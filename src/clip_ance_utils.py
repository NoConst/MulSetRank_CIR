# -*- coding: utf-8 -*-
"""
CLIP-based ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation) utilities
for hard negative sampling in Composed Image Retrieval training.

This module provides hard negative mining capabilities for CLIP-based CIR models,
where the composed query is the element-wise sum of text and image features.

Uses Hugging Face Transformers CLIP models.
Pure PyTorch implementation without FAISS dependency.
Optimized for minimal CPU usage with vectorized operations.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import logging
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CLIP_TEMPERATURE = 0.07
MAX_LOGIT_SCALE = float(np.log(100.0))
LOGIT_SCALE_STATE_FILENAME = "logit_scale.pt"


def get_similarity_scale(
    temperature: float = DEFAULT_CLIP_TEMPERATURE,
    logit_scale: Optional[torch.Tensor] = None,
    max_logit_scale: float = MAX_LOGIT_SCALE,
):
    """Return the multiplicative similarity scale used by CLIP-style losses."""
    if logit_scale is not None:
        if not isinstance(logit_scale, torch.Tensor):
            logit_scale = torch.tensor(logit_scale, dtype=torch.float32)
        return logit_scale.float().clamp(max=max_logit_scale).exp()

    if isinstance(temperature, torch.Tensor):
        return temperature.float().reciprocal()

    return 1.0 / float(temperature)


def get_model_logit_scale(model) -> Optional[torch.Tensor]:
    """Best-effort retrieval of CLIP's learnable logit_scale across wrapper types."""
    candidates = [
        getattr(model, "logit_scale", None),
        getattr(getattr(model, "model", None), "logit_scale", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "logit_scale", None),
    ]

    for candidate in candidates:
        if isinstance(candidate, torch.Tensor):
            return candidate

    for name, param in model.named_parameters():
        if name.endswith("logit_scale"):
            return param

    return None


def temperature_bounds_to_logit_scale_bounds(
    min_temperature: float,
    max_temperature: float,
) -> Tuple[float, float]:
    """Convert temperature bounds to logit_scale bounds."""
    min_temperature = float(min_temperature)
    max_temperature = float(max_temperature)
    if min_temperature <= 0 or max_temperature <= 0:
        raise ValueError("Temperature bounds must be positive.")
    if min_temperature > max_temperature:
        raise ValueError("min_temperature must be <= max_temperature.")

    min_logit_scale = float(np.log(1.0 / max_temperature))
    max_logit_scale = float(np.log(1.0 / min_temperature))
    return min_logit_scale, max_logit_scale


def enable_model_logit_scale_training(model, log: Optional[logging.Logger] = None) -> Optional[torch.Tensor]:
    """Ensure logit_scale stays trainable even when PEFT freezes base parameters."""
    logit_scale = get_model_logit_scale(model)
    if logit_scale is None:
        if log is not None:
            log.warning("Could not find CLIP logit_scale parameter; fixed temperature fallback will be used.")
        return None

    logit_scale.requires_grad_(True)

    if log is not None:
        scale = get_similarity_scale(logit_scale=logit_scale)
        initial_temperature = 1.0 / float(scale.detach().cpu().item())
        log.info(f"Enabled learnable logit_scale (initial temperature={initial_temperature:.4f})")

    return logit_scale


def set_model_logit_scale(
    model,
    temperature: float,
    log: Optional[logging.Logger] = None,
) -> Optional[torch.Tensor]:
    """Reset CLIP logit_scale to a target temperature."""
    logit_scale = get_model_logit_scale(model)
    if logit_scale is None:
        if log is not None:
            log.warning("Could not find CLIP logit_scale parameter; reset skipped.")
        return None

    target_value = float(np.log(1.0 / float(temperature)))
    with torch.no_grad():
        logit_scale.copy_(
            torch.tensor(target_value, dtype=logit_scale.dtype, device=logit_scale.device)
        )

    if log is not None:
        log.info(f"Reset logit_scale to temperature={float(temperature):.4f}")

    return logit_scale


def clamp_logit_scale_(
    logit_scale: Optional[torch.Tensor],
    min_temperature: float,
    max_temperature: float,
) -> Optional[torch.Tensor]:
    """Clamp logit_scale in-place using temperature bounds."""
    if logit_scale is None:
        return None

    min_logit_scale, max_logit_scale = temperature_bounds_to_logit_scale_bounds(
        min_temperature=min_temperature,
        max_temperature=max_temperature,
    )
    with torch.no_grad():
        logit_scale.clamp_(min=min_logit_scale, max=max_logit_scale)
    return logit_scale


def save_model_logit_scale(
    model,
    save_dir: Union[str, Path],
    log: Optional[logging.Logger] = None,
) -> bool:
    """Persist learned logit_scale alongside adapter checkpoints."""
    logit_scale = get_model_logit_scale(model)
    if logit_scale is None:
        return False

    save_path = Path(save_dir) / LOGIT_SCALE_STATE_FILENAME
    torch.save({"logit_scale": logit_scale.detach().cpu()}, save_path)
    if log is not None:
        log.info(f"Saved logit_scale to {save_path}")
    return True


def load_model_logit_scale(
    model,
    load_dir: Union[str, Path],
    log: Optional[logging.Logger] = None,
) -> bool:
    """Load a previously saved logit_scale if present."""
    logit_scale = get_model_logit_scale(model)
    if logit_scale is None:
        return False

    load_path = Path(load_dir) / LOGIT_SCALE_STATE_FILENAME
    if not load_path.exists():
        return False

    state = torch.load(load_path, map_location="cpu")
    saved_logit_scale = state.get("logit_scale")
    if saved_logit_scale is None:
        return False

    with torch.no_grad():
        logit_scale.copy_(
            saved_logit_scale.to(device=logit_scale.device, dtype=logit_scale.dtype)
        )

    if log is not None:
        restored_temperature = 1.0 / float(get_similarity_scale(logit_scale=logit_scale).detach().cpu().item())
        log.info(f"Restored logit_scale from {load_path} (temperature={restored_temperature:.4f})")

    return True


class CLIPHardNegativeMiner:
    """
    ANCE-style hard negative miner for CLIP-based CIR models.
    Uses pure PyTorch for nearest neighbor search (no FAISS dependency).
    Optimized for GPU with minimal CPU operations.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_negatives: int = 16,
        topk_candidates: int = 100,
        refresh_interval: int = 1,
        use_gpu: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        self.topk_candidates = topk_candidates
        self.refresh_interval = refresh_interval
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Embeddings storage (PyTorch tensors on GPU)
        self.target_embeddings: Optional[torch.Tensor] = None
        self.target_names: Optional[List[str]] = None
        self.name_to_idx: dict = {}
        
        # 🆕 预构建的索引tensor，用于快速查找
        self._idx_tensor: Optional[torch.Tensor] = None
        
        # Training state
        self.last_refresh_epoch = -1
        self.is_initialized = False
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def build_index(
        self,
        clip_model,
        dataset,
        device: torch.device,
        batch_size: int = 64,
        num_workers: int = 4
    ):
        """Build embedding index from target images."""
        logger.info("Building PyTorch embedding index...")
        
        from utils import collate_fn
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        clip_model.eval()
        all_features = []
        all_names = []
        
        for names, images in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                # Support both HF CLIP (get_image_features) and LAVIS BLIP2 (extract_target_features)
                if hasattr(clip_model, "get_image_features"):
                    image_features = clip_model.get_image_features(pixel_values=images)
                elif hasattr(clip_model, "extract_target_features"):
                    # BLIP2 returns (features, aux); we only need the projected retrieval embedding
                    image_features, _ = clip_model.extract_target_features(images, mode="mean")
                else:
                    raise AttributeError(
                        "Unsupported model for CLIPHardNegativeMiner.build_index: "
                        "expected `get_image_features` (CLIP) or `extract_target_features` (BLIP2)."
                    )

                # If token-level features are returned, pool to a single vector
                if image_features.dim() == 3:
                    image_features = image_features.mean(dim=1)

                image_features = F.normalize(image_features, dim=-1)
            all_features.append(image_features)
            all_names.extend(names)
        
        self.target_embeddings = torch.cat(all_features, dim=0)
        if not self.use_gpu:
            self.target_embeddings = self.target_embeddings.cpu()
        
        self.target_names = all_names
        self.name_to_idx = {name: idx for idx, name in enumerate(all_names)}
        
        # 🆕 预构建索引tensor
        self._idx_tensor = torch.arange(len(all_names), device=self.target_embeddings.device)
        
        logger.info(f"Built index: {self.target_embeddings.shape}, on {'GPU' if self.target_embeddings.is_cuda else 'CPU'}")
        
        del all_features
        torch.cuda.empty_cache()
        
        self.is_initialized = True
        return self.target_embeddings, self.target_names
    
    def refresh_index(
        self,
        clip_model,
        dataset,
        device: torch.device,
        current_epoch: int,
        batch_size: int = 64,
        num_workers: int = 4,
        force: bool = False
    ) -> bool:
        should_refresh = force or (current_epoch - self.last_refresh_epoch >= self.refresh_interval)
        if should_refresh:
            self.build_index(clip_model, dataset, device, batch_size, num_workers)
            self.last_refresh_epoch = current_epoch
            return True
        return False
    
    def _search_topk(
        self,
        query_features: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-only top-k search."""
        query_features = query_features.to(self.target_embeddings.device)
        similarities = torch.matmul(query_features.float(), self.target_embeddings.T)
        topk_sim, topk_idx = torch.topk(similarities, k=k, dim=-1, largest=True)
        return topk_sim, topk_idx
    
    def _get_positive_indices_tensor(self, positive_names: List[str], device: torch.device) -> torch.Tensor:
        """将positive_names转换为tensor indices（批量操作）"""
        indices = torch.tensor(
            [self.name_to_idx.get(name, -1) for name in positive_names],
            dtype=torch.long,
            device=device
        )
        return indices
    
    def mine_hard_negatives_vectorized(
        self,
        query_features: torch.Tensor,
        positive_names: List[str],
        exclude_reference_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        向量化的硬负样本挖掘 - 最小化CPU操作
        
        Returns:
            hard_negative_indices: [B, num_negatives] on GPU
            hard_negative_names: List[List[str]]
        """
        if not self.is_initialized:
            raise RuntimeError("Index not initialized.")
        
        device = self.target_embeddings.device
        query_features = F.normalize(query_features.to(device), dim=-1)
        batch_size = query_features.shape[0]
        
        # 1. GPU上搜索top-k候选
        _, topk_indices = self._search_topk(query_features, self.topk_candidates)
        # topk_indices: [B, topk_candidates] on GPU
        
        # 2. 获取positive indices (GPU tensor)
        positive_indices = self._get_positive_indices_tensor(positive_names, device)  # [B]
        
        # 3. 在GPU上创建mask排除正样本
        # positive_indices: [B, 1], topk_indices: [B, topk_candidates]
        positive_mask = (topk_indices == positive_indices.unsqueeze(1))  # [B, topk_candidates]
        
        # 4. 如果有exclude_reference_names，也排除
        if exclude_reference_names:
            exclude_indices = self._get_positive_indices_tensor(exclude_reference_names, device)
            exclude_mask = (topk_indices == exclude_indices.unsqueeze(1))
            positive_mask = positive_mask | exclude_mask
        
        # 5. 将正样本位置的值设为一个很大的负数（这样它们在排序后会排到最后）
        # 使用similarity值来重新排序，但我们只需要indices
        # 简单方法：直接过滤
        
        # 6. 向量化选择：使用masked_fill和gather
        # 将要排除的位置填充为-1
        filtered_indices = topk_indices.clone()
        filtered_indices[positive_mask] = -1
        
        # 7. 对每个样本，选择前num_negatives个非-1的索引
        # 由于topk已经按相似度排序，我们只需要取前面的非-1值
        hard_negative_indices = torch.zeros(batch_size, self.num_negatives, dtype=torch.long, device=device)
        
        # 这里仍需要循环，但操作都在GPU tensor上，减少CPU开销
        for i in range(batch_size):
            valid_mask = filtered_indices[i] != -1
            valid_indices = filtered_indices[i][valid_mask]
            
            if len(valid_indices) >= self.num_negatives:
                hard_negative_indices[i] = valid_indices[:self.num_negatives]
            else:
                # 需要随机补充
                hard_negative_indices[i, :len(valid_indices)] = valid_indices
                # 随机采样补充（在GPU上）
                n_needed = self.num_negatives - len(valid_indices)
                random_indices = torch.randint(
                    0, len(self.target_names), (n_needed,), 
                    device=device, dtype=torch.long
                )
                hard_negative_indices[i, len(valid_indices):] = random_indices
        
        # 8. 转换为names（这步必须在CPU上，但indices已经准备好了）
        indices_cpu = hard_negative_indices.cpu().numpy()
        hard_negative_names = [
            [self.target_names[idx] for idx in indices_cpu[i]]
            for i in range(batch_size)
        ]
        
        return hard_negative_indices, hard_negative_names
    
    def mine_hard_negatives(
        self,
        query_features: torch.Tensor,
        positive_names: List[str],
        exclude_reference_names: Optional[List[str]] = None,
        return_names: bool = True
    ) -> Union[Tuple[torch.Tensor, List[List[str]]], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Mine hard negatives - 使用向量化版本
        """
        indices, names = self.mine_hard_negatives_vectorized(
            query_features, positive_names, exclude_reference_names
        )
        
        if return_names:
            return indices, names
        else:
            hard_negative_features = self.target_embeddings[indices]
            return indices, hard_negative_features
    
    def get_features_by_names(self, names: List[str]) -> torch.Tensor:
        """Get precomputed features for a list of image names."""
        indices = [self.name_to_idx[name] for name in names if name in self.name_to_idx]
        if not indices:
            return torch.tensor([])
        return self.target_embeddings[indices]
    
    def get_features_by_indices(self, indices: Union[List[int], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Get precomputed features by indices."""
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long()
        elif isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long)
        return self.target_embeddings[indices.to(self.target_embeddings.device)]
    
    def mine_text_intent_negatives_vectorized(
        self,
        text_features: torch.Tensor,
        ref_features: torch.Tensor,
        captions: List[str],
        positive_names: List[str],
        hard_negative_indices: torch.Tensor,  # 🆕 直接接受indices而非names
        num_negatives: int,
        clip_model=None,
        tokenizer=None
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        向量化的text intent负样本挖掘
        规则：用'and'来split caption，然后对于其中的每一部分描述，
        与ref_features组合去检索一部分最近邻作为负样本，所有的构成最终的负样本
        """
        if not self.is_initialized:
            raise RuntimeError("Index not initialized.")
        
        if clip_model is None or tokenizer is None:
            raise ValueError("clip_model and tokenizer are required for text intent negative mining.")
        
        device = self.target_embeddings.device
        batch_size = text_features.shape[0]
        
        ref_features = F.normalize(ref_features.to(device), dim=-1)
        
        # 获取positive indices
        positive_indices = self._get_positive_indices_tensor(positive_names, device)
        
        # 将hard_negative_indices移到正确设备
        hard_negative_indices = hard_negative_indices.to(device)
        
        # 存储所有负样本indices
        text_intent_negative_indices = torch.zeros(batch_size, num_negatives, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            # 用'and'分割caption
            caption_parts = [part.strip() for part in captions[i].split('and') if part.strip()]
            
            if len(caption_parts) == 0:
                # 如果没有分割出部分，使用原始caption
                caption_parts = [captions[i]]
            
            # 计算每个部分应该分配的负样本数量
            num_parts = len(caption_parts)
            negatives_per_part = num_negatives // num_parts
            remainder = num_negatives % num_parts
            
            all_negative_indices = []
            
            # 对每个部分进行编码和检索
            for part_idx, part_text in enumerate(caption_parts):
                # 编码这个部分的文本
                with torch.no_grad():
                    inputs = tokenizer(
                        [part_text],
                        padding=True,
                        truncation=True,
                        max_length=77,
                        return_tensors="pt"
                    ).to(device)
                    # Use autocast for mixed precision compatibility with FSDP
                    with torch.cuda.amp.autocast():
                        part_text_features = clip_model.get_text_features(**inputs)
                        part_text_features = F.normalize(part_text_features, dim=-1)
                
                # 与ref_features组合
                part_query = F.normalize(ref_features[i:i+1] + part_text_features, dim=-1)
                
                # 搜索top-k候选
                _, topk_indices = self._search_topk(part_query, self.topk_candidates)
                topk_indices = topk_indices[0]  # [topk_candidates]
                
                # 创建排除mask：排除正样本和已有的hard negatives
                positive_mask = (topk_indices == positive_indices[i])
                
                # 排除已有的hard negatives
                for k in range(hard_negative_indices.shape[1]):
                    positive_mask = positive_mask | (topk_indices == hard_negative_indices[i, k])
                
                # 过滤
                filtered_indices = topk_indices.clone()
                filtered_indices[positive_mask] = -1
                
                # 选择这个部分的负样本
                valid_mask = filtered_indices != -1
                valid_indices = filtered_indices[valid_mask]
                
                # 计算这个部分应该选择的负样本数量
                part_num_negatives = negatives_per_part + (1 if part_idx < remainder else 0)
                
                if len(valid_indices) >= part_num_negatives:
                    all_negative_indices.append(valid_indices[:part_num_negatives])
                else:
                    if len(valid_indices) > 0:
                        all_negative_indices.append(valid_indices)
                    # 如果不够，用随机样本补充
                    n_needed = part_num_negatives - len(valid_indices)
                    random_indices = torch.randint(
                        0, len(self.target_names), (n_needed,),
                        device=device, dtype=torch.long
                    )
                    all_negative_indices.append(random_indices)
            
            # 合并所有部分的负样本
            if all_negative_indices:
                combined_indices = torch.cat(all_negative_indices, dim=0)
                # 确保总数不超过num_negatives
                if len(combined_indices) > num_negatives:
                    combined_indices = combined_indices[:num_negatives]
                elif len(combined_indices) < num_negatives:
                    # 如果不够，用随机样本补充
                    n_needed = num_negatives - len(combined_indices)
                    random_indices = torch.randint(
                        0, len(self.target_names), (n_needed,),
                        device=device, dtype=torch.long
                    )
                    combined_indices = torch.cat([combined_indices, random_indices], dim=0)
                
                text_intent_negative_indices[i] = combined_indices
            else:
                # 如果没有找到任何负样本，使用随机样本
                random_indices = torch.randint(
                    0, len(self.target_names), (num_negatives,),
                    device=device, dtype=torch.long
                )
                text_intent_negative_indices[i] = random_indices
        
        # 转换为names
        indices_cpu = text_intent_negative_indices.cpu().numpy()
        text_intent_negative_names = [
            [self.target_names[idx] for idx in indices_cpu[i]]
            for i in range(batch_size)
        ]
        
        return text_intent_negative_indices, text_intent_negative_names
    
    def mine_text_intent_negatives(
        self,
        text_features: torch.Tensor,
        ref_features: torch.Tensor,
        captions: List[str],
        positive_names: List[str],
        hard_negative_names: List[List[str]],
        num_negatives: int,
        return_names: bool = True,
        hard_negative_indices: Optional[torch.Tensor] = None,  # 🆕 可选直接传入indices
        clip_model=None,
        tokenizer=None
    ) -> Union[Tuple[torch.Tensor, List[List[str]]], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Mine text intent-based negatives
        """
        device = self.target_embeddings.device
        
        # 如果没有传入indices，从names构建
        if hard_negative_indices is None:
            batch_size = len(hard_negative_names)
            num_hard_neg = len(hard_negative_names[0])
            hard_negative_indices = torch.zeros(batch_size, num_hard_neg, dtype=torch.long, device=device)
            for i, names in enumerate(hard_negative_names):
                for j, name in enumerate(names):
                    hard_negative_indices[i, j] = self.name_to_idx.get(name, 0)
        
        indices, names = self.mine_text_intent_negatives_vectorized(
            text_features, ref_features, captions, positive_names,
            hard_negative_indices, num_negatives, clip_model, tokenizer
        )
        
        if return_names:
            return indices, names
        else:
            features = self.target_embeddings[indices]
            return indices, features

    @torch.no_grad()
    def mine_partial_intent_negatives(
        self,
        partial_intent_texts: List[str],
        ref_features: torch.Tensor,
        positive_names: List[str],
        hard_negative_indices: torch.Tensor,
        num_negatives: int,
        clip_model=None,
        tokenizer=None
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        使用预生成的 partial intent query + reference image 挖掘负样本。

        对每个样本，将 partial intent text 编码后与 ref image features 组合，
        在索引中检索最近邻作为负样本。

        Args:
            partial_intent_texts: 每个样本对应的 partial intent 文本
            ref_features: [B, D] reference image features
            positive_names: 正样本名称列表
            hard_negative_indices: [B, N] 已有硬负样本索引（排除用）
            num_negatives: 每个样本需要的负样本数量
            clip_model: CLIP 模型
            tokenizer: CLIP tokenizer

        Returns:
            partial_intent_neg_indices: [B, num_negatives]
            partial_intent_neg_names: List[List[str]]
        """
        if not self.is_initialized:
            raise RuntimeError("Index not initialized.")
        if clip_model is None or tokenizer is None:
            raise ValueError("clip_model and tokenizer are required.")

        device = self.target_embeddings.device
        batch_size = ref_features.shape[0]
        ref_features = F.normalize(ref_features.to(device), dim=-1)
        hard_negative_indices = hard_negative_indices.to(device)
        positive_indices = self._get_positive_indices_tensor(positive_names, device)

        inputs = tokenizer(
            partial_intent_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)
        with torch.cuda.amp.autocast():
            partial_text_features = clip_model.get_text_features(**inputs)
            partial_text_features = F.normalize(partial_text_features, dim=-1)

        partial_query = F.normalize(ref_features + partial_text_features, dim=-1)

        _, topk_indices = self._search_topk(partial_query, self.topk_candidates)

        exclude_mask = (topk_indices == positive_indices.unsqueeze(1))
        for k in range(hard_negative_indices.shape[1]):
            exclude_mask = exclude_mask | (topk_indices == hard_negative_indices[:, k:k+1])

        filtered_indices = topk_indices.clone()
        filtered_indices[exclude_mask] = -1

        partial_intent_neg_indices = torch.zeros(
            batch_size, num_negatives, dtype=torch.long, device=device
        )

        for i in range(batch_size):
            valid_mask = filtered_indices[i] != -1
            valid_indices = filtered_indices[i][valid_mask]

            if len(valid_indices) >= num_negatives:
                partial_intent_neg_indices[i] = valid_indices[:num_negatives]
            else:
                partial_intent_neg_indices[i, :len(valid_indices)] = valid_indices
                n_needed = num_negatives - len(valid_indices)
                random_indices = torch.randint(
                    0, len(self.target_names), (n_needed,),
                    device=device, dtype=torch.long
                )
                partial_intent_neg_indices[i, len(valid_indices):] = random_indices

        indices_cpu = partial_intent_neg_indices.cpu().numpy()
        partial_intent_neg_names = [
            [self.target_names[idx] for idx in indices_cpu[i]]
            for i in range(batch_size)
        ]

        return partial_intent_neg_indices, partial_intent_neg_names

    @torch.no_grad()
    def mine_multi_partial_intent_negatives(
        self,
        partial_intent_texts_per_sample: List[List[str]],
        ref_features: torch.Tensor,
        positive_names: List[str],
        hard_negative_indices: torch.Tensor,
        num_negatives: int,
        clip_model=None,
        tokenizer=None
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        对每个样本随机选取一个 partial intent query，与 ref image 组合
        检索 num_negatives 个负样本。

        Args:
            partial_intent_texts_per_sample: [B] 列表，每个元素是该样本的多个 partial intent 文本列表
            ref_features: [B, D] reference image features
            positive_names: 正样本名称列表
            hard_negative_indices: [B, N] 已有硬负样本索引（排除用）
            num_negatives: 需要的负样本数量
            clip_model: CLIP 模型
            tokenizer: CLIP tokenizer

        Returns:
            result_indices: [B, num_negatives]
            result_names: List[List[str]]
        """
        if not self.is_initialized:
            raise RuntimeError("Index not initialized.")
        if clip_model is None or tokenizer is None:
            raise ValueError("clip_model and tokenizer are required.")

        device = self.target_embeddings.device
        batch_size = ref_features.shape[0]
        ref_features = F.normalize(ref_features.to(device), dim=-1)
        hard_negative_indices = hard_negative_indices.to(device)
        positive_indices = self._get_positive_indices_tensor(positive_names, device)

        import random
        selected_texts = [
            random.choice(texts) for texts in partial_intent_texts_per_sample
        ]

        inputs = tokenizer(
            selected_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)
        with torch.cuda.amp.autocast():
            text_features = clip_model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)

        query = F.normalize(ref_features + text_features, dim=-1)

        _, topk_indices = self._search_topk(query, self.topk_candidates)

        exclude_mask = (topk_indices == positive_indices.unsqueeze(1))
        for k in range(hard_negative_indices.shape[1]):
            exclude_mask = exclude_mask | (topk_indices == hard_negative_indices[:, k:k + 1])

        filtered_indices = topk_indices.clone()
        filtered_indices[exclude_mask] = -1

        result_indices = torch.zeros(batch_size, num_negatives, dtype=torch.long, device=device)

        for i in range(batch_size):
            valid_mask = filtered_indices[i] != -1
            valid = filtered_indices[i][valid_mask]

            if len(valid) >= num_negatives:
                result_indices[i] = valid[:num_negatives]
            else:
                if len(valid) > 0:
                    result_indices[i, :len(valid)] = valid
                n_needed = num_negatives - len(valid)
                random_idx = torch.randint(
                    0, len(self.target_names), (n_needed,),
                    device=device, dtype=torch.long
                )
                result_indices[i, len(valid):] = random_idx

        indices_cpu = result_indices.cpu().numpy()
        result_names = [
            [self.target_names[idx] for idx in indices_cpu[i]]
            for i in range(batch_size)
        ]

        return result_indices, result_names


def contrastive_in_batch_loss(
    query,
    target,
    temperature=DEFAULT_CLIP_TEMPERATURE,
    normalized=False,
    logit_scale: Optional[torch.Tensor] = None,
    max_logit_scale: float = MAX_LOGIT_SCALE,
):
    """In-batch contrastive loss."""
    if not normalized:
        query = F.normalize(query, dim=-1)
        target = F.normalize(target, dim=-1)
    scale = get_similarity_scale(
        temperature=temperature,
        logit_scale=logit_scale,
        max_logit_scale=max_logit_scale,
    )
    sim = torch.matmul(query.float(), target.float().T) * scale
    labels = torch.arange(query.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(sim, labels)


def contrastive_loss_hard_negative(
    query,
    positive,
    negatives,
    temperature=DEFAULT_CLIP_TEMPERATURE,
    normalized=False,
    logit_scale: Optional[torch.Tensor] = None,
    max_logit_scale: float = MAX_LOGIT_SCALE,
):
    """Contrastive loss with hard negatives."""
    if not normalized:
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

    scale = get_similarity_scale(
        temperature=temperature,
        logit_scale=logit_scale,
        max_logit_scale=max_logit_scale,
    )
    pos_sim = torch.sum(query.float() * positive.float(), dim=-1, keepdim=True)
    neg_sim = torch.bmm(query.float().unsqueeze(1), negatives.float().transpose(1, 2)).squeeze(1)
    logits = torch.cat([pos_sim, neg_sim], dim=1) * scale
    labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


def compute_local_ranking_loss(query_feat, target_feat, hard_neg_feats, margin=0.05):
    """Local ranking loss."""
    query_feat = F.normalize(query_feat, dim=-1)
    target_feat = F.normalize(target_feat, dim=-1)
    hard_neg_feats = F.normalize(hard_neg_feats, dim=-1)

    pos_sim = torch.sum(query_feat * target_feat, dim=-1, keepdim=True)
    neg_sims = torch.bmm(query_feat.unsqueeze(1), hard_neg_feats.transpose(1, 2)).squeeze(1)
    loss = torch.clamp(neg_sims - pos_sim + margin, min=0.0)
    return loss.mean()


def compute_listwise_ranking_loss(
    query_features: torch.Tensor,
    negative_features_list: List[torch.Tensor],
    temperature: float = 0.1,
    ranking_weights: Optional[List[float]] = None,
    logit_scale: Optional[torch.Tensor] = None,
    max_logit_scale: float = MAX_LOGIT_SCALE,
) -> torch.Tensor:
    """
    使用 Listwise 排序来形成负样本之间的偏序
    
    Args:
        query_features: [B, D] query features
        negative_features_list: List of negative feature tensors, each [B, N_i, D]
                                按照期望的排序顺序排列（例如：[hard, text_intent, ref_hard]）
        temperature: temperature for softmax
        ranking_weights: Optional weights for each negative type, default is [1.0, 0.8, 0.6, ...]
    
    Returns:
        listwise ranking loss
    """
    device = query_features.device
    batch_size = query_features.shape[0]
    
    # 归一化
    query_features = F.normalize(query_features, dim=-1)
    normalized_negatives = [F.normalize(neg, dim=-1) for neg in negative_features_list]
    
    # 计算每个负样本与query的相似度
    all_similarities = []
    
    for neg_features in normalized_negatives:
        # neg_features: [B, N_i, D]
        # 计算相似度: [B, 1, D] x [B, D, N_i] -> [B, 1, N_i] -> [B, N_i]
        sim = torch.bmm(
            query_features.unsqueeze(1),
            neg_features.transpose(1, 2)
        ).squeeze(1)  # [B, N_i]
        
        all_similarities.append(sim)
    
    # 拼接所有相似度: [B, total_negatives]
    all_sim = torch.cat(all_similarities, dim=1)  # [B, total_negatives]
    
    # 使用 softmax 将相似度转换为概率分布 (ListNet approach)
    # 相似度越高，排名应该越高，softmax 会放大较大的值，所以直接使用 sim
    
    # 使用温度缩放
    scale = get_similarity_scale(
        temperature=temperature,
        logit_scale=logit_scale,
        max_logit_scale=max_logit_scale,
    )
    logits = all_sim.float() * scale
    
    # 构建目标概率分布
    # 目标：前面的负样本类型（hard）应该比后面的（text_intent, ref_hard）排名更高
    # 使用指数衰减的权重来构建目标分布
    if ranking_weights is None:
        # 默认权重：每个类型的权重递减
        num_types = len(negative_features_list)
        ranking_weights = [1.0 / (i + 1) for i in range(num_types)]
    
    # 为每个负样本分配目标权重
    target_weights = []
    for neg_idx, neg_features in enumerate(normalized_negatives):
        num_neg = neg_features.shape[1]
        weight = ranking_weights[neg_idx]
        target_weights.extend([weight] * num_neg)
    
    target_weights = torch.tensor(target_weights, device=device, dtype=torch.float32)
    target_weights = target_weights.unsqueeze(0).expand(batch_size, -1)  # [B, total_negatives]
    
    # 归一化目标权重为概率分布
    target_probs = target_weights / target_weights.sum(dim=1, keepdim=True)  # [B, total_negatives]
    
    # 计算 KL 散度损失 (ListNet loss)
    # KL(P_target || P_pred) = sum(P_target * log(P_target / P_pred))
    # 为了避免数值不稳定，使用 log_softmax
    log_pred_probs = F.log_softmax(logits, dim=-1)
    kl_loss = F.kl_div(log_pred_probs, target_probs, reduction='batchmean')
    
    return kl_loss


def compute_clip_ance_loss(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    hard_negative_features: torch.Tensor,
    temperature: float = DEFAULT_CLIP_TEMPERATURE,
    hard_negative_weight: float = 1.0,
    ref_hard_negative_features: Optional[torch.Tensor] = None,
    ref_hard_negative_weight: float = 1.0,
    partial_intent_negative_features: Optional[torch.Tensor] = None,
    partial_intent_negative_weight: float = 0.75,
    logit_scale: Optional[torch.Tensor] = None,
    max_logit_scale: float = MAX_LOGIT_SCALE,
    **legacy_kwargs,
) -> torch.Tensor:
    """Compute ANCE-style contrastive loss with three-level negatives."""
    device = query_features.device

    # Backward compatibility with older training scripts.
    if partial_intent_negative_features is None and legacy_kwargs.get("text_intent_negative_features") is not None:
        partial_intent_negative_features = legacy_kwargs["text_intent_negative_features"]
    if "text_intent_negative_weight" in legacy_kwargs:
        partial_intent_negative_weight = legacy_kwargs["text_intent_negative_weight"]
    
    if isinstance(hard_negative_features, np.ndarray):
        hard_neg_tensor = torch.from_numpy(hard_negative_features).float().to(device)
    else:
        hard_neg_tensor = hard_negative_features

    # ---- Device alignment (important for distributed training) ----
    # Ensure ALL tensors are on the same device as query_features to avoid cuda:x vs cpu mismatches.
    if isinstance(target_features, torch.Tensor) and target_features.device != device:
        target_features = target_features.to(device, non_blocking=True)
    if isinstance(hard_neg_tensor, torch.Tensor) and hard_neg_tensor.device != device:
        hard_neg_tensor = hard_neg_tensor.to(device, non_blocking=True)
    if ref_hard_negative_features is not None and isinstance(ref_hard_negative_features, torch.Tensor):
        if ref_hard_negative_features.device != device:
            ref_hard_negative_features = ref_hard_negative_features.to(device, non_blocking=True)
    if partial_intent_negative_features is not None and isinstance(partial_intent_negative_features, torch.Tensor):
        if partial_intent_negative_features.device != device:
            partial_intent_negative_features = partial_intent_negative_features.to(device, non_blocking=True)
    
    # 统一归一化
    query_features = F.normalize(query_features, dim=-1)
    if target_features.dim() == 3:
        # (B, Q, D) token/query-level target features (e.g., BLIP2)
        target_features = F.normalize(target_features, dim=-1)
    else:
        # (B, D) vector target features (e.g., CLIP)
        target_features = F.normalize(target_features, dim=-1)
    hard_neg_tensor = F.normalize(hard_neg_tensor, dim=-1)
    
    # In-batch loss
    if target_features.dim() == 3:
        scale = get_similarity_scale(
            temperature=temperature,
            logit_scale=logit_scale,
            max_logit_scale=max_logit_scale,
        )
        # query: (B,D), target: (B,Q,D) -> sim matrix (B,B) using max over Q
        sim = torch.einsum("id,jqd->ijq", query_features.float(), target_features.float()).max(dim=-1).values
        sim = sim * scale
        labels = torch.arange(query_features.shape[0], dtype=torch.long, device=device)
        loss_in_batch = F.cross_entropy(sim, labels)
    else:
        loss_in_batch = contrastive_in_batch_loss(
            query_features,
            target_features,
            temperature,
            normalized=True,
            logit_scale=logit_scale,
            max_logit_scale=max_logit_scale,
        )

    # Hard negative loss
    if target_features.dim() == 3:
        scale = get_similarity_scale(
            temperature=temperature,
            logit_scale=logit_scale,
            max_logit_scale=max_logit_scale,
        )
        # Positive similarity per sample: max over Q
        sim_pos = torch.einsum("id,iqd->iq", query_features.float(), target_features.float()).max(dim=-1).values
        # Neg similarities: (B,N)
        sim_hard = torch.bmm(
            query_features.float().unsqueeze(1),
            hard_neg_tensor.float().transpose(1, 2),
        ).squeeze(1)
        logits = torch.cat([sim_pos.unsqueeze(1), sim_hard], dim=1) * scale
        labels_hard = torch.zeros(query_features.shape[0], dtype=torch.long, device=device)
        loss_hard_negative = F.cross_entropy(logits, labels_hard)
    else:
        loss_hard_negative = contrastive_loss_hard_negative(
            query_features,
            target_features,
            hard_neg_tensor,
            temperature,
            normalized=True,
            logit_scale=logit_scale,
            max_logit_scale=max_logit_scale,
        )

    batch_size, num_negatives, dim = hard_neg_tensor.shape
    query_repeated = query_features.unsqueeze(1).repeat(1, num_negatives, 1).view(-1, dim)
    hard_neg_flat = hard_neg_tensor.view(-1, dim)
    loss_hard_in_batch = contrastive_in_batch_loss(
        query_repeated,
        hard_neg_flat,
        temperature,
        logit_scale=logit_scale,
        max_logit_scale=max_logit_scale,
    )

    total_loss = loss_in_batch + hard_negative_weight * loss_hard_negative + loss_hard_in_batch
    # total_loss = loss_in_batch + hard_negative_weight * loss_hard_negative
    # total_loss = loss_in_batch + loss_hard_in_batch
    
    # Partial intent negative loss
    partial_intent_neg_tensor = None
    if partial_intent_negative_features is not None:
        partial_intent_neg_tensor = F.normalize(partial_intent_negative_features, dim=-1)
        loss_partial_intent = contrastive_loss_hard_negative(
            query_features,
            target_features,
            partial_intent_neg_tensor,
            temperature,
            normalized=True,
            logit_scale=logit_scale,
            max_logit_scale=max_logit_scale,
        )
        total_loss = total_loss + partial_intent_negative_weight * loss_partial_intent
    
    # Reference hard negative loss
    ref_hard_neg_tensor = None
    if ref_hard_negative_features is not None:
        ref_hard_neg_tensor = F.normalize(ref_hard_negative_features, dim=-1)
    
    # 使用 Listwise 排序来形成负样本之间的偏序
    # 期望的排序：hard > partial_intent > ref_hard
    negative_features_list = []
    if partial_intent_neg_tensor is not None:
        # 排序为：hard > partial_intent > ref_hard
        negative_features_list.append(hard_neg_tensor)  # 最高优先级
        negative_features_list.append(partial_intent_neg_tensor)  # 中等优先级
        if ref_hard_neg_tensor is not None:
            negative_features_list.append(ref_hard_neg_tensor)  # 最低优先级
    elif ref_hard_neg_tensor is not None:
        # 如果没有 partial_intent，则排序为：hard > ref_hard
        negative_features_list.append(hard_neg_tensor)
        negative_features_list.append(ref_hard_neg_tensor)
    
    # 如果有多个负样本类型，使用 listwise 排序损失
    if len(negative_features_list) > 1:
        # 定义每个类型的权重（权重越高，排名越高）
        ranking_weights = [1.0, 0.7, 0.4][:len(negative_features_list)]
        listwise_loss = compute_listwise_ranking_loss(
            query_features=query_features,
            negative_features_list=negative_features_list,
            temperature=0.1,
            ranking_weights=ranking_weights,
            logit_scale=logit_scale,
            max_logit_scale=max_logit_scale,
        )
        total_loss = total_loss + listwise_loss
    
    return total_loss


def element_wise_sum(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """Compute normalized element-wise sum."""
    return F.normalize(image_features + text_features, dim=-1)
