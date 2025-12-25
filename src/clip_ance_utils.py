# -*- coding: utf-8 -*-
"""
CLIP-based ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation) utilities
for hard negative sampling in Composed Image Retrieval training.

This module provides hard negative mining capabilities for CLIP-based CIR models,
where the composed query is the element-wise sum of text and image features.

Uses Hugging Face Transformers CLIP models.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import faiss
from typing import Tuple, List, Optional, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import logging
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPHardNegativeMiner:
    """
    ANCE-style hard negative miner for CLIP-based CIR models.
    Uses FAISS for approximate nearest neighbor search.
    
    The target features are CLIP image embeddings, and queries are
    element-wise sum of reference image and text embeddings.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,  # CLIP ViT-B/32 dim, adjust for other models
        num_negatives: int = 16,
        topk_candidates: int = 100,
        refresh_interval: int = 1,
        use_gpu: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the CLIP hard negative miner.
        
        Args:
            embedding_dim: Dimension of CLIP embeddings (512 for ViT-B/32, 768 for ViT-L/14)
            num_negatives: Number of hard negatives to sample per query
            topk_candidates: Top-k candidates from which to sample negatives
            refresh_interval: How often to refresh the ANN index (in epochs)
            use_gpu: Whether to use GPU for FAISS search
            cache_dir: Directory to cache embeddings (optional)
        """
        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        self.topk_candidates = topk_candidates
        self.refresh_interval = refresh_interval
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Index and embeddings storage
        self.index = None
        self.target_embeddings = None
        self.target_names = None
        self.name_to_idx = {}
        
        # Training state
        self.last_refresh_epoch = -1
        self.is_initialized = False
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create a FAISS index for the embeddings using inner product (cosine similarity)."""
        dim = embeddings.shape[1]
        num_embeddings = embeddings.shape[0]
        
        logger.info(f"Creating FAISS index for {num_embeddings} embeddings with dim={dim}")
        
        # Use inner product for cosine similarity (embeddings should be normalized)
        index = faiss.IndexFlatIP(dim)
        
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                res.setTempMemory(128 * 1024 * 1024)  # 128MB
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Using GPU for FAISS index")
            except Exception as e:
                logger.warning(f"Failed to use GPU for FAISS: {e}. Falling back to CPU.")
                index = faiss.IndexFlatIP(dim)
        
        logger.info("Adding embeddings to FAISS index...")
        index.add(embeddings)
        logger.info(f"FAISS index built successfully with {index.ntotal} vectors")
        return index
    
    @torch.no_grad()
    def build_index(
        self,
        clip_model,
        dataset,
        device: torch.device,
        batch_size: int = 64,
        num_workers: int = 4
    ):
        """
        Build the FAISS index from target image features using CLIP.
        
        Args:
            clip_model: The CLIP model (transformers CLIPModel) for feature extraction
            dataset: Dataset in 'classic' mode containing target images
            device: torch device
            batch_size: Batch size for feature extraction
            num_workers: Number of data loading workers
        """
        logger.info("Building FAISS index for CLIP hard negative mining...")
        
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
        
        for names, images in tqdm(dataloader, desc="Extracting CLIP target features"):
            images = images.to(device, non_blocking=True)
            
            # Extract image features using transformers CLIP
            # images are already preprocessed, so we pass them directly
            outputs = clip_model.get_image_features(pixel_values=images)
            image_features = outputs
            
            # Normalize features for cosine similarity
            image_features = F.normalize(image_features, dim=-1)
            
            all_features.append(image_features.cpu().numpy())
            all_names.extend(names)
            
            if len(all_features) % 100 == 0:
                torch.cuda.empty_cache()
        
        logger.info("Feature extraction completed. Concatenating features...")
        
        self.target_embeddings = np.vstack(all_features).astype('float32')
        self.target_names = all_names
        self.name_to_idx = {name: idx for idx, name in enumerate(all_names)}
        
        logger.info(f"Total features shape: {self.target_embeddings.shape}")
        
        del all_features
        torch.cuda.empty_cache()
        
        logger.info("Starting FAISS index construction...")
        self.index = self._create_index(self.target_embeddings)
        self.is_initialized = True
        
        logger.info(f"Built index with {len(self.target_names)} target images")
        
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
        """
        Refresh the index if needed based on the refresh interval.
        
        Returns:
            True if the index was refreshed, False otherwise
        """
        should_refresh = force or (current_epoch - self.last_refresh_epoch >= self.refresh_interval)
        
        if should_refresh:
            self.build_index(clip_model, dataset, device, batch_size, num_workers)
            self.last_refresh_epoch = current_epoch
            return True
        
        return False
    
    def mine_hard_negatives(
        self,
        query_features: torch.Tensor,
        positive_names: List[str],
        exclude_reference_names: Optional[List[str]] = None,
        return_names: bool = True
    ) -> Union[Tuple[np.ndarray, List[List[str]]], Tuple[np.ndarray, np.ndarray]]:
        """
        Mine hard negatives for a batch of composed queries.
        
        Args:
            query_features: Composed query embeddings (batch_size, dim)
                           This should be the element-wise sum of image and text features
            positive_names: List of positive target names to exclude
            exclude_reference_names: Additional names to exclude (e.g., reference images)
            return_names: If True, return names; if False, return precomputed features
            
        Returns:
            If return_names=True:
                hard_negative_indices: Indices of hard negatives (batch_size, num_negatives)
                hard_negative_names: Names of hard negatives (batch_size, num_negatives)
            If return_names=False:
                hard_negative_indices: Indices of hard negatives (batch_size, num_negatives)
                hard_negative_features: Features of hard negatives (batch_size, num_negatives, dim)
        """
        if not self.is_initialized:
            raise RuntimeError("Index not initialized. Call build_index first.")
        
        # Normalize and convert to numpy
        query_features = F.normalize(query_features, dim=-1)
        query_np = query_features.cpu().numpy().astype('float32')
        
        batch_size = query_np.shape[0]
        
        # Search for top-k candidates
        _, I = self.index.search(query_np, self.topk_candidates)
        
        # Filter out positives and sample hard negatives
        hard_negative_indices = np.zeros((batch_size, self.num_negatives), dtype=np.int64)
        
        for i in range(batch_size):
            candidates = I[i]
            
            # Get positive index to exclude
            positive_idx = self.name_to_idx.get(positive_names[i], -1)
            
            # Get reference index to exclude (if provided)
            exclude_idx = -1
            if exclude_reference_names and i < len(exclude_reference_names):
                exclude_idx = self.name_to_idx.get(exclude_reference_names[i], -1)
            
            # Filter candidates
            valid_candidates = []
            for cand in candidates:
                if cand != positive_idx and cand != exclude_idx:
                    valid_candidates.append(cand)
                if len(valid_candidates) >= self.num_negatives:
                    break
            
            # Pad if necessary
            while len(valid_candidates) < self.num_negatives:
                rand_idx = np.random.randint(0, len(self.target_names))
                if rand_idx not in valid_candidates and rand_idx != positive_idx:
                    valid_candidates.append(rand_idx)
            
            hard_negative_indices[i] = valid_candidates[:self.num_negatives]
        
        if return_names:
            # Return names instead of precomputed features for gradient flow
            hard_negative_names = []
            for i in range(batch_size):
                batch_names = [self.target_names[idx] for idx in hard_negative_indices[i]]
                hard_negative_names.append(batch_names)
            return hard_negative_indices, hard_negative_names
        else:
            # Return precomputed features (old behavior, breaks gradient)
            hard_negative_features = self.target_embeddings[hard_negative_indices]
            return hard_negative_indices, hard_negative_features
    
    def get_features_by_names(self, names: List[str]) -> np.ndarray:
        """Get precomputed features for a list of image names."""
        indices = [self.name_to_idx[name] for name in names if name in self.name_to_idx]
        if not indices:
            return np.array([])
        return self.target_embeddings[indices]
    
    def get_features_by_indices(self, indices: Union[List[int], np.ndarray]) -> np.ndarray:
        """Get precomputed features by indices."""
        return self.target_embeddings[indices]

def contrastive_in_batch_loss(query, target, temperature=0.07, normalized=False):
    """
    query: [B, D]
    target: [B, D]
    normalized: If True, skip normalization (assume inputs are already normalized)
    """
    if not normalized:
        query = F.normalize(query, dim=-1)
        target = F.normalize(target, dim=-1)
    sim = torch.matmul(query, target.T) / temperature
    labels = torch.arange(query.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(sim, labels)

def contrastive_loss_hard_negative(query, positive, negatives, temperature=0.07, normalized=False):
    """
    query: [B, D]
    positive: [B, D]
    negatives: [B, K, D] (K 是负样本数量)
    normalized: If True, skip normalization (assume inputs are already normalized)
    """
    # 1. 特征归一化 (L2 Normalization) - 使用余弦相似度时必须
    if not normalized:
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

    # 2. 计算正样本相似度: [B, 1]
    # 使用 einsum 或 sum(q*p)
    pos_sim = torch.sum(query * positive, dim=-1, keepdim=True) # [B, 1]

    # 3. 计算负样本相似度: [B, K]
    # query: [B, 1, D], negatives: [B, K, D] -> bmm -> [B, 1, K]
    neg_sim = torch.bmm(query.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) # [B, K]

    # 4. 拼接 logits: [B, K + 1]
    # 约定第 0 列永远是正样本
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    
    # 5. 除以温度系数
    logits /= temperature

    # 6. 生成标签: 目标全为 0 (因为正样本在 index 0)
    labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)

    # 7. 计算交叉熵
    loss = F.cross_entropy(logits, labels)
    return loss

def compute_local_ranking_loss(query_feat, target_feat, hard_neg_feats, margin=0.05):
    """
    计算局部相对边际损失，确保正样本相似度高于硬负样本（False Negatives）
    
    Args:
        query_feat: [B, D] - 组合查询特征
        target_feat: [B, D] - 标注的正样本图像特征
        hard_neg_feats: [B, K, D] - 疑似假负样本的硬负样本张量
        margin: 边际值，建议取值 0.01 ~ 0.05 之间以保护特征空间
    """
    # 1. L2 归一化：保证在超球面上进行微调，不破坏原始特征分布
    query_feat = F.normalize(query_feat, dim=-1)
    target_feat = F.normalize(target_feat, dim=-1)
    hard_neg_feats = F.normalize(hard_neg_feats, dim=-1)

    # 2. 计算正样本相似度 s(q, p): [B, 1]
    pos_sim = torch.sum(query_feat * target_feat, dim=-1, keepdim=True)

    # 3. 计算负样本相似度 s(q, n): [B, K]
    # 使用 bmm 计算 batch 内的矩阵乘法: (B, 1, D) * (B, D, K) -> (B, 1, K)
    neg_sims = torch.bmm(query_feat.unsqueeze(1), hard_neg_feats.transpose(1, 2)).squeeze(1)

    # 4. 计算 Ranking Loss: max(0, neg_sim - pos_sim + margin)
    # 只有当 neg_sim + margin > pos_sim 时才产生梯度，不会过度推开负样本
    loss = torch.clamp(neg_sims - pos_sim + margin, min=0.0)

    # 返回 Batch 的平均损失
    return loss.mean()

def compute_clip_ance_loss(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    hard_negative_features: torch.Tensor,
    temperature: float = 0.07,
    hard_negative_weight: float = 1.0,
    ref_hard_negative_features: Optional[torch.Tensor] = None,
    ref_hard_negative_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute ANCE-style contrastive loss for CLIP CIR.
    
    The loss combines:
    1. In-batch contrastive loss: query vs all targets in the batch
    2. Hard negative loss: query vs hard negatives from the current model
    3. (Optional) Reference hard negative loss: query vs reference image hard negatives
    
    Args:
        query_features: Composed query features (batch_size, dim)
                       Element-wise sum of reference image + text features
        target_features: Positive target features (batch_size, dim)
        hard_negative_features: Hard negative features (batch_size, num_negatives, dim)
                               Now these are freshly encoded through current model with gradients
        temperature: Temperature scaling factor
        hard_negative_weight: Weight for hard negative loss
        ref_hard_negative_features: Reference image hard negatives (batch_size, num_ref_negatives, dim)
                                   Images similar to reference but don't match text description
        ref_hard_negative_weight: Weight for reference hard negative loss
        
    Returns:
        Combined contrastive loss
    """
    device = query_features.device
    
    # Convert hard negatives to tensor if needed (backwards compatibility)
    # Now hard_negative_features should already be a tensor with gradients
    if isinstance(hard_negative_features, np.ndarray):
        hard_neg_tensor = torch.from_numpy(hard_negative_features).float().to(device)
    else:
        hard_neg_tensor = hard_negative_features
    
    # ✅ GPU优化：统一归一化一次（避免在每个loss函数中重复归一化）
    query_features = F.normalize(query_features, dim=-1)
    target_features = F.normalize(target_features, dim=-1)
    hard_neg_tensor = F.normalize(hard_neg_tensor, dim=-1)
    
    # Compute in-batch contrastive loss (传入normalized=True避免重复归一化)
    loss_in_batch = contrastive_in_batch_loss(
        query_features, target_features, temperature, normalized=True
    )
        
    # Compute local ranking loss with hard negatives
    # This loss ensures positive samples are ranked higher than hard negatives
    # loss_local_ranking = compute_local_ranking_loss(query_features, target_features, hard_neg_tensor, margin=0.0)

    loss_hard_negative = contrastive_loss_hard_negative(
        query_features, target_features, hard_neg_tensor, temperature, normalized=True
    )

    # ✅ 优化版本1：向量化循环（保持原始语义，减少Python循环开销）
    # 注意：完全批量化会改变loss的语义（分母范围不同），所以保留循环但优化索引
    batch_size, num_negatives, dim = hard_neg_tensor.shape
    
    # # 使用列表推导和torch.stack减少循环开销
    # losses = []
    # for k in range(num_negatives):
    #     target_k = hard_neg_tensor[:, k, :]  # [B, D] - 直接索引，避免split和squeeze
    #     loss_k = contrastive_in_batch_loss(
    #         query_features, target_k, temperature, normalized=True  # 已归一化
    #     )
    #     losses.append(loss_k)
    
    # # # 堆叠并求和（比累加更高效）
    # loss_hard_in_batch = torch.stack(losses).sum()
    
    # 备注：如果想要更激进的优化（改变训练语义，增加负样本难度），可以使用：
    # query_repeated = query_features.unsqueeze(1).repeat(1, num_negatives, 1).view(-1, dim)
    # hard_neg_flat = hard_neg_tensor.view(-1, dim)
    # loss_hard_in_batch = contrastive_in_batch_loss(query_repeated, hard_neg_flat, temperature)
    # 但这会让训练更难（分母更大），可能影响收敛
    
    # Combined loss
    # total_loss = loss_in_batch + hard_negative_weight * loss_hard_negative + loss_hard_in_batch
    total_loss = loss_in_batch + hard_negative_weight * loss_hard_negative
    
    # 🆕 Add reference hard negative loss if provided
    if ref_hard_negative_features is not None:
        # Normalize reference hard negatives
        ref_hard_neg_tensor = F.normalize(ref_hard_negative_features, dim=-1)
        
        # Compute contrastive loss: query vs reference hard negatives
        # 这个loss让模型学习到：即使reference相似，如果不匹配text描述也不应该被检索
        loss_ref_hard_negative = contrastive_loss_hard_negative(
            query_features, target_features, ref_hard_neg_tensor, temperature, normalized=True
        )
        # query_repeated = query_features.unsqueeze(1).repeat(1, num_negatives, 1).view(-1, dim)
        # ref_hard_neg_flat = ref_hard_neg_tensor.view(-1, dim)
        # loss_ref_hard_in_batch = contrastive_in_batch_loss(query_repeated, ref_hard_neg_flat, temperature)

        # loss_between_hard_and_ref_hard = 0
        # for k in range(hard_neg_tensor.shape[1]):
        #     hard_neg_k = hard_neg_tensor[:, k, :]
        #     loss_between_hard_and_ref_hard += contrastive_loss_hard_negative(
        #         query_features, hard_neg_k, ref_hard_neg_tensor, temperature, normalized=True
        #     )
            
        # total_loss = total_loss + ref_hard_negative_weight * loss_ref_hard_negative + loss_ref_hard_in_batch + loss_between_hard_and_ref_hard

        # 🆕 负样本层次化: 让query硬负样本比reference硬负样本更接近query
        # 使用Pairwise Sigmoid Ranking: 不受负样本数量影响，梯度smooth
        # 
        # 优势：
        # 1. 每个pair独立建模，不受K_hard/K_ref数量比例影响
        # 2. 使用sigmoid提供smooth梯度，训练更稳定
        # 3. 概率化建模，loss范围[0,1]，易于调参
        
        # ⚡ 完全向量化计算（无Python循环）
        # 计算query vs query硬负样本的相似度 [B, K_hard]
        sim_query_hard = torch.bmm(
            query_features.unsqueeze(1), 
            hard_neg_tensor.transpose(1, 2)
        ).squeeze(1)  # [B, K_hard]
        
        # 计算query vs reference硬负样本的相似度 [B, K_ref]
        sim_query_ref_hard = torch.bmm(
            query_features.unsqueeze(1), 
            ref_hard_neg_tensor.transpose(1, 2)
        ).squeeze(1)  # [B, K_ref]
        
        # 计算相似度差异 [B, K_hard, K_ref]
        # sim_diff[b,i,j] = sim(query_b, hard_neg_i) - sim(query_b, ref_hard_neg_j)
        sim_diff = sim_query_hard.unsqueeze(2) - sim_query_ref_hard.unsqueeze(1)  # [B, K_hard, K_ref]
        
        # ⭐ Pairwise Sigmoid Ranking
        # 使用sigmoid将相似度差映射到概率空间
        # sigmoid(sim_diff / T) → 1 表示 hard_neg 明显比 ref_hard_neg 相似度高
        # sigmoid(sim_diff / T) → 0 表示 ref_hard_neg 相似度更高（需要惩罚）
        temperature_ranking = 0.1  # 温度参数：越小sigmoid越陡峭，区分度越高
        
        # 计算logits（未经过sigmoid的原始分数）
        logits = sim_diff / temperature_ranking  # [B, K_hard, K_ref]
        
        # 目标：所有pair的ranking概率都应该接近1
        # 即：每个query硬负样本都应该比所有ref硬负样本相似度更高
        target_probs = torch.ones_like(logits)
        
        # ✅ 使用BCEWithLogitsLoss（数值更稳定，兼容AMP）
        # 内部会先做sigmoid再计算BCE，避免手动sigmoid带来的数值问题
        # 比margin loss的优势：
        # - 即使满足条件(sim_diff > 0)，仍有梯度驱动进一步优化
        # - 梯度大小自适应：接近决策边界时梯度大，远离时梯度小
        # - 兼容混合精度训练(AMP autocast)
        loss_negative_ranking = F.binary_cross_entropy_with_logits(
            logits,
            target_probs,
            reduction='mean'
        )
        
        total_loss = total_loss + ref_hard_negative_weight * loss_ref_hard_negative + 0.5 * loss_negative_ranking
    
    return total_loss

def element_wise_sum(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized element-wise sum of image and text features.
    This is the composed query representation for CLIP CIR.
    
    Args:
        image_features: Reference image features (B, D)
        text_features: Text query features (B, D)
        
    Returns:
        Normalized composed features (B, D)
    """
    return F.normalize(image_features + text_features, dim=-1)

