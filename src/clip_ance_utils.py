# -*- coding: utf-8 -*-
"""
CLIP-based ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation) utilities
for hard negative sampling in Composed Image Retrieval training.

This module provides hard negative mining capabilities for CLIP-based CIR models,
where the composed query can be a learned fusion head or the element-wise sum
fallback of text and image features.

Uses Hugging Face Transformers CLIP models.
Pure PyTorch implementation without FAISS dependency.
Optimized for minimal CPU usage with vectorized operations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import logging

from cir_fusion import compose_image_features, compose_query_features, get_model_cir_fusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOGIT_SCALE_STATE_FILENAME = "logit_scale.pt"


def get_similarity_scale(
    logit_scale: Union[torch.Tensor, float],
):
    """Return the multiplicative similarity scale used by CLIP-style losses."""
    if logit_scale is None:
        raise ValueError("logit_scale is required.")

    if isinstance(logit_scale, torch.Tensor):
        return logit_scale.float().exp()

    return float(np.exp(float(logit_scale)))


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


def enable_model_logit_scale_training(model, log: Optional[logging.Logger] = None) -> Optional[torch.Tensor]:
    """Ensure logit_scale stays trainable even when PEFT freezes base parameters."""
    logit_scale = get_model_logit_scale(model)
    if logit_scale is None:
        if log is not None:
            log.warning("Could not find CLIP logit_scale parameter.")
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


def _iter_wrapped_models(model):
    """Yield a model and common wrapper children without assuming a wrapper type."""
    seen = set()
    stack = [model]

    while stack:
        candidate = stack.pop(0)
        if candidate is None or id(candidate) in seen:
            continue

        seen.add(id(candidate))
        yield candidate

        for attr in ("module", "base_model", "model"):
            child = getattr(candidate, attr, None)
            if child is not None and child is not candidate:
                stack.append(child)


def _get_clip_feature_method(model, method_name: str):
    for candidate in _iter_wrapped_models(model):
        method = getattr(candidate, method_name, None)
        if callable(method):
            return method

    raise AttributeError(
        f"Unsupported CLIP model wrapper: expected `{method_name}` on the model or one of its children."
    )


def _coerce_clip_feature_output(output, feature_name: str) -> torch.Tensor:
    """Normalize CLIP feature outputs across Transformers versions."""
    if isinstance(output, torch.Tensor):
        features = output
    else:
        features = None
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                features = value
                break

        if features is None and isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, torch.Tensor):
                    features = value
                    break

        if features is None:
            raise TypeError(
                f"{feature_name} must be a tensor or a CLIP output containing tensor features, "
                f"got {type(output)!r}"
            )

    if features.dim() == 3:
        features = features.mean(dim=1)

    if features.dim() != 2:
        raise ValueError(f"{feature_name} must have shape [B, D], got {tuple(features.shape)}")

    return features


def extract_clip_image_features(clip_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return projected CLIP image embeddings as a tensor."""
    get_image_features = _get_clip_feature_method(clip_model, "get_image_features")
    return _coerce_clip_feature_output(
        get_image_features(pixel_values=pixel_values),
        "image_features",
    )


def extract_clip_text_features(clip_model, **tokenized_inputs) -> torch.Tensor:
    """Return projected CLIP text embeddings as a tensor."""
    get_text_features = _get_clip_feature_method(clip_model, "get_text_features")
    return _coerce_clip_feature_output(
        get_text_features(**tokenized_inputs),
        "text_features",
    )


def _move_tokenized_inputs_to_device(tokenized_inputs, device: torch.device):
    if hasattr(tokenized_inputs, "to"):
        return tokenized_inputs.to(device)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in tokenized_inputs.items()
    }


def _tokenize_texts_for_intent_learning(
    tokenizer,
    texts: List[str],
    device: torch.device,
):
    tokenized_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return _move_tokenized_inputs_to_device(tokenized_inputs, device)


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
    ):
        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        self.topk_candidates = topk_candidates
        self.refresh_interval = refresh_interval
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Embeddings storage (PyTorch tensors on GPU)
        self.target_embeddings: Optional[torch.Tensor] = None
        self.target_names: Optional[List[str]] = None
        self.name_to_idx: dict = {}
        
        # Training state
        self.last_refresh_epoch = -1
        self.is_initialized = False
    
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
                image_features = extract_clip_image_features(clip_model, pixel_values=images)
                image_features = F.normalize(image_features, dim=-1)
                image_features = compose_image_features(clip_model, image_features)
            if self.use_gpu:
                all_features.append(image_features)
            else:
                all_features.append(image_features.cpu())
            all_names.extend(names)
        
        self.target_embeddings = torch.cat(all_features, dim=0)
        if not self.use_gpu:
            self.target_embeddings = self.target_embeddings.cpu()
        
        self.target_names = all_names
        self.name_to_idx = {name: idx for idx, name in enumerate(all_names)}
        
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
        """Convert positive_names to tensor indices in batch."""
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
        exclude_reference_names: Optional[List[str]] = None,
        additional_exclude_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        Vectorized hard-negative mining with minimal CPU work.
        
        Returns:
            hard_negative_indices: [B, num_negatives] on GPU
            hard_negative_names: List[List[str]]
        """
        if not self.is_initialized:
            raise RuntimeError("Index not initialized.")
        
        device = self.target_embeddings.device
        query_features = F.normalize(query_features.to(device), dim=-1)
        batch_size = query_features.shape[0]
        
        # 1. Search top-k candidates on GPU.
        _, topk_indices = self._search_topk(query_features, self.topk_candidates)
        # topk_indices: [B, topk_candidates] on GPU
        
        # 2. Get positive indices as a GPU tensor.
        positive_indices = self._get_positive_indices_tensor(positive_names, device)  # [B]
        
        # 3. Build a GPU mask to exclude positive samples.
        # positive_indices: [B, 1], topk_indices: [B, topk_candidates]
        positive_mask = (topk_indices == positive_indices.unsqueeze(1))  # [B, topk_candidates]
        
        # 4. Also exclude reference names when provided.
        if exclude_reference_names:
            exclude_indices = self._get_positive_indices_tensor(exclude_reference_names, device)
            exclude_mask = (topk_indices == exclude_indices.unsqueeze(1))
            positive_mask = positive_mask | exclude_mask

        if additional_exclude_indices is not None:
            additional_exclude_indices = additional_exclude_indices.to(device)
            if additional_exclude_indices.dim() == 1:
                additional_exclude_indices = additional_exclude_indices.unsqueeze(1)
            valid_additional = additional_exclude_indices >= 0
            additional_mask = (
                topk_indices.unsqueeze(-1) == additional_exclude_indices.unsqueeze(1)
            ) & valid_additional.unsqueeze(1)
            positive_mask = positive_mask | additional_mask.any(dim=-1)
        
        # 5. Filter out excluded indices directly.
        
        # 6. Mark excluded positions as -1.
        filtered_indices = topk_indices.clone()
        filtered_indices[positive_mask] = -1
        
        # 7. For each sample, take the first num_negatives non--1 indices.
        # topk is already similarity-sorted, so the first valid entries are enough.
        hard_negative_indices = torch.zeros(batch_size, self.num_negatives, dtype=torch.long, device=device)
        
        # This still loops per sample, but the work stays on GPU tensors.
        for i in range(batch_size):
            valid_mask = filtered_indices[i] != -1
            valid_indices = filtered_indices[i][valid_mask]
            
            if len(valid_indices) >= self.num_negatives:
                hard_negative_indices[i] = valid_indices[:self.num_negatives]
            else:
                # Backfill if there are not enough valid candidates.
                hard_negative_indices[i, :len(valid_indices)] = valid_indices
                # Randomly sample the remaining entries on GPU.
                n_needed = self.num_negatives - len(valid_indices)
                random_indices = torch.randint(
                    0, len(self.target_names), (n_needed,), 
                    device=device, dtype=torch.long
                )
                hard_negative_indices[i, len(valid_indices):] = random_indices
        
        # 8. Convert to names on CPU after all indices are ready.
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
        additional_exclude_indices: Optional[torch.Tensor] = None,
        return_names: bool = True
    ) -> Union[Tuple[torch.Tensor, List[List[str]]], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Mine hard negatives using the vectorized implementation.
        """
        indices, names = self.mine_hard_negatives_vectorized(
            query_features, positive_names, exclude_reference_names, additional_exclude_indices
        )
        
        if return_names:
            return indices, names
        else:
            hard_negative_features = self.target_embeddings[indices]
            return indices, hard_negative_features
    
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
        Mine negatives using pregenerated partial-intent queries and reference images.

        For each sample, encode the partial-intent text, compose it with the
        reference image features, and retrieve nearest neighbors from the index
        as negatives.

        Args:
            partial_intent_texts: Partial-intent text for each sample.
            ref_features: [B, D] reference image features
            positive_names: Positive sample names.
            hard_negative_indices: [B, N] existing hard-negative indices to exclude.
            num_negatives: Number of negatives needed for each sample.
            clip_model: CLIP model.
            tokenizer: CLIP tokenizer

        Returns:
            partial_intent_neg_indices: [B, num_negatives]
            partial_intent_neg_names: List[List[str]]
        """
        if not self.is_initialized:
            raise RuntimeError("Index not initialized.")
        if clip_model is None or tokenizer is None:
            raise ValueError("clip_model and tokenizer are required.")

        index_device = self.target_embeddings.device
        model_device = ref_features.device
        batch_size = ref_features.shape[0]
        ref_features = F.normalize(ref_features.to(model_device), dim=-1)
        hard_negative_indices = hard_negative_indices.to(index_device)
        positive_indices = self._get_positive_indices_tensor(positive_names, index_device)

        inputs = tokenizer(
            partial_intent_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(model_device)
        with torch.cuda.amp.autocast():
            partial_text_features = extract_clip_text_features(clip_model, **inputs)
            partial_text_features = F.normalize(partial_text_features, dim=-1)

        partial_query = compose_query_features(clip_model, ref_features, partial_text_features)

        _, topk_indices = self._search_topk(partial_query, self.topk_candidates)

        exclude_mask = (topk_indices == positive_indices.unsqueeze(1))
        for k in range(hard_negative_indices.shape[1]):
            exclude_mask = exclude_mask | (topk_indices == hard_negative_indices[:, k:k+1])

        filtered_indices = topk_indices.clone()
        filtered_indices[exclude_mask] = -1

        partial_intent_neg_indices = torch.zeros(
            batch_size, num_negatives, dtype=torch.long, device=index_device
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
                    device=index_device, dtype=torch.long
                )
                partial_intent_neg_indices[i, len(valid_indices):] = random_indices

        indices_cpu = partial_intent_neg_indices.cpu().numpy()
        partial_intent_neg_names = [
            [self.target_names[idx] for idx in indices_cpu[i]]
            for i in range(batch_size)
        ]

        return partial_intent_neg_indices, partial_intent_neg_names
def _ensure_feature_tensor(
    features: Optional[Union[np.ndarray, torch.Tensor]],
    device: torch.device,
    name: str,
) -> Optional[torch.Tensor]:
    """Convert numpy features to torch and align them to the query device."""
    if features is None:
        return None

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()

    if not isinstance(features, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or np.ndarray, got {type(features)!r}")

    if features.device != device:
        features = features.to(device, non_blocking=True)

    return features


def _compute_target_similarity_matrix(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """Compute query-to-target similarities for both vector and token-level targets."""
    if target_features.dim() == 2:
        return torch.matmul(query_features.float(), target_features.float().T)

    if target_features.dim() == 3:
        return torch.einsum("id,jqd->ijq", query_features.float(), target_features.float()).max(dim=-1).values

    raise ValueError(
        f"target_features must have shape [B, D] or [B, Q, D], got {tuple(target_features.shape)}"
    )


def _compute_group_similarities(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
    """Compute similarities between a query batch and batched candidate groups."""
    if candidate_features.dim() == 2:
        candidate_features = candidate_features.unsqueeze(1)

    if candidate_features.dim() != 3:
        raise ValueError(
            f"{group_name} must have shape [B, N, D] or [B, D], got {tuple(candidate_features.shape)}"
        )

    return torch.bmm(
        query_features.float().unsqueeze(1),
        candidate_features.float().transpose(1, 2),
    ).squeeze(1)


def compute_clip_ance_listwise_loss(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    hard_negative_features: torch.Tensor,
    hard_negative_weight: float = 1.0,
    ref_hard_negative_features: Optional[torch.Tensor] = None,
    ref_hard_negative_weight: float = 1.0,
    partial_intent_negative_features: Optional[torch.Tensor] = None,
    partial_intent_negative_weight: float = 0.75,
    logit_scale: Optional[Union[torch.Tensor, float]] = None,
    **legacy_kwargs,
) -> torch.Tensor:
    """
    ListNet top-one ranking loss over the ordered candidate list:
    target image > hard negatives > partial-intent negatives >
    reference-image negatives > in-batch samples.

    The target distribution is built from manually assigned group-level
    relevance scores while the prediction distribution comes from query-candidate
    similarities.
    """
    device = query_features.device

    # Backward compatibility with older training scripts.
    if partial_intent_negative_features is None and legacy_kwargs.get("text_intent_negative_features") is not None:
        partial_intent_negative_features = legacy_kwargs["text_intent_negative_features"]
    if "text_intent_negative_weight" in legacy_kwargs:
        partial_intent_negative_weight = legacy_kwargs["text_intent_negative_weight"]

    target_features = _ensure_feature_tensor(target_features, device, "target_features")
    hard_neg_tensor = _ensure_feature_tensor(hard_negative_features, device, "hard_negative_features")
    ref_hard_neg_tensor = _ensure_feature_tensor(
        ref_hard_negative_features,
        device,
        "ref_hard_negative_features",
    )
    partial_intent_neg_tensor = _ensure_feature_tensor(
        partial_intent_negative_features,
        device,
        "partial_intent_negative_features",
    )

    query_features = F.normalize(query_features, dim=-1)
    target_features = F.normalize(target_features, dim=-1)
    hard_neg_tensor = F.normalize(hard_neg_tensor, dim=-1)
    if ref_hard_neg_tensor is not None:
        ref_hard_neg_tensor = F.normalize(ref_hard_neg_tensor, dim=-1)
    if partial_intent_neg_tensor is not None:
        partial_intent_neg_tensor = F.normalize(partial_intent_neg_tensor, dim=-1)

    batch_size = query_features.shape[0]
    target_sim_matrix = _compute_target_similarity_matrix(query_features, target_features)
    positive_sim = torch.diagonal(target_sim_matrix, dim1=0, dim2=1).unsqueeze(1)

    ordered_groups: List[Tuple[str, torch.Tensor]] = [("target", positive_sim)]

    hard_sim = _compute_group_similarities(query_features, hard_neg_tensor, "hard_negative_features")
    if hard_sim.shape[1] > 0:
        ordered_groups.append(("hard", hard_sim))

    if partial_intent_neg_tensor is not None:
        partial_sim = _compute_group_similarities(
            query_features,
            partial_intent_neg_tensor,
            "partial_intent_negative_features",
        )
        if partial_sim.shape[1] > 0:
            ordered_groups.append(("partial_intent", partial_sim))

    if ref_hard_neg_tensor is not None:
        ref_sim = _compute_group_similarities(
            query_features,
            ref_hard_neg_tensor,
            "ref_hard_negative_features",
        )
        if ref_sim.shape[1] > 0:
            ordered_groups.append(("ref_hard", ref_sim))

    if batch_size > 1:
        in_batch_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        in_batch_sim = target_sim_matrix.masked_select(in_batch_mask).view(batch_size, batch_size - 1)
        if in_batch_sim.shape[1] > 0:
            ordered_groups.append(("in_batch", in_batch_sim))

    if len(ordered_groups) == 1:
        return torch.zeros((), device=device, dtype=query_features.dtype)

    scale = get_similarity_scale(logit_scale=logit_scale)
    logits = torch.cat([group_sim.float() for _, group_sim in ordered_groups], dim=1) * scale

    # raw_group_scores = {
    #     "target": 4.0,
    #     "hard": 3.0 + float(np.log(max(float(hard_negative_weight), 1e-8))),
    #     "partial_intent": 2.0 + float(np.log(max(float(partial_intent_negative_weight), 1e-8))),
    #     "ref_hard": 1.0 + float(np.log(max(float(ref_hard_negative_weight), 1e-8))),
    #     "in_batch": 0.0,
    # }
    raw_group_scores = {
        "target": 4.0,
        "hard": 3.0,
        "partial_intent": 2.0,
        "ref_hard": 1.0,
        "in_batch": 0.0,
    }

    target_score_blocks = []
    previous_score = None
    for group_name, group_sim in ordered_groups:
        group_score = raw_group_scores[group_name]
        if previous_score is not None:
            group_score = min(group_score, previous_score - 1e-3)
        target_score_blocks.append(
            torch.full(
                group_sim.shape,
                fill_value=group_score,
                device=device,
                dtype=torch.float32,
            )
        )
        previous_score = group_score

    target_scores = torch.cat(target_score_blocks, dim=1)
    target_probs = F.softmax(target_scores, dim=-1)
    log_pred_probs = F.log_softmax(logits, dim=-1)

    return -(target_probs * log_pred_probs).sum(dim=-1).mean()


def compute_intent_consistency_loss(
    model,
    tokenizer,
    ref_features: torch.Tensor,
    query_features: torch.Tensor,
    input_captions: List[str],
    single_intent_map: Optional[dict],
    device: torch.device,
    global_consistency_weight: float = 0.5,
    global_consistency_temperature: float = 0.2,
    consistency_epsilon: float = 0.05,
    return_stats: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Counterfactual-delta Intent Consistency Learning.

    For a multi-intent query T=(A_1, ..., A_k), the contextual representation
    of intent A_i is the marginal effect of that intent on the composed query:

        d_ctx_i = q_full - q_without_i

    It is aligned with the isolated single-intent effect:

        d_iso_i = q_i - q_ref

    The optional global term reconstructs the full edit direction
    q_full - q_ref from a relevance-weighted sum of isolated deltas.

    Active only for cross-attention fusion modules; other fusion types return 0.
    """
    def _join_remaining_intents(intents: List[str], removed_position: int) -> str:
        remaining = [
            intent.strip()
            for pos, intent in enumerate(intents)
            if pos != removed_position and intent.strip()
        ]
        return " ".join(remaining)

    def _empty_result():
        zero = torch.zeros((), device=device)
        stats = {
            "loss_ic": zero,
            "loss_ic_local": zero,
            "loss_ic_global": zero,
            "ic_valid_spans": zero,
            "ic_skipped_spans": zero,
            "ic_valid_queries": zero,
            "ic_multi_queries": zero,
        }
        if return_stats:
            return zero, stats
        return zero

    if single_intent_map is None or len(single_intent_map) == 0:
        return _empty_result()

    fusion_module = get_model_cir_fusion(model)
    if fusion_module is None or getattr(fusion_module, "cross_attn", None) is None:
        return _empty_result()

    multi_indices: List[int] = []
    multi_intent_texts: List[List[str]] = []
    for i, cap in enumerate(input_captions):
        intents = single_intent_map.get(cap)
        if isinstance(intents, list) and len(intents) >= 2:
            multi_indices.append(i)
            multi_intent_texts.append(intents)

    if len(multi_indices) == 0:
        return _empty_result()

    flat_intents: List[str] = []
    for intents in multi_intent_texts:
        flat_intents.extend(intents)

    if len(flat_intents) == 0:
        return _empty_result()

    global_consistency_weight = float(global_consistency_weight)
    global_consistency_temperature = max(float(global_consistency_temperature), 1e-6)
    consistency_epsilon = max(float(consistency_epsilon), 0.0)

    intent_inputs = _tokenize_texts_for_intent_learning(
        tokenizer=tokenizer,
        texts=flat_intents,
        device=device,
    )

    without_intent_texts: List[str] = []
    for intents in multi_intent_texts:
        for intent_position in range(len(intents)):
            without_intent_texts.append(_join_remaining_intents(intents, intent_position))
    without_inputs = _tokenize_texts_for_intent_learning(
        tokenizer=tokenizer,
        texts=without_intent_texts,
        device=device,
    )

    with torch.cuda.amp.autocast():
        intent_text_features = extract_clip_text_features(model, **intent_inputs)
        intent_text_features = F.normalize(intent_text_features, dim=-1)

        # Reuse the corresponding reference image feature for each single intent.
        repeat_indices: List[int] = []
        for idx, intents in zip(multi_indices, multi_intent_texts):
            repeat_indices.extend([idx] * len(intents))
        repeated_ref_features = ref_features[repeat_indices]

        e_iso_all = fusion_module(repeated_ref_features, intent_text_features)

        selected_ref_features = ref_features[multi_indices]
        with torch.no_grad():
            ref_anchor_features = compose_image_features(model, selected_ref_features)
            without_text_features = extract_clip_text_features(model, **without_inputs)
            without_text_features = F.normalize(without_text_features, dim=-1)
            q_without_all = fusion_module(repeated_ref_features, without_text_features)

    local_losses = []
    global_losses = []
    valid_span_count = 0
    valid_query_count = 0
    offset = 0
    for j, idx in enumerate(multi_indices):
        num_intents = len(multi_intent_texts[j])
        end = offset + num_intents

        e_iso_sample = e_iso_all[offset:end]
        q_without = q_without_all[offset:end]
        q_full = query_features[idx].unsqueeze(0).expand_as(q_without)
        ref_anchor = ref_anchor_features[j].unsqueeze(0)
        ctx_delta = q_full - q_without
        iso_delta = e_iso_sample - ref_anchor
        local_ctx = F.normalize(ctx_delta, dim=-1)
        local_iso = F.normalize(iso_delta, dim=-1)

        valid_query_count += 1
        valid_span_count += e_iso_sample.shape[0]

        cos_sim_local = F.cosine_similarity(local_ctx, local_iso, dim=-1)
        loss_local = F.relu((1.0 - cos_sim_local) - consistency_epsilon).mean()
        local_losses.append(loss_local)

        if global_consistency_weight > 0.0:
            full_query_teacher = F.normalize(query_features[idx].detach(), dim=-1)
            ref_anchor = ref_anchor_features[j]
            full_query_teacher_for_loss = F.normalize(
                full_query_teacher - ref_anchor,
                dim=-1,
            )
            with torch.no_grad():
                intent_sims = (local_iso * full_query_teacher_for_loss.unsqueeze(0)).sum(dim=-1)
                intent_weights = F.softmax(
                    intent_sims / global_consistency_temperature,
                    dim=0,
                )
            weighted_delta = (intent_weights.unsqueeze(-1) * iso_delta).sum(dim=0)
            weighted_features_for_loss = F.normalize(weighted_delta, dim=-1)

            cos_sim_global = (
                full_query_teacher_for_loss * weighted_features_for_loss
            ).sum().clamp(min=-1.0, max=1.0)
            loss_global = F.relu((1.0 - cos_sim_global) - consistency_epsilon)
            global_losses.append(loss_global)
        offset = end

    if len(local_losses) == 0 or valid_span_count == 0:
        return _empty_result()

    loss_local = torch.stack(local_losses).mean()
    zero = torch.zeros((), device=device)
    loss_global = torch.stack(global_losses).mean() if global_losses else zero

    loss_ic = loss_local + global_consistency_weight * loss_global
    if not return_stats:
        return loss_ic

    stats = {
        "loss_ic": loss_ic.detach(),
        "loss_ic_local": loss_local.detach(),
        "loss_ic_global": loss_global.detach(),
        "ic_valid_spans": torch.tensor(float(valid_span_count), device=device),
        "ic_skipped_spans": zero,
        "ic_valid_queries": torch.tensor(float(valid_query_count), device=device),
        "ic_multi_queries": torch.tensor(float(len(multi_indices)), device=device),
    }
    return loss_ic, stats


def _compute_soft_orthogonality_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Soft orthogonality penalty:
        2 / (K * (K - 1)) * sum_{i<j} (<e_i, e_j>)^2
    after L2-normalizing embeddings along the feature dimension.
    """
    num_embeddings = embeddings.shape[0]
    if num_embeddings < 2:
        return embeddings.new_zeros(())

    normalized_embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
    gram = torch.mm(normalized_embeddings, normalized_embeddings.t())
    pair_mask = torch.triu(
        torch.ones(
            num_embeddings,
            num_embeddings,
            device=embeddings.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    pairwise_inner_products = gram[pair_mask]
    normalization = 2.0 / float(num_embeddings * (num_embeddings - 1))
    return pairwise_inner_products.square().sum() * normalization


def compute_intent_orthogonality_loss(
    model,
    tokenizer,
    ref_features: torch.Tensor,
    query_features: torch.Tensor,
    input_captions: List[str],
    single_intent_map: Optional[dict],
    device: torch.device,
) -> torch.Tensor:
    """
    Soft orthogonality loss between intents.

    For a multi-intent query T=(A_1, ..., A_k) with k >= 2, encourage the
    retrieval-space query features produced by the fusion head for each single
    intent to be mutually orthogonal:

        <f(I, A_i), f(I, A_j)> ~= 0    (i != j)

    This encourages query representations for different intents to be
    disentangled, so one intent is less likely to be pulled by another. The
    embeddings are first L2-normalized along the feature dimension, then the
    differentiable soft orthogonality penalty is computed as:
    2 / (K * (K - 1)) * sum_{i<j} (<e_i, e_j>)^2

    Active only for cross-attention fusion modules; other fusion types return 0.
    """
    if single_intent_map is None or len(single_intent_map) == 0:
        return torch.zeros((), device=device)

    fusion_module = get_model_cir_fusion(model)
    if fusion_module is None or getattr(fusion_module, "cross_attn", None) is None:
        return torch.zeros((), device=device)

    multi_indices: List[int] = []
    multi_intent_texts: List[List[str]] = []
    for i, cap in enumerate(input_captions):
        intents = single_intent_map.get(cap)
        if isinstance(intents, list) and len(intents) >= 2:
            multi_indices.append(i)
            multi_intent_texts.append(intents)

    if len(multi_indices) == 0:
        return torch.zeros((), device=device)

    flat_intents: List[str] = []
    for intents in multi_intent_texts:
        flat_intents.extend(intents)

    if len(flat_intents) == 0:
        return torch.zeros((), device=device)

    inputs = tokenizer(
        flat_intents,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.cuda.amp.autocast():
        single_text_features = extract_clip_text_features(model, **inputs)
        single_text_features = F.normalize(single_text_features, dim=-1)

        repeat_indices: List[int] = []
        for idx, intents in zip(multi_indices, multi_intent_texts):
            repeat_indices.extend([idx] * len(intents))
        repeated_ref_features = ref_features[repeat_indices]

        # Isolated single-intent fusion outputs: [sum_k, D]
        single_query_features = fusion_module(repeated_ref_features, single_text_features)

    losses = []
    offset = 0
    for j in range(len(multi_indices)):
        num_intents = len(multi_intent_texts[j])
        end = offset + num_intents

        # Compute soft orthogonality directly from each single-intent fused query feature.
        single_queries = single_query_features[offset:end]  # [k, D]
        losses.append(_compute_soft_orthogonality_loss(single_queries))

        offset = end

    if len(losses) == 0:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def compute_clip_ance_loss(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    hard_negative_features: torch.Tensor,
    hard_negative_weight: float = 1.0,
    ref_hard_negative_features: Optional[torch.Tensor] = None,
    ref_hard_negative_weight: float = 1.0,
    partial_intent_negative_features: Optional[torch.Tensor] = None,
    partial_intent_negative_weight: float = 0.75,
    listwise_weight: float = 0.2,
    logit_scale: Optional[Union[torch.Tensor, float]] = None,
    **legacy_kwargs,
) -> torch.Tensor:
    """Hybrid ANCE loss with in-batch CE as the primary objective."""
    device = query_features.device

    if "listwise_weight" in legacy_kwargs:
        listwise_weight = legacy_kwargs["listwise_weight"]

    query_features = F.normalize(query_features, dim=-1)
    target_features = _ensure_feature_tensor(target_features, device, "target_features")
    target_features = F.normalize(target_features, dim=-1)

    target_sim_matrix = _compute_target_similarity_matrix(query_features, target_features)
    scale = get_similarity_scale(logit_scale=logit_scale)
    labels = torch.arange(query_features.shape[0], dtype=torch.long, device=device)
    loss_in_batch = F.cross_entropy(target_sim_matrix * scale, labels)

    if float(listwise_weight) <= 0.0:
        return loss_in_batch

    listwise_loss = compute_clip_ance_listwise_loss(
        query_features=query_features,
        target_features=target_features,
        hard_negative_features=hard_negative_features,
        hard_negative_weight=hard_negative_weight,
        ref_hard_negative_features=ref_hard_negative_features,
        ref_hard_negative_weight=ref_hard_negative_weight,
        partial_intent_negative_features=partial_intent_negative_features,
        partial_intent_negative_weight=partial_intent_negative_weight,
        logit_scale=logit_scale,
        **legacy_kwargs,
    )
    return loss_in_batch + float(listwise_weight) * listwise_loss
