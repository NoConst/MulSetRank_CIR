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
        exclude_reference_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mine hard negatives for a batch of composed queries.
        
        Args:
            query_features: Composed query embeddings (batch_size, dim)
                           This should be the element-wise sum of image and text features
            positive_names: List of positive target names to exclude
            exclude_reference_names: Additional names to exclude (e.g., reference images)
            
        Returns:
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
        
        # Gather hard negative features
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


def compute_clip_ance_loss(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    hard_negative_features: torch.Tensor,
    temperature: float = 0.07,
    hard_negative_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute ANCE-style contrastive loss for CLIP CIR.
    
    The loss combines:
    1. In-batch contrastive loss: query vs all targets in the batch
    2. Hard negative loss: query vs hard negatives from FAISS index
    
    Args:
        query_features: Composed query features (batch_size, dim)
                       Element-wise sum of reference image + text features
        target_features: Positive target features (batch_size, dim)
        hard_negative_features: Hard negative features (batch_size, num_negatives, dim)
        temperature: Temperature scaling factor
        hard_negative_weight: Weight for hard negative loss
        
    Returns:
        Combined contrastive loss
    """
    batch_size = query_features.size(0)
    device = query_features.device
    
    # Ensure features are normalized
    query_features = F.normalize(query_features, dim=-1)
    target_features = F.normalize(target_features, dim=-1)
    
    # ============ Part 1: In-batch contrastive loss ============
    # Compute similarity matrix: query @ target.T -> (B, B)
    sim_matrix = torch.matmul(query_features, target_features.T) / temperature
    
    # Labels: diagonal is positive
    labels = torch.arange(batch_size, dtype=torch.long, device=device)
    
    # In-batch InfoNCE loss
    loss_inbatch = F.cross_entropy(sim_matrix, labels)
    
    # ============ Part 2: Hard negative contrastive loss ============
    # Compute positive similarity: (B,)
    sim_pos = (query_features * target_features).sum(dim=-1)
    
    # Convert hard negatives to tensor if needed
    hard_neg_tensor = torch.from_numpy(hard_negative_features).float().to(device) \
        if isinstance(hard_negative_features, np.ndarray) else hard_negative_features
    
    # Normalize hard negatives and compute similarity
    if hard_neg_tensor.dim() == 2:
        hard_neg_tensor = hard_neg_tensor.unsqueeze(1)
    
    hard_neg_tensor = F.normalize(hard_neg_tensor, dim=-1)
    
    # Similarity with hard negatives: (B, 1, D) @ (B, D, N) -> (B, N)
    sim_hard = torch.bmm(
        query_features.unsqueeze(1),
        hard_neg_tensor.permute(0, 2, 1)
    ).squeeze(1)
    
    # Logits: [positive, hard_negatives]
    logits = torch.cat([sim_pos.unsqueeze(1), sim_hard], dim=1) / temperature
    
    # Labels: positive is at index 0
    labels_hard = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    loss_hard = F.cross_entropy(logits, labels_hard)
    
    # ============ Combined loss ============
    total_loss = loss_inbatch + hard_negative_weight * loss_hard
    
    # return total_loss
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

