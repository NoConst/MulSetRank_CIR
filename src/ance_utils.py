"""
ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation) utilities
for hard negative sampling in Composed Image Retrieval training.
"""

import os
import torch
import numpy as np
import faiss
from typing import Tuple, List, Dict, Optional, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    ANCE-style hard negative miner using FAISS for approximate nearest neighbor search.
    
    This class maintains a FAISS index of all target image features and provides
    methods to mine hard negatives for training queries.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_negatives: int = 16,
        topk_candidates: int = 100,
        refresh_interval: int = 1,  # Refresh every N epochs
        use_gpu: bool = False,  # Default to CPU to avoid GPU memory conflicts with PyTorch
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the hard negative miner.
        
        Args:
            embedding_dim: Dimension of the feature embeddings
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
        """Create a FAISS index for the embeddings."""
        dim = embeddings.shape[1]
        num_embeddings = embeddings.shape[0]
        
        logger.info(f"Creating FAISS index for {num_embeddings} embeddings with dim={dim}")
        
        # Use inner product (cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(dim)
        
        if self.use_gpu:
            try:
                # Use GPU 0 in the visible device context (respects CUDA_VISIBLE_DEVICES)
                res = faiss.StandardGpuResources()
                # Limit GPU memory usage to avoid OOM
                res.setTempMemory(128 * 1024 * 1024)  # 128MB
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Using GPU for FAISS index")
            except Exception as e:
                logger.warning(f"Failed to use GPU for FAISS: {e}. Falling back to CPU.")
                # Recreate CPU index since GPU conversion may have corrupted it
                index = faiss.IndexFlatIP(dim)
        
        logger.info("Adding embeddings to FAISS index...")
        index.add(embeddings)
        logger.info(f"FAISS index built successfully with {index.ntotal} vectors")
        return index
    
    @torch.no_grad()
    def build_index(
        self,
        model,
        dataset,
        device: torch.device,
        batch_size: int = 64,
        num_workers: int = 4
    ):
        """
        Build the FAISS index from target image features.
        
        Args:
            model: The BLIP model for feature extraction
            dataset: Dataset in 'classic' mode containing target images
            device: torch device
            batch_size: Batch size for feature extraction
            num_workers: Number of data loading workers
        """
        logger.info("Building FAISS index for hard negative mining...")
        
        from utils import collate_fn
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        model.eval()
        all_features = []
        all_names = []
        
        for names, images in tqdm(dataloader, desc="Extracting target features"):
            images = images.to(device, non_blocking=True)
            
            # Extract target image features using the model
            image_features, _ = model.extract_target_features(images, mode="mean")
            
            # Use mean pooling across query tokens for a single vector per image
            if image_features.dim() == 3:
                image_features = image_features.mean(dim=1)
            
            # Normalize features
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            
            all_features.append(image_features.cpu().numpy())
            all_names.extend(names)
            
            # Clear GPU cache periodically
            if len(all_features) % 100 == 0:
                torch.cuda.empty_cache()
        
        logger.info("Feature extraction completed. Concatenating features...")
        
        # Concatenate all features
        self.target_embeddings = np.vstack(all_features).astype('float32')
        self.target_names = all_names
        self.name_to_idx = {name: idx for idx, name in enumerate(all_names)}
        
        logger.info(f"Total features shape: {self.target_embeddings.shape}")
        
        # Clear memory before building index
        del all_features
        torch.cuda.empty_cache()
        
        # Build FAISS index
        logger.info("Starting FAISS index construction...")
        self.index = self._create_index(self.target_embeddings)
        self.is_initialized = True
        
        logger.info(f"Built index with {len(self.target_names)} target images")
        
        return self.target_embeddings, self.target_names
    
    def refresh_index(
        self,
        model,
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
            self.build_index(model, dataset, device, batch_size, num_workers)
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
        Mine hard negatives for a batch of queries.
        
        Args:
            query_features: Query embeddings of shape (batch_size, dim) or (batch_size, num_queries, dim)
            positive_names: List of positive target names to exclude
            exclude_reference_names: Additional names to exclude (e.g., reference images)
            
        Returns:
            hard_negative_indices: Indices of hard negatives (batch_size, num_negatives)
            hard_negative_features: Features of hard negatives (batch_size, num_negatives, dim)
        """
        if not self.is_initialized:
            raise RuntimeError("Index not initialized. Call build_index first.")
        
        # Handle 3D features by mean pooling
        if query_features.dim() == 3:
            query_features = query_features.mean(dim=1)
        
        # Normalize and convert to numpy
        query_features = torch.nn.functional.normalize(query_features, dim=-1)
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
                # Sample random indices as fallback
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
    
    def save_cache(self, epoch: int):
        """Save the current embeddings to cache."""
        if self.cache_dir and self.is_initialized:
            cache_file = self.cache_dir / f"target_embeddings_epoch_{epoch}.npz"
            np.savez(
                cache_file,
                embeddings=self.target_embeddings,
                names=np.array(self.target_names)
            )
            logger.info(f"Saved embeddings cache to {cache_file}")
    
    def load_cache(self, epoch: int) -> bool:
        """Load embeddings from cache if available."""
        if self.cache_dir:
            cache_file = self.cache_dir / f"target_embeddings_epoch_{epoch}.npz"
            if cache_file.exists():
                data = np.load(cache_file, allow_pickle=True)
                self.target_embeddings = data['embeddings']
                self.target_names = data['names'].tolist()
                self.name_to_idx = {name: idx for idx, name in enumerate(self.target_names)}
                self.index = self._create_index(self.target_embeddings)
                self.is_initialized = True
                logger.info(f"Loaded embeddings cache from {cache_file}")
                return True
        return False


class ANCETrainingDataset:
    """
    Wrapper that adds hard negative sampling capability to existing datasets.
    """
    
    def __init__(
        self,
        base_dataset,
        hard_negative_miner: HardNegativeMiner,
        model,
        device: torch.device,
        txt_processors=None
    ):
        self.base_dataset = base_dataset
        self.miner = hard_negative_miner
        self.model = model
        self.device = device
        self.txt_processors = txt_processors
        
        # Cache for query features (updated periodically)
        self.query_features_cache = {}
    
    @torch.no_grad()
    def compute_query_features(
        self,
        reference_images: torch.Tensor,
        captions: List[str]
    ) -> torch.Tensor:
        """
        Compute query features (fusion of reference image + text).
        """
        self.model.eval()
        
        # Process captions if text processor is provided
        if self.txt_processors:
            captions = [self.txt_processors["eval"](cap) for cap in captions]
        
        reference_images = reference_images.to(self.device)
        
        # Extract reference image embeddings
        image_embeds = self.model.ln_vision(self.model.visual_encoder(reference_images))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        
        # Get query tokens
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
        
        # Tokenize text
        text_tokens = self.model.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        
        # Fuse image and text
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.model.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        text_output = self.model.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, :query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Get fusion features (same as in forward pass)
        fusion_feats = torch.nn.functional.normalize(
            self.model.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )
        
        return fusion_feats


def compute_ance_loss(
    fusion_feats: torch.Tensor,
    target_feats: torch.Tensor,
    hard_negative_feats: torch.Tensor,
    temperature: float = 0.07,
    hard_negative_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute contrastive loss with hard negatives.
    
    The loss consists of two parts:
    1. In-batch contrastive loss: following BLIP2 style (fusion_feats vs all target_feats in batch)
    2. Hard negative loss: positive vs hard negatives from FAISS index
    
    Args:
        fusion_feats: Query features (batch_size, dim)
        target_feats: Positive target features (batch_size, num_queries, dim) or (batch_size, dim)
        hard_negative_feats: Hard negative features (batch_size, num_negatives, dim)
        temperature: Temperature scaling factor
        hard_negative_weight: Weight for hard negative loss (default 1.0)
        
    Returns:
        Combined loss = inbatch_loss + hard_negative_weight * hard_negative_loss
    """
    batch_size = fusion_feats.size(0)
    device = fusion_feats.device
    
    # Ensure fusion_feats is 2D: (B, D)
    if fusion_feats.dim() == 1:
        fusion_feats = fusion_feats.unsqueeze(0)
    
    # ============ Part 1: In-batch contrastive loss (BLIP2 style) ============
    # Reference: blip2_qformer_cir_align_prompt.py forward()
    
    if target_feats.dim() == 3:
        # fusion_feats: (B, D), target_feats: (B, Q, D)
        # Compute similarity: (B, 1, 1, D) x (B, D, Q) -> (B, B, Q)
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1),  # (B, 1, 1, D)
            target_feats.permute(0, 2, 1)  # (B, D, Q)
        ).squeeze()  # (B, B, Q) or (B, Q) if B=1
        
        # Handle batch size = 1 case
        if batch_size == 1:
            sim_t2q = sim_t2q.unsqueeze(0)  # (1, 1, Q)
        
        # Take max over query tokens: (B, B, Q) -> (B, B)
        sim_i2t, _ = sim_t2q.max(-1)
    else:
        # target_feats is already (B, D)
        # Compute similarity matrix: (B, D) x (D, B) -> (B, B)
        sim_i2t = torch.matmul(fusion_feats, target_feats.T)
    
    # Apply temperature
    sim_i2t = sim_i2t / temperature
    
    # Labels: diagonal is positive, targets = [0, 1, 2, ..., B-1]
    targets = torch.arange(batch_size, dtype=torch.long, device=device)
    
    # In-batch contrastive loss
    loss_inbatch = torch.nn.functional.cross_entropy(sim_i2t, targets)
    
    # ============ Part 2: Hard negative contrastive loss ============
    # Compute positive similarity for each sample
    if target_feats.dim() == 3:
        # fusion_feats: (B, D), target_feats: (B, Q, D)
        # Get each sample's positive similarity: (B, 1, D) x (B, D, Q) -> (B, Q) -> max -> (B,)
        sim_pos = torch.bmm(
            fusion_feats.unsqueeze(1),  # (B, 1, D)
            target_feats.permute(0, 2, 1)  # (B, D, Q)
        ).squeeze(1)  # (B, Q)
        sim_pos, _ = sim_pos.max(dim=-1)  # (B,)
    else:
        # target_feats is (B, D), element-wise dot product for each pair
        sim_pos = (fusion_feats * target_feats).sum(dim=-1)  # (B,)
    
    # Compute similarity with hard negatives
    # hard_negative_feats: (B, N, D) or (B, D)
    if hard_negative_feats.dim() == 2:
        hard_negative_feats = hard_negative_feats.unsqueeze(1)  # (B, 1, D)
    
    # (B, 1, D) x (B, D, N) -> (B, 1, N) -> (B, N)
    sim_hard = torch.bmm(
        fusion_feats.unsqueeze(1),  # (B, 1, D)
        hard_negative_feats.permute(0, 2, 1)  # (B, D, N)
    ).squeeze(1)  # (B, N)
    
    # Handle case where sim_hard might be 1D (single negative)
    if sim_hard.dim() == 1:
        sim_hard = sim_hard.unsqueeze(1)
    
    # Logits for hard negative loss: [positive, hard_negatives]
    logits_hard = torch.cat([
        sim_pos.unsqueeze(1),  # (B, 1) - positive at index 0
        sim_hard  # (B, N) - hard negatives
    ], dim=1) / temperature
    
    # Labels: positive is at index 0
    labels_hard = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Hard negative contrastive loss
    loss_hard = torch.nn.functional.cross_entropy(logits_hard, labels_hard)
    
    # ============ Combined loss ============
    total_loss = loss_inbatch + hard_negative_weight * loss_hard
    
    # return total_loss
    return hard_negative_weight * loss_hard

