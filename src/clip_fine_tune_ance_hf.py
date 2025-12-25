# -*- coding: utf-8 -*-
"""
CLIP Fine-tuning with ANCE using Hugging Face Transformers

This script implements hard negative mining during training using FAISS-based ANN search.
Uses Hugging Face Transformers CLIP models instead of OpenAI's clip library.

Composed query = element_wise_sum(image_features, text_features)
Similarity = dot_product(query_features, target_features)
"""

import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import os
import logging
import time

# Hugging Face Transformers
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, \
    element_wise_sum, device
from clip_ance_utils import CLIPHardNegativeMiner, compute_clip_ance_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model name mapping from simple names to Hugging Face model IDs
CLIP_MODEL_MAPPING = {
    "RN50": "openai/clip-vit-base-patch32",  # Default to ViT-B/32 as RN50 is not directly available
    "RN101": "openai/clip-vit-base-patch32",
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-L/14": "openai/clip-vit-large-patch14",
    "ViT-L/14-336": "openai/clip-vit-large-patch14-336",
}

# Embedding dimensions for different models
EMBEDDING_DIMS = {
    "openai/clip-vit-base-patch32": 512,
    "openai/clip-vit-base-patch16": 512,
    "openai/clip-vit-large-patch14": 768,
    "openai/clip-vit-large-patch14-336": 768,
}


def get_clip_model_and_processor(model_name: str, device):
    """
    Load CLIP model and processor from Hugging Face.
    
    Args:
        model_name: Model name (e.g., "ViT-B/32" or "openai/clip-vit-base-patch32")
        device: torch device
    
    Returns:
        model, processor, tokenizer, embedding_dim
    """
    # Map simple names to HF model IDs
    if model_name in CLIP_MODEL_MAPPING:
        hf_model_name = CLIP_MODEL_MAPPING[model_name]
    else:
        hf_model_name = model_name
    
    logger.info(f"Loading CLIP model: {hf_model_name}")
    
    # Load model and processor
    model = CLIPModel.from_pretrained(hf_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(hf_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(hf_model_name)
    
    # Get embedding dimension
    embedding_dim = EMBEDDING_DIMS.get(hf_model_name, 512)
    
    logger.info(f"Model loaded. Embedding dimension: {embedding_dim}")
    
    return model, processor, tokenizer, embedding_dim


def extract_clip_index_features(dataset, clip_model, batch_size=64, num_workers=4):
    """Extract index features from dataset using CLIP (Transformers version)."""
    from utils import collate_fn
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    clip_model.eval()
    all_features = []
    all_names = []
    
    for names, images in tqdm(dataloader, desc="Extracting index features"):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            # Use get_image_features for transformers CLIP
            image_features = clip_model.get_image_features(pixel_values=images)
            image_features = F.normalize(image_features, dim=-1)
            all_features.append(image_features.cpu())
            all_names.extend(names)
    
    return torch.vstack(all_features).to(device), all_names


def encode_text_hf(clip_model, tokenizer, texts, device):
    """
    Encode text using Hugging Face CLIP model.
    
    Args:
        clip_model: CLIPModel
        tokenizer: CLIPTokenizer
        texts: List of text strings
        device: torch device
    
    Returns:
        text_features: normalized text embeddings
    """
    # Tokenize text
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=77,  # CLIP's max length
        return_tensors="pt"
    ).to(device)
    
    # Get text features
    text_features = clip_model.get_text_features(**inputs)
    
    return text_features


def compute_fiq_val_metrics_clip(
    relative_val_dataset: FashionIQDataset,
    clip_model,
    tokenizer,
    index_features: torch.Tensor,
    index_names: List[str]
):
    """Compute FashionIQ validation metrics."""
    print(f"Computing FashionIQ {relative_val_dataset.dress_types} validation metrics")
    
    clip_model.eval()
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset, batch_size=32,
        num_workers=4, pin_memory=True, collate_fn=collate_fn, shuffle=False
    )
    
    # Get mapping from names to features
    name_to_feat = dict(zip(index_names, index_features))
    
    predicted_features = []
    target_names = []
    
    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):
        # Concatenate captions
        flattened_captions = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
            for i in range(0, len(flattened_captions), 2)
        ]
        
        with torch.no_grad():
            # Get reference image features
            if len(input_captions) == 1:
                from operator import itemgetter
                reference_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                from operator import itemgetter
                reference_features = torch.stack(itemgetter(*reference_names)(name_to_feat))
            
            # Encode text using HF tokenizer
            text_features = encode_text_hf(clip_model, tokenizer, input_captions, device)
            
            # Normalize
            reference_features = F.normalize(reference_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Combine: element-wise sum
            batch_predicted = element_wise_sum(reference_features, text_features)
            predicted_features.append(batch_predicted.cpu())
        
        target_names.extend(batch_target_names)
    
    predicted_features = torch.vstack(predicted_features).to(device)
    
    # Compute similarities
    similarities = predicted_features @ index_features.T
    distances = 1 - similarities
    
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    
    # Compute labels
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1)
    )
    
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    
    return recall_at10, recall_at50


def compute_cirr_val_metrics_clip(
    relative_val_dataset: CIRRDataset,
    clip_model,
    tokenizer,
    index_features: torch.Tensor,
    index_names: List[str]
):
    """Compute CIRR validation metrics."""
    print("Computing CIRR validation metrics")
    
    clip_model.eval()
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset, batch_size=32,
        num_workers=4, pin_memory=True, collate_fn=collate_fn, shuffle=False
    )
    
    name_to_feat = dict(zip(index_names, index_features))
    
    predicted_features = []
    reference_names = []
    target_names = []
    group_members = []
    
    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(relative_val_loader):
        batch_group_members = np.array(batch_group_members).T.tolist()
        
        with torch.no_grad():
            if len(captions) == 1:
                from operator import itemgetter
                reference_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                from operator import itemgetter
                reference_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))
            
            # Encode text using HF tokenizer
            text_features = encode_text_hf(clip_model, tokenizer, captions, device)
            
            reference_features = F.normalize(reference_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            batch_predicted = element_wise_sum(reference_features, text_features)
            predicted_features.append(batch_predicted.cpu())
        
        reference_names.extend(batch_reference_names)
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
    
    predicted_features = torch.vstack(predicted_features).to(device)
    
    similarities = predicted_features @ index_features.T
    distances = 1 - similarities
    
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    
    # Remove reference images
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1)
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], -1)
    
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1)
    )
    
    # Subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)
    
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100
    
    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def clip_finetune_fiq_ance(
    train_dress_types: List[str],
    val_dress_types: List[str],
    num_epochs: int,
    clip_model_name: str,
    learning_rate: float,
    batch_size: int,
    validation_frequency: int,
    transform: str,
    save_training: bool,
    save_best: bool,
    # ANCE specific parameters
    ance_num_negatives: int = 16,
    ance_topk_candidates: int = 100,
    ance_refresh_interval: int = 1,
    ance_weight: float = 1.0,
    ance_warmup_epochs: int = 0,
    experiment_name: str = None,
    **kwargs
):
    """Fine-tune CLIP on FashionIQ with ANCE hard negative mining (Transformers version)."""
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    # Clean model name for path
    model_name_clean = clip_model_name.replace("/", "-")
    
    if experiment_name:
        training_path = Path(base_path / f"models/clip_ance_fiq_{model_name_clean}_{experiment_name}_{training_start}")
    else:
        training_path = Path(base_path / f"models/clip_ance_fiq_{model_name_clean}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    
    # Load CLIP model from Hugging Face
    clip_model, processor, tokenizer, embedding_dim = get_clip_model_and_processor(clip_model_name, device)
    
    # Save hyperparameters
    training_hyper_params = {
        "num_epochs": num_epochs,
        "clip_model_name": clip_model_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "ance_num_negatives": ance_num_negatives,
        "ance_topk_candidates": ance_topk_candidates,
        "ance_refresh_interval": ance_refresh_interval,
        "ance_weight": ance_weight,
        "ance_warmup_epochs": ance_warmup_epochs,
    }
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    
    input_dim = 224
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
    elif transform == "targetpad":
        target_ratio = kwargs.get('target_ratio', 1.25)
        preprocess = targetpad_transform(target_ratio, input_dim)
    else:
        raise ValueError("Preprocess transform should be in ['squarepad', 'targetpad']")
    
    # Define datasets
    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []
    
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_datasets.append(FashionIQDataset('val', [dress_type], 'relative', preprocess))
        classic_val_datasets.append(FashionIQDataset('val', [dress_type], 'classic', preprocess))
    
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    classic_train_dataset = FashionIQDataset('train', train_dress_types, 'classic', preprocess, preload_images=True)  # 预加载所有图像
    
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset, batch_size=batch_size,
        num_workers=kwargs.get('num_workers', 4), 
        pin_memory=True,  # 启用pin_memory加速GPU传输
        collate_fn=collate_fn,
        drop_last=True, 
        shuffle=True,
        prefetch_factor=2,  # 预取2个batch
        persistent_workers=True  # 保持worker进程，减少启动开销
    )
    
    # Initialize ANCE hard negative miner
    hard_negative_miner = CLIPHardNegativeMiner(
        embedding_dim=embedding_dim,
        num_negatives=ance_num_negatives,
        topk_candidates=ance_topk_candidates,
        refresh_interval=ance_refresh_interval,
        use_gpu=False,  # CPU模式避免与PyTorch GPU冲突（FAISS在CPU也很快）
        cache_dir=str(training_path / "ance_cache")
    )
    
    # Build initial index
    logger.info("Building initial ANCE index...")
    hard_negative_miner.build_index(
        clip_model=clip_model,
        dataset=classic_train_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=kwargs.get('num_workers', 4)
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        clip_model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-7,
        weight_decay=0.05
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs,
        div_factor=100., steps_per_epoch=len(relative_train_loader), epochs=num_epochs
    )
    scaler = torch.cuda.amp.GradScaler()
    
    if save_best:
        best_avg_recall = 0
    
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()
    
    logger.info('Training loop started with ANCE (Transformers CLIP)')
    for epoch in range(num_epochs):
        # Refresh ANCE index periodically
        if epoch > 0:
            hard_negative_miner.refresh_index(
                clip_model=clip_model,
                dataset=classic_train_dataset,
                device=device,
                current_epoch=epoch,
                batch_size=batch_size,
                num_workers=kwargs.get('num_workers', 4)
            )
        
        use_ance = epoch >= ance_warmup_epochs
        train_running_results = {'accumulated_train_loss': 0, 'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        
        for idx, (reference_images, target_images, captions, target_names) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            optimizer.zero_grad(set_to_none=True)  # Faster than set_to_none=False
            
            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            
            # Randomize captions
            flattened_captions = np.array(captions).T.flatten().tolist()
            from utils import generate_randomized_fiq_caption
            captions = generate_randomized_fiq_caption(flattened_captions)
            
            clip_model.train()
            
            # Forward pass - encode images and text ONCE (避免重复编码)
            with torch.cuda.amp.autocast():
                # Encode images
                ref_features = clip_model.get_image_features(pixel_values=reference_images)
                target_features = clip_model.get_image_features(pixel_values=target_images)
                
                # Encode text
                text_features = encode_text_hf(clip_model, tokenizer, captions, device)
                
                # Normalize
                ref_features = F.normalize(ref_features, dim=-1)
                target_features = F.normalize(target_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Compose query: element-wise sum
                query_features = element_wise_sum(ref_features, text_features)
                
                # Mine hard negatives using the already computed features (不再重复编码!)
                hard_negative_names = None
                ref_hard_negative_names = None
                if use_ance:
                    with torch.no_grad():
                        # 1. 挖掘query的硬负样本（视觉+文本组合后相似的目标图像）
                        _, hard_negative_names = hard_negative_miner.mine_hard_negatives(
                            query_features=query_features.detach(),  # detach避免影响梯度
                            positive_names=list(target_names),
                            return_names=True
                        )
                        
                        # 2. 挖掘reference image的硬负样本（视觉上相似但不匹配text的图像）
                        # 使用reference_features而非query_features来挖掘
                        _, ref_hard_negative_names = hard_negative_miner.mine_hard_negatives(
                            query_features=ref_features.detach(),  # 使用reference image features
                            positive_names=list(target_names),  # 排除正样本
                            return_names=True
                        )
                
                # Load and encode hard negatives (在autocast内，但在单独的section)
                if use_ance and hard_negative_names is not None:
                    # Flatten all negative sample names
                    all_neg_names = [name for batch_names in hard_negative_names for name in batch_names]
                    num_negatives = len(hard_negative_names[0])
                    total_negs = len(all_neg_names)
                    
                    # 动态chunk size: 增大到512以充分利用GPU（可通过参数调整）
                    chunk_size = min(512, total_negs)
                    
                    # 直接从缓存获取图像（已预加载到内存，无需并行加载）
                    # 使用列表推导式快速获取，比get_images_by_names更快
                    all_neg_images = torch.stack([
                        classic_train_dataset.get_image_by_name(name) for name in all_neg_names
                    ]).to(device, non_blocking=True)
                    
                    # Batch encode in chunks
                    all_neg_features_list = []
                    for i in range(0, total_negs, chunk_size):
                        chunk_images = all_neg_images[i:i+chunk_size]
                        chunk_features = clip_model.get_image_features(pixel_values=chunk_images)
                        chunk_features = F.normalize(chunk_features, dim=-1)
                        all_neg_features_list.append(chunk_features)
                    
                    # Concatenate and reshape
                    all_neg_features = torch.cat(all_neg_features_list, dim=0)
                    hard_negative_features = all_neg_features.view(images_in_batch, num_negatives, -1)
                    
                    # Clean up
                    del all_neg_images, all_neg_features_list, all_neg_features
                    
                    # 🆕 编码reference硬负样本
                    ref_hard_negative_features = None
                    if ref_hard_negative_names is not None:
                        # Flatten reference negative names
                        all_ref_neg_names = [name for batch_names in ref_hard_negative_names for name in batch_names]
                        num_ref_negatives = len(ref_hard_negative_names[0])
                        total_ref_negs = len(all_ref_neg_names)
                        
                        # 从缓存加载reference硬负样本图像
                        all_ref_neg_images = torch.stack([
                            classic_train_dataset.get_image_by_name(name) for name in all_ref_neg_names
                        ]).to(device, non_blocking=True)
                        
                        # Batch encode in chunks
                        all_ref_neg_features_list = []
                        for i in range(0, total_ref_negs, chunk_size):
                            chunk_images = all_ref_neg_images[i:i+chunk_size]
                            chunk_features = clip_model.get_image_features(pixel_values=chunk_images)
                            chunk_features = F.normalize(chunk_features, dim=-1)
                            all_ref_neg_features_list.append(chunk_features)
                        
                        # Concatenate and reshape
                        all_ref_neg_features = torch.cat(all_ref_neg_features_list, dim=0)
                        ref_hard_negative_features = all_ref_neg_features.view(images_in_batch, num_ref_negatives, -1)
                        
                        # Clean up
                        del all_ref_neg_images, all_ref_neg_features_list, all_ref_neg_features
                    
                    loss = compute_clip_ance_loss(
                        query_features=query_features,
                        target_features=target_features,
                        hard_negative_features=hard_negative_features,
                        temperature=0.07,
                        hard_negative_weight=ance_weight,
                        ref_hard_negative_features=ref_hard_negative_features,  # 🆕 传入reference硬负样本
                        ref_hard_negative_weight=kwargs.get('ref_ance_weight', 0.5)  # 🆕 可配置权重
                    )
                else:
                    # Standard in-batch contrastive loss
                    sim_matrix = torch.matmul(query_features, target_features.T) / 0.07
                    labels = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = F.cross_entropy(sim_matrix, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            update_train_running_results(train_running_results, loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)
        
        # Logging
        loss_log_dict = {
            'epoch': epoch,
            'loss': train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']
        }
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)
        
        # Validation
        if epoch % validation_frequency == 0:
            clip_model.eval()
            recalls_at10, recalls_at50 = [], []
            
            for relative_val_dataset, classic_val_dataset, idx in zip(
                relative_val_datasets, classic_val_datasets, idx_to_dress_mapping
            ):
                torch.cuda.empty_cache()
                index_features, index_names = extract_clip_index_features(
                    classic_val_dataset, clip_model
                )
                recall_at10, recall_at50 = compute_fiq_val_metrics_clip(
                    relative_val_dataset, clip_model, tokenizer, index_features, index_names
                )
                recalls_at10.append(recall_at10)
                recalls_at50.append(recall_at50)
                del index_features, index_names
                torch.cuda.empty_cache()
            
            results_dict = {
                f'{idx_to_dress_mapping[i]}_recall_at10': recalls_at10[i]
                for i in range(len(recalls_at10))
            }
            results_dict.update({
                f'{idx_to_dress_mapping[i]}_recall_at50': recalls_at50[i]
                for i in range(len(recalls_at50))
            })
            results_dict.update({
                'average_recall_at10': mean(recalls_at10),
                'average_recall_at50': mean(recalls_at50),
                'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
            })
            print(json.dumps(results_dict, indent=4))
            
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)
            
            if save_training and save_best and results_dict['average_recall'] > best_avg_recall:
                best_avg_recall = results_dict['average_recall']
                # Save using Hugging Face save_pretrained
                clip_model.save_pretrained(str(training_path / 'best_model'))
                tokenizer.save_pretrained(str(training_path / 'best_model'))
                logger.info(f"Saved best model at epoch {epoch} with recall {best_avg_recall:.2f}")


def clip_finetune_cirr_ance(
    num_epochs: int,
    clip_model_name: str,
    learning_rate: float,
    batch_size: int,
    validation_frequency: int,
    transform: str,
    save_training: bool,
    save_best: bool,
    # ANCE specific parameters
    ance_num_negatives: int = 16,
    ance_topk_candidates: int = 100,
    ance_refresh_interval: int = 1,
    ance_weight: float = 1.0,
    ance_warmup_epochs: int = 0,
    experiment_name: str = None,
    **kwargs
):
    """Fine-tune CLIP on CIRR with ANCE hard negative mining (Transformers version)."""
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    model_name_clean = clip_model_name.replace("/", "-")
    
    if experiment_name:
        training_path = Path(base_path / f"models/clip_ance_cirr_{model_name_clean}_{experiment_name}_{training_start}")
    else:
        training_path = Path(base_path / f"models/clip_ance_cirr_{model_name_clean}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    
    # Load CLIP model from Hugging Face
    clip_model, processor, tokenizer, embedding_dim = get_clip_model_and_processor(clip_model_name, device)
    
    # Save hyperparameters
    training_hyper_params = {
        "num_epochs": num_epochs,
        "clip_model_name": clip_model_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "ance_num_negatives": ance_num_negatives,
        "ance_topk_candidates": ance_topk_candidates,
        "ance_refresh_interval": ance_refresh_interval,
        "ance_weight": ance_weight,
        "ance_warmup_epochs": ance_warmup_epochs,
    }
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    
    input_dim = 224
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
    elif transform == "targetpad":
        target_ratio = kwargs.get('target_ratio', 1.25)
        preprocess = targetpad_transform(target_ratio, input_dim)
    else:
        raise ValueError("Preprocess transform should be in ['squarepad', 'targetpad']")
    
    # Define datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    classic_train_dataset = CIRRDataset('train', 'classic', preprocess, preload_images=True)  # 预加载所有图像
    
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset, batch_size=batch_size,
        num_workers=kwargs.get('num_workers', 4), 
        pin_memory=True,  # 启用pin_memory加速GPU传输
        collate_fn=collate_fn,
        drop_last=True, 
        shuffle=True,
        prefetch_factor=2,  # 预取2个batch
        persistent_workers=True  # 保持worker进程，减少启动开销
    )
    
    # Initialize ANCE hard negative miner
    hard_negative_miner = CLIPHardNegativeMiner(
        embedding_dim=embedding_dim,
        num_negatives=ance_num_negatives,
        topk_candidates=ance_topk_candidates,
        refresh_interval=ance_refresh_interval,
        use_gpu=False,  # CPU模式避免与PyTorch GPU冲突（FAISS在CPU也很快）
        cache_dir=str(training_path / "ance_cache")
    )
    
    # Build initial index
    logger.info("Building initial ANCE index for CIRR...")
    hard_negative_miner.build_index(
        clip_model=clip_model,
        dataset=classic_train_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=kwargs.get('num_workers', 4)
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        clip_model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-7,
        weight_decay=0.05
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=learning_rate, pct_start=1/50,
        steps_per_epoch=len(relative_train_loader), epochs=80
    )
    scaler = torch.cuda.amp.GradScaler()
    
    if save_best:
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = 0
    
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()
    
    logger.info('Training loop started with ANCE for CIRR (Transformers CLIP)')
    for epoch in range(num_epochs):
        # Refresh ANCE index periodically
        if epoch > 0:
            hard_negative_miner.refresh_index(
                clip_model=clip_model,
                dataset=classic_train_dataset,
                device=device,
                current_epoch=epoch,
                batch_size=batch_size,
                num_workers=kwargs.get('num_workers', 4)
            )
        
        use_ance = epoch >= ance_warmup_epochs
        train_running_results = {'accumulated_train_loss': 0, 'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        
        for idx, (reference_images, target_images, captions, target_names) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            optimizer.zero_grad(set_to_none=True)  # Faster than set_to_none=False
            
            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            
            clip_model.train()
            
            # Forward pass - encode images and text ONCE (避免重复编码)
            with torch.cuda.amp.autocast():
                ref_features = clip_model.get_image_features(pixel_values=reference_images)
                target_features = clip_model.get_image_features(pixel_values=target_images)
                
                text_features = encode_text_hf(clip_model, tokenizer, captions, device)
                
                ref_features = F.normalize(ref_features, dim=-1)
                target_features = F.normalize(target_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                query_features = element_wise_sum(ref_features, text_features)
                
                # Mine hard negatives using the already computed features (不再重复编码!)
                hard_negative_names = None
                ref_hard_negative_names = None
                if use_ance:
                    with torch.no_grad():
                        # 1. 挖掘query的硬负样本
                        _, hard_negative_names = hard_negative_miner.mine_hard_negatives(
                            query_features=query_features.detach(),  # detach避免影响梯度
                            positive_names=list(target_names),
                            return_names=True
                        )
                        
                        # 2. 挖掘reference image的硬负样本
                        _, ref_hard_negative_names = hard_negative_miner.mine_hard_negatives(
                            query_features=ref_features.detach(),  # 使用reference image features
                            positive_names=list(target_names),
                            return_names=True
                        )
                
                # Load and encode hard negatives
                if use_ance and hard_negative_names is not None:
                    # Flatten all negative sample names
                    all_neg_names = [name for batch_names in hard_negative_names for name in batch_names]
                    num_negatives = len(hard_negative_names[0])
                    total_negs = len(all_neg_names)
                    
                    # 动态chunk size: 增大到512以充分利用GPU
                    chunk_size = min(512, total_negs)
                    
                    # 直接从缓存获取图像（已预加载到内存，无需并行加载）
                    all_neg_images = torch.stack([
                        classic_train_dataset.get_image_by_name(name) for name in all_neg_names
                    ]).to(device, non_blocking=True)
                    
                    # Batch encode in chunks
                    all_neg_features_list = []
                    for i in range(0, total_negs, chunk_size):
                        chunk_images = all_neg_images[i:i+chunk_size]
                        chunk_features = clip_model.get_image_features(pixel_values=chunk_images)
                        chunk_features = F.normalize(chunk_features, dim=-1)
                        all_neg_features_list.append(chunk_features)
                    
                    # Concatenate and reshape
                    all_neg_features = torch.cat(all_neg_features_list, dim=0)
                    hard_negative_features = all_neg_features.view(images_in_batch, num_negatives, -1)
                    
                    # Clean up
                    del all_neg_images, all_neg_features_list, all_neg_features
                    
                    # 🆕 编码reference硬负样本
                    ref_hard_negative_features = None
                    if ref_hard_negative_names is not None:
                        # Flatten reference negative names
                        all_ref_neg_names = [name for batch_names in ref_hard_negative_names for name in batch_names]
                        num_ref_negatives = len(ref_hard_negative_names[0])
                        total_ref_negs = len(all_ref_neg_names)
                        
                        # 从缓存加载reference硬负样本图像
                        all_ref_neg_images = torch.stack([
                            classic_train_dataset.get_image_by_name(name) for name in all_ref_neg_names
                        ]).to(device, non_blocking=True)
                        
                        # Batch encode in chunks
                        all_ref_neg_features_list = []
                        for i in range(0, total_ref_negs, chunk_size):
                            chunk_images = all_ref_neg_images[i:i+chunk_size]
                            chunk_features = clip_model.get_image_features(pixel_values=chunk_images)
                            chunk_features = F.normalize(chunk_features, dim=-1)
                            all_ref_neg_features_list.append(chunk_features)
                        
                        # Concatenate and reshape
                        all_ref_neg_features = torch.cat(all_ref_neg_features_list, dim=0)
                        ref_hard_negative_features = all_ref_neg_features.view(images_in_batch, num_ref_negatives, -1)
                        
                        # Clean up
                        del all_ref_neg_images, all_ref_neg_features_list, all_ref_neg_features
                    
                    loss = compute_clip_ance_loss(
                        query_features=query_features,
                        target_features=target_features,
                        hard_negative_features=hard_negative_features,
                        temperature=0.07,
                        hard_negative_weight=ance_weight,
                        ref_hard_negative_features=ref_hard_negative_features,  # 🆕 传入reference硬负样本
                        ref_hard_negative_weight=kwargs.get('ref_ance_weight', 0.5)  # 🆕 可配置权重
                    )
                else:
                    sim_matrix = torch.matmul(query_features, target_features.T) / 0.07
                    labels = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = F.cross_entropy(sim_matrix, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            update_train_running_results(train_running_results, loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)
        
        # Logging
        loss_log_dict = {
            'epoch': epoch,
            'loss': train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']
        }
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)
        
        # Validation
        if epoch % validation_frequency == 0:
            clip_model.eval()
            torch.cuda.empty_cache()
            
            val_index_features, val_index_names = extract_clip_index_features(
                classic_val_dataset, clip_model
            )
            results = compute_cirr_val_metrics_clip(
                relative_val_dataset, clip_model, tokenizer, val_index_features, val_index_names
            )
            group_recall_at1, group_recall_at2, group_recall_at3, \
                recall_at1, recall_at5, recall_at10, recall_at50 = results
            
            results_dict = {
                'group_recall_at1': group_recall_at1,
                'group_recall_at2': group_recall_at2,
                'group_recall_at3': group_recall_at3,
                'recall_at1': recall_at1,
                'recall_at5': recall_at5,
                'recall_at10': recall_at10,
                'recall_at50': recall_at50,
                'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                'arithmetic_mean': mean(results),
                'harmonic_mean': harmonic_mean(results),
                'geometric_mean': geometric_mean(results)
            }
            print(json.dumps(results_dict, indent=4))
            
            del val_index_features, val_index_names
            torch.cuda.empty_cache()
            
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)
            
            if save_training and save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                best_arithmetic = results_dict['arithmetic_mean']
                clip_model.save_pretrained(str(training_path / 'best_model'))
                tokenizer.save_pretrained(str(training_path / 'best_model'))
                logger.info(f"Saved best model at epoch {epoch} with arithmetic mean {best_arithmetic:.2f}")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 注意：启用benchmark可提升性能，但会牺牲一定的可重复性
    torch.backends.cudnn.deterministic = False  # 允许非确定性以提速
    torch.backends.cudnn.benchmark = True  # 启用cudnn自动调优，显著加速
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    print(f"⚠️  Note: cudnn.benchmark=True for speed, may have slight non-determinism")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--clip-model-name", default="ViT-B/32", type=str,
                        help="CLIP model name: 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', or HF model path")
    parser.add_argument("--learning-rate", default=2e-6, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--validation-frequency", default=1, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float)
    parser.add_argument("--transform", default="targetpad", type=str)
    parser.add_argument("--save-training", dest="save_training", action='store_true')
    parser.add_argument("--save-best", dest="save_best", action='store_true')
    parser.add_argument("--experiment-name", type=str, default=None)
    
    # ANCE specific arguments
    parser.add_argument("--ance-num-negatives", default=16, type=int,
                        help="Number of hard negatives per query")
    parser.add_argument("--ance-topk-candidates", default=100, type=int,
                        help="Top-k candidates from which to sample hard negatives")
    parser.add_argument("--ance-refresh-interval", default=1, type=int,
                        help="How often to refresh the ANN index (in epochs)")
    parser.add_argument("--ance-weight", default=1.0, type=float,
                        help="Weight for hard negative samples in loss")
    parser.add_argument("--ance-warmup-epochs", default=0, type=int,
                        help="Number of epochs to train without ANCE before enabling it")
    parser.add_argument("--ref-ance-weight", default=0.5, type=float,
                        help="Weight for reference image hard negative loss")
    
    args = parser.parse_args()
    
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ'")
    
    set_seed(42)
    
    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "clip_model_name": args.clip_model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "experiment_name": args.experiment_name,
        # ANCE params
        "ance_num_negatives": args.ance_num_negatives,
        "ance_topk_candidates": args.ance_topk_candidates,
        "ance_refresh_interval": args.ance_refresh_interval,
        "ance_weight": args.ance_weight,
        "ance_warmup_epochs": args.ance_warmup_epochs,
        "ref_ance_weight": args.ref_ance_weight,  # 🆕 添加reference硬负样本权重参数
    }
    
    if args.dataset.lower() == 'cirr':
        clip_finetune_cirr_ance(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update({
            'train_dress_types': ['dress', 'toptee', 'shirt'],
            'val_dress_types': ['dress', 'toptee', 'shirt']
        })
        clip_finetune_fiq_ance(**training_hyper_params)

