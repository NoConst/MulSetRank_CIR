# -*- coding: utf-8 -*-
"""
BLIP2 Fine-tuning with ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation)

This script implements hard negative mining during training using FAISS-based ANN search,
inspired by the ANCE paper: "Approximate Nearest Neighbor Negative Contrastive Learning 
for Dense Text Retrieval" (Xiong et al., 2020)
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
from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR
import os
import logging

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results, update_train_running_results_dict, \
    set_train_bar_description_dict, set_train_bar_description, extract_index_blip_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device
from validate_blip import compute_cirr_val_metrics, compute_fiq_val_metrics
from ance_utils import HardNegativeMiner, compute_ance_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def forward_with_hard_negatives(
    blip_model,
    reference_images: torch.Tensor,
    target_images: torch.Tensor,
    captions: List[str],
    hard_negative_features: torch.Tensor = None,
    hard_negative_weight: float = 1.0,
    use_ance: bool = True
):
    """
    Forward pass with optional hard negative mining support.
    
    Args:
        blip_model: The BLIP2 model
        reference_images: Reference images tensor
        target_images: Target images tensor
        captions: List of text captions
        hard_negative_features: Pre-computed hard negative features (B, N, D)
        hard_negative_weight: Weight for hard negative samples
        use_ance: Whether to use ANCE loss
        
    Returns:
        Dictionary of losses
    """
    image = reference_images
    target = target_images
    text = captions
    
    ###============== reference text fusion ===================###
    # reference image feature  
    image_embeds = blip_model.ln_vision(blip_model.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    
    # query tokens
    query_tokens = blip_model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(blip_model.device)
    
    # text tokens
    text_tokens = blip_model.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=blip_model.max_txt_len,
        return_tensors="pt",
    ).to(image.device)
    
    # fusion reference image and text tokens
    attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
    fusion_output = blip_model.Qformer.bert(
        text_tokens.input_ids,
        query_embeds=query_tokens,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    text_output = blip_model.Qformer.bert(
        text_tokens.input_ids,
        query_embeds=fusion_output.last_hidden_state[:, :query_tokens.size(1), :],
        attention_mask=attention_mask,
        return_dict=True,
    )

    fusion_feats = F.normalize(
        blip_model.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
    )

    ###============== Target feature extraction ===================###
    target_embeds = blip_model.ln_vision(blip_model.visual_encoder(target))
    target_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(image.device)
    target_output = blip_model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=target_embeds,
        encoder_attention_mask=target_atts,
        use_cache=True,
        return_dict=True,
    )
    target_feats = F.normalize(
        blip_model.vision_proj(target_output.last_hidden_state), dim=-1
    )

    ###============== Contrastive Loss Computation ===================###
    bs = image.size(0)
    targets = torch.linspace(0, bs - 1, bs, dtype=int).to(image.device)
    
    if use_ance and hard_negative_features is not None:
        # Use ANCE loss with hard negatives
        hard_neg_tensor = torch.from_numpy(hard_negative_features).float().to(image.device)
        loss_itc = compute_ance_loss(
            fusion_feats=fusion_feats,
            target_feats=target_feats,
            hard_negative_feats=hard_neg_tensor,
            temperature=blip_model.temp.item(),
            hard_negative_weight=hard_negative_weight
        )
    else:
        # Standard in-batch contrastive loss
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()
        sim_i2t, _ = sim_t2q.max(-1)
        sim_i2t = sim_i2t / blip_model.temp
        loss_itc = F.cross_entropy(sim_i2t, targets)

    ###============== Relative Contrastive Loss ===================###
    prompt_tokens = blip_model.prompt_tokens.expand(image_embeds.shape[0], -1, -1)

    text_only_output = blip_model.Qformer.bert(
        text_tokens.input_ids,
        query_embeds=prompt_tokens,
        attention_mask=attention_mask,
        return_dict=True,
        no_img=True
    )

    text_only_feat = F.normalize(
        blip_model.text_proj(text_only_output.last_hidden_state[:, 0, :]), dim=-1
    )

    sim_r2t = torch.matmul(
        text_only_feat.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
    ).squeeze()

    sim_r2t, _ = sim_r2t.max(-1)
    sim_r2t = sim_r2t / blip_model.temp
    loss_rtc = F.cross_entropy(sim_r2t, targets)

    ###============== Alignment Loss ===================###
    loss_align = F.mse_loss(
        fusion_output.last_hidden_state[:, :query_tokens.size(1), :].mean(1),
        prompt_tokens.clone().detach().mean(1)
    )

    return {
        'loss_itc': loss_itc,
        'loss_rtc': loss_rtc,
        'loss_align': loss_align
    }, fusion_feats


def clip_finetune_fiq_ance(
    train_dress_types: List[str],
    val_dress_types: List[str],
    num_epochs: int,
    blip_model_name: str,
    backbone: str,
    learning_rate: float,
    batch_size: int,
    validation_frequency: int,
    transform: str,
    save_training: bool,
    save_best: bool,
    save_memory: bool,
    # ANCE specific parameters
    ance_num_negatives: int = 16,
    ance_topk_candidates: int = 100,
    ance_refresh_interval: int = 1,
    ance_weight: float = 1.0,
    ance_warmup_epochs: int = 0,
    experiment_name: str = None,
    **kwargs
):
    """
    Fine-tune BLIP on the FashionIQ dataset with ANCE hard negative mining.
    """
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if experiment_name:
        training_path: Path = Path(
            base_path / f"models/blip_ance_fiq_{blip_model_name}_{experiment_name}_{training_start}")
    else:
        training_path: Path = Path(
            base_path / f"models/blip_ance_fiq_{blip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    
    # Save hyperparameters
    training_hyper_params = {
        "num_epochs": num_epochs,
        "blip_model_name": blip_model_name,
        "backbone": backbone,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "ance_num_negatives": ance_num_negatives,
        "ance_topk_candidates": ance_topk_candidates,
        "ance_refresh_interval": ance_refresh_interval,
        "ance_weight": ance_weight,
        "ance_warmup_epochs": ance_warmup_epochs,
    }
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    
    # Load model
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type=backbone, is_eval=False, device=device
    )
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
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
    classic_train_dataset = FashionIQDataset('train', train_dress_types, 'classic', preprocess)
    
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset, batch_size=batch_size,
        num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
        drop_last=True, shuffle=True
    )

    # Initialize ANCE hard negative miner
    # Use CPU for FAISS to avoid GPU memory conflicts with PyTorch training
    hard_negative_miner = HardNegativeMiner(
        embedding_dim=256,  # BLIP2 projection dim
        num_negatives=ance_num_negatives,
        topk_candidates=ance_topk_candidates,
        refresh_interval=ance_refresh_interval,
        use_gpu=False,  # CPU is more stable and fast enough for this index size
        cache_dir=str(training_path / "ance_cache")
    )
    
    # Build initial index
    logger.info("Building initial ANCE index...")
    hard_negative_miner.build_index(
        model=blip_model,
        dataset=classic_train_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=kwargs['num_workers']
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()),
          'lr': learning_rate, 'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}]
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

    logger.info('Training loop started with ANCE')
    for epoch in range(num_epochs):
        # Refresh ANCE index periodically
        if epoch > 0:
            hard_negative_miner.refresh_index(
                model=blip_model,
                dataset=classic_train_dataset,
                device=device,
                current_epoch=epoch,
                batch_size=batch_size,
                num_workers=kwargs['num_workers']
            )
        
        use_ance = epoch >= ance_warmup_epochs
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        
        for idx, (reference_images, target_images, captions, target_names) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            # Randomize captions
            flattened_captions = np.array(captions).T.flatten().tolist()
            captions = generate_randomized_fiq_caption(flattened_captions)
            captions = [txt_processors["eval"](caption) for caption in captions]
            
            blip_model.train()
            
            # Get hard negatives if ANCE is enabled
            hard_negative_features = None
            if use_ance:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    # Compute query features for hard negative mining
                    blip_model.eval()
                    image_embeds = blip_model.ln_vision(blip_model.visual_encoder(reference_images))
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                    query_tokens = blip_model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    
                    text_tokens = blip_model.tokenizer(
                        captions, padding="max_length", truncation=True,
                        max_length=blip_model.max_txt_len, return_tensors="pt"
                    ).to(device)
                    
                    attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
                    fusion_output = blip_model.Qformer.bert(
                        text_tokens.input_ids,
                        query_embeds=query_tokens,
                        attention_mask=attention_mask,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    text_output = blip_model.Qformer.bert(
                        text_tokens.input_ids,
                        query_embeds=fusion_output.last_hidden_state[:, :query_tokens.size(1), :],
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                    query_features = F.normalize(
                        blip_model.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
                    )
                    
                    # Use real target names to correctly exclude positives from hard negatives
                    _, hard_negative_features = hard_negative_miner.mine_hard_negatives(
                        query_features=query_features,
                        positive_names=list(target_names)
                    )
                blip_model.train()

            # Forward pass with hard negatives
            with torch.cuda.amp.autocast():
                loss_dict, _ = forward_with_hard_negatives(
                    blip_model=blip_model,
                    reference_images=reference_images,
                    target_images=target_images,
                    captions=captions,
                    hard_negative_features=hard_negative_features,
                    hard_negative_weight=ance_weight,
                    use_ance=use_ance
                )
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        # Logging
        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
                    train_running_results[key] / train_running_results['images_in_epoch']
                )
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        # Validation
        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10, recalls_at50 = [], []

            for relative_val_dataset, classic_val_dataset, idx in zip(
                relative_val_datasets, classic_val_datasets, idx_to_dress_mapping
            ):
                torch.cuda.empty_cache()
                index_features, index_names = extract_index_blip_features(
                    classic_val_dataset, blip_model, save_memory=True
                )
                recall_at10, recall_at50 = compute_fiq_val_metrics(
                    relative_val_dataset, blip_model, index_features, index_names,
                    txt_processors, save_memory=True
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
                save_model('tuned_blip_ance_best', epoch, blip_model, training_path)


def clip_finetune_cirr_ance(
    num_epochs: int,
    blip_model_name: str,
    backbone: str,
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
    """
    Fine-tune BLIP on the CIRR dataset with ANCE hard negative mining.
    """
    rtc_weights = kwargs.get('loss_rtc', 0.4)
    align_weights = kwargs.get('loss_align', 0.4)
    
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if experiment_name:
        training_path: Path = Path(
            base_path / f"models/blip_ance_cirr_{blip_model_name}_{experiment_name}_{training_start}")
    else:
        training_path: Path = Path(
            base_path / f"models/blip_ance_cirr_{blip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save hyperparameters
    training_hyper_params = {
        "num_epochs": num_epochs,
        "blip_model_name": blip_model_name,
        "backbone": backbone,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "ance_num_negatives": ance_num_negatives,
        "ance_topk_candidates": ance_topk_candidates,
        "ance_refresh_interval": ance_refresh_interval,
        "ance_weight": ance_weight,
        "ance_warmup_epochs": ance_warmup_epochs,
        "loss_rtc": rtc_weights,
        "loss_align": align_weights,
    }
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # Load model
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type=backbone, is_eval=False, device=device
    )
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
    else:
        raise ValueError("Preprocess transform should be in ['squarepad', 'targetpad']")

    # Define datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    classic_train_dataset = CIRRDataset('train', 'classic', preprocess)
    
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset, batch_size=batch_size,
        num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
        drop_last=True, shuffle=True
    )

    # Initialize ANCE hard negative miner
    # Initialize ANCE hard negative miner
    # Use CPU for FAISS to avoid GPU memory conflicts with PyTorch training
    hard_negative_miner = HardNegativeMiner(
        embedding_dim=256,
        num_negatives=ance_num_negatives,
        topk_candidates=ance_topk_candidates,
        refresh_interval=ance_refresh_interval,
        use_gpu=False,  # CPU is more stable and fast enough for this index size
        cache_dir=str(training_path / "ance_cache")
    )
    
    # Build initial index
    logger.info("Building initial ANCE index for CIRR...")
    hard_negative_miner.build_index(
        model=blip_model,
        dataset=classic_train_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=kwargs['num_workers']
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()),
          'lr': learning_rate, 'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}]
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

    logger.info('Training loop started with ANCE for CIRR')
    for epoch in range(num_epochs):
        # Refresh ANCE index periodically
        if epoch > 0:
            hard_negative_miner.refresh_index(
                model=blip_model,
                dataset=classic_train_dataset,
                device=device,
                current_epoch=epoch,
                batch_size=batch_size,
                num_workers=kwargs['num_workers']
            )
        
        use_ance = epoch >= ance_warmup_epochs
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        
        for idx, (reference_images, target_images, captions, target_names) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            captions = [txt_processors["eval"](caption) for caption in captions]
            
            blip_model.train()
            
            # Get hard negatives if ANCE is enabled
            hard_negative_features = None
            if use_ance:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    blip_model.eval()
                    image_embeds = blip_model.ln_vision(blip_model.visual_encoder(reference_images))
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                    query_tokens = blip_model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
                    
                    text_tokens = blip_model.tokenizer(
                        captions, padding="max_length", truncation=True,
                        max_length=blip_model.max_txt_len, return_tensors="pt"
                    ).to(device)
                    
                    attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
                    fusion_output = blip_model.Qformer.bert(
                        text_tokens.input_ids,
                        query_embeds=query_tokens,
                        attention_mask=attention_mask,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    text_output = blip_model.Qformer.bert(
                        text_tokens.input_ids,
                        query_embeds=fusion_output.last_hidden_state[:, :query_tokens.size(1), :],
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                    query_features = F.normalize(
                        blip_model.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
                    )
                    
                    # Use real target names to correctly exclude positives from hard negatives
                    _, hard_negative_features = hard_negative_miner.mine_hard_negatives(
                        query_features=query_features,
                        positive_names=list(target_names)
                    )
                blip_model.train()

            # Forward pass with hard negatives
            with torch.cuda.amp.autocast():
                loss_dict, _ = forward_with_hard_negatives(
                    blip_model=blip_model,
                    reference_images=reference_images,
                    target_images=target_images,
                    captions=captions,
                    hard_negative_features=hard_negative_features,
                    hard_negative_weight=ance_weight,
                    use_ance=use_ance
                )
                
                # Apply loss weights
                loss = 0.
                for key in loss_dict.keys():
                    if key != 'loss_itc':
                        loss += kwargs.get(key, 1.0) * loss_dict[key]
                    else:
                        loss += loss_dict[key]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        # Logging
        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
                    train_running_results[key] / train_running_results['images_in_epoch']
                )
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        # Validation
        if epoch % validation_frequency == 0:
            blip_model.eval()
            torch.cuda.empty_cache()
            
            val_index_features, val_index_names = extract_index_blip_features(
                classic_val_dataset, blip_model, save_memory=True
            )
            results = compute_cirr_val_metrics(
                relative_val_dataset, blip_model, val_index_features,
                val_index_names, txt_processors, save_memory=True
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
                save_model('tuned_blip_ance_arithmetic', epoch, blip_model, training_path)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--data-path", type=str, default="./cirr_dataset")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", default=300, type=int)
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str)
    parser.add_argument("--backbone", type=str, default="pretrain")
    parser.add_argument("--learning-rate", default=2e-6, type=float)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--loss-align", default=0.4, type=float)
    parser.add_argument("--loss-rtc", default=0.4, type=float)
    parser.add_argument("--loss-itm", default=1, type=float)
    parser.add_argument("--validation-frequency", default=1, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float)
    parser.add_argument("--transform", default="targetpad", type=str)
    parser.add_argument("--save-training", dest="save_training", action='store_true')
    parser.add_argument("--save-best", dest="save_best", action='store_true')
    parser.add_argument("--save-memory", dest="save_memory", action='store_true')
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

    args = parser.parse_args()
    
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "blip_model_name": args.blip_model_name,
        "backbone": args.backbone,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "data_path": args.data_path,
        "loss_rtc": args.loss_rtc,
        "loss_align": args.loss_align,
        "loss_itm": args.loss_itm,
        "save_memory": args.save_memory,
        "experiment_name": args.experiment_name,
        # ANCE params
        "ance_num_negatives": args.ance_num_negatives,
        "ance_topk_candidates": args.ance_topk_candidates,
        "ance_refresh_interval": args.ance_refresh_interval,
        "ance_weight": args.ance_weight,
        "ance_warmup_epochs": args.ance_warmup_epochs,
    }

    if args.dataset.lower() == 'cirr':
        clip_finetune_cirr_ance(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update({
            'train_dress_types': ['dress', 'toptee', 'shirt'],
            'val_dress_types': ['dress', 'toptee', 'shirt']
        })
        clip_finetune_fiq_ance(**training_hyper_params)

