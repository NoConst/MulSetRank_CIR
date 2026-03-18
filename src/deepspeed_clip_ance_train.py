# -*- coding: utf-8 -*-
"""
DeepSpeed ZeRO-2 enabled version of:
CLIP Fine-tuning with ANCE using Hugging Face Transformers

- Keeps training logic identical to the original script.
- Adds: DeepSpeed initialization, ZeRO-2 optimization, rank0-only logging/validation/saving.
"""

import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import os
import logging

# ===== DeepSpeed =====
import deepspeed
from deepspeed import comm as dist

# Hugging Face Transformers
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

# LoRA
from peft import LoraConfig, get_peft_model

from cir_fusion import attach_cir_fusion, compose_query_features, load_cir_fusion, save_cir_fusion
from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description
from clip_ance_utils import (
    CLIPHardNegativeMiner,
    compute_clip_ance_loss,
    enable_model_logit_scale_training,
    get_model_logit_scale,
    get_similarity_scale,
    save_model_logit_scale,
    set_model_logit_scale,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Original mappings (unchanged)
# -------------------------
CLIP_MODEL_MAPPING = {
    "RN50": "openai/clip-vit-base-patch32",
    "RN101": "openai/clip-vit-base-patch32",
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-L/14": "openai/clip-vit-large-patch14",
    "ViT-L/14-336": "openai/clip-vit-large-patch14-336",
    "ViT-H/14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "ViT-G": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
}
EMBEDDING_DIMS = {
    "openai/clip-vit-base-patch32": 512,
    "openai/clip-vit-base-patch16": 512,
    "openai/clip-vit-large-patch14": 768,
    "openai/clip-vit-large-patch14-336": 768,
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": 1024,
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": 1280,
}

# =========================================================
# DeepSpeed distributed helpers
# =========================================================
def init_distributed() -> dict:
    """Initialize distributed environment from DeepSpeed."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        # DeepSpeed will handle init_process_group
        deepspeed.init_distributed(dist_backend="nccl")
        return {"enabled": True, "rank": rank, "world_size": world_size, "local_rank": local_rank}
    return {"enabled": False, "rank": 0, "world_size": 1, "local_rank": 0}

def is_main_process(dist_info: dict) -> bool:
    return (not dist_info["enabled"]) or dist_info["rank"] == 0

def barrier(dist_info: dict) -> None:
    if dist_info["enabled"]:
        dist.barrier()

def broadcast_object(dist_info: dict, obj):
    """Broadcast a Python object from rank0 to all ranks."""
    if not dist_info["enabled"]:
        return obj
    obj_list = [obj] if dist_info["rank"] == 0 else [None]
    torch.distributed.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def all_reduce_sum(dist_info: dict, t: torch.Tensor) -> torch.Tensor:
    if not dist_info["enabled"]:
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

# =========================================================
# Original utilities (minimally adjusted to accept device)
# =========================================================
def get_clip_model_and_processor(
    model_name: str,
    device: torch.device,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    init_temperature: Optional[float] = None,
    fusion_type: str = "adaptive_residual",
    fusion_hidden_dim: Optional[int] = None,
    fusion_dropout: float = 0.1,
    gradient_checkpointing: bool = False,
):
    # Map simple names to HF model IDs
    hf_model_name = CLIP_MODEL_MAPPING.get(model_name, model_name)
    logger.info(f"Loading CLIP model: {hf_model_name}")

    model = CLIPModel.from_pretrained(hf_model_name)
    processor = CLIPProcessor.from_pretrained(hf_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(hf_model_name)
    embedding_dim = EMBEDDING_DIMS.get(hf_model_name, 512)

    if use_lora:
        logger.info(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
            modules_to_save=None,
        )
        model = get_peft_model(model, lora_config)
        enable_model_logit_scale_training(model, logger)
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(
            f"LoRA enabled: trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.4f}%"
        )
    else:
        enable_model_logit_scale_training(model, logger)

    if init_temperature is not None:
        set_model_logit_scale(model, init_temperature, logger)

    model_source_path = Path(hf_model_name)
    loaded_fusion = False
    if model_source_path.exists():
        loaded_fusion = load_cir_fusion(model, model_source_path, logger)
    if not loaded_fusion:
        attach_cir_fusion(
            model,
            embedding_dim=embedding_dim,
            fusion_type=fusion_type,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
            log=logger,
        )

    if gradient_checkpointing:
        if use_lora and hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except NotImplementedError:
                logger.warning(
                    "Skipping enable_input_require_grads(): CLIPModel does not expose get_input_embeddings()."
                )
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            logger.info("Enabled gradient checkpointing for CLIP model")
        else:
            logger.warning("Gradient checkpointing requested, but this model does not support it")

    # Move model to device
    model = model.to(device)
    logger.info(f"Model loaded. Embedding dimension: {embedding_dim}")
    return model, processor, tokenizer, embedding_dim

def encode_text_hf(clip_model, tokenizer, texts, device):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)
    # Use autocast for mixed precision compatibility
    with torch.cuda.amp.autocast():
        text_features = clip_model.get_text_features(**inputs)
    return text_features

def extract_clip_index_features(dataset, clip_model, device, batch_size=64, num_workers=4):
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    clip_model.eval()
    all_features = []
    all_names = []
    for names, images in tqdm(dataloader, desc="Extracting index features", disable=not hasattr(tqdm, "__call__")):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features = clip_model.get_image_features(pixel_values=images)
                image_features = F.normalize(image_features, dim=-1)
            all_features.append(image_features.cpu())
            all_names.extend(names)
    return torch.vstack(all_features).to(device), all_names

# =========================================================
# DeepSpeed save helpers
# =========================================================
def deepspeed_save_hf_pretrained(
    model_engine,
    processor: CLIPProcessor,
    tokenizer: CLIPTokenizer,
    save_dir: Path,
    dist_info: dict,
):
    """Save HF model under DeepSpeed correctly (rank0-only state dict)."""
    if not is_main_process(dist_info):
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Get the underlying model from DeepSpeed engine
    model = model_engine.module

    if hasattr(model, "peft_config"):
        model.save_pretrained(str(save_dir))
    else:
        base_state_dict = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("cir_fusion.")
        }
        model.save_pretrained(str(save_dir), state_dict=base_state_dict)

    processor.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    save_model_logit_scale(model, save_dir, logger)
    save_cir_fusion(model, save_dir, logger)

def deepspeed_save_checkpoint(
    model_engine,
    save_dir: Path,
    client_state: dict = None,
):
    """Save DeepSpeed checkpoint (includes optimizer states for ZeRO)."""
    model_engine.save_checkpoint(str(save_dir), client_state=client_state)


def build_trainable_param_groups(
    model,
    default_learning_rate: float,
    lora_learning_rate: Optional[float] = None,
    fusion_learning_rate: Optional[float] = None,
):
    """Create optimizer groups so LoRA and CIR fusion can use different learning rates."""
    lora_lr = float(lora_learning_rate if lora_learning_rate is not None else default_learning_rate)
    fusion_lr = float(fusion_learning_rate if fusion_learning_rate is not None else default_learning_rate)
    default_lr = float(default_learning_rate)

    lora_params = []
    fusion_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("cir_fusion."):
            fusion_params.append(param)
        elif "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr})
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": fusion_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": default_lr})

    logger.info(
        "Optimizer param groups: "
        f"lora={len(lora_params)} tensors @ {lora_lr:.2e}, "
        f"fusion={len(fusion_params)} tensors @ {fusion_lr:.2e}, "
        f"other={len(other_params)} tensors @ {default_lr:.2e}"
    )
    return param_groups

# =========================================================
# Validation metrics (unchanged except device passed in)
# =========================================================
def compute_fiq_val_metrics_clip(relative_val_dataset, clip_model, tokenizer, index_features, index_names, device):
    print(f"Computing FashionIQ {relative_val_dataset.dress_types} validation metrics")
    clip_model.eval()
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset, batch_size=32,
        num_workers=4, pin_memory=True, collate_fn=collate_fn, shuffle=False
    )
    name_to_feat = dict(zip(index_names, index_features))
    predicted_features = []
    target_names = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):
        flattened_captions = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
            for i in range(0, len(flattened_captions), 2)
        ]
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                from operator import itemgetter
                if len(input_captions) == 1:
                    reference_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
                else:
                    reference_features = torch.stack(itemgetter(*reference_names)(name_to_feat))

                text_features = encode_text_hf(clip_model, tokenizer, input_captions, device)
                reference_features = F.normalize(reference_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                batch_predicted = compose_query_features(clip_model, reference_features, text_features)
            predicted_features.append(batch_predicted.cpu())

        target_names.extend(batch_target_names)

    predicted_features = torch.vstack(predicted_features).to(device)
    with torch.cuda.amp.autocast():
        similarities = predicted_features @ index_features.T
    distances = 1 - similarities
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1)
    )
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    return recall_at10, recall_at50

def compute_cirr_val_metrics_clip(relative_val_dataset, clip_model, tokenizer, index_features, index_names, device):
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
            with torch.cuda.amp.autocast():
                from operator import itemgetter
                if len(captions) == 1:
                    reference_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
                else:
                    reference_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))
                text_features = encode_text_hf(clip_model, tokenizer, captions, device)
                reference_features = F.normalize(reference_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                batch_predicted = compose_query_features(clip_model, reference_features, text_features)
            predicted_features.append(batch_predicted.cpu())

        reference_names.extend(batch_reference_names)
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)

    predicted_features = torch.vstack(predicted_features).to(device)
    with torch.cuda.amp.autocast():
        similarities = predicted_features @ index_features.T
    distances = 1 - similarities
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1)
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], -1)

    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1)
    )

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

# =========================================================
# Training loops: keep identical math, add DeepSpeed/rank0-only io
# =========================================================
def clip_finetune_fiq_ance(
    train_dress_types: List[str],
    val_dress_types: List[str],
    num_epochs: int,
    clip_model_name: str,
    learning_rate: float,
    batch_size: int,  # per-GPU batch
    validation_frequency: int,
    transform: str,
    save_training: bool,
    save_best: bool,
    ance_num_negatives: int = 16,
    ance_topk_candidates: int = 100,
    ance_refresh_interval: int = 1,
    ance_weight: float = 1.0,
    experiment_name: str = None,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    # DeepSpeed args
    dist_info: Optional[dict] = None,
    deepspeed_config: str = None,
    grad_accum_steps: int = 1,
    partial_intent_queries_path: str = None,
    **kwargs
):
    assert dist_info is not None
    device = torch.device(f"cuda:{dist_info['local_rank']}") if torch.cuda.is_available() else torch.device("cpu")
    init_temperature = kwargs.get("init_temperature")
    fusion_type = kwargs.get("fusion_type", "adaptive_residual")
    fusion_hidden_dim = kwargs.get("fusion_hidden_dim")
    fusion_dropout = kwargs.get("fusion_dropout", 0.1)
    lora_learning_rate = kwargs.get("lora_learning_rate", 5e-5)
    fusion_learning_rate = kwargs.get("fusion_learning_rate", 1e-4)
    gradient_checkpointing = kwargs.get("gradient_checkpointing", True)
    ance_index_device = kwargs.get("ance_index_device", "cpu")

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_name_clean = clip_model_name.replace("/", "-")
    lora_suffix = "_lora" if use_lora else ""
    if experiment_name:
        training_path = Path(base_path / f"models/clip_ance_fiq_{model_name_clean}{lora_suffix}_{experiment_name}_{training_start}")
    else:
        training_path = Path(base_path / f"models/clip_ance_fiq_{model_name_clean}{lora_suffix}_{training_start}")

    # rank0 creates dir, broadcast to all
    if is_main_process(dist_info):
        training_path.mkdir(exist_ok=False, parents=True)
    barrier(dist_info)
    training_path = Path(broadcast_object(dist_info, str(training_path)))

    clip_model, processor, tokenizer, embedding_dim = get_clip_model_and_processor(
        clip_model_name, device,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_temperature=init_temperature,
        fusion_type=fusion_type,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        gradient_checkpointing=gradient_checkpointing,
    )

    # Save hyperparameters (rank0 only)
    if is_main_process(dist_info):
        training_hyper_params = {
            "num_epochs": num_epochs,
            "clip_model_name": clip_model_name,
            "learning_rate": learning_rate,
            "batch_size_per_gpu": batch_size,
            "world_size": dist_info["world_size"],
            "global_batch": batch_size * dist_info["world_size"] * grad_accum_steps,
            "embedding_dim": embedding_dim,
            "ance_num_negatives": ance_num_negatives,
            "ance_topk_candidates": ance_topk_candidates,
            "ance_refresh_interval": ance_refresh_interval,
            "ance_weight": ance_weight,
            "listwise_weight": kwargs.get("listwise_weight", 0.2),
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "fusion_type": fusion_type,
            "fusion_hidden_dim": fusion_hidden_dim,
            "fusion_dropout": fusion_dropout,
            "lora_learning_rate": lora_learning_rate,
            "fusion_learning_rate": fusion_learning_rate,
            "learnable_temperature": True,
            "init_temperature": init_temperature,
            "gradient_checkpointing": gradient_checkpointing,
            "ance_index_device": ance_index_device,
            "deepspeed_zero_stage": 2,
            "grad_accum_steps": grad_accum_steps,
        }
        with open(training_path / "training_hyperparameters.json", "w+") as f:
            json.dump(training_hyper_params, f, sort_keys=True, indent=4)

    input_dim = 224
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
    elif transform == "targetpad":
        target_ratio = kwargs.get("target_ratio", 1.25)
        preprocess = targetpad_transform(target_ratio, input_dim)
    else:
        raise ValueError("Preprocess transform should be in ['squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets, classic_val_datasets = [], []
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_datasets.append(FashionIQDataset("val", [dress_type], "relative", preprocess))
        classic_val_datasets.append(FashionIQDataset("val", [dress_type], "classic", preprocess))

    relative_train_dataset = FashionIQDataset("train", train_dress_types, "relative", preprocess)
    classic_train_dataset = FashionIQDataset("train", train_dress_types, "classic", preprocess, preload_images=True)

    # Load partial intent queries for partial intent negative mining
    partial_intent_queries = None
    if partial_intent_queries_path and os.path.exists(partial_intent_queries_path):
        with open(partial_intent_queries_path, "r") as f:
            partial_intent_queries = json.load(f)
        logger.info(f"Loaded {len(partial_intent_queries)} partial intent queries from {partial_intent_queries_path}")

    # DistributedSampler for training
    if dist_info["enabled"]:
        train_sampler = DistributedSampler(relative_train_dataset, shuffle=True, drop_last=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        num_workers=kwargs.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=shuffle,
        sampler=train_sampler,
        prefetch_factor=2,
        persistent_workers=True,
    )

    hard_negative_miner = CLIPHardNegativeMiner(
        embedding_dim=embedding_dim,
        num_negatives=ance_num_negatives,
        topk_candidates=ance_topk_candidates,
        refresh_interval=ance_refresh_interval,
        use_gpu=ance_index_device == "cuda",
        cache_dir=str(training_path / "ance_cache")
    )

    logger.info("Building initial embedding index...")
    hard_negative_miner.build_index(
        clip_model=clip_model,
        dataset=classic_train_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=kwargs.get("num_workers", 4)
    )

    # =========================================================
    # DeepSpeed initialization
    # =========================================================
    # Load DeepSpeed config
    with open(deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    
    # Update config with runtime values
    ds_config["train_micro_batch_size_per_gpu"] = batch_size
    ds_config["gradient_accumulation_steps"] = grad_accum_steps
    ds_config["train_batch_size"] = batch_size * dist_info["world_size"] * grad_accum_steps
    
    # Update optimizer lr
    if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
        ds_config["optimizer"]["params"]["lr"] = learning_rate
    model_parameters = build_trainable_param_groups(
        clip_model,
        default_learning_rate=learning_rate,
        lora_learning_rate=lora_learning_rate,
        fusion_learning_rate=fusion_learning_rate,
    )
    max_lrs = [group.get("lr", learning_rate) for group in model_parameters]
    
    # Initialize DeepSpeed with model and optimizer
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=clip_model,
        model_parameters=model_parameters,
        config=ds_config,
    )
    train_logit_scale = get_model_logit_scale(model_engine.module)

    # Create scheduler manually (DeepSpeed's OneCycle scheduler has limited flexibility)
    # We'll use PyTorch's OneCycleLR instead
    scheduler = OneCycleLR(
        optimizer.optimizer,  # Access underlying optimizer from DeepSpeed
        max_lr=max_lrs,
        pct_start=1.5 / num_epochs,
        div_factor=100.0,
        steps_per_epoch=len(relative_train_loader) // grad_accum_steps,
        epochs=num_epochs
    )

    best_avg_recall = 0.0 if save_best else None
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    logger.info("Training loop started (DeepSpeed ZeRO-2 enabled)")
    for epoch in range(num_epochs):
        if dist_info["enabled"]:
            train_sampler.set_epoch(epoch)

        # Refresh ANCE index periodically
        if epoch > 0:
            # For validation/index building, we need the model in eval mode
            model_engine.eval()
            hard_negative_miner.refresh_index(
                clip_model=model_engine.module,
                dataset=classic_train_dataset,
                device=device,
                current_epoch=epoch,
                batch_size=batch_size,
                num_workers=kwargs.get("num_workers", 4)
            )

        train_running_results = {"accumulated_train_loss": 0.0, "images_in_epoch": 0}
        train_iter = relative_train_loader
        train_bar = tqdm(train_iter, ncols=150, disable=not is_main_process(dist_info))

        for step, (reference_images, target_images, captions, target_names) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            # Process captions: combine the two captions for FashionIQ
            flattened_captions = np.array(captions).T.flatten().tolist()
            input_captions = [
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
                for i in range(0, len(flattened_captions), 2)
            ]

            model_engine.train()

            # DeepSpeed handles mixed precision automatically via config
            ref_features = model_engine.module.get_image_features(pixel_values=reference_images)
            target_features = model_engine.module.get_image_features(pixel_values=target_images)
            text_features = encode_text_hf(model_engine.module, tokenizer, input_captions, device)

            ref_features = F.normalize(ref_features, dim=-1)
            target_features = F.normalize(target_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            query_features = compose_query_features(model_engine.module, ref_features, text_features)

            hard_negative_names = None
            hard_neg_indices = None
            ref_hard_negative_names = None
            with torch.no_grad():
                hard_neg_indices, hard_negative_names = hard_negative_miner.mine_hard_negatives(
                    query_features=query_features.detach(),
                    positive_names=list(target_names),
                    return_names=True
                )
                _, ref_hard_negative_names = hard_negative_miner.mine_hard_negatives(
                    query_features=ref_features.detach(),
                    positive_names=list(target_names),
                    return_names=True
                )

            if hard_negative_names is not None:
                num_negatives = len(hard_negative_names[0])
                all_neg_names = [n for batch_names in hard_negative_names for n in batch_names]
                all_neg_images = classic_train_dataset.get_images_batch(all_neg_names).to(device, non_blocking=True)
                all_neg_features = model_engine.module.get_image_features(pixel_values=all_neg_images)
                all_neg_features = F.normalize(all_neg_features, dim=-1)
                hard_negative_features = all_neg_features.view(images_in_batch, num_negatives, -1)
                del all_neg_images, all_neg_features

                ref_hard_negative_features = None
                if ref_hard_negative_names is not None:
                    num_ref_negatives = len(ref_hard_negative_names[0])
                    all_ref_names = [n for batch_names in ref_hard_negative_names for n in batch_names]
                    all_ref_images = classic_train_dataset.get_images_batch(all_ref_names)
                    all_ref_images = all_ref_images.to(device, non_blocking=True)
                    all_ref_features = model_engine.module.get_image_features(pixel_values=all_ref_images)
                    all_ref_features = F.normalize(all_ref_features, dim=-1)
                    ref_hard_negative_features = all_ref_features.view(images_in_batch, num_ref_negatives, -1)
                    del all_ref_images, all_ref_features

                # Mine partial intent negatives
                partial_intent_negative_features = None
                if partial_intent_queries is not None and hard_neg_indices is not None:
                    partial_intents = [
                        partial_intent_queries.get(cap, cap)
                        for cap in input_captions
                    ]
                    pi_num_neg = kwargs.get("partial_intent_num_negatives", ance_num_negatives)
                    with torch.no_grad():
                        _, partial_intent_neg_names = hard_negative_miner.mine_partial_intent_negatives(
                            partial_intent_texts=partial_intents,
                            ref_features=ref_features.detach(),
                            positive_names=list(target_names),
                            hard_negative_indices=hard_neg_indices,
                            num_negatives=pi_num_neg,
                            clip_model=model_engine.module,
                            tokenizer=tokenizer
                        )
                    num_pi_neg = len(partial_intent_neg_names[0])
                    all_pi_names = [n for batch_names in partial_intent_neg_names for n in batch_names]
                    all_pi_images = classic_train_dataset.get_images_batch(all_pi_names).to(device, non_blocking=True)
                    all_pi_features = model_engine.module.get_image_features(pixel_values=all_pi_images)
                    all_pi_features = F.normalize(all_pi_features, dim=-1)
                    partial_intent_negative_features = all_pi_features.view(images_in_batch, num_pi_neg, -1)
                    del all_pi_images, all_pi_features

                loss = compute_clip_ance_loss(
                    query_features=query_features,
                    target_features=target_features,
                    hard_negative_features=hard_negative_features,
                    hard_negative_weight=ance_weight,
                    ref_hard_negative_features=ref_hard_negative_features,
                    ref_hard_negative_weight=kwargs.get("ref_ance_weight", 1.0),
                    partial_intent_negative_features=partial_intent_negative_features,
                    partial_intent_negative_weight=kwargs.get("partial_intent_weight", 0.75),
                    listwise_weight=kwargs.get("listwise_weight", 0.2),
                    logit_scale=train_logit_scale,
                )
            else:
                scale = get_similarity_scale(logit_scale=train_logit_scale)
                sim_matrix = torch.matmul(query_features.float(), target_features.float().T) * scale
                labels = torch.arange(images_in_batch, dtype=torch.long, device=device)
                loss = F.cross_entropy(sim_matrix, labels)

            loss_for_logging = loss.detach().float().cpu()

            # DeepSpeed handles backward, gradient accumulation, and optimizer step
            model_engine.backward(loss)
            model_engine.step()

            # Step scheduler at the right frequency
            if (step + 1) % grad_accum_steps == 0:
                scheduler.step()

            update_train_running_results(train_running_results, loss_for_logging, images_in_batch)
            if is_main_process(dist_info):
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

        # Global loss reduce for logging (rank0 writes)
        total_loss = torch.tensor([train_running_results["accumulated_train_loss"]], device=device)
        total_imgs = torch.tensor([train_running_results["images_in_epoch"]], device=device, dtype=torch.long)
        total_loss = all_reduce_sum(dist_info, total_loss)
        total_imgs = all_reduce_sum(dist_info, total_imgs)

        if is_main_process(dist_info):
            avg_loss = (total_loss.item() / max(int(total_imgs.item()), 1))
            current_temperature = 1.0 / float(
                get_similarity_scale(logit_scale=train_logit_scale).detach().cpu().item()
            )
            loss_log_dict = {"epoch": epoch, "loss": avg_loss, "temperature": current_temperature}
            training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
            training_log_frame.to_csv(str(training_path / "train_metrics.csv"), index=False)

        barrier(dist_info)

        # Validation (rank0 only, others wait)
        if epoch % validation_frequency == 0:
            if is_main_process(dist_info):
                model_engine.eval()
                recalls_at10, recalls_at50 = [], []

                for relative_val_dataset, classic_val_dataset, idx in zip(
                    relative_val_datasets, classic_val_datasets, idx_to_dress_mapping
                ):
                    torch.cuda.empty_cache()
                    index_features, index_names = extract_clip_index_features(
                        classic_val_dataset, model_engine.module, device=device
                    )
                    r10, r50 = compute_fiq_val_metrics_clip(
                        relative_val_dataset, model_engine.module, tokenizer, index_features, index_names, device=device
                    )
                    recalls_at10.append(r10)
                    recalls_at50.append(r50)
                    del index_features, index_names
                    torch.cuda.empty_cache()

                results_dict = {f"{idx_to_dress_mapping[i]}_recall_at10": recalls_at10[i] for i in range(len(recalls_at10))}
                results_dict.update({f"{idx_to_dress_mapping[i]}_recall_at50": recalls_at50[i] for i in range(len(recalls_at50))})
                results_dict.update({
                    "average_recall_at10": mean(recalls_at10),
                    "average_recall_at50": mean(recalls_at50),
                    "average_recall": (mean(recalls_at50) + mean(recalls_at10)) / 2
                })
                print(json.dumps(results_dict, indent=4))

                log_dict = {"epoch": epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / "validation_metrics.csv"), index=False)

                if save_training and save_best and results_dict["average_recall"] > best_avg_recall:
                    best_avg_recall = results_dict["average_recall"]
                    deepspeed_save_hf_pretrained(model_engine, processor, tokenizer, training_path / "best_model", dist_info)
                    logger.info(f"Saved best model at epoch {epoch} with recall {best_avg_recall:.2f}")

            if save_training:
                deepspeed_save_hf_pretrained(model_engine, processor, tokenizer, training_path / "latest_model", dist_info)
                if is_main_process(dist_info):
                    logger.info(f"Saved latest model at epoch {epoch}")

            barrier(dist_info)

    if save_training:
        deepspeed_save_hf_pretrained(model_engine, processor, tokenizer, training_path / "final_model", dist_info)
        if is_main_process(dist_info):
            logger.info(f"Saved final model to {training_path / 'final_model'}")
    barrier(dist_info)

def clip_finetune_cirr_ance(
    num_epochs: int,
    clip_model_name: str,
    learning_rate: float,
    batch_size: int,
    validation_frequency: int,
    transform: str,
    save_training: bool,
    save_best: bool,
    ance_num_negatives: int = 16,
    ance_topk_candidates: int = 100,
    ance_refresh_interval: int = 1,
    ance_weight: float = 1.0,
    experiment_name: str = None,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    # DeepSpeed args
    dist_info: Optional[dict] = None,
    deepspeed_config: str = None,
    grad_accum_steps: int = 1,
    partial_intent_queries_path: str = None,
    **kwargs
):
    assert dist_info is not None
    device = torch.device(f"cuda:{dist_info['local_rank']}") if torch.cuda.is_available() else torch.device("cpu")
    init_temperature = kwargs.get("init_temperature")
    fusion_type = kwargs.get("fusion_type", "adaptive_residual")
    fusion_hidden_dim = kwargs.get("fusion_hidden_dim")
    fusion_dropout = kwargs.get("fusion_dropout", 0.1)
    lora_learning_rate = kwargs.get("lora_learning_rate", 5e-5)
    fusion_learning_rate = kwargs.get("fusion_learning_rate", 1e-4)
    gradient_checkpointing = kwargs.get("gradient_checkpointing", True)
    ance_index_device = kwargs.get("ance_index_device", "cpu")

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_name_clean = clip_model_name.replace("/", "-")
    lora_suffix = "_lora" if use_lora else ""
    if experiment_name:
        training_path = Path(base_path / f"models/clip_ance_cirr_{model_name_clean}{lora_suffix}_{experiment_name}_{training_start}")
    else:
        training_path = Path(base_path / f"models/clip_ance_cirr_{model_name_clean}{lora_suffix}_{training_start}")

    if is_main_process(dist_info):
        training_path.mkdir(exist_ok=False, parents=True)
    barrier(dist_info)
    training_path = Path(broadcast_object(dist_info, str(training_path)))

    clip_model, processor, tokenizer, embedding_dim = get_clip_model_and_processor(
        clip_model_name, device,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_temperature=init_temperature,
        fusion_type=fusion_type,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        gradient_checkpointing=gradient_checkpointing,
    )

    if is_main_process(dist_info):
        training_hyper_params = {
            "num_epochs": num_epochs,
            "clip_model_name": clip_model_name,
            "learning_rate": learning_rate,
            "batch_size_per_gpu": batch_size,
            "world_size": dist_info["world_size"],
            "global_batch": batch_size * dist_info["world_size"] * grad_accum_steps,
            "embedding_dim": embedding_dim,
            "ance_num_negatives": ance_num_negatives,
            "ance_topk_candidates": ance_topk_candidates,
            "ance_refresh_interval": ance_refresh_interval,
            "ance_weight": ance_weight,
            "listwise_weight": kwargs.get("listwise_weight", 0.2),
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "fusion_type": fusion_type,
            "fusion_hidden_dim": fusion_hidden_dim,
            "fusion_dropout": fusion_dropout,
            "lora_learning_rate": lora_learning_rate,
            "fusion_learning_rate": fusion_learning_rate,
            "learnable_temperature": True,
            "init_temperature": init_temperature,
            "gradient_checkpointing": gradient_checkpointing,
            "ance_index_device": ance_index_device,
            "deepspeed_zero_stage": 2,
            "grad_accum_steps": grad_accum_steps,
        }
        with open(training_path / "training_hyperparameters.json", "w+") as f:
            json.dump(training_hyper_params, f, sort_keys=True, indent=4)

    input_dim = 224
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
    elif transform == "targetpad":
        target_ratio = kwargs.get("target_ratio", 1.25)
        preprocess = targetpad_transform(target_ratio, input_dim)
    else:
        raise ValueError("Preprocess transform should be in ['squarepad', 'targetpad']")

    relative_val_dataset = CIRRDataset("val", "relative", preprocess)
    classic_val_dataset = CIRRDataset("val", "classic", preprocess)
    relative_train_dataset = CIRRDataset("train", "relative", preprocess)

    classic_train_dataset = CIRRDataset("train", "classic", preprocess, preload_images=True)

    # Load partial intent queries for partial intent negative mining
    partial_intent_queries = None
    if partial_intent_queries_path and os.path.exists(partial_intent_queries_path):
        with open(partial_intent_queries_path, "r") as f:
            partial_intent_queries = json.load(f)
        logger.info(f"Loaded {len(partial_intent_queries)} partial intent queries from {partial_intent_queries_path}")

    if dist_info["enabled"]:
        train_sampler = DistributedSampler(relative_train_dataset, shuffle=True, drop_last=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        num_workers=kwargs.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=shuffle,
        sampler=train_sampler,
        prefetch_factor=2,
        persistent_workers=True,
    )

    hard_negative_miner = CLIPHardNegativeMiner(
        embedding_dim=embedding_dim,
        num_negatives=ance_num_negatives,
        topk_candidates=ance_topk_candidates,
        refresh_interval=ance_refresh_interval,
        use_gpu=ance_index_device == "cuda",
        cache_dir=str(training_path / "ance_cache")
    )

    logger.info("Building initial embedding index for CIRR...")
    hard_negative_miner.build_index(
        clip_model=clip_model,
        dataset=classic_train_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=kwargs.get("num_workers", 4)
    )

    # =========================================================
    # DeepSpeed initialization
    # =========================================================
    with open(deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    
    ds_config["train_micro_batch_size_per_gpu"] = batch_size
    ds_config["gradient_accumulation_steps"] = grad_accum_steps
    ds_config["train_batch_size"] = batch_size * dist_info["world_size"] * grad_accum_steps
    
    if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
        ds_config["optimizer"]["params"]["lr"] = learning_rate
    model_parameters = build_trainable_param_groups(
        clip_model,
        default_learning_rate=learning_rate,
        lora_learning_rate=lora_learning_rate,
        fusion_learning_rate=fusion_learning_rate,
    )
    max_lrs = [group.get("lr", learning_rate) for group in model_parameters]
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=clip_model,
        model_parameters=model_parameters,
        config=ds_config,
    )
    train_logit_scale = get_model_logit_scale(model_engine.module)

    scheduler = OneCycleLR(
        optimizer.optimizer,
        max_lr=max_lrs,
        pct_start=1/50,
        steps_per_epoch=len(relative_train_loader) // grad_accum_steps,
        epochs=num_epochs
    )

    best_arithmetic = 0.0 if save_best else None
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    logger.info("Training loop started for CIRR (DeepSpeed ZeRO-2 enabled)")
    for epoch in range(num_epochs):
        if dist_info["enabled"]:
            train_sampler.set_epoch(epoch)

        # Refresh ANCE index periodically
        if epoch > 0:
            model_engine.eval()
            hard_negative_miner.refresh_index(
                clip_model=model_engine.module,
                dataset=classic_train_dataset,
                device=device,
                current_epoch=epoch,
                batch_size=batch_size,
                num_workers=kwargs.get("num_workers", 4)
            )

        train_running_results = {"accumulated_train_loss": 0.0, "images_in_epoch": 0}
        train_bar = tqdm(relative_train_loader, ncols=150, disable=not is_main_process(dist_info))

        for step, (reference_images, target_images, captions, target_names) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            # CIRR captions are already strings, just convert to list
            input_captions = list(captions)

            model_engine.train()
            
            ref_features = model_engine.module.get_image_features(pixel_values=reference_images)
            target_features = model_engine.module.get_image_features(pixel_values=target_images)
            text_features = encode_text_hf(model_engine.module, tokenizer, input_captions, device)

            ref_features = F.normalize(ref_features, dim=-1)
            target_features = F.normalize(target_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            query_features = compose_query_features(model_engine.module, ref_features, text_features)

            hard_negative_names = None
            hard_neg_indices = None
            ref_hard_negative_names = None
            with torch.no_grad():
                hard_neg_indices, hard_negative_names = hard_negative_miner.mine_hard_negatives(
                    query_features=query_features.detach(),
                    positive_names=list(target_names),
                    return_names=True
                )
                _, ref_hard_negative_names = hard_negative_miner.mine_hard_negatives(
                    query_features=ref_features.detach(),
                    positive_names=list(target_names),
                    return_names=True
                )

            if hard_negative_names is not None:
                num_negatives = len(hard_negative_names[0])
                all_neg_names = [n for batch_names in hard_negative_names for n in batch_names]
                all_neg_images = classic_train_dataset.get_images_batch(all_neg_names).to(device, non_blocking=True)
                all_neg_features = model_engine.module.get_image_features(pixel_values=all_neg_images)
                all_neg_features = F.normalize(all_neg_features, dim=-1)
                hard_negative_features = all_neg_features.view(images_in_batch, num_negatives, -1)
                del all_neg_images, all_neg_features

                ref_hard_negative_features = None
                if ref_hard_negative_names is not None:
                    num_ref_negatives = len(ref_hard_negative_names[0])
                    all_ref_names = [n for batch_names in ref_hard_negative_names for n in batch_names]
                    all_ref_images = classic_train_dataset.get_images_batch(all_ref_names).to(device, non_blocking=True)
                    all_ref_features = model_engine.module.get_image_features(pixel_values=all_ref_images)
                    all_ref_features = F.normalize(all_ref_features, dim=-1)
                    ref_hard_negative_features = all_ref_features.view(images_in_batch, num_ref_negatives, -1)
                    del all_ref_images, all_ref_features

                # Mine partial intent negatives
                partial_intent_negative_features = None
                if partial_intent_queries is not None and hard_neg_indices is not None:
                    partial_intents = [
                        partial_intent_queries.get(cap, cap)
                        for cap in input_captions
                    ]
                    pi_num_neg = kwargs.get("partial_intent_num_negatives", ance_num_negatives)
                    with torch.no_grad():
                        _, partial_intent_neg_names = hard_negative_miner.mine_partial_intent_negatives(
                            partial_intent_texts=partial_intents,
                            ref_features=ref_features.detach(),
                            positive_names=list(target_names),
                            hard_negative_indices=hard_neg_indices,
                            num_negatives=pi_num_neg,
                            clip_model=model_engine.module,
                            tokenizer=tokenizer
                        )
                    num_pi_neg = len(partial_intent_neg_names[0])
                    all_pi_names = [n for batch_names in partial_intent_neg_names for n in batch_names]
                    all_pi_images = classic_train_dataset.get_images_batch(all_pi_names).to(device, non_blocking=True)
                    all_pi_features = model_engine.module.get_image_features(pixel_values=all_pi_images)
                    all_pi_features = F.normalize(all_pi_features, dim=-1)
                    partial_intent_negative_features = all_pi_features.view(images_in_batch, num_pi_neg, -1)
                    del all_pi_images, all_pi_features

                loss = compute_clip_ance_loss(
                    query_features=query_features,
                    target_features=target_features,
                    hard_negative_features=hard_negative_features,
                    hard_negative_weight=ance_weight,
                    ref_hard_negative_features=ref_hard_negative_features,
                    ref_hard_negative_weight=kwargs.get("ref_ance_weight", 1.0),
                    partial_intent_negative_features=partial_intent_negative_features,
                    partial_intent_negative_weight=kwargs.get("partial_intent_weight", 0.75),
                    listwise_weight=kwargs.get("listwise_weight", 0.2),
                    logit_scale=train_logit_scale,
                )
            else:
                scale = get_similarity_scale(logit_scale=train_logit_scale)
                sim_matrix = torch.matmul(query_features.float(), target_features.float().T) * scale
                labels = torch.arange(images_in_batch, dtype=torch.long, device=device)
                loss = F.cross_entropy(sim_matrix, labels)

            loss_for_logging = loss.detach().float().cpu()

            model_engine.backward(loss)
            model_engine.step()

            if (step + 1) % grad_accum_steps == 0:
                scheduler.step()

            update_train_running_results(train_running_results, loss_for_logging, images_in_batch)
            if is_main_process(dist_info):
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

        total_loss = torch.tensor([train_running_results["accumulated_train_loss"]], device=device)
        total_imgs = torch.tensor([train_running_results["images_in_epoch"]], device=device, dtype=torch.long)
        total_loss = all_reduce_sum(dist_info, total_loss)
        total_imgs = all_reduce_sum(dist_info, total_imgs)

        if is_main_process(dist_info):
            avg_loss = (total_loss.item() / max(int(total_imgs.item()), 1))
            current_temperature = 1.0 / float(
                get_similarity_scale(logit_scale=train_logit_scale).detach().cpu().item()
            )
            loss_log_dict = {"epoch": epoch, "loss": avg_loss, "temperature": current_temperature}
            training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
            training_log_frame.to_csv(str(training_path / "train_metrics.csv"), index=False)

        barrier(dist_info)

        if epoch % validation_frequency == 0:
            if is_main_process(dist_info):
                model_engine.eval()
                torch.cuda.empty_cache()
                val_index_features, val_index_names = extract_clip_index_features(
                    classic_val_dataset, model_engine.module, device=device
                )
                results = compute_cirr_val_metrics_clip(
                    relative_val_dataset, model_engine.module, tokenizer, val_index_features, val_index_names, device=device
                )
                group_r1, group_r2, group_r3, r1, r5, r10, r50 = results
                results_dict = {
                    "group_recall_at1": group_r1,
                    "group_recall_at2": group_r2,
                    "group_recall_at3": group_r3,
                    "recall_at1": r1,
                    "recall_at5": r5,
                    "recall_at10": r10,
                    "recall_at50": r50,
                    "mean(R@5+R_s@1)": (group_r1 + r5) / 2,
                    "arithmetic_mean": mean(results),
                    "harmonic_mean": harmonic_mean(results),
                    "geometric_mean": geometric_mean(results),
                }
                print(json.dumps(results_dict, indent=4))

                del val_index_features, val_index_names
                torch.cuda.empty_cache()

                log_dict = {"epoch": epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / "validation_metrics.csv"), index=False)

                if save_training and save_best and results_dict["arithmetic_mean"] > best_arithmetic:
                    best_arithmetic = results_dict["arithmetic_mean"]
                    deepspeed_save_hf_pretrained(model_engine, processor, tokenizer, training_path / "best_model", dist_info)
                    logger.info(f"Saved best model at epoch {epoch} with arithmetic mean {best_arithmetic:.2f}")

            if save_training:
                deepspeed_save_hf_pretrained(model_engine, processor, tokenizer, training_path / "latest_model", dist_info)
                if is_main_process(dist_info):
                    logger.info(f"Saved latest model at epoch {epoch}")

            barrier(dist_info)

    if save_training:
        deepspeed_save_hf_pretrained(model_engine, processor, tokenizer, training_path / "final_model", dist_info)
        if is_main_process(dist_info):
            logger.info(f"Saved final model to {training_path / 'final_model'}")
    barrier(dist_info)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--clip-model-name", default="ViT-B/32", type=str)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    parser.add_argument("--batch-size", default=128, type=int)  # per-GPU
    parser.add_argument("--validation-frequency", default=1, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float)
    parser.add_argument("--transform", default="targetpad", type=str)
    parser.add_argument("--save-training", dest="save_training", action="store_true")
    parser.add_argument("--save-best", dest="save_best", action="store_true")
    parser.add_argument("--experiment-name", type=str, default=None)

    # ANCE
    parser.add_argument("--ance-num-negatives", default=16, type=int)
    parser.add_argument("--ance-topk-candidates", default=100, type=int)
    parser.add_argument("--ance-refresh-interval", default=1, type=int)
    parser.add_argument("--ance-weight", default=1.0, type=float)
    parser.add_argument("--ref-ance-weight", default=1.0, type=float)
    parser.add_argument("--partial-intent-num-negatives", default=None, type=int)
    parser.add_argument("--partial-intent-weight", default=0.75, type=float)
    parser.add_argument("--listwise-weight", default=0.2, type=float)
    parser.add_argument("--partial-intent-queries-path", type=str, default=None,
                        help="Path to partial intent queries JSON file")

    # LoRA
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16, type=int)
    parser.add_argument("--lora-dropout", default=0.1, type=float)
    parser.add_argument("--init-temperature", default=0.03, type=float,
                        help="Initial value for the learnable CLIP temperature.")
    parser.add_argument("--fusion-type", default="adaptive_residual", type=str,
                        choices=["sum", "adaptive_residual"])
    parser.add_argument("--fusion-hidden-dim", default=None, type=int)
    parser.add_argument("--fusion-dropout", default=0.1, type=float)
    parser.add_argument("--lora-learning-rate", default=5e-5, type=float)
    parser.add_argument("--fusion-learning-rate", default=1e-4, type=float)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--ance-index-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", default=41, type=int)

    # DeepSpeed
    parser.add_argument("--deepspeed-config", type=str, default="ds_config_zero2.json",
                        help="Path to DeepSpeed config file")
    parser.add_argument("--grad-accum-steps", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank passed by DeepSpeed launcher")
    parser.set_defaults(gradient_checkpointing=True)

    args = parser.parse_args()

    if args.dataset.lower() not in ["fashioniq", "cirr"]:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ'")

    # Initialize distributed
    dist_info = init_distributed()

    # Keep same default seed; distributed sampler handles shuffling via set_epoch
    set_seed(args.seed)

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
        "ance_num_negatives": args.ance_num_negatives,
        "ance_topk_candidates": args.ance_topk_candidates,
        "ance_refresh_interval": args.ance_refresh_interval,
        "ance_weight": args.ance_weight,
        "ref_ance_weight": args.ref_ance_weight,
        "partial_intent_num_negatives": args.partial_intent_num_negatives if args.partial_intent_num_negatives else args.ance_num_negatives,
        "partial_intent_weight": args.partial_intent_weight,
        "listwise_weight": args.listwise_weight,
        "partial_intent_queries_path": args.partial_intent_queries_path,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "fusion_type": args.fusion_type,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "fusion_dropout": args.fusion_dropout,
        "lora_learning_rate": args.lora_learning_rate,
        "fusion_learning_rate": args.fusion_learning_rate,
        "gradient_checkpointing": args.gradient_checkpointing,
        "learnable_temperature": True,
        "init_temperature": args.init_temperature,
        "ance_index_device": args.ance_index_device,
        "seed": args.seed,
        # DeepSpeed
        "dist_info": dist_info,
        "deepspeed_config": args.deepspeed_config,
        "grad_accum_steps": args.grad_accum_steps,
    }

    if args.dataset.lower() == "cirr":
        clip_finetune_cirr_ance(**training_hyper_params)
    else:
        training_hyper_params.update({
            "train_dress_types": ["dress", "toptee", "shirt"],
            "val_dress_types": ["dress", "toptee", "shirt"]
        })
        clip_finetune_fiq_ance(**training_hyper_params)
