# -*- coding: utf-8 -*-
"""
DeepSpeed ZeRO-2 training for fusion module with frozen CLIP.

Compared to deepspeed_clip_ance_train.py:
- CLIP encoder is frozen (used only for feature extraction + ANCE mining index).
- Trainable model is fusion module for text query + reference image composition.
"""

import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import logging


def _early_set_cuda_device_from_env() -> None:
    # Bind each DeepSpeed worker to its target GPU before importing modules that may touch CUDA.
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or not torch.cuda.is_available():
        return
    torch.cuda.set_device(int(local_rank))


_early_set_cuda_device_from_env()

# ===== DeepSpeed =====
import deepspeed
from deepspeed import comm as dist

# Hugging Face Transformers
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description
from clip_ance_utils import CLIPHardNegativeMiner, compute_clip_ance_loss
from fusion_module import get_fusion_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """Initialize distributed environment from DeepSpeed launcher variables."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed(dist_backend="nccl")
        return {"enabled": True, "rank": rank, "world_size": world_size, "local_rank": local_rank}
    return {"enabled": False, "rank": 0, "world_size": 1, "local_rank": 0}


def is_main_process(dist_info: dict) -> bool:
    return (not dist_info["enabled"]) or dist_info["rank"] == 0


def barrier(dist_info: dict) -> None:
    if dist_info["enabled"]:
        dist.barrier()


def broadcast_object(dist_info: dict, obj):
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
# Model helpers
# =========================================================
def get_clip_model_and_processor(
    model_name: str,
    device: torch.device,
    pretrained_path: Optional[str] = None,
) -> Tuple[CLIPModel, CLIPProcessor, CLIPTokenizer, int, str]:
    """Load CLIP model/tokenizer/processor from HF or local checkpoint."""
    hf_model_name = CLIP_MODEL_MAPPING.get(model_name, model_name)

    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"Loading Stage-1 CLIP checkpoint from: {pretrained_path}")
        adapter_config_path = Path(pretrained_path) / "adapter_config.json"
        if adapter_config_path.exists():
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(pretrained_path)
            hf_model_name = peft_config.base_model_name_or_path or hf_model_name
            logger.info(
                "Detected PEFT adapter checkpoint. Loading base CLIP on CPU, merging adapter, then moving to target device."
            )
            base_model = CLIPModel.from_pretrained(hf_model_name)
            model = PeftModel.from_pretrained(base_model, pretrained_path, torch_device="cpu").merge_and_unload()
            model = model.to(device)
        else:
            model = CLIPModel.from_pretrained(pretrained_path).to(device)
        try:
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_path)
            processor = CLIPProcessor.from_pretrained(pretrained_path)
        except Exception:
            logger.info(f"Tokenizer/processor not found in checkpoint. Falling back to {hf_model_name}")
            tokenizer = CLIPTokenizer.from_pretrained(hf_model_name)
            processor = CLIPProcessor.from_pretrained(hf_model_name)
    else:
        logger.info(f"Loading CLIP model from Hugging Face: {hf_model_name}")
        model = CLIPModel.from_pretrained(hf_model_name).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(hf_model_name)
        processor = CLIPProcessor.from_pretrained(hf_model_name)

    embedding_dim = EMBEDDING_DIMS.get(hf_model_name, 512)
    logger.info(f"CLIP loaded. embedding_dim={embedding_dim}")
    return model, processor, tokenizer, embedding_dim, hf_model_name


def freeze_clip_model(clip_model: nn.Module) -> None:
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()
    logger.info("CLIP model frozen")


def encode_text_hf(clip_model, tokenizer, texts, device):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            text_features = clip_model.get_text_features(**inputs)
    return text_features


def compose_query_features(fusion_module, reference_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """
    Forward for fusion module that supports both:
    - Tensor output
    - Tuple output: (features, aux)
    """
    output_dtype = reference_features.dtype

    # Align input dtype with fusion module parameter dtype (important for bf16/fp16 DeepSpeed runs).
    first_param = next(fusion_module.parameters(), None)
    if first_param is not None:
        module_dtype = first_param.dtype
        if reference_features.dtype != module_dtype:
            reference_features = reference_features.to(module_dtype)
        if text_features.dtype != module_dtype:
            text_features = text_features.to(module_dtype)

    out = fusion_module(reference_features, text_features)
    if isinstance(out, tuple):
        out = out[0]

    # Cast back so downstream loss (with CLIP float32 features) keeps dtype-consistent matmul.
    if out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out


def get_fusion_delta_scale(fusion_module: nn.Module) -> Optional[float]:
    delta_gate = getattr(fusion_module, "delta_gate", None)
    if delta_gate is None:
        return None
    return float(torch.tanh(delta_gate.detach()).item())


def extract_clip_index_features(dataset, clip_model, device, batch_size=64, num_workers=4):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(**loader_kwargs)

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


def save_fusion_module(
    model_engine,
    save_path: Path,
    dist_info: dict,
    epoch: int,
    config: dict,
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None,
):
    """Rank-0 only save for fusion module state dict."""
    if not is_main_process(dist_info):
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "fusion_module": model_engine.module.state_dict(),
        "config": config,
    }
    if metric_name is not None and metric_value is not None:
        payload[metric_name] = metric_value
    torch.save(payload, str(save_path))


# =========================================================
# Validation metrics
# =========================================================
def compute_fiq_val_metrics_fusion(relative_val_dataset, clip_model, fusion_module, tokenizer, index_features, index_names, device):
    print(f"Computing FashionIQ {relative_val_dataset.dress_types} validation metrics")
    clip_model.eval()
    fusion_module.eval()
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
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
                batch_predicted = compose_query_features(fusion_module, reference_features, text_features)
                batch_predicted = F.normalize(batch_predicted, dim=-1)
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


def compute_cirr_val_metrics_fusion(relative_val_dataset, clip_model, fusion_module, tokenizer, index_features, index_names, device):
    print("Computing CIRR validation metrics")
    clip_model.eval()
    fusion_module.eval()
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
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
                batch_predicted = compose_query_features(fusion_module, reference_features, text_features)
                batch_predicted = F.normalize(batch_predicted, dim=-1)
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


def create_train_loader(dataset, batch_size, num_workers, shuffle, sampler=None):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
        "drop_last": True,
        "shuffle": shuffle,
        "sampler": sampler,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return DataLoader(**loader_kwargs)


# =========================================================
# Training loops
# =========================================================
def clip_finetune_fusion_fiq_ance(
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
    pretrained_clip_path: Optional[str] = None,
    num_cross_attn_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
    ance_num_negatives: int = 16,
    ance_topk_candidates: int = 100,
    ance_refresh_interval: int = 1,
    ance_weight: float = 1.0,
    ance_warmup_epochs: int = 0,
    experiment_name: str = None,
    # DeepSpeed args
    dist_info: Optional[dict] = None,
    deepspeed_config: str = None,
    grad_accum_steps: int = 1,
    partial_intent_queries_path: str = None,
    **kwargs,
):
    assert dist_info is not None
    device = torch.device(f"cuda:{dist_info['local_rank']}") if torch.cuda.is_available() else torch.device("cpu")
    num_workers = kwargs.get("num_workers", 4)

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_name_clean = clip_model_name.replace("/", "-")
    fusion_suffix = "bicross_attn_pool"
    if experiment_name:
        training_path = Path(
            base_path / f"models/clip_fusion_ance_fiq_{model_name_clean}_{fusion_suffix}_{experiment_name}_{training_start}"
        )
    else:
        training_path = Path(base_path / f"models/clip_fusion_ance_fiq_{model_name_clean}_{fusion_suffix}_{training_start}")

    if is_main_process(dist_info):
        training_path.mkdir(exist_ok=False, parents=True)
    barrier(dist_info)
    training_path = Path(broadcast_object(dist_info, str(training_path)))

    clip_model, _, tokenizer, embedding_dim, hf_model_name = get_clip_model_and_processor(
        clip_model_name, device, pretrained_path=pretrained_clip_path
    )
    freeze_clip_model(clip_model)

    num_aux_tokens = kwargs.get("num_aux_tokens", 4)
    fusion_module = get_fusion_module(
        clip_model_name=hf_model_name,
        num_cross_attn_layers=num_cross_attn_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_aux_tokens=num_aux_tokens,
    ).to(device)

    fusion_trainable_params = sum(p.numel() for p in fusion_module.parameters() if p.requires_grad)
    if is_main_process(dist_info):
        logger.info(f"Trainable fusion params: {fusion_trainable_params:,}")

    if is_main_process(dist_info):
        training_hyper_params = {
            "num_epochs": num_epochs,
            "clip_model_name": clip_model_name,
            "pretrained_clip_path": pretrained_clip_path,
            "fusion_type": "bidirectional_cross_attention",
            "num_cross_attn_layers": num_cross_attn_layers,
            "num_heads": num_heads,
            "num_aux_tokens": num_aux_tokens,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size_per_gpu": batch_size,
            "world_size": dist_info["world_size"],
            "global_batch": batch_size * dist_info["world_size"] * grad_accum_steps,
            "embedding_dim": embedding_dim,
            "trainable_fusion_params": fusion_trainable_params,
            "ance_num_negatives": ance_num_negatives,
            "ance_topk_candidates": ance_topk_candidates,
            "ance_refresh_interval": ance_refresh_interval,
            "ance_weight": ance_weight,
            "ance_warmup_epochs": ance_warmup_epochs,
            "deepspeed_zero_stage": 2,
            "grad_accum_steps": grad_accum_steps,
            "fusion_output_form": "sum_plus_bounded_residual",
        }
        with open(training_path / "training_hyperparameters.json", "w+") as f:
            json.dump(training_hyper_params, f, sort_keys=True, indent=4)
    else:
        training_hyper_params = None
    training_hyper_params = broadcast_object(dist_info, training_hyper_params)

    if is_main_process(dist_info) and save_training:
        tokenizer.save_pretrained(str(training_path / "tokenizer"))

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
    use_ance_init = num_epochs >= ance_warmup_epochs
    classic_train_dataset = FashionIQDataset("train", train_dress_types, "classic", preprocess, preload_images=use_ance_init)

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

    relative_train_loader = create_train_loader(
        relative_train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=train_sampler,
    )

    hard_negative_miner = CLIPHardNegativeMiner(
        embedding_dim=embedding_dim,
        num_negatives=ance_num_negatives,
        topk_candidates=ance_topk_candidates,
        refresh_interval=ance_refresh_interval,
        use_gpu=True,
        cache_dir=str(training_path / "ance_cache"),
    )

    logger.info("Building initial embedding index with frozen CLIP...")
    if use_ance_init:
        hard_negative_miner.build_index(
            clip_model=clip_model,
            dataset=classic_train_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    with open(deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = batch_size
    ds_config["gradient_accumulation_steps"] = grad_accum_steps
    ds_config["train_batch_size"] = batch_size * dist_info["world_size"] * grad_accum_steps
    if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
        ds_config["optimizer"]["params"]["lr"] = learning_rate

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=fusion_module,
        model_parameters=fusion_module.parameters(),
        config=ds_config,
    )

    scheduler = OneCycleLR(
        optimizer.optimizer,
        max_lr=learning_rate,
        pct_start=1.5 / num_epochs,
        div_factor=100.0,
        steps_per_epoch=max(1, len(relative_train_loader) // grad_accum_steps),
        epochs=num_epochs,
    )

    best_avg_recall = 0.0 if save_best else None
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()
    use_ance = 0 >= ance_warmup_epochs

    logger.info("Fusion training loop started for FashionIQ (DeepSpeed ZeRO-2)")
    for epoch in range(num_epochs):
        if dist_info["enabled"]:
            train_sampler.set_epoch(epoch)

        if epoch > 0 and use_ance:
            hard_negative_miner.refresh_index(
                clip_model=clip_model,
                dataset=classic_train_dataset,
                device=device,
                current_epoch=epoch,
                batch_size=batch_size,
                num_workers=num_workers,
            )

        use_ance = epoch >= ance_warmup_epochs
        train_running_results = {"accumulated_train_loss": 0.0, "images_in_epoch": 0}
        train_bar = tqdm(relative_train_loader, ncols=150, disable=not is_main_process(dist_info))

        for step, (reference_images, target_images, captions, target_names) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            flattened_captions = np.array(captions).T.flatten().tolist()
            input_captions = [
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
                for i in range(0, len(flattened_captions), 2)
            ]

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    ref_features = clip_model.get_image_features(pixel_values=reference_images)
                    target_features = clip_model.get_image_features(pixel_values=target_images)
                    text_features = encode_text_hf(clip_model, tokenizer, input_captions, device)
                    ref_features = F.normalize(ref_features, dim=-1)
                    target_features = F.normalize(target_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)

            model_engine.train()
            query_features = compose_query_features(model_engine.module, ref_features, text_features)
            query_features = F.normalize(query_features, dim=-1)

            hard_negative_names = None
            hard_neg_indices = None
            if use_ance:
                with torch.no_grad():
                    hard_neg_indices, hard_negative_names = hard_negative_miner.mine_hard_negatives(
                        query_features=query_features.detach(),
                        positive_names=list(target_names),
                        return_names=True,
                    )
                    _, ref_hard_negative_names = hard_negative_miner.mine_hard_negatives(
                        query_features=ref_features.detach(),
                        positive_names=list(target_names),
                        return_names=True,
                    )

            if use_ance and hard_negative_names is not None:
                num_negatives = len(hard_negative_names[0])
                all_neg_names = [n for batch_names in hard_negative_names for n in batch_names]
                all_neg_images = classic_train_dataset.get_images_batch(all_neg_names).to(device, non_blocking=True)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        all_neg_features = clip_model.get_image_features(pixel_values=all_neg_images)
                        all_neg_features = F.normalize(all_neg_features, dim=-1)
                hard_negative_features = all_neg_features.view(images_in_batch, num_negatives, -1)
                del all_neg_images, all_neg_features

                ref_hard_negative_features = None
                if ref_hard_negative_names is not None:
                    num_ref_negatives = len(ref_hard_negative_names[0])
                    all_ref_names = [n for batch_names in ref_hard_negative_names for n in batch_names]
                    all_ref_images = classic_train_dataset.get_images_batch(all_ref_names).to(device, non_blocking=True)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            all_ref_features = clip_model.get_image_features(pixel_values=all_ref_images)
                            all_ref_features = F.normalize(all_ref_features, dim=-1)
                    ref_hard_negative_features = all_ref_features.view(images_in_batch, num_ref_negatives, -1)
                    del all_ref_images, all_ref_features

                partial_intent_negative_features = None
                if partial_intent_queries is not None and hard_neg_indices is not None:
                    partial_intents = [partial_intent_queries.get(cap, cap) for cap in input_captions]
                    pi_num_neg = kwargs.get("partial_intent_num_negatives", ance_num_negatives)
                    with torch.no_grad():
                        _, partial_intent_neg_names = hard_negative_miner.mine_partial_intent_negatives(
                            partial_intent_texts=partial_intents,
                            ref_features=ref_features.detach(),
                            positive_names=list(target_names),
                            hard_negative_indices=hard_neg_indices,
                            num_negatives=pi_num_neg,
                            clip_model=clip_model,
                            tokenizer=tokenizer,
                        )
                    num_pi_neg = len(partial_intent_neg_names[0])
                    all_pi_names = [n for batch_names in partial_intent_neg_names for n in batch_names]
                    all_pi_images = classic_train_dataset.get_images_batch(all_pi_names).to(device, non_blocking=True)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            all_pi_features = clip_model.get_image_features(pixel_values=all_pi_images)
                            all_pi_features = F.normalize(all_pi_features, dim=-1)
                    partial_intent_negative_features = all_pi_features.view(images_in_batch, num_pi_neg, -1)
                    del all_pi_images, all_pi_features

                loss = compute_clip_ance_loss(
                    query_features=query_features,
                    target_features=target_features,
                    hard_negative_features=hard_negative_features,
                    temperature=0.07,
                    hard_negative_weight=ance_weight,
                    ref_hard_negative_features=ref_hard_negative_features,
                    ref_hard_negative_weight=kwargs.get("ref_ance_weight", 1.0),
                    partial_intent_negative_features=partial_intent_negative_features,
                    partial_intent_negative_weight=kwargs.get("partial_intent_weight", 0.75),
                )
            else:
                sim_matrix = torch.matmul(query_features, target_features.T) / 0.07
                labels = torch.arange(images_in_batch, dtype=torch.long, device=device)
                loss = F.cross_entropy(sim_matrix, labels)

            model_engine.backward(loss)
            model_engine.step()

            if (step + 1) % grad_accum_steps == 0:
                scheduler.step()

            update_train_running_results(train_running_results, loss.detach(), images_in_batch)
            if is_main_process(dist_info):
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

        total_loss = torch.tensor([train_running_results["accumulated_train_loss"]], device=device)
        total_imgs = torch.tensor([train_running_results["images_in_epoch"]], device=device, dtype=torch.long)
        total_loss = all_reduce_sum(dist_info, total_loss)
        total_imgs = all_reduce_sum(dist_info, total_imgs)

        if is_main_process(dist_info):
            avg_loss = total_loss.item() / max(int(total_imgs.item()), 1)
            loss_log_dict = {"epoch": epoch, "loss": avg_loss}
            delta_scale = get_fusion_delta_scale(model_engine.module)
            if delta_scale is not None:
                loss_log_dict["delta_scale"] = delta_scale
            training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
            training_log_frame.to_csv(str(training_path / "train_metrics.csv"), index=False)

        barrier(dist_info)

        if epoch % validation_frequency == 0:
            if is_main_process(dist_info):
                model_engine.eval()
                recalls_at10, recalls_at50 = [], []

                for relative_val_dataset, classic_val_dataset, idx in zip(
                    relative_val_datasets, classic_val_datasets, idx_to_dress_mapping
                ):
                    torch.cuda.empty_cache()
                    index_features, index_names = extract_clip_index_features(
                        classic_val_dataset, clip_model, device=device, num_workers=num_workers
                    )
                    r10, r50 = compute_fiq_val_metrics_fusion(
                        relative_val_dataset,
                        clip_model,
                        model_engine.module,
                        tokenizer,
                        index_features,
                        index_names,
                        device=device,
                    )
                    recalls_at10.append(r10)
                    recalls_at50.append(r50)
                    del index_features, index_names
                    torch.cuda.empty_cache()

                results_dict = {f"{idx_to_dress_mapping[i]}_recall_at10": recalls_at10[i] for i in range(len(recalls_at10))}
                results_dict.update({f"{idx_to_dress_mapping[i]}_recall_at50": recalls_at50[i] for i in range(len(recalls_at50))})
                results_dict.update(
                    {
                        "average_recall_at10": mean(recalls_at10),
                        "average_recall_at50": mean(recalls_at50),
                        "average_recall": (mean(recalls_at50) + mean(recalls_at10)) / 2,
                    }
                )
                print(json.dumps(results_dict, indent=4))

                log_dict = {"epoch": epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / "validation_metrics.csv"), index=False)

                if save_training and save_best and results_dict["average_recall"] > best_avg_recall:
                    best_avg_recall = results_dict["average_recall"]
                    save_fusion_module(
                        model_engine,
                        training_path / "best_fusion_model.pt",
                        dist_info,
                        epoch=epoch,
                        config=training_hyper_params,
                        metric_name="best_average_recall",
                        metric_value=best_avg_recall,
                    )
                    logger.info(f"Saved best fusion model at epoch {epoch}, average_recall={best_avg_recall:.2f}")

            if save_training:
                save_fusion_module(
                    model_engine,
                    training_path / "latest_fusion_model.pt",
                    dist_info,
                    epoch=epoch,
                    config=training_hyper_params,
                )
                if is_main_process(dist_info):
                    logger.info(f"Saved latest fusion model at epoch {epoch}")

            barrier(dist_info)

    if save_training:
        save_fusion_module(
            model_engine,
            training_path / "final_fusion_model.pt",
            dist_info,
            epoch=num_epochs - 1,
            config=training_hyper_params,
        )
        if is_main_process(dist_info):
            logger.info(f"Saved final fusion model to {training_path / 'final_fusion_model.pt'}")
    barrier(dist_info)


def clip_finetune_fusion_cirr_ance(
    num_epochs: int,
    clip_model_name: str,
    learning_rate: float,
    batch_size: int,
    validation_frequency: int,
    transform: str,
    save_training: bool,
    save_best: bool,
    pretrained_clip_path: Optional[str] = None,
    num_cross_attn_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
    ance_num_negatives: int = 16,
    ance_topk_candidates: int = 100,
    ance_refresh_interval: int = 1,
    ance_weight: float = 1.0,
    ance_warmup_epochs: int = 0,
    experiment_name: str = None,
    # DeepSpeed args
    dist_info: Optional[dict] = None,
    deepspeed_config: str = None,
    grad_accum_steps: int = 1,
    partial_intent_queries_path: str = None,
    **kwargs,
):
    assert dist_info is not None
    device = torch.device(f"cuda:{dist_info['local_rank']}") if torch.cuda.is_available() else torch.device("cpu")
    num_workers = kwargs.get("num_workers", 4)

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_name_clean = clip_model_name.replace("/", "-")
    fusion_suffix = "bicross_attn_pool"
    if experiment_name:
        training_path = Path(
            base_path / f"models/clip_fusion_ance_cirr_{model_name_clean}_{fusion_suffix}_{experiment_name}_{training_start}"
        )
    else:
        training_path = Path(base_path / f"models/clip_fusion_ance_cirr_{model_name_clean}_{fusion_suffix}_{training_start}")

    if is_main_process(dist_info):
        training_path.mkdir(exist_ok=False, parents=True)
    barrier(dist_info)
    training_path = Path(broadcast_object(dist_info, str(training_path)))

    clip_model, _, tokenizer, embedding_dim, hf_model_name = get_clip_model_and_processor(
        clip_model_name, device, pretrained_path=pretrained_clip_path
    )
    freeze_clip_model(clip_model)

    num_aux_tokens = kwargs.get("num_aux_tokens", 4)
    fusion_module = get_fusion_module(
        clip_model_name=hf_model_name,
        num_cross_attn_layers=num_cross_attn_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_aux_tokens=num_aux_tokens,
    ).to(device)

    fusion_trainable_params = sum(p.numel() for p in fusion_module.parameters() if p.requires_grad)
    if is_main_process(dist_info):
        logger.info(f"Trainable fusion params: {fusion_trainable_params:,}")

    if is_main_process(dist_info):
        training_hyper_params = {
            "num_epochs": num_epochs,
            "clip_model_name": clip_model_name,
            "pretrained_clip_path": pretrained_clip_path,
            "fusion_type": "bidirectional_cross_attention",
            "num_cross_attn_layers": num_cross_attn_layers,
            "num_heads": num_heads,
            "num_aux_tokens": num_aux_tokens,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size_per_gpu": batch_size,
            "world_size": dist_info["world_size"],
            "global_batch": batch_size * dist_info["world_size"] * grad_accum_steps,
            "embedding_dim": embedding_dim,
            "trainable_fusion_params": fusion_trainable_params,
            "ance_num_negatives": ance_num_negatives,
            "ance_topk_candidates": ance_topk_candidates,
            "ance_refresh_interval": ance_refresh_interval,
            "ance_weight": ance_weight,
            "ance_warmup_epochs": ance_warmup_epochs,
            "deepspeed_zero_stage": 2,
            "grad_accum_steps": grad_accum_steps,
            "fusion_output_form": "sum_plus_bounded_residual",
        }
        with open(training_path / "training_hyperparameters.json", "w+") as f:
            json.dump(training_hyper_params, f, sort_keys=True, indent=4)
    else:
        training_hyper_params = None
    training_hyper_params = broadcast_object(dist_info, training_hyper_params)

    if is_main_process(dist_info) and save_training:
        tokenizer.save_pretrained(str(training_path / "tokenizer"))

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

    use_ance_init = num_epochs >= ance_warmup_epochs
    classic_train_dataset = CIRRDataset("train", "classic", preprocess, preload_images=use_ance_init)

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

    relative_train_loader = create_train_loader(
        relative_train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=train_sampler,
    )

    hard_negative_miner = CLIPHardNegativeMiner(
        embedding_dim=embedding_dim,
        num_negatives=ance_num_negatives,
        topk_candidates=ance_topk_candidates,
        refresh_interval=ance_refresh_interval,
        use_gpu=True,
        cache_dir=str(training_path / "ance_cache"),
    )

    logger.info("Building initial embedding index for CIRR with frozen CLIP...")
    if use_ance_init:
        hard_negative_miner.build_index(
            clip_model=clip_model,
            dataset=classic_train_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    with open(deepspeed_config, "r") as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = batch_size
    ds_config["gradient_accumulation_steps"] = grad_accum_steps
    ds_config["train_batch_size"] = batch_size * dist_info["world_size"] * grad_accum_steps
    if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
        ds_config["optimizer"]["params"]["lr"] = learning_rate

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=fusion_module,
        model_parameters=fusion_module.parameters(),
        config=ds_config,
    )

    scheduler = OneCycleLR(
        optimizer.optimizer,
        max_lr=learning_rate,
        pct_start=1 / 50,
        steps_per_epoch=max(1, len(relative_train_loader) // grad_accum_steps),
        epochs=num_epochs,
    )

    best_arithmetic = 0.0 if save_best else None
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()
    use_ance = 0 >= ance_warmup_epochs

    logger.info("Fusion training loop started for CIRR (DeepSpeed ZeRO-2)")
    for epoch in range(num_epochs):
        if dist_info["enabled"]:
            train_sampler.set_epoch(epoch)

        if epoch > 0 and use_ance:
            hard_negative_miner.refresh_index(
                clip_model=clip_model,
                dataset=classic_train_dataset,
                device=device,
                current_epoch=epoch,
                batch_size=batch_size,
                num_workers=num_workers,
            )

        use_ance = epoch >= ance_warmup_epochs
        train_running_results = {"accumulated_train_loss": 0.0, "images_in_epoch": 0}
        train_bar = tqdm(relative_train_loader, ncols=150, disable=not is_main_process(dist_info))

        for step, (reference_images, target_images, captions, target_names) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            input_captions = list(captions)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    ref_features = clip_model.get_image_features(pixel_values=reference_images)
                    target_features = clip_model.get_image_features(pixel_values=target_images)
                    text_features = encode_text_hf(clip_model, tokenizer, input_captions, device)
                    ref_features = F.normalize(ref_features, dim=-1)
                    target_features = F.normalize(target_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)

            model_engine.train()
            query_features = compose_query_features(model_engine.module, ref_features, text_features)
            query_features = F.normalize(query_features, dim=-1)

            hard_negative_names = None
            hard_neg_indices = None
            if use_ance:
                with torch.no_grad():
                    hard_neg_indices, hard_negative_names = hard_negative_miner.mine_hard_negatives(
                        query_features=query_features.detach(),
                        positive_names=list(target_names),
                        return_names=True,
                    )
                    _, ref_hard_negative_names = hard_negative_miner.mine_hard_negatives(
                        query_features=ref_features.detach(),
                        positive_names=list(target_names),
                        return_names=True,
                    )

            if use_ance and hard_negative_names is not None:
                num_negatives = len(hard_negative_names[0])
                all_neg_names = [n for batch_names in hard_negative_names for n in batch_names]
                all_neg_images = classic_train_dataset.get_images_batch(all_neg_names).to(device, non_blocking=True)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        all_neg_features = clip_model.get_image_features(pixel_values=all_neg_images)
                        all_neg_features = F.normalize(all_neg_features, dim=-1)
                hard_negative_features = all_neg_features.view(images_in_batch, num_negatives, -1)
                del all_neg_images, all_neg_features

                ref_hard_negative_features = None
                if ref_hard_negative_names is not None:
                    num_ref_negatives = len(ref_hard_negative_names[0])
                    all_ref_names = [n for batch_names in ref_hard_negative_names for n in batch_names]
                    all_ref_images = classic_train_dataset.get_images_batch(all_ref_names).to(device, non_blocking=True)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            all_ref_features = clip_model.get_image_features(pixel_values=all_ref_images)
                            all_ref_features = F.normalize(all_ref_features, dim=-1)
                    ref_hard_negative_features = all_ref_features.view(images_in_batch, num_ref_negatives, -1)
                    del all_ref_images, all_ref_features

                partial_intent_negative_features = None
                if partial_intent_queries is not None and hard_neg_indices is not None:
                    partial_intents = [partial_intent_queries.get(cap, cap) for cap in input_captions]
                    pi_num_neg = kwargs.get("partial_intent_num_negatives", ance_num_negatives)
                    with torch.no_grad():
                        _, partial_intent_neg_names = hard_negative_miner.mine_partial_intent_negatives(
                            partial_intent_texts=partial_intents,
                            ref_features=ref_features.detach(),
                            positive_names=list(target_names),
                            hard_negative_indices=hard_neg_indices,
                            num_negatives=pi_num_neg,
                            clip_model=clip_model,
                            tokenizer=tokenizer,
                        )
                    num_pi_neg = len(partial_intent_neg_names[0])
                    all_pi_names = [n for batch_names in partial_intent_neg_names for n in batch_names]
                    all_pi_images = classic_train_dataset.get_images_batch(all_pi_names).to(device, non_blocking=True)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            all_pi_features = clip_model.get_image_features(pixel_values=all_pi_images)
                            all_pi_features = F.normalize(all_pi_features, dim=-1)
                    partial_intent_negative_features = all_pi_features.view(images_in_batch, num_pi_neg, -1)
                    del all_pi_images, all_pi_features

                loss = compute_clip_ance_loss(
                    query_features=query_features,
                    target_features=target_features,
                    hard_negative_features=hard_negative_features,
                    temperature=0.07,
                    hard_negative_weight=ance_weight,
                    ref_hard_negative_features=ref_hard_negative_features,
                    ref_hard_negative_weight=kwargs.get("ref_ance_weight", 1.0),
                    partial_intent_negative_features=partial_intent_negative_features,
                    partial_intent_negative_weight=kwargs.get("partial_intent_weight", 0.75),
                )
            else:
                sim_matrix = torch.matmul(query_features, target_features.T) / 0.07
                labels = torch.arange(images_in_batch, dtype=torch.long, device=device)
                loss = F.cross_entropy(sim_matrix, labels)

            model_engine.backward(loss)
            model_engine.step()

            if (step + 1) % grad_accum_steps == 0:
                scheduler.step()

            update_train_running_results(train_running_results, loss.detach(), images_in_batch)
            if is_main_process(dist_info):
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

        total_loss = torch.tensor([train_running_results["accumulated_train_loss"]], device=device)
        total_imgs = torch.tensor([train_running_results["images_in_epoch"]], device=device, dtype=torch.long)
        total_loss = all_reduce_sum(dist_info, total_loss)
        total_imgs = all_reduce_sum(dist_info, total_imgs)

        if is_main_process(dist_info):
            avg_loss = total_loss.item() / max(int(total_imgs.item()), 1)
            loss_log_dict = {"epoch": epoch, "loss": avg_loss}
            delta_scale = get_fusion_delta_scale(model_engine.module)
            if delta_scale is not None:
                loss_log_dict["delta_scale"] = delta_scale
            training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
            training_log_frame.to_csv(str(training_path / "train_metrics.csv"), index=False)

        barrier(dist_info)

        if epoch % validation_frequency == 0:
            if is_main_process(dist_info):
                model_engine.eval()
                torch.cuda.empty_cache()
                val_index_features, val_index_names = extract_clip_index_features(
                    classic_val_dataset, clip_model, device=device, num_workers=num_workers
                )
                results = compute_cirr_val_metrics_fusion(
                    relative_val_dataset,
                    clip_model,
                    model_engine.module,
                    tokenizer,
                    val_index_features,
                    val_index_names,
                    device=device,
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
                    save_fusion_module(
                        model_engine,
                        training_path / "best_fusion_model.pt",
                        dist_info,
                        epoch=epoch,
                        config=training_hyper_params,
                        metric_name="best_arithmetic_mean",
                        metric_value=best_arithmetic,
                    )
                    logger.info(f"Saved best fusion model at epoch {epoch}, arithmetic_mean={best_arithmetic:.2f}")

            if save_training:
                save_fusion_module(
                    model_engine,
                    training_path / "latest_fusion_model.pt",
                    dist_info,
                    epoch=epoch,
                    config=training_hyper_params,
                )
                if is_main_process(dist_info):
                    logger.info(f"Saved latest fusion model at epoch {epoch}")

            barrier(dist_info)

    if save_training:
        save_fusion_module(
            model_engine,
            training_path / "final_fusion_model.pt",
            dist_info,
            epoch=num_epochs - 1,
            config=training_hyper_params,
        )
        if is_main_process(dist_info):
            logger.info(f"Saved final fusion model to {training_path / 'final_fusion_model.pt'}")
    barrier(dist_info)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--clip-model-name", default="ViT-B/32", type=str)
    parser.add_argument("--pretrained-clip-path", type=str, default=None, help="Path to Stage-1 fine-tuned CLIP")
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--batch-size", default=128, type=int)  # per-GPU
    parser.add_argument("--validation-frequency", default=1, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float)
    parser.add_argument("--transform", default="targetpad", type=str)
    parser.add_argument("--save-training", dest="save_training", action="store_true")
    parser.add_argument("--save-best", dest="save_best", action="store_true")
    parser.add_argument("--experiment-name", type=str, default=None)

    # Fusion module args (Bidirectional Cross-Attention + Attention Pooling)
    parser.add_argument("--num-cross-attn-layers", default=2, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--num-aux-tokens", default=4, type=int, help="Learnable auxiliary tokens per modality")

    # ANCE
    parser.add_argument("--ance-num-negatives", default=16, type=int)
    parser.add_argument("--ance-topk-candidates", default=100, type=int)
    parser.add_argument("--ance-refresh-interval", default=1, type=int)
    parser.add_argument("--ance-weight", default=1.0, type=float)
    parser.add_argument("--ance-warmup-epochs", default=0, type=int)
    parser.add_argument("--ref-ance-weight", default=0.5, type=float)
    parser.add_argument("--partial-intent-num-negatives", default=None, type=int)
    parser.add_argument("--partial-intent-weight", default=0.75, type=float)
    parser.add_argument("--partial-intent-queries-path", type=str, default=None, help="Path to partial intent queries JSON file")

    # DeepSpeed
    parser.add_argument("--deepspeed-config", type=str, default="ds_config_zero2.json", help="Path to DeepSpeed config file")
    parser.add_argument("--grad-accum-steps", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed launcher")

    args = parser.parse_args()

    if args.dataset.lower() not in ["fashioniq", "cirr"]:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ'")

    dist_info = init_distributed()
    set_seed(41)

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "clip_model_name": args.clip_model_name,
        "pretrained_clip_path": args.pretrained_clip_path,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "experiment_name": args.experiment_name,
        # Fusion params
        "num_cross_attn_layers": args.num_cross_attn_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "num_aux_tokens": args.num_aux_tokens,
        # ANCE
        "ance_num_negatives": args.ance_num_negatives,
        "ance_topk_candidates": args.ance_topk_candidates,
        "ance_refresh_interval": args.ance_refresh_interval,
        "ance_weight": args.ance_weight,
        "ance_warmup_epochs": args.ance_warmup_epochs,
        "ref_ance_weight": args.ref_ance_weight,
        "partial_intent_num_negatives": args.partial_intent_num_negatives if args.partial_intent_num_negatives else args.ance_num_negatives,
        "partial_intent_weight": args.partial_intent_weight,
        "partial_intent_queries_path": args.partial_intent_queries_path,
        # DeepSpeed
        "dist_info": dist_info,
        "deepspeed_config": args.deepspeed_config,
        "grad_accum_steps": args.grad_accum_steps,
    }

    if args.dataset.lower() == "cirr":
        clip_finetune_fusion_cirr_ance(**training_hyper_params)
    else:
        training_hyper_params.update(
            {
                "train_dress_types": ["dress", "toptee", "shirt"],
                "val_dress_types": ["dress", "toptee", "shirt"],
            }
        )
        clip_finetune_fusion_fiq_ance(**training_hyper_params)
