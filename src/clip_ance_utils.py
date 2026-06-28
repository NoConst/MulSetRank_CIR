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


def _get_clip_text_components(clip_model):
    for candidate in _iter_wrapped_models(clip_model):
        text_model = getattr(candidate, "text_model", None)
        text_projection = getattr(candidate, "text_projection", None)
        if text_model is not None and text_projection is not None:
            return text_model, text_projection

    raise AttributeError(
        "Unsupported CLIP model wrapper: expected `text_model` and `text_projection` "
        "on the model or one of its children."
    )


def extract_clip_text_token_features(clip_model, **tokenized_inputs) -> torch.Tensor:
    """Return projected CLIP text token embeddings as [B, L, D]."""
    text_model, text_projection = _get_clip_text_components(clip_model)
    tokenized_inputs = {
        key: value for key, value in tokenized_inputs.items() if key != "offset_mapping"
    }

    try:
        text_outputs = text_model(**tokenized_inputs, return_dict=True)
    except TypeError:
        text_outputs = text_model(**tokenized_inputs)

    last_hidden_state = getattr(text_outputs, "last_hidden_state", None)
    if last_hidden_state is None:
        if isinstance(text_outputs, (tuple, list)) and len(text_outputs) > 0:
            last_hidden_state = text_outputs[0]
        else:
            raise TypeError("CLIP text model output does not contain token hidden states.")

    token_features = text_projection(last_hidden_state)
    if token_features.dim() != 3:
        raise ValueError(
            "text_token_features must have shape [B, L, D], "
            f"got {tuple(token_features.shape)}"
        )
    return token_features


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
    return_offsets: bool = False,
):
    tokenize_kwargs = {
        "padding": True,
        "truncation": True,
        "max_length": 77,
        "return_tensors": "pt",
    }

    offset_mappings = None
    if return_offsets:
        try:
            tokenized_inputs = tokenizer(
                texts,
                return_offsets_mapping=True,
                **tokenize_kwargs,
            )
            offset_mappings = tokenized_inputs.pop("offset_mapping", None)
        except (TypeError, NotImplementedError, ValueError):
            tokenized_inputs = tokenizer(texts, **tokenize_kwargs)
    else:
        tokenized_inputs = tokenizer(texts, **tokenize_kwargs)

    if isinstance(offset_mappings, torch.Tensor):
        offset_mappings = offset_mappings.cpu()

    return _move_tokenized_inputs_to_device(tokenized_inputs, device), offset_mappings


def _normalize_with_char_map(text: str) -> Tuple[str, List[int]]:
    normalized_chars: List[str] = []
    char_map: List[int] = []
    previous_was_space = False

    for char_index, char in enumerate(text):
        if char.isspace():
            if not previous_was_space:
                normalized_chars.append(" ")
                char_map.append(char_index)
                previous_was_space = True
            continue

        normalized_chars.append(char.lower())
        char_map.append(char_index)
        previous_was_space = False

    start = 0
    while start < len(normalized_chars) and normalized_chars[start] == " ":
        start += 1
    end = len(normalized_chars)
    while end > start and normalized_chars[end - 1] == " ":
        end -= 1

    return "".join(normalized_chars[start:end]), char_map[start:end]


def _find_all_substrings(text: str, pattern: str) -> List[int]:
    if not pattern:
        return []

    matches = []
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos < 0:
            break
        matches.append(pos)
        start = pos + 1
    return matches


def _find_unique_normalized_char_span(full_text: str, intent_text: str) -> Optional[Tuple[int, int]]:
    strip_chars = " \t\r\n.?,;:!\"'`"
    candidates = []
    for candidate in (intent_text.strip(), intent_text.strip(strip_chars)):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        raw_matches = _find_all_substrings(full_text, candidate)
        if len(raw_matches) == 1:
            pos = raw_matches[0]
            return pos, pos + len(candidate)
        if len(raw_matches) > 1:
            return None

        lower_matches = _find_all_substrings(full_text.lower(), candidate.lower())
        if len(lower_matches) == 1:
            pos = lower_matches[0]
            return pos, pos + len(candidate)
        if len(lower_matches) > 1:
            return None

        full_norm, full_map = _normalize_with_char_map(full_text)
        intent_norm, _ = _normalize_with_char_map(candidate)
        norm_matches = _find_all_substrings(full_norm, intent_norm)
        if len(norm_matches) == 1:
            norm_start = norm_matches[0]
            norm_end = norm_start + len(intent_norm) - 1
            return full_map[norm_start], full_map[norm_end] + 1
        if len(norm_matches) > 1:
            return None

    return None


def _as_1d_int_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], (tuple, list)):
        value = value[0]
    return [int(item) for item in value]


def _get_special_token_ids(tokenizer) -> set:
    special_ids = set()
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        if token_id is not None:
            special_ids.add(int(token_id))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    return special_ids


def _strip_special_token_ids(
    input_ids,
    attention_mask,
    special_token_ids: set,
) -> Tuple[List[int], List[int]]:
    ids = _as_1d_int_list(input_ids)
    mask = _as_1d_int_list(attention_mask)
    if not mask:
        mask = [1] * len(ids)

    kept_ids: List[int] = []
    kept_positions: List[int] = []
    for pos, token_id in enumerate(ids):
        if pos >= len(mask) or mask[pos] == 0:
            continue
        if token_id in special_token_ids:
            continue
        kept_ids.append(token_id)
        kept_positions.append(pos)
    return kept_ids, kept_positions


def _find_unique_token_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if len(needle) == 0 or len(needle) > len(haystack):
        return None

    matches = []
    max_start = len(haystack) - len(needle)
    for start in range(max_start + 1):
        if haystack[start:start + len(needle)] == needle:
            matches.append(start)
            if len(matches) > 1:
                return None
    return matches[0] if len(matches) == 1 else None


def _get_offsets_for_sample(offset_mappings, sample_index: int):
    if offset_mappings is None:
        return None
    if isinstance(offset_mappings, torch.Tensor):
        return offset_mappings[sample_index].detach().cpu().tolist()
    return offset_mappings[sample_index]


def _token_indices_from_offsets(
    offset_mappings,
    sample_index: int,
    char_span: Tuple[int, int],
    input_ids,
    attention_mask,
    special_token_ids: set,
) -> Optional[List[int]]:
    offsets = _get_offsets_for_sample(offset_mappings, sample_index)
    if offsets is None:
        return None

    char_start, char_end = char_span
    ids = _as_1d_int_list(input_ids)
    mask = _as_1d_int_list(attention_mask)
    if not mask:
        mask = [1] * len(ids)

    token_indices: List[int] = []
    for token_index, token_offsets in enumerate(offsets):
        if token_index >= len(ids) or token_index >= len(mask) or mask[token_index] == 0:
            continue
        if ids[token_index] in special_token_ids:
            continue
        token_start, token_end = int(token_offsets[0]), int(token_offsets[1])
        if token_end <= token_start:
            continue
        if token_start < char_end and token_end > char_start:
            token_indices.append(token_index)

    if len(token_indices) == 0:
        return None
    if token_indices[-1] - token_indices[0] + 1 != len(token_indices):
        return None
    return token_indices


def _token_indices_from_token_ids(
    tokenizer,
    full_input_ids,
    full_attention_mask,
    intent_text: str,
    special_token_ids: set,
) -> Optional[List[int]]:
    full_ids, full_positions = _strip_special_token_ids(
        full_input_ids,
        full_attention_mask,
        special_token_ids,
    )

    intent_inputs = tokenizer(
        intent_text,
        padding=False,
        truncation=True,
        max_length=77,
    )
    intent_ids, _ = _strip_special_token_ids(
        intent_inputs.get("input_ids"),
        intent_inputs.get("attention_mask"),
        special_token_ids,
    )
    match_start = _find_unique_token_subsequence(full_ids, intent_ids)
    if match_start is None:
        return None

    token_indices = full_positions[match_start:match_start + len(intent_ids)]
    if len(token_indices) == 0:
        return None
    if token_indices[-1] - token_indices[0] + 1 != len(token_indices):
        return None
    return token_indices


def _find_intent_token_indices(
    tokenizer,
    full_text: str,
    intent_text: str,
    tokenized_full_inputs,
    offset_mappings,
    sample_index: int,
    special_token_ids: set,
) -> Optional[List[int]]:
    char_span = _find_unique_normalized_char_span(full_text, intent_text)
    if char_span is None:
        return None

    full_input_ids = tokenized_full_inputs["input_ids"][sample_index]
    full_attention_mask = tokenized_full_inputs.get("attention_mask")
    if full_attention_mask is not None:
        full_attention_mask = full_attention_mask[sample_index]

    if offset_mappings is not None:
        return _token_indices_from_offsets(
            offset_mappings=offset_mappings,
            sample_index=sample_index,
            char_span=char_span,
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            special_token_ids=special_token_ids,
        )

    return _token_indices_from_token_ids(
        tokenizer=tokenizer,
        full_input_ids=full_input_ids,
        full_attention_mask=full_attention_mask,
        intent_text=intent_text,
        special_token_ids=special_token_ids,
    )


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
    consistency_mode: str = "counterfactual_delta",
    return_stats: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Intent Consistency Learning based on the cross-attention fusion head.

    The isolated intent representation is composed from the reference image and
    a single extracted intent. Depending on the mode, the contextual intent is
    either a counterfactual marginal query delta or a full-text token-span
    representation.

    The counterfactual_delta mode defines the contextual intent representation
    as the marginal effect of that intent on the final retrieval query:
    full_query - query_without_that_intent. It is aligned with the isolated
    single-intent effect: single_intent_query - reference_image_query.

    The direction mode keeps the intent-specific edit direction consistent:
    contextual_intent - reference_image should match isolated_intent -
    reference_image. This keeps the loss as a same-intent consistency objective
    while reducing domination from the shared reference image component. The
    feature mode preserves the previous raw cosine hinge. The contrastive mode
    is kept as an opt-in experiment. In counterfactual_delta mode, the optional
    global term uses the full composed query as a detached teacher for a
    relevance-weighted reconstruction from isolated single-intent deltas.

    Active only for cross-attention fusion modules; other fusion types return 0.
    """
    def _bidirectional_contrastive_loss(
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if anchor_features.shape[0] != positive_features.shape[0]:
            raise ValueError(
                "Contrastive consistency expects paired tensors with the same "
                f"batch size, got {anchor_features.shape[0]} and {positive_features.shape[0]}"
            )
        if anchor_features.shape[0] <= 1:
            cos_sim = F.cosine_similarity(anchor_features, positive_features, dim=-1)
            return F.relu((1.0 - cos_sim) - consistency_epsilon).mean()

        anchor_norm = F.normalize(anchor_features.float(), dim=-1)
        positive_norm = F.normalize(positive_features.float(), dim=-1)
        logits = torch.matmul(anchor_norm, positive_norm.T) / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

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
    consistency_mode = str(consistency_mode).strip().lower()
    if consistency_mode == "counterfactual":
        consistency_mode = "counterfactual_delta"
    if consistency_mode not in {"counterfactual_delta", "direction", "feature", "contrastive", "hybrid"}:
        raise ValueError(
            "Unsupported intent consistency mode: "
            f"{consistency_mode!r}. Expected one of: counterfactual_delta, "
            "direction, feature, contrastive, hybrid."
        )

    intent_inputs, _ = _tokenize_texts_for_intent_learning(
        tokenizer=tokenizer,
        texts=flat_intents,
        device=device,
        return_offsets=False,
    )

    full_texts = [input_captions[idx] for idx in multi_indices]
    if consistency_mode == "counterfactual_delta":
        without_intent_texts: List[str] = []
        for intents in multi_intent_texts:
            for intent_position in range(len(intents)):
                without_intent_texts.append(_join_remaining_intents(intents, intent_position))
        without_inputs, _ = _tokenize_texts_for_intent_learning(
            tokenizer=tokenizer,
            texts=without_intent_texts,
            device=device,
            return_offsets=False,
        )
    else:
        full_inputs, full_offset_mappings = _tokenize_texts_for_intent_learning(
            tokenizer=tokenizer,
            texts=full_texts,
            device=device,
            return_offsets=True,
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
        if consistency_mode in {"counterfactual_delta", "direction"}:
            with torch.no_grad():
                ref_anchor_features = compose_image_features(model, selected_ref_features)

        if consistency_mode == "counterfactual_delta":
            with torch.no_grad():
                without_text_features = extract_clip_text_features(model, **without_inputs)
                without_text_features = F.normalize(without_text_features, dim=-1)
                q_without_all = fusion_module(repeated_ref_features, without_text_features)
        else:
            full_text_features = extract_clip_text_features(model, **full_inputs)
            full_text_features = F.normalize(full_text_features, dim=-1)
            full_token_features = extract_clip_text_token_features(model, **full_inputs)
            full_token_features = F.normalize(full_token_features, dim=-1)
            _, full_fused_token_features = fusion_module(
                selected_ref_features,
                full_text_features,
                text_token_features=full_token_features,
                text_attention_mask=full_inputs.get("attention_mask"),
                return_token_features=True,
            )

    special_token_ids = _get_special_token_ids(tokenizer)

    feature_local_losses = []
    feature_global_losses = []
    contrastive_ctx_features = []
    contrastive_iso_features = []
    global_student_features = []
    global_teacher_features = []
    skipped_span_mismatch = 0
    valid_span_count = 0
    valid_query_count = 0
    offset = 0
    for j, idx in enumerate(multi_indices):
        num_intents = len(multi_intent_texts[j])
        end = offset + num_intents

        e_iso_sample = e_iso_all[offset:end]
        e_iso = e_iso_sample

        if consistency_mode == "counterfactual_delta":
            q_without = q_without_all[offset:end]
            q_full = query_features[idx].unsqueeze(0).expand_as(q_without)
            ref_anchor = ref_anchor_features[j].unsqueeze(0)
            ctx_delta = q_full - q_without
            iso_delta = e_iso - ref_anchor
            local_ctx = F.normalize(ctx_delta, dim=-1)
            local_iso = F.normalize(iso_delta, dim=-1)
            valid_span_count += e_iso.shape[0]
        else:
            e_ctx_list = []
            e_iso_list = []
            for intent_position, intent_text in enumerate(multi_intent_texts[j]):
                token_indices = _find_intent_token_indices(
                    tokenizer=tokenizer,
                    full_text=full_texts[j],
                    intent_text=intent_text,
                    tokenized_full_inputs=full_inputs,
                    offset_mappings=full_offset_mappings,
                    sample_index=j,
                    special_token_ids=special_token_ids,
                )
                if token_indices is None:
                    skipped_span_mismatch += 1
                    continue

                token_index_tensor = torch.tensor(
                    token_indices,
                    device=full_fused_token_features.device,
                    dtype=torch.long,
                )
                e_ctx = full_fused_token_features[j].index_select(0, token_index_tensor).mean(dim=0)
                e_ctx_list.append(e_ctx)
                e_iso_list.append(e_iso_sample[intent_position])

            if len(e_ctx_list) == 0:
                offset = end
                continue

            e_ctx = torch.stack(e_ctx_list, dim=0)
            e_iso = torch.stack(e_iso_list, dim=0)
            if e_iso.shape != e_ctx.shape:
                raise ValueError(
                    "Intent Consistency Learning shape mismatch: "
                    f"e_iso.shape={tuple(e_iso.shape)}, e_ctx.shape={tuple(e_ctx.shape)}"
                )
            valid_span_count += e_ctx.shape[0]

            if consistency_mode == "direction":
                ref_anchor = ref_anchor_features[j].detach().unsqueeze(0)
                local_ctx = F.normalize(e_ctx - ref_anchor, dim=-1)
                local_iso = F.normalize(e_iso - ref_anchor, dim=-1)
            else:
                local_ctx = e_ctx
                local_iso = e_iso

        valid_query_count += 1

        cos_sim_local = F.cosine_similarity(local_ctx, local_iso, dim=-1)
        feature_loss_local = F.relu((1.0 - cos_sim_local) - consistency_epsilon).mean()
        feature_local_losses.append(feature_loss_local)
        contrastive_ctx_features.append(local_ctx)
        contrastive_iso_features.append(local_iso)

        if global_consistency_weight > 0.0:
            full_query_teacher = F.normalize(query_features[idx].detach(), dim=-1)
            if consistency_mode == "counterfactual_delta":
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
            else:
                with torch.no_grad():
                    e_ctx_norm = F.normalize(e_ctx, dim=-1)
                    intent_sims = (e_ctx_norm * full_query_teacher.unsqueeze(0)).sum(dim=-1)
                    intent_weights = F.softmax(
                        intent_sims / global_consistency_temperature,
                        dim=0,
                    )

                weighted_features = (intent_weights.unsqueeze(-1) * e_ctx).sum(dim=0)
                if consistency_mode == "direction":
                    ref_anchor = ref_anchor_features[j].detach()
                    weighted_features_for_loss = F.normalize(weighted_features - ref_anchor, dim=-1)
                    full_query_teacher_for_loss = F.normalize(full_query_teacher - ref_anchor, dim=-1)
                else:
                    weighted_features_for_loss = F.normalize(weighted_features.unsqueeze(0), dim=-1).squeeze(0)
                    full_query_teacher_for_loss = full_query_teacher

            cos_sim_global = (
                full_query_teacher_for_loss * weighted_features_for_loss
            ).sum().clamp(min=-1.0, max=1.0)
            feature_loss_global = F.relu((1.0 - cos_sim_global) - consistency_epsilon)
            feature_global_losses.append(feature_loss_global)
            global_student_features.append(weighted_features_for_loss)
            global_teacher_features.append(full_query_teacher_for_loss)
        offset = end

    if skipped_span_mismatch > 0:
        logger.debug(
            "Intent Consistency Learning skipped %d intent spans due to span mismatch.",
            skipped_span_mismatch,
        )
    if len(feature_local_losses) == 0 or valid_span_count == 0:
        return _empty_result()

    feature_loss_local = torch.stack(feature_local_losses).mean()
    zero = torch.zeros((), device=device)
    feature_loss_global = (
        torch.stack(feature_global_losses).mean()
        if feature_global_losses
        else zero
    )

    if consistency_mode in {"counterfactual_delta", "direction", "feature"}:
        loss_local = feature_loss_local
        loss_global = feature_loss_global
    else:
        all_ctx = torch.cat(contrastive_ctx_features, dim=0)
        all_iso = torch.cat(contrastive_iso_features, dim=0)
        contrastive_loss_local = _bidirectional_contrastive_loss(
            all_ctx,
            all_iso,
            global_consistency_temperature,
        )

        if global_consistency_weight > 0.0 and global_student_features:
            students = torch.stack(global_student_features, dim=0)
            teachers = torch.stack(global_teacher_features, dim=0)
            contrastive_loss_global = _bidirectional_contrastive_loss(
                students,
                teachers.detach(),
                global_consistency_temperature,
            )
        else:
            contrastive_loss_global = zero

        if consistency_mode == "hybrid":
            loss_local = 0.5 * (feature_loss_local + contrastive_loss_local)
            loss_global = 0.5 * (feature_loss_global + contrastive_loss_global)
        else:
            loss_local = contrastive_loss_local
            loss_global = contrastive_loss_global

    loss_ic = loss_local + global_consistency_weight * loss_global
    if not return_stats:
        return loss_ic

    stats = {
        "loss_ic": loss_ic.detach(),
        "loss_ic_local": loss_local.detach(),
        "loss_ic_global": loss_global.detach(),
        "ic_valid_spans": torch.tensor(float(valid_span_count), device=device),
        "ic_skipped_spans": torch.tensor(float(skipped_span_mismatch), device=device),
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
    retrieval-space edit directions produced by the fusion head for each single
    intent to be mutually orthogonal:

        <f(I, A_i) - f(I, blank), f(I, A_j) - f(I, blank)> ~= 0    (i != j)

    This encourages intent-specific changes, not the shared reference-image
    component, to be disentangled. The
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

        selected_ref_features = ref_features[multi_indices]
        with torch.no_grad():
            ref_anchor_features = compose_image_features(model, selected_ref_features)

    losses = []
    offset = 0
    for j in range(len(multi_indices)):
        num_intents = len(multi_intent_texts[j])
        end = offset + num_intents

        # Match the consistency loss isolated side: q_i - q_ref.
        ref_anchor = ref_anchor_features[j].unsqueeze(0)
        single_query_deltas = single_query_features[offset:end] - ref_anchor  # [k, D]
        losses.append(_compute_soft_orthogonality_loss(single_query_deltas))

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
