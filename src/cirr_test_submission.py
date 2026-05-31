import json
from argparse import ArgumentParser
from contextlib import nullcontext
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from cir_fusion import compose_query_features, load_cir_fusion
from clip_ance_utils import get_model_logit_scale
from data_utils import CIRRDataset, base_path, targetpad_transform
from utils import collate_fn, device


DEFAULT_CHECKPOINT_NAME = "best_model"
FALLBACK_CHECKPOINT_NAMES = ("best_model", "latest_model", "final_model")


def autocast_if_available():
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def resolve_local_snapshot_dir(repo_id: str) -> Path:
    try:
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not find a complete local Hugging Face snapshot for '{repo_id}'. "
            "Please confirm the base model is already present in the local Hugging Face cache."
        ) from exc

    snapshot_path = Path(snapshot_dir)
    required_candidates = (
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "model.ckpt.index",
        "flax_model.msgpack",
    )
    if not any((snapshot_path / candidate).exists() for candidate in required_candidates):
        raise RuntimeError(
            f"Found snapshot for '{repo_id}' at '{snapshot_path}', but no standard model weight file was present."
        )

    return snapshot_path


def resolve_pretrained_source(pretrained_name_or_path: str) -> Path:
    direct_path = Path(pretrained_name_or_path).expanduser()
    if direct_path.exists():
        return direct_path.resolve()

    if "/" in pretrained_name_or_path:
        snapshot_path = resolve_local_snapshot_dir(pretrained_name_or_path)
        print(f"Resolved local snapshot for {pretrained_name_or_path}: {snapshot_path}")
        return snapshot_path

    raise FileNotFoundError(f"Local model path does not exist: {direct_path}")


def load_hf_component(component_cls, pretrained_name_or_path: str, **kwargs):
    resolved_source = resolve_pretrained_source(pretrained_name_or_path)
    try:
        return component_cls.from_pretrained(
            str(resolved_source),
            local_files_only=True,
            **kwargs,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not load '{pretrained_name_or_path}' from local source '{resolved_source}'. "
            "The local cache entry may be incomplete or incompatible."
        ) from exc


def resolve_model_dir(model_path: str, checkpoint_name: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Model path must be a directory: {path}")

    if (path / "adapter_config.json").exists() or (path / "config.json").exists():
        return path

    explicit_checkpoint_dir = path / checkpoint_name
    if explicit_checkpoint_dir.is_dir():
        return explicit_checkpoint_dir

    for candidate_name in FALLBACK_CHECKPOINT_NAMES:
        candidate_dir = path / candidate_name
        if candidate_dir.is_dir():
            print(
                f"Checkpoint directory '{checkpoint_name}' not found. "
                f"Automatically using '{candidate_name}'."
            )
            return candidate_dir

    raise FileNotFoundError(
        "Could not find a checkpoint directory. "
        f"Expected one of: {', '.join(FALLBACK_CHECKPOINT_NAMES)} under {path}"
    )


def infer_input_dim(model_dir: Path) -> int:
    preprocessor_path = model_dir / "preprocessor_config.json"
    if not preprocessor_path.exists():
        return 224

    with open(preprocessor_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    crop_size = config.get("crop_size")
    if isinstance(crop_size, dict):
        if "height" in crop_size:
            return int(crop_size["height"])
        if "width" in crop_size:
            return int(crop_size["width"])

    size = config.get("size")
    if isinstance(size, dict):
        if "shortest_edge" in size:
            return int(size["shortest_edge"])
        if "height" in size:
            return int(size["height"])
        if "width" in size:
            return int(size["width"])

    return 224


def load_saved_logit_scale(model, model_dir: Path) -> bool:
    logit_scale_path = model_dir / "logit_scale.pt"
    if not logit_scale_path.exists():
        return False

    state = torch.load(logit_scale_path, map_location="cpu")
    saved_logit_scale = state.get("logit_scale")
    model_logit_scale = get_model_logit_scale(model)

    if saved_logit_scale is None or model_logit_scale is None:
        return False

    with torch.no_grad():
        model_logit_scale.copy_(
            saved_logit_scale.to(device=model_logit_scale.device, dtype=model_logit_scale.dtype)
        )
    return True


def load_trained_clip_model(model_dir: Path):
    adapter_config_path = model_dir / "adapter_config.json"

    if adapter_config_path.exists():
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model_name = peft_config.base_model_name_or_path
        print(f"Loading base CLIP model from local Hugging Face cache: {base_model_name}")
        model = load_hf_component(CLIPModel, base_model_name)
        model = PeftModel.from_pretrained(model, str(model_dir))
    else:
        print(f"Loading CLIP model from local directory: {model_dir}")
        model = CLIPModel.from_pretrained(str(model_dir))

    try:
        tokenizer = CLIPTokenizer.from_pretrained(str(model_dir))
    except OSError:
        if not adapter_config_path.exists():
            raise
        tokenizer = load_hf_component(
            CLIPTokenizer,
            peft_config.base_model_name_or_path,
        )

    load_cir_fusion(model, model_dir)
    load_saved_logit_scale(model, model_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def extract_clip_index_features(
    dataset: CIRRDataset,
    clip_model,
    batch_size: int,
) -> Tuple[torch.Tensor, List[str]]:
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    all_features = []
    all_names = []

    print(f"Extracting CIRR {dataset.split} index features")
    for names, images in tqdm(dataloader):
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            with autocast_if_available():
                image_features = clip_model.get_image_features(pixel_values=images)
                image_features = F.normalize(image_features, dim=-1)
            all_features.append(image_features.cpu())
            all_names.extend(names)

    return torch.vstack(all_features).to(device), all_names


def encode_text(clip_model, tokenizer, captions: List[str]) -> torch.Tensor:
    tokenized = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        with autocast_if_available():
            text_features = clip_model.get_text_features(**tokenized)
    return text_features


def generate_cirr_test_predictions(
    clip_model,
    tokenizer,
    relative_test_dataset: CIRRDataset,
    index_names: List[str],
    index_features: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, List[str], List[List[str]], List[str]]:
    print("Computing CIRR test predictions")

    relative_test_loader = DataLoader(
        dataset=relative_test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        shuffle=False,
    )

    name_to_feat = dict(zip(index_names, index_features))
    pairs_id = []
    group_members = []
    reference_names = []
    similarities = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(relative_test_loader):
        batch_group_members = np.array(batch_group_members).T.tolist()

        with torch.no_grad():
            with autocast_if_available():
                if len(captions) == 1:
                    reference_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
                else:
                    reference_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))

                text_features = encode_text(clip_model, tokenizer, list(captions))
                reference_features = F.normalize(reference_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                query_features = compose_query_features(clip_model, reference_features, text_features)
                batch_similarities = query_features @ index_features.T

            similarities.append(batch_similarities.cpu())

        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    return torch.vstack(similarities), reference_names, group_members, pairs_id


def generate_cirr_test_dicts(
    relative_test_dataset: CIRRDataset,
    clip_model,
    tokenizer,
    index_features: torch.Tensor,
    index_names: List[str],
    batch_size: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    predicted_similarities, reference_names, group_members, pairs_id = generate_cirr_test_predictions(
        clip_model=clip_model,
        tokenizer=tokenizer,
        relative_test_dataset=relative_test_dataset,
        index_names=index_names,
        index_features=index_features,
        batch_size=batch_size,
    )

    print("Computing CIRR prediction dicts")
    distances = 1 - predicted_similarities
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(
            len(sorted_index_names), -1
        )
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0], sorted_index_names.shape[1] - 1
    )

    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    pairid_to_predictions = {
        str(int(pair_id)): prediction[:50].tolist()
        for pair_id, prediction in zip(pairs_id, sorted_index_names)
    }
    pairid_to_group_predictions = {
        str(int(pair_id)): prediction[:3].tolist()
        for pair_id, prediction in zip(pairs_id, sorted_group_names)
    }

    return pairid_to_predictions, pairid_to_group_predictions


def build_submission_name(model_dir: Path, explicit_submission_name: str = None) -> str:
    if explicit_submission_name:
        return explicit_submission_name

    if model_dir.name in FALLBACK_CHECKPOINT_NAMES:
        return f"{model_dir.parent.name}_{model_dir.name}"
    return model_dir.name


def generate_cirr_test_submissions(
    submission_name: str,
    clip_model,
    tokenizer,
    preprocess,
    batch_size: int,
) -> None:
    classic_test_dataset = CIRRDataset("test1", "classic", preprocess)
    index_features, index_names = extract_clip_index_features(
        classic_test_dataset,
        clip_model,
        batch_size=batch_size,
    )
    relative_test_dataset = CIRRDataset("test1", "relative", preprocess)

    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(
        relative_test_dataset=relative_test_dataset,
        clip_model=clip_model,
        tokenizer=tokenizer,
        index_features=index_features,
        index_names=index_names,
        batch_size=batch_size,
    )

    submission = {
        "version": "rc2",
        "metric": "recall",
    }
    group_submission = {
        "version": "rc2",
        "metric": "recall_subset",
    }
    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    submissions_folder_path = base_path / "submission" / "CIRR"
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions to {submissions_folder_path}")
    with open(submissions_folder_path / f"recall_submission_{submission_name}.json", "w", encoding="utf-8") as file:
        json.dump(submission, file, sort_keys=True)

    with open(
        submissions_folder_path / f"recall_subset_submission_{submission_name}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(group_submission, file, sort_keys=True)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the training directory or directly to best_model/latest_model/final_model.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=DEFAULT_CHECKPOINT_NAME,
        help="Checkpoint subdirectory to use when --model-path points to the training directory.",
    )
    parser.add_argument(
        "--submission-name",
        type=str,
        default=None,
        help="Output submission name. Defaults to an automatic name derived from the checkpoint directory.",
    )
    parser.add_argument("--target-ratio", type=float, default=1.25)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_path, args.checkpoint_name)
    submission_name = build_submission_name(model_dir, args.submission_name)
    input_dim = infer_input_dim(model_dir)

    print(f"Using device: {device}")
    print(f"Resolved checkpoint directory: {model_dir}")
    print(f"Submission name: {submission_name}")
    print(f"Input image size: {input_dim}")

    clip_model, tokenizer = load_trained_clip_model(model_dir)
    preprocess = targetpad_transform(args.target_ratio, input_dim)

    generate_cirr_test_submissions(
        submission_name=submission_name,
        clip_model=clip_model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
