#!/usr/bin/env python3
"""
Generate partial-intent queries for CIR datasets using a multimodal API.

For each training query, takes the reference image + full modification text and
produces a shorter "partial-intent" query that captures only part of the
original intent.

Usage:
    # FashionIQ
    python src/generate_partial_intent_queries.py \
        --dataset fashioniq \
        --dress_types dress shirt toptee \
        --max_workers 8 \
        --output_dir outputs/fiq_partial_intent_queries

    # CIRR
    python src/generate_partial_intent_queries.py \
        --dataset cirr \
        --max_workers 8 \
        --output_dir outputs/cirr_partial_intent_queries

    # Fashion200k
    python src/generate_partial_intent_queries.py \
        --dataset fashion200k \
        --max_workers 8 \
        --output_dir outputs/fashion200k_partial_intent_queries_standard

    # Shoes
    python src/generate_partial_intent_queries.py \
        --dataset shoes \
        --max_workers 8 \
        --output_dir outputs/shoes_partial_intent_queries_standard

    # LASCO
    python src/generate_partial_intent_queries.py \
        --dataset lasco \
        --max_workers 8 \
        --output_dir outputs/lasco_partial_intent_queries
"""

import json
import os
import base64
import time
import argparse
import mimetypes
import random
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict
from typing import Optional

import numpy as np
from tqdm import tqdm
from openai import OpenAI

def resolve_dataset_path() -> Path:
    candidates = [
        os.environ.get("MULSETRANK_DATASETS_DIR"),
        "/workspace/CAM-CIR_backup/datasets",
        "/root/siton-data-92a7d2fc7b594215b07e48fd8818598b/CAM-CIR_backup/datasets",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return Path(candidates[-1]).expanduser()


DATASET_PATH = resolve_dataset_path()
FIQ_PATH = DATASET_PATH / "fashionIQ_dataset"
CIRR_PATH = DATASET_PATH / "cirr_dataset" / "CIRR"
FASHION200K_PATH = DATASET_PATH / "fashion200k"
SHOES_PATH = DATASET_PATH / "shoes_dataset" / "shoes_data"
LASCO_PATH = DATASET_PATH / "lasco_dataset"
FASHION200K_STANDARD_SEED = 0

SYSTEM_PROMPT_FIQ = """\
You are an expert in fashion image retrieval.

## Task
Given a reference fashion image and a full modification query describing ALL desired changes, \
generate a **partial-intent query** — a shorter text that captures only part of the modification intent.

## Rules
1. The partial-intent query must be strictly shorter than the full query.
2. If the modification query contains **multiple intents**, the partial-intent query should include **only a subset of those intents** (e.g., keep 1–2 and omit the rest).
3. If the modification query contains **only one intent**, the partial-intent query should express a **weaker or more general version** of that intent.
4. The result must remain a **coherent, natural-sounding text query**.
5. Do NOT describe the reference image itself; only describe the desired changes.
6. Output ONLY the partial-intent query text — no explanation, no quotes.
"""

SYSTEM_PROMPT_CIRR = """\
You are an expert in image retrieval.

## Task
Given a reference image and a full modification query describing ALL desired changes, \
generate a **partial-intent query** — a shorter text that captures only part of the modification intent.

## Rules
1. The partial-intent query must be strictly shorter than the full query.
2. If the modification query contains **multiple intents**, the partial-intent query should include **only a subset of those intents** (e.g., keep 1–2 and omit the rest).
3. If the modification query contains **only one intent**, the partial-intent query should express a **weaker or more general version** of that intent.
4. The result must remain a **coherent, natural-sounding text query**.
5. Do NOT describe the reference image itself; only describe the desired changes.
6. Output ONLY the partial-intent query text — no explanation, no quotes.
"""


def log(message: str) -> None:
    print(message, flush=True)


def load_fiq_training_data(dress_types: list[str]) -> list[dict]:
    """Load FIQ training triplets and build composed queries."""
    samples = []
    for dt in dress_types:
        cap_file = FIQ_PATH / "captions" / f"cap.{dt}.train.json"
        with open(cap_file) as f:
            triplets = json.load(f)
        for idx, t in enumerate(triplets):
            c0 = t["captions"][0].strip(".?, ").capitalize()
            c1 = t["captions"][1].strip(".?, ")
            composed = f"{c0} and {c1}"
            samples.append(
                {
                    "sample_id": f"fiq_{dt}_{idx}",
                    "index": idx,
                    "dress_type": dt,
                    "candidate": t["candidate"],
                    "target": t["target"],
                    "captions": t["captions"],
                    "composed_query": composed,
                    "image_path": str(FIQ_PATH / "images" / f"{t['candidate']}.png"),
                }
            )
    return samples


def load_cirr_training_data() -> list[dict]:
    """Load CIRR training triplets."""
    cap_file = CIRR_PATH / "cirr" / "captions" / "cap.rc2.train.json"
    split_file = CIRR_PATH / "cirr" / "image_splits" / "split.rc2.train.json"

    with open(cap_file) as f:
        triplets = json.load(f)
    with open(split_file) as f:
        name_to_relpath = json.load(f)

    samples = []
    for idx, t in enumerate(triplets):
        reference_name = t["reference"]
        caption = t["caption"]
        image_path = str(CIRR_PATH / name_to_relpath[reference_name])
        samples.append(
            {
                "sample_id": f"cirr_{idx}",
                "index": idx,
                "reference": reference_name,
                "target": t["target_hard"],
                "composed_query": caption,
                "image_path": image_path,
            }
        )
    return samples


def _fashion200k_caption_post_process(caption: str) -> str:
    return (
        caption.strip()
        .replace(".", "dotmark")
        .replace("?", "questionmark")
        .replace("&", "andmark")
        .replace("*", "starmark")
    )


def _get_different_word(
    source_caption: str, target_caption: str
) -> Optional[tuple[str, str, str]]:
    """Match Fashion200k official query construction logic."""
    source_words = source_caption.split()
    target_words = target_caption.split()

    source_word = next((word for word in source_words if word not in target_words), None)
    target_word = next((word for word in target_words if word not in source_words), None)
    if source_word is None or target_word is None:
        return None

    mod_str = f"replace {source_word} with {target_word}"
    return source_word, target_word, mod_str


def _load_fashion200k_train_images() -> list[dict]:
    """Load raw Fashion200k train images and caption index metadata."""
    log("[fashion200k] Reading train label files...")
    imgs = []
    for label_file in sorted((FASHION200K_PATH / "labels").glob("*_train_detect_all.txt")):
        with open(label_file) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                file_path, detection_score, caption = parts
                imgs.append(
                    {
                        "file_path": file_path,
                        "detection_score": float(detection_score),
                        "caption": _fashion200k_caption_post_process(caption),
                        "parent_captions": [],
                        "modifiable": False,
                        "valid_parent_captions": [],
                    }
                )
    log(f"[fashion200k] Loaded {len(imgs)} train images")

    caption2imgids: dict[str, list[int]] = defaultdict(list)
    for idx, img in enumerate(imgs):
        caption2imgids[img["caption"]].append(idx)
    log(f"[fashion200k] Built caption index with {len(caption2imgids)} unique captions")

    parent2children_captions: dict[str, list[str]] = defaultdict(list)
    for caption in caption2imgids.keys():
        for word in caption.split():
            parent_caption = " ".join(caption.replace(word, "").split())
            if caption not in parent2children_captions[parent_caption]:
                parent2children_captions[parent_caption].append(caption)
    log(f"[fashion200k] Built {len(parent2children_captions)} parent captions")

    for parent_caption, children_captions in parent2children_captions.items():
        if len(children_captions) < 2:
            continue
        for caption in children_captions:
            for img_idx in caption2imgids[caption]:
                imgs[img_idx]["modifiable"] = True
                imgs[img_idx]["parent_captions"].append(parent_caption)

    valid_source_indices = []
    skipped_modifiable = 0
    for idx, img in enumerate(imgs):
        if not img["modifiable"]:
            continue

        valid_parents = []
        for parent_caption in img["parent_captions"]:
            has_valid_target = False
            for target_caption in parent2children_captions[parent_caption]:
                if target_caption == img["caption"]:
                    continue
                if _get_different_word(img["caption"], target_caption) is not None:
                    has_valid_target = True
                    break
            if has_valid_target:
                valid_parents.append(parent_caption)

        img["valid_parent_captions"] = valid_parents
        if valid_parents:
            valid_source_indices.append(idx)
        else:
            skipped_modifiable += 1

    log(
        "[fashion200k] "
        f"Modifiable images: {sum(1 for img in imgs if img['modifiable'])}, "
        f"valid sources: {len(valid_source_indices)}, "
        f"skipped noisy sources: {skipped_modifiable}"
    )
    return imgs, caption2imgids, parent2children_captions, valid_source_indices


def _sample_fashion200k_standard_triplet(
    imgs: list[dict],
    caption2imgids: dict[str, list[int]],
    parent2children_captions: dict[str, list[str]],
    valid_source_indices: list[int],
    rng: random.Random,
    idx: int,
) -> tuple[int, int, str, str, str]:
    """Mirror the official TIRG Fashion200k train-time sampling with a fixed RNG."""
    while not imgs[idx]["valid_parent_captions"]:
        idx = rng.choice(valid_source_indices)

    img = imgs[idx]
    source_caption = imgs[idx]["caption"]
    while True:
        parent_caption = rng.choice(img["valid_parent_captions"])
        target_caption = rng.choice(parent2children_captions[parent_caption])
        if target_caption == source_caption:
            continue

        diff = _get_different_word(source_caption, target_caption)
        if diff is None:
            continue

        target_idx = rng.choice(caption2imgids[target_caption])
        source_word, target_word, mod_str = diff
        return idx, target_idx, source_word, target_word, mod_str


def load_fashion200k_training_data() -> list[dict]:
    """Load Fashion200k training triplets using the standard train split logic.

    The official Fashion200k CIR loader samples one target on the fly for each
    train index. We reproduce one deterministic training epoch with a fixed RNG
    so that generated partial-intent queries follow the standard split without
    enumerating every possible caption replacement pair.
    """
    (
        imgs,
        caption2imgids,
        parent2children_captions,
        valid_source_indices,
    ) = _load_fashion200k_train_images()
    rng = random.Random(FASHION200K_STANDARD_SEED)
    log("[fashion200k] Sampling one deterministic standard train epoch...")

    samples = []
    for sample_idx in range(len(imgs)):
        source_idx, target_idx, source_word, target_word, mod_str = (
            _sample_fashion200k_standard_triplet(
                imgs,
                caption2imgids,
                parent2children_captions,
                valid_source_indices,
                rng,
                sample_idx,
            )
        )
        source_img = imgs[source_idx]
        target_img = imgs[target_idx]
        samples.append(
            {
                "sample_id": f"fashion200k_{sample_idx}",
                "index": sample_idx,
                "dataset": "fashion200k",
                "reference": source_img["file_path"],
                "target": target_img["file_path"],
                "source_caption": source_img["caption"],
                "target_caption": target_img["caption"],
                "source_word": source_word,
                "target_word": target_word,
                "composed_query": mod_str,
                "image_path": str(FASHION200K_PATH / source_img["file_path"]),
                "target_image_path": str(FASHION200K_PATH / target_img["file_path"]),
            }
        )
        if (sample_idx + 1) % 10000 == 0:
            log(f"[fashion200k] Prepared {sample_idx + 1}/{len(imgs)} training samples")

    return samples


def _resolve_shoes_image_path(raw_path: str) -> str:
    marker = "shoes_data/"
    if marker in raw_path:
        rel_path = raw_path.split(marker, 1)[1]
        resolved = SHOES_PATH / rel_path
        if resolved.exists():
            return str(resolved)

    name = Path(raw_path).name
    matches = list(SHOES_PATH.rglob(name))
    if len(matches) == 1:
        return str(matches[0])
    raise FileNotFoundError(f"Could not resolve Shoes image path from: {raw_path}")


def load_shoes_training_data() -> list[dict]:
    """Load Shoes train split using unique standard training triplets."""
    train_pairs = np.load(SHOES_PATH / "relative_pairs_train.npy", allow_pickle=True)

    seen_triplets: set[tuple[str, str, str]] = set()
    samples = []
    for item in train_pairs:
        source_name = Path(item["source"]).name
        target_name = Path(item["target"]).name
        composed_query = item["mod"]
        triplet_key = (source_name, target_name, composed_query)
        if triplet_key in seen_triplets:
            continue
        seen_triplets.add(triplet_key)

        sample_idx = len(samples)
        samples.append(
            {
                "sample_id": f"shoes_{sample_idx}",
                "index": sample_idx,
                "dataset": "shoes",
                "reference": source_name,
                "target": target_name,
                "composed_query": composed_query,
                "image_path": _resolve_shoes_image_path(item["source"]),
                "target_image_path": _resolve_shoes_image_path(item["target"]),
            }
        )

    return samples


def load_lasco_training_data() -> list[dict]:
    """Load LASCO training triplets."""
    with open(LASCO_PATH / "lasco_train.json") as f:
        triplets = json.load(f)

    samples = []
    for idx, t in enumerate(triplets):
        reference_id, reference_relpath = t["query-image"]
        target_id, target_relpath = t["target-image"]
        samples.append(
            {
                "sample_id": f"lasco_{t['qid']}",
                "index": idx,
                "dataset": "lasco",
                "qid": t["qid"],
                "reference": reference_id,
                "target": target_id,
                "composed_query": t["query-text"],
                "image_path": str(LASCO_PATH / reference_relpath),
                "target_image_path": str(LASCO_PATH / target_relpath),
            }
        )
    return samples


def encode_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_image_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/png"


def call_api(
    client: OpenAI,
    image_path: str,
    composed_query: str,
    model: str,
    system_prompt: str,
    max_retries: int = 3,
) -> str:
    """Call Qwen multimodal API with exponential-backoff retry."""
    img_b64 = encode_image_base64(Path(image_path))
    mime_type = get_image_mime_type(image_path)
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{img_b64}"},
        },
        {
            "type": "text",
            "text": (
                f'The full modification query is: "{composed_query}"\n'
                "Generate a partial-intent query."
            ),
        },
    ]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=80,
                temperature=0.3,
            )
            text = resp.choices[0].message.content.strip().strip("\"'")
            return text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1) + 0.5
                time.sleep(wait)
            else:
                raise


def process_one(
    client: OpenAI,
    sample: dict,
    model: str,
    system_prompt: str,
    max_retries: int,
) -> dict:
    """Process a single sample: call API and return result dict."""
    try:
        partial = call_api(
            client, sample["image_path"], sample["composed_query"],
            model, system_prompt, max_retries,
        )
        return {
            **sample,
            "partial_intent_query": partial,
            "status": "ok",
            "error": None,
            "model_name": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            **sample,
            "partial_intent_query": None,
            "status": "error",
            "error": str(e),
            "model_name": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def _get_sample_id(entry: dict) -> str:
    """Extract sample_id from a progress entry, with backward compat for old FIQ format."""
    if "sample_id" in entry:
        return entry["sample_id"]
    if "dress_type" in entry:
        return f"fiq_{entry['dress_type']}_{entry['index']}"
    if entry.get("dataset") == "fashion200k":
        return f"fashion200k_{entry['index']}"
    if entry.get("dataset") == "shoes":
        return f"shoes_{entry['index']}"
    if entry.get("dataset") == "lasco":
        return f"lasco_{entry['index']}"
    return f"cirr_{entry['index']}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate partial-intent queries for CIR training sets"
    )
    parser.add_argument(
        "--dataset",
        default="fashioniq",
        choices=["fashioniq", "cirr", "fashion200k", "shoes", "lasco"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--dress_types",
        nargs="+",
        default=["dress", "shirt", "toptee"],
        choices=["dress", "shirt", "toptee"],
        help="FashionIQ dress types (only used when --dataset=fashioniq)",
    )
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: auto based on dataset)",
    )
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the DASHSCOPE_API_KEY environment variable")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if args.dataset == "fashioniq":
        samples = load_fiq_training_data(args.dress_types)
        system_prompt = SYSTEM_PROMPT_FIQ
        default_output_dir = "outputs/fiq_partial_intent_queries"
    elif args.dataset == "cirr":
        samples = load_cirr_training_data()
        system_prompt = SYSTEM_PROMPT_CIRR
        default_output_dir = "outputs/cirr_partial_intent_queries"
    elif args.dataset == "fashion200k":
        samples = load_fashion200k_training_data()
        system_prompt = SYSTEM_PROMPT_FIQ
        default_output_dir = "outputs/fashion200k_partial_intent_queries_standard"
    elif args.dataset == "shoes":
        samples = load_shoes_training_data()
        system_prompt = SYSTEM_PROMPT_FIQ
        default_output_dir = "outputs/shoes_partial_intent_queries_standard"
    else:
        samples = load_lasco_training_data()
        system_prompt = SYSTEM_PROMPT_CIRR
        default_output_dir = "outputs/lasco_partial_intent_queries"

    output_dir = Path(args.output_dir or default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Total training samples: {len(samples)}")

    # ── Resume from progress file ──
    progress_file = output_dir / "progress.jsonl"
    done_keys: set[str] = set()
    if progress_file.exists():
        with open(progress_file) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                done_keys.add(_get_sample_id(entry))
        print(f"Already completed: {len(done_keys)}")

    remaining = [s for s in samples if s["sample_id"] not in done_keys]
    print(f"Remaining to process: {len(remaining)}")
    if not remaining:
        print("Nothing to do — consolidating existing results.")
    else:
        # ── Process with thread pool ──
        write_lock = Lock()
        ok_count = 0
        err_count = 0

        def _worker(sample):
            return process_one(client, sample, args.model, system_prompt, args.max_retries)

        with open(progress_file, "a", buffering=1) as pf:
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {pool.submit(_worker, s): s for s in remaining}
                pbar = tqdm(
                    as_completed(futures),
                    total=len(remaining),
                    desc="Generating partial-intent queries",
                )
                for future in pbar:
                    result = future.result()
                    with write_lock:
                        pf.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if result["status"] == "ok":
                        ok_count += 1
                    else:
                        err_count += 1
                    pbar.set_postfix(ok=ok_count, err=err_count)

        print(f"\nProcessing done: {ok_count} ok, {err_count} errors")

    # ── Consolidate to final JSON (key=composed_query, value=partial_intent_query) ──
    all_entries = []
    with open(progress_file) as f:
        for line in f:
            if not line.strip():
                continue
            all_entries.append(json.loads(line))

    final: dict[str, str] = {}
    for entry in all_entries:
        if entry["status"] == "ok" and entry["partial_intent_query"]:
            final[entry["composed_query"]] = entry["partial_intent_query"]

    output_file = output_dir / "partial_intent_queries.json"
    with open(output_file, "w") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(final)} partial-intent queries to {output_file}")

    # ── Save detail files ──
    if args.dataset == "fashioniq":
        per_type: dict[str, list] = {}
        for entry in all_entries:
            dt = entry.get("dress_type", "unknown")
            per_type.setdefault(dt, []).append(entry)

        for dt, entries in per_type.items():
            entries.sort(key=lambda x: x["index"])
            detail_file = output_dir / f"partial_intent.{dt}.train.jsonl"
            with open(detail_file, "w") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            ok = sum(1 for e in entries if e["status"] == "ok")
            print(f"  {dt}: {ok}/{len(entries)} saved to {detail_file}")
    else:
        all_entries.sort(key=lambda x: x["index"])
        detail_file = output_dir / "partial_intent.train.jsonl"
        with open(detail_file, "w") as f:
            for e in all_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        ok = sum(1 for e in all_entries if e["status"] == "ok")
        print(f"  {args.dataset}: {ok}/{len(all_entries)} saved to {detail_file}")


if __name__ == "__main__":
    main()
