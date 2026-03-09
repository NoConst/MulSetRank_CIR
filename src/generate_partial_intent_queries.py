#!/usr/bin/env python3
"""
Generate partial-intent queries for FashionIQ / CIRR training sets using Qwen-Plus API.

For each training sample, takes the reference image + composed text query and
produces a shorter "partial-intent" query that captures only part of the
original modification intent.

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
"""

import json
import os
import base64
import time
import argparse
import mimetypes
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm
from openai import OpenAI

DATASET_PATH = Path(
    "/root/siton-data-92a7d2fc7b594215b07e48fd8818598b/CAM-CIR_backup/datasets"
)
FIQ_PATH = DATASET_PATH / "fashionIQ_dataset"
CIRR_PATH = DATASET_PATH / "cirr_dataset" / "CIRR"

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
    return f"cirr_{entry['index']}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate partial-intent queries for FashionIQ / CIRR training set"
    )
    parser.add_argument(
        "--dataset",
        default="fashioniq",
        choices=["fashioniq", "cirr"],
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
    else:
        samples = load_cirr_training_data()
        system_prompt = SYSTEM_PROMPT_CIRR
        default_output_dir = "outputs/cirr_partial_intent_queries"

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
        print(f"  CIRR: {ok}/{len(all_entries)} saved to {detail_file}")


if __name__ == "__main__":
    main()
