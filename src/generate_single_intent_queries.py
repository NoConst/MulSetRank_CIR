#!/usr/bin/env python3
"""
Generate single-intent query variants for FashionIQ train/val text queries.

For each composed FashionIQ query:
  - If the query has multiple intents, split it into multiple single-intent
    queries. Each split query must be a contiguous substring copied from the
    original query.
  - If the query has one intent, generate one weaker version of that intent.

The script uses DeepSeek's OpenAI-compatible chat API. It writes a detailed
JSONL progress file for resume, a main JSON mapping query text to a list of
single-intent queries, and a compatibility JSON that maps query text to the
first generated query for existing partial-intent training code.

Example:
    python src/generate_single_intent_queries.py \
        --dataset fashioniq \
        --splits train val \
        --dress-types dress shirt toptee \
        --max-workers 8 \
        --output-dir outputs/fiq_single_intent_queries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


SYSTEM_PROMPT = """\
You are an expert annotator for composed image retrieval fashion queries.

Task:
Given one FashionIQ text query, decide whether it contains one modification
intent or multiple modification intents.

Definitions:
- A modification intent is one separately actionable desired change, such as
  color, pattern, material, sleeve length, neckline, fit, length, style,
  coverage, logo/graphic, or adding/removing a visible attribute.
- Do not double-count repeated or synonymous wording.

Output rules:
1. Return one JSON object only. Do not use markdown or explanations.
2. If the query contains multiple intents, set "intent_type" to
   "multi_intent" and set "single_intent_queries" to multiple strings.
3. For multi_intent, every string in "single_intent_queries" must be an exact
   contiguous substring copied from the input query. Do not rewrite words, do
   not add words, do not change word order.
4. For multi_intent, each string should contain only one modification intent.
   Split adjacent attributes into separate minimal substrings whenever possible.
5. If the query contains one intent, set "intent_type" to "single_intent" and
   set "single_intent_queries" to a list with exactly one weaker or more
   general version of the query.
6. For single_intent, the weaker version may paraphrase, but it must not
   strengthen or add new details.

Examples:
Input: Is solid black with no sleeves and is black with straps
Output: {"intent_type":"multi_intent","single_intent_queries":["Is solid black","with no sleeves","with straps"]}

Input: Is short sleeved and has a collar and is grey with shorter sleeves
Output: {"intent_type":"multi_intent","single_intent_queries":["Is short sleeved","has a collar","is grey","shorter sleeves"]}

Input: Is a lighter color
Output: {"intent_type":"single_intent","single_intent_queries":["is lighter"]}

JSON schema:
{
  "intent_type": "multi_intent" or "single_intent",
  "single_intent_queries": ["..."]
}
"""


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_openai_client(api_key: str, base_url: str) -> OpenAI:
    """Create a client compatible with the pinned OpenAI SDK in this env."""
    http_client = httpx.Client(timeout=60.0, trust_env=False)
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


def normalize_fiq_caption_pair(captions: list[str]) -> str:
    """Match the FashionIQ caption composition used by training/validation."""
    cap0 = captions[0].strip(".?, ").capitalize()
    cap1 = captions[1].strip(".?, ")
    return f"{cap0} and {cap1}"


def load_fiq_samples(
    splits: list[str],
    dress_types: list[str],
    limit_per_group: int | None,
    limit_total: int | None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for split in splits:
        if split not in {"train", "val"}:
            raise ValueError(f"FashionIQ split must be train or val, got: {split}")
        for dress_type in dress_types:
            cap_file = FIQ_PATH / "captions" / f"cap.{dress_type}.{split}.json"
            with open(cap_file) as f:
                triplets = json.load(f)

            group_count = 0
            for idx, triplet in enumerate(triplets):
                if limit_per_group is not None and group_count >= limit_per_group:
                    break
                samples.append(
                    {
                        "sample_id": f"fashioniq_{split}_{dress_type}_{idx}",
                        "dataset": "fashioniq",
                        "split": split,
                        "dress_type": dress_type,
                        "index": idx,
                        "candidate": triplet.get("candidate"),
                        "target": triplet.get("target"),
                        "captions": triplet["captions"],
                        "text_query": normalize_fiq_caption_pair(triplet["captions"]),
                    }
                )
                group_count += 1
                if limit_total is not None and len(samples) >= limit_total:
                    return samples
    return samples


def make_query_id(text_query: str) -> str:
    digest = hashlib.sha1(text_query.encode("utf-8")).hexdigest()
    return f"q_{digest[:16]}"


def build_query_tasks(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks_by_text: dict[str, dict[str, Any]] = {}
    for sample in samples:
        text_query = sample["text_query"]
        task = tasks_by_text.setdefault(
            text_query,
            {
                "query_id": make_query_id(text_query),
                "dataset": "fashioniq",
                "text_query": text_query,
                "sample_ids": [],
                "samples": [],
            },
        )
        task["sample_ids"].append(sample["sample_id"])
        task["samples"].append(sample)
    return sorted(tasks_by_text.values(), key=lambda item: item["query_id"])


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_code_fence(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"Model output is not a JSON object: {text!r}")
    return parsed


def normalize_item(text: Any) -> str:
    if not isinstance(text, str):
        raise ValueError(f"Query item is not a string: {text!r}")
    normalized = " ".join(text.strip().strip("\"'").split())
    if not normalized:
        raise ValueError("Query item is empty")
    return normalized


def find_source_substring(source: str, candidate: str) -> str | None:
    candidate = candidate.strip()
    if candidate in source:
        return candidate

    source_lower = source.lower()
    candidate_lower = candidate.lower()
    pos = source_lower.find(candidate_lower)
    if pos >= 0:
        return source[pos : pos + len(candidate)]

    trimmed = candidate.strip(" .,;:")
    if trimmed != candidate:
        return find_source_substring(source, trimmed)

    return None


def clean_source_substring(source: str, candidate: str) -> str:
    variants = [candidate]
    without_leading_connector = re.sub(
        r"^(?:and|or)\s+",
        "",
        candidate.strip(),
        flags=re.IGNORECASE,
    )
    variants.append(without_leading_connector)

    for variant in variants:
        source_substring = find_source_substring(source, variant)
        if source_substring is not None:
            return source_substring.strip()

    raise ValueError(
        "multi_intent query is not a contiguous substring of the input: "
        f"{candidate!r}"
    )


def validate_model_result(text_query: str, parsed: dict[str, Any]) -> dict[str, Any]:
    intent_type = str(parsed.get("intent_type", "")).strip().lower()
    if intent_type not in {"multi_intent", "single_intent"}:
        raise ValueError(f"Invalid intent_type: {parsed.get('intent_type')!r}")

    raw_queries = parsed.get("single_intent_queries")
    if not isinstance(raw_queries, list):
        raise ValueError("single_intent_queries must be a list")

    queries = [normalize_item(item) for item in raw_queries]
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        if query not in seen:
            deduped.append(query)
            seen.add(query)
    queries = deduped

    if intent_type == "multi_intent":
        if len(queries) < 2:
            raise ValueError("multi_intent output must contain at least two queries")

        queries = [clean_source_substring(text_query, query) for query in queries]
    else:
        if len(queries) != 1:
            raise ValueError("single_intent output must contain exactly one query")
        if queries[0].strip().lower() == text_query.strip().lower():
            raise ValueError("single_intent weakened query is identical to the input")

    return {
        "intent_type": intent_type,
        "single_intent_queries": queries,
    }


def call_deepseek_one(
    client: OpenAI,
    text_query: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> tuple[dict[str, Any], str]:
    user_prompt = (
        "Process this FashionIQ text query according to the rules.\n"
        f"Input query: {text_query}"
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise RuntimeError(
                    "model response was truncated; increase --max-tokens"
                )
            raw_output = response.choices[0].message.content or ""
            if not raw_output.strip():
                raise RuntimeError("model returned empty JSON content")
            parsed = parse_json_object(raw_output)
            return validate_model_result(text_query, parsed), raw_output.strip()
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(2**attempt + 1)

    raise RuntimeError(str(last_error))


def process_one(
    client: OpenAI,
    task: dict[str, Any],
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    try:
        normalized, raw_output = call_deepseek_one(
            client=client,
            text_query=task["text_query"],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        return {
            **task,
            **normalized,
            "raw_model_output": raw_output,
            "status": "ok",
            "error": None,
            "model_name": model,
            "created_at": utc_now(),
        }
    except Exception as exc:
        return {
            **task,
            "intent_type": None,
            "single_intent_queries": [],
            "raw_model_output": "",
            "status": "error",
            "error": str(exc),
            "model_name": model,
            "created_at": utc_now(),
        }


def read_latest_progress(progress_file: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not progress_file.exists():
        return latest
    with open(progress_file) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            query_id = entry.get("query_id")
            if query_id:
                latest[query_id] = entry
    return latest


def sort_entry(entry: dict[str, Any]) -> tuple[str, str]:
    return (entry.get("text_query", ""), entry.get("query_id", ""))


def save_outputs(
    output_dir: Path,
    entries: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    model: str,
    base_url: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_by_query = {entry["query_id"]: entry for entry in entries}
    query_id_by_text = {entry["text_query"]: entry["query_id"] for entry in entries}

    query_mapping: dict[str, list[str]] = {}
    first_query_mapping: dict[str, str] = {}
    for entry in sorted(entries, key=sort_entry):
        if entry.get("status") != "ok":
            continue
        queries = entry.get("single_intent_queries") or []
        if not queries:
            continue
        query_mapping[entry["text_query"]] = queries
        first_query_mapping[entry["text_query"]] = queries[0]

    with open(output_dir / "single_intent_queries.json", "w") as f:
        json.dump(query_mapping, f, ensure_ascii=False, indent=2)

    with open(output_dir / "partial_intent_queries.json", "w") as f:
        json.dump(first_query_mapping, f, ensure_ascii=False, indent=2)

    detail_file = output_dir / "single_intent_details.jsonl"
    with open(detail_file, "w") as f:
        for entry in sorted(entries, key=sort_entry):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    sample_file = output_dir / "single_intent_samples.jsonl"
    with open(sample_file, "w") as f:
        for sample in sorted(
            samples,
            key=lambda item: (
                item.get("split", ""),
                item.get("dress_type", ""),
                int(item.get("index", 0)),
            ),
        ):
            query_id = query_id_by_text.get(sample["text_query"], make_query_id(sample["text_query"]))
            entry = latest_by_query.get(query_id, {})
            row = {
                **sample,
                "query_id": query_id,
                "intent_type": entry.get("intent_type"),
                "single_intent_queries": entry.get("single_intent_queries", []),
                "status": entry.get("status", "missing"),
                "error": entry.get("error"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(
        entries=entries,
        samples=samples,
        query_mapping=query_mapping,
        model=model,
        base_url=base_url,
    )
    with open(output_dir / "single_intent_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def build_summary(
    entries: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    query_mapping: dict[str, list[str]],
    model: str,
    base_url: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "model": model,
        "base_url": base_url,
        "dataset": "fashioniq",
        "total_samples": len(samples),
        "unique_queries": len(entries),
        "ok_queries": 0,
        "error_queries": 0,
        "multi_intent_queries": 0,
        "single_intent_queries": 0,
        "generated_query_keys": len(query_mapping),
        "by_split_dress_type": {},
    }

    for entry in entries:
        if entry.get("status") == "ok":
            summary["ok_queries"] += 1
            if entry.get("intent_type") == "multi_intent":
                summary["multi_intent_queries"] += 1
            elif entry.get("intent_type") == "single_intent":
                summary["single_intent_queries"] += 1
        else:
            summary["error_queries"] += 1

    for sample in samples:
        key = f"{sample['split']}/{sample['dress_type']}"
        group = summary["by_split_dress_type"].setdefault(
            key,
            {
                "samples": 0,
                "unique_queries": set(),
            },
        )
        group["samples"] += 1
        group["unique_queries"].add(make_query_id(sample["text_query"]))

    for group in summary["by_split_dress_type"].values():
        group["unique_queries"] = len(group["unique_queries"])

    return summary


def print_sample_counts(samples: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for sample in samples:
        key = (sample["split"], sample["dress_type"])
        counts[key] = counts.get(key, 0) + 1

    print(f"Loaded {len(samples)} FashionIQ samples")
    for (split, dress_type), count in sorted(counts.items()):
        print(f"  fashioniq/{split}/{dress_type}: {count}")


def print_summary(summary: dict[str, Any]) -> None:
    print(
        "Summary: "
        f"{summary['ok_queries']}/{summary['unique_queries']} unique queries ok, "
        f"errors={summary['error_queries']}, "
        f"multi={summary['multi_intent_queries']}, "
        f"single={summary['single_intent_queries']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate single-intent/weak FIQ text queries with DeepSeek."
    )
    parser.add_argument(
        "--dataset",
        default="fashioniq",
        choices=["fashioniq"],
        help="Dataset to process. Currently implemented for FashionIQ.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="FashionIQ splits to process.",
    )
    parser.add_argument(
        "--dress-types",
        "--dress_types",
        dest="dress_types",
        nargs="+",
        default=["dress", "shirt", "toptee"],
        choices=["dress", "shirt", "toptee"],
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--output-dir", default="outputs/fiq_single_intent_queries")
    parser.add_argument("--max-workers", "--max_workers", dest="max_workers", type=int, default=4)
    parser.add_argument("--max-retries", "--max_retries", dest="max_retries", type=int, default=3)
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--limit-per-group",
        type=int,
        default=None,
        help="Optional smoke-test cap per split/dress_type group.",
    )
    parser.add_argument(
        "--limit-total",
        type=int,
        default=None,
        help="Optional smoke-test cap across all loaded samples.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load and count samples; do not call DeepSeek.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry queries whose latest progress entry is an error.",
    )
    args = parser.parse_args()

    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive")

    samples = load_fiq_samples(
        splits=args.splits,
        dress_types=args.dress_types,
        limit_per_group=args.limit_per_group,
        limit_total=args.limit_total,
    )
    print_sample_counts(samples)

    tasks = build_query_tasks(samples)
    print(f"Unique text queries: {len(tasks)}")

    if args.dry_run:
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Please set the {args.api_key_env} environment variable")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / "single_intent_progress.jsonl"

    latest_progress = read_latest_progress(progress_file)
    task_ids = {task["query_id"] for task in tasks}
    completed_ids = {
        query_id
        for query_id, entry in latest_progress.items()
        if query_id in task_ids and (entry.get("status") == "ok" or not args.retry_errors)
    }
    remaining = [task for task in tasks if task["query_id"] not in completed_ids]

    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {len(remaining)}")

    client = create_openai_client(api_key=api_key, base_url=args.base_url)

    if remaining:
        write_lock = Lock()
        ok_count = 0
        err_count = 0

        with open(progress_file, "a", buffering=1) as pf:
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = [
                    pool.submit(
                        process_one,
                        client,
                        task,
                        args.model,
                        args.max_tokens,
                        args.temperature,
                        args.max_retries,
                    )
                    for task in remaining
                ]
                pbar = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Generating single-intent queries",
                )
                for future in pbar:
                    entry = future.result()
                    with write_lock:
                        pf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if entry["status"] == "ok":
                        ok_count += 1
                    else:
                        err_count += 1
                    pbar.set_postfix(ok=ok_count, err=err_count)

        print(f"Processing done: {ok_count} ok, {err_count} errors")

    latest_progress = read_latest_progress(progress_file)
    entries = []
    for task in tasks:
        entry = latest_progress.get(task["query_id"])
        if entry is None:
            entry = {
                **task,
                "intent_type": None,
                "single_intent_queries": [],
                "raw_model_output": "",
                "status": "error",
                "error": "missing generation",
                "model_name": args.model,
                "created_at": utc_now(),
            }
        entries.append(entry)

    summary = save_outputs(
        output_dir=output_dir,
        entries=entries,
        samples=samples,
        model=args.model,
        base_url=args.base_url,
    )
    print_summary(summary)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
