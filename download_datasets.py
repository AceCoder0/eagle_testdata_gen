"""
Download math datasets from HuggingFace or OpenCompass and normalize to a common JSONL pool format.

Supports two download sources:
  --source huggingface  (default) — download via HuggingFace datasets library
  --source opencompass              — download from OpenCompass Aliyun OSS mirrors

Outputs math_pool.jsonl with fields: id, question, answer, source, question_tokens, answer_tokens, total_tokens.
"""
import os
import json
import random
import argparse
import hashlib
import tempfile
import zipfile
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

# ---------------------------------------------------------------------------
# Dataset registry -- each entry maps a dataset source to HF path + fields
# ---------------------------------------------------------------------------
OPENCOMPASS_BASE = "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data"

DATASET_CONFIGS = [
    {
        "name": "gsm8k",
        "hf_path": "openai/gsm8k",
        "hf_config": "main",
        "split": "train",
        "question_field": "question",
        "answer_field": "answer",
        "filter": None,
        # OpenCompass mirror: files inside gsm8k/ subdirectory in zip
        "oc_url": f"{OPENCOMPASS_BASE}/gsm8k.zip",
        "oc_file": "gsm8k/train.jsonl",
        "oc_question_field": "question",
        "oc_answer_field": "answer",
        "oc_filter": None,
    },
    {
        "name": "gsm8k_test",
        "hf_path": "openai/gsm8k",
        "hf_config": "main",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "filter": None,
        "oc_url": f"{OPENCOMPASS_BASE}/gsm8k.zip",
        "oc_file": "gsm8k/test.jsonl",
        "oc_question_field": "question",
        "oc_answer_field": "answer",
        "oc_filter": None,
    },
    {
        "name": "math_500",
        "hf_path": "HuggingFaceH4/MATH-500",
        "hf_config": None,
        "split": "test",
        "question_field": "problem",
        "answer_field": "solution",
        "filter": None,
        "oc_url": f"{OPENCOMPASS_BASE}/math.zip",
        "oc_file": "math/test_prm800k_500.jsonl",
        "oc_question_field": "problem",
        "oc_answer_field": "solution",
        "oc_filter": None,
    },
    {
        "name": "aime2024",
        "hf_path": "AI-MO/aimo-validation-aime",
        "hf_config": None,
        "split": "train",
        "question_field": "problem",
        "answer_field": "solution",
        "filter": lambda r: r.get("url", "") and "2024" in str(r.get("url", "")),
        "oc_url": f"{OPENCOMPASS_BASE}/aime.zip",
        "oc_file": "aime.jsonl",
        "oc_question_field": "origin_prompt",
        "oc_answer_field": "gold_answer",
        "oc_filter": lambda r: "2024" in str(r.get("source", "")),
    },
    {
        "name": "aime2025",
        "hf_path": "yentinglin/aime_2025",
        "hf_config": None,
        "split": "train",
        "question_field": "problem",
        "answer_field": "solution",
        "filter": None,
        "oc_url": f"{OPENCOMPASS_BASE}/aime2025.zip",
        "oc_file": "aime2025/aime2025.jsonl",
        "oc_question_field": "question",
        "oc_answer_field": "answer",
        "oc_filter": None,
    },
    {
        "name": "mgsm",
        "hf_path": "juletxara/mgsm",
        "hf_config": "en",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer_number",
        "filter": None,
        "oc_url": None,  # No OpenCompass mirror available
    },
    {
        "name": "dapo_math_17k",
        "hf_path": "open-r1/DAPO-Math-17k",
        "hf_config": None,
        "split": "train",
        "question_field": "prompt",
        "answer_field": "solution",
        "filter": None,
        "oc_url": None,  # No OpenCompass mirror available
    },
]

# Fallback configs -- try these if primary HF path fails (same dict format)
FALLBACKS = {
    "aime2024": [
        {"hf_path": "TIGER-Lab/AIME-2024", "hf_config": None, "split": "train",
         "question_field": "problem", "answer_field": "solution", "filter": None},
    ],
    "aime2025": [
        {"hf_path": "TIGER-Lab/AIME-2025", "hf_config": None, "split": "train",
         "question_field": "problem", "answer_field": "solution", "filter": None},
    ],
    "mgsm": [
        {"hf_path": "juletxara/mgsm", "hf_config": "en", "split": "train",
         "question_field": "question", "answer_field": "answer", "filter": None},
    ],
}


def load_tokenizer(model_path):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.model_max_length = int(1e9)
    return tok


def question_hash(question: str) -> str:
    return hashlib.md5(question.strip().encode("utf-8")).hexdigest()


def try_load_dataset(hf_path, hf_config, split, max_retries=2):
    """Try loading a HF dataset with retries."""
    from datasets import load_dataset as hf_load
    import time

    for attempt in range(max_retries):
        try:
            if hf_config:
                ds = hf_load(hf_path, hf_config, split=split)
            else:
                ds = hf_load(hf_path, split=split)
            return ds
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} for {hf_path}: {e}")
                time.sleep(2)
            else:
                raise


def download_oc_dataset(config, tokenizer, seen_hashes):
    """Download one dataset from OpenCompass mirror. Returns list of JSONL-ready dicts."""
    import io

    name = config["name"]
    url = config.get("oc_url")
    if not url:
        print(f"  SKIP: No OpenCompass mirror for {name}")
        return []

    oc_file = config["oc_file"]
    q_field = config["oc_question_field"]
    a_field = config["oc_answer_field"]
    filt = config.get("oc_filter")

    print(f"  Downloading {url}")
    try:
        resp = urlopen(url, timeout=120)
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc="  ") as pbar:
            data = io.BytesIO()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                data.write(chunk)
                pbar.update(len(chunk))
        data.seek(0)
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return []

    records = []
    try:
        with zipfile.ZipFile(data) as zf:
            if oc_file not in zf.namelist():
                print(f"  SKIP: {oc_file} not found in zip (available: {zf.namelist()})")
                return []
            content = zf.read(oc_file).decode("utf-8")
    except Exception as e:
        print(f"  Failed to extract {oc_file} from zip: {e}")
        return []

    for i, line in enumerate(content.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            question = str(row.get(q_field, "")).strip()
            answer = str(row.get(a_field, "")).strip()

            if not question:
                continue

            if filt is not None and not filt(row):
                continue

            qh = question_hash(question)
            if qh in seen_hashes:
                continue
            seen_hashes.add(qh)

            q_tokens = len(tokenizer.encode(question))
            a_tokens = len(tokenizer.encode(answer))

            records.append({
                "id": f"{name}_{i}",
                "question": question,
                "answer": answer,
                "source": name,
                "question_tokens": q_tokens,
                "answer_tokens": a_tokens,
                "total_tokens": q_tokens + a_tokens,
            })
        except Exception:
            continue

    print(f"  Got {len(records)} records")
    return records


def download_dataset(config, tokenizer, seen_hashes, source="huggingface"):
    """Download and normalize one dataset config. Returns list of JSONL-ready dicts."""
    name = config["name"]

    if source == "opencompass":
        return download_oc_dataset(config, tokenizer, seen_hashes)

    # --- HuggingFace path ---
    records = []

    # Primary config + fallbacks
    hf_configs_to_try = [config]
    base_name = name.replace("_test", "")
    if name in FALLBACKS:
        hf_configs_to_try.extend(FALLBACKS[name])
    elif base_name in FALLBACKS:
        hf_configs_to_try.extend(FALLBACKS[base_name])

    ds = None
    for cfg in hf_configs_to_try:
        try:
            ds = try_load_dataset(cfg["hf_path"], cfg.get("hf_config"), cfg["split"])
            actual_cfg = cfg
            print(f"  Loaded from {cfg['hf_path']} (split={cfg['split']})")
            break
        except Exception as e:
            print(f"  Failed {cfg['hf_path']}: {e}")
            continue

    if ds is None:
        print(f"  SKIP: Could not load {name} from any source")
        return records

    q_field = actual_cfg["question_field"]
    a_field = actual_cfg["answer_field"]
    filt = actual_cfg.get("filter")

    for i, row in enumerate(ds):
        try:
            question = str(row.get(q_field, "")).strip()
            answer = str(row.get(a_field, "")).strip()

            if not question:
                continue

            if filt is not None and not filt(row):
                continue

            qh = question_hash(question)
            if qh in seen_hashes:
                continue
            seen_hashes.add(qh)

            q_tokens = len(tokenizer.encode(question))
            a_tokens = len(tokenizer.encode(answer))

            records.append({
                "id": f"{name}_{i}",
                "question": question,
                "answer": answer,
                "source": name,
                "question_tokens": q_tokens,
                "answer_tokens": a_tokens,
                "total_tokens": q_tokens + a_tokens,
            })
        except Exception:
            continue

    print(f"  Got {len(records)} records")
    return records


def main():
    parser = argparse.ArgumentParser(description="Download math datasets for test data generation")
    parser.add_argument("--output_pool", type=str, default="math_pool.jsonl",
                        help="Output JSONL file path (default: math_pool.jsonl)")
    parser.add_argument("--tokenizer_path", type=str, default="./DeepSeekR1",
                        help="Path to DeepSeekR1 tokenizer directory")
    parser.add_argument("--source", type=str, choices=["huggingface", "opencompass"],
                        default="huggingface",
                        help="Download source: huggingface (default) or opencompass")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated dataset names to skip")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer_path = os.path.abspath(args.tokenizer_path)
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Download source: {args.source}")

    skip_set = set(s.strip() for s in args.skip.split(",") if s.strip())

    all_records = []
    seen_hashes = set()

    for config in DATASET_CONFIGS:
        name = config["name"]
        if name in skip_set:
            print(f"\n[{name}] SKIP (requested)")
            continue

        print(f"\n[{name}] Downloading...")
        records = download_dataset(config, tokenizer, seen_hashes, source=args.source)
        all_records.extend(records)

    # Sort by id for deterministic output
    all_records.sort(key=lambda r: r["id"])

    # Save pool
    output_path = os.path.abspath(args.output_pool)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*50}")
    print(f"Total records: {len(all_records)}")
    print(f"Saved to: {output_path}")

    # Per-source stats
    from collections import Counter
    source_counts = Counter(r["source"] for r in all_records)
    for src, cnt in sorted(source_counts.items()):
        q_lens = [r["question_tokens"] for r in all_records if r["source"] == src]
        a_lens = [r["answer_tokens"] for r in all_records if r["source"] == src]
        if q_lens:
            print(f"  {src}: {cnt} records, q_token_avg={sum(q_lens)//len(q_lens)}, "
                  f"a_token_avg={sum(a_lens)//len(a_lens)}")


if __name__ == "__main__":
    main()
