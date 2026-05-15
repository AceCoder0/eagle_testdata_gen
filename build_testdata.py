#!/usr/bin/env python3
"""
Build performance test cases from math datasets.

Orchestrates: load pool -> dedup -> generate few-shot prompts at each target length.

Usage:
  # Step 1: Download datasets (one-time)
  python download_datasets.py --output_pool math_pool.jsonl

  # Step 2: Build test cases
  python build_testdata.py \
      --pool math_pool.jsonl \
      --targets 3500,16000,32000,64000,200000 \
      --samples 100 \
      --tolerance 0.05 \
      --output_dir ./output/
"""
import os
import json
import argparse
import random
from collections import Counter
from dedup import exact_dedup
from fewshot import load_pool, load_tokenizer, FewShotBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build math performance test cases via few-shot construction"
    )
    parser.add_argument("--pool", type=str, default="math_pool.jsonl",
                        help="Path to math Q&A pool JSONL (from download_datasets.py)")
    parser.add_argument("--targets", type=str, default="3500,16000,32000,64000,200000",
                        help="Comma-separated target input token lengths")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples per target length")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Acceptable fractional deviation from target (e.g., 0.05 = +/-5%%)")
    parser.add_argument("--min_qa_pairs", type=int, default=2,
                        help="Minimum exemplar Q&A pairs in few-shot prompt")
    parser.add_argument("--tokenizer_path", type=str, default="./DeepSeekR1",
                        help="Path to tokenizer directory")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory for output JSONL files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    # Parse targets
    targets = [int(t.strip()) for t in args.targets.split(",") if t.strip()]
    targets.sort()

    # Load tokenizer
    tokenizer_path = os.path.abspath(args.tokenizer_path)
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    # Load and deduplicate pool
    pool_path = os.path.abspath(args.pool)
    print(f"Loading math pool from {pool_path}")
    pool = load_pool(pool_path)
    print(f"  Loaded {len(pool)} records")

    # Global dedup
    before = len(pool)
    pool = exact_dedup(pool, key="question")
    print(f"  After exact dedup: {len(pool)} records (removed {before - len(pool)})")

    # Per-source stats
    source_counts = Counter(r["source"] for r in pool)
    print("  Sources:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src}: {cnt}")

    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Estimate max achievable token length
    max_qa_tokens = sum(r["total_tokens"] for r in pool)
    print(f"\n  Total pool tokens (Q+A): {max_qa_tokens:,}")

    # Initialize few-shot builder
    builder = FewShotBuilder(pool, tokenizer, seed=args.seed)

    # Generate test cases for each target length
    for target in targets:
        label = f"{target // 1000}k" if target >= 1000 else str(target)
        output_file = os.path.join(output_dir, f"testdata_{label}.jsonl")

        print(f"\n{'='*50}")
        print(f"Target: {target:,} tokens ({args.samples} samples)")
        print(f"  Tolerance: +/-{args.tolerance * 100:.0f}% "
              f"[{int(target * (1 - args.tolerance)):,}, {int(target * (1 + args.tolerance)):,}]")

        records = builder.build_many(
            target_tokens=target,
            num_samples=args.samples,
            tolerance=args.tolerance,
            min_qa_pairs=args.min_qa_pairs,
        )

        # Write output
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Stats
        lengths = [r["question_token_len"] for r in records]
        if lengths:
            within_tol = sum(
                1 for l in lengths
                if target * (1 - args.tolerance) <= l <= target * (1 + args.tolerance)
            )
            print(f"  Generated: {len(records)} samples")
            print(f"  Length range: {min(lengths):,} - {max(lengths):,}")
            print(f"  Length mean: {sum(lengths) // len(lengths):,}")
            print(f"  Within tolerance: {within_tol}/{len(records)}")
            print(f"  Avg exemplars: {sum(r.get('num_exemplars', 0) for r in records) // len(records)}")
            print(f"  Output: {output_file}")
        else:
            print(f"  FAILED: No samples generated for target={target}")

    print(f"\n{'='*50}")
    print(f"Done. Output files in: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".jsonl"):
            path = os.path.join(output_dir, f)
            count = sum(1 for _ in open(path, "r", encoding="utf-8"))
            print(f"  {f}: {count} records")


if __name__ == "__main__":
    main()
