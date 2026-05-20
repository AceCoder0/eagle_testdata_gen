#!/usr/bin/env python3
"""
Build performance test cases from math datasets.

Orchestrates: load pool -> dedup -> generate few-shot prompts at each target length.

Usage:
  # Step 1: Download datasets (one-time)
  python download_datasets.py --output_pool math_pool.jsonl

  # Step 2: Build test cases (independent samples)
  python build_testdata.py \\
      --pool math_pool.jsonl \\
      --targets 3500,16000,32000,64000,200000 \\
      --samples 100 \\
      --tolerance 0.05 \\
      --output_dir ./output/

  # Step 3: Build test cases WITH prefix cache hit rate control
  python build_testdata.py \\
      --pool math_pool.jsonl \\
      --targets 32000 \\
      --samples 500 \\
      --prefix_rate 0.6 \\
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
    parser.add_argument("--targets", type=str, default="4k,16k,32k,64k,200k",
                        help="Comma-separated target input token lengths. "
                             "Use 'k' suffix for x1024 (e.g. 32k=32768), "
                             "or raw numbers (e.g. 32768)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples per target length")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Acceptable fractional deviation from target (e.g., 0.05 = +/-5%%)")
    parser.add_argument("--min_qa_pairs", type=int, default=2,
                        help="Minimum exemplar Q&A pairs in few-shot prompt")
    parser.add_argument("--prefix_rate", type=float, default=0.0,
                        help="Prefix cache hit rate [0, 1]. 0=independent samples, "
                             "0.6=60%% shared prefix across all samples in the batch, "
                             "0.9=90%% shared. All samples in one batch share the same "
                             "common prefix (same Q&A pairs in same order).")
    parser.add_argument("--tokenizer_path", type=str, default="./DeepSeekR1",
                        help="Path to tokenizer directory")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory for output JSONL files")
    parser.add_argument("--answer_style", type=str, default="mixed",
                        choices=["mixed", "detailed", "concise"],
                        help="Answer style for few-shot exemplars: mixed (default), "
                             "detailed (prefer long step-by-step answers), "
                             "concise (prefer short final answers)")
    parser.add_argument("--min_answer_tokens", type=int, default=0,
                        help="Minimum answer tokens for pool records. "
                             "Filters both exemplars AND final questions. "
                             "Use to guide model toward natural long outputs "
                             "(e.g. 512 for 1024-token target output)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if not 0.0 <= args.prefix_rate <= 1.0:
        parser.error("--prefix_rate must be in [0, 1]")

    random.seed(args.seed)

    # Parse targets: support "32k" suffix (k=1024) and raw numbers
    targets = []
    for t in args.targets.split(","):
        t = t.strip()
        if not t:
            continue
        if t.lower().endswith("k"):
            targets.append(int(float(t[:-1]) * 1024))
        else:
            targets.append(int(t))
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
    builder = FewShotBuilder(pool, tokenizer, seed=args.seed,
                             answer_style=args.answer_style,
                             min_answer_tokens=args.min_answer_tokens)
    if args.min_answer_tokens > 0:
        print(f"  After min_answer_tokens filter ({args.min_answer_tokens}): "
              f"{len(builder.pool)} records (from {len(pool)})")

    use_prefix = args.prefix_rate > 0.0

    # Generate test cases for each target length
    for target in targets:
        label = f"{target // 1024}k" if target >= 1024 else str(target)

        if use_prefix:
            pr_label = f"p{str(args.prefix_rate).replace('.', '_')}"
            output_file = os.path.join(output_dir, f"testdata_{label}_{pr_label}.jsonl")
        else:
            output_file = os.path.join(output_dir, f"testdata_{label}.jsonl")

        print(f"\n{'='*50}")
        print(f"Target: {target:,} tokens ({args.samples} samples)")
        if use_prefix:
            common_len = int(target * args.prefix_rate)
            unique_len = target - common_len
            print(f"  Prefix rate: {args.prefix_rate} "
                  f"(common prefix ~{common_len:,} tokens, unique suffix ~{unique_len:,} tokens)")
        print(f"  Tolerance: +/-{args.tolerance * 100:.0f}% "
              f"[{int(target * (1 - args.tolerance)):,}, {int(target * (1 + args.tolerance)):,}]")

        if use_prefix:
            records = builder.build_prefix_batch(
                target_tokens=target,
                num_samples=args.samples,
                prefix_rate=args.prefix_rate,
                tolerance=args.tolerance,
                min_qa_pairs=args.min_qa_pairs,
            )
        else:
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
            if use_prefix:
                common_avg = sum(r.get("common_exemplars", 0) for r in records) // max(len(records), 1)
                unique_avg = sum(r.get("unique_exemplars", 0) for r in records) // max(len(records), 1)
                print(f"  Avg common exemplars: {common_avg}")
                print(f"  Avg unique exemplars: {unique_avg}")
                # Verify the common prefix is indeed identical across samples
                first_q = records[0]["question"]
                prefix_len = 0
                for r in records[1:]:
                    # Find the divergence point
                    q = r["question"]
                    i = 0
                    while i < min(len(first_q), len(q)) and first_q[i] == q[i]:
                        i += 1
                    if prefix_len == 0:
                        prefix_len = i
                    else:
                        prefix_len = min(prefix_len, i)
                print(f"  Verified common prefix: {prefix_len:,} chars identical across all samples")
            else:
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
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {f}: {count} records ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
