#!/usr/bin/env python3
"""Analyze QA pair token length distribution and packing coverage.

Usage:
  python analyze_pool.py --pool math_pool_enriched.jsonl
  python analyze_pool.py --pool math_pool.jsonl
  python analyze_pool.py --pool math_pool_enriched.jsonl --targets 1500,3500,8000,16000,32000
"""
import json
import argparse
import math
from collections import defaultdict


def load_pool(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def histogram(values, bins=20):
    """Print a text histogram."""
    if not values:
        return
    lo, hi = min(values), max(values)
    if lo == hi:
        print(f"  All values = {lo}")
        return
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    max_count = max(counts)
    bar_width = 40
    for i in range(bins):
        b_lo = int(lo + i * step)
        b_hi = int(lo + (i + 1) * step)
        bar = "█" * max(1, int(counts[i] / max_count * bar_width))
        print(f"  [{b_lo:>6d}, {b_hi:>6d}): {counts[i]:>4d} {bar}")


def gaps_analysis(values):
    """Analyze gaps in the sorted value distribution — larger gaps = worse coverage."""
    if len(values) < 2:
        return
    sorted_vals = sorted(set(values))
    gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
    gaps.sort(reverse=True)
    print(f"\n  Unique values: {len(sorted_vals)}")
    print(f"  Max gap between consecutive values: {max(gaps):,}")
    print(f"  Top-10 gaps: {gaps[:10]}")
    # Coverage: what fraction of the range [min, max] is covered?
    total_range = sorted_vals[-1] - sorted_vals[0]
    covered = sum(gap <= 10 for gap in gaps)
    print(f"  Gaps <= 10 tokens: {covered}/{len(gaps)} ({100*covered/len(gaps):.1f}%)")
    return gaps


def subset_sum_coverage(values, target_ranges):
    """Simulate: with these token counts, what targets are achievable?

    For each target T, check if any 2-item combination lands within 5% of T.
    This is a proxy for how well the distribution "covers" target lengths.
    """
    unique_vals = sorted(set(values))
    print(f"\n  --- Subset-sum coverage (2-item combos) ---")
    for t_lo, t_hi, label in target_ranges:
        count = 0
        total_combos = 0
        for i, a in enumerate(unique_vals):
            if a > t_hi:
                break
            for b in unique_vals[i:]:
                total_combos += 1
                if t_lo <= a + b <= t_hi:
                    count += 1
                if a + b > t_hi:
                    break
        coverage = 100 * count / max(total_combos, 1)
        print(f"  Target ~{label:>6s}: {count}/{total_combos} combos in ±5% "
              f"({coverage:.1f}% coverage)")


def target_capacity_analysis(values, targets):
    """For each target T, estimate how many distinct prompts can be built.

    This checks: given the pool's token distribution, what's the max number
    of non-overlapping few-shot prompts at each target length?
    """
    from collections import Counter
    sorted_vals = sorted(values)
    total_tokens = sum(sorted_vals)
    print(f"\n  --- Capacity estimate ---")
    for t in targets:
        # Greedy estimate: how many times can we hit ~T using available items?
        remaining = list(sorted_vals)
        count = 0
        used = 0
        budget = t
        for v in remaining:
            if v <= budget:
                budget -= v
                used += v
                if budget <= t * 0.05:
                    count += 1
                    budget = t
        print(f"  Target {t:>8,}: ~{count} prompts possible "
              f"(total pool tokens: {total_tokens:,})")


def main():
    parser = argparse.ArgumentParser(description="Analyze QA pair token length distribution")
    parser.add_argument("--pool", default="math_pool_enriched.jsonl")
    parser.add_argument("--targets", default="1500,3500,8000,16000,32000,64000,200000",
                        help="Target token lengths to analyze coverage for")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    targets = []
    for t in args.targets.split(","):
        t = t.strip()
        if t.lower().endswith("k"):
            targets.append(int(float(t[:-1]) * 1024))
        else:
            targets.append(int(t))

    pool = load_pool(args.pool)
    print(f"Pool: {args.pool} ({len(pool)} records)")

    # 1. Total tokens (QA pair size — this is what gets packed)
    total_tokens = [r["total_tokens"] for r in pool]
    print(f"\n{'='*50}")
    print(f"QA total tokens (exemplar granularity)")
    print(f"  Count: {len(total_tokens)}")
    print(f"  Min: {min(total_tokens):,}")
    print(f"  Max: {max(total_tokens):,}")
    print(f"  Mean: {sum(total_tokens)//len(total_tokens):,}")
    print(f"  Median: {sorted(total_tokens)[len(total_tokens)//2]:,}")
    print(f"  Sum: {sum(total_tokens):,}")
    pct = [10, 25, 50, 75, 90, 95, 99]
    sv = sorted(total_tokens)
    print(f"  Percentiles: " + ", ".join(f"P{p}={sv[int(len(sv)*p/100)]:,}" for p in pct))
    print(f"\n  Distribution:")
    histogram(total_tokens, bins=args.bins)

    # 2. Answer tokens
    answer_tokens = [r["answer_tokens"] for r in pool]
    print(f"\n{'='*50}")
    print(f"Answer tokens")
    print(f"  Min: {min(answer_tokens):,}")
    print(f"  Max: {max(answer_tokens):,}")
    print(f"  Mean: {sum(answer_tokens)//len(answer_tokens):,}")
    print(f"  Distribution:")
    histogram(answer_tokens, bins=args.bins)

    # 3. Per-source breakdown
    print(f"\n{'='*50}")
    print(f"Per-source distribution")
    by_source = defaultdict(list)
    for r in pool:
        by_source[r.get("source", "?")].append(r["total_tokens"])
    for src in sorted(by_source):
        vals = by_source[src]
        print(f"  {src}: {len(vals)} records, "
              f"min={min(vals):,}, max={max(vals):,}, "
              f"mean={sum(vals)//len(vals):,}, median={sorted(vals)[len(vals)//2]:,}")

    # 4. Gap analysis — key metric for packing quality
    print(f"\n{'='*50}")
    print(f"Coverage gap analysis")
    gaps = gaps_analysis(total_tokens)

    # 5. Uniformity metrics
    print(f"\n{'='*50}")
    print(f"Uniformity analysis")
    counts = defaultdict(int)
    lo, hi = min(total_tokens), max(total_tokens)
    step = max(1, (hi - lo) // args.bins)
    for v in total_tokens:
        bucket = (v - lo) // step
        counts[bucket] += 1
    bucket_counts = list(counts.values())
    if bucket_counts:
        mean_bucket = sum(bucket_counts) / len(bucket_counts)
        variance = sum((c - mean_bucket) ** 2 for c in bucket_counts) / len(bucket_counts)
        cv = math.sqrt(variance) / mean_bucket if mean_bucket > 0 else float("inf")
        print(f"  Buckets: {len(bucket_counts)}, bucket size: {step:,} tokens")
        print(f"  Mean items/bucket: {mean_bucket:.1f}")
        print(f"  StdDev: {math.sqrt(variance):.1f}")
        print(f"  Coefficient of variation: {cv:.2f} (0=perfectly uniform, lower=more uniform)")
        empty_buckets = sum(1 for c in bucket_counts if c == 0)
        print(f"  Empty buckets: {empty_buckets}/{len(bucket_counts)} "
              f"({100*empty_buckets/len(bucket_counts):.1f}%)")
        if cv < 0.5:
            print(f"  Assessment: Very uniform — excellent for packing")
        elif cv < 1.0:
            print(f"  Assessment: Moderately uniform — good for packing")
        else:
            print(f"  Assessment: Skewed — consider adding items in sparse ranges")

    # 6. Subset-sum coverage
    target_ranges = [(int(t * 0.95), int(t * 1.05), f"{t//1024}k" if t >= 1024 else str(t))
                     for t in targets if t <= max(total_tokens) * 2]
    subset_sum_coverage(total_tokens, target_ranges)

    # 7. Comparison: enriched pool vs original pool
    print(f"\n{'='*50}")
    print(f"Cross-pool comparison")
    try:
        orig = load_pool("math_pool.jsonl")
        orig_tt = [r["total_tokens"] for r in orig]
        print(f"\n  math_pool.jsonl ({len(orig)} records):")
        print(f"    QA tokens: min={min(orig_tt):,}, max={max(orig_tt):,}, "
              f"mean={sum(orig_tt)//len(orig_tt):,}")
        print(f"    Total pool tokens: {sum(orig_tt):,}")

        print(f"\n  math_pool_enriched.jsonl ({len(pool)} records):")
        print(f"    QA tokens: min={min(total_tokens):,}, max={max(total_tokens):,}, "
              f"mean={sum(total_tokens)//len(total_tokens):,}")
        print(f"    Total pool tokens: {sum(total_tokens):,}")

        # Which pool has better coverage at each target?
        print(f"\n  --- Per-target gap near each target ---")
        for t in targets:
            # Find the gap in each pool around target t
            orig_nearby = sorted([v for v in orig_tt if abs(v - t) < t * 0.3])
            enr_nearby = sorted([v for v in total_tokens if abs(v - t) < t * 0.3])
            print(f"  Target {t:>8,}:")
            if orig_nearby:
                print(f"    Original: {len(orig_nearby)} items in ±30%, "
                      f"range [{min(orig_nearby):,}, {max(orig_nearby):,}]")
            else:
                print(f"    Original: 0 items in ±30%")
            if enr_nearby:
                print(f"    Enriched: {len(enr_nearby)} items in ±30%, "
                      f"range [{min(enr_nearby):,}, {max(enr_nearby):,}]")
            else:
                print(f"    Enriched: 0 items in ±30%")
    except FileNotFoundError:
        print("  (math_pool.jsonl not found, skipping comparison)")


if __name__ == "__main__":
    main()
