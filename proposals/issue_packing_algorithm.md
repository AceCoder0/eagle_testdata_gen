# Design Doc: Few-Shot Prompt Packing Algorithm

## Problem Statement

Given a pool of Q&A pairs with known token counts and a target total token length
`T`, select a subset of Q&A pairs (exemplars) plus one final question to
construct a few-shot prompt whose total token count is as close as possible to `T`.

This is exactly the **Subset Sum Problem**: given a set of items with weights
`w_i` (token counts), find a subset whose sum is as close as possible to a
target `T`. Subset sum is **NP-hard** — no polynomial-time algorithm is known
to solve it optimally for all inputs.

### Additional constraints

- **Minimum exemplar count** (`min_qa_pairs`): at least 2 Q&A pairs required for
  meaningful few-shot format.
- **Source diversity**: exemplars should be drawn from different datasets
  (AIME, GSM8K, MATH-500, etc.) to avoid homogeneous prompts.
- **Answer style bias** (`answer_style`): optionally prefer detailed or concise
  exemplars for the final question.
- **Pool separation**: exemplars come from the full (unfiltered) pool, while the
  final question comes from the filtered pool (`min_answer_tokens`) to guide
  output length independently of input length control.

### Input sizes

| Parameter | Typical value |
|-----------|--------------|
| Pool size | 500–10,000 records |
| Target tokens | 1,500–200,000 |
| Exemplar QA tokens | 10–10,000 (high variance) |
| Samples per target | 50–500 |

## Current Algorithm: 4-Phase Greedy + Local Search

### Phase 1: Greedy Stratified Fill

```
budget = target - instruction_tokens - final_q_tokens
exemplars = []
used_ids = {final_q_id}

while budget > target * tolerance:
    candidate = pick_stratified(used_ids)  # from unused source
    if candidate.tokens <= budget + slack:
        exemplars.append(candidate)
        budget -= candidate.tokens
        used_ids.add(candidate.id)
```

**Strategy**: Pick one exemplar from each source dataset in round-robin fashion
(stratified sampling). Accept any exemplar that fits within `budget + 5% * target * 0.5`
(slight overshoot tolerance). This prioritizes source diversity over perfect
packing.

**Time**: O(pool_size) per iteration. Each `_pick_stratified` scans the pool
for unused records from a specific source.

### Phase 2: Force Minimum Pairs

```
while len(exemplars) < min_qa_pairs:
    candidates = sorted(unused_from_full_pool, key=token_count ASC)
    smallest = candidates[0]
    if current_total + smallest.tokens > upper_bound:
        break  # accept fewer exemplars rather than overshoot
    exemplars.append(smallest)
```

**Strategy**: If Phase 1 didn't collect enough exemplars (e.g., target is very
small, only 1-2 large Q&A pairs fit), force-add the smallest available ones.
Guard against overshoot by checking `upper_bound`.

**Why smallest-first**: Minimizes overshoot when operating near the target.

### Phase 3: Gap Filling

```
if budget > 0:
    for c in sorted(unused_from_full_pool, key=token_count ASC):
        if c.tokens <= budget:
            exemplars.append(c)
            budget -= c.tokens
```

**Strategy**: Fill remaining budget with the smallest available Q&A pairs.
This is a "coin change" greedy approach — use the smallest denominations to
make exact change.

**Optimality**: This is optimal for gap filling because (a) we want to get as
close to the target as possible, and (b) using smaller items gives finer
granularity. This is correct because we're always below target at this point
and want to minimize the remaining gap.

### Phase 4: Hill-Climbing Local Search

```
def local_search(exemplars, target, final_q_tokens):
    current = total_tokens(exemplars)
    best_error = |current - target|

    repeat until no improvement (max 10 iterations):
        # Strategy 1: Try removing one exemplar
        for each e in exemplars:
            if |current - e.tokens - target| < best_error:
                remove e and accept

        # Strategy 2: Try swapping one for a better-fit unused one
        for each e in exemplars:
            for each unused candidate c:
                if |current - e.tokens + c.tokens - target| < best_error:
                    swap e with c and accept
```

**Strategy**: After the greedy phases produce a solution, fine-tune it with
local search. Two moves are considered:
1. **Remove**: Helps when we overshot the target (a large exemplar pushed us over).
2. **Swap**: Replaces one exemplar with another that gets closer to target.

**Why not more strategies**: Empirically, remove and swap cover the vast
majority of cases. More complex moves (remove 1, add 2) add O(n³) complexity
with diminishing returns.

**Convergence**: Guaranteed in at most `len(exemplars)` iterations since each
step strictly reduces error. Capped at 10 iterations for performance.

## Why Not Dynamic Programming?

Classic DP for subset sum: `O(n * T)` time and space, where `n` is pool size
and `T` is target.

For our use case:
- `n ≈ 10,000` (full pool)
- `T ≤ 200,000` (max target)

**DP would require ~2 billion operations and ~200 MB of memory per prompt.**
With 100+ prompts to generate, this is prohibitive.

Additionally, DP can't easily incorporate:
- Source diversity constraints (stratified sampling)
- Answer style preferences
- Minimum exemplar count

## Why the Greedy Approach Works Well

The key insight is that we have **many small items**:

| Item size (tokens) | Count in enriched pool |
|-------------------|----------------------|
| 10–100 | ~330 (AIME short answers, small GSM8K) |
| 100–500 | ~260 (GSM8K, Math-500) |
| 500–2,000 | 0 (gap — filled by original pool) |
| 2,000–10,000 | ~182 (enriched long answers) |

With abundant small items (Phase 3 gap filling) and the enriched pool providing
large items, the greedy algorithm can achieve any target within ~1% error.

The **Phase 3 gap-filling step** is particularly effective because:
1. It uses smallest-first ordering (optimal for coin-change with unlimited
   supply of small denominations)
2. The pool has items as small as 10 tokens, giving 0.03% granularity at 32K

## Empirical Results

Tested with enriched pool (482 records: 182 long + 300 short):

| Target | Samples | Within 5% | Mean Error | Range |
|--------|---------|-----------|------------|-------|
| 1,500 | 10 | 10/10 | 0.4% | 1,485–1,501 |
| 3,500 | 10 | 10/10 | 0.0% | 3,494–3,501 |
| 8,000 | 10 | 10/10 | 0.1% | 7,975–8,001 |
| 16,000 | 10 | 10/10 | 0.2% | 15,785–16,001 |
| 32,000 | 10 | 10/10 | 0.0% | 31,959–32,001 |

All samples within ±5% tolerance across all target sizes. Mean error under 0.5%.

## Future Improvements

### 1. Simulated Annealing

For very tight tolerances (<1%), replace the deterministic hill-climbing in
Phase 4 with simulated annealing. This accepts occasional uphill moves to escape
local minima. Given current results (mean error 0.0–0.4%), this is unnecessary
for now.

### 2. Beam Search for Phase 1

Instead of greedy single-path, maintain top-k partial solutions during Phase 1.
This would explore more of the search space at the cost of k× runtime. Useful
when source diversity requirements conflict with tight length targets.

### 3. DP for Small Targets

For targets < 5,000 tokens, DP becomes feasible (50M operations, negligible
memory). Could switch to exact DP when `n * T < 10^7`.

### 4. Item Pruning

Pre-filter the pool to only include items that could possibly fit:
`token_count < target * 1.05`. This reduces the search space significantly for
small targets while maintaining correctness.

## Trade-offs Summary

| Approach | Time | Accuracy | Diversity | Complexity |
|----------|------|----------|-----------|------------|
| Pure greedy (3-phase) | O(n log n) | ±3–5% | Good | Low |
| Greedy + local search (current) | O(n log n + k·n) | ±0.5% | Good | Medium |
| Beam search (k=5) | O(k·n log n) | ±0.2% | Better | Medium-High |
| DP (small targets only) | O(n·T) | Optimal | None | High |
| Full ILP solver | Exponential | Optimal | Configurable | Very High |

The current 4-phase approach (greedy + local search) hits the sweet spot:
excellent accuracy with linear runtime, while preserving source diversity and
answer style preferences.
