# Issue #5: 不靠 ignore_eos，让模型自然输出到目标长度

## 问题

性能测试需要测指定输出长度（如 1024、2048 tokens）下的 decode 速度。目前靠 `ignore_eos=True` 强制生成，但实际场景中模型会主动输出 EOS 停下。需要让 **prompt 本身** 引导模型自然输出到目标长度以上。

## 分析

模型输出长度主要由 few-shot 示例的**答案风格**决定：

| 数据集 | 答案风格 | 典型 answer_tokens |
|--------|----------|-------------------|
| GSM8K | 分步推理 "Let's think step by step..." | ~100 tokens |
| MATH-500 | 详细推导 | ~200 tokens |
| DAPO-Math-17k | 完整解析 | ~200 tokens |
| AIME 2024/2025 | 仅最终答案 "25" | ~2 tokens |
| MGSM | 仅数字答案 | ~5 tokens |

如果 few-shot 示例中夹杂大量短答案（AIME），模型会"学"到简短回答，提前输出 EOS，达不到目标输出长度。

## 方案

新增 `--min_answer_tokens` 参数，过滤数据池，只保留 `answer_tokens >= N` 的 Q&A 对作为 few-shot 示例来源。配合已有的 `--answer_style detailed`，双向保证。

### 参数设计

```bash
# 要求 few-shot 示例的答案至少 512 tokens → 引导模型输出 512+ tokens
python build_testdata.py \
    --pool math_pool.jsonl \
    --targets 32k \
    --samples 100 \
    --min_answer_tokens 512 \
    --answer_style detailed

# 目标输出 1024 tokens，用 1024+ token 的示例
python build_testdata.py \
    --pool math_pool.jsonl \
    --targets 32k \
    --samples 100 \
    --min_answer_tokens 1024

# 目标输出 2048 tokens
python build_testdata.py \
    --pool math_pool.jsonl \
    --targets 32k \
    --samples 100 \
    --min_answer_tokens 2048
```

### 改动点

**1. `fewshot.py` — `FewShotBuilder.__init__`**

新增 `min_answer_tokens` 参数，初始化时过滤 `self.pool`。

**关键设计**：`self.pool` 在 `FewShotBuilder` 中是**唯一数据源**——既用于选取 few-shot exemplar，也用于选取 final question（最终要模型回答的题）。过滤一次，两边同时生效：

```
self.pool (全部 ~26000 条)
    │
    ├── 过滤: answer_tokens >= min_answer_tokens
    │
    ▼
self.pool (过滤后 ~8000 条)
    │
    ├── 选 few-shot exemplar ── 只有长答案的 Q&A 对
    └── 选 final question   ── 只有"需要长答案"的复杂题
```

这样既能保证 **exemplar 示范** 是详细推理，又能保证 **最终问题本身** 也是需要长篇回答的复杂题——不会出现"exemplar 都是长答案，但最终题是 AIME 那种一句话答案"的矛盾。

```python
def __init__(self, pool, tokenizer, seed=42, answer_style="mixed",
             min_answer_tokens=0):
    if min_answer_tokens > 0:
        self.pool = [r for r in pool if r.get("answer_tokens", 0) >= min_answer_tokens]
        if len(self.pool) == 0:
            raise ValueError(f"No records with answer_tokens >= {min_answer_tokens}")
    else:
        self.pool = pool
    # ... rest unchanged (pre-compute tokens, build _by_source, etc.)
```

**2. `build_testdata.py`**

新增 `--min_answer_tokens` CLI 参数，传递给 `FewShotBuilder`。

### 效果验证

预期数据池记录数变化（基于各数据集典型分布）：

| min_answer_tokens | 可用记录 | 主要来源 |
|-------------------|---------|---------|
| 0 (默认) | ~26,000 | 全部 |
| 100 | ~20,000 | GSM8K + MATH + DAPO |
| 512 | ~8,000 | MATH + DAPO + 部分 GSM8K |
| 1024 | ~3,000 | 少量 MATH + DAPO |
| 2048 | ~500 | 极少数长答案 |

如果 `min_answer_tokens` 设置过高导致可用记录不足，`build_many` 会打印 warning 并生成尽可能多的样本。

### 与 `--answer_style` 的配合

| 配置 | exemplar 效果 | final question 效果 |
|------|-------------|-------------------|
| `--min_answer_tokens 512 --answer_style detailed` | 只用 512+ token 的长推理示例，优先最长的 | 只选需要长答案的复杂题 |
| `--min_answer_tokens 256 --answer_style mixed` | 中等过滤，保留多样性 | 中等复杂度题目 |
| 不设置（默认） | 混合长短答案，可能引导模型输出短答案 | 可能选到 AIME 那种简单答案的题 |

### 限制

- 不能**保证**模型输出一定达到目标长度，只能通过 exemplar + 题目复杂度双维度**引导**
- 如果 `min_answer_tokens` 过高，pool 可能不够大（exemplar 不够用 + final question 不够用），`build_many` 会打印 warning
- 经验法则：`min_answer_tokens` 设为目标输出长度的 30%~50%（如目标输出 2048 tokens，设 `min_answer_tokens 512~1024`）
