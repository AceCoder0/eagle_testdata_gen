# Proposal: MiniMax 长答案增强数据池

## 问题

当前 `--min_answer_tokens` 靠过滤原始数据池来控制输出长度引导，但原始数据集中很少有 `answer_tokens > 2048` 的记录（AIME 答案只有 ~2 tokens，GSM8K ~100 tokens，MATH-500 ~200 tokens）。即使设 `--min_answer_tokens 2048`，过滤后可用记录极少，few-shot 构造无法完成。

**核心矛盾**：需要长答案来引导模型自然长输出，但数据池里没有长答案。

## 方案

用 MiniMax 模型为一批难题生成极其详细的 step-by-step 答案，产出一个"长答案增强池"。之后构造测试数据时，`--min_answer_tokens` 自然从增强池中挑选记录。

### 整体流程

```
math_pool.jsonl (~26,000 条，短/中答案)
    │
    ├── 步骤 1: 筛选难题 (~200 条)
    │     ├── AIME 2024 + 2025: 全部 (~60 题, 最难)
    │     ├── MATH-500: answer_tokens top ~100
    │     └── DAPO-Math-17k: 随机 ~40 (多样性)
    │
    ├── 步骤 2: 逐个调用 MiniMax，prompt 引导极其详细的推理
    │     输出: 每个题 ~1500-4000 tokens 的详细解答
    │
    └── 输出: math_pool_enriched.jsonl (~200 条，长答案)
```

### 使用方式

```bash
# 步骤 1: 生成增强池（一次性，约 30-60 分钟，取决于 API 速率）
python enrich_pool.py \
    --pool math_pool.jsonl \
    --output math_pool_enriched.jsonl \
    --api_key_file ~/llm_keys/minimax \
    --num_samples 200

# 步骤 2: 用增强池构造测试数据（2k+ 目标输出）
python build_testdata.py \
    --pool math_pool.jsonl \
    --extra_pool math_pool_enriched.jsonl \
    --targets 32k \
    --samples 100 \
    --min_answer_tokens 2048
```

### `enrich_pool.py` 设计

**CLI 参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pool` | `math_pool.jsonl` | 输入数据池 |
| `--output` | `math_pool_enriched.jsonl` | 输出（仅增强记录） |
| `--num_samples` | `200` | 目标生成数量 |
| `--api_key_file` | (必填) | MiniMax API key 文件路径 |
| `--api_base` | `https://api.minimaxi.com/v1` | API 地址（OpenAI 兼容） |
| `--model` | `MiniMax-M1` | 模型名 |
| `--temperature` | `0.7-0.9` | 每个请求随机波动，增加多样性 |
| `--max_output_tokens` | `4096` | 单次最大输出 token 数 |
| `--seed` | `42` | 随机种子 |
| `--checkpoint_interval` | `10` | 每 N 条保存一次检查点 |

**题目筛选策略（Diversity-aware selection）**：

```
来源分层:
  AIME 2024        → 全部 30 题    (最难竞赛题)
  AIME 2025        → 全部 30 题    (最新竞赛题)
  MATH-500         → 抽 100 题      (按 answer_tokens 排序取 top-100，确保题目本身复杂)
  DAPO-Math-17k    → 抽 40 题       (随机采样，保证领域多样性)

总数: ~200 题，覆盖代数、几何、数论、组合等
```

**Prompt 模板**：

```
You are a mathematics professor. Please solve the following problem with
EXTREMELY detailed, step-by-step reasoning. Your solution should be
comprehensive enough to fill several pages.

For each step:
- State the theorem or concept being applied
- Explain WHY this approach is correct
- Show all algebraic manipulations in full detail
- Verify intermediate results
- Consider edge cases
- If multiple approaches exist, briefly mention them

Make your answer as long and detailed as possible while remaining
mathematically rigorous.

Problem: {question}
```

**输出格式**：与 `math_pool.jsonl` 完全相同的字段结构，确保下游兼容：

```json
{
  "id": "enriched_aime2024_0",
  "question": "Find the number of ordered triples (a,b,c)...",
  "answer": "Let's approach this step by step...\n\nStep 1:...",
  "source": "enriched_aime2024",
  "question_tokens": 45,
  "answer_tokens": 2847,
  "total_tokens": 2892
}
```

**容错设计**：

- 断点续传：每 10 条保存 checkpoint JSON，中断后 `--resume` 继续
- API 重试：429/5xx 指数退避重试 3 次
- 超时控制：单次请求超时 120s

### 集成方式

新增 `--extra_pool` 参数到 `build_testdata.py`，加载额外的高质量长答案记录，合并到主池：

```python
# build_testdata.py
parser.add_argument("--extra_pool", type=str, default=None,
                    help="Additional pool with long-answer records (from enrich_pool.py)")

# 合并逻辑
pool = load_pool(pool_path)
if args.extra_pool:
    extra = load_pool(args.extra_pool)
    pool.extend(extra)
    # 重新去重
    pool = exact_dedup(pool, key="question")
```

这样 `--min_answer_tokens 2048` 过滤后，增强池中的长答案记录自然被选中，用作 few-shot exemplar 和 final question。

### 多样性保证

| 维度 | 机制 |
|------|------|
| **题目来源** | 分层采样 AIME + MATH + DAPO，覆盖不同难度和题型 |
| **答案风格** | 每个请求 temperature 在 0.7-0.9 随机波动 |
| **最终问题** | `--min_answer_tokens` 过滤后，exemplar 和 final question 从不同 source 挑选（已有 stratified 逻辑） |
| **MoE 负载** | 200 条来自 4 个不同数据集 + 不同数学分支（代数/几何/数论/组合），激活不同 expert |

### 工作量估算

| 文件 | 改动量 |
|------|--------|
| `enrich_pool.py` (新) | ~250 行（选择逻辑 + API 调用 + checkpoint） |
| `build_testdata.py` | +15 行（`--extra_pool` 参数） |
| `test_eagle.py` | +5 个测试（enrich pool 格式、合并逻辑、过滤后分布） |
| `README.md` | +1 节（增强池使用说明） |

### 限制

- **API 成本**：200 题 × 4096 tokens ≈ 800K output tokens，MiniMax 计费
- **耗时**：串行调用约 20-40 分钟（假设 10s/请求）
- **答案质量依赖模型**：MiniMax 生成的答案质量决定引导效果
- **不是 guarantee**：仍然靠 few-shot 引导，不能 100% 保证模型输出到指定长度
- **MiniMax API 兼容性**：假设 OpenAI 兼容接口，需要验证具体 base URL 和 model name
