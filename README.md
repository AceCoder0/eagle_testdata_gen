# eagle_testdata_gen

从 AISBench 开源数学数据集中构造固定输入长度的 LLM 推理性能测试用例。

## 数据来源

基于 [AISBench 开源数据集表格](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/get_started/datasets.html) 中的**数学推理**类数据集：

| 数据集 | HuggingFace 来源 | 说明 |
|--------|-----------------|------|
| GSM8K | `openai/gsm8k` (main) | 小学数学应用题，~8,800 条，MIT 协议 |
| MATH-500 | `HuggingFaceH4/MATH-500` | 竞赛数学题，500 条 |
| AIME 2024 | `AI-MO/aimo-validation-aime` | 美国数学邀请赛 2024，30 题 |
| AIME 2025 | `yentinglin/aime_2025` | 美国数学邀请赛 2025，30 题 |
| MGSM | `juletxara/mgsm` (en) | 多语言小学数学，250 条 |
| DAPO-Math-17k | `open-r1/DAPO-Math-17k` | 数学推理 RL 评估，~17k 条 |

## 安装

```bash
git clone https://github.com/AceCoder0/eagle_testdata_gen.git
cd eagle_testdata_gen

# 安装依赖
pip install transformers datasets tqdm langid matplotlib scipy numpy
```

项目自带 DeepSeekR1 tokenizer，无需联网下载分词器。

## 快速开始

### 第一步：下载数据集（一次性，需要网络）

```bash
# 方式一：从 HuggingFace 下载（默认）
python download_datasets.py --output_pool math_pool.jsonl

# 方式二：从 OpenCompass 下载（国内服务器友好，无需 HuggingFace 访问）
python download_datasets.py --source opencompass --output_pool math_pool.jsonl
```

下载后产生 `math_pool.jsonl`，约 26,000+ 条 Q&A 对，每条包含 `question`、`answer`、`question_tokens`、`answer_tokens` 等字段。

> **注意**：`mgsm` 和 `dapo_math_17k` 暂无 OpenCompass 镜像，使用 `--source opencompass` 时会自动跳过。

### 第二步：构造测试用例

```bash
# 单个长度（32k = 32768 tokens）
python build_testdata.py --pool math_pool.jsonl --targets 32k --samples 500

# 多个长度一起生成（支持 k 后缀和纯数字混合）
python build_testdata.py \
    --pool math_pool.jsonl \
    --targets 4k,16k,32k,64k,200k \
    --samples 500 \
    --tolerance 0.05 \
    --output_dir ./output/

# 使用其他模型的 tokenizer（首次自动下载）
python build_testdata.py \
    --pool math_pool.jsonl \
    --targets 32k \
    --samples 500 \
    --tokenizer_path Qwen/Qwen2.5-7B-Instruct
```

### 第三步（可选）：生成 Prefix Cache 命中率测试数据

```bash
# 60% 共同前缀：同一个 batch 内所有样本前 60% 的 token 序列完全一致
python build_testdata.py \
    --pool math_pool.jsonl \
    --targets 32000 \
    --samples 500 \
    --prefix_rate 0.6 \
    --output_dir ./output/
```

输出文件：
```
output/
├── testdata_3k.jsonl         # 3.5k tokens × 500 条（独立样本）
├── testdata_16k.jsonl        # 16k tokens × 500 条
├── testdata_32k.jsonl        # 32k tokens × 500 条
├── testdata_32k_p0_6.jsonl   # 32k tokens × 500 条，60% 共同前缀
├── testdata_32k_p0_9.jsonl   # 32k tokens × 500 条，90% 共同前缀
├── testdata_64k.jsonl        # 64k tokens × 500 条
└── testdata_200k.jsonl       # 200k tokens × 500 条
```

## 输出数据结构

每条 JSON 记录包含以下字段：

```json
{
  "question": "<完整的 few-shot prompt 文本>",
  "question_token_len": 31996,
  "final_question": "<模型需要回答的最终问题>",
  "source": "fewshot_141_exemplars",
  "num_exemplars": 141,
  "target_tokens": 32000
}
```

| 字段 | 含义 |
|------|------|
| `question` | 完整的 prompt，直接作为模型的输入文本 |
| `question_token_len` | 实际 token 数（DeepSeekR1 tokenizer） |
| `final_question` | 最后一个问题——模型需要回答的目标题 |
| `source` | 来源标记 `fewshot_{N}_exemplars` |
| `num_exemplars` | few-shot 示例的数量 |
| `target_tokens` | 构造时的目标长度 |

## Prompt 结构分析

以一条 32k 的样本为例：

```
┌──────────────────────────────────────────────┐
│ Part 0: 指令文本 (1 条，69 字符)              │
│ "Solve the following math problems.          │
│  Show your step-by-step reasoning."          │
├──────────────────────────────────────────────┤
│ Part 1~141: Few-shot 示例 (141 对 Q&A)       │ ← 已知问题和答案，用于"占满"输入长度
│                                              │
│  Q: What is the smallest odd number with     │
│     four different prime factors?            │
│  A: 1155                                     │
│                                              │
│  Q: What is the tens digit of $5^{2005}$?    │
│  A: 2                                        │
│                                              │
│  Q: Each vertex of a regular octagon is...   │
│  A: Notice that the question's condition     │  ← gsm8k 的带步骤长答案
│     mandates all blues to go to reds...      │
│                                              │
│  ... (共 141 对，混合多个数据集) ...          │
├──────────────────────────────────────────────┤
│ Part 142: 最终问题 (1 条)                     │ ← 模型真正要回答的题
│                                              │
│  Q: cos18°+2cos36°+...+20cos360°=_______。   │
│  A: (空 -- 留给模型生成)                      │
└──────────────────────────────────────────────┘
```

**构造逻辑：**
- **Part 0**：固定的指令文本（~10-20 tokens）
- **Part 1~N**：从数据池中贪婪选取 Q&A 对，一条一条往上加，直到总长度逼近目标值。优先从不同数据集选取，保证多样性
- **Part N+1**：随机选一道题作为"最终问题"，答案留空

实际效果（500 条 × 32k 的统计）：

```
长度范围:  31,979 - 32,796 tokens
长度均值:  32,026 (偏离目标仅 +0.08%)
标准差:    128 tokens (0.4%)
在 ±5% 内: 500/500 (100%)
平均示例数: 132 对 Q&A
```

这种构造方式天然兼容性能测试——输入 token 数精确可控，且内容是真实的数学题，比随机字符串更接近实际推理场景。

## 构造策略

### 短输入 (3.5k)
少量 Q&A 示例（10-30 对）+ 最终问题即可达到目标。

### 长输入 (16k / 32k / 64k / 200k)
使用 **few-shot 拼接**：贪婪打包大量 Q&A 对，填充到目标长度。单个 prompt 可能包含上百到上千对 Q&A。

### 多样性保证

| 机制 | 说明 |
|------|------|
| **全局去重** | MD5 hash 去重，移除完全相同的题目 |
| **跨样本去重** | 每个长度下，最终问题不重复 |
| **样本内去重** | 同一个 prompt 内不会出现重复 Q&A |
| **分层采样** | 轮转从不同数据集选取示例，避免单一来源 |

## Prefix Cache 命中率测试

支持生成带有**共同前缀**的测试数据，用于评估推理框架（vLLM, SGLang）的 prefix cache / automatic prefix caching 效果。

### 核心思路

通过 `--prefix_rate` 参数控制所有样本的**共同前缀比例**。共同前缀部分使用**完全相同的 Q&A 对（相同顺序）**拼接，唯一后缀部分使用**不同的 Q&A 对**拼接。

```
样本 1: ┌─── 共同前缀 (60%) ───┐┌─ 唯一后缀 1 ─┐
样本 2: ┌─── 共同前缀 (60%) ───┐┌─ 唯一后缀 2 ─┐
样本 3: ┌─── 共同前缀 (60%) ───┐┌─ 唯一后缀 3 ─┐
        ^^^^^^ 完全相同 ^^^^^^    ^^^ 各不相同 ^^^
        → KV cache 命中          → 需重新计算
```

- **共同前缀**：所有样本的第 1 到第 N 个 token **完全一致** → 推理框架只需计算一次 KV cache
- **唯一后缀**：每个样本从 N+1 个 token 开始**不同** → 各自独立计算

### 使用方式

```bash
# prefix_rate=0.6: 60% 共同前缀，40% 唯一后缀
python build_testdata.py \
    --pool math_pool.jsonl \
    --targets 32000 \
    --samples 500 \
    --prefix_rate 0.6 \
    --output_dir ./output/

# prefix_rate=0.9: 90% 共同前缀，10% 唯一后缀  
python build_testdata.py \
    --pool math_pool.jsonl \
    --targets 32000 \
    --samples 500 \
    --prefix_rate 0.9 \
    --output_dir ./output/

# prefix_rate=0: 无共同前缀（等价于普通模式）
python build_testdata.py \
    --pool math_pool.jsonl \
    --targets 32000 \
    --samples 500 \
    --prefix_rate 0
```

### 实测效果（500 条 × 32k，prefix_rate=0.6）

```
Generated:        500/500 samples
Within tolerance:  500/500 (100%)
Length mean:       32,029 tokens
Length range:      31,975 - 32,763 tokens
Common exemplars:  82 (same Q&A pairs across all samples)
Unique exemplars:  76 (different Q&A pairs per sample)
Verified:          53,730 chars identical across ALL 500 samples
Actual hit rate:   60.0% of tokens are shared
```

### 输出数据额外字段

使用 prefix cache 模式时，每条记录额外包含：

| 字段 | 含义 |
|------|------|
| `prefix_rate` | 共同前缀比例（如 0.6） |
| `batch_id` | 批次 ID，同一批次共享共同前缀 |
| `common_exemplars` | 共同前缀中包含的 Q&A 对数 |
| `unique_exemplars` | 唯一后缀中包含的 Q&A 对数 |

## 长答案增强池（自然长输出引导）

当需要测试 **输出 token > 2K** 的 decode 性能时，原始数据池中很少有 `answer_tokens >= 2048` 的记录。`enrich_pool.py` 通过调用 MiniMax API 为精选难题生成极其详细的解答，产出一个"长答案增强池"，与 `--min_answer_tokens` 配合使用。

### 核心思路

```
math_pool.jsonl (~26,000 条，短/中答案)
    │
    ├── 步骤 1: enrich_pool.py 筛选难题
    │     ├── AIME 2024 + 2025: 全部 (~60 题)
    │     ├── MATH-500: answer_tokens top ~100
    │     └── DAPO-Math-17k: 随机 ~40
    │
    ├── 步骤 2: MiniMax API 生成详细解答
    │     每个题生成 1500-4000 token 的详细推理
    │
    └── 输出: math_pool_enriched.jsonl (~200 条，长答案)
```

### 使用方式

```bash
# 第一步：生成增强池（一次性，需 MiniMax API key）
python enrich_pool.py \
    --pool math_pool.jsonl \
    --output math_pool_enriched.jsonl \
    --api_key_file ~/llm_keys/minimax \
    --num_samples 200

# 支持断点续传
python enrich_pool.py \
    --pool math_pool.jsonl \
    --output math_pool_enriched.jsonl \
    --api_key_file ~/llm_keys/minimax \
    --resume math_pool_enriched.jsonl.checkpoint.json

# 第二步：用增强池构造测试数据（2k+ 目标输出）
python build_testdata.py \
    --pool math_pool.jsonl \
    --extra_pool math_pool_enriched.jsonl \
    --targets 32k \
    --samples 100 \
    --min_answer_tokens 2048 \
    --answer_style detailed
```

### 多样性保证

| 维度 | 机制 |
|------|------|
| **题目来源** | 分层采样 AIME + MATH + DAPO，覆盖代数/几何/数论/组合 |
| **答案风格** | 每个请求 temperature 在 0.7-0.9 随机波动 |
| **MoE 均衡** | 200 条来自 4 个数据集 + 不同数学分支，避免激活同一 expert |
| **最终问题** | `--min_answer_tokens` 过滤后，exemplar 和 final question 从不同 source 挑选 |

### `enrich_pool.py` 命令行参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pool` | `math_pool.jsonl` | 输入数据池 |
| `--output` | `math_pool_enriched.jsonl` | 输出（仅增强记录） |
| `--num_samples` | `200` | 目标生成数量 |
| `--api_key_file` | (必填) | MiniMax API key 文件路径 |
| `--api_base` | `https://api.minimaxi.com/v1` | API 地址（OpenAI 兼容） |
| `--model` | `MiniMax-M2.5` | 模型名 |
| `--temperature_low` | `0.7` | 最低温度（每个请求随机波动） |
| `--temperature_high` | `0.9` | 最高温度 |
| `--max_output_tokens` | `8192` | 单次最大输出 token 数 |
| `--tokenizer_path` | `./DeepSeekR1` | Tokenizer 路径 |
| `--seed` | `42` | 随机种子 |
| `--resume` | (空) | 从 checkpoint JSON 续传 |
| `--checkpoint_interval` | `10` | 每 N 条保存检查点 |

## 命令行参考

### `download_datasets.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_pool` | `math_pool.jsonl` | 输出 JSONL 文件路径 |
| `--source` | `huggingface` | 下载源：`huggingface` 或 `opencompass`（国内服务器推荐） |
| `--tokenizer_path` | `./DeepSeekR1` | DeepSeekR1 tokenizer 路径 |
| `--skip` | (空) | 跳过指定数据集，逗号分隔，如 `gsm8k_test,aime2025` |
| `--seed` | `42` | 随机种子，保证可复现 |

```bash
# 默认 HuggingFace 下载
python download_datasets.py --output_pool math_pool.jsonl

# OpenCompass 下载（适合国内服务器 / 无法访问 HuggingFace 的环境）
python download_datasets.py --source opencompass --output_pool math_pool.jsonl
```

### `build_testdata.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pool` | `math_pool.jsonl` | 数据池路径 |
| `--extra_pool` | (空) | 附加增强池 JSONL（来自 `enrich_pool.py`），与主池合并后使用 |
| `--targets` | `4k,16k,32k,64k,200k` | 目标 token 长度，逗号分隔。支持 `k` 后缀 (×1024，如 `32k`=32768) 或纯数字 |
| `--samples` | `100` | 每个长度生成条数 |
| `--prefix_rate` | `0.0` | 共同前缀比例 [0, 1]。0=独立样本；0.6=60% 共享前缀 |
| `--tolerance` | `0.05` | 允许偏差 (±5%) |
| `--min_qa_pairs` | `2` | 最少 few-shot 示例数 |
| `--answer_style` | `mixed` | 答案风格：`mixed`（混合）、`detailed`（优先长答案带推理）、`concise`（优先短答案） |
| `--min_answer_tokens` | `0` | 过滤数据池，只保留答案 ≥N tokens 的记录。同时影响 exemplar 和最终问题，用于引导模型自然长输出 |
| `--tokenizer_path` | `./DeepSeekR1` | Tokenizer 路径。支持本地路径或 HF 模型名（如 `deepseek-ai/DeepSeek-V3`） |
| `--output_dir` | `./output/` | 输出目录 |
| `--seed` | `42` | 随机种子 |

### 其他工具

```bash
# 查看数据分布
python data_dist.py --input_filename output/testdata_32k.jsonl

# 随机打乱
python shuffle.py --input_filename output/testdata_32k.jsonl \
                  --output_filename output/testdata_32k_shuffle.jsonl
```

## 旧版工具 (v1.x)

`create_dataset.py` 和 `data_augment.py` 仍保留在仓库中。旧版从通用文本中按长度筛选，使用中文同音替换/字符换位做增强。新版 (`build_testdata.py`) 建议替代旧版使用。

## 更新日志

### 2026-05-21

- **长答案增强池**：新增 `enrich_pool.py`，通过 MiniMax API 为难题生成详细长答案，支持断点续传。`build_testdata.py` 新增 `--extra_pool` 参数，配合 `--min_answer_tokens` 实现自然长输出引导

### 2026-05-15

- **OpenCompass 下载源**：新增 `--source opencompass`，从阿里云 OSS 镜像下载数据集，解决国内服务器无法访问 HuggingFace 的问题。支持 GSM8K、MATH-500、AIME 2024/2025。`mgsm` 和 `dapo_math_17k` 暂无镜像会自动跳过
- **种子复现**：`download_datasets.py` 新增 `--seed` 参数；`FewShotBuilder` 使用实例级独立 RNG，同一参数保证输出完全一致

### 2026-05-14

- **测试套件**：新增 `test_eagle.py`，36 个测试覆盖去重、few-shot 构建、prefix cache、shuffle、数据池加载、token 计数、复现性
- **Prefix Cache 测试**：新增 `--prefix_rate` 参数（0~1），支持生成带共同前缀的测试数据，用于评估 KV cache 命中率

### 2026-05-13

- **v2.0 重写**：从 AISBench 开源数学数据集下载 7 个数据源（GSM8K、MATH-500、AIME 2024/2025、MGSM、DAPO-Math-17k），通过 few-shot Q&A 拼接构造固定长度性能测试用例，支持 3.5k / 16k / 32k / 64k / 200k tokens

## License

MIT
