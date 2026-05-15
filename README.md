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
python download_datasets.py --output_pool math_pool.jsonl
```

下载后产生 `math_pool.jsonl`，约 26,000+ 条 Q&A 对，每条包含 `question`、`answer`、`question_tokens`、`answer_tokens` 等字段。

### 第二步：构造测试用例

```bash
# 单个长度
python build_testdata.py --pool math_pool.jsonl --targets 32000 --samples 500

# 多个长度一起生成
python build_testdata.py \
    --pool math_pool.jsonl \
    --targets 3500,16000,32000,64000,200000 \
    --samples 500 \
    --tolerance 0.05 \
    --output_dir ./output/
```

输出文件：
```
output/
├── testdata_3k.jsonl      # 3.5k tokens × 500 条
├── testdata_16k.jsonl     # 16k tokens × 500 条
├── testdata_32k.jsonl     # 32k tokens × 500 条
├── testdata_64k.jsonl     # 64k tokens × 500 条
└── testdata_200k.jsonl    # 200k tokens × 500 条
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

## 命令行参考

### `download_datasets.py`

```bash
python download_datasets.py \
    --output_pool math_pool.jsonl \
    --tokenizer_path ./DeepSeekR1 \
    --skip gsm8k_test,aime2025    # 跳过某些数据集
```

### `build_testdata.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pool` | `math_pool.jsonl` | 数据池路径 |
| `--targets` | `3500,16000,32000,64000,200000` | 目标 token 长度，逗号分隔 |
| `--samples` | `100` | 每个长度生成条数 |
| `--tolerance` | `0.05` | 允许偏差 (±5%) |
| `--min_qa_pairs` | `2` | 最少 few-shot 示例数 |
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

## License

MIT
