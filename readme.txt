eagle_testdata_gen -- 数学数据集性能测试用例生成工具
========================================================

从 AISBench 开源数学数据集中抽取/构造固定输入长度的测试用例，
用于 LLM 推理性能压测。

支持的目标输入长度：3.5k, 16k, 32k, 64k, 200k tokens

数据来源：AISBench 开源数据集表格中的数学推理类数据集
  - gsm8k (openai/gsm8k)
  - math (lighteval/MATH)
  - aime2024 / aime2025 / aime2026
  - mgsm (juletxara/mgsm)
  - dapo-math-17k (open-r1/DAPO-Math-17k)


===== 使用方式 =====

1. 安装依赖
   pip install transformers datasets tqdm langid matplotlib scipy numpy

2. 下载数据集（一次性，需要网络）
   python download_datasets.py --output_pool math_pool.jsonl
   
   可选参数：
     --tokenizer_path ./DeepSeekR1   # 分词器路径
     --skip gsm8k,aime2024           # 跳过某些数据集

3. 构造测试用例
   python build_testdata.py \
       --pool math_pool.jsonl \
       --targets 3500,16000,32000,64000,200000 \
       --samples 100 \
       --tolerance 0.05 \
       --output_dir ./output/

   参数说明：
     --pool          步骤2产生的数据池
     --targets       目标输入长度（tokens），逗号分隔
     --samples       每个长度生成多少条（默认100）
     --tolerance     允许偏差比例（默认0.05 = ±5%）
     --min_qa_pairs  最少 few-shot 示例数（默认2）
     --output_dir    输出目录
     --seed          随机种子（默认42）

4. 查看分布
   python data_dist.py --input_filename ./output/testdata_16k.jsonl

5. 随机打乱
   python shuffle.py --input_filename ./output/testdata_16k.jsonl \
                     --output_filename ./output/testdata_16k_shuffle.jsonl


===== 构造策略 =====

短输入（3.5k）：少量 Q&A 示例 + 最终问题即可达到目标长度。
长输入（16k, 32k, 64k, 200k）：使用 few-shot 方式，将多条 Q&A
  示例拼接在一起，组成一个长 prompt，最后一个问题作为模型需要
  回答的"最终问题"。

Few-shot prompt 格式：
  Solve the following math problems. Show your step-by-step reasoning.

  Q: {question_1}
  A: {answer_1}

  Q: {question_2}
  A: {answer_2}

  ...

  Q: {final_question}
  A:

多样性保证：
  - 全局去重：移除完全相同的题目
  - 跨样本去重：每个长度下最终问题不重复
  - 样本内去重：同一个 prompt 内不出现重复的 Q&A
  - 分层采样：尽量从不同数据集选取示例


===== 输出格式 =====

JSONL 文件，每行一个 JSON 对象：
  {
    "question": "<完整的 few-shot prompt>",
    "question_token_len": 16234,
    "final_question": "<模型需要回答的问题>",
    "source": "fewshot_45_exemplars",
    "num_exemplars": 45,
    "target_tokens": 16000
  }

与 AISBench 开源数据集格式兼容（question + question_token_len 字段）。


===== 其他工具 =====

create_dataset.py / data_augment.py -- 旧版工具（v1.x），
  用于从 merged_dataset.jsonl 中按长度筛选+中文增强。
  新版建议使用 build_testdata.py + download_datasets.py。


===== 更新记录 =====

v2.0
  - 新增 download_datasets.py：从 HuggingFace 下载 AISBench 数学数据集
  - 新增 build_testdata.py：基于 few-shot 构造指定长度的测试用例
  - 新增 fewshot.py：few-shot prompt 构造算法
  - 新增 dedup.py：去重工具
  - 支持 200k tokens 长序列
  - 数据来源从通用文本迁移到开源数学数据集

v1.3 - v1.0
  见旧版 readme
