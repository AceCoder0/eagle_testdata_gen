# Issue #2: 支持按照用户指定的模型来计算 token

## 问题
用户需要指定不同模型的 tokenizer 来计算 token 数，确保生成数据的 token 长度与目标模型一致。

## 方案
**不改代码**。`build_testdata.py` 已有 `--tokenizer_path` 参数，`AutoTokenizer.from_pretrained()` 同时支持本地路径和 HF 模型 ID。

只需在 README 中补充文档说明。

## 文档补充
```bash
# 使用本地 tokenizer
python build_testdata.py --tokenizer_path ./DeepSeekR1 --targets 32000 --samples 100

# 使用 HuggingFace 模型名（首次自动下载）
python build_testdata.py --tokenizer_path deepseek-ai/DeepSeek-V3 --targets 32000 --samples 100
python build_testdata.py --tokenizer_path Qwen/Qwen2.5-7B-Instruct --targets 32000 --samples 100
```
