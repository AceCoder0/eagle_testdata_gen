# Issue #1: 生成的数据集结果需要加上 answer 键

## 问题
jsonl 文件里每行需要有一个 `answer` 键，值为空字符串 `""`。

## 方案
在 `fewshot.py` 的 `build_one()` 和 `build_prefix_batch()` 返回 dict 中加 `"answer": ""`。

## 改动
- `fewshot.py` `build_one()` 返回字典加一行 `"answer": ""`
- `fewshot.py` `build_prefix_batch()` 返回字典加一行 `"answer": ""`
- `test_eagle.py` 更新相关测试断言

## 输出格式
```json
{"question": "Solve the following...", "answer": "", "question_token_len": 31996, ...}
```
