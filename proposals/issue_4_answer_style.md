# Issue #4: fewshot 的答案是怎么来的？有解析过程吗？

## 问题
不同数据集的答案风格差异大：GSM8K/MATH 带步骤推理，AIME 仅最终答案。few-shot 示例的答案风格会影响模型行为。

## 方案
新增 `--answer_style` 参数：`mixed`（默认）、`detailed`（优先长答案带推理）、`concise`（优先短答案）。

## 改动
- `fewshot.py` `FewShotBuilder.__init__` 新增 `answer_style` 参数
- `detailed` 模式：按 `answer_tokens` 降序排列数据池，优先选长答案
- `concise` 模式：按 `answer_tokens` 升序排列
- `mixed` 模式：保持现有行为不变
- `build_testdata.py` 新增 `--answer_style` 参数，传递给 `FewShotBuilder`
- README 补充各数据集答案风格说明表格
