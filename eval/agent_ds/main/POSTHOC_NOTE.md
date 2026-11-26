# 事后分析文件说明（不在预注册里）

- `thought_memory_posthoc.json`：答对的题里，答案字符串出现在哪里（最终上下文、Agent 自己的笔记，或都不在）。
- `conditional_accuracy_posthoc.json`：按最终证据是否齐全分开算的 F1。
- `dropped_evidence_posthoc.json`：证据不齐全的题，分成"看过全部 gold 后又丢掉"和"从没搜齐"两类。
- `bridge_memory_posthoc.json`：**作废**。它用 `gold_hop` 判断证据是否齐全，但 HotpotQA / 2Wiki 的 `gold_hop` 为空，口径错误。文件按"不删产物"的规则保留，正确口径见 `dropped_evidence_posthoc.json`。
- 汇总解读见 `posthoc_report.md`。
