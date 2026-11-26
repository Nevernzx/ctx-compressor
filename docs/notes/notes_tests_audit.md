# 测试与数据验收实现说明（并入 NOTES / HANDOFF）

1. 测试直接使用任务书要求的奖励分支及 PLAN §5 的三组精确成本数值。随机状态测试使用 `random.Random(seed)`，10 个种子 × 200 个状态 × 2 种模式，共 4,000 个可行状态；状态生成器保证已保留 token 未超预算，符合真实 rollout 的前置条件。重选状态额外检查噪声策略合法性。
2. `audit_data.py` 对 train/dev 原始 `(title, paragraph_text)` 元组做精确集合统计，不做大小写或空白归一化。标题口径按唯一标题数计算双方比例，并额外给出对应段落数，以避免把同标题不同正文误称为精确泄漏。
3. dev 的 P50/P90/P99 针对去重语料段落，使用原始 `tokenizer.json` 的 `Tokenizer.encode(..., add_special_tokens=False)`；渲染格式统一调用 `render_paragraph`，分位数采用 NumPy 线性插值。输入模板、编号元数据不计入“保留片段”预算。
4. 对 `#k` 以外引用格式采用保守词法扫描：非标准 `#`、`[n]`、`$n`、`<n>` / ref/answer/hop XML 标签、花括号引用。命中是“疑似”而非语义判定；相关题跳过并在 JSON 报告记录 ID、查询、原因。不声称有限正则能证明不存在任意自然语言引用。`>>` 为关系分隔符，保留原样。
5. 真实验收固定默认 seed=20261117，跨任务无放回抽 12 道 dev 题形成 3 个任务；同一任务两种模式复用相同检索步骤。各任务检索打乱种子为 seed+任务号。验收只建 dev BM25 索引，预算默认 3000，明确标为占位，不执行 PLAN §6。
6. `data_audit.md` 内打印 2/3/4 跳各 5 条完整原始 question_decomposition，供主 README / DATA_CARD 嵌入。机器可读 `data_audit.json`、`acceptance_checks.json` 及 `data/traj/acceptance.jsonl` 仅保留在 5880；镜像同步 Markdown 文档及代码。

复现（在 5880 项目根目录，使用 root 提供的环境包装器）：

```bash
bash scripts/run_python.sh -m pytest tests -q
bash scripts/run_python.sh scripts/audit_data.py
bash scripts/run_python.sh scripts/acceptance.py --seed 20261117 --budget 3000
```

若命令超过两分钟，应依任务书用 `setsid nohup`、日志及 PID 文件在远端运行。
