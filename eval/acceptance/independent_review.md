# 任务 01：独立复核记录

复核日期：2026-11-17。结论：**通过，未发现阻塞交付的错误**。

复核对象为 `scripts/audit_data.py`、`scripts/acceptance.py`、`tests/`、`eval/acceptance/traj_samples.md`，并在 5880 上直接读取 MuSiQue-Ans dev 原始 JSONL 和 `data/traj/acceptance.jsonl` 进行独立交叉检查。没有将原始数据复制到 3080。

## 可复现的独立检查

`scripts/review_acceptance.py` 不导入生产轨迹函数、原数据审计函数或原验收函数；直接按原始 dev 文件顺序，使用 `(title, paragraph_text)` 重建段落映射。脚本只读取已有文件并打印结果，不下载数据、不写入产物、不使用 GPU 或模型权重。

在 3080 执行：

```bash
ssh 5880 'cd /root/workplace/ctx-compressor && env PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES= .conda/envs/ctxc/bin/python scripts/review_acceptance.py'
```

实际结果：

```json
{
  "result": "PASS",
  "records_independently_checked": 58,
  "unique_candidate_fragments_retokenized": 489,
  "dev_corpus_size": 21100,
  "distinct_questions": 12,
  "records_per_mode": {"reselect": 29, "append": 29},
  "task_hop_coverage": {"1": [2], "2": [2, 3], "3": [2, 3, 4]}
}
```

逐条检查内容：

1. 原始子问题来自相应题号与跳号；`#k` 只引用题内此前第 k 跳答案，替换结果与记录相同且无残留数字占位符。
2. 根据原始 `paragraph_support_idx` 定位的证据段，其标题和正文映射到记录中的同一语料 pid。
3. 每步工具返回 20 个不重复段落，包含本跳证据；状态的 new 正确排除已保留段落并保持工具返回顺序。
4. 轨迹内同一 pid 的小编号保持稳定；候选编号与 `pid_by_lid` 一一对应；本步 kept 等于上一步 keep_after。
5. 当前 gold 等于此前已执行跳的证据与本步可获得候选的交集，符合 PLAN 对当前可用金标准的定义。
6. 用原始标题、换行、原正文重新构造候选文本，并用本地 Qwen3 tokenizer 重新计算 489 个不同候选的 token；与所有状态记录一致。
7. add/drop 编号合法；独立执行动作后的顺序与 keep_after 相同；预算、used_after 与 recall 重算一致；append 模式没有 drop 且前缀不变。
8. 三任务各有 4 道题，共 12 题不重复；同一任务的两种模式使用相同题目；58 条记录等于所选各题跳数之和的两倍。

## 代码与报告人工检查

- 审阅了奖励的格式错误、非法编号、append 禁止 drop、预算失败、部分/全部召回、空金标准和重复 add 编号测试；成本测试包含任务书要求的三组精确数值。
- 随机合成状态测试为两模式 × 10 种子 × 200 状态，验证 oracle 合法性；噪声测试验证干净目标、下一步 drop、预算上限和 append 不加噪声。
- 验收脚本仅建立 dev 语料；每跳先记录自然命中，未命中时替换末位证据，然后按显式种子打乱。自然命中率字段的计算流程经静态审阅，本次独立脚本没有重新建立 BM25 索引。
- 抽查任务 1 第 2 跳的 `Neil Patrick Harris` 替换、任务 2 第 5 步的 `#2 → Republicans`，以及任务 3 的四跳 `J. P. Hayes → Appleton → Outagamie County → Brown County` 链。报告的证据标题、前 200 字符和编号均对应原始标注，未擅自改写看似间接的证据。
- 数据审计按原始字符串精确去重，不混用 train/dev；分位数计算使用完整 dev 去重段落；占位符及缺失证据检查有明确保守跳过路径。2／3／4 跳各 5 条原始分解供人工核对。

该复核只验证任务 01 框架及报告一致性，不据此推断 §6 可行性试验结果或正式预算。
