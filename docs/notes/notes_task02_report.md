# 任务 02 报告接口与口径

新增文件：`src/feasibility_report.py`。唯一公开接口 `render_report(results) -> str`，只渲染 Markdown，不读写文件、不加载语料、不发起网络请求。runner 负责将返回字符串写入 `eval/feasibility/feasibility_report.md`。

## 顶层 schema

```text
settings: {seed, n_tasks, top_n, r_read?: 0.05, r_write?: 1.25,
           train_reference_n_tasks?: 150}
budgets: {tight, mid, loose, train_reference_p90}
dev_filter/train_filter: {total, valid, rejected, reasons: {原因: 数目}}
distributions: {
  task_raw_tokens, task_unique_tokens, task_steps, question_hops,
  task_evidence_tokens, question_evidence_tokens
}
retrieval: {
  overall: rate,
  by_hop: {"1": rate, "2": rate, "3": rate, "4": rate},
  by_question_hops: {"2": rate, "3": rate, "4": rate}
}
answer_leak: {
  all: {paragraph: rate, hop: rate},
  multiword: {paragraph: rate, hop: rate},
  rendered_all?: {paragraph: rate, hop: rate},
  rendered_multiword?: {paragraph: rate, hop: rate},
  body_duplicates: {count, n, rate}
}
configs: {配置名: {
  policy: no_compress/latest/bm25/oracle,
  mode: reselect/append（no_compress 可为任意表示等价模式的字符串）,
  budget_tier: tight/mid/loose（no_compress 可用 unbounded）,
  budget: 正整数,
  metrics: aggregate_tasks 的返回对象（下述）,
  input_tokens: distribution,
  fragment_input_tokens: distribution
}}
criteria: 可选，render_report 从 configs 同份聚合指标计算差值，不另依赖此字段
validation: 任意 JSON 对象，逐项打印校验结果、计数和异常解释
```

`distribution = {n, mean, p10, p50, p90, p99, max}`。`rate = {hits, n, rate}`；所有 rate 是 0–1 小数，报告格式化为百分比。零分母可用 `rate: null`，报告显示“未提供”，同时保留分母 n。

配置必须恰好 19 个：一个 no_compress，加三个策略 × 两种模式 × 三档预算；缺项/重复组合会明确报错。no_compress 的 `input_tokens` 和 `fragment_input_tokens` 都传每步执行后累计上下文分布，报告只打印一次“累计上下文”，不伪称为压缩器输入。

## aggregate_tasks schema（与 metrics 代理对齐）

```text
{
  n_tasks, n_steps,
  step_recall: {micro: {n,mean}, task_macro: {n,mean}},
  final_recall: {n,mean},
  evidence_loss: {
    n,count,rate,
    by_hop: {"1": group,...,"4": group},
    by_question_position: {"1": group,...,"4": group},
    by_mechanism: {never_kept: group,dropped: group}
  },
  evidence_complete_questions: {n,count,rate},
  occupancy: {n,mean,max},
  action_validity: {n,invalid_count,over_budget_count},
  cost: {
    plain: {n,mean,relative_to_no_compress},
    cached: {n,mean,relative_to_no_compress}
  }
}
group = {n,count,rate,evidence_n,lost_evidence_count}
```

跳位/题位的 group.n 是实际有该分组证据的任务数，count 是其中有该分组最终缺失证据的任务数；evidence_n 按每步证据关联计数，同 pid 如在多个跳出现可重复关联。丢失方式 group.n 为全部任务数，evidence_n 为每任务去重金标准数之和。报告明确指出分组不可直接相加。

## 需求覆盖与保守处理

1. 所有 6 类整体分布和每配置单步输入分布显示 n、mean、P10/P50/P90/P99/max；压缩器模板和片段分别报告。
2. 所有 19 配置显示两种单步召回、任务末召回、证据丢失、完整题、预算占用、动作非法/超预算、两种缓存成本及相对 no_compress 比例。
3. 答案泄漏正文为主，标题+正文为敏感性口径（runner 若提供即打印）；非证据分母排除整任务全部证据，多词子集独立分母。正文重复另报计数及分母。
4. 判据表第一列原样保留 PLAN 四个观察条件，其中 `F1 低 5 个点以上` 是原规格阈值，**不是实测结果**。本任务没有 F1 / EM 结果数字；阅读模型相关格一律“未测（需阅读模型）”。不复制 PLAN 的“开工”等决策列，不新定“接近”阈值，不按结果调任务。
5. 两模式、三预算下 oracle 减 BM25 的差值由 configs 的同份聚合指标计算，以百分点表示；中档列出任务末、按步单步、按任务单步差；前两者标记是否 ≥15。
6. 不改任务 01 文件；该新增文件及说明均只写 mirror。runner/指标的真实执行与最终校验由 root 负责。
