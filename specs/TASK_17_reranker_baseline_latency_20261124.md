# 任务17：同数据微调的重排模型基线（Qwen3-Reranker-0.6B）+ 按跳拆分 + 压缩器单步延迟 — 5880 GPU 1

Claude 2026-11-24 17:50 写。Codex 执行、Claude 核查。动手前通读全文，并通读同时派出的任务18（两者共用 GPU 1，顺序见任务18 §5）。下面的判据和设置都是预注册的，看到结果后不许改。

## 0. 为什么做

**来由**
- 11-24 用户回"**1和2一起吧**"。1 = 补重排基线和延迟。

**PLAN_v2 §7 原文要求**
- 对比方法里有"同数据微调的重排模型：用同一份训练轨迹微调一个 cross-encoder，按分数装满预算；**最可能打平本项目的基线**"。到任务16 为止，这一行一直没做。
- "预期与检验"一节写道："本项目应该在靠后的跳上赢过微调的重排模型……重排模型一次只看一个片段。按跳数拆分结果来检验；如果两者打平，如实报告。"
- "效率"一项：压缩器的单步延迟（vLLM）。

**基线怎么定（Claude 的选择）**
- 模型：**Qwen3-Reranker-0.6B**。
  - 它是现成的 cross-encoder 式重排模型，参数比我们的 1.7B 压缩器小。
  - ModelScope `Qwen/Qwen3-Reranker-0.6B`，5880 上已测可以访问。
- 两个变体，都在完整的 rollout 里闭环跑：
  - **RR-fill**：PLAN 原文的"按分数装满预算"。
  - **RR-τ**：只装分数 ≥ τ 的片段。τ 在训练集划出的验证任务上按读者 F1 选，held-out 不参与。
    - 加这一行是因为任务16 发现，读者对无关段很敏感，"装满"可能吃亏。
    - RR-τ 是在目标指标上调过的、更强的基线，对本项目是保守的比较。
- 另加一行 **RR-zs-fill**：未微调的原模型装满预算，作为"现成重排模型"的参照。

**不在范围内**
- 锁定测试集 `eval/test_dev_v1/` 不碰。
- LLMLingua-2 / Provence 和真实 Agent 测试（PLAN §8）都不做。

## 1. 机器、资源、硬规则

- 同任务18 §1：只用 GPU 1、过闸门、MRI_SAM 不碰；`/` ≤ 300 MB；不装包；不删不覆盖；不碰 `.git`、不 commit；只按确切 PID 杀；读者评测 `disable_cascade_attn=True` 并核对生效值；发最终消息前关掉子代理。
- **模型下载**
  - 只下 `Qwen/Qwen3-Reranker-0.6B`，到 `/root/userfolder/ctx-compressor/models/Qwen3-Reranker-0.6B/`。
  - 缓存目录设在 `/root/userfolder/ctx-compressor/cache/modelscope`。
  - 记录 revision、各文件 sha256 和 license，写进 `eval/reranker_v1/model_card.json`。
  - 下载失败就打 `NEED:` 停下，**不要自行换成别的模型**（包括 `/root/commonfile/models/Qwen3-Reranker-4B`）。
- 新目录：
  - 大文件：`/root/userfolder/ctx-compressor/reranker_v1/`，含训练、checkpoint、分数缓存、rollout、读者预测；
  - 小结果：`eval/reranker_v1/`。
- 不写任务03–16、18 的目录。

## 2. 数据（CPU）

- **划分**
  - 训练任务取 `data/traj_train/train_tasks.jsonl`，共 4,735 个。
  - 用 `random.Random(20261128).sample(sorted(task_ids), 200)` 取 200 个作 **train-val**，按抽样顺序记录；前 100 个另外记为 **val100**。
  - 其余 4,535 个用于训练。
  - 检查：与 held-out、锁定测试集重叠都是 0。
- **候选与标签**
  - 每个任务的候选 = 所有步的 `new` 的并集（去重）。
  - 标签：pid 在该任务全部 gold 集合（所有步的 `gold`）里记 1，否则记 0。
  - 报告：每任务候选数和正例数的分布。
- **输入文本**
  - query = 该任务 4 个问题，按顺序用 `"\n"` 连接。
  - document = 该片段在压缩器 prompt 里的正文，逐字相同。从现有渲染函数取，不要自己拼；用 3 个任务抽查逐字一致。
  - 模板：Qwen3-Reranker 官方格式（system 提示 + `<Instruct>: …\n<Query>: …\n<Document>: …` + 空 think 的 assistant 前缀）。
    - 打分 s = logit("yes") − logit("no")，取最后一个位置；prob = sigmoid(s)。
    - 模板原文写进 run_config。
  - Instruct 固定为：`Given the four questions of a multi-hop QA task, judge whether the passage is a supporting fact needed to answer any of them.`
  - 每对最长 512 token，超出时从 document 末尾截断，并报截断条数。

## 3. 训练（S1，GPU）

- 全参微调，bf16，AdamW，lr 1e-5，cosine，warmup 3%，1 个 epoch，每步 64 对（可以用梯度累积凑），种子 20261128。
- 损失：对 s 的逐对 BCE。
- 每个 epoch 的样本：训练任务的全部正例，每个正例配 7 个同任务的负例，无放回均匀抽取；同任务负例不够时全取。
- **耗时上限**：冒烟测吞吐后，如果预计超过 3 h，负例改为每个正例 3 个，并在 run_config 里记一笔。
- **选点**：每 1,000 步和训练结束时，在 train-val 200 上算每任务的 AP（gold 对该任务全部候选），取均值。取均值最高的 checkpoint，平局取更早的。
  - 同时报原模型（zero-shot）的 AP。
- **分数缓存**：只依赖 (query, document)，每个 (任务, pid) 算一次。
  - 选中模型：val100 + held-out 249；
  - 原模型：held-out 249。
  - 写 `reranker_v1/scores/*.jsonl`，并报每个文件的 sha256。

## 4. 策略与闭环 rollout（CPU，读分数缓存）

- **策略**：与 `src/policies.py::bm25_policy` 相同，只把分数来源换成重排分数。
  - 候选：reselect 用 kept + new，append 只用 new。
  - 按分数从高到低排，分数相同时排在后面的条目优先。
  - 用 `_pack` 做 first-fit。**直接 import `_pack`，不改 policies.py。**
- **RR-τ**：只有 prob ≥ τ 的候选进入排序。reselect 下低于 τ 的已保留片段会被丢掉；append 下低于 τ 的新片段不加。
- **rollout 协议**：同任务12 的闭环评测，249 任务 × 2 模式 × 3 档，预算口径相同。输出 steps.jsonl / metrics.json，字段与 `eval_rollouts/bm25/` 一致。

## 5. 选 τ（S2 的第一步，GPU 读者）

- 网格：τ ∈ {0（= fill）, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}。
- 每个 τ 在 val100 × 6 格上闭环 rollout，再跑读者，共 9 × 2,400 = 21,600 条。
- 读者引擎同任务18 G。
- 取 6 格合并读者 F1 最高的 τ，差 < 1e-9 时取更小的 τ。如果选中 τ = 0，RR-τ 就等于 RR-fill，照实写。
- 报告每个 τ 的：合并 F1 / EM、丢证据率、平均保留段数、平均无关段数。

## 6. held-out 评测（S2，GPU 读者）

- 行：RR-fill、RR-τ、RR-zs-fill，加同批重跑 r2 ckpt-125 和 v3 ckpt-3000（这两行的 rollout 用已有的）。
- 每个任务一批，共 5 × 24 = 120 条，顺序固定为 RR-fill、RR-τ、RR-zs-fill、r2、v3。开头照旧先跑任务03 的 40 条小样。
- **预注册指标**（只报数，不写"有效 / 无效"；每个数附 n）
  1. 每行的丢证据率（每格、合并）、最终召回、单步召回、平均保留段数、平均无关段数、读者 prompt 平均 token、cost.py 开缓存 / 不开缓存两种成本、读者 F1 / EM（每格、合并）、按题的证据齐全率。
  2. 配对主比较（n = 5,976 题），报 ΔF1、ΔEM、逐题高 / 平 / 低计数、EM 2×2、每格 ΔF1，用同批的数：
     - r2 − RR-τ；
     - v3 − RR-τ；
     - r2 − RR-fill。
  3. **按跳拆分**（PLAN §7 的预期检验）。每个 gold 片段按其所在问题的 `hop_index` 分组，看任务结束时是否还在：
     - 报每个跳位的保留率，对象是 RR-fill、RR-τ、r2、v3、bm25、oracle；每个模式分开，并报 6 格合并；
     - 报"每道题第一个丢失的 gold 在第几跳"的分布；
     - 并列 r2 − RR-τ 在第 1 跳和第 ≥ 2 跳上的保留率差。
  4. 一致性：r2、v3 两行与任务16 逐题 raw_output 不同的条数，以及本次运行的同次运行内不一致率（口径同任务15 §3.1）。
- 表里另外引用任务16 的 no_compress、latest、bm25、oracle、H1 和 r3 各行，标明出处。

## 7. 单步延迟（S2 最后一步，GPU）

- **状态**：从 r2 ckpt-125 的 held-out rollout 里抽 300 个输入状态，每格 50 个，种子 20261129。
- **压缩器**：r2 ckpt-125，即 `grpo_r2/merged_start` 加 LoRA，与评测相同。
  - 两种引擎设置各测一次：与评测相同的 `enforce_eager=True`；以及 `enforce_eager=False`（CUDA graph）。后者加载失败就记下原因，不硬修。
  - (a) batch 1 顺序逐条请求：先热身 20 条，每条记墙钟时间，报 p50 / p90 / 均值，以及 prompt token 和输出 token 的分布；
  - (b) 300 条一起提交（max_num_seqs 16）：报每秒处理的步数。
- **重排模型**：选中的 checkpoint，HF transformers，bf16。
  - 每步把全部候选放进一个 batch 打分（reselect 用 kept + new，append 用 new）；
  - 报同样的 p50 / p90 / 均值和吞吐。
- 测量时记录 GPU 0 的占用（别人的卡），只作背景信息。

## 8. 流程与交付

- **Phase P（CPU，与任务18 Phase P 并行）**：下载、§2 数据、训练 / 打分 / 策略 / 读者 / 延迟驱动及测试。
  - 测试至少覆盖：
    - 划分不重叠；
    - 标签；
    - 模板与打分符号（yes 高时 s > 0）；
    - 策略在给定分数下与 bm25_policy 的排序规则一致（用 BM25 分数代入，要与 `eval_rollouts/bm25` 的动作逐步一致，抽 10 个任务）；
    - τ 过滤；
    - 拒绝覆盖。
  - 全量 CPU 测试必须全过。
- **S1**：训练、选点、分数缓存。
- **CPU**：rollout。
- **S2**：选 τ、held-out 读者、延迟。各 GPU 会话都按任务18 §5 排锁、过闸门。运行方式：detached、pidfile、状态文件 `reranker_v1/status.json`、每阶段一份 receipt、不自动重试。
- **报告**：`eval/reranker_v1/reranker_v1_report.md`（中文）、`reranker_v1_results.json`、run_config、按跳拆分表、延迟表，以及在任务16 的两张 SVG 上加 RR 两行的新图（新文件名）。
- `eval/reranker_v1/TASK17_SHA256SUMS`，并入 SHA256SUMS。任务06–16 的清单仍要校验通过。README / NOTES / HANDOFF 只追加；镜像用 `rsync -a`，不带 `--delete`。
- **最终消息**（任务17、18 合并一条，中文）：
  - 任务17：τ、三组主比较、按跳拆分要点、延迟；
  - 任务18：C / D 构成、训练时长、各 checkpoint 的 reward v3 与丢证据、选中的 checkpoint、两组读者主比较、无关段数与成本。
  - 以这一行结束：

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务17+18 <一句话，含 r2−RR-τ 和 r4−r2 的 ΔF1>
```

## 9. 时间估计（Claude）

| 阶段 | 估计 |
|---|---|
| Phase P | 约 2–3 h（和任务18 的 GPU 阶段重叠） |
| S1 | 约 1–2 h（训练约 40 万对） |
| rollout | 约 15 min（CPU） |
| S2 | 约 1.5 h（读者 21,600 + 29,880 条，约 55 min；延迟约 20 min） |
| 闸门 | 每个会话约 10–15 min |

任务17 自身约 3–4 h 的 GPU 时间，排在任务18 之后。
