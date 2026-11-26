# 任务12：held-out 端到端读者 F1（Qwen3-8B，沿用任务03 协议）— 5880 GPU 1

Claude 2026-11-23 16:40 写。Codex 执行、Claude 核查。动手前通读全文。下面的判据和设置都是预注册的，看到结果后不许改。

## 0. 为什么做、做什么

- 11-23 用户看完任务11 结果后回"**1和2都做吧**"，2 = 端到端读者评测。
- 目的：任务11 的 GRPO r2 ckpt-125 把 held-out 丢证据率从 59.3% 降到 47.8%。要看这个下降能不能变成 PLAN_v2 §7 的端到端指标："阅读模型分别回答任务里的 4 个问题，报告平均 EM / F1"。
- **只做推理和统计**：
  - 不训练，不重跑压缩器，不改任何已有 rollout；
  - 不调用任何付费或在线 API。PLAN §7 的"API 模型在子集上复核"这次不做；
  - 不碰锁定测试集 `eval/test_dev_v1/`。

## 1. 机器、资源、硬规则（同任务11 §1）

**GPU**
- 只用 GPU 1：`CUDA_VISIBLE_DEVICES=1`、`CUDA_DEVICE_ORDER=PCI_BUS_ID`，GPU 0 永远不碰。
- 每个 GPU 阶段开始前都过任务10 的等待闸门（`scripts/sft_v3_wait_gpu.py` 的逻辑），这条阈值不许改：
  - 每 5 分钟读一次；
  - 连续 3 次 < 2,048 MiB 才开始；
  - 最长等 36 h；
  - 状态文件写 `waiting_gpu`。
- GPU 1 和用户的 MRI_SAM 线共用，**绝不碰 MRI_SAM 的进程或目录**。

**目录和环境**
- 大文件只放 `/root/userfolder/ctx-compressor/reader_heldout/`，小结果放 `eval/reader_heldout/`。
- 读者环境：先 source `scripts/reader_environment.sh`，用 `envs/reader` 和 `models/Qwen3-8B`。
- `/` 下新增 ≤ 300 MB（任务03 峰值曾到 501 MB，这次开工前、结束时各记一次 `df`，峰值也记）。
- 不装包；缺东西就打印 `NEED:` 并停下。

**禁止**
- 删除任何东西、覆盖已有输出（遇到就拒绝）；
- 写任务03–11 的目录；
- 碰 `mirror/.git`、commit、push；
- 用非确切 PID 的方式杀进程（禁止 `pkill -f`）。

**每次发最终消息前关掉所有子代理。**

## 2. 评测对象

每个压缩策略取 held-out 的 249 任务 × 2 模式 × 3 档 = 1,494 条轨迹。读者回答每条轨迹的 4 道题，每格 249 × 4 = 996 题。

| 名称 | rollout 路径（`/root/userfolder/ctx-compressor/`下） |
|---|---|
| `grpo_r2:checkpoint-125`（任务11 选中） | `eval_rollouts_grpo_r2/checkpoint-125` |
| `sft_v3:checkpoint-3000`（任务11 起点） | `eval_rollouts_v3/checkpoint-3000` |
| `sft_v2:checkpoint-2000`（任务06） | `eval_rollouts_v2/checkpoint-2000` |
| `grpo_r1:checkpoint-500`（任务08） | `eval_rollouts_grpo_v2/checkpoint-500` |
| `h1:tau_star`（任务09） | `eval_rollouts_hedge_v2/tau_star` |
| `bm25`、`latest`、`oracle`（任务05） | `eval_rollouts/<名称>` |
| `no_compress`（任务05） | `eval_rollouts/no_compress` |
| `no_context`（闭卷对照） | 无上下文 |

- 每条 rollout 读之前，先核对 sha256 与各任务的签名清单一致，不一致就停下报告。
- `no_compress`：先检查每个任务在 6 格里的最终上下文（pid 列表和顺序）是否完全相同。
  - 全相同：只跑一次，996 题，报告里作为一行；
  - 有不同：6 格全跑，并报告不同的格数。
- `no_context`：996 题，每题一次。

## 3. 阅读协议（与任务03 §3 完全一致，不许改模板）

**上下文**
- 取每条轨迹**最后一步**的 `keep_after`，按记录的顺序排列。
- 每段渲染成 `title + "\n" + paragraph_text`，不带编号；段与段之间用 `"\n\n"` 连接。
- 文本取自 `data/traj_train/corpus_train.jsonl`。

**题目和答案**
- 题目取 `heldout_tasks.jsonl` 的 `questions`。
- 答案按 `example_ids` 到原始 `musique_ans_v1.0_train.jsonl` 取 `answer` 和 `answer_aliases`。

**模板、生成、解析、评分**
- 直接 import 任务03 的 `src/reader.py` 和 `src/qa_metrics.py`，不复制、不修改：
  - user 模板和 `no_context` 模板逐字沿用；
  - 解析规则和 MuSiQue 官方 EM / F1 口径沿用。
- 生成设置与任务03 的 `eval/reader/run_config.json` 一致：
  - enable_thinking=False，greedy，max_tokens=48；
  - bf16，max_model_len=40960，不开 YaRN；
  - gpu_memory_utilization 0.85，prefix caching、chunked prefill；
  - max_num_seqs 16，max_num_batched_tokens 8192，enforce_eager；
  - seed 20261117。
- 同一轨迹的 4 道题共用上下文，按任务03 的顺序提交，让 prefix cache 生效。

**超长**
- 每个 prompt 的 token 数 + 48 必须 ≤ 40960。超出就停下报告，**不许截断**。
- 先做全量 token 预检，再上 GPU。

**一致性小样**
- 用同一引擎重跑任务03 `smoke_samples.md` 里的 40 个 prompt，与 `data/reader/predictions.jsonl` 比 raw_output。
- 不一致的条数如实记录，不作为停止条件。

## 4. 指标（预注册）

**主指标**
- 每个策略合并 6 格的平均 **F1** 和 EM，n = 5,976 题。

**主比较**（只报数，不写"有效 / 无效"）
1. `grpo_r2:checkpoint-125` 对 `sft_v3:checkpoint-3000`：
   - 合并 F1 差（百分点）和 EM 差；
   - 逐题配对计数：r2 高 / 相等 / v3 高；
   - EM 的 2×2 配对表。
2. `grpo_r2:checkpoint-125` 对 `h1:tau_star`：同上。

**每个策略都报**
- 每格（mode × tier）的 F1 / EM，每格 n = 996。
- 按"该题证据是否齐全"分组的 F1，各组附 n。证据齐全指该题全部 gold 段都在最终上下文里。
- 按题目跳数（2 / 3 / 4）、题位（1–4）分组的 F1。
- 每题 prompt token 数：mean、P50、P90、max。

**两张图**（PLAN §7 的核心图，SVG）
- **预算–F1**：横轴是三档预算 1876 / 2814 / 5628，纵轴是 F1，每个策略 × 每个模式一条线；`no_compress`、`no_context` 画水平参考线。
- **成本–F1**：横轴是每条轨迹的缓存口径成本，取各任务已有的 `<策略>_results.json` 里的 `cost.cached.mean`，与 rollout 复算一致；纵轴是 F1，每格一个点。

**证据丢失和 F1 的关系**（补充表）
- 每个模型策略，对比以下两组的平均 F1，各附 n：
  - 轨迹丢了证据的题；
  - 轨迹没丢证据的题。

## 5. 流程

**P（CPU）**
- 写 `scripts/run_reader_heldout.py`（只有它 import vLLM）、`scripts/reader_heldout_report.py`，配测试。
- 测试至少覆盖：
  - 上下文渲染与任务03 一致；
  - 答案取 alias 后取最大值；
  - `no_compress` 去重判定；
  - 拒绝覆盖。
- 全量 CPU 测试必须全过。

**R（GPU，过闸门）**
- 以 detached 方式运行：pidfile、状态文件 `reader_heldout/status.json`、每阶段一份 receipt。
- 不自动重试。

**报告**
- 中文报告写 `eval/reader_heldout/reader_heldout_report.md`，每个数都附 n。
- 结果写 `reader_heldout_results.json`。
- 两张 SVG 和 `run_config.json`：版本、参数、耗时、峰值显存、`df` 记录。
- 预测逐题写 `reader_heldout/predictions.jsonl`，字段同任务03。

**交付**
- 生成 `eval/reader_heldout/TASK12_SHA256SUMS` 并并入 SHA256SUMS；任务06–11 的清单仍要校验通过。
- README / NOTES / HANDOFF 只追加。
- 镜像用 `rsync -a`，不带 `--delete`，不碰 `.git`，不 commit。

## 6. 时间估计（Claude）

- P 约 1 h，可与任务13 的 Phase P 并行。
- R 约 2–2.5 h：约 8 × 5,976 + 2 × 996 ≈ 4.98 万个 prompt，任务03 实测 6.9 个/s。
- 报告约 30 min。

## 7. 和任务13 的关系

- 两个任务都用 GPU 1，所以排队：**先跑任务12 的 R，再开任务13 的 Phase A**。
- 任务13 的 G 阶段会用本任务的脚本，给任务13 选中的模型补一行读者 F1。本任务的脚本要支持"给一个新的 rollout 目录、补算一行"。
- 任务12 报告写完后**不打 DONE 行**，直接进入任务13 Phase A，详见任务13 §2。
