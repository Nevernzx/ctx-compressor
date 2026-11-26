# 任务18：GRPO 第四轮（reward v3：给无关段按读者 F1 定价），其余全同任务11 — 5880 GPU 1

Claude 2026-11-24 17:40 写。Codex 执行、Claude 核查。动手前通读全文，并通读同时派出的任务17（两者共用 GPU 1，顺序见 §5）。下面的判据和设置都是预注册的，看到结果后不许改。

## 0. 为什么做、改了什么

**来由**
- 11-24 用户回"**1和2一起吧**"。2 = 改奖励，对齐读者 F1。

**依据**（Claude 已在 5880 上只读算过）
- 任务16 的分解见 `claude_checks/reader_v2_r3r2_decomp_out_20261124.txt`：r3 比 r2 少丢证据，带来 +0.62；但多留无关段等带来的损失抵消有余，合并 −0.61。
- 无关段的定价见 `claude_checks/reward_v3_calib.py` 及其输出 `_out_20261124.txt`。
  - 做法：在任务16 held-out 新表上，同一道题在 6 个学习型压缩器和 oracle 之间比较（题目固定效应），按下式回归：
    - F1 ~ a·证据齐全 + b·最终上下文无关段数；
    - 结果 a = +17.80 F1，b = −1.046 F1/题，n = 41,832 行。
  - 一段无关段影响一条轨迹的 4 道题，丢一段 gold 约等于一道题不齐。所以按读者 F1 折算，一段无关段的价格是 κ_F1 = 4|b|/a = **0.235**（以丢一段 gold = 1 为单位）。
    - 稳健性：reselect 0.238，append 0.234，不含 oracle 0.246。
  - reward v2 只通过 token 项给无关段定价：0.5 × 149/1000 ≈ 0.075，约为 κ_F1 的 1/3。
- **局限，要写进报告**：κ 是从 held-out 数据上估出来的单个系数；最终结论以锁定测试集为准（不在本任务范围）。

**本轮只改一处：reward v3**

```
R_v3 = −n_lost − 0.16·n_nongold_after − 0.5·used_after/1000 − 1[invalid]
```
- 0.16 = 0.235 − 0.5 × 0.149，四舍五入到两位。这样一段平均长度的无关段，总价格约等于 κ_F1。
- `n_nongold_after`：执行动作后 `keep_after` 里，pid 不在**该任务全部 gold 集合**里的片段数。
  - 全部 gold 集合指所有步的 `gold`，包括还没走到的后续跳。
  - invalid 按 no-op 执行，此时 keep_after = kept_ids，照样计数。
- 其余三项与 `score_completion_v2(·, μ=0.5)` 逐项相同。
- 实现：新文件 `src/grpo_reward_v3.py::score_completion_v3`，内部调用 `score_completion_v2`，只加这一项。**不改 v2。**
- reward state 要带上任务的全部 gold pid（`task_gold_pids`）。从 `data/traj_train/train_tasks.jsonl` 按 task id 拼进来；held-out 从 `heldout_tasks.jsonl` 拼。

**不变的设置（全部同任务11）**
- 起点模型：`grpo_r2/merged_start`，即合并后的 SFT v3 ckpt-3000。直接复用，只读，按任务11 的签名清单核对 sha256。KL 参考模型也是它。
- 不另做 F0：任务11 F0 已有。
- B：直接复用任务11 B 的 `grpo_r2/states/states.jsonl`，核对 sha256。
- C：直接复用任务11 C 的 `grpo_r2/filter/*_samples.jsonl`，含原始输出。**只在 CPU 上用 v3 重新打分，不重新采样。**
- D：规则同任务11 D。2,000 条，只取 informative（std > 1e-9）；每格目标 334 / 334 / 333 / 333 / 333 / 333，缺口轮转分给有余量的格；种子 20261124，最后打乱；informative < 500 就停。
- E：同任务11 E。`--scale_rewards none`，LR 1.5e-5，BETA 0.04，NGEN 8，PER_DEVICE 4，GRAD_ACC 8，500 步，SAVE 125，新 LoRA r16 / α32 / dropout 0.05 all-linear，**GRPO_SEED 20261123（与任务11 相同，使两轮只差奖励）**。

**预注册指标**（只报数，不写"有效 / 无效"；每个数附 n）
1. 读者 F1 / EM（G 阶段）的配对比较，n = 5,976 题。报合并 ΔF1、ΔEM、逐题高 / 平 / 低计数、EM 2×2 和每格 ΔF1。
   - r4 选中 − r2 ckpt-125：同起点、同种子、只差奖励；
   - r4 选中 − SFT v3 ckpt-3000。
2. 每条轨迹最终上下文的平均无关段数和平均段数，读者 prompt 平均 token，以及 cost.py 开缓存 / 不开缓存两种成本。
3. PLAN §4 指标：tight 档每个模式的丢证据率，合并丢证据率，合并最终召回。
4. 按题的证据齐全率。
5. r4 选中 − r2 的分解，口径同 `claude_checks/reader_v2_r3r2_decomp.py`：证据齐全的四种转移，以及都齐全时按多留无关段数分组。
6. held-out reward v3 的每步均值及四个分项，对象是 r4 各 checkpoint、r2 ckpt-125、v3 ckpt-3000。后两者用已有 rollout 离线算，不重跑。

**不在范围内**
- 锁定测试集 `eval/test_dev_v1/` 不碰。

## 1. 机器、资源、硬规则（同任务11 §1、任务16 §1）

- 只用 GPU 1，`CUDA_VISIBLE_DEVICES=1`，`CUDA_DEVICE_ORDER=PCI_BUS_ID`。0 号卡即使空着也不用。
- 每个 GPU 阶段前过闸门：每 5 min 读一次，连续 3 次 < 2,048 MiB 才开始，36 h 超时；状态文件写 `waiting_gpu`。
- MRI_SAM 的进程和目录一律不碰。
- `/` 下新增 ≤ 300 MB；不装包，缺东西就打 `NEED:` 停下。
- 不删、不覆盖已有产物（遇到就拒绝）、不碰 `mirror/.git`、不 commit、只按确切 PID 杀。
- 读者评测一律 `disable_cascade_attn=True`，并从引擎读出生效值写进 run_config；不是 True 就停。
- 新目录：
  - 大文件：`grpo_r4/`、`eval_rollouts_grpo_r4/`、`reader_t18/`（都在 `/root/userfolder/ctx-compressor/` 下）；
  - 小结果：`eval/grpo_r4/`。
- 不写任务03–17 的目录；任务11 的产物只读。
- 发最终消息前关掉所有子代理。

## 2. Phase P（CPU，现在开始）

- 新功能全部写进新文件，例如 `src/grpo_reward_v3.py`、`scripts/*grpo_r4*`。任务11–16 的文件只 import、不改。
- `score_completion_v3` 的测试至少覆盖：
  - 每个分支的手算例：合法、invalid（no-op 时 kept 里的无关段照样计数）、超预算、append 下 drop；
  - 后续跳的 gold 不算无关段；
  - 在任务11 C 的全部样本上，v3 − v2 恰好等于 −0.16 × n_nongold_after，不一致 0 条。
- **C 重打分**：
  - 先用 v2 重算任务11 C 的每个样本，要与 C 记录的 reward 一致（不一致 0 条）；
  - 再用 v3 打分。每格和合并都报：zero-std 率、gold-varying 率、nongold-varying 率（组内 n_nongold_after 有差别）、token-only 率、有样本赢过 greedy 的组占比。
  - 并与任务11 C 并列。
- **D**：按 §0 的规则建数据集。检查项同任务11 D：
  - user 文本与 B 状态逐字节一致；
  - 全部来自 train split；
  - 插件在记录的 greedy 输出上算出的 v3 reward 等于重打分的 greedy reward，不一致 0 条。
  - 另报与任务11 D 数据集相同 state_id 的条数。
- 插件：`scripts/grpo_r4_swift_plugin.py`，照任务11 插件的写法，奖励换成 v3。每步记录五个分项：n_lost、n_nongold_after、used/1000、invalid、add-count。
- 全量 CPU 测试必须全过。

## 3. 流水线（GPU，detached）

运行方式同任务11 A.5：pidfile；状态文件 `grpo_r4/pipeline_status.json`；每阶段一份 receipt，带 require 检查；不自动重试；每 30 s 监控一次。

1. **冒烟**：2 步训练，放在 `grpo_r4/smoke_train/`，8 条 prompt 取自本轮 D。复算 reward 和 advantage，期望不一致 0 条。
2. **E 训练**：见 §0，`OUT = grpo_r4/train/`。
   - loss 或 grad_norm 非有限、或 OOM 就停。
   - 用 `completions.jsonl` 复核 reward 和 advantage，期望不一致 0 条。
3. **F 评测**：checkpoint 125 / 250 / 375 / 500（实际存下的那些），以 LoRA 叠在 `grpo_r2/merged_start` 上。协议同任务11 F：held-out 249 任务 × 6 格，greedy，`max_tokens` 256。
   - **选点规则**（预注册，只在 r4 的 checkpoint 里选）：
     1. held-out reward v3 每步均值最高（6 格合并、全部步）；
     2. 平局时合并丢证据数更少；
     3. 再平局取步数最早。
   - 理由：这是训练目标本身，不用读者；κ 已按读者 F1 定过价。
4. **G 读者**：引擎设置同任务16（取 `eval/reader/run_config.json` 全部参数，加 `disable_cascade_attn=True`），greedy，max_tokens 48，seed 20261117。
   - 行：r4 的 4 个 checkpoint，同批重跑 r2 ckpt-125 和 v3 ckpt-3000。
   - 每个任务一批，共 6 × 24 = 144 条，顺序固定为：r4-125、250、375、500、r2、v3。开头照旧先跑任务03 的 40 条小样。
   - 主比较用**同批的数**；另报 r2、v3 两行与任务16 逐题 raw_output 不同的条数，以及本次运行的同次运行内不一致率（口径同任务15 §3.1）。
   - 选点以 F 的规则为准。其余 3 个 checkpoint 的 F1 只报数，不参与选点。

## 4. 报告与交付

- `eval/grpo_r4/grpo_r4_report.md`（中文，每个数附 n）和 `grpo_r4_results.json`，含 §0 的全部预注册指标。
- 训练曲线 csv + svg，包含 n_nongold_after 分项。
- 对比表的行：r4 选中、r4 其余 checkpoint、r2 ckpt-125、r3 ckpt-250（引用任务16）、v3 ckpt-3000、H1（引用任务16）、bm25、oracle（引用任务16）。
- 报告要写明 §0 的局限：κ 来自 held-out。
- `eval/grpo_r4/TASK18_SHA256SUMS`，并入 SHA256SUMS。任务06–16 的清单仍要校验通过。
- README / NOTES / HANDOFF 只追加；镜像用 `rsync -a`，不带 `--delete`。

## 5. 与任务17 共用 GPU 1

- 同一时间 GPU 1 上只跑我们的一个阶段。两条流水线共用一个锁：`/root/userfolder/ctx-compressor/rt/gpu1_t17_t18.lock`（flock）。拿到锁后照样过闸门。
- 默认顺序：**任务18 冒烟 → E → F → G，然后任务17 的 S1 → S2**。
  - 如果任务18 的 Phase P 卡住，而任务17 已准备好，任务17 可以先上，并在状态文件里记一笔。
- 两条流水线都启动（或排上锁）以后，打印 `WAITING: 任务17/18 Phase P 完成，GPU 队列已启动（附 C 重打分的合并 zero-std 率和 D 的构成）`，**不打 DONE 行**。全部 GPU 阶段结束后，Claude 会发收尾消息。

## 6. 时间估计（Claude）

| 阶段 | 估计 |
|---|---|
| Phase P | 约 1–1.5 h（C 重打分只用 CPU，几分钟） |
| 冒烟 | 约 20 min |
| E | 约 5.5 h（任务11 实测 5.3 h） |
| F | 4 × 约 0.75 h |
| G | 约 40 min（35,856 + 40 条） |
| 闸门 | 每个 GPU 阶段约 10–15 min |

合计约 11–12 h。
