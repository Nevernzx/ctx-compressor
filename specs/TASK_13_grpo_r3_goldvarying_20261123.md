# 任务13：GRPO 第三轮（从 r2 ckpt-125 出发，训练集优先取"丢不丢 gold 有差别"的组）— 5880 GPU 1

Claude 2026-11-23 16:50 写。Codex 执行、Claude 核查。动手前通读全文。下面的判据和设置都是预注册的，看到结果后不许改。

## 0. 为什么做、改了什么

**来由**
- 11-23 用户回"**1和2都做吧**"，1 = GRPO 第三轮。

**任务11 留下的问题**（Claude 独立复核过，见 `claude_checks/check_grpo_r2_eval_out_20261123.txt`）
- r2 ckpt-125 合并丢证据 714/1494，之后的 250 / 375 / 500 是 718 / 735 / 724，**第 125 步之后不再进步**。
- 训练中"8 个样本全同分"的组从 27% 升到 47%（每 50 步平均）。
- 训练集 2,000 条里只有 556 条（27.8%）是 gold-varying（组内 n_lost 或 invalid 有差别），其余 1,444 条只在 token 开销上有差别。
- 首个错误里最多的仍是"当步新 gold 没加"：510/1494。

**本轮只改两处，其余全部同任务11**
1. **起点**：任务11 选中的 `grpo_r2/train/checkpoint-125`，即 LoRA 叠在 `grpo_r2/merged_start` 上。合并成 `grpo_r3/merged_start`，脚本与任务11 A 相同（CPU fp32 merge，存 bf16）。新一轮的 KL 参考模型就是它。
2. **数据集规则（D）**：先取 gold-varying 的组，不够再用 token-only 的组补足，见 §3 D。

**不变的设置**
- reward v2：μ = 0.5，`score_completion_v2`；
- `--scale_rewards none`，LR 1.5e-5，BETA 0.04；
- NGEN 8，PER_DEVICE 4，GRAD_ACC 8；500 步，SAVE 125；
- 新 LoRA r16 / α32 / dropout 0.05，all-linear；
- 采样设置、F 的评测协议、选点规则。

**新的种子**
- B 取任务：20261125；C 采样：20261125；D 抽样：20261126；GRPO_SEED：20261125。

**预注册指标**（只报数，不写"有效 / 无效"）
1. tight 档每个模式的丢证据率；
2. 合并丢证据率，其次合并最终召回；
3. held-out reward v2 每步均值及三个分项；
4. 选中模型的读者 F1（G 阶段，任务12 协议）。

主比较对象是 r2 ckpt-125。

**不在范围内**
- 锁定测试集 `eval/test_dev_v1/` 不碰。

## 1. 机器、资源、硬规则

- 与任务11 §1 完全相同：GPU 1、等待闸门、MRI_SAM 不碰、`/` ≤ 300 MB、不装包、不删不覆盖、不碰 `.git`、不 commit、只按确切 PID 杀。
- 新目录：
  - 外部大文件：`grpo_r3/`、`eval_rollouts_grpo_r3/`、`reader_heldout/grpo_r3_<ckpt>/`；
  - 小结果：`eval/grpo_r3/`。
- 不写任务03–12 的目录。

## 2. 阶段安排

### Phase P（现在开始，CPU，与任务12 并行）

- 新功能写进**新文件**，例如 `scripts/*grpo_r3*`。任务11 的文件只能 import、不能改，否则任务11 的签名清单会失效。
  - 实在要改共享文件，推迟到 Phase A，改后重跑全量测试。
- 写 B / C / D / F / G 的驱动与测试。D 的新规则至少测四种情况：
  - gold-varying 足够；
  - 不够、需要用 token-only 补；
  - 某格 gold-varying 不足、按轮转分给有余量的格；
  - informative < 500 时停下。
- 全量 CPU 测试必须全过。
- **不用 GPU**：任务12 的 R 在用 GPU 1。

### Phase A（任务12 报告写完后直接开始，不用等 Claude）

1. **合并**：写出 `grpo_r3/merged_start`。
2. **贪心等价检查**：用 n = 30 的 step-0 fixture，比较 base + r2 merged_start + ckpt-125 LoRA 与新 merged 模型单独运行的输出。报告文本和 token id 的不一致数；不一致只记录，不作为停止条件。
3. **2 步训练冒烟**：同任务11 §7，放在 `grpo_r3/smoke_train/`，8 条 prompt 取自任务11 的 D 数据集。
4. **启动流水线**：冒烟通过后启动 §3 的 detached 流水线，要求与任务11 A.5 相同：
   - pidfile；
   - 状态文件 `grpo_r3/pipeline_status.json`；
   - 每阶段一份 receipt，带 require 检查；
   - 不自动重试，每 30 s 监控一次。
5. **汇报**：确认流水线在跑以后，打印下面这行，**不打 DONE 行**。收尾时 Claude 会发消息。
   - `WAITING: 任务12 完成（附合并 F1 主比较的两组数），任务13 流水线已启动`

## 3. 流水线

### F0：merged-only held-out 评测

- 协议同任务11 F0，输出到 `eval_rollouts_grpo_r3/merged_start/`。
- 报告与 r2 ckpt-125 rollout 的逐格差异，以及输入状态相同、raw 输出不同的步数。

### B：训练状态

- 用起点模型做 greedy 闭环 rollout：400 个训练任务，种子 20261125，6 格。
- 这 400 个任务必须与任务08 B、任务10 C、任务11 B 的任务**都不重叠**。
- `verify_rows` 检查与报告项同任务11 B。

### C：采样过滤

- 同任务11 C：8 个样本，T = 1，种子 20261125，μ = 0.5，informative = std > 1e-9。
- 每格和合并后都报：
  - zero-std 率、gold-varying 率、token-only 率；
  - 有样本赢过 greedy 的组占比；
  - 采样的 add-count 分布、各类 invalid 率。
- 与任务11 C 对比。

### D：GRPO 数据集（本轮的新规则）

- **目标 2,000 条**。格子按排序后的顺序，目标条数 334 / 334 / 333 / 333 / 333 / 333。
1. **第一层**：只从 gold-varying 的组里抽，按上面的目标每格均衡抽取。哪格不够，缺口轮转分给 gold-varying 有余量的格。
2. **第二层**：如果 gold-varying 总数 < 2,000，用 token-only-varying 的 informative 组补足到 2,000，补法同样是按格均衡加轮转。
3. 抽样种子 20261126，最后整体打乱。
4. informative 总数 < 2,000 时全部用上；**informative < 500 就在 E 之前停下**，状态写 `status=stopped_insufficient_signal`。
- 报告每格两层各取了多少条，以及 gold-varying 的总占比。
- 检查项同任务11 D：
  - user 文本与 B 状态逐字节一致；
  - 全部来自 train split；
  - 插件在记录的 greedy 输出上算出的 reward 等于 C 的 greedy reward，不一致 0 条。

### E：GRPO 训练

- 同任务11 E，三处路径改为：`MERGED = grpo_r3/merged_start`、`DATA = <D>`、`OUT = grpo_r3/train/`。
- `MAX_STEPS = floor(n_prompts / 4)`。
- 每步都记录 reward 分项。
- loss 或 grad_norm 非有限、或 OOM 就停。
- 用 `completions.jsonl` 复核 reward 和 advantage，期望不一致 0 条。

### F：评测 checkpoint

- 在 held-out 上评测 125 / 250 / 375 / 500（实际存下的那些），协议同任务11 F。
- **选点规则**（只在 r3 的 checkpoint 里选）：合并丢证据率最低 → 合并最终召回最高 → 步数最早。
- extra stats 同任务11 §3F，包括 held-out reward v2 及分项。

### G：选中模型的读者 F1

- 用任务12 的脚本和协议，给 r3 选中 checkpoint 的 rollout 补一行：6 格 × 996 题。
- 输出：
  - 预测写 `reader_heldout/grpo_r3_<ckpt>/`；
  - 小结果并入 `eval/grpo_r3/`，**不改任务12 的产物**。
- 报告：
  - 合并 F1 / EM；
  - 与 r2 ckpt-125 的配对比较，方式同任务12 §4。

## 4. 报告（收尾，等 Claude 的消息）

- `eval/grpo_r3/grpo_r3_report.md`（中文，每个数附 n）和 `grpo_r3_results.json`，章节同任务11 §4。
- 对比表的行：
  - r3 选中；
  - r3 起点（F0）；
  - r2 ckpt-125；
  - SFT v3 ckpt-3000；
  - H1；
  - bm25、oracle（引用）。
- 对比表的列：同任务11，再加**读者 F1**。r3 以外各行的 F1 引用任务12。
- 训练曲线：csv + svg。
- 写清两项预注册数字（r2 ckpt-125 对 r3 选中）和读者 F1 的差。

## 5. 交付物

- `eval/grpo_r3/TASK13_SHA256SUMS`，并入 SHA256SUMS。任务06–12 的清单仍要校验通过。
- README / NOTES / HANDOFF 只追加。
- 镜像用 `rsync -a`，不带 `--delete`，不碰 `.git`，不 commit。
- 最终消息用中文，包括：
  - 任务12 的主比较；
  - 任务13 的起点、F0 差异；
  - B / C / D 的计数，包括 D 两层的构成；
  - 训练时长；
  - 每个 checkpoint 的合并丢证据数和选中的 checkpoint；
  - tight 和 mid 的对比；
  - held-out reward v2；
  - 读者 F1。
- 以这一行结束：

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务13 <一句话，含任务12 的 F1 主比较>
```

## 6. 时间估计（Claude）

| 阶段 | 估计 |
|---|---|
| Phase A | 约 40 min |
| F0 | 约 0.7 h |
| B | 约 1.1 h |
| C | 约 2.4 h |
| D | 几分钟 |
| E | 约 5.5 h |
| F | 4 × 约 0.75 h |
| G | 约 0.3 h |
| 等待闸门 | 每个 GPU 阶段约 10 min |

任务12 完成后，合计约 14–15 h。
