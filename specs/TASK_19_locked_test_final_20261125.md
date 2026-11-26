# 任务19：锁定测试集最终对比（MuSiQue dev，第一次使用）— 5880 GPU 1

Claude 2026-11-25 10:40 写。Codex 执行、Claude 核查。动手前通读全文，并通读同时派出的任务20（两者共用 GPU 1，顺序见 §6）。下面的行、设置和指标都是预注册的，看到结果后不许改。

本任务就是 `eval/test_dev_v1/LOCK.md` 里说的"first use requires a Claude-written task"。

## 0. 为什么做

**来由**
- 11-25 用户回"**1和2都做吧**"。1 = 锁定测试集最终对比，2 = 真实 Agent 测试（任务20）。

**held-out 的局限**（Claude 已在 5880 上只读复算，见 `claude_checks/check_rr_memorization_out_20261125.txt`、`check_rr_reader_out_20261125.txt`）
- held-out 的 gold 段有 82.8% 在训练任务里当过 gold。
- 微调重排模型对"训练时见过的 gold"判为 gold 的比例是 94.2%，没见过的是 84.5%；压缩器 v3 两者都约 90%。
- held-out 上 RR-τ 与 r2 / v3 只差 0.5–0.9 F1。但只看"有 gold 没见过"的题（n = 2,316），压缩器高 3.2–4.3 F1。
- reward v3 的无关段价格 κ 也是在 held-out 上估的。

**锁定测试集为什么能回答这些**
- 用 dev 语料（21,100 段），test gold 只有 26.6% 在 train 语料里逐字出现；在 SFT v2 训练行里当过 gold 的是 0（见 `structural_stats.json`）。
- 所有选择（checkpoint、τ、预算、读者设置）都已在训练集 / held-out 上定死，测试集只跑一次。

## 1. 行（全部冻结）

| 行 | 来源 | 读者条数 / 任务 |
|---|---|---|
| no_compress | 全部检索内容，`policies.no_compress_policy`，与模式和档位无关 | 4 |
| no_context | 不给上下文，同任务12 | 4 |
| latest | `policies.latest_policy` | 24 |
| bm25 | `policies.bm25_policy` | 24 |
| oracle | `build_traj.oracle_action` | 24 |
| RR-fill | 任务17 选中的重排 checkpoint，τ = 0 | 24 |
| RR-τ | 同上，τ = 0.5（任务17 在 val100 上选定） | 24 |
| v3 | SFT v3 ckpt-3000（= `grpo_r2/merged_start`，不加 LoRA） | 24 |
| r2 | grpo_r2 ckpt-125，LoRA 叠在 merged_start 上 | 24 |
| **r4** | grpo_r4 ckpt-500（任务18 F 规则选中），LoRA 叠在 merged_start 上。**本项目的主行** | 24 |

- 每个 checkpoint 和重排模型都按各自任务的签名清单核对 sha256，对不上就停。
- 预算：tight 1,876 / mid 2,814 / loose 5,628（LOCK.md），每条轨迹固定一档。
- 不加任何别的行，不做任何选择。

## 2. 机器、资源、硬规则（同任务18 §1）

- 只用 GPU 1（`CUDA_VISIBLE_DEVICES=1`，`CUDA_DEVICE_ORDER=PCI_BUS_ID`）；每个 GPU 阶段前过闸门（每 5 min 读一次，连续 3 次 < 2,048 MiB，36 h 超时，状态写 `waiting_gpu`）。
- MRI_SAM 的进程和目录一律不碰。`/` 下新增 ≤ 300 MB；不装包，缺东西就打 `NEED:` 停下。
- 不删、不覆盖（遇到就拒绝）、不碰 `mirror/.git`、不 commit、只按确切 PID 杀。
- 读者评测一律 `disable_cascade_attn=True`，从引擎读出生效值写进 run_config；不是 True 就停。
- **锁定测试集的文件一个都不改**：`eval/test_dev_v1/` 和 `data/test_dev_v1/` 只读。动手前和交付前各核对一次 LOCK.md 里 5 个文件的 sha256。
- 新目录：大文件 `/root/userfolder/ctx-compressor/test_v1/`；小结果 `eval/test_v1/`。不写任务03–18 的目录。
- 发最终消息前关掉所有子代理。

## 3. Phase P（CPU，现在开始）

- 新代码全部写进新文件（如 `scripts/test_v1_*.py`），旧模块只 import、不改。
- **数据**
  - 核对 LOCK.md 的 sha256 后，用任务10 锁定时的同一套函数（`build_task10_locked_dev.py` 调的 `build_corpus`，同一分词器）重建 dev 语料。
  - 检查：段数 21,100；对 3 个任务，每步 gold pid 对应的 (title, paragraph_text) 与原始 dev 的支撑段逐字一致；不一致就停。
- **重排**：任务17 选中的 checkpoint，模板、Instruct、512 截断、query（4 个问题用 `"\n"` 连接）全部同任务17。每个 (任务, pid) 打一次分，写 `test_v1/scores/`。策略直接 import 任务17 的实现。
- **压缩器 rollout**：协议同任务18 F（greedy，`max_tokens` 256，`enforce_eager=True`，invalid 按 no-op），只是把任务换成 451 个测试任务、语料换成 dev 语料。451 任务 × 2 模式 × 3 档。
- **读者请求**：用任务12 / 16 的同一套渲染函数（`src/reader.py`、`src/reader_heldout_inputs.py`）。
- **测试**至少覆盖：
  - LOCK.md 的 sha256 核对、语料重建检查；
  - 重排策略回归：用任务17 的 held-out 分数缓存跑 10 个 held-out 任务，动作与任务17 的 rollout 逐步一致；
  - 读者批次构成（§5）；
  - 拒绝覆盖；
  - 生效值检查（`disable_cascade_attn` 不是 True 就拒绝）。
- 全量 CPU 测试必须全过。

## 4. GPU 阶段

1. **G0 冒烟**：r4 ckpt-500 在 3 个 held-out 任务 × 6 格上重跑，与任务18 F 的 held-out rollout 逐步比较动作。报不同的步数；超过 5% 就停（说明加载的 adapter 或渲染不对）。
2. **G1 重排打分**：选中的重排模型给测试任务打分。
3. **G2 压缩器 rollout**：v3、r2、r4 三个模型依次跑完整测试集。
4. **G3 读者**：见 §5。

运行方式同任务18 §3：detached、pidfile、状态文件 `test_v1/status.json`、每阶段一份 receipt（带 require 检查）、不自动重试、每 30 s 监控一次。

## 5. 读者

- 引擎同任务18 G：`eval/reader/run_config.json` 的全部参数加 `disable_cascade_attn=True`；greedy，max_tokens 48，seed 20261117，enable_thinking=False。
- 开头照旧先跑任务03 的 40 条小样，单独一批。
- 每个任务一批，共 200 条，顺序固定为：no_compress 4、no_context 4、latest、bm25、oracle、RR-fill、RR-τ、v3、r2、r4（后 8 行各 24）。
- 任务顺序同 test_tasks.jsonl。合计 451 × 200 = 90,200 条。

## 6. 预注册指标（只报数，不写"有效 / 无效"；每个数附 n；单次运行，不算 CI）

1. **每行**：读者 F1 / EM（每格、每模式、每档、合并）；丢证据率（任务结束时至少丢一段 gold 的轨迹比例，每格、合并）；最终召回；单步召回；非法和超预算比例（模型行）；平均保留段数、平均无关段数；读者 prompt 平均 token；cost.py 开缓存 / 不开缓存两种成本；按题的证据齐全率。
2. **配对主比较**（每格每题配对，n = 451 × 6 × 4 = 10,824）。每组报合并 ΔF1、ΔEM、逐题高 / 平 / 低计数、EM 2×2、每格 ΔF1：
   - a. **r4 − RR-τ**（PLAN §7：同数据微调的重排模型）；
   - b. r4 − v3（GRPO 相对 SFT）；
   - c. r4 − r2（reward v3 相对 v2）；
   - d. r4 − bm25；
   - e. r4 − no_compress（no_compress 的同一道题与 r4 每格配对）；
   - f. oracle − r4（与上限的差）。
3. **按跳拆分**（PLAN §7 的预期检验）：
   - 每个 gold 片段按 `hop_index` 分组，报任务结束时的保留率。对象是全部 24 条 / 任务的行，每个模式分开，并报 6 格合并。
   - "每道题第一个丢失的 gold 在第几跳"的分布。
   - r4 − RR-τ 在第 1 跳和第 ≥ 2 跳上的保留率差。
   - 按题的跳数（2 / 3 / 4）分组的 F1，以及 r4 − RR-τ 的 ΔF1。
4. **两张 SVG**：预算–F1、成本–F1（开缓存），画全部行。
5. **"见没见过"**：test gold 段里，有多少在任何 traj_train 任务（训练 + held-out，也就是重排模型和压缩器训练时的 gold）中当过 gold。按 (title, paragraph_text) 逐字比较。比例 > 0 时，主比较 a 再按"该题 gold 全见过 / 有没见过的"拆开报。
6. **一致性**：40 条小样的不一致数；本次运行的同次运行内不一致率（口径同任务15 §3.1，全部行合算一次、每行各算一次）。
7. **held-out 对照表**：每行并列 held-out F1（任务16 / 17 / 18 的同批数，标明出处）和测试集 F1。只并列，不做配对。

## 7. 与任务20 共用 GPU 1

- 共用一个锁：`/root/userfolder/ctx-compressor/rt/gpu1_t19_t20.lock`（flock）。拿到锁后照样过闸门。同一时间只跑一个阶段。
- 默认顺序：**任务19 G0 → G1 → G2 → G3，然后任务20 的冒烟和主运行**。
- 例外：任务20 的 Phase P 先做完时，任务20 冒烟可以插在任务19 两个阶段之间跑一次，并在两边状态文件里各记一笔。任务20 主运行一律排在任务19 G3 之后。

## 8. 报告与交付

- `eval/test_v1/test_v1_report.md`（中文）、`test_v1_results.json`、run_config、按跳拆分表、两张 SVG。
- 报告开头写明：这是锁定测试集的第一次使用；所有设置来自哪个任务；没做任何选择。
- `eval/test_v1/TASK19_SHA256SUMS`，并入 SHA256SUMS。任务06–18 的清单仍要校验通过。
- README / NOTES / HANDOFF 只追加，写一句"锁定测试集已于任务19 第一次使用"。镜像用 `rsync -a`，不带 `--delete`。
- 任务19 交付完成后打印一行（**不打 DONE**，继续任务20）：

```
WAITING: 任务19 完成（r4−RR-τ ΔF1=…，r4−v3 ΔF1=…，r4 F1=…，n=10,824）
```

## 9. 时间估计（Claude）

| 阶段 | 估计 |
|---|---|
| Phase P | 约 1–1.5 h |
| G0 + G1 | 约 30 min |
| G2 | 约 3.5 h（任务18 F 实测 249 任务 34.5 min / 模型；451 任务约 70 min / 模型 × 3） |
| G3 | 约 30–45 min（90,240 条；任务18 G 实测 35,856 条 10.5 min） |
| 闸门 | 每个阶段约 10–15 min |
| 报告 | 约 30 min |
| **合计** | **约 6–7 h** |
