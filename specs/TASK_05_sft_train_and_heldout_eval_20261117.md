# 任务 05：LoRA SFT 正式训练（2 万条 × 2 轮）+ 留出集闭环评测选 checkpoint

日期：2026-11-17　派发：Claude（大脑 / 核查）　执行：Codex（3080 上通过 `ssh 5880` 操作）
规格：`tasks/PLAN_v2.md` §4（SFT 目标、推理时非法输出的处理）、§7（训练配置、"在开发集上按证据丢失率选 checkpoint"）。
前置：任务 04 的 SFT 数据、`envs/train`、Qwen3-1.7B、100 步基准。标"Claude 的理解"的条目照抄进 NOTES。

## 0. 范围

用户 11-17 选择"方案 1 + 附加项 A"。

**做**：
- **A.** 测 per-device batch 4 / 8 的吞吐，选定训练配置；
- **B.** 抽 20,000 条样本，正式 LoRA SFT 2 个 epoch；
- **C.** 在留出部分（249 个任务）上对每个保存的 checkpoint 做闭环评测，按证据丢失率选出最佳 checkpoint，并和不训练的策略、未训练的原始模型对比。

**不做（硬性）**：
- 不在 dev（测试）任务上评测这个模型，即不碰 `data/feasibility` 的 150 个任务和其他 dev 任务。
- 不做 GRPO，不算阅读模型 F1，不调用 API。
- 不改任务 01–04 的产物；README / NOTES / HANDOFF 只追加。
- 不动 `ctxc` 和 `reader` 环境的安装内容（可以用 `reader` 环境里的 vLLM 跑推理，但不往里装东西），不动别人的环境。

## 1. 机器与资源（同任务 03 / 04）

- **只用 1 号卡**（`CUDA_VISIBLE_DEVICES=1`）。开跑前确认 1 号卡已用显存 < 2 GB，否则停下，用 `NEED:` 报告。
- **大文件放 `/root/userfolder/ctx-compressor/`**，包括 `runs/sft_20k_*`（checkpoint、日志）和 `eval_rollouts/`。`/` 的最终增量 ≤ 300 MB。
- **训练必须脱离 Codex 会话运行**（5880 上用 `nohup` 或 `tmux`），日志写文件。这样 ssh 断开或 Codex 重连都不会中断训练。
  - 记录 PID，只按确切 PID 管理进程。
- 训练期间每 30 分钟以内检查一次：进程存活、step 进度、loss 是否为 NaN、显存、磁盘。出现 NaN 或 OOM 就停下报告，不要擅自改超参重跑。

## 2. A：batch 吞吐对比（约 15 分钟）

- **对比的三种配置**：有效 batch 都是 16，其余配置同任务 04。
  - per-device 1 × 累积 16（任务 04 的基准）；
  - per-device 4 × 累积 4；
  - per-device 8 × 累积 2。
- **跑法**：每种跑 30 个 optimizer step，记录 token/s 和峰值显存。另外用"最长的 8 条样本凑成一个 batch"单独测一次，确认 per-device 8 不会 OOM。
- `group_by_length` 可以作为可选项一起测；用了就在 NOTES 里写明，它会改变样本顺序。
- **选择规则**：最长 batch 不 OOM、峰值显存 ≤ 40 GB 的配置里，选 token/s 最高的那个。用选中配置的实测速度更新 2 万 × 2 轮的预计墙钟时间，写进报告，然后**直接开始 B**，不必等我。

## 3. B：正式训练

- **抽样（Claude 的理解）**：从 `data/sft/train.jsonl` 抽 20,000 条，reselect、append 各 10,000 条。
  - 每种模式内先保证每个任务至少有 2 条样本（不足 2 条的任务全取），剩下的名额从其余样本中均匀随机补齐；
  - 用 `random.Random(20261118)`；
  - 抽中的 (task, step, mode) 写进 `runs/sft_20k_*/sample_ids.json`，并统计每种 tier 的条数。
- **超参**：同任务 04（PLAN §7），只有 per-device batch / 累积步数按 A 的选择调整。
  - LoRA r=16、alpha=32、dropout 0.05，target 全部线性层；
  - lr 1e-4，cosine，warmup 3%；bf16，gradient checkpointing；
  - `max_length=8392`，`truncation_strategy=raise`；
  - 2 个 epoch，约 2,500 个 optimizer step；seed 20261118；
  - 模板与任务 04 核对过的完全一致（qwen3，`enable_thinking=False`，只有 assistant 计 loss）。
- **保存与验证**：每 500 个 optimizer step 保存一次 LoRA checkpoint（共 5 个，含最后一个）。
  - 同时在留出 SFT 数据上随机固定抽 1,000 条，算一次 eval loss（只算 loss，用于画曲线，不用于选 checkpoint）。

## 4. C：留出集闭环评测

**评测网格**：249 个留出任务 × 两种模式 × 三档预算（1876 / 2814 / 5628），共 1,494 条轨迹。检索步骤沿用任务 04 已经生成的 `heldout_tasks.jsonl`，不重新检索。

**要评测的策略**：
- 5 个 checkpoint；
- **未训练的原始 Qwen3-1.7B**（同样的 prompt），作为 PLAN §7 里"Qwen3-1.7B 原始"那一行；
- 不训练的策略：`no_compress`、`latest`、`bm25`、`oracle`，规则与任务 02 相同。在 CPU 上跑；BM25 用 train 语料库的索引。

**闭环规则**（PLAN §4 / §7）：
- 每一步的输入是 `render_input(s)` 套 chat template（`enable_thinking=False`），与训练时的格式逐 token 一致：抽一条核对 token ids，写进报告。
- 模型输出解析为 JSON 后，交给现有的 `apply_action` 执行。以下情况都按"本步什么都不改"处理（`rollout` 现有逻辑），并分别计数：
  - 非法 JSON；
  - 编号不在候选里；
  - 只追加模式下 drop 不为空；
  - 超预算。
- 下一步的输入用模型自己的保留结果。

**推理设置**：
- greedy，`max_tokens=256`；记录达到上限的次数。
- 跨轨迹按步并行（同一步的所有轨迹放进一个 batch）。
- 用 `reader` 环境的 vLLM 加载 LoRA（`enable_lora`，`max_lora_rank=16`），或把每个 checkpoint 合并成完整权重后加载，两种都可以，写明用了哪种。

**指标**：口径同任务 02，按策略 × 模式 × 预算分别报告。
- 单步召回（按步平均、按任务平均）；
- 任务末召回；
- 证据丢失率，以及按跳位、按题位、按丢失方式的拆分；
- 证据完整题比例；
- 预算占用；
- 四类非法输出各自的比例；
- 成本（不开缓存 / 开缓存）。

**选 checkpoint（Claude 的理解）**：
- 主指标：把 6 个"模式 × 预算"格合起来（每格 249 个任务）算证据丢失率，最低者胜出；
- 打平时，比合起来的任务末召回；
- 还打平，选步数靠前的 checkpoint。

**报告**中写明：
- 每个 checkpoint 的主指标；
- 选中的 checkpoint；
- 选中模型、原始模型与 `bm25`、`oracle` 在**中档预算**下两种模式的对比；
- 不下"训练是否成功"的结论，只列数。

## 5. 产物

**代码**（项目目录，同步到 mirror）：
- `scripts/bench_batch.py`、`scripts/sample_sft_subset.py`、`scripts/train_sft.sh`、`scripts/rollout_model.py`（只有这里 import vLLM）、`scripts/heldout_eval.py`、`scripts/sft_run_report.py`；
- 必要的 `src/` 纯函数，配单元测试。原有 264 个测试必须继续全部通过。

**大数据**（userfolder）：
- checkpoint；
- 训练日志；
- 每条轨迹每一步的模型原始输出、解析结果、执行结果：`eval_rollouts/<策略>/steps.jsonl`。

**结果**（`eval/sft_run/`，同步到 mirror）：
- `sft_run_report.md`（中文，所有数附 n）；
- `sft_run_results.json`；
- `batch_bench.json`；
- `loss_curve.csv` / `.svg`（train loss 和 eval loss）；
- `train_run_config.json`（版本、超参、耗时、峰值显存、GPU 编号、`df`）。

**文档**：README / NOTES / HANDOFF 追加；SHA256SUMS 更新，大数据记录路径 + sha256；任务 01–04 的不变文件和文档前缀继续核对通过。

## 6. 完成标准

1. 训练 2 个 epoch 跑完，没有 NaN / OOM；5 个 checkpoint 都完成闭环评测。
2. 只用了 1 号卡，用完释放；`/` 的增量 ≤ 300 MB。
3. 测试全部通过。
4. 最后一条消息用中文汇报，以下面这行结尾：
   - A 的选择和实测速度；
   - 训练耗时；
   - 每个 checkpoint 的合并证据丢失率；
   - 选中的 checkpoint；
   - 中档预算下选中模型、原始模型、bm25、oracle 在两种模式下的任务末召回、证据丢失率和非法输出比例。

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务05 <一句话>
```
