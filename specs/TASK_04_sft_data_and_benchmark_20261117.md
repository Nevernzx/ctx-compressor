# 任务 04：PLAN §7 训练准备——SFT 数据、训练环境与吞吐实测（不正式训练）

日期：2026-11-17　派发：Claude（大脑 / 核查）　执行：Codex（3080 上通过 `ssh 5880` 操作）
规格：`tasks/PLAN_v2.md` §3（数据与划分）、§4（SFT 目标）、§7（训练配置）。标"Claude 的理解"的条目照抄进 NOTES。

## 0. 范围

用户 11-17 选择进入 §7。训练样本量要由用户拍板，所以本任务只做与样本量无关的准备，并实测训练速度；**不跑正式训练**。

**做**：
1. 修 "#9 Dream"；
2. 划分 train / 5% 留出；
3. 建 train 语料库和 BM25；
4. 生成训练与留出轨迹、SFT 样本；
5. 新建训练环境（ms-swift）；
6. 下载 Qwen3-1.7B；
7. 跑一小段 LoRA SFT 吞吐基准；
8. 给出若干样本量方案的估时表。

**不做（硬性）**：
- 不跑完整训练，也不跑超过本任务书 §5 规定步数（100 步）的训练。
- 不做 GRPO，不做闭环评测。
- 不调用 API。
- 不动 `ctxc` 和 `reader` 环境，不动别人的环境。
- 不改任务 01–03 的任何产物；README / NOTES / HANDOFF 只追加。
- 不改 dev 侧任何结果。

## 1. 机器与资源（同任务 03）

- **只用 1 号卡**（`CUDA_VISIBLE_DEVICES=1`）。开跑前确认 1 号卡已用显存 < 2 GB，否则停下，用 `NEED:` 报告。
- **大文件放 `/root/userfolder/ctx-compressor/`**：
  - 新环境 `envs/train`；
  - 模型 `models/Qwen3-1.7B`；
  - SFT 数据 `data/sft/`、轨迹 `data/traj_train/`；
  - 基准输出 `runs/bench_*`。
- 缓存目录沿用任务 03 的设置，指向 userfolder。`/` 的最终增量 ≤ 300 MB。
- 不走代理；pip 用清华源。ms-swift 选一个稳定版并写明理由。
  - 环境装好后写 `requirements.train.lock.txt`。
  - attention 用 sdpa。flash-attn 编译超过 20 分钟就放弃，不影响本任务。
- 项目正式目录 `/root/workplace/ctx-compressor` 只放代码、小结果和文档。mirror 只同步代码、文档和 `eval/sft_prep/` 小结果。

## 2. 修 "#9 Dream"

- **现象**：2 道 train 题的子问题里，"#9 Dream" 是歌名（John Lennon 的歌），被当成了"第 9 跳答案"的占位符。
- **改法（Claude 的理解）**：`hop_queries` 只把 `1 ≤ k ≤ 已完成跳数` 的 `#k` 当作引用并替换；超出范围的 `#k` 保留原文，不抛错，同时计数。在结果里记录受影响题的 id。
- **测试**：
  - 新增单元测试，覆盖 "#9 Dream" 保留原文、`#1` 正常替换，以及原有的越界测试按新语义更新。
  - 对全部 dev 题重跑 `hop_queries`，输出必须与任务 02 的 `tasks_dev150.jsonl` 和全部 dev 结果**逐字节一致**。不一致就停下报告。

## 3. 划分与轨迹

- **留出**：从修复后全部可用的 train 题中，按题用 `random.Random(20261118)` 留出 5%，向下取整；ID 写进 `data/splits/train_heldout_ids.json`，与 PLAN §3 一致。
- **语料与检索**：train 语料库与 BM25 用**全部 train 题**的段落建（PLAN §3："train 和 dev 各建一个"），沿用 `build_corpus` 和 `simulate_task(top_n=20)`。
- **拼任务**：
  - 训练部分和留出部分各自用 `sample_tasks(..., seed=20261118)`，每题只用一次；
  - 拼不满 4 题的余数丢弃并计数；
  - 每个任务的检索种子等于"部分内的任务序号"。
- **预算档（Claude 的理解）**：每条轨迹固定一档，用 `random.Random(20261118 + 任务序号)` 在 tight / mid / loose（1876 / 2814 / 5628）中均匀抽一档。
- **两种模式**：每个任务都生成 reselect 和 append 两份 rollout。
  - reselect 用 `make_oracle_noise_policy(seed=任务序号)` 生成状态；
  - append 用纯 oracle；
  - `sft_target` 一律是纯 oracle（与任务 01 的约定一致）。

## 4. SFT 样本

- **每个状态一条样本**：
  - `messages = [{"role":"user","content":render_input(s)}, {"role":"assistant","content":render_target(s)}]`；
  - 另附 meta：task、step、mode、tier、split、n_input_tokens、n_target_tokens。
- **chat template**：Qwen3，`enable_thinking=False`。训练串必须和以后推理时的串格式一致。
  - 把一条完整样本经 ms-swift 实际编码后的字符串（输入可截断显示）和 labels 掩码，写进 `eval/sft_prep/template_check.md`；
  - 要求只有 assistant 部分计 loss，并写清 assistant 前是否有空的 `<think></think>` 块。
- **长度**：统计"套模板后的输入 token + 目标 token"的分布（P50 / P90 / P99 / max），按 mode × tier 分组。
  - `max_length` 取能覆盖全部样本的值；如果因此超过 16384，就停下报告分布，不要截断或丢样本。
- **其他统计**（按 mode × tier）：
  - 样本数；
  - 目标里 add 为空 / drop 不为空的比例；
  - 预算真正卡住的步数比例：理想动作没能放下全部可得证据的步。

## 5. 吞吐基准（只测速度，不是正式训练）

- **模型**：从 ModelScope 下载 `Qwen/Qwen3-1.7B`，逐文件校验 sha256，生成 `eval/sft_prep/model_manifest.json`。
  - 核对它的 `tokenizer.json` 与任务 01 用的 `data/raw/qwen3_tokenizer/tokenizer.json` 是否一致。不一致就报告，并说明对 ntok 的影响。
- **配置**（PLAN §7）：
  - LoRA r=16、alpha=32、dropout 0.05，target 全部线性层；
  - bf16，gradient checkpointing，per-device batch 1；
  - 梯度累积 16；lr 1e-4，cosine，warmup 3%；不用 packing。
- **跑法**：从训练部分按"一半 reselect、一半 append"均匀抽 1,600 条样本，跑 **100 个 optimizer step**。
- **记录**：
  - 每秒 token 数、每条样本的秒数、峰值显存；
  - loss 曲线：是否下降，前后各 10 步的均值；
  - 有没有 OOM。
  - 另外用 max_length 附近的最长样本单独测一次，确认不会 OOM。
- **用实测速度估时**：分别估算
  - 训练部分的**全部**样本 × 1 / 2 / 3 个 epoch；
  - 抽样 20k / 40k 条 × 2 个 epoch。
  
  每种都写出预计墙钟时间，以及占用 1 号卡的时长。**估时表只列出来，不选方案。**

## 6. 产物

**代码**：
- `src/build_traj.py`：只改 `hop_queries` 的引用规则，配新测试。
- `scripts/make_train_traj.py`、`scripts/make_sft_data.py`、`scripts/setup_train_env.sh`、`scripts/download_policy_model.py`、`scripts/bench_sft.sh`（或等价的 .py）、`scripts/sft_prep_report.py`。
- 新测试放在 `tests/` 里，在 `ctxc` 环境跑。原有 202 个测试必须继续全部通过。

**数据**（userfolder）：
- `data/traj_train/{train,heldout}_tasks.jsonl`；
- `data/sft/{train,heldout}.jsonl`；
- `data/splits/train_heldout_ids.json`（这个文件小，同时放进项目目录）。

**结果**（`eval/sft_prep/`）：
- `sft_prep_report.md`（中文：数据统计表、长度分布、模板核对、基准结果、估时表，每个数都附 n）
- `sft_prep_results.json`
- `template_check.md`
- `model_manifest.json`
- `bench_run_config.json`

**文档**：README / NOTES / HANDOFF 追加；SHA256SUMS 更新，大数据不纳入，记录路径、行数和 sha256；任务 01–03 的不变文件和文档前缀继续核对通过。

## 7. 完成标准

1. 测试全部通过；dev 侧逐字节不变。
2. SFT 数据生成完毕，长度不超过 max_length，没有被截断的样本。
3. 基准 100 步跑完、没有 OOM，loss 有下降。
4. 只用了 1 号卡，用完释放；`/` 增量 ≤ 300 MB。
5. 最后一条消息用中文汇报：
   - 两部分各自的任务数和样本数；
   - 长度的 P50 / P99 / max；
   - 基准每秒 token 数和峰值显存；
   - 估时表。
   
   以下面这行结尾：

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务04 <一句话>
```
