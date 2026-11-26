# 任务16：关掉 cascade attention，重跑整张 held-out 读者表 — 5880 GPU 1

Claude 2026-11-24 11:10 写。Codex 执行、Claude 核查。动手前通读全文。下面的设置和指标都是预注册的，看到结果后不许改。

## 0. 为什么做

- 任务15 诊断结论（Claude 已独立复算，见 `claude_checks/check_reader_diag15_out_20261124.txt`）：
  - cascade attention 开着时，读者输出取决于批次构成：
    - try2（每批 24 条，只有 r3）：同次运行内不一致 30.1%，r3 F1 50.71；
    - 臂 b（每批 48 条，r2 和 r3 混合）：r2 F1 从任务12 的 56.12 掉到 50.15。
  - 关掉以后（臂 a），r3 同次运行内不一致降到 1.7%，F1 55.72。
  - 任务12 自己是在 cascade 开着时跑的，同次运行内不一致 6.6%。
- 11-24 用户选"**方案1**"：所有读者行都关掉 cascade，**整张表重跑**，旧结果全部保留。

## 1. 机器、资源、硬规则

**与任务12 §1 相同**
- 只用 GPU 1，闸门阈值不改；MRI_SAM 不碰。
- `/` 下新增 ≤ 300 MB；不装包。
- 不删、不覆盖、不碰 `mirror/.git`、不 commit；只按确切 PID 杀进程。
- 发最终消息前关掉所有子代理。

**与任务15 相同**
- 不改 vLLM 安装目录里的任何文件。
- 用短 TMPDIR：`/root/userfolder/ctx-compressor/rt/g16`，并写 `ipc_check.json`。

**目录**
- 不写任务03–15 的目录。
- 新的大文件：`/root/userfolder/ctx-compressor/reader_heldout_v2/`。
- 新的小结果：`eval/reader_heldout_v2/`。
- 任务12 / 13 的报告和清单都不改。NOTES / HANDOFF 末尾追加一句："任务12 / 13 的读者 F1 被任务16（关 cascade）取代，旧数保留"。

## 2. 输入和批次

**任务12 的全部请求**
- 直接复用任务12 的 `input_preflight/requests.jsonl`（共 249 × 200 = 49,800 条），按任务12 的签名清单核对 sha256。
- 构成：每个任务 200 条，依次是：
  - no_compress 4 条；
  - 8 个策略各 24 条：latest、bm25、oracle、sft_v2、grpo_r1、h1、sft_v3、grpo_r2；
  - no_context 4 条。

**加一行 r3**
- `grpo_r3:checkpoint-250` 的 5,976 条请求，复用 try2 的 `input_preflight/requests.jsonl`，按任务13 的签名清单核对。

**批次**
- 每个任务一批，共 224 条：任务12 原来的 200 条（顺序不变），后面接 r3 的 24 条。
- 任务顺序同任务12。合计 249 × 224 = 55,776 条。
- 开头照旧先跑任务03 的 40 条一致性小样，单独一批。

**引擎**
- 取 `eval/reader/run_config.json` 的全部参数，**只加一项** `disable_cascade_attn=True`。
- 生成参数不变：greedy，max_tokens 48，seed 20261117，enable_thinking=False。
- 从引擎读出实际生效的 `disable_cascade_attn`，写进 `run_config.json`；不是 True 就停。

## 3. 指标（预注册，每个数都附 n）

**任务12 §4 的全部指标重算一遍，表里多一行 r3 ckpt-250**
- 每个策略的合并 F1 / EM；
- 每格 F1 / EM；
- 按证据是否齐全、跳数、题位分组的 F1；
- prompt token 统计；
- 两张 SVG（预算–F1、成本–F1），图中加 r3；
- 证据丢失与 F1 的补充表。

**三组主比较**（只报数，不写"有效 / 无效"）
- 每组都报：合并 F1 差、EM 差、逐题配对计数、EM 2×2。
1. r2 ckpt-125 − sft_v3 ckpt-3000（任务12 主比较 1）；
2. r2 ckpt-125 − h1 tau_star（任务12 主比较 2）；
3. r3 ckpt-250 − r2 ckpt-125（任务13 G 的比较）。

**新旧对照表**
- 每个策略并列以下几列：
  - 任务12 的 F1（cascade 开）；
  - 本次 F1；
  - 差；
  - 逐题 raw_output 不同的条数。
- r3 那一行的"旧值"列，同时写 try2 的 50.709 和任务15 臂 a 的 55.719。

**一致性**
- 本次运行的同次运行内不一致率，口径同任务15 §3.1：
  - 全部策略合在一起分组算一次；
  - 每个策略各算一次。
- 40 条小样中的不一致数。

## 4. 任务14 的后续（等任务14 F 跑完再做）

**任务14 C**
- 原定用任务12 的脚本，给 DeepSeek 压缩器 B 的 60 条轨迹补一行 Qwen3-8B 读者 F1，共 240 题次。
- 改为用本任务的引擎设置：关掉 cascade，其余同上。
- 与 §2 的主运行可以合在同一次 GPU 会话，也可以单独过一次闸门。
- 输出：
  - 大文件：`reader_heldout_v2/deepseek_compressor/`；
  - 小结果并入 `eval/deepseek_api/`，**只追加新文件**，不改任务14 已交付的文件。

**任务14 A 的 Qwen 列重算**
- 任务14 A 里"同一批提示上 Qwen3-8B 的 F1 / EM"原来取自任务12。
- 用本任务的新预测按同键重算一遍：只用 CPU，不重跑 DeepSeek。
- 写进 `eval/deepseek_api/qwen_column_v2.md` / `.json`，并列新旧两列。

## 5. 流程

**P（CPU）**
- 新文件：`scripts/reader_heldout_v2.py`；只 import 任务12 / 15 的模块，不改。
- 测试至少覆盖：
  - 批次构成（每批 224 条，前 200 条与任务12 逐条一致）；
  - 两份输入的签名核对；
  - 生效值检查（disable_cascade_attn 不是 True 就拒绝）；
  - 拒绝覆盖；
  - IPC 路径长度检查。
- 全量 CPU 测试必须全过。

**R（GPU）**
- 闸门 → 小样 → 主运行。
- 运行方式：detached；pidfile；状态文件 `reader_heldout_v2/status.json`；每阶段一份 receipt；不自动重试。

**报告**
- `eval/reader_heldout_v2/reader_heldout_v2_report.md`（中文）；
- `reader_heldout_v2_results.json`；
- `run_config.json`；
- 两张 SVG；
- `TASK16_SHA256SUMS`，并入 SHA256SUMS。任务06–15 的清单仍要校验通过。
- 镜像用 `rsync -a`，不带 `--delete`。

## 6. 时间估计（Claude）

| 阶段 | 估计 |
|---|---|
| P | 约 30–40 min |
| 闸门 | 约 10 min |
| 主运行 | 约 1 h（任务12 实测 53 min，加 r3 约 4 min） |
| 报告 | 约 30 min |
| **合计** | **约 2–2.5 h** |

§4 在任务14 F 完成后再做，约 20 min。
