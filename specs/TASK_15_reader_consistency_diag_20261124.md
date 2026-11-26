# 任务15：读者一致性诊断（cascade attention 假设）— 5880 GPU 1

Claude 2026-11-24 10:30 写。Codex 执行、Claude 核查。动手前通读全文。下面的臂设置、指标和判读规则都是预注册的，看到结果后不许改。

## 0. 为什么做

**来由**
- 11-24 用户看完任务13 后回"**用方案1吧**"：先诊断读者为什么不稳定，再决定重跑范围。
- 本任务**只诊断**，不重跑任务12 / 13 的正式结果。重跑范围诊断后由用户定。

**Claude 复核时发现的问题**
- 复核脚本和输出见 `claude_checks/check_reader_r3_out_20261124.txt`、`claude_checks/reader_r3_consistency_out_20261124.txt`。
- **跨运行不一致**：任务13 G try2（r3 ckpt-250）和任务12 的 r2 ckpt-125 有 2,116 道题，最终上下文相同，prompt token 也逐个相同。
  - 两边 raw_output 不同的有 397 道；
  - 这 2,116 道题的 F1 是 53.84 对 58.40。
- **同次运行内不一致**：token 完全相同的请求分组后，输出不止一种的组：
  - try2 有 485/1,612 组（30.1%）；
  - 任务12 有 383/5,812 组（6.6%）。
- **输出形态变了**：输出 ≥ 20 token 的条数，try2 是 383，任务12 r2 是 52。
- **运行条件**：
  - 引擎参数、日志与任务12 相同；
  - 唯一明显不同的是批次构成：任务12 每个任务提交一批约 200 条，混了 8 个策略；try2 每批只有 r3 的 24 条（6 格 × 4 题）。
- **任务13 的 −5.4 pp 读者 F1 差因此不可信。**

**假设（未验证）**
- vLLM 0.11.0 的 cascade attention 在同批请求共享长前缀时启用，让输出偏移。
  - 源码：`vllm/v1/attention/backends/flash_attn.py`，`use_cascade_attention`。
  - 启用条件：共享前缀 ≥ 256 token，且请求数 ≥ 8。
  - 关闭开关：`disable_cascade_attn`，见 `config/model.py:198` 和 `engine/arg_utils.py:354`。

## 1. 机器、资源、硬规则

**与任务12 §1 相同**
- 只用 GPU 1，闸门阈值不改。
- MRI_SAM 的进程和目录不碰。
- `/` 下新增 ≤ 300 MB，不装包。
- 不删、不覆盖、不碰 `mirror/.git`、不 commit。
- 只按确切 PID 杀进程。
- 发最终消息前关掉所有子代理。

**本任务另加**
- **不改 vLLM 安装目录里的任何文件**，只通过 `LLM(...)` 的参数控制。
- **TMPDIR 用短目录**：`/root/userfolder/ctx-compressor/rt/g15`，启动前像 try2 一样写 `ipc_check.json`，IPC 路径要 < 100 字节。
- **不写任务03–14 的目录。**
  - 任务13 的报告和 `TASK13_SHA256SUMS` 不改，改了清单就会失效。
  - 只在 NOTES / HANDOFF 末尾追加一句："任务13 的读者 F1（r3 50.709）待复测，见任务15"。
- **与任务14 F 互不干扰**：任务14 F 不用 GPU，任务14 C 继续推迟。

## 2. 两个臂

**两臂共用**
- 顺序跑，过一次闸门后连着跑完；每个臂单独一个进程、一个引擎。
- 引擎参数与 try2 相同：取 `eval/reader/run_config.json`，生成参数也相同（greedy，max_tokens 48，seed 20261117）。

| 臂 | 名称 | 输入 | 批次 | 和 try2 的唯一区别 |
|---|---|---|---|---|
| a | `cascade_off_r3` | try2 的 5,976 个请求 | 与 try2 完全相同：按任务分批，每批 24 条，顺序不变 | `LLM(..., disable_cascade_attn=True)` |
| b | `mixed_r2r3` | r2 ckpt-125 的 5,976 个请求 + r3 的 5,976 个请求 | 每个任务一批 48 条：先 r2 的 24 条，再 r3 的 24 条 | 批次构成。cascade 保持默认（开） |

**臂 a 的输入**
- 直接复用 try2 的 requests 输入，sha256 必须等于 `5f584e73b5bcb48d6670da18951d02447d52f8064778f59d98fce11a35a88b32`（try2 的 `receipt_inputs.json`）。

**臂 b 的输入和顺序**
- r2 的请求用任务12 的构造代码重建，不复制、不修改。
- **身份检查**：每条请求的 `prompt_sha1` 要等于任务12 `reader_heldout/predictions.jsonl` 里同键 `(strategy, mode, tier, task, question_index)` 的值，不一致 > 0 就停。
- 每个策略内部的请求顺序与任务12 / try2 相同，任务顺序同 try2。

**两臂都要做**
- **开头先跑一致性小样**：任务03 的 40 条，单独一批，同任务12 §3。记 raw_output 不一致数。
- **记录 cascade 的实际设置**：把引擎实际生效的 `disable_cascade_attn` 值写进 `run_config.json`。
  - 先从引擎的 vllm_config 读；
  - 读不到就从启动日志里的 config 行截取；
  - 两种都拿不到，就写明"未取得"。

## 3. 指标（每个臂、每个策略各报一份，每个数都附 n）

1. **同次运行内一致性**
   - 把 token_ids 完全相同的请求分成一组，只看组大小 ≥ 2 的组。
   - 报"raw_output 不止一种的组数 / 组数"。
   - 口径同 `claude_checks/reader_r3_within_run.py`，只读参考，不改它。
   - 臂 b 分三种报：r2 内、r3 内、r2 与 r3 合在一起分组。
2. **与任务12 对照**
   - 2,116 道题：r3 中与任务12 r2 同键 prompt token 完全相同的那些题，报 raw_output 与任务12 不同的条数（try2 是 397）。
     - 口径同 `claude_checks/reader_r3_cmp_tokens.py`；
     - 同时报这 2,116 题的 F1。
   - 臂 b 的 r2：5,976 题中 raw_output 与任务12 r2 不同的条数。
3. **与 try2 对照**：r3 的 5,976 题中 raw_output 与 try2 不同的条数，以及这些题的平均输出 token 数（两边各一个）。
4. **一致性小样**：40 条中的不一致数（任务12 是 1/40）。
5. **F1 / EM**
   - 合并：n = 5,976；
   - 每格：n = 996；
   - 按"证据是否齐全"分组。
6. **输出形态**
   - 平均输出 token 数；
   - ≥ 20 token 的条数（try2 383，任务12 r2 52）；
   - 顶到 48 token 的条数。
7. **臂 b 的配对比较**
   - r2 − r3 的 F1 差和 EM 差；
   - 逐题配对计数：r2 高 / 相等 / r3 高。
8. **资源**：每臂耗时、峰值显存。

## 4. 判读规则（预注册，Claude 用；Codex 只报数，不下结论）

- 记 p_a、p_b 为臂 a、臂 b 中 r3 的同次运行不一致率。参考值：try2 30.1%，任务12 6.6%。

| 结果 | 判读 | 后续 |
|---|---|---|
| p_a ≤ 10% | 支持"cascade attention 是主因" | 之后的重跑都加 `disable_cascade_attn=True` |
| p_a > 20% 且 p_b > 20% | 假设被否 | 另查原因，暂不重跑 |
| 其他 | 如实报 | Claude 再给方案 |

- 无论哪种结果，重跑范围（只重跑 r2 + r3，还是任务12 全部）都由用户定，**本任务不做**。

## 5. 流程

**P（CPU）**
- 新文件：`scripts/reader_diag_cascade.py` 加测试；只 import 任务12 的模块，不改。
- 测试至少覆盖：
  - 臂 a 的批次与 try2 逐条一致；
  - 臂 b 的批次顺序；
  - r2 的 prompt_sha1 身份检查；
  - 一致性指标在小 fixture 上与 claude_checks 两个脚本口径一致；
  - 拒绝覆盖；
  - IPC 路径长度检查。
- 全量 CPU 测试必须全过。

**R（GPU）**
- 顺序：闸门 → 臂 a → 臂 b。
- 运行方式：detached；pidfile；状态文件；每阶段一份 receipt；不自动重试。

**产物**
- 大文件：`/root/userfolder/ctx-compressor/reader_diag/{cascade_off_r3,mixed_r2r3}/`。
- 小结果放 `eval/reader_diag/`：
  - 中文报告 `reader_diag_report.md`；
  - `reader_diag_results.json`；
  - `run_config.json`；
  - `TASK15_SHA256SUMS`，并入 SHA256SUMS。任务06–13 的清单仍要校验通过。
- 镜像用 `rsync -a`，不带 `--delete`。

## 6. 时间估计（Claude）

| 阶段 | 估计 |
|---|---|
| P | 约 40 min |
| 闸门 | 约 10 min |
| 臂 a | 约 6 min（try2 推理实测 5.5 min） |
| 臂 b | 约 12 min |
| 报告 | 约 20 min |
| **合计** | **约 1.5 h** |
