# 任务14：DeepSeek API 子集实验（换读者复核 + API 压缩器基线）— 5880，不用 GPU

Claude 2026-11-23 22:10 写。Codex 执行、Claude 核查。动手前通读全文。下面的判据、提示词和设置都是预注册的，看到结果后不许改。

## 0. 为什么做、做什么

**来由**
- 11-23 用户要求"请你用deepseek的API来做对应的实验……最好在低峰期做实验"，随后选了方案 1（子集，总花费上限 ¥8），并同意下面这行：
  - 用 `deepseek-flash`，关闭思考，错峰价；总花费上限 ¥8；外发数据是 MuSiQue 的维基段落和问题（CC BY 4.0，不含个人信息）；key 放在 5880 的 600 权限文件里，不写进任何日志。
- 账户余额 ¥9.40（22:07 查询）。这把 key 和其他研究线共用，**上限 ¥8 是硬上限**。

**对应 PLAN_v2 §7 的两项**（此前因为没有 API 额度没做）
- **A. 换读者复核**："本地用 Qwen3-8B，再用一个 API 模型在子集上复核"。
- **B. API 压缩器基线**："用同样的提示词让 API 大模型输出 add / drop"。

**不在范围内**
- 不用 GPU；不碰任务13 的流水线（PID 3166200 及其子进程）和 `grpo_r3/` 目录。
- 不碰锁定测试集 `eval/test_dev_v1/`。
- 不调用 `deepseek-flash` 以外的模型；不开思考模式；不用联网搜索等服务端工具。
- PLAN §8 真实 Agent 测试不在本任务内。

## 1. 机器、资源、硬规则

**API 与 key**
- key 文件：`/root/.config/ctx-compressor/deepseek.env`（600 权限），变量名 `DEEPSEEK_API_KEY`。
  - 只在进程内读取；不打印、不写进任何日志、receipt、报告、测试快照或 shell 历史；不要 `set -x`；不复制到别处。
  - 交付前对所有新产物做一次检查：不含 `sk-` 开头的串，也不含 key 原文。结果写进报告。
- 端点：`https://api.deepseek.com/chat/completions`（5880 直连，不走代理）。余额查询 `GET /user/balance` 免费，每个阶段开始前和结束后各查一次并记录。
- 固定请求参数：
  - `model="deepseek-flash"`，`thinking={"type":"disabled"}`（OpenAI SDK 用 `extra_body`，或直接写进 HTTP body）；
  - `temperature=0`，`stream=false`；
  - 读者 `max_tokens=48`，压缩器 `max_tokens=256`（同 held-out 评测）。
  - 每次调用记录 `id`、`model`、`system_fingerprint`、`usage` 全部字段（含 `prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`）、`finish_reason`、原始 content、请求开始和结束的 UTC 时间、尝试次数。

**错峰**
- 高峰定义（官方定价页）：北京时间周一至周五 9:00–12:00、14:00–18:00。其余时段为空闲，含周末和法定节假日全天。
- 法定节假日不做特判，工作日一律按上面的高峰时段算（保守）。
- **每次调用前**检查北京时间。落在高峰时段就暂停，睡到时段结束再继续，并在状态文件里写 `paused_peak`。
- 本任务预期全部在 11-24 09:00 前跑完。

**花费上限 ¥8（含小样）**
- 单价（元 / 百万 token）：

  | 时段 | 输入缓存命中 | 输入缓存未命中 | 输出 |
  |---|---|---|---|
  | 空闲 | 0.02 | 1 | 4 |
  | 高峰 | 0.04 | 2 | 8 |

- 账本 `ledger.jsonl`：按 usage 和调用时段逐次累计花费。
- **每次调用前**做上限检查：已花 + 本次最坏花费 > ¥8.00 就停下，状态写 `stopped_cost_cap`。本次最坏花费 = 估计输入 token × 未命中价 + max_tokens × 输出价。
- 余额查询低于 ¥1.00 也停。余额的减少量与账本对账，差异写进报告（余额是共用的，别的线可能同时在花）。

**错误处理**
- 429、5xx、超时：指数退避（2、4、8、16、32 s），最多 6 次，全部记录。
- 402（余额不足）或其他 4xx：立即停下，报告。
- 并发：最多 8 条轨迹同时进行。同一轨迹内的调用按顺序发：读者的 4 道题按 q0→q3，压缩器逐步。

**其他（同任务11 / 12 §1）**
- 大文件放 `/root/userfolder/ctx-compressor/deepseek_api/`，小结果放 `eval/deepseek_api/`。
- `/` 下新增 ≤ 300 MB；不装包（缺 `openai` 包就直接用 `urllib` / `requests`，都没有就打印 `NEED:` 停下）。
- 不删、不覆盖；不写任务03–13 的目录；不碰 `mirror/.git`、不 commit；只按确切 PID 杀进程；README / NOTES / HANDOFF 只追加。
- 每次发最终消息前关掉所有子代理。

## 2. 子集（预注册）

- 留出任务编号排序后，用 `random.Random(20261127).sample(sorted_task_ids, 60)` 抽 60 个，记为列表 S60（保持抽样顺序）。
- S30 = S60 的前 30 个。
- 只用中档预算（mid，2814）。

| 部分 | 对象 | 任务 | 格 | 题次 / 轨迹 |
|---|---|---|---|---|
| A | oracle、`grpo_r2:checkpoint-125`、`sft_v3:checkpoint-3000` 的已有最终上下文 | S60 | reselect\|mid、append\|mid | 每个策略 480 题次 |
| A | `no_compress` | S30 | 一份（6 格上下文相同，任务12 已核） | 120 题次 |
| A | `no_context` | S60 | — | 240 题次 |
| B | DeepSeek 当压缩器，闭环 | S30 | reselect\|mid、append\|mid | 60 条轨迹（约 560 步）|
| B | DeepSeek 当读者，读 B 的最终上下文 | S30 | 同上 | 240 题次 |

## 3. A：换读者复核

**提示词**
- 与任务12 逐字相同：单条 user 消息，没有 system 消息，模板和上下文渲染都来自 `src/reader.py`。
- 直接复用任务12 构造提示的代码，不复制、不修改。
- **身份检查**：A 的每个提示的 `prompt_sha1`，必须等于任务12 `reader_heldout/predictions.jsonl` 里同键（strategy, mode, tier, task, question_index）的值。不一致 > 0 就停下。

**解析与评分**
- `src/reader.py::parse_answer` 和 `src/qa_metrics.py`，不改。

**报告（各附 n）**
1. 每个对象：DeepSeek 的 F1 / EM，并列同一批提示上 Qwen3-8B 的 F1 / EM（取自任务12），以及两个读者的逐题 F1 配对计数。
2. **主比较**，两个读者各报一次：
   - r2 − v3：F1 差、EM 差、逐题配对计数（r2 高 / 相等 / v3 高）、EM 2×2；
   - oracle − no_compress（S30）：F1 差、EM 差、逐题配对计数。
3. 按"该题证据是否齐全"分组的 F1。
4. **无关段**：在 r2 与 v3 该题证据都齐全的题上，按 r2 比 v3 多留的非 gold 段数分组（≤0 / 1 / ≥2；非 gold = 不是该任务任何一道题的 gold），报两个读者的平均 F1 差和 n。
5. DeepSeek 输出：`finish_reason` 分布；顶到 48 token 的条数；空答案的条数。

## 4. B：API 压缩器基线

**闭环**
- 沿用任务11 F（held-out 闭环评测）的全部代码：
  - 状态构造、编号映射、`render_input` 渲染；
  - 解析、四类非法判定（invalid_json / invalid_candidate_id / append_drop / over_budget）；
  - 执行动作（非法 = 本步什么都不改）、指标和成本计算。
- 只把"生成"这一步换成 DeepSeek 调用。不复制、不修改共享代码；需要的话写新的 policy 包装。
- **身份检查**：每条轨迹 step 0 的 user 文本 sha256，必须等于 r2 ckpt-125 rollout 同 (task, mode, tier) 的 step-0 `user_text_sha256`。不一致 > 0 就停下。

**消息**
- system 消息逐字如下，预注册不许改：

```
你是一个多步检索 Agent 的上下文压缩器。每一步你会看到：任务问题、模式、预算、已用 token、"已保留"的片段和本步"新返回"的片段，每个片段标有编号和 token 数。
你要决定：新返回的片段里保留哪些（add），已保留的片段里丢掉哪些（drop），使保留下来的内容包含回答全部任务问题所需的证据，同时保留片段的 token 总数不超过预算。
只输出一个 JSON 对象，不要任何其他文字，也不要代码块标记，格式：{"add": [编号, ...], "drop": [编号, ...]}
规则：add 只能是"新返回"里的编号；drop 只能是"已保留"里的编号；模式为 append 时 drop 必须为空；执行后保留片段的 token 总和不能超过预算。不合法的输出视为本步什么都不改。
```

- user 消息：与我们模型完全相同的 `render_input` 文本。

**解析**
- 用与我们模型相同的严格解析，不做任何放宽。
- 诊断项（只统计，不改判）：去掉首尾代码块标记或首尾空白后就能合法的条数。

**读者**
- DeepSeek 当读者，读 B 的 60 条轨迹的最终上下文（240 题次）。
- 提示按任务12 的模板构造，参数同 A。

**报告（每格 n = 30 个任务，并报两格合并）**
- 证据丢失率、任务末召回、单步召回；
- 非法率及按类型拆分，加代码块诊断；
- 每步 add 数分布、`used_after` 均值；
- 缓存口径和不开缓存的成本；
- DeepSeek 读者的 F1 / EM。
- 同一批 30 个任务 × 2 格上的对照行，用已有 rollout 和 A 的结果算：r2 ckpt-125、v3 ckpt-3000、oracle、bm25。
  - 丢失率、召回、成本：取自 rollout；
  - DeepSeek 读者 F1：r2 / v3 / oracle 取 A 的结果；bm25 不读（省钱），只报压缩指标。
- 每次调用的延迟（P50 / P90）、输入 token、缓存命中率。

## 5. 流程

### P（CPU，现在开始）
- 新文件：`scripts/deepseek_api_*.py` 和测试。
- 测试至少覆盖：
  - 高峰判断：周一 08:59 闲、09:00 忙、11:59 忙、12:00 闲、13:59 闲、14:00 忙、17:59 忙、18:00 闲，周六 10:00 闲；
  - 账本上限：模拟 usage，跨过 ¥8 前一次调用被拦下；
  - 402 立即停、429 退避；
  - A 的提示身份检查、B 的 step-0 身份检查；
  - 拒绝覆盖；
  - 产物里不含 key。
- 全量 CPU 测试必须全过。

### S（小样，约 40 次调用，约 ¥0.2）
- 内容：
  - A：S60 前 2 个任务 × reselect|mid × oracle（8 题次）；no_compress 取 S30 第 1 个任务（4 题次，用来看缓存命中）。
  - B：S30 第 1 个任务 × 2 格闭环（约 19 步），加 DeepSeek 读者 8 题次。
- 输出放 `deepseek_api/smoke/`。
- 跑完**停下**，打印下面这行，不打 DONE 行：
  - `WAITING: 任务14 小样完成（实际花费 ¥x.xx，输入 token 命中 / 未命中 a / b，B 非法 k/m 步，余额 ¥y.yy），等 Claude 放行`
- 同时报告：
  - 按小样实测单价重估的全量花费；
  - 有没有被高峰暂停。

### F（全量，Claude 发"放行"后才开始）
- 输出放 `deepseek_api/full/`，从头跑全部 §2 的内容。小样的结果不并入。
- 账本累计小样的花费，上限 ¥8 含小样。
- 跑完写：
  - `eval/deepseek_api/deepseek_api_report.md`（中文，每个数附 n）；
  - `deepseek_api_results.json`；
  - `run_config.json`（参数、system_fingerprint 分布、时段、余额前后、账本合计）；
  - `TASK14_SHA256SUMS`，并入 SHA256SUMS。任务06–12 的清单仍要校验通过。
- 镜像用 `rsync -a`，不带 `--delete`。
- 最终消息用中文，包括 A 的两组主比较（两个读者）、B 每格的丢失率 / F1 / 非法率、总花费和余额。以这一行结束，不打 DONE 行（任务13 收尾时一起打）：
  - `WAITING: 任务14 完成（总花费 ¥x.xx；DeepSeek 读者 r2−v3 F1 差 …；DeepSeek 压缩器合并丢失率 …）`

### C（推迟：GPU，等 Claude 消息）
- 任务13 跑完后，Claude 会让你用任务12 的脚本，给 B 的 60 条轨迹补一行 Qwen3-8B 读者 F1：
  - 过 GPU 闸门；
  - 输出 `reader_heldout/deepseek_compressor/`；
  - 小结果并入 `eval/deepseek_api/`。
- **现在不做。**

## 6. 时间与花费估计（Claude）

**时间**

| 阶段 | 估计 |
|---|---|
| P | 约 1–1.5 h |
| S | 约 10 min |
| F | 全量 < 1 h（8 并发；读者每次约 1 s，压缩器约 1–2 s）|

**花费**（按 Qwen token 数估，最坏情况 = 缓存全不命中）

| 项 | 最坏花费 |
|---|---|
| A 压缩上下文：3 × 480 × 约 1.4k token | ≈ ¥2.0 |
| A no_compress：120 × 约 2.0 万 token | ≈ ¥2.4 |
| B 压缩器：约 560 步 × 约 3.6k token | ≈ ¥2.0 |
| B 读者 | ≈ ¥0.35 |
| 输出 | ≈ ¥0.2 |
| 小样 | ≈ ¥0.2 |
| **合计** | **≈ ¥7.2** |

缓存命中后，估计实际约 ¥5–6。
