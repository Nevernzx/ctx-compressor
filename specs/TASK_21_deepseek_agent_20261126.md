# 任务21：真实 Agent 换 DeepSeek 主模型（子集）— 5880 GPU 1 + DeepSeek API

Claude 2026-11-26 14:30 写，**Claude 自己实现和执行**（用户 11-26："可以，你来实现，不用交给gpt"）。下面的设置和指标在看到任何结果前定下，之后不改。

## 0. 来由

- 用户先选了报价里的方案 1，又同意了这一行：`deepseek-flash`，关思考、错峰价；小样 ≤ ¥0.5，总花费硬上限 ¥8（含小样）；外发数据是 MuSiQue（CC BY 4.0）、HotpotQA（CC BY-SA 4.0）、2Wiki（Apache-2.0）公开 dev 集的问题和维基段落；key 用 5880 上现有的 600 权限文件，不写进日志。
- 要回答的问题：任务20 的主模型是 Qwen3-8B。换成更强的 API 主模型后，r4 相对 RR-τ、不压缩、LLMLingua-2 的差距会怎么变。
- 开工时余额 ¥24.94（11-26 13:55 查询），这把 key 三条线共用。

## 1. 设置（和任务20 的差别只有主模型）

- **主模型**：`deepseek-flash`，`thinking: disabled`，temperature 0，max_tokens 256，stop `["\nObservation"]`。
  - 用 DeepSeek 的前缀续写接口（`/beta/chat/completions`，最后一条 assistant 消息带 `prefix: true`）。
  - 历史拼法和任务20 逐字一致：一整段 assistant 文本，每轮是 `Thought:` + 输出 + `\nObservation:\n` + 观测 + `\n\n`，末尾预填 `Thought:`。
  - 强制作答时：assistant 文本结束，加一条 user 消息 `Search limit reached. Give your final answers now.`，再预填 `Thought:`。
- **原样沿用任务20**：
  - system 和 user 提示词、K = 3、BM25 top-20、最多 12 次搜索、连续 2 次格式无效就强制作答；
  - 长度上限按 Qwen3 token 计（40,448）、各数据集预算 B_d、观测渲染规则；
  - 压缩器全部沿用：r4 ckpt-500、RR-τ（任务17 ckpt-5300，τ = 0.5）、LLMLingua-2（rate 0.33 加 latest 装箱）、no_compress、oracle。
- **行**（6 行）：no_compress、oracle、LLMLingua-2、RR-τ reselect、r4 reselect、r4 append。
- **范围**：任务20 正式题里每个数据集的前 40 个任务（按任务20 的抽样顺序），共 120 个任务、360 题。
- **小样**：任务20 冒烟文件里每个数据集的前 3 个任务（不和正式题重叠），6 行都跑。
- **小样之前的两步**（14:40 补记，未看任何结果）：
  - 干跑（dryrun）：假的 API 回复（每题搜一次再 finish）、真的压缩器，每个数据集 1 个冒烟任务，只验证 GPU 路径，不联网不花钱，账本单独放在 `agent_ds/dry/`。
  - 探针（probe）：18:00 后发 3 次真实调用（首轮、续写、强制作答三种消息形状），确认前缀续写的返回格式，上限 ¥0.1。
- **顺序**：每 12 个任务一组（每个数据集各 4 个），一组 6 行一起跑完再跑下一组。撞到上限时，已完成的组仍然全部配对。

## 1.5 偏差（小样发现，在任何行间比较之前定）

- **11-26 14:41 小样作废**，原因有两个：
  - DeepSeek 在 12/170 轮里一次连写多组 Thought/Action，不写 `Observation`，所以 stop 不生效，这些轮按格式无效处理；连续两次无效后，它不搜索直接作答。
  - 最坏预留按 UTF-8 字节数乘高峰价估，16 个并发同时预留，加上已花的 ¥0.28，超过了小样上限 ¥0.5。
- **修正**：
  - stop 改为 `["\nObservation", "\nThought:"]`，每轮只保留第一组 Thought/Action，即任务20 的"每轮恰好一个动作"；6 行统一适用。
  - 预留改为 min(字节数, 1.5 × Qwen3 token 数 + 512)。
  - 用同样的 9 个任务重跑小样（smoke2），上限 ¥1。
  - 作废的小样产物保留，收尾记录在 `eval/agent_ds/smoke/closeout.json`。

- **11-26 14:57 第二版小样（smoke2）停下**：
  - 结果：253 次成功调用，无法解析 0 次（停止符修正生效）。
  - 停下的原因：1 次 HTTP 400。原样重发后拿到错误信息 `Content Exists Risk`，即 DeepSeek 内容审核拦截，当时上下文里有一段 Free China / Falun Gong 的维基段落。
- **全量前定下的规则**（还没看任何行间比较）：
  - 被拒的 episode 直接结束，答案记空串；不重试，不改语料。
  - 某个任务只要有任意一行被拒，就整体退出所有配对比较。
  - 每行的被拒次数单独报告；另报"被拒按 F1 = 0 计"的敏感性数字，只并列，不作主结论。
  - 不再跑第三版小样，直接全量：用户 11-26 说"小样跑完直接跑全量吧"，总上限 ¥20。

## 2. 花费规则

**11-26 14:30 后用户改了两条（都在看到小样结果之前）**：
- "直接开始，不用错峰"：允许高峰时段调用，账本按调用时刻的时段计价（高峰是错峰的 2 倍），预留一律按高峰价。
- "小样跑完直接跑全量吧"：小样核查无异常就直接启动全量，不再等放行。
- "调整上限到20"：总上限从 ¥8 改为 ¥20（小样上限 ¥0.5 不变）。下面写的"¥8"以此为准改读成 ¥20。

- 只在错峰时段发请求（北京时间工作日 9–12、14–18 点之外，加上周末），高峰时自动暂停。
- 本任务单独记账（`agent_ds/ledger.jsonl`）。每次调用前按最坏花费预留：已花加预留超过 ¥8 就停；小样阶段超过 ¥0.5 就停。
- 余额每个阶段前后各查一次，余额 < ¥1 就停；402 或其他 4xx 立即停。
- 交付前扫描所有新产物，确认不含 key。

## 3. 预注册指标（只报数，附 n，单次运行不算 CI）

1. 每行、每数据集和合并：F1、EM、搜索次数 / 任务、自己 finish 的比例、格式无效轮数、检索覆盖齐全率（gold 全被搜到过）、最终证据齐全率、丢证据数（搜到但没留住）、平均保留段数。
2. **配对比较**（逐题）：r4 reselect − RR-τ reselect、r4 reselect − no_compress、r4 append − no_compress、r4 reselect − LLMLingua-2、oracle − r4 reselect。每组报 ΔF1、ΔEM、逐题高 / 平 / 低，按数据集拆开。
3. **成本**：
   - 每行按我们的成本模型（Qwen3 token，前缀命中 0.05、未命中 1.25）算开缓存和不开缓存两种成本；
   - DeepSeek 实际计费：每行每任务的 ¥、命中和未命中 token；
   - 每行"预测命中率"（我们按 token 前缀算的）对照"实际命中率"（DeepSeek 返回的）。
4. **和 Qwen3-8B 对照**：同 120 个任务上，任务20 的 Qwen 结果和本次逐行并列。比较 F1、检索覆盖齐全率、搜索次数，以及每行的 DeepSeek − Qwen 配对 ΔF1。
5. 只用配对完整的任务（6 行全部完成）算比较；不完整的任务单独列出。

## 4. 产物

- 代码：`scripts/agent_ds_client.py`、`scripts/agent_ds_run.py`、`scripts/agent_ds_report.py`、`tests/test_agent_ds.py`（只新增，旧文件只 import 不改）。
- 大文件：`/root/userfolder/ctx-compressor/agent_ds/{probe,smoke,main}/`；运行缓存：`/root/userfolder/ctx-compressor/rt/ads/`。
- 小结果：`eval/agent_ds/`（报告、results JSON、run_config、SHA256SUMS）。
- 硬规则同任务20：只用 GPU 1 且先过闸门；`/` 存储四条规则；不碰 MRI_SAM；不写旧任务目录（用测试证明）；不装包；不 commit。
