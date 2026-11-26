# 任务20：真实 Agent 测试（PLAN §8）：ReAct 搜索 Agent + 压缩器，MuSiQue / HotpotQA / 2Wiki — 5880 GPU 1

Claude 2026-11-25 10:55 写。Codex 执行、Claude 核查。动手前通读全文，并通读同时派出的任务19（共用 GPU 1，顺序见任务19 §7）。下面的设置和指标都是预注册的，看到结果后不许改。

## 0. 为什么做

- 11-25 用户回"**1和2都做吧**"。2 = PLAN §8 真实 Agent 测试。
- PLAN §8 原文："这一节决定项目能不能叫'Agent 上下文压缩'，至少跑 200 个任务。"
  - 模拟轨迹用的是标准子问题，真实 Agent 要自己写查询、会走弯路。
  - 跨数据集泛化在这里测：HotpotQA 和 2Wiki 对所有方法都是没训练过的数据。
  - 和现成压缩器（LLMLingua-2 / Provence）的比较以这里为准。
- **主模型用本地 Qwen3-8B**（PLAN §8 允许"API 模型或本地 Qwen3-8B"）。不花钱，与读者同一个模型。API 主模型不在本任务范围。
- **不在范围内**：归档取回工具（`search_archive`）、无标注适应、API 主模型。

## 1. 机器、资源、硬规则

- 同任务19 §2：只用 GPU 1、过闸门、MRI_SAM 不碰；`/` ≤ 300 MB；不装包；不删不覆盖；不碰 `.git`、不 commit；只按确切 PID 杀；Qwen3-8B 一律 `disable_cascade_attn=True` 并核对生效值；发最终消息前关掉子代理。
- 新目录：大文件 `/root/userfolder/ctx-compressor/agent_v1/`；小结果 `eval/agent_v1/`。不写任务03–19 的目录。
- **下载**（只下这三样，全部不走 3080 的代理）
  1. LLMLingua-2：ModelScope `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`（MIT，约 2.25 GB），到 `models/llmlingua-2-xlm-roberta-large-meetingbank/`。用 `curl -L -C - --retry` 断点续传（任务17 的教训）。
  2. `llmlingua` 源码：从国内 PyPI 镜像（aliyun 或 tuna）下最新版的 wheel，解压到 `/root/userfolder/ctx-compressor/vendor/`，用 PYTHONPATH 在 grpo 环境里 import。**不 pip install。** import 时缺包就打 `NEED:` 停下。
  3. （可选行）open-provence：ModelScope `hotchpotch/open-provence-reranker-v1-gte-modernbert-base`（MIT，英文，约 600 MB），到 `models/open-provence-en-gte/`。
     - 这是第三方对 Provence 的开源复现，**不是 naver 官方版**（官方版 ModelScope 上没有）。报告里必须这样写。
     - 只有在 grpo 环境里用 `trust_remote_code` 能直接加载、不需要新装包、不需要再下别的数据（如 nltk 语料）时才跑。做不到就记"未跑"和原因，**不打 NEED**，其余照常。
- 每个下载记录 revision、各文件 sha256 和 license，写进 `eval/agent_v1/model_cards.json`。

## 2. 数据（CPU）

**每个任务 K = 3 道题。每个数据集 100 个任务（300 题），三个数据集共 300 个任务。**

| 数据集 | 题源 | 抽样 | 语料 |
|---|---|---|---|
| MuSiQue | 锁定测试集的 1,804 题（`test_tasks.jsonl`，只读） | `random.Random(20261125).sample(sorted(题 id), 300)`，按抽样顺序每 3 题一组 | 任务19 重建的 dev 语料（21,100 段） |
| HotpotQA | `hotpot_dev_distractor_v1.json`（7,405 题） | 同上，种子 20261126 | 全部 dev distractor 题的 context 段落去重合并 |
| 2Wiki | `2wikimultihopqa/dev.json` | 同上，种子 20261127 | 全部 dev 题的 context 段落去重合并 |

- 段落去重键：(title, text)。HotpotQA / 2Wiki 的 text = 句子按原样拼接（HotpotQA 句子自带前导空格，直接连接；2Wiki 用一个空格连接），写进 run_config。
- **gold**：MuSiQue 用 `question_gold`；HotpotQA / 2Wiki 用 `supporting_facts` 里出现的标题对应的段落（该题自己 context 里的那一段）。
- 答案：MuSiQue 用 answer + aliases；HotpotQA 用 answer；2Wiki 用 answer（文件里有 aliases 就一起用）。EM / F1 用 `src/qa_metrics.py`。
- token 数用同一个 Qwen3 分词器。
- **预算**（PLAN §6 的中档规则）：B_d = round(1.5 × 该数据集 100 个任务"每任务 gold token 总和"的 P90)。跑任何东西之前算好，写进 run_config。
- **冒烟任务**（与正式题不重叠）：每个数据集 10 个任务。
  - MuSiQue：从 `excluded_ids.json` 里的 dev 题抽（这些题以前只用于可行性试验，不在锁定集内），语料同上。
  - HotpotQA / 2Wiki：从未被正式抽样选中的 dev 题里抽，种子各加 100。
- 报告：每个数据集的语料段数、每任务 gold token 的 P50 / P90、题型分布（HotpotQA 的 bridge / comparison，2Wiki 的四种类型）、yes/no 答案占比、B_d。

## 3. Agent（预注册）

- ReAct 循环，只有一个工具 `search(query)`：bm25s 检索该数据集语料，取 top-20。索引和分词同 `build_corpus`。
- **主模型**：Qwen3-8B，`eval/reader/run_config.json` 的引擎参数加 `disable_cascade_attn=True`，enable_thinking=False，greedy，每轮 max_tokens 256，stop 设为 `["\nObservation"]`。
  - 为了吞吐和与压缩器共卡，允许改 `max_num_seqs`（≤ 64）和 `gpu_memory_utilization`，实际值写进 run_config。其余参数不改。
- **提示**：system 说明任务（回答全部 K 道题，每轮一个动作）；user 列出编号的 K 道题。每轮模型输出：

```
Thought: <一两句>
Action: search[<query>]
```

或

```
Thought: <一两句>
Action: finish[{"1": "<答案>", "2": "<答案>", "3": "<答案>"}]
```

- **历史是交错的**：每轮的 Thought + Action 后面跟一条 Observation，列出这次搜索返回、且**此刻仍被保留**的段落，格式 `[Pk] title: text`（k 是本 episode 内按首次出现编号）。
  - 段落显示在它第一次进入上下文的那次搜索下面；被丢掉的段落从原处删去；一条 Observation 里一段都不剩时写 `(no passages retained)`。
  - 已在上下文里的段落再次被搜到时不重复显示。
  - 这样"只追加"模式的前缀不变，"重选"模式丢段会让其后的缓存失效，与 PLAN §5 的成本模型一致。
- **上限**
  - 最多 4K = 12 次搜索。用满后，或连续 2 次格式无效后，追加一条 user 消息 `Search limit reached. Give your final answers now.`，再取一轮强制 finish。
  - 格式无效的一轮：Observation 写 `Invalid action format.`，算一轮、不算搜索。
  - 下一轮 prompt 会超过 max_model_len − 512（40,448 token）时：这次搜索结果不再加入，直接强制 finish，并计数。
- **答案**：finish 里的 JSON 按题号取；缺失或解析不了记空串（F1 = 0）。
- 提示词原文在 Phase P 写好，冒烟后冻结，sha256 写进 run_config。

## 4. 压缩方法（每行跑一遍完整的 Agent）

每次搜索后，压缩器拿到：K 道题、当前保留的段落（kept）、本次返回里不在 kept 中的段落（new）、预算 B_d、模式；输出新的保留列表。new 为空时不调用，保留不变（所有行同一规则）。

| 行 | 做法 |
|---|---|
| no_compress | 所有段落都留下，不受预算限制 |
| latest | `policies.latest_policy`，重选 |
| oracle | gold 段优先装满预算（`oracle_action` 的规则；gold = 该任务 K 道题的全部 gold），重选。用了标签，是上限 |
| RR-τ 重选 / 只追加 | 任务17 选中的重排 checkpoint，τ = 0.5，query = K 道题用 `"\n"` 连接，模板同任务17；策略直接 import 任务17 的实现 |
| LLMLingua-2 | 见下 |
| open-provence（可选） | 见下 |
| **r4 重选 / 只追加** | grpo_r4 ckpt-500 叠在 merged_start 上。prompt 用训练时的同一个渲染函数（只是题数是 K = 3），引擎设置同任务18 F；invalid 按 no-op |

**LLMLingua-2**（与问题无关的 token 级压缩）
- 每段的正文单独压缩：`compress_prompt([正文], rate=0.33, force_tokens=['\n', '?'])`，即 llmlingua README 里 LLMLingua-2 的示例设置。标题原样保留。
- 压缩后的段落按 latest 的规则装进预算（新的优先，旧的先被挤出）。
- 计数时段落"在上下文里"以压缩后的版本为准；另报压缩后上下文里含标准答案字符串的比例。

**open-provence**（可选，按问题剪句并给重排分）
- 每个新段落调用 `model.process(question=K 道题用 "\n" 连接, context=正文, threshold=0.1)`，得到剪过的正文和重排分。剪完为空的段落不保留。
- 用剪后的 token 数和重排分，按 `bm25_policy` 的排序规则（重选：kept + new）和 `_pack` 装进预算。

## 5. 预注册指标（只报数，不写"有效 / 无效"；每个数附 n；单次运行，不算 CI）

每个数据集分开报，再报三个数据集合并。

1. **每行的 Agent 最终 EM / F1**（每个数据集 n = 300 题）。
2. **配对比较**（逐题配对）：报 ΔF1、ΔEM、逐题高 / 平 / 低计数、EM 2×2。
   - r4 重选 − RR-τ 重选；
   - r4 只追加 − RR-τ 只追加；
   - r4 重选 − no_compress；r4 只追加 − no_compress；
   - r4 重选 − LLMLingua-2；
   - r4 重选 − open-provence（跑了才报）；
   - r4 重选 − latest；
   - oracle − r4 重选。
3. **过程**：每任务搜索次数（均值和分布）、在上限前自己 finish 的比例、强制 finish 数、格式无效轮数、超长强制 finish 数、重复查询比例。
4. **成本**（主模型输入）：
   - 每任务的 prompt token 总和（不开缓存）；
   - 开缓存：同一 episode 相邻两轮按 token 的最长公共前缀算命中，命中按 0.05、其余按 1.25 计（cost.py 的倍率）；
   - 输出 token；
   - 每个数据集一张成本–F1 散点 SVG。
5. **证据**（按题）：
   - gold 是否被搜到过（检索覆盖）；
   - 是否留在最终上下文里；
   - 证据齐全率；
   - 丢证据 = 搜到过但没留下。用来区分"Agent 没搜到"和"压缩器丢了"。
   - MuSiQue 另按 hop_index 报保留率。
6. **压缩器诊断**：r4 的非法 / 超预算比例；每行平均保留段数和无关段数；每次压缩调用的墙钟时间 p50 / p90（r4、RR-τ、LLMLingua-2、open-provence）。
7. **模拟与真实对照**（仅 MuSiQue）：并列任务19 测试集中档的 F1（同名的行）。题不同、设置不同，只并列，不配对，报告里写明。

## 6. 流程

**Phase P（CPU，与任务19 的 GPU 阶段并行）**
- 下载、§2 数据、Agent 驱动、各压缩方法的适配层、指标和成本计算。新代码全部写进新文件（如 `scripts/agent_v1_*.py`、`src/agent_v1.py`），旧模块只 import、不改。
- 测试至少覆盖：
  - 动作解析（合法、格式无效、finish JSON 缺题号）；
  - Observation 渲染（丢段后删除、不重复显示、全空时的占位）；
  - 每种方法都不超预算（no_compress 除外），只追加模式前缀不变；
  - 超长保护和搜索上限；
  - 缓存成本的最长公共前缀计算（玩具例子手算）；
  - RR-τ 适配层：给定 held-out 状态和任务17 的分数，动作与任务17 的 rollout 一致（抽 5 个状态）；
  - r4 适配层：同一状态下渲染出的 prompt 与训练时的渲染函数逐字节相同；
  - 数据抽样的种子和不重叠（冒烟 vs 正式、MuSiQue 冒烟 vs 锁定集）；
  - 拒绝覆盖。
- 全量 CPU 测试必须全过。

**冒烟（GPU）**
- 3 个数据集 × 10 个任务 × 全部行。
- 只允许修 bug（解析、渲染、驱动）。提示词只能为"模型不按格式输出"这一类问题改，不能为了分数改；每处改动都记进 run_config。
- 实测吞吐，按它估主运行的时长。
- 冻结提示词和代码（写 sha256），然后打印下面这行，**等 Claude 回复再开主运行**：

```
WAITING: 任务20 冒烟完成（每行格式无效率 / 强制 finish 率 / 超长数，主运行 ETA …）
```

**主运行（GPU）**
- 300 个任务 × 全部行。排在任务19 G3 之后（见任务19 §7）。
- 组织方式由 Codex 定（例如 8B 和压缩器各起一个常驻 vLLM 引擎，按轮推进所有 episode）。要求：
  - 前缀缓存能在同一 episode 的相邻两轮之间命中；
  - 同一时间 GPU 1 上只有本任务的进程；
  - 显存总占用写进 run_config。
- 运行方式：detached、pidfile、状态文件 `agent_v1/status.json`、每阶段一份 receipt、不自动重试、每 30 s 监控一次。每个 episode 的完整对话（每轮 prompt token 数、输出、动作、保留列表）落盘，便于事后核查。

## 7. 报告与交付

- `eval/agent_v1/agent_v1_report.md`（中文，每个数附 n）、`agent_v1_results.json`、run_config、model_cards.json、三张成本–F1 SVG。
- 报告写明：主模型是本地 Qwen3-8B；open-provence 是第三方复现（或写明未跑和原因）；MuSiQue 的题取自锁定测试集。
- `eval/agent_v1/TASK20_SHA256SUMS`，并入 SHA256SUMS。任务06–19 的清单仍要校验通过。README / NOTES / HANDOFF 只追加；镜像用 `rsync -a`，不带 `--delete`。
- **最终消息**（任务19、20 合并一条，中文）：
  - 任务19：主比较 a–f 的 ΔF1、按跳拆分要点、"见没见过"比例；
  - 任务20：每个数据集每行的 F1、r4 − RR-τ 和 r4 − no_compress 的 ΔF1、成本、open-provence 跑没跑。
  - 以这一行结束：

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务19+20 <一句话，含测试集 r4−RR-τ 和 Agent 合并 r4 重选−RR-τ 的 ΔF1>
```

## 8. 时间估计（Claude）

| 阶段 | 估计 |
|---|---|
| Phase P | 约 3–4 h（和任务19 的 GPU 阶段重叠；LLMLingua-2 下载约 30–60 min） |
| 冒烟 | 约 30–45 min |
| 主运行 | 约 4–8 h（9 行 × 300 任务，约 3 万次 8B 调用；以冒烟实测为准） |
| 闸门 | 每个阶段约 10–15 min |
| 报告 | 约 30–45 min |

任务20 的 GPU 部分排在任务19 之后，两个任务合计约 14–18 h 墙钟时间（含等我看冒烟）。
