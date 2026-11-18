<!-- 来源：https://claude.ai/artifact/JP2ihXyuzKiWfaZV15pQ7Y （rev 24）。2026-11-17 Claude 从 Claude Docs 导出成文本；表格是简化排版，有出入以原文档为准 -->

# Agent 上下文压缩器小模型：项目方案
2026-11-16 · @nevvv
v2（2026-11-16 修订）。 v1 的任务太短太容易：整条轨迹只有几千 token，压缩器比不出优势。本版把轨迹加长到几万 token，补上同数据训练的基线，成本按 prompt caching 计价，并要求训练前先做 2–3 天的可行性试验。
| v1 的问题
 | v2 的改动
 | 位置
 | 
| 轨迹只有几千 token，预算太松
 | 每步从整个语料库检索 top-20；4 道题拼成一个多目标任务
 | 第 3 节
 | 
| “token 减少 X%”其实由预算决定
 | 主结果改为预算–F1、成本–F1 两条曲线和证据丢失率
 | 第 7 节
 | 
| 基线没在同样的数据上训练
 | 加同数据微调的重排模型；与现成压缩器的比较放到跨数据集测试
 | 第 7、8 节
 | 
| 没回答为什么要用 RL
 | 写明 GRPO 应该赢在哪两个指标；加无标注适应实验
 | 第 4、7 节
 | 
| 成本没算缓存
 | 成本模型含缓存；新增“只追加”模式；输出改为 add / drop
 | 第 1、5 节
 | 
| 真实 Agent 测试是可选项
 | 改为必做；归档取回作为加分项
 | 第 8 节
 | 
| 没有开工前的验证
 | 新增可行性试验和通过标准
 | 第 6 节
 | 

## 1. 项目定位
用 Qwen3-1.7B + LoRA SFT + GRPO 训练一个本地小模型。Agent 每调用一次工具，它就从工具返回的内容里挑出对任务有用的片段，在 token 预算内交给大模型继续推理。目标结论：在同样的成本下答得更准，成本按含 prompt caching 的价格计算。
输入：任务问题、已保留的片段、本步新返回的片段（都带稳定编号和 token 数）、预算、模式。输出固定 JSON：

```
{"add": [47, 52], "drop": [12]}
```
| 字段
 | 含义
 | 约束
 | 
| add
 | 本步新返回的片段里，保留哪些
 | 只能是新返回片段的编号
 | 
| drop
 | 已保留的片段里，丢掉哪些
 | 只能是已保留片段的编号；“只追加”模式下必须为空
 | 
执行后，保留内容的总 token 不能超过预算。
同一个模型支持两种模式，由输入里的模式字段指定：
| 模式
 | 每步能做什么
 | 特点
 | 
| 重选（reselect）
 | 加新片段，也可以丢旧片段
 | 能反悔，占用的上下文最少；丢旧片段会让它之后的缓存失效
 | 
| 只追加（append）
 | 只加新片段，已保留的不动
 | 前缀不变，缓存全部命中；加错了没法反悔
 | 
为什么用 add / drop，而不是 v1 的“每步重新列出全部保留编号”：轨迹变长后，保留的片段有二十多个，每步重列一遍，漏写一个编号就等于误删一段证据。默认保留、显式丢弃更稳，也更利于缓存。
为什么做抽取式（选编号）而不是生成式摘要：

- 不会引入幻觉：保留的都是原文，压缩器编不出不存在的事实。

- 奖励可以自动验证：多跳问答数据集标注了哪些段落是支撑证据，选没选中一目了然。

- 推理便宜：输出只有几个数字，小模型处理一步的延迟很低。
v1 的 `note` 字段（一句进展备忘）本版先去掉：它是生成的文字，没有现成的训练目标。第二版再加，用子问题的中间答案自动构造目标。

## 2. 相关工作与差异点
这个方向 2026 年非常热，已有不少用 RL 做 Agent 上下文管理的工作。本项目不追求方法上的首创，而是做一个可插拔、可溯源、训练便宜的小压缩器，并在面试中讲清楚与下列工作的区别。
| 工作
 | 做法
 | 与本项目的区别
 | 
| [RECOMP](https://arxiv.org/abs/2310.04408)（2023）
 | 抽取式 / 生成式压缩检索文档，无关时可返回空
 | 单轮 RAG；本项目是多步 Agent，且用 GRPO
 | 
| [LLMLingua-2](https://arxiv.org/abs/2403.12968)（2024）
 | 蒸馏大模型标注，token 级删除，与任务无关
 | 本项目是任务感知的片段级选择
 | 
| [Provence](https://arxiv.org/abs/2501.16214)（ICLR 2025）
 | DeBERTa 序列标注做句子级剪枝，与重排合一
 | 监督学习、单轮；本项目显式控制预算并用 RL
 | 
| [Acon](https://arxiv.org/abs/2510.00615)（ICML）
 | 优化对观测和历史的压缩，使小模型也能做长程 Agent
 | 生成式压缩；本项目抽取式、可溯源
 | 
| [ContextBudget / BACM-RL](https://arxiv.org/abs/2604.01664)（2026）
 | 把多道独立问题拼成复合任务；主 Agent 自己做压缩（选片段再合并成摘要）；GRPO 加预算逐步收紧的课程；奖励是最终答案 F1，超预算则清零
 | 任务拼接和预算课程可借鉴；本项目的压缩器是独立小模型，抽取式，用证据召回做密集奖励
 | 
| [SUPO](https://aclanthology.org/2026.acl-long.966/)（ACL 2026）
 | 端到端 RL 同时优化工具使用和摘要策略
 | 要训练主 Agent 本身；本项目可配任意闭源 API
 | 
| [MemexRL](https://arxiv.org/abs/2603.04257)（2026）
 | RL 学习在预算下写入、归档、索引和读取记忆
 | 体量大、任务重；本项目是其中“选什么留下”的轻量版
 | 
本项目的四个差异点（面试主要讲这些）：

- 解耦：压缩器是独立的 1.7B 小模型，主 Agent 可以是任意闭源 API，不需要训练大模型。

- 可溯源：只选编号、不改写原文，每条保留内容都能追溯到工具的哪一次返回。

- 密集且便宜的奖励：用数据集自带的证据标注算召回，训练循环里不调用大模型，而不是只靠“最终是否答对”这种稀疏奖励。

- 按含缓存的成本评测：比较“重选”和“只追加”两种模式在成本–F1 上的取舍。ContextBudget 的论文没有提到 prompt caching（我通过工具读的网页版，精读时再确认）；其他工作是否做过，也要确认。
ContextBudget 我通过工具读了论文网页版的要点，其余工作只读了摘要和介绍页。开工前至少精读 Provence、Acon 和 ContextBudget 三篇，确认差异点成立。

## 3. 数据与任务构造
用多跳问答模拟 Agent 的工具调用，再把 4 道题拼成一个多目标任务，让一条轨迹有十几步、几万 token。数据集标注的支撑证据就是奖励信号。
| 数据
 | 用途
 | 说明
 | 
| [MuSiQue-Ans](https://github.com/stonybrooknlp/musique) train
 | 训练轨迹
 | 2–4 跳，有逐跳子问题和段落级证据标注；CC BY 4.0
 | 
| MuSiQue-Ans dev
 | 模拟测试
 | 单独建语料库，不与 train 混用
 | 
| MuSiQue dev、HotpotQA、2WikiMultiHopQA
 | 真实 Agent 测试（第 8 节）
 | 后两者没有 MuSiQue 这样的逐跳子问题，不能照搬模拟流程；许可待确认
 | 
轨迹模拟：

- 建语料库：把一个 split 里所有题目的段落去重合并，建 BM25 索引。train 和 dev 各建一个。

- 一跳 = 一次工具调用：用这一跳的子问题检索，取 top-20 作为工具返回。据我所知，后几跳的子问题用“#1”指代前一跳的答案，部分子问题是“实体 >> 关系”的形式；检索前先把占位符换成答案。仓库 README 没有写明格式，开工时先打印几条确认。

- 保证证据出现：这一跳的证据段没被检索到时，用它替换排名最后的一段，并统计自然命中率。

- 拼成多目标任务：每个任务随机取 4 道题，按题依次执行，步数等于各题跳数之和（约 8–16 步）。任务描述里列出全部 4 个问题。

- 每步的金标准：已经执行过的各跳的证据段中，当前还拿得到的那些（在已保留或新返回里）。此前丢掉的证据不再计入。

- 预算：保留内容的总 token 上限，每条轨迹固定一档。三档数值由第 6 节的试验统计来定。
多题拼接是已有做法：ContextBudget 沿用 MEM1，把多道独立问题拼成一个复合任务，训练用 2 题，评测用 2 到 32 题。
长度估算（待试验实测）：每段按 100–200 token，一步 20 段约 2k–4k；4 道题平均 3 跳，整条轨迹约 2.4 万–4.8 万 token。压缩器单步输入 = 已保留 + 新返回，约 4k–8k。
v1 的做法是每步只从题目自带的 20 段里取 5 段，再混入其他题的随机段落。随机段落主题无关、很好认，整条轨迹也只有几千 token。
划分：训练轨迹来自 train，其中按题留出 5% 做开发集，用来选 checkpoint。dev 的题每题只用一次，拼成测试任务。train 与 dev 的段落是否有重叠，用脚本核对后写进 README。

## 4. 奖励设计
主奖励是证据召回；证据全部召回时，再奖励省下的预算。训练循环里不调用大模型，两种模式共用一个公式。
记执行动作后保留的片段总 token 为 used，预算为 B，recall = 保留下来的金标准数 ÷ 当前拿得到的金标准数：

```
R_t = \begin{cases} -1 & \text{invalid action, or } used > B \\ recall_t + \lambda \cdot \mathbb{1}[recall_t = 1] \cdot \left(1 - \frac{used}{B}\right) & \text{otherwise} \end{cases}
```

- 非法动作包括：JSON 无法解析、编号不在候选里、“只追加”模式下 drop 不为空。

- λ 取 0.2–0.5，由消融确定。

- 只有全部召回才奖励省预算：防止模型靠少选来刷分。漏选罚得重，多留罚得轻。

- 超预算直接记 -1：预算是硬约束，对应真实系统里的上下文上限。推理时遇到非法或超预算的输出，按“本步什么都不改”处理。
SFT 目标：预算内保留尽量多的证据段；放不下时按 token 从小到大贪心，这样保留的段数最多。生成重选模式的 SFT 数据时，在每步的理想动作上随机多加 1–3 个非证据段，让下一步的状态里有该丢的东西，模型在这一阶段就能学到 drop。
GRPO 的训练状态：先用 SFT 模型把训练轨迹闭环跑一遍，用它自己每一步的保留结果作为输入状态。这样训练时见到的状态和实际使用时一致。
GRPO 相对 SFT 应该赢在哪里（事先写明，用数据验证）：

- 拿不准时多留。SFT 的目标里只有证据段，学不到在不确定时留有余量；不对称奖励能学到。看指标：紧预算下的证据丢失率。

- 出错后的状态。SFT 见到的状态来自理想动作；闭环使用时，模型会遇到自己漏选、多选之后的状态。看指标：闭环评测下 SFT 与 GRPO 的差距。
参考 KnowMe 的结果：GRPO 相对 SFT，门控 F1 只高约 0.4 个点，检索命中率高约 5.6 个点。增益可能不大，如实报告。
消融：

- λ = 0、0.2、0.5：准确率与占用上下文的取舍。

- 重选 vs 只追加：在成本–F1 平面上比较（第 7 节）。

- 理想状态训练 vs 闭环状态训练。

- （可选）固定预算 vs 随机预算训练。

## 5. 实现：轨迹构造、奖励与成本模型
产出：训练和测试轨迹（每条是一步的输入状态、金标准和 SFT 目标）、奖励函数、成本模型，以及它们的单元测试。下面三段代码已在一份按 MuSiQue 字段格式造的合成数据上跑通，检查了主要逻辑和奖励的每个分支。它们还没接触真实数据，字段名以实际文件为准。
目录结构：

```
ctx-compressor/
├── data/raw/            # MuSiQue / HotpotQA / 2Wiki 原始数据
├── data/traj/           # 模拟的多步轨迹
├── src/build_traj.py    # 语料库、检索、拼接、闭环展开
├── src/reward.py        # GRPO 奖励
├── eval/cost.py         # 含缓存的成本模型
├── eval/                # 闭环评测与结果
└── tests/               # 单元测试
```
轨迹构造。`ntok` 是数 token 的函数，用 Qwen3 的分词器实现：

```
# src/build_traj.py —— 语料库、逐跳检索、多题拼接、闭环展开
import random, re
import bm25s

def build_corpus(examples, ntok):
    """一个 split 内的全部段落去重合并，并建 BM25 索引。"""
    key2pid, corpus = {}, []
    for ex in examples:
        for p in ex["paragraphs"]:
            key = (p["title"], p["paragraph_text"])
            if key not in key2pid:
                key2pid[key] = len(corpus)
                corpus.append({"title": p["title"], "text": p["paragraph_text"],
                               "ntok": ntok(p["paragraph_text"])})
    index = bm25s.BM25()
    index.index(bm25s.tokenize([c["title"] + " " + c["text"] for c in corpus],
                               show_progress=False), show_progress=False)
    return corpus, index, key2pid

def hop_queries(ex, key2pid):
    """逐跳返回 (检索查询, 证据段 pid)。子问题里的 #k 换成第 k 跳的答案。"""
    by_idx = {p["idx"]: p for p in ex["paragraphs"]}
    answers, out = [], []
    for hop in ex["question_decomposition"]:
        def fill(m):
            k = int(m.group(1))
            return answers[k - 1] if k <= len(answers) else m.group(0)
        q = re.sub(r"#(\d+)", fill, hop["question"])
        answers.append(hop["answer"])
        g = by_idx[hop["paragraph_support_idx"]]
        out.append((q, key2pid[(g["title"], g["paragraph_text"])]))
    return out

def simulate_task(exs, index, key2pid, top_n=20):
    """K 道题依次执行，一跳 = 一次检索。返回每步新返回的 pid 和本跳的证据 pid。"""
    steps = []
    for ex in exs:
        for q, gold in hop_queries(ex, key2pid):
            ids, _ = index.retrieve(bm25s.tokenize([q], show_progress=False),
                                    k=top_n, show_progress=False)
            new = [int(i) for i in ids[0]]
            hit = gold in new
            if not hit:
                new[-1] = gold          # 保证本跳证据出现；自然命中率单独统计
            random.shuffle(new)
            steps.append({"new": new, "gold": gold, "natural_hit": hit})
    return steps

def make_sample(questions, kept, new, gold_upto, corpus, budget, mode, lid):
    """一步的输入状态。kept/new 是 pid 列表；lid 把 pid 映射成轨迹内稳定的小编号。"""
    new = [p for p in new if p not in kept]
    for p in new:
        lid.setdefault(p, len(lid))
    cand = kept + new
    return {"questions": questions, "mode": mode, "budget": budget,
            "kept_ids": [lid[p] for p in kept], "new_ids": [lid[p] for p in new],
            "ntok": {lid[p]: corpus[p]["ntok"] for p in cand},
            "text": {lid[p]: corpus[p]["text"] for p in cand},
            "gold": [lid[p] for p in cand if p in gold_upto]}

def oracle_action(s):
    """SFT 目标：预算内保留尽量多的证据段（按 token 从小到大贪心）。"""
    kept, gold, nt = set(s["kept_ids"]), set(s["gold"]), s["ntok"]
    if s["mode"] == "append":            # 已保留的不能动，只决定加哪些
        used, add = sum(nt[i] for i in kept), []
        for i in sorted(gold - kept, key=lambda i: nt[i]):
            if used + nt[i] <= s["budget"]:
                add.append(i)
                used += nt[i]
        return {"add": sorted(add), "drop": []}
    keep, used = set(), 0
    for i in sorted(gold, key=lambda i: (nt[i], i not in kept)):
        if used + nt[i] <= s["budget"]:
            keep.add(i)
            used += nt[i]
    return {"add": sorted(keep - kept), "drop": sorted(kept - keep)}

def apply_action(s, act):
    """执行动作，返回保留的编号（旧的保持原顺序，新加的追加在末尾）；非法动作返回 None。"""
    add, drop = set(act.get("add", [])), set(act.get("drop", []))
    if not add <= set(s["new_ids"]) or not drop <= set(s["kept_ids"]):
        return None
    if s["mode"] == "append" and drop:
        return None
    keep = [i for i in s["kept_ids"] if i not in drop] + [i for i in s["new_ids"] if i in add]
    if sum(s["ntok"][i] for i in keep) > s["budget"]:
        return None
    return keep

def rollout(questions, steps, corpus, budget, mode, policy=oracle_action):
    """闭环展开一条轨迹。policy 换成模型推理函数，就得到 GRPO 的训练状态和闭环评测。"""
    kept, gold_upto, lid, samples = [], set(), {}, []
    for st in steps:
        gold_upto.add(st["gold"])
        s = make_sample(questions, kept, st["new"], gold_upto, corpus, budget, mode, lid)
        s["sft_target"] = oracle_action(s)
        keep = apply_action(s, policy(s))
        if keep is None:                 # 兜底：非法动作视为"本步什么都不改"
            keep = s["kept_ids"]
        s["keep_after"] = keep
        samples.append(s)
        pid_of = {v: k for k, v in lid.items()}
        kept = [pid_of[i] for i in keep]
    return samples
```
给模型的输入按固定模板渲染：先列任务问题、模式、预算和已用 token，再列“已保留”和“新返回”两组片段，每个片段前标出编号和 token 数。小模型数不准 token，不标就很难遵守预算。SFT 的目标串就是 `sft_target` 的 JSON。
奖励函数。接入 ms-swift 时再包一层：继承 `ORM`、注册到 `orms`，用 `--external_plugins` 和 `--reward_funcs` 加载；KnowMe 的 `scripts/grpo_gate_reward.py` 是现成的例子。

```
# src/reward.py —— GRPO 奖励，两种模式共用
import json
from build_traj import apply_action

def step_reward(completion, s, lam=0.3):
    try:
        act = json.loads(completion)
        ids = act["add"] + act.get("drop", [])
        assert all(type(i) is int for i in ids)
    except Exception:
        return -1.0                      # 无法解析或格式不对
    keep = apply_action(s, act)
    if keep is None:
        return -1.0                      # 编号越界、只追加模式里丢了旧片段、或超预算
    gold = set(s["gold"])
    recall = len(gold & set(keep)) / len(gold) if gold else 1.0
    used = sum(s["ntok"][i] for i in keep)
    bonus = lam * (1 - used / s["budget"]) if recall == 1.0 else 0.0
    return recall + bonus
```
成本模型：

```
# eval/cost.py —— 大模型输入成本（以"基础输入价 × token"为单位）
def trajectory_cost(contexts, ntok, r_read=0.05, r_write=1.25):
    """contexts: 每一步交给大模型的片段编号序列。
    返回 (不开缓存的成本, 开缓存的成本)。开缓存时，与上一步的最长公共前缀按读取价计，其余按写入价计。"""
    plain = cached = 0.0
    prev = []
    for cur in contexts:
        k = 0
        while k < min(len(prev), len(cur)) and prev[k] == cur[k]:
            k += 1
        hit = sum(ntok[i] for i in cur[:k])
        miss = sum(ntok[i] for i in cur[k:])
        plain += hit + miss
        cached += r_read * hit + r_write * miss
        prev = cur
    return plain, cached
```
每种方法取“开缓存”和“不开缓存”两个成本里更低的一个。缓存倍率是参数：以 Claude Opus 5.5 为例，写入是基础输入价的 1.25 倍，读取是 0.05 倍；其他模型和厂商各不相同。
为什么要这样算，看一个例子：10 步，每步工具返回 3k token，成本单位是“按基础输入价折算的 token”。
| 做法
 | 不开缓存
 | 开缓存
 | 
| 不压缩
 | 16.5 万
 | 4.4 万
 | 
| 重选，每步只留 2k，且每步留的内容都不同
 | 2 万
 | 2.5 万
 | 
| 只追加，每步留 300 token
 | 1.65 万
 | 0.44 万
 | 
开缓存后，读历史很便宜，贵的是每步新写入的内容。所以压缩器省钱主要靠过滤新返回的内容；频繁丢旧内容省不了多少，还会让缓存失效。
本阶段验收：

- 单元测试全部通过：奖励的每个分支、SFT 目标一定合法、保留内容不超预算、“只追加”模式下前缀不变。

- 换成真实的 MuSiQue 数据后，打印 3 条轨迹人工检查：占位符是否替换、证据段是否正确。

## 6. 可行性试验：训练前先过这一关（2–3 天）
不训练任何模型，只回答一个问题：在这个任务上，学习型压缩器有没有赢的空间。
用 dev 的题拼 150 个任务，做三件事：

- 统计：整条轨迹的 token、压缩器单步输入的 token、每个任务证据 token 的分布；检索的自然命中率；非证据段里含有该跳答案字符串的比例。最后一项是标注遗漏的上界，比例高说明奖励有噪声。

- 定预算：紧档取证据 token 的 P90，即九成任务能放下全部证据；中档是它的 1.5 倍；松档是 3 倍。

- 闭环跑四个不用训练的策略：不压缩、保留最新、BM25 装满预算（用任务问题给候选片段打分，从高到低装）、oracle。记录证据召回、证据丢失率和阅读模型 F1。
| 观察到
 | 说明
 | 怎么办
 | 
| 中档预算下，BM25 的证据召回比 oracle 低 15 个点以上，F1 低 5 个点以上
 | 有学习空间
 | 开工
 | 
| BM25 已接近 oracle
 | 任务还是太容易
 | 收紧预算、top-20 提到 top-30、每个任务多拼几题，再测一次
 | 
| 不压缩的 F1 与 oracle 接近
 | 这个长度下，无关内容不影响阅读模型
 | 项目只讲成本，不讲“提升准确率”
 | 
| 不压缩的 F1 明显低于 oracle
 | 无关内容确实伤害效果
 | 效果和成本都可以讲
 | 
15 和 5 是建议的经验线，不是硬标准。调整两轮后仍拉不开差距，就该换题，而不是硬做。试验的数字直接写进 README。

## 7. 训练、基线与评测
核心结论用两张图呈现：预算–F1 曲线和成本–F1 曲线。所有评测都是闭环的：压缩器上一步的输出就是下一步的输入。
训练配置（ms-swift 或 TRL）：

- SFT：Qwen3-1.7B + LoRA（r=16），两种模式的样本各占一半，2–3 个 epoch，在开发集上按证据丢失率选 checkpoint。

- GRPO：从 SFT checkpoint 继续，每个 prompt 采样 8 条，奖励见第 4 节。单步输入约 4k–8k token，先在 A10 上测单步显存，不够再租 A100。

- 可借鉴 ContextBudget 的做法：预算从宽到紧的课程学习，在 README 里注明出处。
对比的方法（同一批测试任务、同一预算、同一阅读模型）：
| 方法
 | 说明
 | 
| 不压缩
 | 全部内容交给阅读模型；参考线
 | 
| oracle
 | 只保留证据段；上限
 | 
| 保留最新
 | 按时间倒序装满预算
 | 
| BM25 / 向量重排
 | 用任务问题给片段打分，装满预算；不训练
 | 
| 同数据微调的重排模型
 | 用同一份训练轨迹微调一个 cross-encoder，按分数装满预算；最可能打平本项目的基线
 | 
| LLMLingua-2 / Provence
 | 现成压缩器，没在 MuSiQue 上训练过；和它们的公平比较看第 8 节
 | 
| API 模型
 | 用同样的提示词让 API 大模型输出 add / drop
 | 
| 本项目
 | Qwen3-1.7B 原始、+ SFT、+ SFT + GRPO，各跑重选和只追加两种模式
 | 
指标：

- 压缩质量：证据召回；证据丢失率（任务结束时至少丢了一段证据的任务比例，并按丢在第几跳拆分）；非法输出和超预算的比例。

- 端到端：阅读模型分别回答任务里的 4 个问题，报告平均 EM / F1。本地用 Qwen3-4B 或 8B，再用一个 API 模型在子集上复核。

- 成本：大模型的输入成本，按第 5 节的成本模型算，报告不开缓存和开缓存两种。

- 效率：压缩器的单步延迟（vLLM）。
另外报告：三档预算各自的结果、按跳数拆分的结果、第 4 节的消融。跨数据集泛化在第 8 节做。
预期与检验：本项目应该在靠后的跳上赢过微调的重排模型。后几跳的证据和问题字面上不相关，要结合已保留的内容才能判断，而重排模型一次只看一个片段。按跳数拆分结果来检验；如果两者打平，如实报告。
无标注适应实验（核心实验和第 8 节做完后再做）：证据召回奖励依赖标注，真实任务没有。实验设计如下。

- 用第 8 节的真实 Agent 在 2Wiki 上跑一遍，记录每步的工具返回，之后固定重放。

- 从证据奖励训好的模型出发，只用“阅读模型最终答对”作奖励继续 GRPO，全程不看证据标注。

- 比较适应前后在 2Wiki 上的 F1。
时间不够时退回 v1 的做法：在 20% 的样本上对比两种奖励，并在面试中如实说明这一局限。

## 8. 真实 Agent 测试与可逆压缩
模拟轨迹用的是标准子问题，真实 Agent 要自己写查询，会走弯路。这一节决定项目能不能叫“Agent 上下文压缩”，至少跑 200 个任务。

- Agent：ReAct 循环，只有一个 `search(query)` 工具（BM25，返回 top-20 段）。主模型用一个 API 模型，或本地的 Qwen3-8B。

- 任务：MuSiQue dev、HotpotQA、2Wiki 各取一部分，每个任务拼 2–3 道题。语料库是所取题目的段落合集。

- 接入：工具返回先交给压缩器，主模型只看到保留下来的片段。对比不压缩、保留最新、微调重排、现成压缩器和本项目的两种模式。

- 指标：Agent 最终答案的 EM / F1、步数、含缓存的输入成本、证据丢失率。

- 跨数据集泛化也在这里测：HotpotQA 和 2Wiki 对所有方法都是没训练过的数据，和现成压缩器的比较以这里为准。
先用 20 个任务估算 API 费用，再放量。费用主要花在“不压缩”这一行。
可逆压缩（加分项）：被丢掉的片段不删除，存进归档。给 Agent 加一个 `search_archive(query)` 工具，可以在归档里检索并取回片段。对比开、关这个工具时的 F1、取回次数和找回的证据段数。Agent 发现信息不够时能自己找回，压缩器漏选的代价就小了。

## 9. 时间、算力与风险
业余推进约 8 周，比 v1 多 2 周。第一个阶段只要 2–3 天，过不了关就不往下投入。
| 阶段
 | 时间
 | 任务
 | 算力
 | 
| 0
 | 2–3 天
 | 轨迹构造、奖励函数、可行性试验，决定是否继续
 | CPU + A10
 | 
| 1
 | 第 1–2 周
 | 全量轨迹、全部基线（含微调重排）、闭环评测和成本模型
 | A10
 | 
| 2
 | 第 3 周
 | LoRA SFT
 | A10
 | 
| 3
 | 第 4–5 周
 | GRPO 和消融
 | 先试 A10，不够再租 A100 约 2–3 天
 | 
| 4
 | 第 6–7 周
 | 真实 Agent 测试，可选归档工具
 | A10 或 API
 | 
| 5
 | 第 8 周
 | 无标注适应（可选）、README、demo
 | A10
 | 
只有 5–6 周时：做到阶段 3，再跑 100 个真实 Agent 任务。归档工具和无标注适应不做，简历里对应的句子也不写。
风险与应对：

- 试验没过关：按第 6 节调整；两轮后仍不行就换题。

- 微调的重排模型打平本项目：如实报告。这时项目的价值是完整的后训练流程和两种模式的成本对比，而不是“RL 更强”。

- GRPO 相对 SFT 的增益很小：第 4 节已写明看哪两个指标；增益小就如实写，并分析原因。

- 标注遗漏带来奖励噪声：试验里先量化。比例高时，把“含答案字符串的非证据段”设为中性，不奖不罚。

- 显存不够：把 top-20 降到 top-15，或把每个片段截到前 200 token。

- 模拟轨迹与真实 Agent 有差距：第 8 节的真实测试就是为了回答这个问题。

- 缓存价格因模型和厂商而异：倍率做成参数，同时报告开缓存和不开缓存两种成本。

- 与已有工作重叠：第 2 节的差异点要在 README 开头说清，并引用相关工作。

## 10. 简历写法
X 在实验完成后填入真实数字；方括号里是写这一条的前提。
面向长程 Agent 的预算感知上下文压缩模型（个人项目，开源）

- 基于 Qwen3-1.7B，用 LoRA SFT + GRPO 训练可插拔的抽取式上下文压缩器，在 Agent 每步工具调用后按 token 预算增删保留片段，可配合任意闭源大模型使用。

- 基于 MuSiQue 多跳问答构造 X 条多目标长轨迹（单条约 X 万 token）；设计不对称奖励：以证据召回为主，仅在全召回时奖励预算节省，训练中不调用大模型。

- 闭环评测：同等预算下答题 F1 比同数据微调的重排模型高 X 个点，证据丢失率低 X 个点；按含 prompt caching 的价格计，同等 F1 下大模型输入成本降低 X%。

- 接入真实 ReAct 搜索 Agent，在 HotpotQA、2Wiki 上验证跨数据集泛化，F1 比 LLMLingua-2 高 X 个点。［前提：做完第 8 节］

- 丢弃的片段归档并支持检索取回，F1 再提高 X 个点。［前提：做了归档工具］
如果没做真实 Agent 测试，项目名改成“预算感知的多跳证据选择模型”，和内容相符。结果如果与上面的句子方向相反（比如没有赢过重排模型），就按实际结果改写这一条。
面试前自查：为什么选抽取式；为什么只在全召回时奖励省预算；SFT 已有标签，为什么还要 GRPO；为什么不直接用重排模型；开了缓存还能省多少；真实任务没有证据标注怎么办；和 Provence、ContextBudget 的区别。

## Sources

- [Provence（ICLR 2025）](https://arxiv.org/abs/2501.16214)

- [RECOMP](https://arxiv.org/abs/2310.04408) · [LLMLingua-2](https://arxiv.org/abs/2403.12968)

- [Acon](https://arxiv.org/abs/2510.00615) · [ContextBudget](https://arxiv.org/abs/2604.01664)（[网页版](https://arxiv.org/html/2604.01664v1)） · [MemexRL](https://arxiv.org/abs/2603.04257) · [SUPO（ACL 2026）](https://aclanthology.org/2026.acl-long.966/)

- [MuSiQue（TACL 2022）](https://arxiv.org/abs/2108.00573) · [数据仓库](https://github.com/stonybrooknlp/musique)

- [KnowMe Memory Gate 仓库](https://github.com/Longlong418/KnowMe-memory_gate_model_post_train)（GRPO 相对 SFT 的增益、ms-swift 奖励插件的写法）

- [Claude API 定价页](https://platform.claude.com/docs/en/about-claude/pricing)（缓存写入和读取的倍率）
