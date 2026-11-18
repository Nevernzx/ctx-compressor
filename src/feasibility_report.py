"""Chinese task-02 report renderer; pure function, no file or network access."""
import json


def _number(value):
    if value is None:
        return "未提供"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _percent(value):
    return "未提供" if value is None else f"{value * 100:.3f}%"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(headers, rows):
    lines = ["| " + " | ".join(map(_cell, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(_cell, row)) + " |" for row in rows)
    return "\n".join(lines)


def _distribution_row(label, distribution):
    return [label, distribution["n"]] + [
        _number(distribution[key]) for key in ("mean", "p10", "p50", "p90", "p99", "max")
    ]


def _rate(rate):
    return f"{rate['hits']:,} / {rate['n']:,}；{_percent(rate['rate'])}"


def _loss_group(group):
    percentage = "未适用" if group["n"] == 0 else _percent(group["rate"])
    return (f"{group['count']:,}/{group['n']:,}；{percentage}"
            f"；证据 {group['lost_evidence_count']:,}/{group['evidence_n']:,}")


def _ordered_configs(configs):
    policy_order = {"no_compress": 0, "latest": 1, "bm25": 2, "oracle": 3}
    mode_order = {"reselect": 0, "append": 1}
    tier_order = {"tight": 0, "mid": 1, "loose": 2}
    return sorted(configs.items(), key=lambda item: (
        policy_order[item[1]["policy"]], mode_order.get(item[1]["mode"], -1),
        tier_order.get(item[1]["budget_tier"], -1), item[0],
    ))


def _lookup(configs, policy, mode, tier):
    matches = [config for config in configs.values()
               if config["policy"] == policy and config["mode"] == mode
               and config["budget_tier"] == tier]
    if len(matches) != 1:
        raise ValueError(f"expected one configuration: {policy}/{mode}/{tier}, got {len(matches)}")
    return matches[0]


def _difference(configs, mode, tier):
    oracle = _lookup(configs, "oracle", mode, tier)["metrics"]
    bm25 = _lookup(configs, "bm25", mode, tier)["metrics"]
    return {
        "final": 100 * (oracle["final_recall"]["mean"] - bm25["final_recall"]["mean"]),
        "step": 100 * (oracle["step_recall"]["micro"]["mean"] - bm25["step_recall"]["micro"]["mean"]),
        "task_step": 100 * (oracle["step_recall"]["task_macro"]["mean"] - bm25["step_recall"]["task_macro"]["mean"]),
        "n_tasks": oracle["n_tasks"], "n_steps": oracle["n_steps"],
    }


def _diff_sentence(mode, tier, delta):
    return (f"{mode}/{tier}：任务末 {delta['final']:.3f} pp"
            f"（≥15：{'是' if delta['final'] >= 15 else '否'}）；"
            f"单步按步 {delta['step']:.3f} pp"
            f"（≥15：{'是' if delta['step'] >= 15 else '否'}）；"
            f"单步按任务 {delta['task_step']:.3f} pp；"
            f"n={delta['n_tasks']} 任务/{delta['n_steps']} 步")


def render_report(results):
    """Render final metrics to Markdown using the schema in notes_task02_report.md.

    Rates are fractions in the input and percentages in the report.  Criteria
    differences are computed from the same displayed aggregates, never inferred
    from recall as reading-model performance. Missing required fields raise.
    """
    settings, budgets = results["settings"], results["budgets"]
    configs = results["configs"]
    ordered = _ordered_configs(configs)
    baseline = [item for item in ordered if item[1]["policy"] == "no_compress"]
    if len(ordered) != 19 or len(baseline) != 1:
        raise ValueError("report requires 19 configurations, including one no_compress")
    for policy in ("latest", "bm25", "oracle"):
        for mode in ("reselect", "append"):
            for tier in ("tight", "mid", "loose"):
                _lookup(configs, policy, mode, tier)
    baseline_name, baseline_config = baseline[0]
    n_tasks = settings["n_tasks"]
    distributions = results["distributions"]
    text = [
        "# §6 可行性试验：无阅读模型部分", "",
        f"固定 seed={settings['seed']}；{n_tasks} 个任务，每任务 4 题，共 {4 * n_tasks} 道互不重复的 dev 题；top-{settings['top_n']}。"
        "单次运行，报告点估计及 n，不计算置信区间。",
        "全程纯 CPU、离线；未下载或训练模型，未调用 API。阅读模型 F1 / EM：**未测（需阅读模型）**。"
        "以下指标仅描述证据选择及输入成本，证据完整题比例是代理指标，**不是 F1**。",
        "语料库与 BM25 均仅使用 MuSiQue-Ans dev；片段为 `title + \"\\n\" + paragraph_text`，"
        "Qwen3 分词器使用 `add_special_tokens=False`。任务索引从 0 开始，检索种子等于任务索引；19 个配置共用每个任务的一份检索步骤。", "",
        "## 数据过滤与三档预算", "",
    ]
    filter_rows = []
    for split in ("dev", "train"):
        entry = results[f"{split}_filter"]
        filter_rows.append([split, entry["total"], entry["valid"], entry["rejected"],
                            json.dumps(entry["reasons"], ensure_ascii=False, sort_keys=True)])
    text.extend([_table(["split", "题数 n", "通过", "拒绝", "拒绝原因计数"], filter_rows), "",
                 "dev 全部题先通过 `hop_queries` 校验，再从通过的题中无放回抽样。train 拒绝题保持原样；"
                 "train 只抽取参照任务并计算金标准段 token，不建立 train 语料库或 BM25。", "",
                 _table(["项目", "token", "样本数 / 定义"], [
                     ["dev 证据 token P90", _number(distributions["task_evidence_tokens"]["p90"]), f"n={distributions['task_evidence_tokens']['n']} 任务；linear 分位数"],
                     ["紧档 tight", budgets["tight"], "ceil(dev 每任务去重证据 token P90)"],
                     ["中档 mid", budgets["mid"], "ceil(1.5 × 紧档)"],
                     ["松档 loose", budgets["loose"], "3 × 紧档"],
                     ["train 参照 P90", _number(budgets["train_reference_p90"]), f"n={settings.get('train_reference_n_tasks', 150)} 任务；仅参照，不参与定档"],
                 ]), "",
                 "预算按上述公式一次确定；本任务不按结果修改预算公式、top-20 或每任务题数。任务 01 的 3000 token 占位验收保持原样。", "",
                 "## 轨迹长度与证据长度", ""])
    labels = {
        "task_raw_tokens": "每任务原始轨迹 token（跨步重复计入）",
        "task_unique_tokens": "每任务去重轨迹 token",
        "task_steps": "每任务步数",
        "question_hops": "每题跳数",
        "task_evidence_tokens": "每任务去重金标准段 token",
        "question_evidence_tokens": "每题去重金标准段 token",
    }
    text.extend([_table(["分布", "n", "mean", "P10", "P50", "P90", "P99", "max"],
                        [_distribution_row(label, distributions[key]) for key, label in labels.items()]), "",
                 "原始轨迹 token 是每一步全部返回片段 token 的累加，重复返回仍计入；去重口径是整任务出现过的不同语料 pid。"
                 "证据 token 对相应任务或题的全部金标准 pid 去重后求和。所有分位数使用 NumPy 默认 linear 插值。", "",
                 "## 检索自然命中与答案泄漏", ""])
    retrieval = results["retrieval"]
    retrieval_rows = [["总体", _rate(retrieval["overall"])]]
    for group, label in (("by_hop", "题内跳位"), ("by_question_hops", "题目总跳数")):
        retrieval_rows.extend([f"{label} {key}", _rate(value)] for key, value in sorted(retrieval[group].items(), key=lambda pair: int(pair[0])))
    text.extend([_table(["分组", "自然命中跳数 / 跳数 n；比例"], retrieval_rows), "",
                 "自然命中统计发生在强制补入证据之前；候选用于策略评估时已保证本跳证据出现。", ""])
    leaks = results["answer_leak"]
    leak_rows = []
    for key, label in (("all", "正文主口径：全部答案"), ("multiword", "正文主口径：答案归一化后 ≥2 词"),
                       ("rendered_all", "标题+正文敏感性：全部答案"),
                       ("rendered_multiword", "标题+正文敏感性：答案归一化后 ≥2 词")):
        if key in leaks:
            leak_rows.append([label, _rate(leaks[key]["paragraph"]), _rate(leaks[key]["hop"])])
    text.extend([_table(["口径", "含答案非证据段 / 非证据段 n；比例", "至少一次命中跳 / 跳数 n；比例"], leak_rows), "",
                 "每跳从返回片段中排除整任务全部金标准 pid（所有题、所有跳），剩余片段才进入非证据分母。"
                 "匹配先做 SQuAD 风格归一化：小写、去标点、去冠词 a/an/the、合并空白，再按词边界匹配本跳答案。"
                 "多词口径只纳入归一化答案至少两个词的跳，其段级、跳级分母均独立统计。", ""])
    duplicates = leaks["body_duplicates"]
    text.extend([f"非证据段归一化正文与本跳金标准正文完全相同：**{duplicates['count']:,} / {duplicates['n']:,}** 次候选出现，"
                 f"比例 {_percent(duplicates['rate'])}；其中**标题不同 {duplicates['different_title_count']:,} 次**。"
                 "总数同时包含不同标题的正文重复，以及同标题但原正文仅有标点、冠词、空白等差异而归一化后相同的情况。"
                 "该计数针对段落出现次数，跨跳可重复计入。",
                 "答案子串命中是标注遗漏的上界，也可能来自巧合；"
                 "[20 条固定种子命中样例](answer_leak_samples.md)供人工核对。", "",
                 "## 闭环证据选择指标", "",
                 "单步按步平均对全部状态等权；单步按任务平均先在每任务内求均值，再对任务等权。"
                 "两者的单步召回分母都是 PLAN 奖励口径下当前仍可获得的历史证据。"
                 "任务末召回则对整任务全部唯一金标准 pid 计算，再对任务等权，是本报告判据表的主要证据召回口径。", ""])
    recall_rows = []
    for name, config in ordered:
        metrics = config["metrics"]
        recall = metrics["step_recall"]
        lost = metrics["evidence_loss"]
        complete = metrics["evidence_complete_questions"]
        recall_rows.append([
            name, config["budget"], metrics["n_tasks"], metrics["n_steps"],
            _percent(recall["micro"]["mean"]), _percent(recall["task_macro"]["mean"]),
            _percent(metrics["final_recall"]["mean"]),
            f"{lost['count']}/{lost['n']}；{_percent(lost['rate'])}",
            f"{complete['count']}/{complete['n']}；{_percent(complete['rate'])}",
        ])
    text.extend([_table(["配置", "预算 B", "n 任务", "n 步", "单步按步", "单步按任务", "任务末召回", "证据丢失任务", "证据完整题（非 F1）"], recall_rows), "",
                 "证据丢失任务指最终保留集合至少缺一个整任务金标准 pid；证据完整题要求该题全部金标准 pid 都在最终保留集合中。"
                 "no_compress 使用足够大的非约束预算，两种模式等价，只运行和报告一次。", "",
                 "## 证据丢失拆分", "",
                 "下表每个单元依次给出：有该类最终丢失的任务数 / 含该类证据的任务数 n；任务比例；丢失证据关联数 / 证据关联总数。"
                 "同一 pid 如支撑多跳，可在跳位、题位分组中按关联重复计入；任务率不能跨列相加。", ""])
    for key, title in (("by_hop", "题内跳位"), ("by_question_position", "任务内题目位置")):
        rows = [[name] + [_loss_group(config["metrics"]["evidence_loss"][key][str(i)]) for i in range(1, 5)]
                for name, config in ordered]
        text.extend([f"### 按{title}", "", _table(["配置"] + [f"第 {i} 位" for i in range(1, 5)], rows), ""])
    mechanism_rows = []
    for name, config in ordered:
        mechanisms = config["metrics"]["evidence_loss"]["by_mechanism"]
        mechanism_rows.append([name, _loss_group(mechanisms["never_kept"]), _loss_group(mechanisms["dropped"])])
    text.extend(["### 按丢失方式", "",
                 "两列只针对最终缺失证据：从未进入任一步保留集合，或曾保留过而最终缺失。"
                 "此处任务分母为全部任务，证据分母为各任务去重金标准 pid 的总数；同一任务可同时出现两种方式。", "",
                 _table(["配置", "从未被保留", "保留过后被 drop"], mechanism_rows), "",
                 "## 预算占用与动作合法性", ""])
    occupation_rows = []
    for name, config in ordered:
        metrics = config["metrics"]
        occupancy, actions = metrics["occupancy"], metrics["action_validity"]
        occupation_rows.append([name, occupancy["n"], _percent(occupancy["mean"]), _percent(occupancy["max"]),
                                actions["n"], actions["invalid_count"], actions["over_budget_count"]])
    text.extend([_table(["配置", "n 任务", "最终 used/B 均值", "最终 used/B 最大", "n 动作", "非法次数", "超预算次数"], occupation_rows), "",
                 "no_compress 的预算仅用于允许保留全部返回，其 used/B 不表示实际部署预算。所有策略动作都应合法且不超预算。", "",
                 "## 大模型输入成本估计", "",
                 f"单位为基础输入价折算 token；读取倍率 {settings.get('r_read', 0.05)}，写入倍率 {settings.get('r_write', 1.25)}。"
                 "逐步执行后的 keep_after 顺序交给 trajectory_cost，与上一步的最长共同前缀按读取价、其余按写入价计算。"
                 "这里仅估计输入成本，没有调用阅读模型，也不包括压缩器推理成本。", ""])
    costs = []
    for name, config in ordered:
        cost = config["metrics"]["cost"]
        costs.append([name, cost["plain"]["n"], _number(cost["plain"]["mean"]),
                      _percent(cost["plain"]["relative_to_no_compress"]),
                      _number(cost["cached"]["mean"]), _percent(cost["cached"]["relative_to_no_compress"])])
    text.extend([_table(["配置", "n 任务", "不开缓存均值", "相对 no_compress", "开缓存均值", "相对 no_compress"], costs), "",
                 "相对比例分别使用同一缓存条件下的 no_compress 成本作为分母，两种缓存条件并列保留。", "",
                 "## 单步输入长度", "",
                 "压缩器完整输入对 render_input(s) 整串重新分词，包含任务问题、模式、预算、分组标题与片段编号；"
                 "片段部分为 kept + new 的 ntok 之和。二者分开报告，避免把模板开销藏入预算。"
                 "no_compress 没有压缩器，单列每步执行后累计上下文 token。", ""])
    input_rows = [_distribution_row(f"{baseline_name}：累计上下文", baseline_config["input_tokens"])]
    for name, config in ordered:
        if config["policy"] != "no_compress":
            input_rows.append(_distribution_row(f"{name}：完整模板", config["input_tokens"]))
            input_rows.append(_distribution_row(f"{name}：仅片段", config["fragment_input_tokens"]))
    text.extend([_table(["配置 / 输入部分", "n 步", "mean", "P10", "P50", "P90", "P99", "max"], input_rows), "",
                 "## PLAN §6 判据：只填观察值", "",
                 "下表第一列保留 PLAN 的四条观察条件。条件中的阈值属于原规格，不是本次测得的阅读模型结果；"
                 "F1 / EM 均未测。差值统一为 oracle 减 BM25，单位为百分点（pp）；不据此判断是否开工，也不执行任务调整。", ""])
    medium = "<br>".join(_diff_sentence(mode, "mid", _difference(configs, mode, "mid")) for mode in ("reselect", "append"))
    all_differences = "<br>".join(_diff_sentence(mode, tier, _difference(configs, mode, tier))
                                 for mode in ("reselect", "append") for tier in ("tight", "mid", "loose"))
    text.extend([_table(["PLAN 原观察条件", "本次可测数字", "需阅读模型的部分"], [
        ["中档预算下，BM25 的证据召回比 oracle 低 15 个点以上，F1 低 5 个点以上", medium, "未测（需阅读模型）"],
        ["BM25 已接近 oracle", all_differences + "<br>PLAN 未定义‘接近’阈值，本报告不新增阈值。", "未测（需阅读模型）"],
        ["不压缩的 F1 与 oracle 接近", "未测（需阅读模型）", "未测（需阅读模型）"],
        ["不压缩的 F1 明显低于 oracle", "未测（需阅读模型）", "未测（需阅读模型）"],
    ]), "", "## 自动校验与边界", "",
        "oracle 在每档预算的任务末召回是否高于 latest / bm25，由最终保留证据直接核对；"
        "如出现反例，按下列校验记录保留并解释，不修改结果来满足预期。", ""])
    validations = results["validation"]
    if isinstance(validations, dict):
        rows = [[key, json.dumps(value, ensure_ascii=False, sort_keys=True)] for key, value in validations.items()]
        text.extend([_table(["校验项", "结果 / 计数 / 说明"], rows), ""])
    else:
        text.extend(["```json", json.dumps(validations, ensure_ascii=False, indent=2), "```", ""])
    text.extend(["本报告仅覆盖任务 02 授权的无阅读模型部分。"
                 "[机器可读结果](feasibility_results.json)与[运行配置及校验和](run_config.json)保留同次运行的数字和来源。", ""])
    return "\n".join(text)
