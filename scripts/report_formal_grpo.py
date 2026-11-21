#!/usr/bin/env python3
"""Compile Task 08's formal GRPO report from completed experiment receipts on CPU."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "eval/grpo_v2"
EXTERNAL = Path("/root/userfolder/ctx-compressor/grpo_v2")
SFT = "checkpoint-2000"
MERGED = "merged_sft_v2"
CELLS = tuple(f"{mode}|{tier}" for mode in ("reselect", "append") for tier in ("tight", "mid", "loose"))
INVALID = ("invalid_json", "invalid_candidate_id", "append_drop", "over_budget")
FIRST = ("invalid_output", "new_gold_not_added", "old_gold_dropped", "both", "none")


def read(path):
    return json.loads(path.read_bytes())


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure(test, reason):
    if not test:
        raise ValueError(reason)


def gather():
    sources = {}
    def record(path):
        sources[str(path)] = file_sha(path)
        return read(path)

    pipeline = record(EXTERNAL / "pipeline_status.json")
    selected = record(OUT / "selected_checkpoint.json")
    check = record(OUT / "training_checks.json")
    ensure(pipeline["status"] == "evaluation_complete" and all(s["status"] == "complete" for s in pipeline["stages"]),
           "Incomplete pipeline")
    models = {}
    evaluations = {}
    for name in [MERGED, *[f"checkpoint-{s}" for s in check["checkpoint_steps"]]]:
        directory = OUT / name
        run = record(directory / "run.json")
        ensure(run["status"] == "complete" and run["engine_shutdown"]["completed"]
               and run["step0"]["n"] == 1494 and not run["step0"]["user_byte_mismatches"]
               and not run["step0"]["token_id_mismatches"], f"{name}: held-out run checks")
        models[name] = {"metrics": record(directory / f"{name}_results.json"),
                        "diagnostics": record(directory / "diagnostics.json"),
                        "extra": record(directory / "extra.json")}
        evaluations[name] = {key:run[key] for key in ("checks", "heldout_input", "engine_parameters", "step0",
                                                       "engine_shutdown", "completed_at_utc")}
    extra0 = models[MERGED]["extra"]
    for alias, strategy in (("SFT v2", SFT), ("BM25", "bm25"), ("Oracle", "oracle")):
        version = "v2" if strategy == SFT else "v1"
        folder = "sft_run_v2" if version == "v2" else "sft_run"
        diag = record(PROJECT / "eval/sft_run_v2" / f"diagnostics_{version}.json")["strategies"][strategy]
        extra = extra0["sft_v2_baseline"] if version == "v2" else extra0["cited_rule_baselines"][strategy]
        models[alias] = {"metrics":record(PROJECT / "eval" / folder / f"{strategy}_results.json"),
                         "diagnostics":diag, "extra":extra}
    ranked = sorted([n for n in models if n.startswith("checkpoint-")],
                    key=lambda n:(models[n]["metrics"]["combined"]["evidence_loss"]["count"],
                                  -models[n]["metrics"]["combined"]["final_recall"]["mean"],
                                  models[n]["metrics"]["checkpoint_step"]))
    winner = ranked[0]
    ensure(winner == selected["strategy"] == pipeline["selected_checkpoint"] and len(ranked) == selected["n_candidates"],
           "Stored checkpoint selection does not match ranking")

    prereg = {}
    for name in ("SFT v2", winner):
        m = models[name]["metrics"]
        prereg[name] = {"pooled_loss":m["combined"]["evidence_loss"], "pooled_final_recall":m["combined"]["final_recall"],
                        "tight_loss_by_mode":{mode:m["configs"][f"{m['strategy']}|{mode}|tight"]["metrics"]["evidence_loss"]
                                              for mode in ("reselect", "append")}}
    for name, obj in models.items():
        m, diag, ex = obj["metrics"], obj["diagnostics"], obj["extra"]
        ensure(len(m["configs"]) == 6 and len(diag["configs"]) == 6 and len(ex["configs"]) == 6, f"{name}: grid")
        ensure(m["combined"]["n_steps"] == ex["combined"]["n_steps"] == 13956
               and m["combined"]["n_tasks"] == 1494, f"{name}: denominators")
        for cell in CELLS:
            v = m["configs"][m["strategy"] + "|" + cell]["metrics"]
            c, e = diag["configs"][cell], ex["configs"][cell]
            ensure(v["n_tasks"] == c["n_trajectories"] == 249 and v["n_steps"] == c["n_steps"] == e["n_steps"] == 2326,
                   f"{name} {cell}: denominators")
            ensure(sum(x["count"] for x in c["first_error"].values()) == 249, f"{name} {cell}: first errors")
            ensure(sum(x["count"] for x in e["add_counts"].values()) == e["n_valid"], f"{name} {cell}: valid additions")

    states = record(EXTERNAL / "states/state_checks.json")
    probe = record(EXTERNAL / "filter/sampling_probe.json")
    data = record(EXTERNAL / "data/dataset_stats.json")
    n = states["n_states"]
    ensure(states["available_checks_pass"] and states["scope"] == "train" and states["n_tasks"] == 400
           and states["n_trajectories"] == 2400 and n == probe["n_states"] == data["n_states"]
           and not any(states[k] for k in ("gold_mismatches","reencode_mismatches","rerender_mismatches","heldout_overlap")),
           "B/C/D input checks")
    ensure(probe["pooled"]["n_completions"] == 8*n and probe["pooled"]["informative"]["count"] == data["n_informative"]
           and data["n"] == min(2000,data["n_informative"]) == sum(data["per_cell_counts"].values())
           and data["user_text_mismatches"] == data["heldout_rows"] == 0, "Filter/dataset checks")
    csv_path = OUT / "training_curve.csv"
    with csv_path.open() as f:
        curve = list(csv.DictReader(f))
    ensure(file_sha(csv_path) == check["curve_sha256"] and len(curve) == check["optimizer_steps"] == data["n"] // 4
           and check["status"] == "ok" and not check["runtime"]["oom"] and not check["runtime"]["nonfinite"]
           and check["reward_recheck"]["mismatches"] == check["reward_recheck"]["unmatched_prompts"] == 0, "E checks")
    sources[str(csv_path)] = check["curve_sha256"]
    train_means = {key:{"mean":statistics.mean(float(row[key]) for row in curve),"n":len(curve)}
                   for key in ("reward","reward_std","frac_reward_zero_std","kl","step_seconds")}
    inputs = record(OUT / "input_checks.json")
    merge = record(OUT / "a_checks.json")
    ensure(merge["n_reference"] == inputs["fixture_n"] == 30 and inputs["mismatches"] == inputs["overlapping_question_ids"] == 0,
           "Merge fixture or training projection checks")
    ensure(pipeline["resume"]["verified_state_file"]["sha256"] == states["state_file"]["sha256"] == probe["state_file"]["sha256"],
           "Resumed pipeline state hash differs from filter input")
    ensure(len({v["step0"]["identity_and_hashes_sha256"] for v in evaluations.values()}) == 1
           and len({v["heldout_input"]["sha256"] for v in evaluations.values()}) == 1, "F0 and F inputs differ")
    return {"schema_version":1,"experiment":"Task 08 GRPO", "status":pipeline["status"], "source_sha256":sources,
            "input_checks":inputs, "merge":merge,
            "pipeline":pipeline, "states":states, "filter":probe, "dataset":data,
            "training":check, "optimizer_step_means":train_means,
            "evaluations":evaluations, "models":models, "selected":selected, "preregistered":prereg}


def rate(v):
    if not v["n"]:
        return "不适用（n=0）"
    return f"{v['rate']*100:.3f}%（{v['count']:,}/{v['n']:,}）"


def mean(v, percent=False, digits=3):
    value = v["mean"] * (100 if percent else 1)
    return f"{value:.{digits}f}{'%' if percent else ''}（n={v['n']:,}）"


def number(value, n, digits=3):
    return f"{value:.{digits}f}（n={n:,}）"


def table(headers, rows):
    rows = list(rows)
    ensure(all(len(row) == len(headers) for row in rows), f"Table column mismatch for {headers}")
    def line(values):
        return "| " + " | ".join(str(v).replace("|","\\|") for v in values) + " |"
    return "\n".join([line(headers),line(["---"]*len(headers)),*[line(row) for row in rows]])


def cn(cell):
    return cell.replace("reselect","重选").replace("append","只追加").replace("loose","松").replace("tight","紧").replace("mid","中").replace("|"," · ")


def title(name):
    if name == MERGED:
        return "仅合并 SFT（F0）"
    if name.startswith("checkpoint-"):
        return f"GRPO {name}"
    return name


def f0_tables(result):
    ex = result["models"][MERGED]["extra"]
    sft = result["models"]["SFT v2"]["metrics"]["combined"]
    f0 = result["models"][MERGED]["metrics"]["combined"]
    def delta(a, percent=True):
        return f"{a['delta']*(100 if percent else 1):+.4f}（n={a['n']:,}）"
    rows, invalid = [], []
    delta_data = ex["metric_deltas_vs_sft_v2"]
    for key,d in [("合并",delta_data["combined"]), *[(cn(k),delta_data["configs"][k]) for k in CELLS]]:
        e = ex["combined"] if key == "合并" else ex["configs"][next(k for k in CELLS if cn(k) == key)]
        rows.append([key,delta(d["evidence_loss.rate"]),delta(d["final_recall.mean"]),delta(d["step_recall.micro.mean"]),
                     delta(d["step_recall.task_macro.mean"]),delta(d["occupancy.mean"]),delta(d["cost.cached.mean"],False),
                     rate(e["raw_diff_identical_state"])])
        invalid.append([key,*[delta(d[f"invalid_outputs.{k}.rate"]) for k in INVALID]])
    return [f"同一批留出任务上，F0 合并证据丢失率为 {rate(f0['evidence_loss'])}，SFT v2 为 {rate(sft['evidence_loss'])}。"
            f"两者的轨迹末召回均值分别为 {mean(f0['final_recall'],True)} 和 {mean(sft['final_recall'],True)}。",
            "下表给出 F0 减 SFT v2 的差。比例差用百分点表示，缓存成本差用等价基础输入 token 表示。"
            "最后一列先对齐模式、预算、任务和步号，再核对用户文本哈希、提示 token ID 及完整输入状态，统计其中原始输出字符串不同的步骤。",
            table(["模式与预算","丢失率差","末召回差","单步召回差","按任务平均的单步召回差","占用率差","缓存成本差","同状态下输出不同"],rows),
            table(["模式与预算","非法 JSON 率差","无效编号率差","追加模式删除率差","超预算率差"],invalid)]


def build_report(r):
    winner = r["selected"]["strategy"]
    models = r["models"]
    pre = r["preregistered"]
    rows = []
    for label,key in [("紧预算 · 重选 · 证据丢失率","reselect"),("紧预算 · 只追加 · 证据丢失率","append"),
                      ("合并 · 证据丢失率","pooled_loss"),("合并 · 末召回","pooled_final_recall")]:
        values = []
        for name in ("SFT v2",winner):
            if key in ("reselect","append"):
                values.append(rate(pre[name]["tight_loss_by_mode"][key]))
            elif key == "pooled_loss":
                values.append(rate(pre[name][key]))
            else:
                values.append(mean(pre[name][key],True))
        rows.append([label,*values])
    report = ["# 任务 08　GRPO 训练与留出集闭环评测",
              f"按预注册规则，在 {r['selected']['n_candidates']} 个 GRPO 保存点（n={r['selected']['n_candidates']}）中选中 **{winner}**。"
              "下表为一次训练的点估计（n=1 次训练）。丢失率指末步未保全全部已揭示证据的轨迹比例，末召回是轨迹末步已保留证据比例的均值。",
              table(["预注册指标","任务 06 SFT v2 checkpoint-2000",title(winner)],rows),
              "每格由同一组留出任务（n=249）各运行一条轨迹。合并统计覆盖模式与预算的全部组合（n=1,494 条轨迹），其中同一任务跨格重复。表内分数为分子/分母，均值后列出样本数。"]
    a,i=r["merge"],r["input_checks"]
    report.extend(["## A　合并与输入核对",
                   f"将基础模型与任务 06 选定的 LoRA 在 CPU 上以 fp32 合并，保存为 bf16。固定贪心样例中，生成文本差异 {a['text_mismatches']}/{a['n_reference']}，"
                   f"生成 token ID 差异 {a['token_id_mismatches']}/{a['n_reference']}。"
                   f"新旧训练输入在构造器读取字段上的差异为 {i['mismatches']:,}（n={i['n_train_tasks']:,} 个任务，n={i['n_train_steps']:,} 步）。"
                   f"训练与留出问题 ID 交集为 {i['overlapping_question_ids']}（训练 n={i['n_train_question_ids']:,}，留出 n={i['n_heldout_question_ids']:,}）。"
                   " [合并回执](a_checks.json) · [输入回执](input_checks.json)。","### F0 与原 SFT v2 的闭环差异",*f0_tables(r)])
    s = r["states"]
    report.extend(["## B　训练状态",
                   f"任务来自训练集（n={s['n_tasks']:,}），使用原 SFT v2 适配器生成轨迹（n={s['n_trajectories']:,}），共得到 n={s['n_states']:,} 个状态。"
                   "“保留非金标”表示操作前保留区含有已知金标以外的段落。“金标不可得”表示截至本步已有金标不在保留区和新返回集合的并集中。"])
    rows, lengths=[],[]
    for cell,d in [(c,s["cells"][c]) for c in CELLS]+[("合并",s["pooled"])]:
        v = s["prompt_lengths"] if cell=="合并" else d["prompt_lengths"]
        n = v["n"]
        rows.append([cn(cell),f"n={n:,}",*[rate(d[k]) for k in ("invalid","kept_non_gold","lost_gold","prior_gold_missing_from_kept")]])
        lengths.append([cn(cell),*[number(v[k],n, 2 if k=="p99" else 0) for k in ("min","p50","p99","max")],
                        f"{v['exceeds_8392']}/{n:,}",f"{v['prompt_plus_256_exceeds_8392']}/{n:,}"])
    report.extend([table(["模式与预算","状态数","动作无效","保留非金标","金标不可得","既往金标不在保留区"],rows),
                   table(["模式与预算","最短 token","中位 token","P99 token","最长 token","提示超过 8,392 token","提示加 256 token 后超过 8,392"],lengths)])
    v = s["step0_sft"]
    report.append(f"金标重算差异 {s['gold_mismatches']:,}/{s['gold_recompute_n']:,}，提示重编码差异 {s['reencode_mismatches']:,}/{s['prompt_reencode_n']:,}，"
                  f"文本重渲染差异 {s['rerender_mismatches']:,}/{s['n_states']:,}，训练状态涉及的任务与留出集问题 ID 交集为 {s['heldout_overlap']:,}（训练 n={s['train_example_ids']:,}，留出 n={s['heldout_example_ids']:,}）。"
                  f"原预算下逐字节核对首步，差异 {v['direct_mismatches']}/{v['direct_n']:,}。其余预算只替换 SFT 首步模板的预算行后核对，差异 {v['budget_substituted_mismatches']}/{v['budget_substituted_n']:,}。")
    p = r["filter"]
    g = p["generation"]
    report.extend(["## C　离线采样与奖励方差筛选",
                   f"每状态采样 n={g['n']} 条完成，温度和 top_p 均为 {g['temperature']:.1f}，上限 {g['max_tokens']} token，种子 {g['seed']}（n=1 套采样配置）。"
                   f"登记 top_k={g['top_k']}，vLLM 0.11.0 实际默认 top_k={g['top_k_actual']}，两者均关闭 top-k 截断。"
                   f"首次启动在检查默认值时退出（n=1 次）。续跑复核 F0 回执（n={r['pipeline']['resume']['f0_steps']:,} 步）、B 回执与状态文件 SHA-256（n={s['n_states']:,} 状态），随后从 C 开始。"
                   f"奖励使用 score_completion，λ={p['lambda']:.1f}。每组奖励取总体标准差（ddof={p['ddof']}，每组 n={g['n']}），严格大于 {p['informative_threshold']:g} 即入围。"
                   "下表吞吐按采样阶段实测时间计算，计时排除了引擎初始化和状态校验。"])
    groups, invalid, adds, executed = [],[],[],[]
    for cell,d in [(c,p["cells"][c]) for c in CELLS]+[("合并",p["pooled"])]:
        n = d["n_groups"]
        groups.append([cn(cell),rate(d["zero_std_groups"]),number(d["mean_within_group_std"],n,5),rate(d["some_sample_beats_greedy"]),
                       rate(d["informative"]),number(d["states_per_second"],n),number(d["elapsed_seconds"],n,1)])
        invalid.append([cn(cell),rate(d["invalid"]),*[rate(d["invalid_by_category"][k]) for k in INVALID]])
        adds.append([cn(cell),*[rate(d["sampled_add_counts_parsed"][k]) for k in ("0","1","2",">=3")],rate(d["unparseable_add_count"])])
        executed.append([cn(cell),*[rate(d["executed_add_counts_invalid_noop"][k]) for k in ("0","1","2",">=3")]])
    report.extend([table(["模式与预算","零标准差组","组内标准差均值","有样本奖励超过贪心","入围","状态/秒","采样秒数"],groups),
                   table(["模式与预算","全部无效","非法 JSON","无效候选编号","追加模式删除","超预算"],invalid),
                   "按可解析输出请求添加的不同片段数统计。无法解析占比另外以该格全部采样输出为分母。",
                   table(["模式与预算","请求零个","请求一个","请求两个","请求至少三个","无法解析"],adds),
                   "无效操作执行时保留原状态，按实际执行的新增片段数计数如下。",
                   table(["模式与预算","新增零个","新增一个","新增两个","新增至少三个"],executed)])
    ds = r["dataset"]
    report.extend(["## D　训练提示抽样",
                   f"从方差入围的 n={ds['n_informative']:,} 个状态中抽取 n={ds['n']:,} 条提示，目标缺口 {ds['shortfall_from_target']}/{ds['target_n']:,}。"
                   f"按格名排序分配配额，使用种子 {ds['seed']} 抽样并在末尾洗牌（n=1 次抽样）。"
                   f"用户文本相对 B 状态的字节差异 {ds['user_text_mismatches']}/{ds['rows_checked']:,}，留出集行 {ds['heldout_rows']}/{ds['rows_checked']:,}。",
                   table(["模式与预算（原键排序）","入围/该格状态","实际抽取/入围"],[
                       [cn(k),f"{ds['per_cell_available'][k]:,}/{s['cells'][k]['n']:,}",f"{ds['per_cell_counts'][k]:,}/{ds['per_cell_available'][k]:,}"]
                       for k in sorted(ds["per_cell_counts"])])])
    t = r["training"]; rc=t["reward_recheck"]; n=t["optimizer_steps"]
    report.extend(["## E　GRPO 训练",
                   "在合并权重上新建 all-linear LoRA。单次训练配置（n=1）为秩 16、α=32、dropout=0.05，学习率 5×10⁻⁶，cosine 调度，warmup=0.03，β=0.04。"
                   "每卡 batch=4，梯度累积 8，每个优化步包含 4 条提示、每提示 8 个样本，温度=1.0，top_p=1.0，top_k=−1，完成上限 256 token，总长上限 8,392 token。"
                   "共训练 1 个 epoch，每 125 步保存一次。vLLM 使用 colocate、利用率 0.30、eager、sleep_level=0，关闭动态采样，种子 20261120。",
                   f"总耗时 {number(t['runtime']['wall_seconds'],1)} 秒（约 4 小时 52 分，n={n:,} 个优化步）。设备监控峰值为 {number(t['peak_gpu1_used_mib'],t['n_gpu_samples'],0)} MiB。"
                   f"PyTorch 已分配与预留内存的峰值分别为 {number(t['peak_torch_allocated_bytes'],n,0)} 和 {number(t['peak_torch_reserved_bytes'],n,0)} 字节。"
                   f"OOM 为 {int(t['runtime']['oom'])}/1 次运行，非有限损失或梯度为 {int(t['runtime']['nonfinite'])}/{n:,} 步。",
                   f"优化步计划生成 n={rc['n_planned_completions']:,} 条完成，Swift 末尾重复记录一组（n={rc['extra_logged_completions']:,}）。"
                   f"对全部奖励日志重算，奖励差异 {rc['mismatches']:,}/{rc['n']:,}，无法匹配提示 {rc['unmatched_prompts']:,}/{rc['n']:,}。",
                   table(["逐步均值","值"],[[label,mean(r["optimizer_step_means"][k])] for k,label in
                        [("reward","奖励"),("reward_std","组内奖励标准差"),("frac_reward_zero_std","零标准差组比例"),("kl","KL"),("step_seconds","每步秒数")]]),
                   "[逐步训练数据](training_curve.csv) · [四项训练曲线](training_curve.svg) · [训练回执](training_checks.json)。"])
    report.extend(eval_tables(r))
    return "\n\n".join(report)+"\n"


def eval_tables(r):
    winner = r["selected"]["strategy"]
    models = r["models"]
    checkpoints = [f"checkpoint-{v}" for v in r["training"]["checkpoint_steps"]]
    names = ["SFT v2",MERGED,*checkpoints]
    combined=[]
    for name in names:
        d=models[name]["metrics"]["combined"]
        ex=models[name]["extra"]["combined"]["mean_added_per_valid"]
        combined.append([title(name),rate(d["evidence_loss"]),mean(d["final_recall"],True),
                         f"{ex['mean']:.6f}（{ex['sum']:,}/{ex['n']:,}）"])
    n_runs=len(r["evaluations"])
    heldout=next(iter(r["evaluations"].values()))["heldout_input"]["sha256"]
    report=["## F　留出集闭环评测",
            f"留出文件 SHA-256 为 {heldout}，与任务 05 输入相同（n=249 个任务）。F0 与全部 GRPO 保存点各进行一次贪心解码"
            f"（n={n_runs} 次评测，各 n=1,494 条轨迹、n=13,956 步）。首步用户文本和提示 token ID 均与任务 06 对齐（每次各 0/1,494 差异）。"
            "沿用固定检索返回和推理参数，完成长度上限 256 token，种子 20261118（n=1 套评测配置）。"
            "保存点先按合并丢失率升序，再按合并末召回降序，最后按保存步升序选择。"
            " [保存点选择回执](selected_checkpoint.json)。",
            table(["策略","合并丢失率","合并末召回","每有效步新增片段（片段/步）"],combined),
            "### 紧预算与中预算的策略比较",
            "紧、中预算分别为 1,876 和 2,814 token（每策略每预算 n=498 条轨迹）。单步召回先逐步计算，再按步平均，占用率使用末步的保留 token 数除以预算。缓存成本为每条轨迹累计的等价输入 token。"
            "无效动作按原协议保持保留区不变，添加数量按有效步报告。"
            " [SFT v2 指标](../sft_run_v2/checkpoint-2000_results.json) · [BM25 指标](../sft_run/bm25_results.json) · [Oracle 指标](../sft_run/oracle_results.json) · "
            "[SFT v2 诊断](../sft_run_v2/diagnostics_v2.json) · [规则策略诊断](../sft_run_v2/diagnostics_v1.json)。"]
    comparisons,invalids,diags,firsts=[],[],[],[]
    for tier in ("tight","mid"):
        for mode in ("reselect","append"):
            cell=f"{mode}|{tier}"
            for name in (winner,"SFT v2",MERGED,"BM25","Oracle"):
                obj=models[name]
                met=obj["metrics"]; v=met["configs"][met["strategy"]+"|"+cell]["metrics"]
                diag=obj["diagnostics"]["configs"][cell]
                ex=obj["extra"]["configs"][cell]
                start=[cn(cell),title(name)]
                comparisons.append(start+[mean(v["final_recall"],True),rate(v["evidence_loss"]),mean(v["step_recall"]["micro"],True),
                                          mean(v["occupancy"],True),mean(v["cost"]["cached"]),mean(ex["mean_added_per_valid"],digits=6)])
                invalids.append(start+[rate(v["invalid_outputs"][k]) for k in INVALID])
                diags.append(start+[rate(diag[k]) for k in ("new_gold_add","kept_gold_survival","all_gold_valid_drop")])
                firsts.append(start+[rate(diag["first_error"][k]) for k in FIRST])
    report.extend([table(["模式与预算","策略","末召回","丢失率","单步召回","占用率","缓存成本","有效步平均新增"],comparisons),
                   table(["模式与预算","策略","非法 JSON","无效候选编号","追加模式删除","超预算"],invalids),
                   "新金标添加率按片段计数，包含本步返回、此前未保留且截至本步已揭示的金标。旧金标存活率对重选模式逐片段统计。"
                   "纯金标保留区的删除率以符合条件的有效重选步为分母。",
                   table(["模式与预算","策略","新金标添加率","旧金标存活率","纯金标区发生删除"],diags),
                   "首次错误按每条轨迹的最早一步计数，无效动作优先，同步漏加与误删计入“两者同时”。末列为全程未触发上述错误的轨迹。",
                   table(["模式与预算","策略","无效动作","漏加新金标","误删旧金标","两者同时","全程无错误"],firsts),
                   "### 按模式统计添加行为与同标题干扰",
                   "新增片段数分布以有效步为分母。同标题组表示当次原始检索返回中还有另一不同片段与目标新金标标题完全相同，另一组为没有同标题片段。"
                   "两组的添加率均以合格新金标片段数为分母，覆盖有效和无效步。"])
    extras=[]
    for name in names:
        for mode,ex in models[name]["extra"]["modes"].items():
            extras.append([title(name),cn(mode),mean(ex["mean_added_per_valid"],digits=6),
                           *[rate(ex["add_counts"][k]) for k in ("0","1","2","3+")],
                           *[rate(ex["new_gold_by_title"][k]) for k in ("same","different")]])
    report.extend([table(["策略","模式","有效步平均新增","新增零个","新增一个","新增两个","新增至少三个","同标题组添加率","无同标题组添加率"],extras),
                   "各格诊断、未舍入指标、抽样与训练回执的 SHA-256 见 [机器可读结果](grpo_v2_results.json)。"])
    return report


def main():
    result=gather()
    payloads={OUT/"grpo_v2_report.md":build_report(result),
              OUT/"grpo_v2_results.json":json.dumps(result, ensure_ascii=False, separators=(",",":"),allow_nan=False)+"\n"}
    for path, payload in payloads.items():
        if path.exists():
            ensure(path.read_text() == payload, f"Refusing to replace changed output: {path}")
    for path, payload in payloads.items():
        if not path.exists():
            with path.open("x", encoding="utf-8") as handle:
                handle.write(payload)


if __name__ == "__main__":
    main()

