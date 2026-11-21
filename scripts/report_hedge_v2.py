#!/usr/bin/env python3
"""Build the Task 09 Chinese report and machine-readable tables from frozen CPU-readable receipts."""
import argparse
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EVAL = PROJECT / "eval"
OUT = EVAL / "hedge_v2"
EXTERNAL = Path("/root/userfolder/ctx-compressor/hedge_v2")
RUNS = {"H0": "nohedge", "H1": "tau_star", "H2": "always"}
CELLS = [f"{mode}|{tier}" for mode in ("reselect", "append") for tier in ("tight", "mid", "loose")]
SCOPES = ["pooled", *CELLS]
DISPLAY = ["pooled", "reselect|tight", "append|tight", "reselect|mid", "append|mid"]
INVALID = ("invalid_json", "invalid_candidate_id", "append_drop", "over_budget", "any_invalid")
NAMES = {"H0": "H0 关闭加段", "H1": "H1 训练阈值", "H2": "H2 预算允许即加段",
         "SFT v2": "SFT v2 checkpoint-2000", "GRPO": "GRPO checkpoint-500",
         "BM25": "BM25", "Oracle": "Oracle"}
BUCKETS = ("0", "1", "2", "3+")


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def scope_name(scope):
    if scope == "pooled":
        return "合并"
    mode, tier = scope.split("|")
    return {"append": "只追加", "reselect": "重选"}[mode] + " · " + {"tight":"紧","mid":"中","loose":"松"}[tier]


def metric_row(metrics, extra, scope, has_wrapper):
    if scope == "pooled":
        m = metrics["combined"]
        e = extra["combined"]
    else:
        m = metrics["configs"][f"{metrics['strategy']}|{scope}"]["metrics"]
        e = extra["configs"][scope]
    trajectories, steps = (1494,13956) if scope == "pooled" else (249,2326)
    ensure((m["n_tasks"], m["n_steps"], e["n_steps"]) == (trajectories, steps, steps), "Incomplete held-out grid")
    for item in (m["final_recall"], m["evidence_loss"], m["occupancy"], m["cost"]["cached"]):
        ensure(item["n"] == trajectories, "Trajectory denominator changed")
    invalid = m["invalid_outputs"]
    ensure(all(invalid[k]["n"] == steps for k in INVALID), "Invalid action denominator changed")
    additions = e["mean_added_per_valid"]
    ensure(additions["n"] == steps - invalid["any_invalid"]["count"], "Valid step denominator changed")
    ensure(sum(invalid[k]["count"] for k in INVALID[:-1]) == invalid["any_invalid"]["count"], "Invalid counts do not sum")
    return {"n_trajectories": trajectories, "n_steps": steps, "final_recall": m["final_recall"],
            "evidence_loss": {k:m["evidence_loss"][k] for k in ("count","n","rate")},
            "step_recall": m["step_recall"]["micro"], "occupancy": m["occupancy"],
            "cached_cost": m["cost"]["cached"], "invalid_rates": {k:invalid[k] for k in INVALID},
            "mean_added_per_step": {"sum": additions["sum"], "n": steps, "mean": additions["sum"] / steps},
            "mean_added_per_valid_step": additions, "wrapper_applicable": has_wrapper,
            "hedge_rate": e.get("hedge_rate") if has_wrapper else None,
            "hedge_hit_rate": e.get("hedge_hit_rate") if has_wrapper else None}


def gather():
    sources = {}
    def record(path):
        key = str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path)
        data = path.read_bytes()
        sources[key] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    check = record(OUT / "score_checks.json")
    replay = record(OUT / "score_checks_cpu_replay.json")
    train = record(OUT / "train_selection.json")
    pipe = record(EXTERNAL / "pipeline_status.json")
    ensure(pipe["status"] == "evaluation_complete" and
           [(s["stage"], s["exit_code"]) for s in pipe["stages"]] ==
           [("train_scores",0), ("H0",0), ("H1",0), ("H2",0)], "Unfinished pipeline")
    ensure(train["split"] == "train" and train["heldout_overlap"] == 0 and train["n_states"] == 22158,
           "Train source changed")
    grid = [0., .5, .6, .7, .8, .9, .95, .99, 1.01]
    ensure(train["grid"] == grid and len(train["cells"]) == 6 and
           all(cell[str(t)]["n"] == 3693 for cell in train["cells"].values() for t in grid),
           "Missing train grid entries")
    selected = min(grid, key=lambda t: (-train["pooled"][str(t)]["mean_reward"]["mean"], t))
    ensure(selected == train["tau_star"] == pipe["tau_star"], "Train selection differs from registered rule")
    ensure(check["status"] == "ok" and replay["n"] == check["sample_n"] == 300
           and replay["argmax_matches"] == check["argmax"]["matched"]
           and replay["recomputed_candidate_scores"] == check["n_candidate_scores_reconstructed"]
           and replay["full_token_round_trips"] == 300 and replay["max_abs_logprob_error_nats"] <= .02,
           "Score implementation checks failed")
    models, h0, diagnostics, provenance = {}, {}, {}, {}
    for label, name in RUNS.items():
        folder = OUT / name
        run = record(folder / "run.json")
        ensure(run["status"] == "complete" and run["tau"] == {"H0":0., "H1": selected, "H2":1.01}[label]
               and run["step0"]["n"] == 1494 and run["step0"]["user_byte_mismatches"] == 0
               and run["step0"]["token_id_mismatches"] == 0 and run["engine_shutdown"]["completed"],
               f"Run validation failed: {name}")
        provenance[label] = {k:run[k] for k in ("name","tau","model","adapter","adapter_weight","heldout_input","step0",
                                               "checks","selection","score_checks","seed","max_tokens","temperature")}
        metrics, extra = record(folder/"metrics.json"), record(folder/"extra.json")
        models[label] = (metrics, extra)
        diagnostics[label] = {}
        for scope in SCOPES:
            e = extra["combined"] if scope == "pooled" else extra["configs"][scope]
            diagnostics[label][scope] = {k:e[k] for k in (
                "executed_add_counts_all_steps","mean_kept_tokens_per_step","mean_extra_tokens_per_step",
                "new_gold_by_title","current_gold_valid_new_by_title","add_counts")}
            if label == "H0":
                deltas = extra["metric_deltas_vs_sft_v2"]
                h0[scope] = {"raw_diff": e["raw_diff_identical_state"],
                             "raw_diff_identical_model_prompt": e["raw_diff_identical_model_prompt"],
                             "deltas": deltas["combined"] if scope == "pooled" else deltas["configs"][scope]}
    archived = record(EVAL/"grpo_v2/merged_sft_v2/extra.json")
    grpo = EVAL/"grpo_v2/checkpoint-500"
    models["SFT v2"] = (record(EVAL/"sft_run_v2/checkpoint-2000_results.json"),archived["sft_v2_baseline"])
    models["GRPO"] = (record(grpo/"checkpoint-500_results.json"),record(grpo/"extra.json"))
    for strategy in ("BM25","Oracle"):
        key = strategy.lower()
        models[strategy] = (record(EVAL/"sft_run"/f"{key}_results.json"),archived["cited_rule_baselines"][key])
    comparison = {scope: {name: metric_row(m,e,scope,name in RUNS) for name,(m,e) in models.items()}
                  for scope in SCOPES}
    for scope, info in h0.items():
        current, baseline = comparison[scope]["H0"], comparison[scope]["SFT v2"]
        for source_key, key, field in (("evidence_loss.rate","evidence_loss","rate"),
                                       ("final_recall.mean","final_recall","mean"),
                                       ("cost.cached.mean","cached_cost","mean")):
            ensure(abs(current[key][field]-baseline[key][field]-info["deltas"][source_key]["delta"]) < 1e-12,
                   f"H0 delta is inconsistent for {source_key}")
    return {"schema_version":1, "status":"evaluation_complete","experiment":"任务09：低置信度时额外保留一个片段",
            "definitions_zh": {"scope":"合并或模式|预算。相同留出任务在六个模式与预算组合中重复。",
               "cached_cost":"每条轨迹内部累计缓存成本，再按轨迹平均，单位为等价基础输入 token。",
               "occupancy":"每条轨迹末步保留 token 除以有效预算，再按轨迹平均。",
               "hedge_rate":"执行了额外加段的步数 / 全部步数。未使用加段器的历史基线记为 null。",
               "hedge_hit_rate":"额外片段属于当前可用金标的步数 / 执行加段的步数。分母为零时 rate 为 null。",
               "mean_added_per_step":"全部动作步内平均新增的不同片段数。无效动作保持原状，新增数为零。",
               "current_gold_miss_rate":"当前金标在本步新候选中且此前未保留，但执行后仍未保留的比例，以有资格的步数为分母。",
               "raw_diff":"同一任务、模式、预算和步号对齐后，模型提示和完整输入状态均一致的步中原始贪心输出不同的数量。"},
            "score_checks":check, "score_cpu_replay":replay, "train":train, "heldout_runs":provenance,
            "h0_consistency":h0, "comparison":comparison, "run_diagnostics":diagnostics,
            "cost_vs_loss": {n:{k:comparison["pooled"][n][k] for k in ("cached_cost","evidence_loss")}
                             for n in ("H0","H1","H2","Oracle")},
            "pipeline": {"status":pipe["status"], "tau_star":selected,
                         "stages":[{k:s[k] for k in ("stage","status","exit_code","started_utc","ended_utc")}
                                   for s in pipe["stages"]]}, "source_sha256":sources}


def rate(m, digits=2):
    if m is None:
        return "不适用"
    if not m["n"]:
        return "未定义（n=0）"
    return f"{100*m['rate']:.{digits}f}%（{m['count']:,}/{m['n']:,}）"


def mean(m, digits=3, percentage=False):
    value = m["mean"] * (100 if percentage else 1)
    unit = "%" if percentage else ""
    return f"{value:,.{digits}f}{unit}（n={m['n']:,}）"


def row(*values):
    return "| " + " | ".join(map(str,values)) + " |"


def header(*fields):
    return [row(*fields),row(*(["---"]*len(fields)))]


def report(data):
    train = data["train"]
    tau = train["tau_star"]
    comp = data["comparison"]
    c = data["score_checks"]
    a, b, z = c["argmax"], c["logprob"], c["reencode"]
    tstar = train["pooled"][str(tau)]
    N = c["sample_n"]
    lines = ["# 任务09　低置信度时多保留一个片段的成本与证据丢失", "",
             f"在 {train['n_states']:,} 个训练状态上，平均单步奖励最高的阈值为 **τ={tau}**，平均奖励 {mean(tstar['mean_reward'],6)}。"
             f"留出集采用这一阈值时（H1），有 {rate(comp['pooled']['H1']['evidence_loss'])} 的轨迹在末步缺失至少一个金标片段，"
             f"末步平均召回为 {mean(comp['pooled']['H1']['final_recall'],3,True)}，每条轨迹平均缓存成本为 "
             f"{mean(comp['pooled']['H1']['cached_cost'],2)} 等价输入 token。", "",
             "## 打分核对", "",
             "策略使用原始 Qwen3-1.7B 加任务06的 SFT v2 checkpoint-2000 适配器。对每个本步新到且尚未保留的片段，"
             '在原评测提示下累加 JSON 前缀 `{"add":[`、片段本地编号和结束分隔符 `],"` 的 token 对数概率。'
             "候选分数经 softmax 归一化，最高概率低于阈值时，先执行有效的贪心动作，再尝试加入排名最高且尚未加入的候选。"
             "增加片段后仍须满足预算。贪心动作无效时沿用原评测的空操作。", "",
             f"从任务06的全部 {c['valid_population']:,} 个有效步骤中随机固定 n={N} 步。打分器对所有候选的共同贪心前缀，"
             "由模型原始 logits 在 float64 中独立归一化，分叉后的 token 按指定候选强制解码。评估时读取同一次前向计算中 "
             "vLLM 自行返回的生成对数概率。", "",
             *header("检查","观测"),row("最高分候选与原贪心添加编号一致",
                     f"{a['rate']:.2%}（{a['matched']}/{a['n']}，门槛 {a['required_rate']:.0%}）"),
             row(f"贪心候选分数与生成对数概率的差值不超过 {b['tolerance']} nat",
                 f"{b['within_tolerance']}/{b['n']}，最大绝对差 {b['max_abs_error']:.8g} nat（n={b['n']}）"),
             row("包含终止符的诊断原文重编码后与完整生成 ID 相同",
                 f"{z['returned_raw_full_id_matches']}/{z['n']}"),
             row("任务06默认展示正文重编码后与正文 token 相同",f"{z['text_token_matches']}/{z['n']}"),
             row("任务06默认展示正文直接重编码后与含终止符的完整 ID 相同",f"{z['strict_full_id_matches']}/{z['n']}"),
             "",f"原档案有 {z['terminal_eos_count']}/{z['n']} 个输出以 EOS 终止。按原始记录补回 EOS，完整序列逐条相同 "
             f"（{z['full_with_terminal_matches']}/{z['n']}）。CPU 逐项复算 {c['n_candidate_scores_reconstructed']:,}/"
             f"{c['n_candidate_scores_reconstructed']:,} 个候选分数，概率核对覆盖 n={c['n_independently_normalized_tokens']:,} 个生成 token。"
             " [打分回执](score_checks.json) · [CPU 复核](score_checks_cpu_replay.json)", "",
             "## 训练状态上的阈值选择", "",
             f"固定 SFT 策略提供 n={train['n_tasks']:,} 个训练任务、n={train['n_trajectories']:,} 条轨迹、n={train['n_states']:,} 个状态。"
             f"在每个状态上对同一贪心动作试算各阈值；按合并平均单步奖励取最大值，并列时选较小阈值。"
             f"奖励系数 λ={train['lambda']}。阈值表的当前金标漏加率只计当前金标在新候选中、此前未保留的步骤。"
             "加段率与额外 token 按全部状态平均，命中率按实际额外加段的次数计算。 [训练集选点回执](train_selection.json)"]
    for scope in SCOPES:
        lines += ["", "### " + scope_name(scope), "",
                  *header("阈值 τ","加段率","当前金标漏加率","平均额外 token/步","平均单步奖励","加段命中率")]
        group = train["pooled"] if scope == "pooled" else train["cells"][scope]
        for threshold in train["grid"]:
            m = group[str(threshold)]
            lines.append(row(f"{threshold:g}",rate(m["hedge_rate"]),rate(m["current_gold_miss_rate"]),
                             mean(m["mean_extra_tokens"],3),mean(m["mean_reward"],6),rate(m["hedge_hit_rate"])))

    lines += ["","## H0 与任务06的一致性","",
              "H0 关闭额外加段。每次闭环复用任务05留出文件的 SHA-256，并与任务06对齐首步用户文本与提示 token ID；"
              "H0、H1、H2 的两项首步差异均为 0/1,494（每次运行）。下表逐步对齐模式、预算、任务、步号，核对提示和完整输入状态"
              "（包括金标与片段编号映射），再比对原始贪心输出。差值采用 H0 减 SFT v2，比例差为百分点，成本差为等价输入 token。"
              " [H0 逐项对照](nohedge/extra.json) · [任务06指标](../sft_run_v2/checkpoint-2000_results.json)", "",
              *header("模式与预算","相同状态上原始输出不同","末召回差","证据丢失率差","单步召回差","按轨迹平均的单步召回差","末步占用率差","缓存成本差")]
    def delta(d, key, pct=True):
        v = d[key]
        return f"{v['delta']*(100 if pct else 1):+.4f}（n={v['n']:,}）"
    for scope in SCOPES:
        info = data["h0_consistency"][scope]
        d = info["deltas"]
        lines.append(row(scope_name(scope),rate(info["raw_diff"],3),delta(d,"final_recall.mean"),
                         delta(d,"evidence_loss.rate"),delta(d,"step_recall.micro.mean"),delta(d,"step_recall.task_macro.mean"),
                         delta(d,"occupancy.mean"),delta(d,"cost.cached.mean",False)))
    lines += ["",*header("模式与预算","非法 JSON 率差","无效候选编号率差","只追加模式删除率差","超预算率差")]
    for scope in SCOPES:
        d=data["h0_consistency"][scope]["deltas"]
        lines.append(row(scope_name(scope),*[delta(d,"invalid_outputs."+k+".rate") for k in INVALID[:-1]]))
    lines += ["","## 留出集闭环比较","",
              "每种策略在 n=249 个相同的留出任务上执行两种模式与三档预算，合并 n=1,494 条轨迹。"
              "每种加段策略各运行一次（n=1 次/策略）。末召回按轨迹平均，证据丢失指末步缺失至少一个金标片段；"
              "单步召回以步为单位汇总。末步占用率为最后一步保留 token 数除以有效预算。"
              "缓存成本为每条轨迹累计的等价基础输入 token，延续原评测的读写计价。"
              "每步新增的分母包括无效动作的空操作，同时列出仅按有效步计算的均值。历史基线未使用额外加段机制，两个加段指标记为不适用。"
              " [SFT v2](../sft_run_v2/checkpoint-2000_results.json) · "
              "[GRPO](../grpo_v2/checkpoint-500/checkpoint-500_results.json) · [BM25](../sft_run/bm25_results.json) · "
              "[Oracle](../sft_run/oracle_results.json) · [历史新增片段统计](../grpo_v2/merged_sft_v2/extra.json)"]
    for scope in DISPLAY:
        lines += ["","### " + scope_name(scope),"",
                  *header("策略","末召回","证据丢失率","单步召回","末步占用率","缓存成本/轨迹","新增片段/步","新增片段/有效步","加段率","加段命中率")]
        for name,m in comp[scope].items():
            lines.append(row(NAMES[name],mean(m["final_recall"],3,True),rate(m["evidence_loss"]),
                             mean(m["step_recall"],3,True),mean(m["occupancy"],2,True),mean(m["cached_cost"],2),
                             mean(m["mean_added_per_step"],4),mean(m["mean_added_per_valid_step"],4),
                             rate(m["hedge_rate"]),rate(m["hedge_hit_rate"])))
        lines += ["",*header("策略","非法 JSON","无效候选编号","只追加模式删除","超预算","全部无效")]
        for name,m in comp[scope].items():
            lines.append(row(NAMES[name],*[rate(m["invalid_rates"][k],3) for k in INVALID]))

    lines += ["","## 加段后的动作与证据","",
              "每步新增数量按执行后实际多保留的片段计数。无效动作维持原状，计入新增零段。"
              "同标题组指原始检索返回中另有不同片段与合格金标的标题完全相同；异标题组指没有这样的同标题片段。"
              "两组均按新返回、此前未保留、截至本步已揭示的金标片段计数，包含无效步。"
              " [H0](nohedge/extra.json) · [H1](tau_star/extra.json) · [H2](always/extra.json)"]
    for name in RUNS:
        lines += ["","### " + NAMES[name],"",*header("模式与预算","新增 0 段","新增 1 段","新增 2 段","新增 ≥3 段",
                                                  "平均保留 token/步","平均额外 token/步","同标题新金标添加率","异标题新金标添加率")]
        for scope in SCOPES:
            d = data["run_diagnostics"][name][scope]
            split = d["new_gold_by_title"]
            lines.append(row(scope_name(scope),*[rate(d["executed_add_counts_all_steps"][k],3) for k in BUCKETS],
                             mean(d["mean_kept_tokens_per_step"],2),mean(d["mean_extra_tokens_per_step"],2),
                             rate(split["same"]),rate(split["different"])))

    lines += ["","## 缓存成本与证据丢失","",
              *header("策略","每条轨迹的平均缓存成本（等价输入 token）","末步证据丢失率")]
    for name,m in data["cost_vs_loss"].items():
        lines.append(row(NAMES[name],mean(m["cached_cost"],2),rate(m["evidence_loss"],3)))
    lines += ["","![合并留出集上各策略平均缓存成本与末步证据丢失率](cost_vs_loss.svg)","",
              "[结构化结果与数据来源](hedge_v2_results.json)"]
    return "\n".join(lines)+"\n"


def deliver(path, text, verify):
    if verify or path.exists():
        ensure(path.read_bytes() == text.encode("utf-8"), f"Report output differs: {path}")
    else:
        with path.open("x",encoding="utf-8") as stream:
            stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Compare both existing outputs byte for byte")
    args = parser.parse_args()
    data = gather()
    deliver(OUT/"hedge_v2_results.json",json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",args.verify)
    deliver(OUT/"hedge_v2_report.md",report(data),args.verify)


if __name__ == "__main__":
    main()

