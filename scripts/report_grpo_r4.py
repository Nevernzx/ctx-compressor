"""Publish the Task 18 report from the completed C through G receipts."""
import csv
import io
from math import isclose, isfinite
import os
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import close_reranker_grpo_metrics as m
from scripts import grpo_r4_eval as eval_r4
from scripts import grpo_r4_reader as reader
from src.reader_heldout_inputs import write_new
from src.reranker_v1 import need, read, record


def curve():
    path = m.T18 / "training_curve.csv"
    tr = read(m.T18 / "training_checks.json")
    need(m.checked(tr["curve"]) == path and tr["optimizer_steps"] == 500, "training curve receipt differs")
    with path.open() as f:
        entries = [{k: float(v) for k,v in a.items()} for a in csv.DictReader(f)]
    need(len(entries) == tr["optimizer_steps"], "incomplete curve")
    for step, a in enumerate(entries, 1):
        reward = -a["mean_n_lost"] - .16*a["mean_n_nongold_after"] - .5*a["mean_used_per_1000"] - a["invalid_rate"]
        need(a["step"] == step and all(isfinite(x) for x in a.values()) and a["reward_component_n"] == 32
             and isclose(a["reward"], reward, rel_tol=0, abs_tol=1e-5), f"training curve mismatch: step {step}")
    return entries, tr["curve"]


def plot(entries):
    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter
    panels = (
        ("reward","Mean reward v3","#245D96"),
        ("reward_std","Within-group reward SD","#128276"),
        ("frac_reward_zero_std","Groups with zero reward SD","#9B5B13"),
        ("kl","KL divergence","#74549A"),
        ("mean_n_lost","Lost gold passages / generation","#A93554"),
        ("mean_n_nongold_after","Non-gold kept after action / generation","#326A47"),
    )
    x = [r["step"] for r in entries]
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":9,"svg.fonttype":"none",
                         "svg.hashsalt":"ctx-task18","path.simplify":False}):
        fig, axes = plt.subplots(2,3,sharex=True,figsize=(12.3,6.6),layout="constrained")
        fig.suptitle(f"Reward v3 training  (n = {len(x)} optimizer steps)",fontsize=15,fontweight="semibold")
        for ax,(key, label, color) in zip(axes.flat,panels,strict=True):
            line, = ax.plot(x,[r[key] for r in entries],color=color,lw=.84)
            line.set_gid(f"trace-{key}")
            ax.set_title(label,loc="left",fontsize=10,fontweight="semibold")
            ax.set_xlim(x[0],x[-1])
            ax.set_xticks([1,125,250,375,500])
            if key=="frac_reward_zero_std":
                ax.yaxis.set_major_formatter(PercentFormatter(1.0,decimals=0))
            ax.grid(axis="y",color="#D4DCE0",lw=.55)
            ax.set_axisbelow(True)
            ax.spines[["top","right"]].set_visible(False)
        fig.supxlabel("Optimizer step")
        output = io.BytesIO()
        fig.savefig(output,format="svg",metadata={"Date":None,"Title":"Task 18 GRPO training with reward v3"})
        plt.close(fig)
    result = output.getvalue()
    root = ET.fromstring(result)
    ns = {"s":"http://www.w3.org/2000/svg"}
    for key, _, _ in panels:
        path = root.find(f".//s:g[@id='trace-{key}']/s:path", ns)
        need(path is not None and path.attrib["d"].split().count("L") == len(entries)-1, f"incomplete curve: {key}")
    return result


def f_results():
    selected = read(m.T18 / "selected_checkpoint.json")
    history = read(m.T18 / "historical_reward_v3.json")
    need(selected["n_candidates"] == 4 and history["status"] == "complete", "F selection or historical reward missing")
    data = {}
    for step in m.r4.CHECKPOINT_ORDER:
        run, compression, extra = eval_r4.check_eval(step)
        name = f"grpo_r4:checkpoint-{step}"
        need(m.checked(run["adapter_weight"]).parent == m.r4.TRAIN / f"checkpoint-{step}", "F adapter differs")
        pooled = extra["combined"]
        candidate = next(r for r in selected["candidates"] if r["checkpoint_step"]==step)
        need(candidate["heldout_reward_v3"] == pooled["heldout_reward_v3"]
             and candidate["final_context"] == pooled["final_context"]
             and candidate["evidence_loss"] == compression["combined"]["evidence_loss"]
             and candidate["final_recall"] == compression["combined"]["final_recall"]
             and m.checked(candidate["run"]) == m.T18 / f"checkpoint-{step}/run.json", "F candidate differs")
        data[name] = {"compression":compression, "heldout": {
            **{k:{name:v[name] for name in ("heldout_reward_v3","final_context")} for k,v in extra["configs"].items()},
            "pooled":{name:pooled[name] for name in ("heldout_reward_v3","final_context")},
        }}
    choice = eval_r4.select_checkpoint(selected["candidates"])
    need(all(value==selected[key] for key,value in choice.items()), "F objective selection differs")
    for name, ref in history["strategies"].items():
        compression = read(m.checked(ref["metrics"]))
        need(compression["steps_artifact"]==ref["steps"], "historical rollout receipt differs")
        m.checked(ref["steps"])
        data[name] = {"compression":compression,"heldout":ref["heldout"]}
    return selected, data, history


def g_results(f, g):
    pre = read(m.HELD18 / "input_preflight.json")
    need(g["status"] == "complete" and g["n_predictions"] == 35856 and g["batch_size"] == 144
         and pre["input_truncations"] == pre["historical_prompt_mismatches"] == 0, "G coverage or past prompt mismatch")
    finals = read(m.checked(pre["final_context_file"]))
    need(len(finals) == len(f) and all(len(g)==1494 for g in finals.values()), "G final contexts incomplete")
    enriched = {}
    for k, data in f.items():
        compression = data["compression"]
        by_cell = {f"{a['mode']}|{a['budget_tier']}":a["metrics"] for a in compression["configs"].values()}
        by_cell["pooled"] = compression["combined"]
        enriched[k] = {}
        for cell, comp in by_cell.items():
            answer = g["strategies"][k]["pooled"] if cell=="pooled" else g["strategies"][k]["cells"][cell]
            ctx = data["heldout"][cell]["final_context"]
            qa_count = comp["evidence_complete_questions"]
            need(answer["n"] == qa_count["n"] and answer["by_evidence_complete"]["true"]["n"] == qa_count["count"]
                 and all(ctx[x]["n"] == comp["n_tasks"] for x in ("segments","nongold")), f"G context denominator: {k}/{cell}")
            if cell!="pooled":
                cost = pre["costs"][k][cell]
                need(isclose(cost["recomputed_cached_mean"],comp["cost"]["cached"]["mean"],rel_tol=1e-12),
                     f"cached cost mismatch: {k}/{cell}")
            enriched[k][cell] = {
                "reader":answer, "final_context":ctx, "heldout_reward_v3":data["heldout"][cell]["heldout_reward_v3"],
                "cost":comp["cost"], "evidence_loss":comp["evidence_loss"], "final_recall":comp["final_recall"],
                "step_recall":comp["step_recall"], "evidence_complete_questions":qa_count,
            }
    return enriched


def render(r):
    c, d, e, smoke, g = (r[k] for k in ("C","D","E","smoke","G"))
    sel, perf = r["selected_checkpoint"], r["strategies"]
    selected_name = g["selected"]
    train_pairs = e["reward_recheck"]
    smoke_pairs = smoke["reward_recheck"]
    c_rows=[]
    for cell in ("pooled",*m.CELLS):
        shared_v3 = c["pooled"] if cell=="pooled" else c["cells"][cell]
        prev = c["task11_c"]["pooled"] if cell=="pooled" else c["task11_c"]["cells"][cell]
        for tag, a in (("v2（任务11）",prev),("v3",shared_v3)):
            c_rows.append((cell,tag, f"{a['n_groups']:,}", m.rate(a["zero_std_groups"]),m.rate(a["gold_varying"]),
                m.rate(shared_v3["nongold_varying"]),m.rate(a["token_only_varying"]),m.rate(a["some_sample_beats_greedy"])))
    d_rows = []
    for cell in ("pooled",*m.CELLS):
        total = d["n"] if cell=="pooled" else d["per_cell_counts"][cell]
        comp = d["composition"] if cell=="pooled" else d["per_cell_composition"][cell]
        d_rows.append((cell,f"{total:,}",*(m.rate(comp[key]) for key in
                       ("gold_varying","nongold_only_varying","token_only_varying"))))
    order = [selected_name,*[k for k in m.r4.CHECKPOINT_ORDER if k!=sel["checkpoint_step"]],m.R2,m.V3]
    order = [f"grpo_r4:checkpoint-{k}" if isinstance(k,int) else k for k in order]
    answers, compression, reward, refs = [], [], [], []
    for name in order:
        view = perf[name]
        a = view["pooled"]
        label = m.LABEL[name]
        qa = a["reader"]
        answers.append((label,"任务18 G",f"{qa['n']:,}",m.pct(qa["f1"],3),m.pct(qa["em"],3),
                        m.rate(a["evidence_complete_questions"]),f"{qa['prompt_tokens']['mean']:,.1f}"))
        compression.append((label,m.rate(view["append|tight"]["evidence_loss"]),m.rate(view["reselect|tight"]["evidence_loss"]),
            m.rate(a["evidence_loss"]), f"{m.pct(a['final_recall']['mean'])} (n={a['final_recall']['n']:,})",
            m.mean(a["final_context"]["segments"]),m.mean(a["final_context"]["nongold"]),
            m.mean(a["cost"]["plain"],1),m.mean(a["cost"]["cached"],1)))
        for cell in ("pooled",*m.CELLS):
            z = view[cell]["heldout_reward_v3"]
            reward.append((label,cell,f"{z['n_steps']:,}",*(f"{z[k]['mean']:.6f}" for k in (
                          "reward","n_lost","n_nongold_after","used_per_1000")),m.rate(z["invalid"])))
    for name, obj in r["task16_reference_rows"].items():
        q = obj["pooled"]
        complete = q["by_evidence_complete"]["true"]["n"]
        answers.append((m.LABEL.get(name,name),"任务16",f"{q['n']:,}",m.pct(q["f1"],3),m.pct(q["em"],3),
                        f"{complete:,}/{q['n']:,}",f"{q['prompt_tokens']['mean']:,.1f}"))
        n = sum(v["cost"]["n"] for v in obj["cells"].values())
        avg = sum(v["cost"]["cached_mean"] for v in obj["cells"].values())/len(obj["cells"])
        refs.append((m.LABEL.get(name,name),f"{avg:,.1f}",f"{n:,}"))
    decomp = r["decomposition"]
    transitions = {"false_to_false":"不齐→不齐","false_to_true":"不齐→齐全","true_to_false":"齐全→不齐","true_to_true":"齐全→齐全"}
    elapsed = e["runtime"]["wall_seconds"]
    def delta(d):
        return "—" if d is None else f"{d:+.3f}"
    return "\n\n".join([
        "# 任务18 · GRPO 第四轮与 reward v3",
        f"5880 流水线于北京时间 {r['completion']['beijing_end']} 完成（n={r['completion']['n_stages']} 阶段）。"
        f"按留出集逐步 reward v3 选中 checkpoint-{sel['checkpoint_step']}（n={sel['n_candidates']} 个候选）。"
        f"最终 CPU 全量测试通过 {r['cpu_tests']['tests']:,}/{r['cpu_tests']['tests']:,} 项。[结果 JSON](grpo_r4_results.json)",
        "## 奖励和实验设置",
        "以合并后的 SFT v3 ckpt-3000 起步，基座权重及 KL 参考沿用任务11。"
        "本轮奖励为 `−n_lost − 0.16 × n_nongold_after − 0.5 × used_after/1000 − 1[invalid]`。"
        "无关段按动作后上下文与该任务全部 gold 比较，后续 gold 也在集合内；非法动作为 no-op，继续对现存片段计价。",
        "κ=0.235 来自任务16留出集的读者 F1 回归（n=41,832 行），按平均段长扣除 token 项后得到无关段额外系数 0.16。"
        "定价使用留出集估计的单个系数，最终结论以锁定测试集为准。",
        "## C · 重打分",
        f"复用任务11的 n={c['n_states']:,} 组、{c['n_samples']:,} 条采样输出。"
        f"逐条重算 v2 奖励与原记录不一致 {c['parity']['v2_reward_mismatches']:,}/{c['parity']['v2_reward_checks']:,}，"
        f"v3 − v2 与新增项不一致 {c['parity']['v3_delta_mismatches']:,}/{c['parity']['v3_delta_checks']:,}。"
        "无关段变化列按同一批执行后的动作计算，两种奖励下列值相同。各指标使用归档的组内变化判据。",
        m.table(("格","奖励","组数 n","zero-std","gold-varying","nongold-varying","token-only","有采样胜过 greedy"),c_rows),
        "## D · 训练数据",
        f"从 {d['n_informative']:,}/{c['n_states']:,} 个 informative 状态抽取 n={d['n']:,} 条。"
        f"与任务11 D 同 state_id {d['same_state_id_as_task11_D']['count']:,}/{d['same_state_id_as_task11_D']['n']:,} 条，"
        f"user 字节差异 {d['user_text_mismatches']}/{d['rows_checked']:,}，非训练来源 {d['heldout_rows']}/{d['rows_checked']:,}，"
        f"插件 greedy 奖励差异 {d['plugin_greedy_mismatches']}/{d['plugin_greedy_checked']:,}。",
        m.table(("格","样本 n","gold-varying","仅无关段变化","token-only"), d_rows),
        "## E · 训练",
        "沿用任务11的训练种子 20261123、新 LoRA r16、α32、dropout 0.05，lr 1.5e-5、β 0.04、每组 8 个生成、"
        "梯度累积 8，advantage 用奖励减组均值，scale_rewards 为 none。",
        f"冒烟 n={smoke['optimizer_steps']} 步，reward / advantage 不一致各 {smoke_pairs['mismatches']}/{smoke_pairs['n']}。"
        f"正式训练用时 {elapsed/3600:.3f} 小时（{elapsed:,.2f} 秒，n={e['optimizer_steps']} 优化器步）。"
        f"逐条复算共 n={train_pairs['n']:,} 条日志记录，含末尾重复的 {train_pairs['extra_logged_completions']} 条；"
        f"reward 异常 {train_pairs['mismatches']}/{train_pairs['n']:,}，advantage 异常 {train_pairs['advantage_mismatches']}/{train_pairs['n']:,}。",
        "曲线 CSV 保留每步的无关段、丢失、token、非法和 add-count 分项。[训练曲线 CSV](training_curve.csv) · [训练曲线 SVG](training_curve.svg)",
        "## F · 留出集压缩、选点与奖励分项",
        f"按合并 reward v3 每步均值降序选 checkpoint-{sel['checkpoint_step']}，平局先取丢证据轨迹数较少者，再取更早步数。"
        "开缓存和无缓存成本均用 `eval/cost.py` 折合输入 token，以轨迹为单位。"
        "r2 和 v3 的 reward v3 从已有动作离线计算。",
        m.table(("策略","append tight 丢证据","reselect tight 丢证据","合并丢证据","最终召回",
                 "平均段数","平均无关段数","无缓存成本","开缓存成本"),compression),
        m.table(("策略","格","步数 n","reward v3","平均丢 gold","平均动作后无关段","平均 used/1000","非法率"),reward),
        "## G · 读者结果",
        f"Qwen3-8B 生效值 `disable_cascade_attn=True`，n={g['n_predictions']:,} 条留出题次。"
        f"任务03小样原始输出差异 {g['smoke']['raw_output_mismatches']}/{g['smoke']['n']} 条。主比较使用 G 同批结果，选点取自 F。",
        m.table(("策略","读者来源","题次 n","F1","EM","按题证据齐全","平均 prompt token"),answers),
        m.table(("任务16引用策略","开缓存成本","轨迹 n"),refs),
        "### 同批配对比较", *m.paired_md(r["paired_comparisons"]),
        "### r4 − r2 的 F1 分解",
        f"证据齐全转移按题次（合并 n={decomp['n']:,}）分组，贡献等于组内差值总和除以合并 n。"
        "对两侧证据均齐全的题，再按任务末新增的无关段数分组。",
        m.table(("r2→r4","题次 n","组内 ΔF1（百分点）","对合并 ΔF1 的贡献（百分点）"),
                ((transitions[k],f"{v['n']:,}",delta(v["mean_f1_difference_pp"]),delta(v["contribution_f1_pp"]))
                 for k,v in decomp["evidence_complete_transitions_r2_to_r4"].items())),
        m.table(("r4 − r2 无关段数","题次 n","组内 ΔF1（百分点）","平均 prompt token 差"),
                ((k,f"{v['n']:,}",delta(v["mean_f1_difference_pp"]),delta(v["mean_prompt_token_difference"]))
                 for k,v in decomp["both_complete_by_added_nongold_count_r4_minus_r2"].items())),
        "### 读者一致性",
        m.table(("策略","相同输入原始输出不同 / 任务16 配对题次"),
                ((m.LABEL[k],f"{a['count']:,}/{a['n']:,}") for k,a in g["different_raw_vs_task16"].items())),
        "同次运行对完整提示 token 序列分组，分母为出现至少两次的组，分子为组内有不同原始输出的组。",
        m.table(("范围","不一致率"),((m.LABEL.get(k,k),m.rate(a,3) if a["n"] else "0/0") for k,a in
                [("合并",g["same_run_inconsistent"]["all"]),*g["same_run_inconsistent"]["strategies"].items()])),
    ])+"\n"


def main():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and Path.cwd()==PROJECT,"run from the project on CPU")
    audit = read(m.EXT/"close17_18_20261125/reader_checks.json")
    C,D,E,G = [read(m.T18/name) for name in ("sampling_probe.json","dataset_stats.json","training_checks.json","reader_f1_results.json")]
    smoke = read(m.W18/"smoke_train/smoke_checks.json")
    need(C["status"]==D["status"]=="complete" and E["status"]==smoke["status"]=="ok"
         and C["n_samples"]==179952 and C["parity"]["v2_reward_mismatches"]==C["parity"]["v3_delta_mismatches"]==0
         and D["n"]==2000 and D["user_text_mismatches"]==D["plugin_greedy_mismatches"]==D["heldout_rows"]==0,
         "C, D, smoke or E checks did not pass")
    need(all(a["reward_recheck"]["mismatches"]==a["reward_recheck"]["advantage_mismatches"]==0
             and not a["runtime"]["nonfinite"] and not a["runtime"]["oom"] for a in (E,smoke)), "training checks incomplete")
    for key in ("source_groups","source_informative","output","smoke_data"):
        m.checked(D[key])
    selected,f,historical = f_results()
    need(G["selected"]==f"grpo_r4:{selected['strategy']}"==m.R4_500,"selected checkpoint differs from G")
    stats = g_results(f,G)
    sig = m.inputs.sums(m.T16/"TASK16_SHA256SUMS")
    m.verify(m.C16,sig[str(m.C16.relative_to(PROJECT))])
    t16 = read(m.C16)["strategies"]
    rows, curve_ref = curve()
    result = {
        "status":"complete","task":18,"completion":audit["stage_completion"]["task18"],
        "C":C,"D":D,"E":E,"smoke":smoke,"selected_checkpoint":selected,"G":G,"strategies":stats,
        "historical_reward_v3":historical,"task16_reference":record(m.C16),
        "task16_reference_rows":{name:t16[name] for name in (m.R3,"h1:tau_star","bm25","oracle")},
        "paired_comparisons":audit["pairs18"],"decomposition":audit["decomposition18"],
        "reader_input_sha256":audit["reader18_input_hashes"],"training_curve_csv":curve_ref,
        "cpu_tests":read(m.T17/"full_close_tests.json"),
        "calibration":{"kappa_f1":.235,"n":41832,"source":"Task 18 §0, Task 16 held-out F1",
                       "non_gold_coefficient":.16,"final_claim_requires_locked_test":True},
    }
    files = {"grpo_r4_report.md":render(result).encode(),"training_curve.svg":plot(rows)}
    for name in [*files,"grpo_r4_results.json"]:
        need(not (m.T18/name).exists(),f"will not overwrite {name}")
    for name, raw in files.items():
        with (m.T18/name).open("xb") as fp:
            fp.write(raw)
    result["generated_artifacts"] = {name:record(m.T18/name) for name in files}
    write_new(m.T18/"grpo_r4_results.json",result)


if __name__=="__main__":
    main()
