"""Publish the Task 17 report, statistical tables and updated Task 16 plots."""
import csv
import io
from math import ceil, fsum, isclose
import os
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import close_reranker_grpo_metrics as m
from scripts import reader_heldout_v2_report as reader16
from src.reader_heldout_inputs import rows, write_new
from src.reranker_v1 import need, read, record

RR = ("RR-fill", "RR-τ")


def encode_csv(entries):
    s = io.StringIO(newline="")
    writer = csv.DictWriter(s, fieldnames=list(entries[0]))
    writer.writeheader()
    writer.writerows(entries)
    return s.getvalue().encode()


def heldout_tables(comparison):
    compression, answer = [], []
    for name, by in comparison["by_strategy"].items():
        for cell, block in [("合并", by["pooled"]), *by["cells"].items()]:
            c, q, ctx = (block[k] for k in ("compression", "reader", "final_context"))
            n = ctx["n_trajectories"]
            ev = c["evidence_complete_questions"]
            need(q["n"] == ev["n"] == 4*n and q["by_evidence_complete"]["true"]["n"] == ev["count"], "reader evidence mismatch")
            label = m.LABEL.get(name, name)
            compression.append((label, cell, m.rate(c["evidence_loss"]),
                f"{m.pct(c['final_recall']['mean'])} (n={n:,})",
                f"{m.pct(c['step_recall']['micro']['mean'])} (n={c['step_recall']['micro']['n']:,})",
                f"{ctx['mean_keep']:.3f} / {ctx['mean_nongold']:.3f} (n={n:,})",
                m.mean(c["cost"]["plain"], 1), m.mean(c["cost"]["cached"], 1)))
            answer.append((label, cell, f"{q['n']:,}", m.pct(q["f1"], 3), m.pct(q["em"], 3),
                           m.rate(ev), f"{q['prompt_tokens']['mean']:,.1f}"))
    refs = []
    for name, by in comparison["task16_reference_rows"].items():
        q = by["pooled"]
        costs = [g["cost"] for g in by["cells"].values()]
        complete = q["by_evidence_complete"]["true"]["n"]
        refs.append((m.LABEL.get(name, name), f"{q['n']:,}", m.pct(q["f1"], 3), m.pct(q["em"], 3),
                     f"{complete:,}/{q['n']:,}", f"{q['prompt_tokens']['mean']:,.1f}",
                     f"{fsum(g['cached_mean'] for g in costs)/len(costs):,.1f} (n={sum(g['n'] for g in costs):,})", "任务16"))
    return [
        m.table(("策略", "格", "丢证据", "最终召回", "单步召回", "平均段数 / 无关段数", "无缓存", "开缓存"), compression),
        m.table(("策略", "格", "题次 n", "F1", "EM", "按题证据齐全", "平均 prompt token"), answer),
        "### 任务16引用行",
        m.table(("策略", "题次 n", "F1", "EM", "按题证据齐全", "平均 prompt token", "开缓存成本", "来源"), refs),
    ]


def hop_tables(hops):
    keep, first, csv_rows = [], [], []
    for name, scopes in hops["by_strategy"].items():
        for scope in ("append", "reselect", "pooled"):
            a = scopes[scope]
            label = m.LABEL.get(name, name)
            keep.append((label, scope, *(m.rate(a["retained"][h]) for h in ("1","2","3","4","2+"))))
            first.append((label, scope, *(m.rate(a["first_loss"][h]) for h in ("1","2","3","4","none"))))
            need(sum(x["count"] for x in a["first_loss"].values()) == a["first_loss"]["none"]["n"], "first loss partition mismatch")
            csv_rows.extend({"strategy": name, "scope": scope, "metric": metric, "hop": h, **value}
                            for metric in ("retained", "first_loss") for h,value in a[metric].items())
    return [
        m.table(("策略", "范围", "第1跳", "第2跳", "第3跳", "第4跳", "第≥2跳"), keep),
        m.table(("策略", "范围", "首失第1跳", "首失第2跳", "首失第3跳", "首失第4跳", "全部保留"), first),
        m.table(("范围", "跳位", "r2 保留率", "RR-τ 保留率", "r2 − RR-τ（百分点）"),
                ((scope,h,m.rate(a["left"]),m.rate(a["right"]),f"{a['delta_pp']:+.3f}")
                 for scope, by in hops["r2_minus_rr_tau"].items() for h,a in by.items())),
    ], encode_csv(csv_rows)


def latency_tables(latencies):
    wall, token, gpu, csv_rows = [], [], [], []
    for mode, a in latencies.items():
        if a["status"] == "load_failed":
            wall.append((mode, "加载失败", "—", "—", "—", "—", a["reason"]))
            csv_rows.append({"engine":mode,"unit":"state","statistic":"load_failed","value":a["reason"],"n":0})
            continue
        timings = list(rows(m.checked(a["per_state"])))
        s, b, n = a["sequential_seconds"], a["submitted_together"], a["n_states"]
        need(len(timings) == n == s["n"] == b["n_states"] == 300 and a["warmup_n"] == 20
             and isclose(b["steps_per_second"], n/b["wall_seconds"], abs_tol=1e-10)
             and isclose(s["mean"], fsum(z["wall_seconds"] for z in timings)/n, abs_tol=1e-10), f"latency receipt: {mode}")
        wall.append((mode,n,*(f"{s[k]*1000:.2f}" for k in ("p50","p90","mean")),
                     f"{b['steps_per_second']:.3f}",f"{b['wall_seconds']:.3f}"))
        csv_rows.extend({"engine":mode,"unit":"ms","statistic":k,"value":s[k]*1000,"n":n} for k in ("p50","p90","mean"))
        csv_rows.append({"engine":mode,"unit":"steps/s","statistic":"submitted_together","value":b["steps_per_second"],"n":n})
        for k in ("prompt_tokens","output_tokens","pair_prompt_tokens"):
            if k in a:
                dist = a[k]
                metrics = ("min","p50","p90","mean","max")
                token.append((mode,k,f"{dist['n']:,}",*(f"{dist[h]:,.1f}" for h in metrics)))
                csv_rows.extend({"engine":mode,"unit":"tokens","statistic":f"{k}.{h}","value":dist[h],"n":dist["n"]}
                                for h in metrics)
        samples = list(rows(m.checked(a["gpu0_background"])))
        need(samples and all(z["index"] == 0 for z in samples), "latency background telemetry")
        gpu.append((mode,len(samples),
                    f"{min(z['memory_used_mib'] for z in samples):,}–{max(z['memory_used_mib'] for z in samples):,}",
                    f"{min(z['utilization_percent'] for z in samples)}–{max(z['utilization_percent'] for z in samples)}"))
    return [
        m.table(("设置", "状态 n", "p50 毫秒", "p90 毫秒", "均值毫秒", "整批步/秒", "整批秒"), wall),
        m.table(("设置", "token 口径", "样本 n", "最小", "p50", "p90", "均值", "最大"), token),
        m.table(("设置", "GPU 0 采样 n", "显存 MiB 范围", "利用率 % 范围"), gpu),
    ], encode_csv(csv_rows)


def figures(t16, comparison):
    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    data = {name: s["cells"] for name,s in t16["strategies"].items()}
    for name in RR:
        data[name] = {cell: {"f1": b["reader"]["f1"],
                            "cost":{"cached_mean":b["compression"]["cost"]["cached"]["mean"]}}
                      for cell,b in comparison["by_strategy"][name]["cells"].items()}
    names = (*reader16.NAMES,*RR)
    colors = ("#173f88","#c75713","#6e4897","#508168","#b44470","#678812",
              "#396978","#868686","#141414","#876344","#525a68","#008491","#c22c4e")
    markers = ("o","s","D","^","v","P","X","p","*","h","+","<",">")
    labels = {**reader16.DISPLAY, **{name:name for name in RR}}
    values = [b for by in data.values() for b in by.values()]
    upper = min(100,5*ceil((max(100*z["f1"] for z in values)+3)/5))
    bounds = 1.13*max(z["cost"]["cached_mean"] for z in values)
    charts = {}
    with mpl.rc_context({"font.family":"DejaVu Sans","font.size":9,"svg.fonttype":"none","svg.hashsalt":"reader17",
                         "axes.spines.top":False,"axes.spines.right":False}):
        for axis in ("budget","cost"):
            fig, axes = plt.subplots(1,2,figsize=(11.4,6.25),sharey=True)
            fig.subplots_adjust(left=.085,right=.978,bottom=.30,top=.87,wspace=.13)
            legend = []
            for ax,mode in zip(axes,m.qa.MODES,strict=True):
                ax.set_title(mode.capitalize(),loc="left",fontweight="bold")
                ax.grid(axis="y",color="#d9dcde",lw=.6)
                ax.set_axisbelow(True)
                ax.set_ylim(0,upper)
                budgets = m.qa.BUDGETS
                if axis == "budget":
                    ax.set_xticks(list(budgets.values()),[f"{v:,}" for v in budgets.values()])
                    ax.set_xlim(1500,6000)
                    ax.set_xlabel("Passage budget (tokens)")
                else:
                    ax.set_xscale("symlog",linthresh=1500,linscale=.35,base=10)
                    ax.set_xlim(-90,bounds)
                    ax.set_xticks([k for k in (0,1500,3000,10000,30000,100000) if k<=bounds])
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:,.0f}"))
                    ax.set_xlabel("Cached cost (equivalent input tokens / trajectory)")
                for i,(name,color,marker) in enumerate(zip(names,colors,markers,strict=True)):
                    by = data[name]
                    line = "--" if name=="no_compress" else ":" if name=="no_context" else "-"
                    marker_arg = marker
                    if "shared" in by and axis=="budget":
                        artist = ax.axhline(100*by["shared"]["f1"],color=color,linestyle=line,lw=1.2)
                        marker_arg = ""
                    else:
                        if "shared" in by:
                            b = by["shared"]
                            c = b["cost"]
                            x = [c.get("by_cell",{}).get(f"{mode}|{tier}",c)["cached_mean"] for tier in budgets]
                            y = [100*b["f1"]]*len(budgets)
                        else:
                            blocks = [by[f"{mode}|{tier}"] for tier in budgets]
                            x = list(budgets.values()) if axis=="budget" else [b["cost"]["cached_mean"] for b in blocks]
                            y = [100*b["f1"] for b in blocks]
                        artist, = ax.plot(x,y,color=color,linestyle=line if axis=="budget" else "none",marker=marker,
                                          markersize=6.4 if name in RR else 4.7,lw=2 if name in RR else 1.15)
                    artist.set_gid(f"{mode}-{i}-curve")
                    if ax==axes[0]:
                        legend.append(Line2D([],[],label=labels[name],color=color,marker=marker_arg,
                                             linestyle=line if axis=="budget" else "none"))
            axes[0].set_ylabel("Answer F1 (%)")
            title = "Reader F1 by passage budget" if axis=="budget" else "Reader F1 by cached trajectory cost"
            fig.suptitle(title,x=.085,y=.978,ha="left",fontweight="bold",fontsize=14)
            fig.legend(handles=legend,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(.50,.038),fontsize=8.5)
            fig.text(.085,.008,"RR lines: Task 17. Other lines: Task 16. Each cell: n = 996 question instances.",fontsize=8)
            raw = io.BytesIO()
            fig.savefig(raw,format="svg",metadata={"Date":None})
            plt.close(fig)
            root = ET.fromstring(raw.getvalue())
            ids = {x.attrib.get("id") for x in root.iter()}
            need(all(f"{mode}-{i}-curve" in ids for mode in m.qa.MODES for i in range(len(names))), "missing plot lines")
            charts[f"{axis}_f1_with_reranker.svg"] = raw.getvalue()
    return charts


def posthoc_md(a):
    labels = {"all_seen":"该题全部 gold 当过", "any_unseen":"该题至少一段 gold 未当过"}
    return [
        "## 事后分析（Claude 提出，非预注册）",
        f"从 `split.json` 的 `train_tasks` 取 n={a['n_train_tasks']:,} 个重排训练任务，"
        f"汇总这些任务全部步里当过 gold 的 {a['gold_training_pid_count']:,} 个不同语料 PID。"
        f"留出集 n={a['n_heldout_tasks']:,} 个任务。按 PID 归组，以 `keep_after` 判定任务末保留。"
        "同一任务内 gold PID 互不重复，六格分别计数。",
        m.table(("策略","训练中当过 gold","训练中未当过 gold"),
                ((m.LABEL.get(k,k),m.rate(v["seen"],3),m.rate(v["unseen"],3)) for k,v in a["kept_gold"].items())),
        "题的所有 gold PID 都在上述集合则归入全部当过组。配对键为模式、预算格、任务、题位，另核对原题 ID 与跳数。"
        "r2、v3 和 RR-τ 用任务17同批预测，r4-500 用任务18 G 预测与任务17 RR-τ 跨批配对。",
        m.table(("题组","原题 n","六格配对题次 n","比较","ΔF1（百分点）","ΔEM（百分点）","F1 高 / 平 / 低"),
                ((labels[group], a["question_strata"][group]["unique_questions"], f"{p['n']:,}",
                  f"{m.LABEL[name]} − RR-τ",f"{p['f1_difference_pp']:+.3f}",f"{p['em_difference_pp']:+.3f}",
                  " / ".join(f"{p['f1_counts'][k]:,}" for k in ("left_higher","equal","right_higher")))
                 for group, by in a["paired"].items() for name,p in by.items())),
    ]


def render(r, hop_md, lat_md):
    tr, split, val, comp = (r[k] for k in ("training","split","tau_selection","heldout"))
    f1, f2 = r["completion"]["failure_records"]
    probe = tr["throughput_probe"]
    n = tr["n_train_pairs"]
    phases = [(k,t) for k,t in split["partitions"].items()]
    model_rows = [("原模型",tr["zero_shot_validation"]),
                  *((f"checkpoint-{v['step']}",v) for v in tr["checkpoints"])]
    consistency = r["consistency"]
    nrr = r["latency"]["reranker"]
    return "\n\n".join([
        "# 任务17 · 同数据重排基线与单步延迟",
        f"5880 流水线于北京时间 {r['completion']['beijing_end']} 完成（n={r['completion']['n_stages']} 阶段）。"
        f"阈值 τ={val['tau']:g}，由 n={val['n_tasks']} 个训练验证任务选出，覆盖 {val['n_predictions']:,} 条预测。"
        f"全量 CPU 测试通过 {r['cpu_tests']['tests']:,}/{r['cpu_tests']['tests']:,} 项。[运行配置](run_config.json) · [结果 JSON](reranker_v1_results.json)",
        "## 数据和训练",
        f"Qwen3-Reranker-0.6B revision `{r['model']['revision']}`，许可证 `{r['model']['license']}`。"
        "训练、train-val 和留出题交集 0。以四道问题组成 query，使用压缩器 prompt 的片段正文。"
        "在 n=3 个任务抽查的 540 段出现位置中，正文有 0 处差异。用末位 yes/no 的 logit 差，取 sigmoid 得分。"
        "模板原文见运行配置。[模型清单](model_card.json)",
        m.table(("划分","每任务数量","任务数 n","最小","p50","p90","最大","均值"),
                ((name,title,f"{d['n']:,}",d["min"],f"{d['p50']:.1f}",f"{d['p90']:.1f}",d["max"],f"{d['mean']:.3f}")
                 for name,t in phases for k,title in (("candidates","候选"),("positives","正例"))
                 for d in (t[f"{k}_per_task"],))),
        m.table(("编码划分","样本对 n","截断对数","截断率"),
                ((k,f"{d['pairs']:,}",f"{d['truncated']:,}",m.pct(d['truncated']/d['pairs'],3)) for k,d in r["tokenization"]["partitions"].items())),
        f"全参 bf16、AdamW、lr 1e-5、cosine、warmup 3%、每步 64 对，逐对 BCE 训练一轮。"
        f"吞吐冒烟 n={probe['measured_updates']} 步估时 {probe['estimated_fit_seconds'][str(tr['negative_ratio'])]/3600:.3f} 小时，"
        f"沿用每正例最多 {tr['negative_ratio']} 个同任务负例。训练 n={n:,} 对，截断 {tr['n_truncated_train_pairs']:,}/{n:,}。"
        f"训练含验证用时 {tr['fit_wall_seconds_including_checkpoints']/3600:.3f} 小时（n={tr['n_optimizer_steps']:,} 步）。"
        f"按 train-val 任务平均 AP 选择 checkpoint-{tr['chosen']['step']}。",
        m.table(("权重","验证任务 n","验证样本对 n","任务平均 AP"),
                ((k,a["n_tasks"],f"{a['n_pairs']:,}",m.pct(a["mAP"],4)) for k,a in model_rows)),
        m.table(("分数缓存","SHA256"),((k,a["sha256"]) for k,a in tr["score_cache"].items())),
        "## val100 选阈值",
        "τ=0 对应装满策略。按六格合并的读者 F1 取最高点，差小于 1e-9 时取较小阈值。",
        m.table(("τ","题次 n","F1","EM","丢证据率","平均段数 / 无关段数"),
                ((d["tau"],f"{d['n']:,}",m.pct(d["f1"],4),m.pct(d["em"],4),m.rate(d["evidence_loss"]),
                  f"{d['mean_keep']:.3f} / {d['mean_nongold']:.3f} (n={d['n_trajectories']:,})") for d in val["grid"].values())),
        "## 留出集预注册结果",
        f"Qwen3-8B 启用 `disable_cascade_attn=True`。同批读取 n={comp['n_predictions']:,} 题次，"
        f"任务03小样原始输出不同 {comp['reader_smoke']['raw_output_mismatches']}/{comp['reader_smoke']['n']} 条。"
        "丢证据按任务末缺至少一段 gold 的轨迹计，最终召回按轨迹平均，单步召回按步合并。成本以 `eval/cost.py` 折合输入 token 计。",
        *heldout_tables(comp),
        "### 同批配对比较", *m.paired_md(r["paired_comparisons"]),
        "### 按跳保留与首次丢失",
        "保留率分母为 gold 出现次数，首次丢失分母为题次。首次丢失指任务末缺失 gold 的最早跳位。[按跳表 CSV](hop_retention.csv)",
        *hop_md,
        "### 读者一致性",
        m.table(("策略","相同输入原始输出不同 / 任务16配对题次"),
                ((m.LABEL.get(k,k),f"{v['count']:,}/{v['n']:,}") for k,v in consistency["historical_task16_raw_diff"].items())),
        "同次运行对完整提示 token 序列分组。分母为至少出现两次的组，分子为组内有不同原始输出的组。",
        m.table(("范围","不一致率"),
                ((m.LABEL.get(k,k),m.rate(v,3) if v["n"] else "0/0")
                 for k,v in [("合并",consistency["within_all"]),*consistency["within_by_strategy"].items()])),
        "## 单步延迟",
        "种子 20261129，从 r2 轨迹每格取 n=50 个状态，共 n=300。每种设置先顺序热身 n=20，再逐条计时。"
        "吞吐按整批提交 n=300 算，压缩器最大并发 16，重排把每步全部候选作为一个 batch。GPU 0 采样仅供同机背景参考。",
        *lat_md,
        f"重排吞吐 {nrr['candidates_per_second']:,.1f} 对/秒（n={nrr['pair_prompt_tokens']['n']:,} 对），"
        f"截断 {nrr['n_truncated_pairs']:,}/{nrr['pair_prompt_tokens']['n']:,} 对。[延迟表 CSV](latency_table.csv)",
        "## 续跑记录",
        f"- {f1['beijing']}（北京时间，PID {f1['controller_pid']}），`prepare-heldout` 返回 `KeyError: 'task'`。"
        "加载器把 task ID 放在映射键中，回放器要求每条记录有 `task` 字段。修复在 held-out 准备入口把键复制进记录，并用小夹具回归。"
        "校验复用前四阶段，沿用 τ=0.5。[第一次续跑回执](prepare_heldout_resume.json)\n"
        f"- {f2['beijing']}（北京时间，PID {f2['controller_pid']}），`latency-eager` 初始化 vLLM 时 Triton 编译 LoRA kernel 缺少 C 编译器。"
        "压缩器延迟沿用任务11和18 F 阶段使用的 `envs/reader/bin/python`，加载签名的 `grpo_r2_environment.sh` 获取 CC/CXX，"
        "将缓存放到任务17外盘路径，并在 GPU 闸门前校验 CC。校验复用六个完成阶段。"
        "[第二次续跑回执](latency_resume.json) · [CPU 编译验证](latency_compiler_cpu_preflight.json)",
        "## 图",
        "在任务16原数据上增加 RR-fill 和 RR-τ。[预算–F1](budget_f1_with_reranker.svg) · [成本–F1](cost_f1_with_reranker.svg)",
        *posthoc_md(r["posthoc"]),
    ]) + "\n"


def main():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and Path.cwd() == PROJECT, "run from the project on CPU")
    audit = read(m.EXT / "close17_18_20261125/reader_checks.json")
    comp = read(m.T17 / "heldout_comparison.json")
    ref16 = m.checked(comp["task16_reference"])
    need(ref16==m.C16, "unexpected Task 16 reference")
    sig = m.inputs.sums(m.T16/"TASK16_SHA256SUMS")
    m.verify(ref16,sig[str(ref16.relative_to(PROJECT))])
    source = read(ref16)
    need(all(v==source["strategies"][k] for k,v in comp["task16_reference_rows"].items()),"Task 16 reference rows differ")
    hop = read(m.T17/"hop_retention.json")
    lat = {k:read(m.T17/f"latency_{k}.json") for k in ("eager","graph","reranker")}
    result = {
        "task":17,"status":"complete","completion":audit["stage_completion"]["task17"],
        "model":read(m.T17/"model_card.json"),"split":read(m.T17/"data_preflight.json"),
        "training":read(m.T17/"training.json"),"tokenization":read(m.T17/"tokenization.json"),
        "tau_selection":read(m.T17/"tau_selection.json"),"heldout":comp,"paired_comparisons":audit["pairs17"],
        "consistency":comp["consistency"],"hop_retention":hop,"latency":lat,"posthoc":audit["posthoc"],
        "reader_input_sha256":audit["reader17_input_hashes"]|audit["reader18_input_hashes"],
        "cpu_tests":read(m.T17/"full_close_tests.json"),
    }
    need(result["tau_selection"]["tau"]==.5 and result["cpu_tests"]["status"]=="passed","tau or CPU tests missing")
    hops, hop_csv = hop_tables(hop)
    latency, lat_csv = latency_tables(lat)
    items = {"reranker_v1_report.md":render(result,hops,latency).encode(),
             "hop_retention.csv":hop_csv,"latency_table.csv":lat_csv,**figures(source,comp)}
    for name in [*items,"reranker_v1_results.json"]:
        need(not (m.T17/name).exists(),f"will not overwrite {name}")
    for name,data in items.items():
        with (m.T17/name).open("xb") as f:
            f.write(data)
    result["generated_artifacts"] = {name:record(m.T17/name) for name in items}
    write_new(m.T17/"reranker_v1_results.json",result)


if __name__ == "__main__":
    main()
