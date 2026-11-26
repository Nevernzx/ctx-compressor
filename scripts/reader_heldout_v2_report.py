"""Publish the frozen, cascade-off reader comparison from 55,776 saved predictions."""
from collections import defaultdict
from hashlib import sha256
from itertools import chain
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import reader_diag_metrics as diag
from scripts import reader_heldout_report as previous
from src.reader_heldout_inputs import checksum, sums

WORK = previous.EXTERNAL / "reader_heldout_v2"
OUT = PROJECT / "eval/reader_heldout_v2"
NAMES = (diag.R3, *previous.STRATEGIES)
BASES = (diag.HIST, diag.TRY)
HISTORY = (
    ("task12", PROJECT/"eval/reader_heldout/TASK12_SHA256SUMS",
     "a0b093d2cd8d777995fb51dffd1ec569491a24bd3007a24e00a0138b28d2515e"),
    ("try2", PROJECT/"eval/grpo_r3/TASK13_SHA256SUMS",
     "45aebbf0daa4bee61037da433c0c7c7534bd2d48488a82d9984bea4d76694206"),
    ("task15_arm_a", PROJECT/"eval/reader_diag/TASK15_SHA256SUMS",
     "f04a85cd13f95c031801d6b1f59862f29d534816906692b619de8e9fb5135404"),
)
DISPLAY = {**dict(zip(previous.STRATEGIES, previous.DISPLAY)), diag.R3:"GRPO r3 · 250"}
require = previous.require


def signature_files():
    signed = {}
    global_sums = sums(PROJECT/"SHA256SUMS")
    for label, path, expected in HISTORY:
        require(checksum(path) == global_sums[str(path.relative_to(PROJECT))] == expected,
                f"historical manifest differs: {label}")
        signed[label] = sums(path)
    return signed


def signed_rows(path, manifest, hashes):
    yield from previous.input_rows(path, hashes)
    require(hashes[str(path)] == manifest[str(path)], f"signed byte stream differs: {path}")


def signed_json(path, manifest, hashes):
    key = str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path)
    content = path.read_bytes()
    hashes[str(path)] = sha256(content).hexdigest()
    require(hashes[str(path)] == manifest[key], f"signed JSON differs: {path}")
    return json.loads(content)


def key(row):
    return tuple(row[k] for k in previous.IDENTITY)


def index_inputs(rows):
    """Use the exact packed token sequence as the dictionary key; retain only integer group IDs per question."""
    token_index, token_ids = {}, {}
    by_stream = defaultdict(lambda: defaultdict(list))
    def checked_rows():
        for row in rows:
            k = key(row)
            token_ids[k] = token_index.setdefault(diag.signature(row), len(token_index))
            by_stream[row["task"]][row["strategy"] == diag.R3].append(k)
            yield row
    sources = previous.index_requests(checked_rows())
    expected_order = []
    for task in sorted(by_stream):
        blocks = by_stream[task]
        require(len(blocks[False]) == 200 and len(blocks[True]) == 24, f"wrong batch layout for task {task}")
        expected_order.extend(blocks[False] + blocks[True])
    return sources, token_ids, expected_order


def index_outputs(rows, expected_order):
    raw = {}
    def checked_rows():
        for row, expected in zip(rows, expected_order, strict=True):
            k = key(row)
            require(k == expected, "new prediction order differs from registered input order")
            raw[k] = row["raw_output"]
            yield row
    return checked_rows(), raw


def historical_pair(rows, sources, raw, expected, old_summary):
    """Pair every historical prediction to its new prediction and its signed prompt metadata."""
    seen = set()
    deltas = defaultdict(lambda: {"n":0,"raw_different":0,"f1":[],"em":[]})
    for row in rows:
        k = key(row)
        name = row["strategy"]
        require(k in raw and k not in seen and name in expected, "unmatched or duplicate historical key")
        ref, answer, aliases = sources[k]
        require(tuple(row[f] for f in previous.MATCH) == ref and
                row["answer"] == answer and row["answer_aliases"] == aliases, "historical prompt metadata differ")
        seen.add(k)
        item = deltas[name]
        item["n"] += 1
        item["raw_different"] += raw[k] != row["raw_output"]
        item["f1"].append(row["f1"])
        item["em"].append(row["em"])
    expected_keys = {k for k,m in sources.items() if m[0][0] in expected}
    require(seen == expected_keys and set(deltas)==set(expected), "incomplete historical coverage")
    for name, item in deltas.items():
        old = old_summary[name]["pooled"]
        require(old["n"] == item["n"], "historical sample size differs from signed summary")
        for metric in ("f1","em"):
            score = math.fsum(item[metric])/item["n"]
            require(math.isclose(score, old[metric], rel_tol=0, abs_tol=1e-12),
                    "historical prediction scores differ from signed summary")
            item[metric] = score
    return dict(deltas)


def calculate():
    signed = signature_files()
    hashes = {}
    pre = [signed_json(folder/"input_preflight/preflight.json", signed[label], hashes)
           for label,folder in zip(("task12","try2"),BASES)]
    require(all(p["status"] == "passed" and p["input_truncations"] == 0 for p in pre), "failed historical preflight")
    reads = [signed_rows(folder/"input_preflight/requests.jsonl", signed[label], hashes)
             for label,folder in zip(("task12","try2"),BASES)]
    sources, token_ids, order = index_inputs(chain.from_iterable(reads))
    require(len(sources)==len(order)==55776, "unexpected request count")
    for folder, pf in zip(BASES,pre):
        require(hashes[str(folder/"input_preflight/requests.jsonl")] == pf["requests"]["sha256"],
                "historical requests/preflight disagree")
    config = json.loads((OUT/"run_config.json").read_bytes())
    require(config["status"]=="complete" and config["effective_disable_cascade_attn"] is True and
            config["predictions"]==55776 and config["parameters"]["disable_cascade_attn"] is True,
            "missing completed cascade-off engine")
    smoke_count, smoke_n = 0, 0
    smoke_src = signed_rows(diag.HIST/"input_preflight/smoke.jsonl", signed["task12"], hashes)
    for row, prompt in zip(previous.input_rows(WORK/"smoke_predictions.jsonl", hashes), smoke_src, strict=True):
        require(key(row) == key(prompt) and row["prompt_sha1"] == prompt["prompt_sha1"] and
                row["expected_raw_output"] == prompt["expected_raw_output"], "smoke reference differs")
        smoke_count += row["raw_output"] != prompt["expected_raw_output"]
        smoke_n += 1
    smoke = dict(n=smoke_n, raw_output_mismatches=smoke_count)
    require(smoke_n == 40 and smoke == {k:config["smoke"][k] for k in smoke}, "smoke count differs")
    checked, raw = index_outputs(previous.input_rows(WORK/"predictions.jsonl", hashes), order)
    data = previous.collect(checked,sources,NAMES)
    require(config["predictions_sha256"] == hashes[str(WORK/"predictions.jsonl")], "unsigned new predictions")
    hashes[str(OUT/"run_config.json")] = checksum(OUT/"run_config.json")
    groups = defaultdict(list)
    for k, (metadata,answer,aliases) in sources.items():
        groups[metadata[0]].append((token_ids[k],raw[k]))
    within = {"all":diag.inconsistent(chain.from_iterable(groups.values())),
              "by_strategy":{name:diag.inconsistent(groups[name]) for name in NAMES}}
    costs = {**pre[0], "costs":{**pre[0]["costs"],**pre[1]["costs"]}}
    result = previous.make_results(data,costs)
    extra = previous.make_results(data,costs,extra=diag.R3)
    result["strategies"] = {**extra["strategies"],**result["strategies"]}
    result["comparisons"] += extra["comparisons"]
    result["n_predictions"] += extra["n_predictions"]
    result["smoke"], result["within_run"] = smoke, within
    require(result["n_predictions"]==55776 and all(v["n"]==5976 for v in result["comparisons"]),
            "unexpected analysis population")
    t12 = signed_json(PROJECT/"eval/reader_heldout/reader_heldout_results.json",signed["task12"],hashes)
    t13 = signed_json(PROJECT/"eval/grpo_r3/reader_f1_results.json",signed["try2"],hashes)
    t15 = signed_json(PROJECT/"eval/reader_diag/reader_diag_results.json",signed["task15_arm_a"],hashes)
    history = (
        ("task12", diag.HIST, previous.STRATEGIES, t12["strategies"]),
        ("try2", diag.TRY,(diag.R3,),t13["strategies"]),
        ("task15_arm_a",diag.X/diag.A,(diag.R3,),t15["arms"][diag.A]["strategies"]),
    )
    comparisons = {}
    for label, folder, strategies, old in history:
        extracted = historical_pair(signed_rows(folder/"predictions.jsonl",signed[label],hashes),
                                    sources,raw,strategies,old)
        for name,v in extracted.items():
            new = result["strategies"][name]["pooled"]
            require(new["n"]==v["n"],"historical pairing n differs")
            comparisons.setdefault(name,{})[label] = {
                "n":v["n"],"old_f1":v["f1"],"new_f1":new["f1"],
                "f1_difference_pp":100*(new["f1"]-v["f1"]),"raw_different":v["raw_different"],
            }
    result["historical_comparisons"] = comparisons
    result["input_sha256"] = hashes
    result["plots"] = {"budget":"budget_f1.svg","cost":"cost_f1.svg"}
    return result


def render_markdown(result):
    from scripts.reader_report import _table
    content = previous.render_markdown(result)
    content = content.replace("# MuSiQue 留出集阅读模型评测","# 任务16 · 关闭 cascade attention 的留出集读者评测",1)
    content = content.replace("Qwen3-8B 根据", "引擎确认 disable_cascade_attn=True。Qwen3-8B 根据",1)
    content = content.replace("](reader_heldout_results.json)","](reader_heldout_v2_results.json)")
    labels = {"task12":"任务12","try2":"任务13 try2","task15_arm_a":"任务15 臂 a"}
    rows = []
    for name in NAMES:
        for tag,p in result["historical_comparisons"][name].items():
            rows.append([name,labels[tag],p["n"],f"{100*p['old_f1']:.3f}%",f"{100*p['new_f1']:.3f}%",
                         f"{p['f1_difference_pp']:+.3f}",p["raw_different"]])
    rates = result["within_run"]
    grouped = [[name,r["count"],r["n"],f"{100*r['rate']:.3f}%" if r["n"] else "—"]
               for name,r in [("合并",rates["all"]),*rates["by_strategy"].items()]]
    return content+"\n## 新旧读者逐题对照\n\n"+_table(["策略","旧运行","配对题次 n","旧 F1","本次 F1","本次减旧值（百分点）","原始输出不同（条）"],rows)+(
        "\n\n## 同次运行的一致性\n\n按完整输入 token 序列分组，分母为出现至少两次的组。分子为组内出现多种原始输出的组，每组计一次。\n\n"
        )+_table(["范围","原始输出不一致组数","重复组数 n","不一致率"],grouped)+"\n"


def figures(result):
    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    colors = ("#173f88","#c75713","#6e4897","#508168","#b44470","#678812",
              "#396978","#868686","#141414","#876344","#525a68")
    markers = ("o","s","D","^","v","P","X","p","*","h","+")
    refs = {"no_compress","no_context"}
    charts = {}
    with mpl.rc_context({"font.family":"DejaVu Sans","font.size":9,"svg.fonttype":"none",
                         "svg.hashsalt":"reader16","axes.spines.top":False,"axes.spines.right":False}):
        for axis in ("budget","cost"):
            fig, axes = plt.subplots(1,2,figsize=(11.4,6.0),sharey=True)
            fig.subplots_adjust(left=.085,right=.978,bottom=.29,top=.87,wspace=.13)
            labels = []
            all_cells = [g for s in result["strategies"].values() for g in s["cells"].values()]
            upper = min(100, 5*math.ceil((max(100*g["f1"] for g in all_cells)+3)/5))
            for ax, mode in zip(axes,previous.MODES):
                ax.set_title(mode.capitalize(),loc="left",fontweight="bold")
                ax.grid(axis="y",color="#d9dcde",lw=.6);ax.set_axisbelow(True);ax.set_ylim(0,max(10,upper))
                if axis=="budget":
                    ax.set_xticks(list(previous.BUDGETS.values()),[f"{v:,}" for v in previous.BUDGETS.values()])
                    ax.set_xlim(1500,6000);ax.set_xlabel("Passage budget (tokens)")
                else:
                    bound=1.13*max(g["cost"]["cached_mean"] for g in all_cells)
                    ax.set_xscale("symlog",linthresh=1500,linscale=.35,base=10)
                    ax.set_xlim(-90,bound)
                    ax.set_xticks([v for v in (0,1500,3000,10000,30000,100000) if v<=bound])
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:,.0f}"))
                    ax.set_xlabel("Cached cost (equivalent input tokens / trajectory)")
                for name,color,marker in zip(NAMES,colors,markers,strict=True):
                    cells=result["strategies"][name]["cells"]
                    reference=name in refs
                    line = ("--" if name=="no_compress" else ":") if reference else "-"
                    if axis=="budget" and reference:
                        ax.axhline(100*cells["shared"]["f1"],color=color,linestyle=line,lw=1.2)
                    else:
                        if reference:
                            g=cells["shared"]
                            by_cell=g["cost"].get("by_cell",{})
                            xs=[by_cell.get(f"{mode}|{tier}",g["cost"])["cached_mean"] for tier in previous.BUDGETS]
                            ys=[100*g["f1"]]*len(xs)
                        else:
                            ordered=[cells[f"{mode}|{tier}"] for tier in previous.BUDGETS]
                            xs=list(previous.BUDGETS.values()) if axis=="budget" else [g["cost"]["cached_mean"] for g in ordered]
                            ys=[100*g["f1"] for g in ordered]
                        ax.plot(xs,ys,linestyle="-" if axis=="budget" else "none",marker=marker,color=color,
                                markersize=5.6 if name==diag.R3 else 4.7,lw=1.9 if name==diag.R3 else 1.25,
                                zorder=5 if name==diag.R3 else 3)
                    if ax==axes[0]:
                        labels.append(Line2D([],[],color=color,label=DISPLAY[name],marker="" if reference and axis=="budget" else marker,
                                             linestyle=line if axis=="budget" else "none"))
            axes[0].set_ylabel("Answer F1 (%)")
            fig.suptitle("Reader F1 by passage budget" if axis=="budget" else "Reader F1 by cached trajectory cost",
                         x=.085,y=.978,ha="left",fontweight="bold",fontsize=14)
            fig.legend(handles=labels,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(.50,.035),fontsize=8.5)
            fig.text(.085,.008,"Each cell: n = 996 question instances from 249 tasks. Shared controls: n = 996 each.",fontsize=8)
            buffer=io.BytesIO();fig.savefig(buffer,format="svg",metadata={"Date":None})
            charts[f"{axis}_f1.svg"]=buffer.getvalue();plt.close(fig)
    return charts


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and Path.cwd()==PROJECT,"run on project CPU")
    result=calculate()
    payloads={"reader_heldout_v2_report.md":render_markdown(result).encode(),
              "reader_heldout_v2_results.json":previous.encoded(result)}
    # The project training environment contains Matplotlib; keep all on-disk caches with task16.
    import base64
    env=dict(os.environ,MPLCONFIGDIR=str(WORK/"plot_cache"),MPLBACKEND="Agg",CUDA_VISIBLE_DEVICES="",
             PYTHONDONTWRITEBYTECODE="1")
    plot=subprocess.run([str(previous.EXTERNAL/"envs/train/bin/python"),"-B",str(Path(__file__).resolve()),"--plot"],
                        input=previous.encoded(result),capture_output=True,check=True,env=env)
    payloads.update({name:base64.b64decode(data) for name,data in json.loads(plot.stdout).items()})
    previous.publish(OUT,payloads)
    print(json.dumps({"n":result["n_predictions"],"comparisons":result["comparisons"],
                      "within":result["within_run"]["all"]},ensure_ascii=False))


if __name__=="__main__":
    if sys.argv[1:]==["--plot"]:
        import base64
        print(json.dumps({k:base64.b64encode(v).decode() for k,v in figures(json.load(sys.stdin)).items()}))
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("unexpected arguments")

