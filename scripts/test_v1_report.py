"""Write the registered Task 19 report, hop table, and budget/cost plots."""
import csv
import io
from math import ceil
import os
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import close_reranker_grpo_metrics as style
from scripts import test_v1_data as data, test_v1_metrics as metrics
from src.reader_heldout_inputs import write_new
from src.reranker_v1 import need, read, record, require_new

W,OUT=data.WORK,data.SMALL
CSV="hop_retention.csv"
NAMES=data.NAMES
CELLS=metrics.CELLS


def f(x,digits=3):
    return f"{x:.{digits}f}" if x is not None else "—"


def signed_num(x):
    return f"{x:+.3f}" if x is not None else "—"


def rate(a):
    if a["n"]==0:
        return "— (n=0)"
    return style.rate(a,3)


def pct_avg(a):
    return f"{style.pct(a['mean'],3)} (n={a['n']:,})"


def answers(r):
    result=[]
    for name,a in r["reader"].items():
        groups=[("合并",a["pooled"]),*a["by_mode"].items(),*a["by_tier"].items(),*a["cells"].items()]
        for group,q in groups:
            complete=q["by_evidence_complete"]["true"]["n"]
            result.append((name,group,f"{q['n']:,}",style.pct(q["f1"],3),style.pct(q["em"],3),
                           rate(metrics.rate(complete,q["n"])),f"{q['prompt_tokens']['mean']:,.1f}"))
    return style.table(("策略","范围","题次 n","F1","EM","按题证据齐全","平均 prompt token"),result)


def compression(r):
    groups, invalid = [], []
    for name in NAMES:
        block=r["compression"][name]
        for group,a in block.items():
            groups.append((name,group,rate(a["evidence_loss"]),pct_avg(a["final_recall"]),pct_avg(a["step_recall"]["micro"]),
                style.mean(a["final_segments"]),style.mean(a["final_nongold"]),style.mean(a["cost"]["plain"],1),
                style.mean(a["cost"]["cached"],1)))
            if name in ("v3","r2","r4"):
                q=a["invalid_outputs"]
                invalid.append((name,group,rate(q["any_invalid"]),rate(q["over_budget"])))
    return [
        style.table(("策略","格","轨迹丢证据率","最终召回","单步召回","平均段数","平均无关段数","无缓存","开缓存"),groups),
        style.table(("模型行","格","非法动作","超预算动作"),invalid),
    ]


def pairs(paired):
    body=[]
    cells=[]
    for tag,p in zip("abcdef",paired,strict=True):
        label=f"{p['left']} − {p['right']}"
        counts=p["f1_counts"]
        m=p["em_matrix_correct_then_incorrect"]
        body.append((tag,label,f"{p['n']:,}",signed_num(p["f1_difference_pp"]),signed_num(p["em_difference_pp"]),
                     f"{counts['left_higher']:,} / {counts['equal']:,} / {counts['right_higher']:,}",
                     " / ".join(f"{x:,}" for row in m for x in row)))
        cells.extend((tag,label,k,f"{v['n']:,}",signed_num(v["f1_difference_pp"])) for k,v in p["by_cell"].items())
    return [
        style.table(("比较","策略","配对题次 n","ΔF1（百分点）","ΔEM（百分点）","F1 高 / 平 / 低",
                     "EM 同对 / 仅左对 / 仅右对 / 同错"),body),
        style.table(("比较","策略","格","配对题次 n","ΔF1（百分点）"),cells),
    ]


def by_hop(r):
    kept, lost, csv_rows=[],[],[]
    for name,cells in r["hop_retention"].items():
        for scope,a in cells.items():
            kept.append((name,scope,*(rate(a["retained"][h]) for h in ("1","2","3","4","2+"))))
            lost.append((name,scope,*(rate(a["first_loss"][h]) for h in ("1","2","3","4","none"))))
            csv_rows.extend({"strategy":name,"scope":scope,"statistic":s,"hop":h,**z}
                            for s,zs in a.items() for h,z in zs.items())
    c=r["hop_r4_minus_rr_tau"]
    byq=[]
    for name in NAMES:
        for h,a in r["reader"][name]["pooled"]["by_hops"].items():
            byq.append((name,h,f"{a['n']:,}",style.pct(a["f1"],3),style.pct(a["em"],3)))
    data16=[
        style.table(("策略","范围","第1跳","第2跳","第3跳","第4跳","第≥2跳"),kept),
        style.table(("策略","范围","首失第1跳","首失第2跳","首失第3跳","首失第4跳","全部保留"),lost),
        style.table(("范围","跳位","r4","RR-τ","r4 − RR-τ（百分点）"),
                     ((scope,h,rate(z["left"]),rate(z["right"]),signed_num(z["delta_pp"])) for scope,b in c.items() for h,z in b.items())),
        "### 按题的跳数",
        style.table(("策略","问题跳数","题次 n","F1","EM"),byq),
        style.table(("问题跳数","配对题次 n","r4 − RR-τ ΔF1（百分点）"),
                     ((h,f"{a['n']:,}",signed_num(a["f1_difference_pp"])) for h,a in r["paired_f1_by_question_hops"].items())),
    ]
    fh=io.StringIO(newline="")
    writer=csv.DictWriter(fh,fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    return data16,fh.getvalue().encode()


def charts(r):
    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    colors=("#171717","#848484","#3b6e84","#648219","#7353a3","#b84926","#d12659","#ba7b14","#247274","#143c82")
    markers=("o","v","d","P","X","<",">","^","s","D")
    refs=("no_compress","no_context")
    labels={"no_compress":"No compression","no_context":"No context","latest":"Latest","bm25":"BM25","oracle":"Oracle",
            "v3":"SFT v3","r2":"GRPO r2","r4":"GRPO r4","RR-τ":"RR-τ","RR-fill":"RR-fill"}
    all_y=[100*x["f1"] for row in r["reader"].values() for x in row["cells"].values()]
    y_top=min(100,max(10,5*ceil((max(all_y)+3)/5)))
    all_cost=[c["cost"]["cached"]["mean"] for s in r["compression"].values() for c in s.values()]
    x_max=1.12*max(all_cost)
    out={}
    with mpl.rc_context({"font.family":"DejaVu Sans","font.size":9,"svg.fonttype":"none","svg.hashsalt":"test-dev-v1",
                         "axes.spines.top":False,"axes.spines.right":False}):
        for measure in ("budget","cost"):
            fig,axes=plt.subplots(1,2,figsize=(11.4,5.9),sharey=True)
            fig.subplots_adjust(left=.085,right=.978,bottom=.27,top=.87,wspace=.12)
            handles=[]
            for ax,mode in zip(axes,("append","reselect"),strict=True):
                ax.set_title(mode.capitalize(),loc="left",fontweight="bold")
                ax.set_ylim(0,y_top)
                ax.grid(axis="y",color="#d9dcde",lw=.6)
                ax.set_axisbelow(True)
                if measure=="budget":
                    xs=list(data.BUDGETS.values())
                    ax.set_xticks(xs,[f"{v:,}" for v in xs])
                    ax.set_xlim(1500,6000)
                    ax.set_xlabel("Passage budget (tokens)")
                else:
                    ax.set_xscale("symlog",linthresh=1500,linscale=.35,base=10)
                    ax.set_xlim(-90,x_max)
                    ax.set_xticks([v for v in (0,1500,3000,10000,30000,100000) if v <= x_max])
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:,.0f}"))
                    ax.set_xlabel("Cached cost (equivalent input tokens / trajectory)")
                for name,color,marker in zip(NAMES,colors,markers,strict=True):
                    info=r["reader"][name]
                    line="--" if name=="no_compress" else ":" if name=="no_context" else "-"
                    style_marker=marker
                    if name in refs and measure=="budget":
                        artist=ax.axhline(100*info["pooled"]["f1"],color=color,linestyle=line,lw=1.25)
                        style_marker=""
                    else:
                        costs=r["compression"][name]
                        cells=[f"{mode}|{tier}" for tier in data.BUDGETS]
                        x=([costs[k]["cost"]["cached"]["mean"] for k in cells] if measure=="cost"
                           else list(data.BUDGETS.values()))
                        y=([100*info["pooled"]["f1"]]*len(cells) if name in refs
                           else [100*info["cells"][k]["f1"] for k in cells])
                        artist,=ax.plot(x,y,color=color,marker=marker,linestyle=line if measure=="budget" else "none",
                                       lw=1.7 if name=="r4" else 1.15,markersize=6 if name=="r4" else 4.6)
                    artist.set_gid(f"{mode}-{NAMES.index(name)}")
                    if ax is axes[0]:
                        handles.append(Line2D([],[],label=labels[name],color=color,marker=style_marker,
                                             linestyle=line if measure=="budget" else "none"))
            axes[0].set_ylabel("Answer F1 (%)")
            title="Locked test reader F1 by passage budget" if measure=="budget" else "Locked test reader F1 by cached trajectory cost"
            fig.suptitle(title,x=.085,y=.978,ha="left",fontweight="bold",fontsize=14)
            fig.legend(handles=handles,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.50,.045),fontsize=8.5)
            fig.text(.085,.008,"Each cell: n = 1,804 questions from 451 tasks. Each shared row: n = 1,804 actual predictions.",fontsize=8)
            buffer=io.BytesIO()
            fig.savefig(buffer,format="svg",metadata={"Date":None})
            plt.close(fig)
            raw=buffer.getvalue()
            svg=ET.fromstring(raw)
            ids={el.attrib.get("id") for el in svg.iter()}
            need(all(f"{mode}-{i}" in ids for mode in ("append","reselect") for i in range(len(NAMES))), "incomplete SVG")
            out[f"{measure}_f1.svg"]=raw
    return out


def render(r,hops):
    g0=r["g0"]["comparison"]
    g1=r["g1"]
    s=r["consistency"]
    smoke=s["smoke"]
    ex=r["gold_exposure"]
    expo_cmps=[]
    restored=r["completion"].get("reader_recovery")
    g3_recovery=[]
    if restored:
        ev=restored["effective_flag_from_log"]
        n=restored["coverage"]["n"]
        g3_recovery=[
            "### G3 的 CPU 收尾",
            f"G3 在写完 n={n:,} 条预测并退出引擎后，读取预检的 `df_opening` 时触发 `KeyError`。"
            "修复在请求预检中保存磁盘快照，并在发动引擎前校验该字段。"
            f"本轮只读原始预测及请求，在 CPU 上逐条核对 {restored['coverage']['n_tasks']:,} 个批次、题目 ID、配对键、prompt 哈希和原始答案得分。"
            f"生效值引用 vLLM 启动日志 `launcher.log` 第 {ev['line_number']} 行。"
            f"原 GPU 子进程退出码为 {restored['failure']['original_g3_exit_code']}，CPU 验收在北京时间 {r['completion']['cpu_recovery_beijing']} 入档。"
            f"回归后全量 CPU 测试通过 {restored['cpu_test_count']:,}/{restored['cpu_test_count']:,} 项。"
            "[G3 核验回执](G3_recovery.json) · [读者运行配置](reader_run_config.json)",
        ]
    for key,x in ex["comparison_a_by_group"].items():
        k=x["f1_counts"]
        expo_cmps.append(("题的 gold 全见过" if key=="all_seen" else "至少一段未见过",f"{x['n']:,}",
                           signed_num(x["f1_difference_pp"]),signed_num(x["em_difference_pp"]),
                           f"{k['left_higher']:,} / {k['equal']:,} / {k['right_higher']:,}"))
    heldout=[]
    for name in NAMES:
        old=r["heldout_reference"][name]
        new=r["reader"][name]["pooled"]
        source={"reader_heldout_v2":"任务16","reranker_v1":"任务17同批","grpo_r4":"任务18 G 同批"}
        heldout.append((name,style.pct(old["f1"],3),f"{old['n']:,}",source[Path(old["source"]).parent.name],
                        style.pct(new["f1"],3),f"{new['n']:,}"))
    return "\n\n".join([
        "# 任务19 · MuSiQue dev 锁定测试集最终对比",
        "这是锁定测试集的第一次使用。行、检查点、阈值、预算和推理设置均在本轮前冻结，按登记协议执行一次。"
        "预算来自 LOCK.md，重排 checkpoint-5300 和 τ=0.5 来自任务17。SFT v3 ckpt-3000 用任务11合并基座，"
        "r2 ckpt-125 与 r4 ckpt-500 使用各自 LoRA，压缩器引擎沿用任务18 F，读者配置沿用任务16及任务18 G。"
        "任务19 最后一条预测落盘时间为北京时间 "+r["completion"]["finished_beijing"]+"。"
        f"锁定集 n={r['n_tasks']:,} 任务、{r['n_questions']:,} 道题。"
        f"Phase P 全量 CPU 测试通过 {r['cpu_tests']['tests']:,}/{r['cpu_tests']['tests']:,} 项。"
        "[运行配置](run_config.json) · [结果 JSON](test_v1_results.json) · [交付前锁定 SHA256](lock_final.json)",
        "## 执行核验",
        f"G0 重跑 n={r['g0']['n_tasks']} 个 held-out 任务×六格，执行动作不同 {rate(metrics.rate(g0['executed_action_different'],g0['n']))}。"
        f"同状态 prompt 哈希核验 n={g0['same_state_prompt_checks']:,}。"
        f"G1 重排打分 n={g1['n_pairs']:,} 对，截断 {g1['n_truncated']:,}/{g1['n_pairs']:,}。"
        "Qwen3-8B 读者确认有效配置 `disable_cascade_attn=True`，每任务按固定顺序提交 200 条（n=90,200 条）。",
        *g3_recovery,
        "## 读者 F1 / EM 和按题证据齐全",
        "no_compress 和 no_context 各对每题推理一次。两行的模式与档位引用相同的 n=1,804 条共享结果。",
        answers(r),
        "## 压缩轨迹",
        "最终召回以轨迹为单位，单步召回按步合并。成本为 `eval/cost.py` 折合输入 token。"
        "no_context 在每步和题末的上下文均为空。",
        *compression(r),
        "## 预注册配对比较",
        "六格内逐题配对。比较 e 在每格复用同一批 no_compress 答案。",
        *pairs(r["paired"]),
        "## 按跳保留与首次丢失",
        "保留率按 gold 出现计，首次丢失按题次计。用任务末保留列表判定。"
        "[按跳表 CSV](hop_retention.csv)",
        *hops,
        "## 训练轨迹中见过的 gold",
        f"逐字比较 (title, paragraph_text)。训练源 n={ex['source_task_counts']['train']:,} 任务，held-out 源 n={ex['source_task_counts']['heldout']:,} 任务。"
        f"锁定集去重 gold 当过 gold 的比例为 **{rate(ex['unique_test_gold'])}**，按跳出现计为 {rate(ex['gold_hop_occurrences'])}。"
        f"全部 gold 都当过的题为 {rate(ex['question_all_seen'])}。",
        *( [style.table(("题组","配对题次 n","r4 − RR-τ ΔF1（百分点）","ΔEM（百分点）","F1 高 / 平 / 低"),expo_cmps)]
           if expo_cmps else []),
        "## 读者一致性",
        f"任务03小样原始输出不同 {smoke['raw_output_mismatches']}/{smoke['n']} 条。"
        "同次运行按完整 prompt token 序列分组，分母为至少出现两次的组。",
        style.table(("范围","组内原始输出不一致率"),
                     ((name,rate(item)) for name,item in [("合并",s["within_all"]),*s["within_by_strategy"].items()])),
        "## 与 held-out 的同名策略对照",
        "两列按各自样本汇总。",
        style.table(("策略","held-out F1","held-out 题次 n","held-out 来源","锁定集 F1","锁定集题次 n"),heldout),
        "## 图",
        "[预算–F1](budget_f1.svg) · [开缓存成本–F1](cost_f1.svg)",
    ])+"\n"


def main():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","report must be CPU")
    result=read(W/"reader_metrics.json")
    need(result["status"]=="complete" and result["task"]==19 and result["locked_test_first_use"],"Task 19 CPU metrics missing")
    need(read(OUT/"lock_final.json")["files"]==read(OUT/"lock_preflight.json")["files"],"locked files changed")
    hops,csv_data=by_hop(result)
    documents={"test_v1_report.md":render(result,hops).encode(),CSV:csv_data,**charts(result)}
    require_new(*[OUT/k for k in (*documents,"test_v1_results.json")])
    for k,value in documents.items():
        with (OUT/k).open("xb") as out:
            out.write(value)
    result["delivery"]={k:record(OUT/k) for k in documents}
    result["lock_final"]=record(OUT/"lock_final.json")
    result["settings_fresh_selection"]=False
    write_new(OUT/"test_v1_results.json",result)
    print("Task 19 report and figures complete",flush=True)


if __name__=="__main__":
    main()
