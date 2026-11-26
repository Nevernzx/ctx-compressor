"""Build task17 closed loops and run the frozen Qwen reader in registered task batches."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain, groupby
import json
from math import fsum, isfinite
import os
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import reader_heldout_report as report, reader_heldout_v2 as v16, reranker_v1_data as data
from scripts import run_reader_heldout as reader
from src import reader_heldout_inputs as reuse
from src.heldout_eval import HeldoutReplay
from src.reranker_v1 import (DATA, GRID, R2, SMALL, STRATEGIES, V3, WORK, candidates, choose_tau,
                            manifest_entries, need, read, record, require_new, reranker_action, verify)

SHORT = reuse.EXTERNAL/"rt/r17"
VAL = WORK/"val_reader"
HELD = WORK/"heldout_reader"
LOOPS = WORK/"rollouts"
SETUP = {"val":(VAL,100,21600), "heldout":(HELD,249,29880)}
TAU_NAMES = tuple(f"tau-{tau:g}" for tau in GRID)


@dataclass(frozen=True)
class Strategy:
    name: str


def load_scores(which, tasks):
    proof=read(SMALL/"training.json")
    entry=proof["score_cache"][which]
    path=Path(entry["path"])
    need(proof["status"]=="complete" and path.is_relative_to(WORK/"scores"),"incomplete S1 score cache")
    verify(path,entry["sha256"])
    labels={k:dict(candidates(t["steps"])) for k,t in tasks.items()}
    scores=defaultdict(dict)
    for row in reuse.rows(path):
        task,pid,prob,score=row["task"],row["pid"],row["prob"],row["score"]
        need(task in labels and row["label"]==labels[task][pid] and pid not in scores[task]
             and 0<=prob<=1 and isfinite(score),"score cache identity, value or duplication")
        scores[task][pid]=prob
    need(scores.keys()==labels.keys() and all(scores[k].keys()==labels[k].keys() for k in scores),"score coverage differs")
    return scores


def val_task_inputs():
    split=data.verify_split()
    wanted=set(split["val100_tasks"])
    tasks={r["task"]:r for r in reuse.rows(data.DATA_OUT/"train-val.jsonl") if r["task"] in wanted}
    need(len(tasks)==100,"incomplete val100 tasks")
    raw=split["input_files"]["raw_train"]
    verify(Path(raw["path"]),raw["sha256"])
    eids={eid for t in tasks.values() for eid in t["example_ids"]}
    definitions={r["id"]: (r["question"],r["answer"],r["answer_aliases"])
                 for r in reuse.rows(Path(raw["path"])) if r["id"] in eids}
    need(definitions.keys()==eids and len(eids)==400,"incomplete val100 answer keys")
    for t in tasks.values():
        metadata=[]
        for i,eid in enumerate(t["example_ids"]):
            question,answer,aliases=definitions[eid]
            hops=[s for s in t["steps"] if s["question_index"]==i]
            need(question==t["questions"][i] and len(hops) in (2,3,4)
                 and all(h["example_id"]==eid and h["hop_index"]==j for j,h in enumerate(hops)),
                 "val100 label identity differs")
            metadata.append({"example_id":eid,"question":question,"answer":answer,
                             "answer_aliases":aliases,"question_hops":len(hops),
                             "gold":{h["gold"] for h in hops}})
        t["metadata"]=metadata
        t["all_gold"]=set.union(*(m["gold"] for m in metadata))
    idx=split["input_files"]["corpus"]
    verify(Path(idx["path"]),idx["sha256"])
    corpus=list(reuse.rows(Path(idx["path"])))
    return corpus,tasks


def per_context(finals,tasks):
    by_cell=defaultdict(list)
    for ((mode,tier),task),kept in finals.items():
        not_gold=len(set(kept)-tasks[task]["all_gold"])
        by_cell[f"{mode}|{tier}"].append((len(kept),not_gold))
    all_rows=list(chain.from_iterable(by_cell.values()))
    result={}
    for cell, items in [("pooled",all_rows),*by_cell.items()]:
        n=len(items)
        result[cell]={"n_trajectories":n,"mean_keep":fsum(a for a,b in items)/n,
                      "mean_nongold":fsum(b for a,b in items)/n}
    return result


def run_rollout(kind,name,tau,tasks,corpus,scores,baseline=None):
    destination=LOOPS/kind/name
    require_new(destination)
    destination.mkdir(parents=True)
    replay=HeldoutReplay(tasks.values(),corpus,name)
    archive=destination/"steps.jsonl"
    n=0
    with archive.open("x",encoding="utf-8") as stream:
        while not replay.completed():
            batch=replay.next_batch()
            out=[{"raw_output":data.compact(reranker_action(item["state"],scores[item["task"]],tau)),
                  "generation_backend":"cpu_policy"} for item in batch]
            for row in replay.advance_batch(out):
                state=row["state"]
                row["reranker_scores"]={str(pid):scores[row["task"]][pid] for pid in state["pid_by_lid"].values()}
                stream.write(data.compact(row)+"\n")
                n+=1
    metrics=replay.aggregate(baseline=baseline)
    need(metrics["combined"]["n_tasks"]==len(tasks)*6 and metrics["combined"]["action_validity"]["invalid_count"]==0,
         "closed-loop coverage or validity differs")
    del replay
    steps={**record(archive),"lines":n}
    metrics["run"]={"steps_file":steps,"tau":tau}
    small=destination/"metrics.json"
    reuse.write_new(small,metrics)
    certified=reuse.Source(name,small,reuse.checksum(small),archive,steps["sha256"],f"Task17 {kind} trajectory")
    finals,costs=reuse.replay(certified,tasks,[p["ntok"] for p in corpus])
    print(name,"rollout",n,"steps",flush=True)
    return finals,{"metrics":metrics,"steps":steps,"metrics_file":record(small),"costs":costs,
                   "final_context":per_context(finals,tasks)}


def historical_reader_parts():
    signature=manifest_entries(PROJECT/"eval/reader_heldout/TASK12_SHA256SUMS")
    p=reader.STANDARD/"input_preflight/requests.jsonl"
    verify(p,signature[str(p)])
    grouped=defaultdict(lambda: defaultdict(list))
    for row in reuse.rows(p):
        if row["strategy"] in (R2,V3):
            grouped[row["task"]][row["strategy"]].append(row)
    expected=[(name,mode,tier,q) for name in (R2,V3) for mode,tier in reuse.CELLS for q in range(4)]
    by_task={}
    for task,strategies in grouped.items():
        need(set(strategies)=={R2,V3},"signed reader task lacks a comparator")
        batch=[item for name in (R2,V3) for item in strategies[name]]
        need([(r["strategy"],r["mode"],r["tier"],r["question_index"]) for r in batch]==expected,"signed r2/v3 order differs")
        by_task[task]=batch
    need(len(by_task)==249, "signed r2/v3 tasks differ")
    return by_task,{"path":str(p),"sha256":signature[str(p)]}


def splice_rr_and_historical(rr,old):
    for task,block in groupby(rr,key=lambda r:r["task"]):
        yield from block
        yield from old.pop(task)
    need(not old,"signed baseline tasks not covered")


def validate_batches(source,names,n):
    counts=Counter()
    expected=[(name,mode,tier,q) for name in names for mode,tier in reuse.CELLS for q in range(4)]
    for task,block in groupby(source,key=lambda r:r["task"]):
        batch=list(block)
        need([(r["strategy"],r["mode"],r["tier"],r["question_index"]) for r in batch]==expected,"unregistered reader batch")
        counts[task]+=len(batch)
        need(counts[task]==len(expected),"duplicate reader task")
        yield task,batch
    need(len(counts)==n and set(counts.values())=={len(expected)},"reader batch count differs")


def hop_diagnostics(finals,tasks):
    totals=defaultdict(Counter)
    for ((mode,tier),taskid),pids in finals.items():
        kept=set(pids)
        hops=defaultdict(list)
        for row in tasks[taskid]["steps"]:
            hops[row["question_index"]].append((1+row["hop_index"],row["gold"]))
        for question_hops in hops.values():
            flags=[(pos,pid in kept) for pos,pid in sorted(question_hops)]
            first=next((pos for pos,ok in flags if not ok),None)
            for label in (mode,"pooled"):
                agg=totals[label]
                agg["questions"]+=1
                agg["first_lost_"+(str(first) if first is not None else "none")]+=1
                for pos,ok in flags:
                    agg[f"n_hop_{pos}"]+=1
                    agg[f"retained_hop_{pos}"]+=ok
    result={}
    for where,agg in totals.items():
        cells={}
        for h in range(1,5):
            n=agg[f"n_hop_{h}"]
            count=agg[f"retained_hop_{h}"]
            cells[str(h)]={"count":count,"n":n,"rate":count/n if n else None}
        later=[cells[str(h)] for h in range(2,5)]
        n=sum(x["n"] for x in later)
        count=sum(x["count"] for x in later)
        cells["2+"]=dict(count=count,n=n,rate=count/n if n else None)
        result[where]={"retained":cells,
                      "first_loss":{k:{"count":agg["first_lost_"+k],"n":agg["questions"],
                       "rate":agg["first_lost_"+k]/agg["questions"]} for k in ("1","2","3","4","none")}}
    return result


def prepare(kind):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU preflight only")
    directory,n_tasks,n_requests=SETUP[kind]
    require_new(directory)
    directory.mkdir()
    history={}
    rolls={}
    if kind=="val":
        corpus,tasks=val_task_inputs()
        scores=load_scores("selected_val100",tasks)
        names=list(TAU_NAMES)
        settings=[(name,tau,scores) for name,tau in zip(names,GRID,strict=True)]
        nc=None
    else:
        proof=read(SMALL/"tau_selection.json")
        need(proof["status"]=="complete" and proof["n_tasks"]==100,"val100 tau result is absent")
        tau=proof["tau"]
        idx=reuse.sums(PROJECT/"SHA256SUMS")
        corpus,tasks,history=reuse.load_task_inputs(idx)
        tasks = {task_id: {**task, "task": task_id} for task_id, task in tasks.items()}
        tuned=load_scores("selected_heldout",tasks)
        zero=load_scores("zero_shot_heldout",tasks)
        names=list(STRATEGIES)
        settings=[("RR-fill",None,tuned),("RR-τ",tau,tuned),("RR-zs-fill",None,zero)]
        nc_source=data.eval_sources()["no_compress"]
        nc=reuse.checked_json(nc_source.metrics,nc_source.metrics_sha256)
    finalists={}
    for name,tau,scores in settings:
        finalists[name],rolls[name]=run_rollout(kind,name,tau,tasks,corpus,scores,nc)
    if kind=="heldout":
        pid_lengths=[p["ntok"] for p in corpus]
        historical=data.eval_sources()
        for name in (R2,V3,"bm25","oracle"):
            src=historical[name]
            finalists[name],costs=reuse.replay(src,tasks,pid_lengths)
            rolls[name]={"metrics":read(src.metrics),"metrics_file":record(src.metrics),"steps":{"path":str(src.steps),"sha256":src.steps_sha256},
                         "costs":costs,"final_context":per_context(finalists[name],tasks)}
        are_for_hops=("RR-fill","RR-τ",R2,V3,"bm25","oracle")
        hops={name:hop_diagnostics(finalists[name],tasks) for name in are_for_hops}
        contrasts={}
        for scope in ("reselect","append","pooled"):
            contrasts[scope]={}
            for hop in ("1","2+"):
                left=hops[R2][scope]["retained"][hop]
                right=hops["RR-τ"][scope]["retained"][hop]
                need(left["n"]==right["n"],"hop denominators differ")
                contrasts[scope][hop]={"left":left,"right":right,"delta_pp":100*(left["rate"]-right["rate"])}
        reuse.write_new(SMALL/"hop_retention.json",{"status":"complete","by_strategy":hops,
                         "r2_minus_rr_tau":contrasts,"unit":"question-hop occurrence of the supporting paragraph"})
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(reuse.EXTERNAL/"models/Qwen3-8B"),
                                              local_files_only=True,trust_remote_code=False)
    inputs=reuse.make_requests([Strategy(name) for name,_,_ in settings],finalists,tasks,corpus,tokenizer,False)
    if kind=="heldout":
        old,reference=historical_reader_parts()
        history["same_batch_r2_v3_inputs"]=reference
        inputs=splice_rr_and_historical(inputs,old)
    ordered=(r for _,batch in validate_batches(inputs,names,n_tasks) for r in batch)
    spool=reuse.export_requests(directory/"requests.jsonl",ordered)
    need(spool["n"]==n_requests,"number of prompts differs from registered grid")
    preflight={"status":"passed","phase":kind,"strategies":names,"n_tasks":n_tasks,"n_prompts":n_requests,
               "requests":spool,"source_signatures":history,"rollouts":rolls,"df_opening":reader.original.root_df()}
    reuse.write_new(directory/"input_preflight.json",preflight)
    print(f"{kind}: {n_requests} reader prompts preflighted",flush=True)


def infer(kind):
    directory,n_tasks,n_predictions=SETUP[kind]
    pre=directory/"input_preflight.json"
    input_path=directory/"requests.jsonl"
    info=read(pre)
    need(info["status"]=="passed" and info["n_prompts"]==n_predictions and info["n_tasks"]==n_tasks,
         "reader preflight differs")
    verify(input_path,info["requests"]["sha256"])
    batched=validate_batches(reuse.rows(input_path),info["strategies"],n_tasks)
    with patch.multiple(v16, WORK=directory, PRE=pre, SHORT=SHORT):
        v16.run_engine(directory,SMALL/f"{kind}_reader_config.json",batched,n_tasks,n_predictions,
                       smoke=kind=="heldout",provenance={"preflight_sha256":reuse.checksum(pre),
                       "n_tasks":n_tasks,"batch_size":n_predictions//n_tasks,"task17_phase":kind})


def validated_predictions(kind):
    directory,n_tasks,n_predictions=SETUP[kind]
    config=read(SMALL/f"{kind}_reader_config.json")
    info=read(directory/"input_preflight.json")
    need(config["status"]=="complete" and config["effective_disable_cascade_attn"] is True
         and config["predictions"]==n_predictions,"incomplete or changed reader")
    preds=directory/"predictions.jsonl"
    verify(preds,config["predictions_sha256"])
    input_path=directory/"requests.jsonl"
    verify(input_path,info["requests"]["sha256"])
    ref=report.index_requests(reuse.rows(input_path))
    observed=report.collect(reuse.rows(preds),ref,info["strategies"])
    report.audit_matrix(observed,{"all_identical":False},n_tasks)
    return observed, info, config


def select():
    observed,info,config=validated_predictions("val")
    named={}
    for name,tau in zip(TAU_NAMES,GRID,strict=True):
        rollout=info["rollouts"][name]
        m=rollout["metrics"]["combined"]
        val=report.average([s for c in observed[name].values() for s in c.values()])
        need(val["n"]==2400,"val100 arm did not cover six cells")
        named[str(tau)]={"tau":tau,**val,"evidence_loss":m["evidence_loss"],**rollout["final_context"]["pooled"]}
    tau=choose_tau({v["tau"]:v["f1"] for v in named.values()})
    reuse.write_new(SMALL/"tau_selection.json",{"status":"complete","n_tasks":100,"n_predictions":21600,
                     "task_order":data.verify_split()["val100_tasks"],"tau":tau,"RR_tau_equals_fill":tau==0,
                     "tolerance":1e-9,"selection":"val100 six-cell F1; ties within 1e-9 take smaller tau",
                     "grid":named,"predictions_sha256":config["predictions_sha256"]})
    print(f"val100 selected tau={tau:g}",flush=True)


def consistency():
    from scripts.reader_diag_metrics import inconsistent, signature
    p=HELD/"predictions.jsonl"
    pred={(r["config"],r["task"],r["question_index"]):r for r in reuse.rows(p)}
    signs=manifest_entries(PROJECT/"eval/reader_heldout_v2/TASK16_SHA256SUMS")
    old= reuse.EXTERNAL/"reader_heldout_v2/predictions.jsonl"
    verify(old,signs[str(old)])
    changed=Counter()
    match=Counter()
    for line in reuse.rows(old):
        if line["strategy"] not in (R2,V3):
            continue
        key=line["config"],line["task"],line["question_index"]
        new=pred[key]
        need(line["prompt_sha1"]==new["prompt_sha1"] and line["example_id"]==new["example_id"],"historical same-prompt pair drift")
        match[line["strategy"]]+=1
        changed[line["strategy"]]+=line["raw_output"]!=new["raw_output"]
    need(match=={R2:5976,V3:5976},"wrong historical paired count")
    items=defaultdict(list)
    for request in reuse.rows(HELD/"requests.jsonl"):
        key=request["config"],request["task"],request["question_index"]
        items[request["strategy"]].append((signature(request),pred[key]["raw_output"]))
    return {"historical_task16_raw_diff":{name:{"count":changed[name],"n":match[name]} for name in (R2,V3)},
            "within_all":inconsistent(chain.from_iterable(items.values())),
            "within_by_strategy":{name:inconsistent(values) for name,values in items.items()}}


def compare(left,right,observed):
    paired=report.paired(left,right,observed)
    each={}
    for cell in report.CELLS:
        a,b=observed[left][cell], observed[right][cell]
        need(a.keys()==b.keys() and len(a)==996,"paired cell identity differs")
        each[cell]={"n":len(a),"f1_difference_pp":100*fsum(v.f1-b[k].f1 for k,v in a.items())/len(a)}
    return {**paired,"by_cell":each}


def summarize():
    observed,p,config=validated_predictions("heldout")
    table={}
    for name in STRATEGIES:
        groups=observed[name]
        row=p["rollouts"][name]
        cells={}
        for cell, group in groups.items():
            metrics=row["metrics"]["configs"][row["metrics"]["strategy"]+"|"+cell]["metrics"]
            cells[cell]={"reader":report.summarize_group(group.values()),"compression":metrics,"final_context":row["final_context"][cell]}
        table[name]={"pooled":{"reader":report.summarize_group(chain.from_iterable(g.values() for g in groups.values())),
                               "compression":row["metrics"]["combined"],"final_context":row["final_context"]["pooled"]},
                     "cells":cells}
    v16file=PROJECT/"eval/reader_heldout_v2/reader_heldout_v2_results.json"
    signed=manifest_entries(PROJECT/"eval/reader_heldout_v2/TASK16_SHA256SUMS")
    verify(v16file,signed[str(v16file.relative_to(PROJECT))])
    historical=read(v16file)
    refs=[name for name in ("no_compress","latest","bm25","oracle","h1:tau_star") if name in historical["strategies"]]
    refs.append(next(name for name in historical["strategies"] if name.startswith("grpo_r3:")))
    reuse.write_new(SMALL/"heldout_comparison.json",{"status":"complete","reader_smoke":config["smoke"],
        "n_predictions":29880,"n_tasks":249,"by_strategy":table,"task16_reference_rows":
        {name:historical["strategies"][name] for name in refs},
        "task16_reference":record(v16file),"primary":[compare(a,b,observed)
        for a,b in ((R2,"RR-τ"),(V3,"RR-τ"),(R2,"RR-fill"))],"consistency":consistency()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare-val","reader-val","select-tau","prepare-heldout","reader-heldout","summarize"))
    opts=p.parse_args()
    need(Path.cwd()==PROJECT,"wrong project root")
    is_reader=opts.action.startswith("reader-")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")==("1" if is_reader else ""),"wrong CPU/GPU visibility")
    {"prepare-val":lambda:prepare("val"),"reader-val":lambda:infer("val"),"select-tau":select,
     "prepare-heldout":lambda:prepare("heldout"),"reader-heldout":lambda:infer("heldout"),"summarize":summarize}[opts.action]()


if __name__=="__main__":
    main()

