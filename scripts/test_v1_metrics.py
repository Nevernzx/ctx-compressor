"""Recompute all preregistered Task 19 metrics from signed inputs and frozen outputs."""
import argparse
from collections import Counter, defaultdict
from itertools import chain
from math import fsum, isclose
import os
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

from scripts import close_reranker_grpo_metrics as compare
from scripts import reader_heldout_report as qa, reader_diag_metrics as diag
from scripts import reranker_v1_rollouts as rr
from scripts import test_v1_assets as models, test_v1_data as data, test_v1_reader as reader
from scripts import test_v1_rollout as rollout
from src import reader_heldout_inputs as reuse
from src.reranker_v1 import need, read, record, require_new, verify

OUT, W=data.SMALL,data.WORK
CELLS=qa.CELLS
PAIRS=(("r4","RR-τ"),("r4","v3"),("r4","r2"),("r4","bm25"),("r4","no_compress"),("oracle","r4"))
EXPOSURE=OUT/"gold_exposure.json"
NPAIRS=10824


def rate(count,n):
    return {"count":count,"n":n,"rate":count/n if n else None}


def mean(values):
    values=list(values)
    return {"n":len(values),"mean":fsum(values)/len(values) if values else None}


def gold_exposure(corpus,tasks):
    sig=reuse.sums(PROJECT/"SHA256SUMS")
    t17=PROJECT/"eval/reranker_v1/data_preflight.json"
    verify(t17,sig[str(t17.relative_to(PROJECT))])
    old=read(t17)["input_files"]
    gold_ids=set()
    counts, refs={},{}
    for key,n in (("train",4735),("heldout",249)):
        source=old[key]
        p=Path(source["path"])
        need(p==data.E/f"data/traj_train/{key}_tasks.jsonl","wrong trajectory corpus")
        h={}
        count=0
        for task in qa.input_rows(p,h):
            need(task["split"]==key,"training trajectory split differs")
            gold_ids.update(step["gold"] for step in task["steps"])
            count+=1
        need(count==n and h[str(p)]==source["sha256"],"training trajectory receipt differs")
        counts[key]=n
        refs[key]=source
    path=Path(old["corpus"]["path"])
    train_gold=set()
    hashes={}
    for i,row in enumerate(qa.input_rows(path,hashes)):
        need(row["pid"]==i,"train corpus PID order differs")
        if i in gold_ids:
            train_gold.add((row["title"],row["paragraph_text"]))
    need(hashes[str(path)]==old["corpus"]["sha256"] and len(train_gold)==len(gold_ids),"training corpus key differs")
    refs["corpus"]=old["corpus"]
    test_ids={s["gold"] for t in tasks.values() for s in t["steps"]}
    hits={p for p in test_ids if (corpus[p]["title"],corpus[p]["paragraph_text"]) in train_gold}
    all_seen=[]
    n_hops=sum(len(t["steps"]) for t in tasks.values())
    for task,t in tasks.items():
        for q,row in enumerate(t["metadata"]):
            if row["gold"] <= hits:
                all_seen.append((task,q))
    return {"status":"passed","unique_test_gold":rate(len(hits),len(test_ids)),"gold_hop_occurrences":
            rate(sum(s["gold"] in hits for t in tasks.values() for s in t["steps"]),n_hops),
            "question_all_seen":rate(len(all_seen),len(tasks)*4),"all_seen_questions":all_seen,
            "source_files":refs,"source_task_counts":counts,"source_gold_pid_count":len(gold_ids),
            "identity":"Exact UTF-8 title and paragraph_text in any train trajectory or held-out trajectory."}


def exposure_phase():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","compute gold exposure on CPU")
    require_new(EXPOSURE)
    corpus,tasks=data.load_test()
    r=gold_exposure(corpus,tasks)
    reuse.write_new(EXPOSURE,r)
    print("gold exposure denominator",r["unique_test_gold"],r["question_all_seen"],flush=True)


def completed():
    p=W/"status.json"
    stat=read(p)
    need(stat["status"]=="evaluation_complete" and stat["stage"]=="all" and stat["lock_released"]
         and not stat["automatic_retry"] and not Path(f"/proc/{stat['pid']}").exists(),"Task 19 GPU work is not finished")
    cfg_path=compare.checked(stat["config"])
    need(cfg_path==OUT/"run_config.json","unexpected task run configuration")
    cfg=read(cfg_path)
    names=("G0","G1","RR","G2-v3","G2-r2","G2-r4","G3_prepare","G3")
    need(tuple(cfg["stages"])==names and cfg["gpu"]==1 and cfg["locked_first_use"] and not cfg["automatic_retry"],
         "pipeline differs from registered order")
    for name, stamp in cfg["stages"].items():
        label="prepare-reader" if name=="G3_prepare" else name
        logdir=W/"stages"/label
        need(stamp==read(logdir/"receipt_complete.json") and stamp["exit_code"]==0
             and stamp["stage"]=="complete",f"stage receipt mismatch: {name}")
        for a in stamp["outputs"].values():
            compare.checked(a)
        if stamp["gpu"]:
            adm=read(logdir/"receipt_gate.json")["gpu"]
            need(adm["index"]==1 and adm["memory_used_mib"]<2048,"different GPU was admitted")
    result={"status":record(p),"config":stat["config"],"n_stages":len(names),
            "finished_beijing":compare.beijing(stat["utc"])}
    if "G3_recovery" in cfg:
        audit=read(compare.checked(cfg["G3_recovery"]))
        need(audit["status"]=="passed" and audit["coverage"]["n"]==reader.N_PREDICTIONS
             and stat["cpu_recovery"]==cfg["G3_recovery"] and cfg["effective_disable_cascade_attn"] is True,
             "G3 CPU recovery receipt differs")
        result.update(finished_beijing=compare.beijing(audit["completed_prediction_archive_utc"]),
                      cpu_recovery_beijing=compare.beijing(audit["recovered_utc"]),
                      reader_recovery=audit)
    return result


def reader_scores(tasks):
    d=reader.READER
    cfg=read(reader.CONFIG)
    pre=read(reader.PREFLIGHT)
    inf=read(d/"receipt_inference.json")
    smoke=read(d/"receipt_smoke.json")
    need(cfg["engine_shutdown"] and cfg["effective_disable_cascade_attn"] is True and cfg["status"]=="complete"
         and cfg["parameters"]==models.load()["reader"]["engine"] and cfg["generation"]==models.load()["reader"]["generation"]
         and cfg["tasks"]==pre["n_tasks"]==len(tasks)==451 and pre["tasks"]==list(tasks)
         and cfg["predictions"]==pre["requests"]["n"]==inf["n"]==reader.N_PREDICTIONS
         and inf["run_config_sha256"]==reuse.checksum(reader.CONFIG),"reader receipts differ")
    small=d/"smoke_predictions.jsonl"
    verify(small,smoke["predictions_sha256"])
    rows=list(reuse.rows(small))
    n=sum(r["raw_output"]!=r["expected_raw_output"] for r in rows)
    need(len(rows)==smoke["n"]==cfg["smoke"]["n"]==40
         and n==smoke["raw_output_mismatches"]==cfg["smoke"]["raw_output_mismatches"],"reader smoke differs")
    hashset={}
    req=d/"requests.jsonl"
    pred=d/"predictions.jsonl"
    outs=iter(qa.input_rows(pred,hashset))
    scores={name:defaultdict(dict) for name in data.NAMES}
    signature_groups={name:[] for name in data.NAMES}
    for t,batch in data.batches(qa.input_rows(req,hashset),list(tasks)):
        out=[next(outs,None) for _ in batch]
        refs=qa.index_requests(batch)
        need(all(r is not None and tuple(r[k] for k in qa.IDENTITY)==tuple(x[k] for k in qa.IDENTITY)
                 for r,x in zip(out,batch,strict=True)),f"reader output batch order differs: {t}")
        scored=qa.collect(out,refs,data.NAMES)
        for name,by in scored.items():
            for cell,rows in by.items():
                need(not scores[name][cell].keys() & rows.keys(),"duplicate reader question")
                scores[name][cell].update(rows)
        for r,x in zip(out,batch,strict=True):
            signature_groups[r["strategy"]].append((diag.signature(x),r["raw_output"]))
    need(next(outs,None) is None and hashset[str(req)]==pre["requests"]["sha256"]
         and hashset[str(pred)]==cfg["predictions_sha256"]==inf["predictions_sha256"],"reader spool bytes differ")
    qa.audit_matrix(scores,{"all_identical":True},n_tasks=len(tasks))
    return scores,{
        "inference":record(d/"receipt_inference.json"),"smoke":cfg["smoke"],"source_sha256":hashset,
        "within_all":diag.inconsistent(chain.from_iterable(signature_groups.values())),
        "within_by_strategy":{name:diag.inconsistent(v) for name,v in signature_groups.items()},
    }


def reader_table(scores):
    result={}
    for name,data_by_cell in scores.items():
        c={"cells":{cell:qa.summarize_group(rows.values()) for cell,rows in data_by_cell.items()}}
        c["pooled"]=qa.summarize_group(chain.from_iterable(x.values() for x in data_by_cell.values()))
        if "shared" in data_by_cell:
            c["by_mode"]={mode:c["pooled"] for mode in qa.MODES}
            c["by_tier"]={tier:c["pooled"] for tier in data.BUDGETS}
        else:
            c["by_mode"]={mode:qa.summarize_group(chain.from_iterable(v.values() for k,v in data_by_cell.items()
                                                                    if k.startswith(f"{mode}|"))) for mode in qa.MODES}
            c["by_tier"]={tier:qa.summarize_group(chain.from_iterable(v.values() for k,v in data_by_cell.items()
                                                                    if k.endswith(f"|{tier}"))) for tier in data.BUDGETS}
        result[name]=c
    return result


def finals_summary(tasks):
    pre=read(reader.PREFLIGHT)
    path=compare.checked(pre["final_contexts"])
    finals=defaultdict(dict)
    for r in reuse.rows(path):
        cell=(r["mode"],r["tier"])
        key=cell,r["task"]
        note=finals[r["strategy"]]
        need(key not in note and r["task"] in tasks and cell in reuse.CELLS and len(r["kept"])==len(set(r["kept"])),
             "unexpected or duplicate final context")
        note[key]=r["kept"]
    expected={(c,t) for c in reuse.CELLS for t in tasks}
    need(set(finals)==set(data.NAMES)-{"no_context"} and all(set(x)==expected for x in finals.values()),"incomplete final contexts")
    results={}
    for name, by in finals.items():
        m=read(compare.checked(pre["sources"][name]["metrics"]))
        rows=m["configs"]
        stats={cell:rows["|".join((name,*cell))]["metrics"] for cell in reuse.CELLS}
        buckets=defaultdict(list)
        for (cell,t),kept in by.items():
            buckets[cell].append((len(kept),len(set(kept)-tasks[t]["all_gold"])))
        buckets[None]=list(chain.from_iterable(buckets.values()))
        stats[None]=m["combined"]
        result={}
        for cell,c in stats.items():
            here=buckets[cell]
            n=len(here)
            need(c["n_tasks"]==n and c["cost"]["cached"]["n"]==c["cost"]["plain"]["n"]==n,"final context denominator")
            if cell is not None:
                reference=pre["costs_recomputed"][name]["|".join(cell)]
                need(reference["n"]==n and isclose(reference["recomputed_cached_mean"],c["cost"]["cached"]["mean"],rel_tol=1e-12),
                     "cost.py verification differs")
            result["pooled" if cell is None else "|".join(cell)]={
                **c,"final_segments":mean(v[0] for v in here),"final_nongold":mean(v[1] for v in here),
            }
        results[name]=result
    results["no_context"]={}
    for cell in ("pooled",*CELLS):
        n=len(tasks)*(6 if cell=="pooled" else 1)
        steps=sum(len(t["steps"]) for t in tasks.values())*(6 if cell=="pooled" else 1)
        results["no_context"][cell]={"n_tasks":n,"n_steps":steps,
            "evidence_loss":rate(n,n),"final_recall":{"mean":0,"n":n},
            "step_recall":{"micro":{"mean":0,"n":steps}},
            "cost":{k:{"mean":0,"n":n} for k in ("plain","cached")},
            "final_segments":{"mean":0,"n":n},"final_nongold":{"mean":0,"n":n},
            "evidence_complete_questions":rate(0,n*4),"definition":"No context at any step or at the final question."}
    return results,finals


def heldout_table():
    out={}
    p16=PROJECT/"eval/reader_heldout_v2/reader_heldout_v2_results.json"
    p17=PROJECT/"eval/reranker_v1/reranker_v1_results.json"
    p18=PROJECT/"eval/grpo_r4/grpo_r4_results.json"
    original=reuse.sums(PROJECT/"SHA256SUMS")
    for p in (p16,p17,p18):
        verify(p,original[str(p.relative_to(PROJECT))])
    f16,f17,f18=[read(p) for p in (p16,p17,p18)]
    for name in data.NAMES:
        if name in ("RR-fill","RR-τ"):
            values=f17["heldout"]["by_strategy"][name]["pooled"]["reader"]
            ref=p17
        elif name in ("v3","r2","r4"):
            arm={"v3":compare.V3,"r2":compare.R2,"r4":compare.R4_500}[name]
            values=f18["strategies"][arm]["pooled"]["reader"]
            ref=p18
        else:
            values=f16["strategies"][name]["pooled"]
            ref=p16
        out[name]={k:values[k] for k in ("f1","em","n")}
        out[name]["source"]=str(ref.relative_to(PROJECT))
    return out


def summarize():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","report on CPU")
    end=completed()
    corpus,tasks=data.load_test()
    score,consistency=reader_scores(tasks)
    readers=reader_table(score)
    comp,fin=finals_summary(tasks)
    for name in data.NAMES:
        col=readers[name]["pooled"]
        q=comp[name]["pooled"]["evidence_complete_questions"]
        factor=6 if name in ("no_compress","no_context") else 1
        need(q["count"]==factor*col["by_evidence_complete"]["true"]["n"] and q["n"]==factor*col["n"],
             f"reader evidence differs from final context: {name}")
    score["no_compress"]={cell:score["no_compress"]["shared"] for cell in CELLS}
    paired=[compare.paired(score,a,b) for a,b in PAIRS]
    need(all(r["n"]==NPAIRS for r in paired),"primary comparisons do not cover 10,824 paired question-cells")

    hopping={name:rr.hop_diagnostics(fin[name],tasks) for name in data.NONSHARED}
    contrast={}
    for mode in ("append","reselect","pooled"):
        left,right=[hopping[name][mode]["retained"] for name in ("r4","RR-τ")]
        contrast[mode]={h:{"left":left[h],"right":right[h],"n":left[h]["n"],
                           "delta_pp":100*(left[h]["rate"]-right[h]["rate"])} for h in ("1","2+")}
    chosen={h:{(t,i) for t,x in tasks.items() for i,q in enumerate(x["metadata"]) if q["question_hops"]==h}
            for h in (2,3,4)}
    by_hops={str(h):compare.paired(score,"r4","RR-τ",qkeys) for h,qkeys in chosen.items()}

    exposure=read(EXPOSURE)
    exp_pair={}
    if exposure["unique_test_gold"]["count"]:
        keys={tuple(q) for q in exposure["all_seen_questions"]}
        allq={(t,i) for t in tasks for i in range(4)}
        for k,g in (("all_seen",keys),("any_unseen",allq-keys)):
            exp_pair[k]=compare.paired(score,"r4","RR-τ",g) if g else {
                "left":"r4","right":"RR-τ","n":0,"f1_difference_pp":None,"em_difference_pp":None,
                "f1_counts":{"left_higher":0,"equal":0,"right_higher":0},"em_matrix_correct_then_incorrect":[[0,0],[0,0]]}
        need(sum(v["n"] for v in exp_pair.values())==NPAIRS,"exposure partitions differ")
    exposure["comparison_a_by_group"]=exp_pair
    return {"status":"complete","task":19,"locked_test_first_use":True,"n_tasks":len(tasks),"n_questions":len(tasks)*4,
            "n_paired_question_cells":NPAIRS,"completion":end,
            "settings":read(models.PRE),"g0":read(OUT/"G0_smoke.json"),"g1":read(OUT/"G1_scores.json"),
            "reader":readers,"compression":comp,"paired":paired,"hop_retention":hopping,
            "hop_r4_minus_rr_tau":contrast,"paired_f1_by_question_hops":by_hops,
            "consistency":consistency,"gold_exposure":exposure,"heldout_reference":heldout_table(),
            "cpu_tests":read(OUT/"cpu_tests.json")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("exposure","score"))
    flag=p.parse_args().action
    if flag=="exposure":
        exposure_phase()
    else:
        dest=W/"reader_metrics.json"
        require_new(dest)
        reuse.write_new(dest,summarize())
        print("Task 19 CPU metrics complete",dest,flush=True)
