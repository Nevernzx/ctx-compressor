"""Run the frozen Task 19 reranker and compressor through saved retrieval steps."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

from scripts import test_v1_assets as assets, test_v1_data as data
from scripts import reranker_v1_rollouts as old_rr
from scripts import train_reranker_v1 as fit
from scripts.run_reader_heldout import gpu
from src import reader_heldout_inputs as reuse
from src.build_traj import oracle_action
from src.heldout_eval import HeldoutReplay, load_heldout_assets
from src.model_inference import compact, generation_record, guard_prompt, inference_prompt
from src.policies import bm25_policy, latest_policy, no_compress_policy, score_query
from src.reranker_v1 import PairTokenizer, candidates, need, probability, read, record, require_new, reranker_action, verify

W, SMALL = data.WORK, data.SMALL
LOOPS = W/"rollouts"
SCORES=W/"scores/selected_test.jsonl"
G1=SMALL/"G1_scores.json"
HIST = data.E/"data/traj_train"
R17 = data.E/"reranker_v1"
PROOF_RR=SMALL/"rr_heldout_parity.json"
ROW_POLICY=("no_compress","latest","bm25","oracle","RR-fill","RR-τ")
MODE="locked-test-dev"


def gpu_ready():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="1" and os.environ.get("CUDA_DEVICE_ORDER")=="PCI_BUS_ID",
         "licensed GPU 1 admission required")
    sample=gpu()
    need(sample["index"]==1 and sample["memory_used_mib"]<2048,"GPU 1 became busy after its gate")
    return sample


def chunks(items, size):
    # Python 3.11 is shared by the reader and GRPO environments.
    from itertools import islice
    iterator=iter(items)
    while chunk:=tuple(islice(iterator,size)):
        yield chunk


def score():
    import torch
    from transformers import AutoTokenizer

    frozen=assets.load()["reranker"]
    corpus,tasks=data.load_test()
    require_new(SCORES.parent,G1)
    SCORES.parent.mkdir()
    tokenizer=PairTokenizer(AutoTokenizer.from_pretrained(frozen["tokenizer"],local_files_only=True,trust_remote_code=False))
    need(tokenizer.config()==frozen["template"],"test reranker template differs from Task 17")
    gpu_before=gpu_ready()
    started=time.monotonic()
    model=fit.load_model(Path(frozen["path"]),torch.bfloat16).eval()
    count=truncated=0
    with SCORES.open("x",encoding="utf-8") as out,torch.inference_mode():
        for task,row in tasks.items():
            ids=[pid for pid,_ in candidates(row["steps"])]
            docs=[corpus[p]["text"] for p in ids]
            encoded=tokenizer.encode("\n".join(row["questions"]),docs)
            offset=0
            for block in chunks(encoded,fit.EVAL_MICROBATCH):
                values=fit.score_pairs(model,[value for value,_ in block],frozen["template"]).tolist()
                for (_,trunc),value in zip(block,values,strict=True):
                    out.write(compact({"task":task,"pid":ids[offset],"score":value,"prob":probability(value)})+"\n")
                    count+=1
                    offset+=1
                    truncated+=trunc
            need(offset==len(ids),"missing candidate scores")
    need(count==sum(len(candidates(t["steps"])) for t in tasks.values()),"wrong number of test scores")
    reuse.write_new(G1,{"status":"complete","n_tasks":len(tasks),"n_pairs":count,"n_truncated":truncated,
                        "file":record(SCORES),"checkpoint":frozen["checkpoint"],"template":frozen["template"],
                        "wall_seconds":time.monotonic()-started,"gpu_before":gpu_before,
                        "peak_bytes_allocated":torch.cuda.max_memory_allocated()})


def read_scores(tasks):
    info=read(G1)
    need(info["status"]=="complete" and info["n_tasks"]==len(tasks) and info["file"]["path"]==str(SCORES),"test score receipt")
    verify(SCORES,info["file"]["sha256"])
    result=defaultdict(dict)
    for row in reuse.rows(SCORES):
        task,pid,prob=row["task"],row["pid"],row["prob"]
        need(task in tasks and pid not in result[task] and 0<=prob<=1,"wrong probability or duplicate score")
        result[task][pid]=prob
    need(result.keys()==tasks.keys() and all(result[t].keys()==dict(candidates(x["steps"])).keys() for t,x in tasks.items()),
         "test reranker score coverage differs")
    return result


def run_loop(tasks, corpus, name, destination, supply, *, base=None, split=MODE):
    require_new(destination)
    destination.mkdir()
    replay=HeldoutReplay(tasks.values(),corpus,name,budgets=data.BUDGETS)
    steps=destination/"steps.jsonl"
    n=0
    with steps.open("x",encoding="utf-8") as fp:
        while not replay.completed():
            batch=replay.next_batch()
            outputs=supply(batch)
            for row in replay.advance_batch(outputs):
                fp.write(compact(row)+"\n")
                n+=1
            fp.flush()
            print(name,"completed step wave",batch[0]["step"],"total rows",n,flush=True)
    expected=sum(len(t["steps"]) for t in tasks.values())*len(reuse.CELLS)
    result=replay.aggregate(base)
    need(n==expected==result["combined"]["n_steps"] and result["combined"]["n_tasks"]==len(tasks)*len(reuse.CELLS),
         f"incomplete loop for {name}")
    result.update(status="complete",steps_artifact=record(steps))
    result["settings"]["split"]=split
    reuse.write_new(destination/"metrics.json",result)
    return result


def cpu_supply(action):
    def run(batch):
        return [{"raw_output":compact(action(item)),"generation_backend":"cpu_policy"} for item in batch]
    return run


def baseline():
    import bm25s

    corpus,tasks=data.load_test()
    pre=read(SMALL/"data_preflight.json")
    for ref in pre["bm25_index"]:
        verify(Path(ref["path"]),ref["sha256"])
    index=bm25s.BM25.load(str(data.DATA/"bm25"),load_corpus=False)
    policies={k:v for k,v in (("no_compress",no_compress_policy),("latest",latest_policy()),("oracle",oracle_action))}
    bm25={task:bm25_policy(index=None,scores=score_query(index," ".join(row["questions"]))) for task,row in tasks.items()}
    require_new(SMALL/"cpu_baselines.json")
    LOOPS.mkdir(exist_ok=True)
    refs={}
    nc=None
    for name in ROW_POLICY[:4]:
        policy=(lambda x:bm25[x["task"]](x["state"])) if name=="bm25" else (lambda x:policies[name](x["state"]))
        dat=run_loop(tasks,corpus,name,LOOPS/name,cpu_supply(policy),base=nc)
        if name=="no_compress":
            nc=dat
        refs[name]=record(LOOPS/name/"metrics.json")
    reuse.write_new(SMALL/"cpu_baselines.json",{"status":"complete","n_tasks":len(tasks),"rollout_metrics":refs})


def loop_reranker():
    corpus,tasks=data.load_test()
    scores=read_scores(tasks)
    base=read(LOOPS/"no_compress/metrics.json")
    require_new(SMALL/"rr_rollouts.json")
    refs={}
    for name,tau in (("RR-fill",None),("RR-τ",.5)):
        action=lambda x:reranker_action(x["state"],scores[x["task"]],tau)
        result=run_loop(tasks,corpus,name,LOOPS/name,cpu_supply(action),base=base)
        need(result["combined"]["invalid_outputs"]["any_invalid"]["count"]==0,"invalid reranker action")
        refs[name]=record(LOOPS/name/"metrics.json")
    reuse.write_new(SMALL/"rr_rollouts.json",{"status":"complete","n_tasks":len(tasks),"metrics":refs,"tau":.5})


def exact_keys(row):
    return row["mode"],row["budget_tier"],row["task"],row["step"]


def archived_heldout():
    proof=read(R17/"heldout_reader/input_preflight.json")
    for file in (HIST/"corpus_train.jsonl",HIST/"heldout_tasks.jsonl"):
        verify(file,proof["source_signatures"][str(file)])
    tasks,corpus=load_heldout_assets(HIST)
    return {row["task"]:row for row in tasks},corpus


def rr_parity():
    """Execute 10 full archived held-out tasks under both Task 17 RR policies."""
    all_tasks,corpus=archived_heldout()
    scores=old_rr.load_scores("selected_heldout",all_tasks)
    ids=sorted(all_tasks)[:10]
    subset={t:all_tasks[t] for t in ids}
    index=reuse.sums(assets.T17/"TASK17_SHA256SUMS")
    summary={}
    for name,tau in (("RR-fill",None),("RR-τ",.5)):
        path=R17/f"rollouts/heldout/{name}/steps.jsonl"
        signature=index[str(path)]
        verify(path,signature)
        archive={exact_keys(r):r for r in reuse.rows(path) if r["task"] in subset}
        replay=HeldoutReplay(subset.values(),corpus,name,budgets=data.BUDGETS)
        seen=set()
        while not replay.completed():
            batch=replay.next_batch()
            supply=cpu_supply(lambda x:reranker_action(x["state"],scores[x["task"]],tau))
            for row in replay.advance_batch(supply(batch)):
                k=exact_keys(row)
                need(k not in seen and k in archive and all(row[f]==archive[k][f] for f in
                     ("parsed_action","keep_after","kept_before","gold","used_after","invalid_reason")),
                     f"Task 17 reranker action mismatch: {name} / {k}")
                seen.add(k)
        need(set(archive)==seen,f"incomplete parity run: {name}")
        summary[name]={"n":len(seen),"action_mismatches":0,"reference":record(path)}
    reuse.write_new(PROOF_RR,{"status":"passed","n_tasks":len(ids),"tasks":ids,"policies":summary})


def changes(row):
    old,new=set(row["kept_before"]),set(row["keep_after"])
    return tuple(p for p in row["keep_after"] if p not in old),tuple(p for p in row["kept_before"] if p not in new)


def compare_g0(steps, reference, subset, threshold):
    original={exact_keys(r):r for r in reuse.rows(reference) if r["task"] in subset}
    coverage=set()
    different=parsed=finals=identical=0
    for r in reuse.rows(steps):
        k=exact_keys(r)
        old=original[k]
        need(k not in coverage and all(r[f]==old[f] for f in ("returned","gold","hop_index","question_index")),
             f"G0 retrieval differs: {k}")
        if r["kept_before"]==old["kept_before"] and r["state"]["pid_by_lid"]==old["state"]["pid_by_lid"]:
            need(all(r["generation_metadata"][f]==old["generation_metadata"][f] for f in ("prompt_sha256","user_text_sha256")),
                 f"G0 same state prompt differs: {k}")
            identical+=1
        different+=changes(r)!=changes(old)
        parsed+=r["parsed_action"]!=old["parsed_action"]
        finals+=r["keep_after"]!=old["keep_after"]
        coverage.add(k)
    need(coverage==set(original) and coverage,"G0 trace incomplete")
    n=len(coverage)
    return {"status":"passed" if different/n<=threshold else "failed","n":n,"executed_action_different":different,
            "parsed_action_different":parsed,"keep_after_different":finals,"same_state_prompt_checks":identical,
            "rate":different/n,"threshold":threshold}


def compress(name, smoke=False):
    frozen=assets.load()
    sampling=frozen["compressor"]["sampling"]
    engine=frozen["compressor"]["engine"]
    need(engine["enforce_eager"] is True and sampling["temperature"]==0 and sampling["max_tokens"]==256,
         "frozen compressor configuration")
    if smoke:
        tasks,corpus=archived_heldout()
        tasks={t:tasks[t] for t in frozen["G0"]["ids"]}
        dest=W/"g0"
        base=None
        ref=assets.R4_TRACE
        verify(ref,frozen["G0"]["trace"]["sha256"])
    else:
        corpus,tasks=data.load_test()
        dest=LOOPS/name
        base=read(LOOPS/"no_compress/metrics.json")
    output=SMALL/("G0_smoke.json" if smoke else f"G2_{name}.json")
    require_new(dest,output)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    initial=gpu_ready()
    start=time.monotonic()
    llm=LLM(**engine)
    try:
        tok=llm.get_tokenizer()
        sample=SamplingParams(**sampling)
        adapter=LoRARequest(name,1,frozen["adapters"][name]["path"]) if name!="v3" else None

        def generate(batch):
            prompts=[inference_prompt(tok,row["state"]) for row in batch]
            for row,p in zip(batch,prompts,strict=True):
                guard_prompt(p,engine["max_model_len"],sampling["max_tokens"],(row["trajectory_id"],row["step"]))
            values=llm.generate([{"prompt_token_ids":p["prompt_token_ids"]} for p in prompts],
                                sampling_params=sample,lora_request=adapter,use_tqdm=False)
            return [generation_record(p,v,sampling["max_tokens"]) for p,v in zip(prompts,values,strict=True)]

        result=run_loop(tasks,corpus,name,dest,generate,base=base,split="heldout-smoke" if smoke else MODE)
    finally:
        llm.llm_engine.engine_core.shutdown()
    proof={"status":"complete","wall_seconds":time.monotonic()-start,"n_tasks":len(tasks),"gpu_before":initial,
           "engine_shutdown":True,"engine":engine,"sampling":sampling,"metrics":record(dest/"metrics.json"),
           "adapter":frozen["adapters"].get(name),"strategy":name}
    if smoke:
        test=compare_g0(Path(result["steps_artifact"]["path"]),ref,tasks,frozen["G0"]["threshold"])
        proof["comparison"]=test
        proof["reference"]=frozen["G0"]["trace"]
        proof["status"]=test["status"]
    reuse.write_new(output,proof)
    need(proof["status"] in ("passed","complete"),"NEED: G0 executed-action difference exceeds 5%")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("score","baseline","rr","rr-parity","G0","v3","r2","r4"))
    flag=parser.parse_args().action
    if flag in ("baseline","rr","rr-parity"):
        need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU loop must hide CUDA")
    tasks={"score":score,"baseline":baseline,"rr":loop_reranker,"rr-parity":rr_parity}
    if flag in tasks:
        tasks[flag]()
    else:
        compress("r4" if flag=="G0" else flag,smoke=flag=="G0")
