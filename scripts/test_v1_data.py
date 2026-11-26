"""Read and validate the locked Task 19 inputs, and rebuild the exact dev corpus."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import build_task10_locked_dev as freeze
from src import reader_heldout_inputs as reuse
from src.build_traj import build_corpus, hop_queries
from src.model_inference import compact
from src.reranker_v1 import need, read, record, require_new, verify

E=reuse.EXTERNAL
WORK=E/"test_v1"
SMALL=PROJECT/"eval/test_v1"
DATA=WORK/"data"
LOCK=PROJECT/"eval/test_dev_v1/LOCK.md"
DEPOT=E/"data/test_dev_v1"
RAW=PROJECT/"data/raw/musique/musique_ans_v1.0_dev.jsonl"
TOK=PROJECT/"data/raw/qwen3_tokenizer/tokenizer.json"
BUDGETS=dict(freeze.BUDGETS)
NAMES=("no_compress","no_context","latest","bm25","oracle","RR-fill","RR-τ","v3","r2","r4")
NONSHARED=NAMES[2:]
ORDER=tuple((s,None,None,i) for s in NAMES[:2] for i in range(4))+tuple(
    (s,mode,tier,i) for s in NONSHARED for mode,tier in reuse.CELLS for i in range(4))


def lock_check(lock, expected, allowed_paths):
    verify(lock,expected)
    matches=re.findall(r"^- (/[^ ]+) {2}([a-f0-9]{64})$",lock.read_text(),re.M)
    signed={Path(p):s for p,s in matches}
    need(len(matches)==len(signed)==len(allowed_paths) and set(signed)==set(allowed_paths),
         "lock inventory differs")
    records={}
    for p,digest in signed.items():
        need(p.resolve()==p and allowed_paths[p]==digest, f"locked signature or destination differs: {p}")
        verify(p,digest)
        records[str(p)]={"sha256":digest,"size":p.stat().st_size}
    return records


def verify_locked():
    idx=reuse.sums(PROJECT/"SHA256SUMS")
    paths={DEPOT/f:idx[f"data/test_dev_v1/{f}"] for f in ("test_tasks.jsonl","steps.jsonl","excluded_ids.json")}
    paths.update({LOCK.parent/f:idx[f"eval/test_dev_v1/{f}"] for f in ("exclusion_audit.json","structural_stats.json")})
    hashes=lock_check(LOCK,idx[str(LOCK.relative_to(PROJECT))],paths)
    need(hashes==read(SMALL/"lock_preflight.json")["files"],"preflight and original lock differ")
    return {"status":"passed","first_use_task":19,"files":hashes,"lock":record(LOCK),
            "verified_utc":datetime.now(timezone.utc).isoformat()}


def jsonl(path, items):
    with path.open("x",encoding="utf-8") as output:
        for item in items:
            output.write(compact(item)+"\n")
    return record(path)


def normalize(item):
    base=item["questions"]
    gold=item["question_gold"]
    hops=item["question_hops"]
    metadata=[{"example_id":q["id"],"question":q["question"],"answer":q["answer"],
               "answer_aliases":q["answer_aliases"],"question_hops":n,"gold":g}
              for q,g,n in zip(base,gold,hops,strict=True)]
    need(len(metadata)==4 and [q["example_id"] for q in metadata]==item["example_ids"],"locked question identity")
    need(sum(hops)==len(item["steps"]) and set(item["all_gold"])==set().union(*map(set,gold)),
         "locked gold or hop coverage")
    return {"task":item["task"],"split":"locked-test-dev","example_ids":item["example_ids"],
            "questions":[q["question"] for q in base],"steps":item["steps"],"metadata":metadata,"all_gold":item["all_gold"]}


def verify_gold(subset, raw, corpus, key_to_pid):
    checked=[]
    for task in subset:
        task_steps=0
        for q,ref in enumerate(task["metadata"]):
            ex=raw[ref["example_id"]]
            need((ex["question"],ex["answer"],ex["answer_aliases"])==(ref["question"],ref["answer"],ref["answer_aliases"]),
                 f"dev reference differs in task {task['task']}/{q}")
            listed=[s for s in task["steps"] if s["question_index"]==q]
            gold=hop_queries(ex,key_to_pid)
            by_idx={p["idx"]:(p["title"],p["paragraph_text"]) for p in ex["paragraphs"]}
            for h,(row,(_,pid),hop) in enumerate(zip(listed,gold,ex["question_decomposition"],strict=True)):
                p=corpus[row["gold"]]
                need(row["hop_index"]==h and row["gold"]==pid
                     and (p["title"],p["paragraph_text"])==by_idx[hop["paragraph_support_idx"]],
                     f"dev corpus gold bytes differ at {task['task']}/{q}/{h}")
                task_steps+=1
        checked.append({"task":task["task"],"steps":task_steps,"gold_mismatches":0})
    return checked


def reader_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(E/"models/Qwen3-8B"),local_files_only=True,trust_remote_code=False)


def full_length_preflight(tasks, corpus, tokenizer):
    longest=None
    too_long=[]
    count=0
    for task in tasks:
        pids=list(dict.fromkeys(p for step in task["steps"] for p in step["new"]))
        passages=reuse.render_passages(corpus,pids)
        for i,q in enumerate(task["metadata"]):
            ids,_=reuse.request_tokens(tokenizer,passages,q["question"])
            entry={"task":task["task"],"question_index":i,"prompt_tokens":len(ids)}
            if longest is None or len(ids)>longest["prompt_tokens"]:
                longest=entry
            if len(ids)+reuse.MAX_NEW>reuse.MAX_LEN:
                too_long.append(entry)
            count+=1
    return {"status":"failed" if too_long else "passed","n":count,"max_prompt":longest,
            "overflow":too_long,"context_limit":reuse.MAX_LEN,"answer_reserve":reuse.MAX_NEW}


def data():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Task 19 data must run on CPU")
    verify_locked()
    require_new(DATA,SMALL/"data_preflight.json")
    DATA.mkdir()
    index=reuse.sums(PROJECT/"SHA256SUMS")
    for p in (RAW,TOK):
        verify(p,index[str(p.relative_to(PROJECT))])
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(TOK))
    ntok=lambda s:len(tok.encode(s,add_special_tokens=False).ids)
    dev=list(reuse.rows(RAW))
    corpus,bm25,keys=build_corpus(dev,ntok)
    need(len(corpus)==21100 and len(dev)==2417,"dev corpus size differs from lock")
    frozen=list(reuse.rows(DEPOT/"test_tasks.jsonl"))
    tasks=[normalize(a) for a in frozen]
    need(len(tasks)==451 and len({q for t in tasks for q in t["example_ids"]})==1804,"locked test coverage")
    need(len({t["task"] for t in tasks})==451,"duplicate test task")
    step_rows=list(reuse.rows(DEPOT/"steps.jsonl"))
    expected=[dict(s,task=t["task"],step=i) for t in frozen for i,s in enumerate(t["steps"])]
    need(step_rows==expected,"locked step file differs from test tasks")
    inspected=verify_gold([tasks[i] for i in (0,len(tasks)//2,len(tasks)-1)],{q["id"]:q for q in dev},corpus,keys)
    corp=jsonl(DATA/"corpus_dev.jsonl",(dict(c,pid=i) for i,c in enumerate(corpus)))
    norm=jsonl(DATA/"tasks.jsonl",tasks)
    bm25.save(str(DATA/"bm25"))
    print("rebuilt",len(corpus),"paragraphs; audited",sum(a["steps"] for a in inspected),"gold hops",flush=True)
    lengths=full_length_preflight(tasks,corpus,reader_tokenizer())
    reuse.write_new(SMALL/"context_limit_preflight.json",lengths)
    print("no_compress context preflight",lengths["n"],lengths["max_prompt"],"overflow",len(lengths["overflow"]),flush=True)
    need(lengths["status"]=="passed",
         f"NEED: {len(lengths['overflow'])} locked no_compress reader requests exceed the frozen context limit")
    reuse.write_new(SMALL/"data_preflight.json",{
        "status":"passed","n_tasks":len(tasks),"n_questions":1804,"n_steps":len(expected),"n_paragraphs":len(corpus),
        "budgets":BUDGETS,"gold_audits":inspected,"source":{"raw":record(RAW),"tokenizer":record(TOK)},
        "corp":corp,"tasks":norm,"reader_context_limit":record(SMALL/"context_limit_preflight.json"),
        "bm25_index":[record(p) for p in sorted((DATA/"bm25").iterdir()) if p.is_file()],
    })


def load_test():
    info=read(SMALL/"data_preflight.json")
    need(info["status"]=="passed","missing Task 19 data proof")
    for k in ("corp","tasks"):
        verify(Path(info[k]["path"]),info[k]["sha256"])
    corpus=list(reuse.rows(DATA/"corpus_dev.jsonl"))
    tasks={}
    for row in reuse.rows(DATA/"tasks.jsonl"):
        row["all_gold"]=set(row["all_gold"])
        for q in row["metadata"]:
            q["gold"]=set(q["gold"])
        tasks[row["task"]]=row
    need(len(tasks)==info["n_tasks"] and len(corpus)==info["n_paragraphs"],"data no longer matches receipts")
    return corpus,tasks


def batches(stream, ids):
    from itertools import groupby
    for task,(key,block) in zip(ids,groupby(stream,key=lambda x:x["task"]),strict=True):
        batch=list(block)
        need(key==task and [(r["strategy"],r["mode"],r["tier"],r["question_index"]) for r in batch]==list(ORDER),
             f"unexpected reader batch at task {task}")
        yield task,batch


def requests(finals,tasks,corpus,tokenizer):
    ordinary=[SimpleNamespace(name=s) for s in NONSHARED]
    for task,row in tasks.items():
        # The shared context generated by the existing renderer also provides no_context.
        paired=list(reuse.make_requests([SimpleNamespace(name="no_compress")],finals,{task:row},corpus,tokenizer,True))
        need(len(paired)==8,"shared reader rows must have four requests each")
        yield from paired
        yield from reuse.make_requests(ordinary,finals,{task:row},corpus,tokenizer,False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","final-lock"))
    args=parser.parse_args()
    if args.action=="prepare":
        data()
    else:
        reuse.write_new(SMALL/"lock_final.json",verify_locked())
