"""Build and run the exact 200-request-per-task batches for Task 19."""
import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import reader_heldout_v2 as v16
from scripts import test_v1_assets as assets, test_v1_data as data, test_v1_rollout as rollout
from src import reader_heldout_inputs as reuse
from src.reranker_v1 import need, read, record, require_new, verify

W, SMALL=data.WORK,data.SMALL
READER=W/"reader"
SHORT=data.E/"rt/t19"
PREFLIGHT=READER/"input_preflight.json"
CONFIG=SMALL/"reader_run_config.json"
N_TASKS=451
BATCH=200
N_PREDICTIONS=N_TASKS*BATCH


def final_contexts(corpus, tasks):
    finals, references = {}, {}
    counts={}
    need(len(tasks)==N_TASKS, "test task count differs")
    for name in ("no_compress",*data.NONSHARED):
        directory=rollout.LOOPS/name
        metrics,steps=directory/"metrics.json",directory/"steps.jsonl"
        digest=reuse.checksum(metrics)
        metric=read(metrics)
        trace=metric["steps_artifact"]
        needed=sum(len(task["steps"]) for task in tasks.values())*len(reuse.CELLS)
        need(trace["path"]==str(steps) and metric["status"]=="complete" and metric["strategy"]==name
             and metric["settings"]["split"]==rollout.MODE and metric["settings"]["budgets"]==data.BUDGETS
             and metric["combined"]["n_tasks"]==N_TASKS*6 and metric["combined"]["n_steps"]==needed,
             f"locked rollout provenance differs: {name}")
        src=reuse.Source(name,metrics,digest,steps,trace["sha256"],"Task 19 frozen retrieval")
        finals[name], costs=reuse.replay(src,tasks,[p["ntok"] for p in corpus])
        counts[name]=costs
        references[name]={"steps":trace,"metrics":record(metrics)}
    for (cell,task),p in finals["no_compress"].items():
        need(p==finals["no_compress"][reuse.CELLS[0],task],"shared no_compress final differs by configuration")
    return finals, references, counts


def prepare():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","prepare reader requests on CPU")
    require_new(READER)
    df_opening=v16.reader.original.root_df()
    READER.mkdir()
    corpus,tasks=data.load_test()
    finals,refs,costs=final_contexts(corpus,tasks)
    # Retain ordered final contexts for independent evidence and cost verification.
    contexts=data.jsonl(READER/"final_contexts.jsonl",(
        {"strategy":arm,"mode":mode,"tier":tier,"task":task,"kept":ids}
        for arm, final in finals.items() for ((mode,tier),task),ids in final.items()
    ))
    tok=data.reader_tokenizer()
    ordered=data.batches(data.requests(finals,tasks,corpus,tok),list(tasks))
    packet=reuse.export_requests(READER/"requests.jsonl",(r for _,batch in ordered for r in batch))
    need(packet["n"]==N_PREDICTIONS and set(packet["n_by_config"].values())=={N_TASKS*4}
         and len(packet["n_by_config"])==50,"reader spool differs from registered rows")
    reuse.write_new(PREFLIGHT,{"status":"passed","n_tasks":N_TASKS,"requests":packet,"batch_size":BATCH,
                              "strategies":list(data.NAMES),"tasks":list(tasks),"final_contexts":contexts,
                              "sources":refs,"costs_recomputed":costs,"df_opening":df_opening})
    print("prepared",packet["n"],"requests",flush=True)


def run():
    pre=read(PREFLIGHT)
    need(isinstance(pre.get("df_opening"),dict) and type(pre["df_opening"].get("used_bytes")) is int,
         "reader preflight is missing its df_opening disk snapshot")
    input_p=READER/"requests.jsonl"
    need(pre["status"]=="passed" and pre["requests"]["path"]==str(input_p)
         and pre["requests"]["n"]==N_PREDICTIONS and pre["n_tasks"]==N_TASKS
         and pre["strategies"]==list(data.NAMES),"test reader preflight differs")
    verify(input_p,pre["requests"]["sha256"])
    ordered=data.batches(reuse.rows(input_p),pre["tasks"])
    with patch.multiple(v16,WORK=READER,PRE=PREFLIGHT,SHORT=SHORT):
        config=v16.run_engine(READER,CONFIG,ordered,N_TASKS,N_PREDICTIONS,smoke=True,
                              provenance={"preflight":record(PREFLIGHT),"requests":pre["requests"],
                                          "task":19,"batch_size":BATCH,"order":list(data.NAMES)})
    fixed=assets.load()["reader"]
    need(config["parameters"]==fixed["engine"] and config["generation"]==fixed["generation"]
         and config["effective_disable_cascade_attn"] is True and config["engine_shutdown"],
         "reader flags, package or shutdown differ")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","run"))
    args=p.parse_args()
    (prepare if args.action=="prepare" else run)()
