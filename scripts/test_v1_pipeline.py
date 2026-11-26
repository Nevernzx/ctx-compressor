"""Detach Task 19, hold its priority lock through G3, and verify each stage receipt."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_grpo_r4_pipeline as r4
from scripts import run_reranker_v1_pipeline as r17
from scripts import test_v1_assets as assets, test_v1_data as data, test_v1_reader as reader
from src import reader_heldout_inputs as reuse
from src.reranker_v1 import need, read, record, require_new, verify

W, OUT = data.WORK,data.SMALL
GPU_LOCK=data.E/"rt/gpu1_t19_t20.lock"
CPU=PROJECT/".conda/envs/ctxc/bin/python"
PY=data.E/"envs/reader/bin/python"
PHASE_P=OUT/"phase_p.json"
TESTS=OUT/"cpu_tests.json"
STATUS=W/"status.json"
PID=W/"pipeline.pid"
RUNTIME=W/"runtime"


def publish(stage,status,**fields):
    r17.reader.publish(W,{"task":19,"stage":stage,"status":status,"automatic_retry":False,**fields})


def environment(name,gpu):
    py=PY if gpu or name.startswith("prepare-") else CPU
    with patch.object(r4,"RUNTIME",RUNTIME),patch.object(r4.qa,"SHORT",reader.SHORT):
        setup=r4.worker_environment(name,py)
    inherited={k:v for k,v in os.environ.items() if k.lower() not in
               ("http_proxy","https_proxy","all_proxy","deepseek_api_key")}
    return {**inherited,**setup,"CUDA_VISIBLE_DEVICES":"1" if gpu else "","CUDA_DEVICE_ORDER":"PCI_BUS_ID"}


def phase_p():
    locks=data.verify_locked()
    a=assets.load()
    b=read(OUT/"data_preflight.json")
    rr=read(OUT/"rr_heldout_parity.json")
    suite=read(TESTS)
    baseline=read(OUT/"cpu_baselines.json")
    old_n=read(PROJECT/"eval/reranker_v1/full_close_tests.json")["tests"]
    need(suite["status"]=="passed" and suite["tests"]>=old_n+8 and suite["matched_task18_close_tests"]==old_n
         and not any(suite[k] for k in ("errors","failures","skipped","cpu_guard_events")),
         "full CPU tests incomplete")
    verify(Path(suite["xml"]),suite["xml_sha256"])
    need(a["reranker"]["tau"]==.5 and a["compressor"]["sampling"]["max_tokens"]==256
         and a["reader"]["engine"]["disable_cascade_attn"] is True,"frozen settings differ")
    need(b["n_tasks"]==baseline["n_tasks"]==451 and b["n_questions"]==1804 and b["n_paragraphs"]==21100
         and b["n_steps"]==4771 and b["budgets"]==data.BUDGETS
         and len(b["gold_audits"])==3 and all(x["gold_mismatches"]==0 for x in b["gold_audits"]),
         "locked corpus prep differs")
    need(rr["n_tasks"]==10 and set(rr["policies"])=={"RR-fill","RR-τ"}
         and all(o["n"]>0 and o["action_mismatches"]==0 for o in rr["policies"].values()),
         "reranker CPU regression differs")
    for f in baseline["rollout_metrics"].values():
        verify(Path(f["path"]),f["sha256"])
    names=("lock_preflight.json","model_preflight.json","data_preflight.json","context_limit_preflight.json",
           "rr_heldout_parity.json","cpu_baselines.json","cpu_tests.json")
    need(W.resolve()==W and OUT.resolve()==OUT and W.stat().st_dev!=Path("/").stat().st_dev,
         "Task 19 large artifacts require external filesystem")
    return {"status":"passed","locked_files":locks["files"],"inputs":{k:record(OUT/k) for k in names},
            "n_cpu_tests":suite["tests"],"priority":"task19_before_task20","shared_lock":str(GPU_LOCK)}


def worker():
    need(os.getpid()==os.getsid(0) and os.environ.get("CUDA_VISIBLE_DEVICES")=="","expected detached CPU controller")
    with PID.open("x") as f:
        print(os.getpid(),file=f)
    try:
        receipt=read(PHASE_P)
        need(receipt==phase_p(),"CPU preflight changed after launch")
        need(GPU_LOCK.resolve()==GPU_LOCK,"joint lock path redirected")
        with GPU_LOCK.open("a+b") as lock:
            publish("queue","waiting_lock",lock=str(GPU_LOCK),priority=receipt["priority"])
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            publish("queue","holding_lock",lock=str(GPU_LOCK),priority=receipt["priority"])
            (W/"stages").mkdir()
            opened=r17.reader.original.root_df()
            with patch.multiple(r17,WORK=W,READER_PY=PY,CPU_PY=CPU,publish=publish,python_environment=environment):
                steps={}
                steps["G0"]=r17.stage("G0","test_v1_rollout.py",["G0"],[OUT/"G0_smoke.json"],gpu=True,expected=("passed",))
                steps["G1"]=r17.stage("G1","test_v1_rollout.py",["score"],[OUT/"G1_scores.json"],gpu=True)
                steps["RR"]=r17.stage("RR","test_v1_rollout.py",["rr"],[OUT/"rr_rollouts.json"])
                for arm in ("v3","r2","r4"):
                    label=f"G2-{arm}"
                    steps[label]=r17.stage(label,"test_v1_rollout.py",[arm],[OUT/f"G2_{arm}.json"],gpu=True)
                steps["G3_prepare"]=r17.stage("prepare-reader","test_v1_reader.py",["prepare"],[reader.PREFLIGHT],expected=("passed",))
                with patch.object(r17.r,"SHORT",reader.SHORT):
                    steps["G3"]=r17.stage("G3","test_v1_reader.py",["run"],[reader.CONFIG],gpu=True,reader_dir=reader.READER)
            size=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
            need(size<300_000_000, "Task 19 small result budget exceeded")
            cfg=OUT/"run_config.json"
            reuse.write_new(cfg,{"status":"evaluation_complete","task":19,"locked_first_use":True,
                   "automatic_retry":False,"gpu":1,"shared_lock":str(GPU_LOCK),"priority":receipt["priority"],
                   "settings_frozen":record(assets.PRE),"budgets":data.BUDGETS,"strategy_order":list(data.NAMES),
                   "reader_batch_size":200,"phase_p":record(PHASE_P),"stages":steps,
                   "root_before":opened,"root_after":r17.reader.original.root_df(),"small_artifact_bytes":size})
        publish("all","evaluation_complete",config=record(cfg),lock_released=True)
    except BaseException as exc:
        publish("terminal","failed",error=f"{type(exc).__name__}: {exc}")
        raise


def detach():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","launch with CUDA hidden")
    require_new(PID,STATUS,W/"status.next",W/"events.jsonl",W/"launcher.log",W/"stages",PHASE_P)
    reuse.write_new(PHASE_P,phase_p())
    with (W/"launcher.log").open("xb") as log:
        job=subprocess.Popen([str(CPU),"-B","-u",str(Path(__file__).resolve()),"worker"],cwd=PROJECT,
            env=environment("controller",False),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    print("Task 19 detached",job.pid,STATUS,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("detach","worker"))
    args=parser.parse_args()
    (detach if args.action=="detach" else worker)()
