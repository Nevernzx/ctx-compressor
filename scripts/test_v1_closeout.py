"""Finish the CPU-only Task 19 closeout once the GPU supervisor has exited."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import test_v1_data as data, test_v1_seal as seal
from scripts.run_reader_heldout import publish
from src.reranker_v1 import need, read, require_new

W=seal.SYNC
PID=W/"pid"
STATUS=W/"status.json"
CPU=PROJECT/".conda/envs/ctxc/bin/python"
PLOT=data.E/"envs/grpo/bin/python"


def environment():
    env={k:v for k,v in os.environ.items() if k.lower() not in
         ("http_proxy","https_proxy","all_proxy","deepseek_api_key")}
    env.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1",MPLCONFIGDIR=str(W/"mpl"),OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4",NO_PROXY="*",no_proxy="*")
    return env


def worker():
    need(os.getpid()==os.getsid(0),"Task 19 closeout controller must be detached")
    with PID.open("x") as out:
        print(os.getpid(),file=out)
    publish(W,{"stage":"gpu_receipts","status":"waiting","task":19,"automatic_retry":False})
    try:
        while True:
            state=read(data.WORK/"status.json")
            if state["status"]=="evaluation_complete":
                if not Path(f"/proc/{state['pid']}").exists():
                    break
            elif state["status"]=="failed" or not Path(f"/proc/{state['pid']}").exists():
                raise RuntimeError(f"GPU workflow did not complete: {state}")
            time.sleep(30)
        phases=(
            ("metrics",CPU,"test_v1_metrics.py","score"),
            ("lock_final",CPU,"test_v1_data.py","final-lock"),
            ("report",PLOT,"test_v1_report.py"),
            ("seal",CPU,"test_v1_seal.py","seal"),
            ("verify",CPU,"test_v1_seal.py","verify"),
        )
        for stage,python,name,*args in phases:
            log=W/f"{stage}.log"
            publish(W,{"stage":stage,"status":"running","task":19,"log":str(log),"automatic_retry":False})
            with log.open("xb") as f:
                subprocess.run([str(python),"-B","-u",str(PROJECT/"scripts"/name),*args],
                                env=environment(),cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                                check=True)
        publish(W,{"stage":"all","status":"complete","task":19,"automatic_retry":False})
    except Exception as e:
        publish(W,{"stage":"terminal","status":"failed","task":19,"automatic_retry":False,"error":repr(e)})
        raise


def detach():
    need(read(W/"history_before.json")["18"]["rehashed"]>0,"Task 19 historical preflight missing")
    require_new(PID,STATUS,W/"status.next",W/"events.jsonl",W/"launch.log")
    with (W/"launch.log").open("xb") as log:
        job=subprocess.Popen([str(CPU),"-B","-u",str(Path(__file__).resolve()),"worker"],cwd=PROJECT,env=environment(),
                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print("Task 19 CPU closeout detached",job.pid,STATUS,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("detach","worker"))
    args=p.parse_args()
    need(Path.cwd()==PROJECT and os.environ.get("CUDA_VISIBLE_DEVICES")=="","run closeout on Task 19 CPU")
    (detach if args.action=="detach" else worker)()
