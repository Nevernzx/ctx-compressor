"""Detach Task 17, retain the joint lock, gate GPU 1 and verify every stage receipt."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import reranker_v1_data as data, reranker_v1_rollouts as r, train_reranker_v1 as fit
from scripts import run_reader_heldout as reader
from scripts.reader_diag_cascade import ipc_record
from scripts.sft_v3_wait_gpu import wait_for_gpu
from src.reader_heldout_inputs import rows, write_new
from src.reranker_v1 import LOCK, MODEL, SMALL, WORK, PairTokenizer, need, read, record, require_new, signed_model, signed_project, verify

CPU_PY = PROJECT/".conda/envs/ctxc/bin/python"
READER_PY = reader.PYTHON
PID = WORK/"pipeline.pid"
POLL=30
COMPRESSOR_LATENCY = {"latency-eager", "latency-graph"}
GRPO_ENVIRONMENT = PROJECT / "scripts/grpo_r2_environment.sh"


def python_environment(stage,gpu):
    cache=WORK/"runtime_cache"/stage
    cache.mkdir(parents=True,exist_ok=True)
    r.SHORT.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k.lower() not in ("http_proxy","https_proxy","all_proxy")}
    env.update(CUDA_VISIBLE_DEVICES="1" if gpu else "",CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTHONPATH=str(PROJECT),PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false",VLLM_WORKER_MULTIPROC_METHOD="spawn",VLLM_NO_USAGE_STATS="1",
               NO_PROXY="*",no_proxy="*",TMPDIR=str(r.SHORT),TMP=str(r.SHORT),TEMP=str(r.SHORT),RAY_TMPDIR=str(r.SHORT),
               OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",RAYON_NUM_THREADS="4")
    for name,entry in {
        "XDG_CACHE_HOME":"xdg","XDG_CONFIG_HOME":"config","XDG_DATA_HOME":"data","XDG_STATE_HOME":"state",
        "HF_HOME":"hf","HF_HUB_CACHE":"hf/hub","HUGGINGFACE_HUB_CACHE":"hf/hub","HF_DATASETS_CACHE":"hf/datasets",
        "MODELSCOPE_CACHE":"modelscope","TORCH_HOME":"torch","TORCH_EXTENSIONS_DIR":"torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR":"torchinductor","VLLM_CACHE_ROOT":"vllm","VLLM_CONFIG_ROOT":"vllm_config",
        "FLASHINFER_WORKSPACE_BASE":"flashinfer","TRITON_CACHE_DIR":"triton","CUDA_CACHE_PATH":"cuda",
        "CCACHE_DIR":"ccache","NUMBA_CACHE_DIR":"numba","CUPY_CACHE_DIR":"cupy","MPLCONFIGDIR":"matplotlib",
    }.items():
        target=cache/entry
        target.mkdir(parents=True,exist_ok=True)
        env[name]=str(target)
    env.pop("VLLM_RPC_BASE_PATH",None)
    return env


def compressor_environment(stage):
    """Load the F-stage toolchain with a Task 17 cache and CUDA hidden for preflight."""
    env_hash = signed_project(GRPO_ENVIRONMENT)
    initial = {k: v for k, v in os.environ.items()
               if k not in ("BASH_ENV", "ENV", "GRPO_R2_SMOKE_ATTEMPT")}
    initial.update(GRPO_GPU="1", GRPO_CPU_ONLY="1", CUDA_VISIBLE_DEVICES="")
    # The signed script hardcodes Task 11's cache. Defer mkdir until its paths are rebased.
    shell = 'mkdir() { :; }; source "$1"; unset -f mkdir; env -0'
    raw = subprocess.check_output(
        ["/bin/bash", "--noprofile", "--norc", "-e", "-c", shell, "task17", str(GRPO_ENVIRONMENT)],
        env=initial,
    )
    env = dict(os.fsdecode(x).split("=", 1) for x in raw.split(b"\0") if x)
    compiler = Path(env.get("CC", ""))
    need(compiler.is_absolute() and compiler.is_file() and os.access(compiler, os.X_OK),
         f"Task 17 {stage}: CC must point to an executable C compiler, got {env.get('CC')!r}")
    need(READER_PY.is_file() and os.access(READER_PY, os.X_OK), "missing historical F-stage Python")
    need(env["CUDA_VISIBLE_DEVICES"] == "" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID",
         "compiler preflight must hide CUDA")

    source_cache = Path(env["GRPO_CACHE"])
    runtime = WORK / "runtime_cache" / stage
    cache = runtime / "cache"
    short = r.SHORT
    ipc = ipc_record(short)
    need(cache.resolve() == cache and runtime.is_relative_to(WORK),
         "compressor latency cache redirected outside Task 17")
    runtime.mkdir(parents=True, exist_ok=True)
    short.mkdir(exist_ok=True)
    for key, value in env.items():
        if value == str(source_cache) or value.startswith(f"{source_cache}/"):
            path = cache / Path(value).relative_to(source_cache)
            need(path.resolve() == path, f"redirected compiler cache path: {path}")
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
    env.update(GRPO_PREP=str(runtime), PATH=os.pathsep.join((str(READER_PY.parent), env["PATH"])))
    for key in ("TMPDIR", "TMP", "TEMP", "RAY_TMPDIR", "VLLM_RPC_BASE_PATH"):
        env[key] = str(short)
    return env, {
        "status": "passed",
        "stage": stage,
        "python": str(READER_PY),
        "environment_script": {"path": str(GRPO_ENVIRONMENT), "sha256": env_hash},
        "CC": str(compiler),
        "CXX": env["CXX"],
        "TRITON_CACHE_DIR": env["TRITON_CACHE_DIR"],
        "cache_root": str(cache),
        "ipc": ipc,
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
    }

def weight_smoke():
    out=SMALL/"weight_smoke.json"
    require_new(out)
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","weight smoke is CPU-only")
    card=signed_model()
    split=data.verify_split()
    verify(data.TOKENS,data.checksum(SMALL/"tokenization.json"))
    tokenization=read(data.TOKENS)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer=PairTokenizer(AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False))
    need(tokenizer.config()==tokenization["template"],"tokenizer constants differ from encoded data")
    task=next(rows(data.DATA_OUT/"train.jsonl"))
    corpus=list(rows(data.DATA/"corpus_train.jsonl"))
    docs=[corpus[p]["text"] for p,_ in data.candidates(task["steps"])[:4]]
    enc=[ids for ids,_ in tokenizer.encode("\n".join(task["questions"]),docs)]
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False,
                                              dtype=torch.float32,attn_implementation="sdpa").eval()
    with torch.inference_mode():
        ids,mask=fit.tensor_batch(enc,tokenizer.config(),"cpu")
        full=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1).logits[:,-1,:]
        full=full[:,tokenizer.yes]-full[:,tokenizer.no]
        selected=fit.score_pairs(model,enc,tokenizer.config(),device="cpu")
        torch.testing.assert_close(full,selected,rtol=1e-5,atol=1e-4)
    need(not torch.cuda.is_initialized(),"CPU-only smoke initialized CUDA")
    write_new(out,{"status":"passed","input_snapshot_revision":card["revision"],"n_pairs":len(enc),
                   "max_abs_logit_difference":float((full-selected).abs().max()),
                   "selective_head_matches_full_head":True,"cuda_initialized":False,
                   "weights":record(MODEL/"model.safetensors"),"data_preflight":record(SMALL/"data_preflight.json"),
                   "tokenization":record(SMALL/"tokenization.json")})
    print("real model CPU smoke passed",len(enc),"pairs",flush=True)


def audit_ready():
    milestones={
        "download":SMALL/"model_card.json","split":SMALL/"data_preflight.json","tokens":SMALL/"tokenization.json",
        "bm25":SMALL/"bm25_action_check.json","latency_inputs":SMALL/"latency_input.json",
        "cpu_tests":SMALL/"cpu_tests.json","weight_cpu_smoke":SMALL/"weight_smoke.json",
    }
    for name,path in milestones.items():
        obj=read(path)
        if name=="download":
            need(obj["directory"]==str(MODEL) and obj["model"]=="Qwen/Qwen3-Reranker-0.6B","wrong model")
        else:
            need(obj["status"]=="passed",f"Task17 preflight failed: {name}")
    suite=read(milestones["cpu_tests"])
    need(suite["tests"]>0 and all(suite.get(k,0)==0 for k in ("errors","failures","skipped")) and
         not suite.get("cpu_guard_events"),"full CPU regressions are incomplete")
    need(WORK.stat().st_dev != Path("/").stat().st_dev and SMALL.resolve()==SMALL,"large files must be external")
    return {name:record(path) for name,path in milestones.items()}


def publish(stage,status,**detail):
    reader.publish(WORK,{"task":17,"stage":stage,"status":status,"automatic_retry":False,**detail})


def stage(name, script, args, outputs, *, gpu=False, reader_dir=None, expected=("complete",)):
    log_dir=WORK/"stages"/name
    require_new(log_dir)
    log_dir.mkdir()
    env = None
    if name in COMPRESSOR_LATENCY:
        env, proof = compressor_environment(name)
        write_new(log_dir / "receipt_environment.json", proof)
    admitted=None
    if gpu:
        admitted=wait_for_gpu(name,lambda reading: publish(name,"waiting_gpu",**{
            k:v for k,v in reading.items() if k not in ("stage","status")}))
        reader.receipts_write(log_dir,"gate",gpu=admitted)
    if env is None:
        env = python_environment(name, gpu)
    else:
        env["CUDA_VISIBLE_DEVICES"] = "1" if gpu else ""
    if reader_dir is not None:
        reader.receipts_write(reader_dir,"gate",gpu=admitted)
        write_new(reader_dir/"ipc_check.json",ipc_record(r.SHORT))
    python=READER_PY if gpu or name.startswith("prepare-") else CPU_PY
    cmd=[str(python),"-B","-u",str(PROJECT/"scripts"/script),*args]
    df=reader.original.root_df()
    start=time.monotonic()
    archive=(reader_dir or log_dir)/"launcher.log"
    with archive.open("xb") as stdout, (log_dir/"samples.jsonl").open("x",buffering=1) as samples:
        proc=subprocess.Popen(cmd,env=env,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=stdout,
                              stderr=subprocess.STDOUT,close_fds=True)
        publish(name,"running",child_pid=proc.pid,gpu=admitted,log=str(archive))
        while True:
            try:
                code=proc.wait(timeout=POLL)
                break
            except subprocess.TimeoutExpired:
                snapshot={"child_pid":proc.pid,"elapsed_seconds":time.monotonic()-start,
                          "root":reader.original.root_df()}
                if gpu:
                    snapshot["gpu"]=reader.gpu()
                if reader_dir is not None and (reader_dir/"status.json").is_file():
                    child=read(reader_dir/"status.json")
                    snapshot["reader_progress"]={k:child[k] for k in ("stage","status","predictions","tasks") if k in child}
                samples.write(data.compact(snapshot)+"\n")
                publish(name,"running",**snapshot)
    need(code==0,f"{name} exited {code}; see {archive}")
    facts={}
    for path in outputs:
        result=read(path)
        need(result["status"] in expected,f"{name} did not complete: {path}")
        facts[path.name]=record(path)
    reader.receipts_write(log_dir,"complete",gpu=gpu,exit_code=code,child_pid=proc.pid,outputs=facts,
                          wall_seconds=time.monotonic()-start,root_before=df,root_after=reader.original.root_df())
    publish(name,"stage_complete",child_pid=proc.pid,receipt=str(log_dir/"receipt_complete.json"))
    return read(log_dir/"receipt_complete.json")


def run_pipeline(order):
    need(os.getpid()==os.getsid(0) and os.environ.get("CUDA_VISIBLE_DEVICES")=="","expected detached CPU supervisor")
    with PID.open("x") as pid:
        pid.write(str(os.getpid())+"\n")
    try:
        receipts=audit_ready()
        LOCK.parent.mkdir(exist_ok=True)
        need(LOCK.resolve()==LOCK,"shared GPU lock path redirected")
        with LOCK.open("a+b") as handle:
            publish("queue","waiting_lock",lock=str(LOCK),requested_order=order)
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
            publish("queue","holding_lock",lock=str(LOCK),requested_order=order)
            (WORK/"stages").mkdir()
            opening=reader.original.root_df()
            results={}
            results["S1"]=stage("S1","train_reranker_v1.py",[],[SMALL/"training.json"],gpu=True)
            val_pre=r.VAL/"input_preflight.json"
            results["prepare_val"]=stage("prepare-val","reranker_v1_rollouts.py",["prepare-val"],[val_pre],expected=("passed",))
            results["reader_val"]=stage("reader-val","reranker_v1_rollouts.py",["reader-val"],
                 [SMALL/"val_reader_config.json"],gpu=True,reader_dir=r.VAL)
            results["tau"]=stage("select-tau","reranker_v1_rollouts.py",["select-tau"],[SMALL/"tau_selection.json"])
            held_pre=r.HELD/"input_preflight.json"
            results["prepare_heldout"]=stage("prepare-heldout","reranker_v1_rollouts.py",["prepare-heldout"],
                                               [held_pre],expected=("passed",))
            results["reader_heldout"]=stage("reader-heldout","reranker_v1_rollouts.py",["reader-heldout"],
                 [SMALL/"heldout_reader_config.json"],gpu=True,reader_dir=r.HELD)
            results["compare"]=stage("compare","reranker_v1_rollouts.py",["summarize"],[SMALL/"heldout_comparison.json"])
            for mode in ("eager","graph","reranker"):
                expected=("complete","load_failed") if mode=="graph" else ("complete",)
                results["latency_"+mode]=stage("latency-"+mode,"reranker_v1_latency.py",[mode],
                 [SMALL/f"latency_{mode}.json"],gpu=True,expected=expected)
            train=read(SMALL/"training.json")
            files=[p for p in SMALL.rglob("*") if p.is_file()]
            size=sum(p.stat().st_size for p in files)
            need(size<300_000_000,"task small artifacts exceed the root limit")
            config={"status":"complete","task":17,"reader_shares_v16_protocol":True,"template":train["template"],
                    "stages":results,"phase_p_artifacts":receipts,"shared_lock":str(LOCK),
                    "GPU":1,"requested_order":order,"automatic_retry":False,
                    "owned_small_root_bytes":size,"root_start":opening,"root_end":reader.original.root_df()}
            write_new(SMALL/"run_config.json",config)
            publish("all","complete",lock=str(LOCK),config=record(SMALL/"run_config.json"))
    except BaseException as exc:
        publish("terminal","failed",error=f"{type(exc).__name__}: {exc}")
        raise


def detach(order):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","launch with CUDA hidden")
    audit_ready()
    require_new(PID,WORK/"status.json",WORK/"status.next",WORK/"events.jsonl",WORK/"launcher.log",WORK/"stages")
    env=python_environment("controller",False)
    with (WORK/"launcher.log").open("xb") as stream:
        job=subprocess.Popen([str(CPU_PY),"-B","-u",str(Path(__file__).resolve()),"pipeline","--order",order],
                             cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                             start_new_session=True,close_fds=True)
    print(f"Task17 detached PID {job.pid}; status {WORK/'status.json'}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("weight-smoke","detach","pipeline"))
    p.add_argument("--order",choices=("after_task18","task18_phase_p_blocked"),default="after_task18")
    args=p.parse_args()
    need(Path.cwd()==PROJECT,"run from the project on 5880")
    if args.action=="weight-smoke":
        weight_smoke()
    elif args.action=="detach":
        detach(args.order)
    else:
        run_pipeline(args.order)


if __name__=="__main__":
    main()

