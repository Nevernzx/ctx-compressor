"""Run the signed held-out reader requests with cascade attention disabled."""
from collections import Counter
from importlib.metadata import version
from itertools import groupby
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import reader_diag_cascade as diag
from scripts import run_reader_heldout as reader
from src import reader_heldout_inputs as inputs

WORK = reader.ROOT / "reader_heldout_v2"
SMALL = PROJECT / "eval/reader_heldout_v2"
SHORT = reader.ROOT / "rt/g16"
R12 = reader.STANDARD / "input_preflight/requests.jsonl"
R13 = diag.R3_SPOOL
SMOKE = diag.SMOKE
M12 = PROJECT / "eval/reader_heldout/TASK12_SHA256SUMS"
M13 = PROJECT / "eval/grpo_r3/TASK13_SHA256SUMS"
HIST_CONFIG = PROJECT / "eval/reader/run_config.json"
PRE = WORK / "input_preflight.json"

STRATEGIES = ("latest", "bm25", "oracle", "sft_v2:checkpoint-2000", "grpo_r1:checkpoint-500",
              "h1:tau_star", "sft_v3:checkpoint-3000", "grpo_r2:checkpoint-125")
BASE_ORDER = [("no_compress", None, None, q) for q in range(4)] + [
    (s,m,t,q) for s in STRATEGIES for m,t in inputs.CELLS for q in range(4)
] + [("no_context", None, None, q) for q in range(4)]


def main_batches(historical_rows, r3_rows):
    """Yield the original ordered 200 followed by the 24 original r3 records."""
    original = groupby(historical_rows, key=lambda r: r["task"])
    r3 = diag.task_groups(r3_rows, diag.R3)
    seen = set()
    for (task, rows), (other, extra) in zip(original, r3, strict=True):
        batch = list(rows)
        inputs.need(task == other and task not in seen, "different or repeated task")
        seen.add(task)
        actual = [(r["strategy"],r["mode"],r["tier"],r["question_index"]) for r in batch]
        inputs.need(actual == BASE_ORDER, "signed 200-row block order differs")
        for r in batch:
            inputs.need(r["prompt_tokens"] == len(r["token_ids"]) > 0, "wrong prompt length")
            inputs.check_length(r["prompt_tokens"], (r["strategy"], task, r["question_index"]))
        yield task, batch + extra


def verify_sources():
    index = inputs.sums(PROJECT / "SHA256SUMS")
    result = {}
    for manifest, paths in ((M12,(R12,SMOKE,reader.STANDARD/"input_preflight/preflight.json")),
                            (M13,(R13,diag.TRY2/"input_preflight/preflight.json"))):
        result[str(manifest)] = diag.verify_source(manifest, index)
        signed = inputs.sums(manifest)
        for path in paths:
            result[str(path)] = diag.verify_source(path, signed)
    result[str(HIST_CONFIG)] = diag.verify_source(HIST_CONFIG, index)
    return result


def prepare():
    inputs.need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preflight required")
    inputs.need(WORK == WORK.resolve() and PROJECT == Path("/root/workplace/ctx-compressor"), "wrong project/output")
    reader.require_new([PRE])
    WORK.mkdir(exist_ok=True)
    opening = reader.original.root_df()
    signatures = verify_sources()
    base = reader.read(reader.STANDARD/"input_preflight/preflight.json")
    r3 = reader.read(diag.TRY2/"input_preflight/preflight.json")
    for info, path, n in ((base,R12,49800),(r3,R13,5976)):
        inputs.need(info["status"] == "passed" and info["requests"]["path"] == str(path) and
                    info["requests"]["sha256"] == signatures[str(path)] and info["requests"]["n"] == n,
                    "signed preflight and spool differ")
    inputs.need(base["smoke"]["n"] == 40 and base["smoke"]["sha256"] == signatures[str(SMOKE)] and
                base["no_compress"]["all_identical"], "historical smoke/singleton inputs differ")
    tasks, counts = [], Counter()
    maximum = 0
    for task, batch in main_batches(inputs.rows(R12), inputs.rows(R13)):
        tasks.append(task)
        counts.update(r["strategy"] for r in batch)
        maximum = max(maximum,max(r["prompt_tokens"] for r in batch))
    expected = {s:5976 for s in (*STRATEGIES,diag.R3)} | {"no_context":996, "no_compress":996}
    inputs.need(tasks == list(range(249)) and dict(counts) == expected and sum(counts.values()) == 55776,
                "incomplete task or strategy coverage")
    proof = dict(status="passed", source_sha256=signatures, task_order=tasks, batch_size=224,
                 n_tasks=len(tasks), n_predictions=sum(counts.values()), strategy_counts=dict(counts),
                 max_prompt_tokens=maximum, input_truncations=0, df_opening=opening,
                 requests_12=base["requests"], requests_13=r3["requests"], smoke=base["smoke"])
    inputs.write_new(PRE,proof)
    print("Preflight passed: 249 tasks, 55,776 requests, max prompt tokens",maximum,flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["prepare"]:
    prepare()
    raise SystemExit(0)

def checked_preflight():
    p = reader.read(PRE)
    inputs.need(p["status"] == "passed" and p["n_predictions"] == 55776 and
                p["batch_size"] == 224 and p["task_order"] == list(range(249)) and
                p["input_truncations"] == 0 and p["source_sha256"] == verify_sources(),
                "signed input preflight no longer holds")
    return p


def check_suite():
    tests = reader.read(SMALL / "cpu_tests.json")
    inputs.need(tests.get("status") == "passed" and tests["tests"] > 0 and
                tests.get("passed", tests["tests"]) == tests["tests"] and
                all(tests[k] == 0 for k in ("failures", "errors", "skipped")), "CPU suite failed")


def require_effective_cascade_off(value, source):
    inputs.need(value is True and source.startswith("llm_engine."),
                "effective disable_cascade_attn must be True in the engine configuration")


def deployable_ipc():
    proof = diag.ipc_record(SHORT)
    inputs.need(os.environ.get("VLLM_RPC_BASE_PATH", str(SHORT)) == str(SHORT), "unexpected IPC base")
    return proof


def run_engine(directory, config_path, batch_iter, expected_tasks, expected_predictions,
               *, smoke=True, provenance=None):
    """Use one admitted process and engine for the specified task-ordered batches."""
    inputs.need(directory.resolve() == directory and
                (directory == WORK or directory == WORK / "deepseek_compressor"),
                "unexpected inference directory")
    reader.require_new([directory/"resources.jsonl", directory/"predictions.jsonl", config_path,
                        directory/"receipt_inference.json"])
    if smoke:
        reader.require_new([directory/"smoke_predictions.jsonl", directory/"receipt_smoke.json"])
    inputs.need(reader.read(directory/"receipt_gate.json")["gpu"]["index"] == 1,
                "GPU 1 gate receipt missing")
    os.environ["READER_HELDOUT_BOOTSTRAP"] = str(directory)
    reader.reader_environment(directory, [])
    inputs.need(all(os.environ[k] == str(SHORT) for k in ("TMPDIR","TMP","TEMP","RAY_TMPDIR")) and
                Path(tempfile.gettempdir()) == SHORT and reader.read(directory/"ipc_check.json") == deployable_ipc(),
                "unexpected temporary or IPC environment")
    index = inputs.sums(PROJECT / "SHA256SUMS")
    config_digest = diag.verify_source(HIST_CONFIG, index)
    historical = reader.read(HIST_CONFIG)
    params = historical["parameters"].copy()
    gen = historical["generation"]
    inputs.need("disable_cascade_attn" not in params and params["seed"] == gen["seed"] == 20261117 and
                gen["max_tokens"] == 48 and gen["temperature"] == 0 and gen["enable_thinking"] is False,
                "historical reader protocol differs")
    params["disable_cascade_attn"] = True
    versions = {name:version(name) for name in historical["package_versions"]}
    inputs.need(versions == historical["package_versions"], "reader package versions differ")
    cfg = dict(status="running",pid=os.getpid(),started_utc=reader.now(), parameters=params, generation=gen,
               source_sha256=config_digest, package_versions=versions, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
               cuda_device_order=os.environ["CUDA_DEVICE_ORDER"], before=reader.original.root_df(),
               input_provenance=provenance, automatic_retry=False)
    monitor = reader.Monitor(directory)
    started = time.monotonic()
    llm = None
    monitor.start()
    n, tasks = 0, set()
    try:
        from vllm import LLM, SamplingParams
        inputs.need(reader.gpu()["memory_used_mib"] < 2048, "GPU 1 occupied after admission")
        reader.publish(directory, {"status":"running", "stage":"engine"})
        llm = LLM(**params)
        actual, source, line = diag.effective_cascade(llm.llm_engine, directory/"launcher.log")
        cfg.update(effective_disable_cascade_attn=actual,effective_disable_cascade_attn_source=source)
        if line:
            cfg["effective_disable_cascade_attn_log_excerpt"] = line
        try:
            require_effective_cascade_off(actual,source)
        except ValueError:
            cfg.update(status="failed_effective_flag",ended_utc=reader.now(),predictions=0)
            inputs.write_new(config_path,cfg)
            raise
        sampling = SamplingParams(n=1,temperature=gen["temperature"],max_tokens=gen["max_tokens"],seed=gen["seed"])
        cfg["sampling_params_repr"] = repr(sampling)
        if smoke:
            reader.publish(directory, {"status":"running", "stage":"smoke"})
            inputs.verify(SMOKE,inputs.sums(M12)[str(SMOKE)])
            fixtures = list(inputs.rows(SMOKE))
            inputs.need(len(fixtures) == 40,"historical smoke size changed")
            observed = reader.generate(llm,sampling,fixtures,monitor)
            mismatches = 0
            smoke_path = directory/"smoke_predictions.jsonl"
            with smoke_path.open("x") as f:
                for predicted,fixture in zip(observed,fixtures,strict=True):
                    mismatches += predicted["raw_output"] != fixture["expected_raw_output"]
                    print(reader.original.compact(dict(predicted,expected_raw_output=fixture["expected_raw_output"])),file=f)
            cfg["smoke"] = {"n":len(fixtures),"raw_output_mismatches":mismatches}
            reader.receipts_write(directory,"smoke",**cfg["smoke"],predictions_sha256=reader.digest(smoke_path))
        archive = directory/"predictions.jsonl"
        inference_start = time.monotonic()
        with archive.open("x") as f:
            for task,prompts in batch_iter:
                inputs.need(task not in tasks and prompts and all(r["task"]==task for r in prompts),
                            "duplicate task or mixed task batch")
                for r in prompts:
                    inputs.need(len(r["token_ids"]) == r["prompt_tokens"] > 0,"input token count differs")
                    inputs.check_length(r["prompt_tokens"],(task,r["question_index"]))
                observed = reader.generate(llm,sampling,prompts,monitor)
                inputs.need(len(observed) == len(prompts),"incomplete engine batch")
                for r in observed:
                    print(reader.original.compact(r),file=f)
                tasks.add(task)
                n += len(observed)
                f.flush()
                if len(tasks) == 1 or len(tasks)%5 == 0:
                    reader.publish(directory,dict(status="running",stage="inference",tasks=len(tasks),predictions=n))
        inputs.need(len(tasks)==expected_tasks and n==expected_predictions,"incomplete run")
        cfg["inference_seconds"] = time.monotonic()-inference_start
    finally:
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
        finally:
            monitor.stop()
    cfg.update(status="complete",ended_utc=reader.now(),elapsed_seconds=time.monotonic()-started,
               predictions=n,tasks=len(tasks),engine_shutdown=True,peak_gpu1_mib=monitor.peak,
               n_resource_samples=monitor.samples,df_after_reader=reader.original.root_df(),
               predictions_sha256=reader.digest(archive))
    cfg["peak_observed_root_used_bytes"] = max(row["root"]["used_bytes"] for row in inputs.rows(directory/"resources.jsonl"))
    cfg["peak_observed_root_delta_bytes"] = max(0,cfg["peak_observed_root_used_bytes"]-cfg["before"]["used_bytes"])
    if directory == WORK:
        cfg["df_opening"] = reader.read(PRE)["df_opening"]
        cfg["peak_observed_root_delta_bytes"] = max(0,cfg["peak_observed_root_used_bytes"]-cfg["df_opening"]["used_bytes"])
    inputs.write_new(config_path,cfg)
    reader.receipts_write(directory,"inference",n=n,run_config_sha256=reader.digest(config_path),
                          predictions_sha256=cfg["predictions_sha256"])
    return cfg


def detach_process(directory, script, argv):
    inputs.need(directory == directory.resolve() and
                (directory == WORK or directory == WORK/"deepseek_compressor"), "unexpected detached directory")
    reader.require_new([directory/"ipc_check.json",directory/"launcher.log"])
    SHORT.mkdir(parents=True,exist_ok=True)
    proof = deployable_ipc()
    print(f"IPC sample {proof['sample']}: path={proof['path_bytes']} bytes, URI={proof['uri_bytes']} bytes (<100)",flush=True)
    inputs.write_new(directory/"ipc_check.json",proof)
    shell = ('source scripts/reader_environment.sh && export TMPDIR="$1" TMP="$1" TEMP="$1" RAY_TMPDIR="$1" '
             'CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 && shift && exec "$READER_ENV/bin/python" -B -u "$@"')
    with (directory/"launcher.log").open("xb") as log:
        proc = subprocess.Popen(["/bin/bash","-c",shell,"reader-v2",str(SHORT),str(script),*argv],
                                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                close_fds=True,start_new_session=True)
    return proc.pid


def detach():
    reader.require_new([WORK/f for f in ("pipeline.pid","launcher.log","ipc_check.json","events.jsonl","status.json",
                                         "status.next","resources.jsonl","predictions.jsonl","smoke_predictions.jsonl",
                                         "receipt_preflight.json","receipt_gate.json","receipt_smoke.json","receipt_inference.json")])
    reader.require_new([SMALL/"run_config.json"])
    check_suite()
    checked_preflight()
    pid = detach_process(WORK,Path(__file__).resolve(),["pipeline"])
    print(f"detached PID {pid} -> {WORK}",flush=True)


def pipeline():
    inputs.need(os.getpid()==os.getsid(0) and Path(sys.prefix).resolve()==reader.ROOT/"envs/reader" and
                os.environ.get("CUDA_VISIBLE_DEVICES")=="1" and os.environ.get("CUDA_DEVICE_ORDER")=="PCI_BUS_ID",
                "expected a new detached session in the reader environment on GPU 1")
    with (WORK/"pipeline.pid").open("x") as f:
        print(os.getpid(),file=f)
    try:
        reader.publish(WORK,dict(stage="preflight",status="running",automatic_retry=False))
        check_suite()
        prep = checked_preflight()
        reader.receipts_write(WORK,"preflight",sha256=reader.digest(PRE),n=prep["n_predictions"],
                              tests_sha256=reader.digest(SMALL/"cpu_tests.json"))
        gpu = reader.wait_for_gpu("reader_heldout_v2",lambda v:reader.publish(WORK,v))
        reader.receipts_write(WORK,"gate",gpu=gpu)
        requests = main_batches(inputs.rows(R12),inputs.rows(R13))
        cfg = run_engine(WORK,SMALL/"run_config.json",requests,249,55776,
                         provenance={"path":str(PRE),"sha256":reader.digest(PRE),"batch_size":224})
        reader.publish(WORK,dict(stage="finished",status="complete",predictions=cfg["predictions"],automatic_retry=False))
    except BaseException as exc:
        reader.publish(WORK,dict(status="failed",error=repr(exc),automatic_retry=False))
        raise


if __name__ == "__main__":
    inputs.need(Path.cwd()==PROJECT,"run from the project root")
    if sys.argv[1:] == ["detach"]:
        detach()
    elif sys.argv[1:] == ["pipeline"]:
        pipeline()
    else:
        raise SystemExit("expected prepare, detach or internal pipeline action")

