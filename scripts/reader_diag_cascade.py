"""Replay signed Qwen reader inputs in two preregistered batch configurations."""
from importlib.metadata import version
from itertools import groupby
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_reader_heldout as reader
from src import reader_heldout_inputs as inputs

WORK = reader.ROOT / "reader_diag"
SMALL = PROJECT / "eval/reader_diag"
HIST = reader.STANDARD
TRY2 = HIST / "grpo_r3_checkpoint-250_try2"
PREFLIGHT = WORK / "input_preflight"
R2_SPOOL = PREFLIGHT / "r2_requests.jsonl"
R3_SPOOL = TRY2 / "input_preflight/requests.jsonl"
SMOKE = HIST / "input_preflight/smoke.jsonl"
SHORT = reader.ROOT / "rt/g15"
R2 = "grpo_r2:checkpoint-125"
R3 = "grpo_r3:checkpoint-250"
ARMS = ("cascade_off_r3", "mixed_r2r3")
P_HASH = "5f584e73b5bcb48d6670da18951d02447d52f8064778f59d98fce11a35a88b32"
KEYS = ("strategy", "mode", "tier", "task", "question_index")


def identity(row):
    return tuple(row[k] for k in KEYS)


def checked_r2_rows(rebuilt, archived, predictions):
    """Compare the rebuilt order, prompt hash and token list before writing the new spool."""
    for row, reference, prediction in zip(rebuilt, archived, predictions, strict=True):
        inputs.need(identity(row) == identity(reference) == identity(prediction), "r2 key/order mismatch")
        inputs.need(row["prompt_sha1"] == prediction["prompt_sha1"] == reference["prompt_sha1"],
                    "r2 prompt_sha1 mismatch")
        inputs.need(row["token_ids"] == reference["token_ids"], "r2 prompt token identity mismatch")
        yield row


def task_groups(rows, strategy):
    seen = set()
    for task, stream in groupby(rows, key=lambda r: r["task"]):
        batch = list(stream)
        expected = [(strategy, m, tier, task, qi) for m, tier in inputs.CELLS for qi in range(4)]
        inputs.need(task not in seen and list(map(identity, batch)) == expected,
                    f"task order or cell/question batch mismatch at {task}")
        for r in batch:
            inputs.need(r["prompt_tokens"] == len(r["token_ids"]) and len(r["token_ids"]) > 0,
                        "unexpected prompt length")
            inputs.check_length(r["prompt_tokens"], identity(r))
        seen.add(task)
        yield task, batch


def batches(arm, r3_rows, r2_rows=None):
    if arm == ARMS[0] and r2_rows is None:
        yield from task_groups(r3_rows, R3)
    elif arm == ARMS[1] and r2_rows is not None:
        groups = zip(task_groups(r3_rows, R3), task_groups(r2_rows, R2), strict=True)
        for (task, r3), (r2_task, r2) in groups:
            inputs.need(task == r2_task, "strategies have different task order")
            yield task, r2 + r3
    else:
        raise ValueError("unexpected diagnostic arm/inputs")


def verify_source(path, manifest):
    name = str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path)
    inputs.verify(path, manifest[name])
    return manifest[name]


def prepare():
    inputs.need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preflight requires CUDA_VISIBLE_DEVICES empty")
    inputs.need(WORK.resolve() == WORK, "redirected output directory")
    PREFLIGHT.mkdir(parents=True, exist_ok=False)
    index = inputs.sums(PROJECT / "SHA256SUMS")
    m12 = PROJECT / "eval/reader_heldout/TASK12_SHA256SUMS"
    m13 = PROJECT / "eval/grpo_r3/TASK13_SHA256SUMS"
    for manifest in (m12, m13):
        verify_source(manifest, index)
    s12, s13 = inputs.sums(m12), inputs.sums(m13)
    pre_path = TRY2 / "input_preflight/preflight.json"
    receipt_path = TRY2 / "receipt_inputs.json"
    proof = {str(p): verify_source(p, s13) for p in (pre_path, receipt_path, R3_SPOOL)}
    receipt = reader.read(receipt_path)
    r3 = reader.read(pre_path)
    inputs.need(r3["status"] == "passed" and receipt["n"] == r3["requests"]["n"] == 5976 and
                receipt["sha256"] == proof[str(pre_path)] == P_HASH and
                r3["requests"]["path"] == str(R3_SPOOL) and r3["requests"]["sha256"] == proof[str(R3_SPOOL)],
                "try2 receipt/preflight/request chain mismatch")
    baseline_pre = HIST / "input_preflight/preflight.json"
    baseline_spool = HIST / "input_preflight/requests.jsonl"
    baseline_predictions = HIST / "predictions.jsonl"
    proof.update({str(p): verify_source(p, s12) for p in (baseline_pre, baseline_spool, baseline_predictions, SMOKE)})
    base = reader.read(baseline_pre)
    inputs.need(base["status"] == "passed" and base["smoke"]["sha256"] == proof[str(SMOKE)] and
                base["requests"]["sha256"] == proof[str(baseline_spool)] and base["smoke"]["n"] == 40,
                "historical input signature mismatch")
    from transformers import AutoTokenizer
    corpus, tasks, source_files = inputs.load_task_inputs(index)
    src, = [s for s in inputs.historical_sources(index) if s.name == R2]
    finals, costs = inputs.replay(src, tasks, [p["ntok"] for p in corpus])
    tokenizer = AutoTokenizer.from_pretrained(str(reader.ROOT / "models/Qwen3-8B"),
                                               local_files_only=True, trust_remote_code=False)
    def selected(path):
        return (r for r in inputs.rows(path) if r["strategy"] == R2)
    rebuilt = inputs.make_requests([src], {R2: finals}, tasks, corpus, tokenizer, False)
    exported = inputs.export_requests(R2_SPOOL, checked_r2_rows(rebuilt,selected(baseline_spool),
                                                                selected(baseline_predictions)))
    inputs.need(exported["n"] == 5976, "r2 coverage mismatch")
    task_order = []
    matching = 0
    for task, combined in batches(ARMS[1], inputs.rows(R3_SPOOL), inputs.rows(R2_SPOOL)):
        task_order.append(task)
        matching += sum(a["token_ids"] == b["token_ids"] for a,b in zip(combined[:24], combined[24:], strict=True))
    inputs.need(task_order == sorted(tasks) and len(task_order) == 249 and matching == 2116,
                "paired input coverage differs")
    hist_config = PROJECT / "eval/reader/run_config.json"
    try2_config = PROJECT / "eval/grpo_r3/reader_run_config.json"
    proof[str(hist_config)] = verify_source(hist_config, index)
    proof[str(try2_config)] = verify_source(try2_config, s13)
    a, b = reader.read(hist_config), reader.read(try2_config)
    inputs.need(all(a[k] == b[k] for k in ("parameters","generation","package_versions")),
                "historical reader configurations differ")
    facts = {"status":"passed","source_hashes":proof, "rollout_inputs":source_files,
             "r2_requests":exported, "r3_requests":r3["requests"], "smoke_requests":base["smoke"],
             "r2_historical_prompt_sha1_mismatches":0, "r2_historical_token_mismatches":0,
             "n_r2_identities":exported["n"],"task_order":task_order, "r2_r3_equal_tokens":matching,
             "r2_rollout_costs_checked":costs, "try2_receipt_sha256_field":receipt["sha256"],
             "hash_note":"The receipt's sha256 identifies preflight.json. Both the signed preflight and its signed requests.jsonl were independently verified."}
    inputs.write_new(PREFLIGHT / "preflight.json", facts)
    print(f"CPU preflight passed: n={exported['n']} per strategy, same input n={matching}", flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["prepare"]:
    prepare()
    raise SystemExit(0)




def ipc_record(short):
    sample = short / "00000000-0000-0000-0000-000000000000"
    path_bytes = len(os.fsencode(sample))
    uri_bytes = path_bytes + len(b"ipc://")
    inputs.need(uri_bytes < 100 and short == short.resolve(), "IPC path too long or redirected")
    return {"sample": str(sample), "path_bytes": path_bytes, "uri_bytes": uri_bytes, "limit_exclusive": 100}


def checked_inputs():
    proof = reader.read(PREFLIGHT / "preflight.json")
    inputs.need(proof["status"] == "passed" and proof["n_r2_identities"] == 5976 and
                proof["r2_historical_prompt_sha1_mismatches"] == proof["r2_historical_token_mismatches"] == 0 and
                proof["r2_r3_equal_tokens"] == 2116, "incomplete diagnostic preflight")
    for key,path in (("r2_requests", R2_SPOOL), ("r3_requests", R3_SPOOL), ("smoke_requests", SMOKE)):
        inputs.need(proof[key]["path"] == str(path), "unexpected source spool")
        inputs.verify(path,proof[key]["sha256"])
    return proof


def check_tests():
    tests = reader.read(SMALL / "cpu_tests.json")
    inputs.need(tests.get("status") == "passed" and int(tests["tests"]) > 0 and
                int(tests.get("passed", tests["tests"])) == int(tests["tests"]) and
                all(int(tests[k]) == 0 for k in ("errors", "failures", "skipped")),
                "full CPU suite did not pass")


def effective_cascade(engine, logfile):
    # Some vLLM builds expose this through the accessor rather than the attribute.
    config = getattr(engine, "vllm_config", None)
    where = "llm_engine.vllm_config.model_config.disable_cascade_attn"
    value = getattr(getattr(config, "model_config", None), "disable_cascade_attn", None)
    if not isinstance(value,bool) and callable(getattr(engine, "get_vllm_config", None)):
        config = engine.get_vllm_config()
        value = getattr(getattr(config, "model_config", None), "disable_cascade_attn", None)
        where = "llm_engine.get_vllm_config().model_config.disable_cascade_attn"
    if isinstance(value, bool):
        return value, where, None
    if logfile.is_file():
        matches = []
        for line in logfile.read_text(errors="replace").splitlines():
            if "config" not in line.lower() or "non-default args" in line.lower():
                continue
            match = re.search(r"""['"]?disable_cascade_attn['"]?\s*[:=]\s*(true|false)\b""", line, re.I)
            if match:
                lo, hi = match.span()
                matches.append((match.group(1).lower() == "true", line[max(0,lo-50):hi+50]))
        if matches and len({v for v,_ in matches}) == 1:
            return matches[-1][0], "engine startup config line", matches[-1][1]
    return None, "未取得", None


def arm_run(arm):
    directory = WORK / arm
    reader.require_new([directory / name for name in ("resources.jsonl","predictions.jsonl","smoke_predictions.jsonl",
                                                      "run_config.json","receipt_smoke.json","receipt_inference.json")])
    inputs.need(int((WORK / "pipeline.pid").read_text()) == os.getppid() and
                reader.read(WORK/"receipt_gate.json")["gpu"]["index"] == 1, "arm not owned by admitted pipeline")
    os.environ["READER_HELDOUT_BOOTSTRAP"] = str(directory)
    reader.reader_environment(directory, [])
    inputs.need(all(os.environ[k] == str(SHORT) for k in ("TMPDIR", "TMP", "TEMP", "RAY_TMPDIR")) and
                Path(tempfile.gettempdir()) == SHORT and deployable_ipc() == reader.read(WORK/"ipc_check.json"),
                "incorrect IPC environment")
    proof = checked_inputs()
    historic_path = PROJECT/"eval/reader/run_config.json"
    inputs.verify(historic_path, proof["source_hashes"][str(historic_path)])
    historic = reader.read(historic_path)
    params = dict(historic["parameters"])
    gen = historic["generation"]
    inputs.need("disable_cascade_attn" not in params and gen["temperature"] == 0 and not gen["enable_thinking"] and
                gen["max_tokens"] == 48 and params["seed"] == gen["seed"] == 20261117, "frozen reader settings differ")
    if arm == ARMS[0]:
        params["disable_cascade_attn"] = True
    versions = {k:version(k) for k in historic["package_versions"]}
    inputs.need(versions == historic["package_versions"], "reader versions differ from try2")
    cfg = {"arm":arm,"pid":os.getpid(),"status":"running","started_utc":reader.now(),"parameters":params,
           "generation":gen,"package_versions":versions,"source_sha256":reader.digest(historic_path),
           "preflight_sha256":reader.digest(PREFLIGHT/"preflight.json"),"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],
           "cuda_device_order":os.environ["CUDA_DEVICE_ORDER"],"df_before_reader":reader.original.root_df()}
    began = time.monotonic()
    monitor = reader.Monitor(directory)
    monitor.start()
    llm = None
    n = 0
    n_tasks = 0
    try:
        from vllm import LLM, SamplingParams
        inputs.need(reader.gpu()["memory_used_mib"] < 2048, "GPU 1 occupied after admission")
        reader.publish(directory, {"status":"running", "stage":"engine"})
        llm = LLM(**params)
        actual, where, line = effective_cascade(llm.llm_engine, directory/"launch.log")
        if actual is not None:
            inputs.need(actual == (arm == ARMS[0]), "requested cascade flag differs from effective value")
        cfg.update(effective_disable_cascade_attn=actual, effective_disable_cascade_attn_source=where,
                   effective_disable_cascade_attn_log_excerpt=line)
        sampling = SamplingParams(n=1, temperature=gen["temperature"], max_tokens=gen["max_tokens"], seed=gen["seed"])
        cfg["sampling_params_repr"] = repr(sampling)
        smoke_inputs = list(inputs.rows(SMOKE))
        smoke = reader.generate(llm,sampling,smoke_inputs,monitor)
        inputs.need(len(smoke) == 40, "incomplete smoke")
        mismatches = 0
        with (directory/"smoke_predictions.jsonl").open("x") as f:
            for row, src in zip(smoke, smoke_inputs, strict=True):
                mismatches += row["raw_output"] != src["expected_raw_output"]
                print(reader.original.compact(dict(row,expected_raw_output=src["expected_raw_output"])),file=f)
        cfg["smoke"] = {"n":len(smoke), "raw_output_mismatches":mismatches}
        reader.receipts_write(directory,"smoke",**cfg["smoke"],predictions_sha256=reader.digest(directory/"smoke_predictions.jsonl"))
        r2_rows = inputs.rows(R2_SPOOL) if arm == ARMS[1] else None
        infer_start = time.monotonic()
        with (directory/"predictions.jsonl").open("x") as f:
            for task, batch in batches(arm,inputs.rows(R3_SPOOL),r2_rows):
                inputs.need(task == proof["task_order"][n_tasks], "historical task order changed")
                for row in reader.generate(llm,sampling,batch,monitor):
                    print(reader.original.compact(row),file=f)
                n += len(batch)
                n_tasks += 1
                f.flush()
                if n_tasks == 1 or n_tasks % 5 == 0:
                    reader.publish(directory,{"status":"running","stage":"inference","tasks":n_tasks,"predictions":n})
        inputs.need(n_tasks == 249 and n == (5976 if arm == ARMS[0] else 11952), "incomplete diagnostic arm")
        cfg["inference_seconds"] = time.monotonic() - infer_start
    finally:
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
        finally:
            monitor.stop()
    cfg.update(status="complete",ended_utc=reader.now(),engine_shutdown=True,elapsed_seconds=time.monotonic()-began,
               predictions=n,tasks=n_tasks,batch_size=(24 if arm == ARMS[0] else 48),
               peak_gpu1_mib=monitor.peak,n_resource_samples=monitor.samples,df_after_reader=reader.original.root_df(),
               predictions_sha256=reader.digest(directory/"predictions.jsonl"))
    cfg["peak_observed_root_used_bytes"] = max(r["root"]["used_bytes"] for r in inputs.rows(directory/"resources.jsonl"))
    inputs.write_new(directory/"run_config.json",cfg)
    reader.receipts_write(directory,"inference",n=n,run_config_sha256=reader.digest(directory/"run_config.json"),
                          predictions_sha256=cfg["predictions_sha256"])
    reader.publish(directory,{"stage":"finished","status":"complete","tasks":n_tasks,"predictions":n})


def deployable_ipc():
    proof = ipc_record(SHORT)
    inputs.need(os.environ.get("VLLM_RPC_BASE_PATH",str(SHORT)) == str(SHORT), "VLLM_RPC_BASE_PATH conflicts with short IPC dir")
    return proof


def pipeline():
    inputs.need(os.getpid() == os.getsid(0) and Path(sys.prefix).resolve() == reader.ROOT/"envs/reader" and
                os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
                "detached reader environment required")
    with (WORK/"pipeline.pid").open("x") as f:
        print(os.getpid(),file=f)
    try:
        check_tests()
        checked_inputs()
        reader.receipts_write(WORK,"preflight",sha256=reader.digest(PREFLIGHT/"preflight.json"),
                              tests_sha256=reader.digest(SMALL/"cpu_tests.json"),n_r2=5976,n_r3=5976)
        admitted = reader.wait_for_gpu("reader_diag",lambda v:reader.publish(WORK,v))
        reader.receipts_write(WORK,"gate",gpu=admitted)
        results = {}
        for arm in ARMS:
            directory = WORK/arm
            directory.mkdir(exist_ok=False)
            reader.publish(WORK,{"status":"running","stage":arm})
            with (directory/"launch.log").open("xb") as log:
                child = subprocess.Popen([str(reader.PYTHON),"-B","-u",str(Path(__file__).resolve()),"arm",arm],
                                         stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
                reader.receipts_write(directory,"launch",pid=child.pid,parent=os.getpid())
                code = child.wait()
            inputs.need(code == 0, f"{arm} failed; exit {code}")
            cfgpath = directory/"run_config.json"
            status = reader.read(directory/"status.json")
            receipt = reader.read(directory/"receipt_inference.json")
            inputs.need(status["status"] == "complete" and receipt["run_config_sha256"] == reader.digest(cfgpath)
                        and receipt["predictions_sha256"] == reader.digest(directory/"predictions.jsonl"),
                        f"incomplete {arm} receipt")
            results[arm] = reader.read(cfgpath)
            reader.receipts_write(WORK,arm,exit_code=code,pid=child.pid,predictions=receipt["n"],
                                  inference_receipt_sha256=reader.digest(directory/"receipt_inference.json"))
        reader.publish(WORK,{"status":"complete","stage":"finished","automatic_retry":False,
                             "predictions":{k:v["predictions"] for k,v in results.items()}})
    except BaseException as exc:
        reader.publish(WORK,{"status":"failed","error":repr(exc),"automatic_retry":False})
        raise


def detach():
    check_tests()
    checked_inputs()
    inputs.need(WORK.resolve() == WORK and SMALL.resolve() == SMALL and SHORT.resolve() == SHORT,
                "diagnostic output path redirected")
    reader.require_new([WORK/f for f in ("pipeline.pid","launcher.log","ipc_check.json","events.jsonl","status.json",
                                         "receipt_gate.json","receipt_preflight.json",*ARMS)])
    SHORT.mkdir(parents=True,exist_ok=True)
    proof = deployable_ipc()
    print(f"IPC sample {proof['sample']}, path={proof['path_bytes']} bytes, URI={proof['uri_bytes']} bytes; must be <100",flush=True)
    inputs.write_new(WORK/"ipc_check.json",proof)
    # The reader export script owns the original Task 03 settings. Override only
    # temporary paths and enforce reader_environment's original spawning/device setup.
    shell = ('source scripts/reader_environment.sh && export TMPDIR="$1" TMP="$1" TEMP="$1" RAY_TMPDIR="$1" '
             'CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 && exec "$READER_ENV/bin/python" -B -u "$2" pipeline')
    with (WORK/"launcher.log").open("xb") as log:
        p = subprocess.Popen(["/bin/bash","-c",shell,"reader-diag",str(SHORT),str(Path(__file__).resolve())],
                             start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                             close_fds=True)
    print(f"detached PID {p.pid} -> {WORK}",flush=True)


if __name__ == "__main__":
    if Path.cwd() != PROJECT:
        raise SystemExit("run from the project directory")
    if sys.argv[1:] == ["detach"]:
        detach()
    elif sys.argv[1:] == ["pipeline"]:
        pipeline()
    elif len(sys.argv) == 3 and sys.argv[1] == "arm" and sys.argv[2] in ARMS:
        try:
            arm_run(sys.argv[2])
        except BaseException as exc:
            reader.publish(WORK/sys.argv[2], {"status":"failed", "error":repr(exc)})
            raise
    else:
        raise SystemExit("expected prepare, detach, pipeline, or internal arm action")

