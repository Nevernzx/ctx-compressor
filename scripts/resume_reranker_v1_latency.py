"""Resume Task 17 latency using the six completed-stage receipts."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts import resume_reranker_v1_heldout as first
from scripts import run_reranker_v1_pipeline as pipeline
from src.reader_heldout_inputs import rows, write_new
from src.reranker_v1 import LOCK, SMALL, WORK, need, read, record, require_new, verify

HELD = pipeline.r.HELD
PID = pipeline.PID
OLD_PID = 3441385
ARCHIVE = WORK / f"failed_latency_eager_{OLD_PID}"
AUDIT = SMALL / "latency_resume.json"
TESTS = SMALL / "resume_latency_cpu_tests.json"
PROBE = SMALL / "latency_compiler_cpu_preflight.json"
HELD_STAGES = (
    ("prepare-heldout", "prepare_heldout", HELD / "input_preflight.json", "passed", False),
    ("reader-heldout", "reader_heldout", SMALL / "heldout_reader_config.json", "complete", True),
)


def audit_completed(events):
    initial = read(first.AUDIT)
    original_events = list(rows(first.ARCHIVE / "events.jsonl"))
    prior = first.verify_completed(original_events)
    need(prior == initial["reused"], "the original four-stage audit changed")

    files = {}
    def check(item):
        path = Path(item["path"])
        need(path.is_absolute() and path == path.resolve()
             and (path.is_relative_to(WORK.parent) or path.is_relative_to(PROJECT))
             and not {"test_dev_v1", ".git"}.intersection(path.parts),
             f"unexpected completed-stage output path: {path}")
        size = path.stat().st_size
        need(size == item.get("size", item.get("bytes", size)), f"receipt size differs for {path}")
        if str(path) not in files:
            verify(path, item["sha256"])
            files[str(path)] = {"size": size, "sha256": item["sha256"]}
        else:
            need(files[str(path)]["sha256"] == item["sha256"], f"conflicting hashes for {path}")

    complete = [e for e in events if e["status"] == "stage_complete"]
    need([e["stage"] for e in complete] == [s[0] for s in HELD_STAGES],
         "unexpected held-out stage completion set")
    values = {}
    receipts = dict(prior["stages"])
    for event, (name, key, output, status, gpu) in zip(complete, HELD_STAGES, strict=True):
        path = WORK / "stages" / name / "receipt_complete.json"
        rec = read(path)
        need(event["receipt"] == str(path) and event["child_pid"] == rec["child_pid"]
             and rec["stage"] == "complete" and rec["gpu"] is gpu and rec["exit_code"] == 0
             and set(rec["outputs"]) == {output.name} and rec["outputs"][output.name]["path"] == str(output),
             f"inconsistent completion receipt: {name}")
        check(rec["outputs"][output.name])
        result = read(output)
        need(result["status"] == status, f"inconsistent completed output: {name}")
        for item in first.references(result):
            check(item)
        values[key] = result
        receipts[key] = record(path)

    prep, config = values["prepare_heldout"], values["reader_heldout"]
    infer_path = HELD / "receipt_inference.json"
    smoke_path = HELD / "receipt_smoke.json"
    inference, smoke = read(infer_path), read(smoke_path)
    need(prep["phase"] == "heldout" and tuple(prep["strategies"]) == pipeline.r.STRATEGIES
         and prep["n_tasks"] == config["tasks"] == 249
         and prep["n_prompts"] == config["predictions"] == inference["n"] == 29880
         and prep["rollouts"]["RR-τ"]["metrics"]["run"]["tau"] == prior["tau"] == .5
         and config["engine_shutdown"] is True and config["effective_disable_cascade_attn"] is True
         and config["input_provenance"]["preflight_sha256"] == files[str(HELD / "input_preflight.json")]["sha256"]
         and inference["run_config_sha256"] == files[str(SMALL / "heldout_reader_config.json")]["sha256"]
         and inference["predictions_sha256"] == config["predictions_sha256"]
         and config["smoke"] == {k: smoke[k] for k in ("n", "raw_output_mismatches")} and smoke["n"] == 40,
         "held-out reader evidence differs")
    for name, digest in (("predictions.jsonl", inference["predictions_sha256"]),
                         ("smoke_predictions.jsonl", smoke["predictions_sha256"])):
        check({"path": str(HELD / name), "sha256": digest})
    for path, digest in prep["source_signatures"].items():
        if isinstance(digest, str):
            check({"path": path, "sha256": digest})
    for item in first.references(read(SMALL / "latency_input.json")):
        check(item)
    hops = SMALL / "hop_retention.json"
    need(read(hops)["status"] == "complete", "held-out retention table is incomplete")
    return {"stages": receipts, "files": prior["files"] | files, "tau": prior["tau"],
            "val_inference_receipt": prior["val_inference_receipt"],
            "heldout_inference_receipt": record(infer_path), "heldout_smoke_receipt": record(smoke_path),
            "heldout_hop_retention": record(hops)}


def audit_environment():
    suite = read(TESTS)
    old_n = read(first.TESTS)["tests"]
    need(suite["status"] == "passed" and suite["tests"] > old_n
         and suite["matched_heldout_resume_tests"] == old_n
         and not any(suite[k] for k in ("failures", "errors", "skipped", "cpu_guard_events")),
         "full CPU suite with compiler regressions has not passed")
    verify(Path(suite["xml"]), suite["xml_sha256"])
    probe = read(PROBE)
    env, current = pipeline.compressor_environment("compiler-cpu-preflight")
    need(probe["compiled"]["compile_and_load_passed"] and env["CUDA_VISIBLE_DEVICES"] == ""
         and all(probe[key] == current[key] for key in current), "compiler CPU proof differs")
    sources = {}
    for name in ("grpo_r2", "grpo_r4"):
        status_file = WORK.parent / name / "pipeline_status.json"
        historical = read(status_file)
        stages = [s for s in historical["stages"] if s["stage"].startswith("F_checkpoint-")]
        need(stages and all(s["argv"][0] == probe["python"] and s["status"] == "complete"
                            and s["exit_code"] == 0 for s in stages),
             f"historical F interpreter differs in {name}")
        sources[name] = {"pipeline_status": record(status_file), "python": probe["python"],
                         "completed_F": [s["stage"] for s in stages]}
    return {"cpu_tests": record(TESTS), "n_cpu_tests": suite["tests"], "compiler": record(PROBE),
            "historical_F": sources}


def detach():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "launch with CUDA hidden")
    phase_p = pipeline.audit_ready()
    environment = audit_environment()
    status = read(WORK / "status.json")
    events = list(rows(WORK / "events.jsonl"))
    launch = [e for e in events if e["stage"] == "latency-eager" and e["status"] == "running"]
    need(int(PID.read_text()) == status["pid"] == OLD_PID and status == events[-1]
         and status["status"] == "failed" and "latency-eager exited" in status["error"] and launch
         and all(e["pid"] == OLD_PID and not e["automatic_retry"] for e in events),
         "unexpected previous latency pipeline state")

    failed = WORK / "stages/latency-eager"
    log = failed / "launcher.log"
    reason = "RuntimeError: Failed to find C compiler"
    need(reason in log.read_text(), "the archived eager failure is different")
    for pid in (OLD_PID, launch[0]["child_pid"]):
        proc = Path(f"/proc/{pid}/stat")
        need(not proc.exists() or proc.read_text().rsplit(")", 1)[1].split()[0] == "Z",
             f"prior PID {pid} is still active")
    require_new(ARCHIVE, AUDIT, WORK / "status.next", failed / "receipt_complete.json",
                SMALL / "run_config.json", SMALL / "heldout_comparison.json",
                *(SMALL / f"latency_{mode}.json" for mode in ("eager", "graph", "reranker")))
    paths = tuple(map(Path, ("pipeline.pid", "status.json", "events.jsonl", "launcher.log", "stages/latency-eager",
                            "latency/eager", "runtime_cache/latency-eager")))
    need(all((WORK / p).exists() and (WORK / p).resolve() == WORK / p for p in paths),
         "missing or redirected failed-attempt artifacts")
    reused = audit_completed(events)
    print(f"Verified 6 completed stages and {len(reused['files'])} file hashes, tau={reused['tau']}", flush=True)

    ARCHIVE.mkdir()
    for p in paths:
        target = ARCHIVE / p
        target.parent.mkdir(parents=True, exist_ok=True)
        require_new(target)
        (WORK / p).rename(target)
    first_failure = read(first.AUDIT)["failure"]
    write_new(AUDIT, {
        "status": "passed", "resume_from": "latency-eager", "reused": reused, "phase_p_artifacts": phase_p,
        "automatic_retry": False, "previous_resume": record(first.AUDIT), "first_failure": first_failure,
        "failure": {
            "utc": status["utc"],
            "beijing": datetime.fromisoformat(status["utc"]).astimezone(
                timezone(timedelta(hours=8))).isoformat(),
            "controller_pid": OLD_PID, "child_pid": launch[0]["child_pid"],
            "exception": reason, "archive": str(ARCHIVE), "log": record(ARCHIVE / log.relative_to(WORK)),
            "cause": "The latency controller did not load the F-stage compiler setup; Triton needed CC for LoRA kernels.",
            "fix": "Use the recorded F-stage Python and source grpo_r2_environment.sh, isolate Task 17 caches, and check CC before GPU admission.",
        },
        **environment,
    })
    with (WORK / "launcher.log").open("xb") as output:
        job = subprocess.Popen(
            [str(pipeline.CPU_PY), "-B", "-u", str(Path(__file__).resolve()), "worker"],
            cwd=PROJECT, env=pipeline.python_environment("controller", False), stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
        )
    print(f"Task17 latency resumed PID {job.pid}; status {WORK / 'status.json'}", flush=True)


def worker():
    need(os.getsid(0) == os.getpid() and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
         "expected a detached CPU supervisor")
    with PID.open("x") as fp:
        print(os.getpid(), file=fp)
    try:
        audit = read(AUDIT)
        need(audit["status"] == "passed" and audit["reused"]["tau"] == .5, "invalid stage reuse audit")
        results = {}
        for name, key, output, _, _ in first.STAGES + HELD_STAGES:
            path = WORK / "stages" / name / "receipt_complete.json"
            need(audit["reused"]["stages"][key]["path"] == str(path), "unexpected receipt pointer")
            verify(path, audit["reused"]["stages"][key]["sha256"])
            verify(output, audit["reused"]["files"][str(output)]["sha256"])
            results[key] = read(path)

        need(LOCK.resolve() == LOCK, "shared GPU lock path redirected")
        order = first.ORDER
        with LOCK.open("a+b") as handle:
            pipeline.publish("queue", "waiting_lock", lock=str(LOCK), requested_order=order)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            pipeline.publish("queue", "holding_lock", lock=str(LOCK), requested_order=order)
            opening = pipeline.reader.original.root_df()
            for mode in ("eager", "graph", "reranker"):
                results[f"latency_{mode}"] = pipeline.stage(
                    f"latency-{mode}", "reranker_v1_latency.py", [mode], [SMALL / f"latency_{mode}.json"],
                    gpu=True, expected=("complete", "load_failed") if mode == "graph" else ("complete",))
            results["compare"] = pipeline.stage(
                "compare", "reranker_v1_rollouts.py", ["summarize"], [SMALL / "heldout_comparison.json"])
            size = sum(p.stat().st_size for p in SMALL.rglob("*") if p.is_file())
            need(size < 300_000_000, "task small artifacts exceed the root limit")
            config = SMALL / "run_config.json"
            write_new(config, {
                "status": "complete", "task": 17, "reader_shares_v16_protocol": True,
                "template": read(SMALL / "training.json")["template"], "stages": results,
                "phase_p_artifacts": audit["phase_p_artifacts"], "shared_lock": str(LOCK),
                "GPU": 1, "requested_order": order, "automatic_retry": False,
                "owned_small_root_bytes": size, "root_start": opening,
                "root_end": pipeline.reader.original.root_df(), "resume": record(AUDIT),
                "compressor_environments": {
                    mode: record(WORK / f"stages/latency-{mode}/receipt_environment.json")
                    for mode in ("eager", "graph")
                },
            })
            pipeline.publish("all", "complete", lock=str(LOCK), config=record(config))
    except BaseException as exc:
        pipeline.publish("terminal", "failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("detach", "worker"))
    opts = parser.parse_args()
    need(Path.cwd() == PROJECT, "run from the project on 5880")
    (detach if opts.action == "detach" else worker)()
