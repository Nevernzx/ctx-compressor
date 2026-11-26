"""Resume Task 17 after the first prepare-heldout failure."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts import run_reranker_v1_pipeline as pipeline
from src.reader_heldout_inputs import rows, write_new
from src.reranker_v1 import LOCK, SMALL, WORK, need, read, record, require_new, verify

ROLLOUT = pipeline.r
PID = pipeline.PID
PREVIOUS_PID = 3345821
ARCHIVE = WORK / f"failed_prepare_heldout_{PREVIOUS_PID}"
AUDIT = SMALL / "prepare_heldout_resume.json"
TESTS = SMALL / "resume_heldout_cpu_tests.json"
ORDER = "after_task18"
STAGES = (
    ("S1", "S1", SMALL / "training.json", "complete", True),
    ("prepare-val", "prepare_val", ROLLOUT.VAL / "input_preflight.json", "passed", False),
    ("reader-val", "reader_val", SMALL / "val_reader_config.json", "complete", True),
    ("select-tau", "tau", SMALL / "tau_selection.json", "complete", False),
)


def references(obj):
    """Yield the file records embedded in an existing stage output."""
    if isinstance(obj, dict):
        if {"path", "sha256"} <= obj.keys():
            yield obj
        for value in obj.values():
            yield from references(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from references(value)


def verify_completed(events):
    """Verify completed stage outputs and the file hashes that they commit to."""
    completions = [e for e in events if e["status"] == "stage_complete"]
    need([e["stage"] for e in completions] == [stage[0] for stage in STAGES],
         "completed stages differ from the original run")
    files = {}
    old_receipts = {}
    results = {}

    def check(item, path):
        need(item["path"] == str(path) and path.resolve() == path
             and (path.is_relative_to(WORK) or path.is_relative_to(SMALL)),
             f"invalid Task 17 output path: {path}")
        size = item.get("size", item.get("bytes", path.stat().st_size))
        prior = files.get(str(path))
        if prior is None:
            need(path.stat().st_size == size, f"output size changed: {path}")
            verify(path, item["sha256"])
            files[str(path)] = {"sha256": item["sha256"], "size": size}
        else:
            need(prior == {"sha256": item["sha256"], "size": size},
                 f"receipts disagree for {path}")

    for event, (name, key, output, status, gpu) in zip(completions, STAGES, strict=True):
        pointer = WORK / "stages" / name / "receipt_complete.json"
        receipt = read(pointer)
        need(event["receipt"] == str(pointer) and event["child_pid"] == receipt["child_pid"]
             and receipt["stage"] == "complete" and receipt["exit_code"] == 0
             and receipt["gpu"] is gpu and set(receipt["outputs"]) == {output.name},
             f"inconsistent completion receipt: {name}")
        check(receipt["outputs"][output.name], output)
        result = read(output)
        need(result["status"] == status, f"inconsistent recorded status: {name}")
        for item in references(result):
            check(item, Path(item["path"]))
        old_receipts[key] = record(pointer)
        results[key] = result

    prep, config, chosen = (results[k] for k in ("prepare_val", "reader_val", "tau"))
    inference = ROLLOUT.VAL / "receipt_inference.json"
    infer = read(inference)
    pred_hash = infer["predictions_sha256"]
    need(prep["phase"] == "val" and prep["n_tasks"] == config["tasks"] == chosen["n_tasks"] == 100
         and prep["n_prompts"] == infer["n"] == config["predictions"] == chosen["n_predictions"] == 21600
         and config["engine_shutdown"] is True and config["effective_disable_cascade_attn"] is True
         and config["input_provenance"]["preflight_sha256"] == files[str(ROLLOUT.VAL / "input_preflight.json")]["sha256"]
         and infer["run_config_sha256"] == files[str(SMALL / "val_reader_config.json")]["sha256"]
         and pred_hash == config["predictions_sha256"] == chosen["predictions_sha256"]
         and chosen["tau"] == .5, "validation reader or selected tau differs")
    preds = ROLLOUT.VAL / "predictions.jsonl"
    check({"path": str(preds), "sha256": pred_hash}, preds)
    return {"stages": old_receipts, "files": files, "tau": chosen["tau"],
            "val_inference_receipt": record(inference)}


def detach():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "launch with CUDA hidden")
    phase_p = pipeline.audit_ready()
    suite = read(TESTS)
    previous_n = read(SMALL / "cpu_tests.json")["tests"]
    need(suite["status"] == "passed" and suite["tests"] == previous_n + 1
         and suite["matched_original_task17_tests"] == previous_n
         and not any(suite[key] for key in ("failures", "errors", "skipped", "cpu_guard_events")),
         "full CPU tests with the held-out regression have not passed")
    verify(Path(suite["xml"]), suite["xml_sha256"])

    status = read(WORK / "status.json")
    events = list(rows(WORK / "events.jsonl"))
    need(int(PID.read_text()) == status["pid"] == PREVIOUS_PID
         and status == events[-1] and status["status"] == "failed"
         and all(e["pid"] == PREVIOUS_PID and not e["automatic_retry"] for e in events)
         and events[-2]["stage"] == "prepare-heldout" and events[-2]["status"] == "running",
         "unexpected previous pipeline state")
    failure = WORK / "stages/prepare-heldout"
    failure_log = failure / "launcher.log"
    need("KeyError: 'task'" in failure_log.read_text(), "original failure does not match")
    for pid in (PREVIOUS_PID, events[-2]["child_pid"]):
        proc = Path(f"/proc/{pid}/stat")
        need(not proc.exists() or proc.read_text().rsplit(")", 1)[1].split()[0] == "Z",
             f"old PID {pid} is still active")
    require_new(ARCHIVE, AUDIT, WORK / "status.next", failure / "receipt_complete.json",
                SMALL / "hop_retention.json", SMALL / "run_config.json")
    archived = (
        Path("pipeline.pid"), Path("status.json"), Path("events.jsonl"), Path("launcher.log"),
        Path("stages/prepare-heldout"), Path("heldout_reader"), Path("rollouts/heldout"),
        Path("runtime_cache/prepare-heldout"),
    )
    need(all((WORK / p).exists() and (WORK / p).resolve() == WORK / p for p in archived),
         "old attempt files are missing or redirected")
    reuse = verify_completed(events)
    print(f"Verified 4 completed stages and {len(reuse['files'])} output hashes; tau={reuse['tau']}",
          flush=True)

    ARCHIVE.mkdir()
    for p in archived:
        target = ARCHIVE / p
        target.parent.mkdir(parents=True, exist_ok=True)
        require_new(target)
        (WORK / p).rename(target)
    archived_log = ARCHIVE / failure_log.relative_to(WORK)
    write_new(AUDIT, {
        "status": "passed", "resume_from": "prepare-heldout", "reused": reuse,
        "failure": {
            "utc": status["utc"],
            "beijing": datetime.fromisoformat(status["utc"]).astimezone(
                timezone(timedelta(hours=8))).isoformat(),
            "controller_pid": PREVIOUS_PID, "child_pid": events[-2]["child_pid"],
            "exception": "KeyError: 'task'", "archive": str(ARCHIVE), "log": record(archived_log),
            "cause": "The held-out loader stores the task ID in the mapping key, while HeldoutReplay expects a task field in each row.",
            "fix": "Copy the mapping key into the task field at the held-out preparation boundary.",
        },
        "cpu_tests": record(TESTS), "n_cpu_tests": suite["tests"],
        "phase_p_artifacts": phase_p, "automatic_retry": False,
    })
    env = pipeline.python_environment("controller", False)
    with (WORK / "launcher.log").open("xb") as stream:
        proc = subprocess.Popen(
            [str(pipeline.CPU_PY), "-B", "-u", str(Path(__file__).resolve()), "worker"],
            cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    print(f"Task17 resumed PID {proc.pid}; status {WORK / 'status.json'}", flush=True)


def worker():
    need(os.getsid(0) == os.getpid() and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
         "expected a detached CPU supervisor")
    with PID.open("x") as stream:
        stream.write(f"{os.getpid()}\n")
    try:
        audit = read(AUDIT)
        need(audit["status"] == "passed" and audit["reused"]["tau"] == .5, "invalid reuse audit")
        need(LOCK.resolve() == LOCK, "shared GPU lock path redirected")
        with LOCK.open("a+b") as handle:
            pipeline.publish("queue", "waiting_lock", lock=str(LOCK), requested_order=ORDER)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            pipeline.publish("queue", "holding_lock", lock=str(LOCK), requested_order=ORDER)
            opening = pipeline.reader.original.root_df()
            results = {}
            for name, key, output, _, _ in STAGES:
                ref = audit["reused"]["stages"][key]
                receipt = WORK / "stages" / name / "receipt_complete.json"
                verify(receipt, ref["sha256"])
                verify(output, audit["reused"]["files"][str(output)]["sha256"])
                results[key] = read(receipt)
            results["prepare_heldout"] = pipeline.stage(
                "prepare-heldout", "reranker_v1_rollouts.py", ["prepare-heldout"],
                [ROLLOUT.HELD / "input_preflight.json"], expected=("passed",))
            results["reader_heldout"] = pipeline.stage(
                "reader-heldout", "reranker_v1_rollouts.py", ["reader-heldout"],
                [SMALL / "heldout_reader_config.json"], gpu=True, reader_dir=ROLLOUT.HELD)
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
                "GPU": 1, "requested_order": ORDER, "automatic_retry": False,
                "owned_small_root_bytes": size, "root_start": opening,
                "root_end": pipeline.reader.original.root_df(), "resume": record(AUDIT),
            })
            pipeline.publish("all", "complete", lock=str(LOCK), config=record(config))
    except BaseException as exc:
        pipeline.publish("terminal", "failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("detach", "worker"))
    args = parser.parse_args()
    need(Path.cwd() == PROJECT, "run from the project on 5880")
    (detach if args.action == "detach" else worker)()
