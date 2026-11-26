"""Queue the Task 20 smoke behind the registered Task 19 G3 completion."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import agent_v1_smoke as smoke
from scripts.run_reader_heldout import publish
from src.reader_heldout_inputs import need, read, verify, write_new
from src.reranker_v1 import record, require_new

Q = smoke.WORK / "smoke_queue"
PID = Q / "queue.pid"
ANCHOR = smoke.PROJECT / "eval/test_v1/run_config.json"
QUEUE_TESTS = smoke.SMALL / "smoke_queue_cpu_tests.json"
INTERVAL = 30


def ready(state):
    need(state["task"] == 19, "unexpected Task 19 status source")
    return (state["status"] == "evaluation_complete" and state["stage"] == "all"
            and state.get("lock_released") is True)


def task19_evidence(state):
    receipt = state["config"]
    need(receipt["path"] == str(ANCHOR), "Task 19 completion references an unexpected run configuration")
    verify(ANCHOR, receipt["sha256"])
    cfg = read(ANCHOR)
    gate = cfg["stages"]["G3"]
    need(cfg["status"] == "evaluation_complete" and cfg["shared_lock"] == str(smoke.LOCK)
         and gate["stage"] == "complete" and gate["exit_code"] == 0 and gate["gpu"] is True,
         "Task 19 G3 completion receipt differs")
    return receipt


def environment():
    inherited = {k:v for k,v in os.environ.items() if k.lower() not in
                 ("http_proxy", "https_proxy", "all_proxy", "deepseek_api_key")}
    return {**inherited, "CUDA_VISIBLE_DEVICES":"", "PYTHONDONTWRITEBYTECODE":"1", "HF_HUB_OFFLINE":"1",
            "TRANSFORMERS_OFFLINE":"1", "HF_DATASETS_OFFLINE":"1", "NO_PROXY":"*", "no_proxy":"*"}


def queue_receipt():
    tests = read(QUEUE_TESTS)
    baseline = read(smoke.CPU_TESTS)
    need(tests["status"] == "passed" and tests["matched_phase_p_tests"] == baseline["tests"]
         and tests["tests"] >= baseline["tests"] + 2
         and all(tests[k] == 0 for k in ("failures", "errors", "skipped"))
         and not tests["cpu_guard_events"], "Task 20 queue regressions have not passed")
    verify(Path(tests["xml"]), tests["xml_sha256"])
    phase = smoke.phase_p(verify_models=True)
    return {"status":"passed", "phase_p":phase, "queue_cpu_tests":record(QUEUE_TESTS),
            "queue_source":record(Path(__file__).resolve()), "lock":str(smoke.LOCK),
            "release_condition":"Task 19 G3 completed and released the shared lock",
            "automatic_retry":False, "main_approval":False}


def update(status, **extras):
    publish(Q, {"task":20, "stage":"smoke_queue", "status":status, "lock":str(smoke.LOCK),
                "automatic_retry":False, "main_approval":False, **extras})


def ensure_smoke_fresh():
    require_new(smoke.STATUS, smoke.PID, smoke.P_RECEIPT, smoke.LOGS, smoke.RUNTIME / "smoke",
                smoke.WORK/"launcher.log", smoke.SMOKE, smoke.SMOKE_REPORT, smoke.SMOKE_CONFIG)


def worker():
    need(os.getsid(0) == os.getpid() and os.environ["CUDA_VISIBLE_DEVICES"] == "",
         "expected a detached CPU queue")
    with PID.open("x") as fp:
        print(os.getpid(), file=fp)
    try:
        proof = read(Q/"preflight.json")
        need(proof["status"] == "passed" and proof["queue_source"] == record(Path(__file__).resolve())
             and proof["phase_p"] == smoke.phase_p(verify_models=False), "queued setup changed")
        prev = None
        while True:
            state = read(smoke.TASK19_STATUS)
            if ready(state):
                break
            observed = (state["pid"], state["stage"], state["status"])
            if observed != prev:
                update("waiting_task19", dependency={"pid":observed[0], "stage":observed[1], "status":observed[2]})
                prev = observed
            time.sleep(INTERVAL)
        completed = task19_evidence(state)
        ensure_smoke_fresh()
        update("launching_smoke", task19=completed)
        output = Q/"smoke_launch.log"
        with output.open("xb") as log:
            subprocess.run(smoke.invocation(smoke.CPU, "launch"),
                           cwd=smoke.PROJECT, env=environment(), stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        deadline = time.monotonic() + 30
        while not smoke.PID.is_file() and time.monotonic() < deadline:
            time.sleep(.2)
        need(smoke.PID.is_file(), f"smoke supervisor did not create its PID file: {output}")
        child = int(smoke.PID.read_text())
        need(os.getsid(child) == child, "smoke supervisor is not detached")
        update("smoke_launched", smoke_pid=child, smoke_status=str(smoke.STATUS), task19=completed)
    except Exception as exc:
        update("failed", error=repr(exc))
        raise


def enqueue():
    need(Path.cwd() == smoke.PROJECT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "queue on 5880 with CUDA hidden")
    ensure_smoke_fresh()
    require_new(Q)
    proof = queue_receipt()
    Q.mkdir()
    write_new(Q/"preflight.json", proof)
    queue_command = [str(smoke.CPU), "-B", "-u", "-m", "scripts.agent_v1_queue", "worker"]
    if smoke.ATTEMPT:
        queue_command.extend(("--attempt", smoke.ATTEMPT))
    with (Q/"launcher.log").open("xb") as log:
        child = subprocess.Popen(queue_command,
                                 cwd=smoke.PROJECT, env={**environment(), **smoke.runtime_environment("queue")},
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(f"Task 20 detached smoke queue PID {child.pid}. Status {Q/'status.json'}", flush=True)


def main():
    global Q, PID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("enqueue", "worker"))
    parser.add_argument("--attempt", choices=smoke.ATTEMPTS)
    args = parser.parse_args()
    if args.attempt:
        smoke.select_attempt(args.attempt)
        Q = smoke.WORK / "smoke_queue"
        PID = Q / "queue.pid"
    (enqueue if args.action == "enqueue" else worker)()


if __name__ == "__main__":
    main()
