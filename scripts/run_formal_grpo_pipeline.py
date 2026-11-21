"""Run formal F0 → B → C → D → E → F once, publishing stage receipts on the external disk."""
import argparse
from contextlib import contextmanager
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.prepare_formal_grpo import ADAPTER, BASE, MERGED, OUT, ROOT, RUN, SHA
from scripts.rollout_model import file_record
from src.heldout_eval import select_checkpoint

LOGS = RUN / "logs"
STATUS = RUN / "pipeline_status.json"
PYTHON = str(ROOT / "envs/grpo/bin/python")
STATES = RUN / "states/states.jsonl"
GROUPS = RUN / "filter/groups.jsonl"
DATA = RUN / "data/train.jsonl"
TRAIN = RUN / "train"
GPU_UUID = "GPU-4318dc90-4126-038a-9402-1323533db94f"
LABEL = "merged_sft_v2"
CHECKPOINT_NAME = re.compile(r"checkpoint-([1-9][0-9]*)\Z")
OOM = re.compile(r"CUDA (?:error: )?out of memory|(?:torch\.(?:cuda\.)?)?OutOfMemoryError:|CUBLAS_STATUS_ALLOC_FAILED", re.I)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def put_new(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def cli(name, *args):
    return [PYTHON, "-u", "-B", str(PROJECT / "scripts" / name), *map(str, args)]


def gpu():
    output = subprocess.check_output(
        ["nvidia-smi", "-i", "1", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=10)
    idx, uuid, used, util = map(str.strip, output.strip().split(","))
    require((idx, uuid) == ("1", GPU_UUID), "Physical GPU 1 identity changed")
    return {"index": 1, "uuid": uuid, "memory_used_mib": int(used), "utilization_pct": int(util)}


def check_a():
    a = read(OUT / "a_checks.json")
    require(a["status"] == "passed" and a["n_reference"] == 30 and a["adapter_sha256"] == SHA,
            "A model/fixture check missing")
    t = read(OUT / "cpu_tests.json")
    require(t["status"] == "passed" and t["matched_old_tests"] == 501
            and t["tests"] >= 501 and t["failures"] == t["errors"] == t["skipped"] == 0,
            "A CPU tests missing or incomplete")
    require(read(OUT / "input_checks.json")["overlapping_question_ids"] == 0, "Data isolation check missing")


def check_eval(name):
    directory = OUT / name
    info = read(directory / "run.json")
    require(info["status"] == "complete" and info["engine_shutdown"]["completed"], f"{name}: engine check")
    require(info["checks"]["n_steps"] == 13956 and info["checks"]["n_trajectories"] == 1494,
            f"{name}: incomplete held-out grid")
    for fname in ("extra.json", "diagnostics.json"):
        require(bool(read(directory / fname)), f"{name}: {fname} missing")
    result = read(directory / (name + "_results.json"))
    require(result["combined"]["n_tasks"] == 1494 and result["combined"]["n_steps"] == 13956,
            f"{name}: pooled metric denominators")
    return result


def check_states():
    s = read(STATES.parent / "state_checks.json")
    require(s["available_checks_pass"] and s["scope"] == "train" and s["n_tasks"] == 400
            and s["n_trajectories"] == 2400 and s["n_states"] == s["gold_recompute_n"] == s["prompt_reencode_n"],
            "B state coverage or split mismatch")
    require(all(s[k] == 0 for k in ["gold_mismatches", "reencode_mismatches", "rerender_mismatches", "heldout_overlap"])
            and all(s["step0_sft"][k] == 0 for k in ["direct_mismatches", "budget_substituted_mismatches"]),
            "B integrity mismatch")
    require(s["source_adapter_sha256"] == SHA and len(s["cells"]) == 6, "B policy/grid mismatch")
    return s["n_states"]


RESUME_SUFFIX = ".failed_C_topk"


def archived(path):
    return path.with_name(path.name + RESUME_SUFFIX)


def archive_sources():
    # Read either original or archived on explicitly restarting an interrupted rename.
    files = (RUN / "pipeline.pid", RUN / "pipeline_monitor.jsonl", LOGS / "C.log", STATUS)
    sources = {}
    for path in files:
        dest = archived(path)
        require(not path.is_symlink() and not dest.is_symlink(), f"Unexpected symlink at {path}")
        require(path.is_file() != dest.is_file(), f"Missing or ambiguous previous artifact at {path}")
        sources[path] = path if path.exists() else dest
    return sources


def prepare_resume_c():
    sources = archive_sources()
    previous = read(sources[STATUS])
    require(previous["status"] == "failed" and previous["stage"] == "C" and previous["stage_pid"] is None
            and previous.get("ended_utc"), "Expected an exited pipeline that failed in C")
    prior = previous["stages"]
    require([(s["stage"], s["status"]) for s in prior] == [("F0", "complete"), ("B", "complete"), ("C", "failed")]
            and all(s["exit_code"] == 0 for s in prior[:2]) and prior[-1]["exit_code"] != 0,
            "Cannot reuse unfinished stages")
    pid = int(sources[RUN / "pipeline.pid"].read_text())
    require(pid == previous["pipeline_pid"], "Failure status and pidfile disagree")
    # The first controller predates the resume lock. It must have exited before rotation.
    for old_pid in (pid, prior[-1]["pid"]):
        stat = Path(f"/proc/{old_pid}/stat")
        if stat.is_file():
            require(stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z", f"Previous PID {old_pid} is live")
    for path in (GROUPS.parent, DATA.parent, TRAIN):
        require(not os.path.lexists(path), f"Refusing to overwrite stage output: {path}")
    metrics = check_eval(LABEL)
    n_states = check_states()
    state_file = file_record(STATES)
    require(state_file == read(STATES.parent / "state_checks.json")["state_file"], "B states file hash or size mismatch")
    f0_file = file_record(ROOT / "eval_rollouts_grpo_v2" / LABEL / "steps.jsonl")
    require(f0_file == metrics["steps_artifact"], "F0 raw steps file hash or size mismatch")
    receipt = {"from_stage": "C", "rechecked_utc": now(), "prior_status": str(archived(STATUS)),
               "prior_pid": pid, "prior_first_error": previous["first_error"], "prior_started_utc": previous["started_utc"],
               "reused_stages": ["F0", "B"], "verified_state_file": state_file, "n_states": n_states,
               "verified_f0_file": f0_file, "f0_trajectories": metrics["combined"]["n_tasks"],
               "f0_steps": metrics["combined"]["n_steps"],
               "archived_paths": {str(src): str(archived(src)) for src in sources}}
    # All checks finish before the first rename. Keep the failed status available until last.
    for path, source in sources.items():
        if path == source:
            path.rename(archived(path))
    return receipt, prior[:2]


def checkpoint_paths(receipt):
    require(receipt["status"] == "ok", "Training verification did not pass")
    entries = [Path(p) for p in receipt["checkpoints"]]
    steps = []
    for p in entries:
        match = CHECKPOINT_NAME.fullmatch(p.name)
        require(match and p.resolve().parent == TRAIN.resolve() and (p / "adapter_model.safetensors").is_file(),
                f"Invalid checkpoint path {p}")
        steps.append(int(match.group(1)))
    require(steps and sorted(set(steps)) == steps and steps == receipt["checkpoint_steps"],
            "Invalid checkpoint order")
    actual = {p for p in TRAIN.glob("checkpoint-*") if p.is_dir()}
    require(set(entries) == actual, "Some saved checkpoint is absent from evaluation queue")
    return entries


@contextmanager
def child_process(args, env, output):
    proc = subprocess.Popen(args, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                            stdout=output, stderr=subprocess.STDOUT)
    try:
        yield proc
    except BaseException:
        # This is the unreaped direct child, so the PID cannot have been reused.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise


def log_failure(log):
    with log.open("rb") as f:
        f.seek(max(0, os.fstat(f.fileno()).st_size - 48000))
        text = f.read().decode("utf-8", "replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pattern = r"(?:[A-Za-z]*Error|Exception):|^NEED:|^FATAL"
    return next((line[-1600:] for line in lines if re.search(pattern, line)), lines[-1][-1600:] if lines else "no output")


class Pipeline:
    def __init__(self):
        self.record = {"status": "running", "stage": "preflight", "pipeline_pid": os.getpid(),
                       "session_id": os.getsid(0), "started_utc": now(), "ended_utc": None,
                       "stage_pid": None, "automatic_retry": False, "gpu_index": 1, "stages": []}
        self.active = None

    def save(self):
        tmp = STATUS.with_suffix(".writing")
        tmp.write_text(json.dumps(self.record, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, STATUS)

    def stage(self, name, args, on_gpu, extra_env=None):
        stage = {"stage": name, "started_utc": now(), "ended_utc": None, "pid": None,
                 "argv": list(args), "uses_gpu": on_gpu, "status": "starting"}
        self.record["stages"].append(stage)
        self.record.update(stage=name, stage_pid=None, last_monitor=None)
        self.active = stage
        self.save()
        before = gpu()
        stage["gpu_before"] = before
        if on_gpu:
            require(before["memory_used_mib"] < 2048, f"{name}: GPU 1 busy ({before['memory_used_mib']} MiB)")
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1" if on_gpu else "",
               "GRPO_GPU": "1", "GRPO_CPU_ONLY": "0" if on_gpu else "1", **(extra_env or {})}
        log = LOGS / (name + ".log")
        stage["log"] = str(log)
        with log.open("xb") as out, child_process(args, env, out) as proc:
            stage.update(pid=proc.pid, status="running")
            self.record["stage_pid"] = proc.pid
            self.save()
            with log.open("rb") as reader, (RUN / "pipeline_monitor.jsonl").open("a") as history:
                tail = ""
                oom = False
                while True:
                    code = proc.poll()
                    chunk = tail + reader.read().decode("utf-8", "replace")
                    oom = oom or bool(OOM.search(chunk))
                    tail = chunk[-200:]
                    monitor = {"utc": now(), "stage": name, "pid": proc.pid, "alive": code is None,
                               "nonfinite": False, "oom": oom, **gpu()}
                    # Training's watchdog checks unrounded loss and grad norm and stops the specific trainer PID.
                    failure = TRAIN / "training_failure.json"
                    if name == "E" and failure.exists():
                        incident = read(failure)
                        monitor["training_failure"] = incident
                        monitor["nonfinite"] = bool(incident["nonfinite"])
                        monitor["oom"] |= bool(incident["oom"])
                    self.record["last_monitor"] = monitor
                    history.write(json.dumps(monitor) + "\n")
                    history.flush()
                    self.save()
                    if code is None and (monitor["oom"] or monitor["nonfinite"]):
                        raise RuntimeError(f"{name} flagged OOM/nonfinite: {log_failure(log)}")
                    if code is not None:
                        break
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
        stage.update(exit_code=code, ended_utc=now(), gpu_after=monitor, status="exited")
        self.save()
        if code != 0 or monitor["nonfinite"] or monitor["oom"]:
            raise RuntimeError(f"{name} exit {code}: {log_failure(log)}")
        stage["status"] = "complete"
        self.save()

    def evaluate(self, name, adapter=None):
        args = cli("eval_formal_grpo.py", "--name", name, "--model", MERGED)
        if adapter:
            args += ["--adapter", str(adapter)]
        self.stage("F0" if adapter is None else "F_" + name, args, True)
        return check_eval(name)

    def run(self, resume=None):
        check_a()
        if resume is None:
            self.evaluate(LABEL)
            self.stage("B", cli("build_formal_grpo_states.py", "--stage", "build", "--base-model", BASE,
                                 "--adapter", ADAPTER, "--task-file", ROOT / "data/traj_train_v2/train_tasks.jsonl",
                                 "--n-tasks", 400, "--seed", 20261120, "--gpu", 1, "--output-dir", STATES.parent), True)
            n_states = check_states()
        else:
            n_states = resume["n_states"]
        self.stage("C", cli("probe_formal_grpo.py", "--all-states", "--states", STATES, "--adapter", ADAPTER,
                             "--gpu", 1, "--output-dir", GROUPS.parent), True)
        probe = read(GROUPS.parent / "sampling_probe.json")
        with GROUPS.open("rb") as f:
            n_groups = sum(1 for _ in f)
        require(n_groups == n_states == probe["n_states"] == probe["pooled"]["n_groups"]
                and probe["pooled"]["n_completions"] == n_states * 8
                and probe["generation"]["n"] == 8 and probe["generation"]["seed"] == 20261120
                and probe["generation"]["temperature"] == probe["generation"]["top_p"] == 1.0
                and probe["generation"]["max_tokens"] == 256 and probe["generation"]["top_k"] == -1
                and probe["generation"]["top_k_actual"] in (0, -1)
                and probe["lambda"] == 0.3 and probe["informative_threshold"] == 1e-9,
                "Filter coverage or generation configuration differs from registration")
        self.stage("D", cli("make_formal_grpo_dataset.py", "--states", STATES, "--informative", GROUPS,
                             "--seed", 20261121, "--out", DATA), False)
        dataset = read(DATA.parent / "dataset_stats.json")
        n = dataset["n"]
        require(n == min(2000, dataset["n_informative"]) and dataset["n_informative"] == probe["pooled"]["informative"]["count"]
                and dataset["n_states"] == n_states and dataset["user_text_mismatches"] == dataset["heldout_rows"] == 0,
                "Informative dataset failed count, split or prompt equality checks")
        with DATA.open("rb") as f:
            require(sum(1 for _ in f) == n and 4 <= n <= 2000, "Insufficient or inconsistent optimizer data")
        self.stage("E", ["bash", "scripts/train_formal_grpo.sh"], True,
                   {"MERGED": str(MERGED), "DATA": str(DATA), "OUT": str(TRAIN),
                    "MAX_STEPS": str(n // 4), "SAVE_STEPS": "125", "GRPO_SEED": "20261120",
                    "LR": "5e-6", "BETA": "0.04", "LAMBDA": "0.3", "VLLM_UTIL": "0.30",
                    "VLLM_MAX_LEN": "9216", "SLEEP_LEVEL": "0", "NGEN": "8", "GEN_TEMP": "1.0",
                    "TOP_P": "1.0", "PER_DEVICE": "4", "GRAD_ACC": "8", "EPOCHS": "1"})
        self.stage("E_check", cli("check_formal_grpo_training.py", "--data", DATA, "--train-dir", TRAIN), False)
        results = []
        for adapter in checkpoint_paths(read(OUT / "training_checks.json")):
            results.append(self.evaluate(adapter.name, adapter))
        selected = select_checkpoint(results)
        name = selected["strategy"]
        put_new(OUT / "selected_checkpoint.json", {"strategy": name, "path": str(TRAIN / name),
                "n_candidates": len(results), "checkpoint_step": selected["checkpoint_step"],
                "evidence_loss": selected["combined"]["evidence_loss"], "final_recall": selected["combined"]["final_recall"],
                "rule": "pooled loss ascending, pooled recall descending, earliest optimizer step"})
        self.record.update(stage="finished", status="evaluation_complete", selected_checkpoint=name)


def run_once(resume_from):
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" and os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
            "Launch via the formal GPU 1 environment")
    resume = None
    prior_stages = []
    if resume_from == "C":
        check_a()
        resume, prior_stages = prepare_resume_c()
    else:
        require(not STATUS.exists(), f"Existing pipeline status: {STATUS}")
    with (RUN / "pipeline.pid").open("x") as f:
        f.write(str(os.getpid()) + "\n")
    def stop(signum, frame):
        raise InterruptedError(f"Controller received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    p = Pipeline()
    if resume is not None:
        p.record.update(resume=resume, stages=prior_stages)
    p.save()
    try:
        p.run(resume)
    except BaseException as exc:
        p.record.update(status="failed", first_error=str(exc), traceback=traceback.format_exc())
        if p.active:
            p.active.update(status="failed", ended_utc=now())
        raise
    finally:
        p.record.update(ended_utc=now(), stage_pid=None)
        p.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", choices=["C"], help="recheck F0 and B, preserve failed C and continue")
    args = parser.parse_args()
    with (RUN / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_once(args.resume_from)


if __name__ == "__main__":
    main()
