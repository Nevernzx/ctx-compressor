"""Run the registered Task 09 train-only selection and three held-out evaluations exactly once."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.prepare_formal_grpo import ADAPTER, ROOT, SHA
from scripts.rollout_model import file_record
from scripts.run_formal_grpo_pipeline import OOM, child_process, gpu, log_failure, now, read, require

RUN = ROOT / "hedge_v2"
RAW = ROOT / "eval_rollouts_hedge_v2"
OUT = PROJECT / "eval/hedge_v2"
LOGS = RUN / "logs"
STATUS = RUN / "pipeline_status.json"
PYTHON = str(ROOT / "envs/grpo/bin/python")
STATES = ROOT / "grpo_v2/states/states.jsonl"
TRAIN_SHA = "33fdea6942278c99a815fc931f1be7354c1ace3ecb78a486e58baafc716e4eb8"
GRID = (0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01)
RUNS = (("H0", "nohedge"), ("H1", "tau_star"), ("H2", "always"))
SELECTION = OUT / "train_selection.json"


def check_scores(record):
    a, b, c = (record[k] for k in ("argmax", "logprob", "reencode"))
    n = record["sample_n"]
    require(record["status"] == "ok" and n == 300, "Incomplete score implementation check")
    require(0 < a["n"] <= n and a["matched"] <= a["n"] and 100 * a["matched"] >= 99 * a["n"],
            "Candidate argmax/greedy agreement below 99%")
    require(b["n"] == a["n"] == b["within_tolerance"] and b["tolerance"] == 0.02
            and 0 <= b["max_abs_error"] <= 0.02, "Greedy logprob discrepancy exceeds tolerance")
    require(c["n"] == c["returned_raw_full_id_matches"] == c["text_token_matches"]
            == c["full_with_terminal_matches"] == n, "Completion token round trip failed")


def verify_preflight():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
            "Incorrect physical GPU environment")
    checks = OUT / "score_checks.json"
    production = OUT / "score_production_checks.json"
    diagnostic = OUT / "score_checks_initial.json"
    score = read(checks)
    prod = read(production)
    check_scores(score)
    check_scores(prod)
    sample = file_record(RUN / "implementation/sample.json")
    require(score["production_receipt_sha256"] == file_record(production)["sha256"]
            and score["sample_sha256"] == prod["sample_sha256"] == read(diagnostic)["sample_sha256"] == sample["sha256"]
            and prod["adapter_model_sha256"] == SHA and prod["sample_seed"] == 20261121,
            "Score validation does not match the frozen implementation sample and adapter")
    verifier = Path(prod["verifier_path"])
    require(verifier == verifier.resolve() and verifier.parent == PROJECT / "scripts"
            and verifier.match("check_hedge_*.py"), "Unexpected score verifier path")
    for key, name in (("scorer", "hedge_scorer.py"), ("processor", "hedge_forced_tokens.py"),
                      ("raw_processor", "hedge_raw_logits.py"), ("check", verifier.name)):
        require(prod["source_sha256"][key] == file_record(PROJECT / "scripts" / name)["sha256"],
                f"Checked implementation changed: {name}")
    tests = read(OUT / "pipeline_tests.json")
    require(tests["status"] == "passed" and tests["legacy_tests_matched"] == 533
            and tests["tests"] == tests["passed"] >= 533, "Missing full CPU test receipt")
    state = read(STATES.with_name("state_checks.json"))
    require(state["scope"] == "train" and state["n_states"] == 22158 and state["n_tasks"] == 400
            and state["n_trajectories"] == 2400 and state["source_adapter_sha256"] == SHA
            and state["available_checks_pass"] and state["heldout_overlap"] == 0,
            "Task 08 train state receipt failed integrity checks")
    artifact = file_record(STATES)
    require(artifact == state["state_file"] and artifact["sha256"] == TRAIN_SHA, "Train states changed")
    require(file_record(ADAPTER / "adapter_model.safetensors")["sha256"] == SHA, "SFT v2 adapter changed")
    for path in [SELECTION, RUN / "train", RUN / "pipeline_monitor.jsonl",
                 *[RAW / name for _, name in RUNS], *[OUT / name for _, name in RUNS]]:
        require(not os.path.lexists(path), f"Existing output: {path}")
    return {"score_checks": file_record(checks), "production_checks": file_record(production), "sample": sample, "tests": file_record(OUT / "pipeline_tests.json"), "train_states": artifact,
            "adapter_sha256": SHA}


def check_selection():
    info = read(SELECTION)
    require(info["status"] == "complete" and info["n_states"] == 22158 and info["n_tasks"] == 400
            and info["n_trajectories"] == 2400 and info["split"] == "train"
            and info["heldout_overlap"] == 0, "Train scope or denominators changed")
    require(tuple(info["grid"]) == GRID and len(info["cells"]) == 6, "Train selection grid changed")
    require(info["state_file"]["sha256"] == TRAIN_SHA
            and info["score_file"] == file_record(RUN / "train/state_scores.jsonl"), "Train score artifact changed")
    means = {}
    expected_keys = {str(float(t)) for t in GRID}
    require(set(info["pooled"]) == expected_keys, "Incorrect pooled tau keys")
    require(set(info["cells"]) == {f"{mode}|{tier}" for mode in ("append", "reselect")
                                   for tier in ("tight", "mid", "loose")}, "Unrecognized train cells")
    for cell in info["cells"].values():
        require(set(cell) == expected_keys and all(row["n"] == 3693 for row in cell.values()),
                "Incomplete train cell grid")
    for tau in GRID:
        row = info["pooled"][str(float(tau))]
        mean = row["mean_reward"]
        require(row["n"] == mean["n"] == 22158 and math.isfinite(mean["mean"]), "Invalid pooled reward")
        means[tau] = mean["mean"]
    require(set(means) == set(GRID) and info["tau_star"] == min(GRID, key=lambda t: (-means[t], t)),
            "Selected tau does not follow the registered rule")
    baseline = info["pooled"]["0.0"]
    misses = baseline["current_gold_miss_rate"]
    require(info["lambda"] == 0.3 and baseline["hedge_rate"]["count"] == baseline["mean_extra_tokens"]["sum"] == 0
            and (misses["count"], misses["n"]) == (1529, 22146), "Zero-threshold train replay changed")
    return {"tau_star": info["tau_star"], "selection": file_record(SELECTION), "score_file": info["score_file"]}


def check_heldout(name, expected_tau):
    directory = OUT / name
    record = read(directory / "run.json")
    checks = record["checks"]
    require(record["status"] == "complete" and record["engine_shutdown"]["completed"] and record["tau"] == expected_tau
            and checks["n_steps"] == 13956 and checks["n_cells"] == 6
            and checks["n_trajectories"] == checks["step0_matched"] == 1494
            and record["step0"]["user_byte_mismatches"] == record["step0"]["token_id_mismatches"] == 0
            and record["selection"] == file_record(SELECTION),
            f"Incomplete evaluation or incorrect tau: {name}")
    metrics = read(directory / "metrics.json")
    pooled = metrics["combined"]
    require(len(metrics["configs"]) == 6 and pooled["n_tasks"] == 1494 and pooled["n_steps"] == 13956,
            "Incorrect held-out denominators")
    require(metrics["steps_artifact"] == file_record(RAW / name / "steps.jsonl"), "Held-out raw file changed")
    for key in ("extra", "diagnostics"):
        require(bool(read(directory / (key + ".json"))), f"Missing {key} receipt for {name}")
    return {"tau": expected_tau, "run": file_record(directory / "run.json"), "steps": metrics["steps_artifact"]}


class Pipeline:
    def __init__(self):
        self.record = {"status": "running", "stage": "preflight", "started_utc": now(),
                       "pipeline_pid": os.getpid(), "session_id": os.getsid(0),
                       "stage_pid": None, "gpu_index": 1, "automatic_retry": False,
                       "stages": [], "last_monitor": None}

    def save(self):
        tmp = STATUS.with_name("pipeline_status.writing")
        with tmp.open("x") as output:
            json.dump(self.record, output, indent=2, ensure_ascii=False)
            output.write("\n")
        tmp.replace(STATUS)

    def stage(self, label, filename, args, verify):
        before = gpu()
        command = [PYTHON, "-B", "-u", str(PROJECT / "scripts" / filename), *args]
        path = LOGS / f"{label}.log"
        stage = {"stage": label, "argv": command, "started_utc": now(), "status": "starting",
                 "gpu_before": before, "log": str(path), "uses_gpu": True, "pid": None}
        self.record["stages"].append(stage)
        self.record.update(stage=label, stage_pid=None, last_monitor=None)
        self.save()
        require(before["memory_used_mib"] < 2048, f"GPU 1 has {before['memory_used_mib']} MiB in use before {label}")
        with path.open("xb") as output, child_process(command, os.environ.copy(), output) as child:
            stage.update(status="running", pid=child.pid)
            self.record["stage_pid"] = child.pid
            self.save()
            with path.open("rb") as stream, (RUN / "pipeline_monitor.jsonl").open("a") as monitor_file:
                tail = ""
                while True:
                    rc = child.poll()
                    chunk = tail + stream.read().decode("utf-8", errors="replace")
                    sample = {"utc": now(), "stage": label, "pid": child.pid, "alive": rc is None,
                              "oom": bool(OOM.search(chunk)), **gpu()}
                    tail = chunk[-200:]
                    monitor_file.write(json.dumps(sample) + "\n")
                    monitor_file.flush()
                    self.record["last_monitor"] = sample
                    self.save()
                    require(not sample["oom"], f"{label}: {log_failure(path)}")
                    if rc is not None:
                        break
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
        stage.update(exit_code=rc, ended_utc=now())
        self.record["stage_pid"] = None
        if rc != 0:
            raise RuntimeError(f"{label} exited {rc}: {log_failure(path)}")
        stage["receipt"] = verify()
        stage.update(status="complete", verified_utc=now())
        self.save()
        return stage["receipt"]

    def run(self):
        self.record["preflight"] = verify_preflight()
        selection = self.stage("train_scores", "score_hedge_train.py", [], check_selection)
        for label, name in RUNS:
            require(file_record(SELECTION) == selection["selection"], "Train threshold receipt changed")
            tau = {"H0": 0, "H1": selection["tau_star"], "H2": 1.01}[label]
            self.stage(label, "eval_hedge_v2.py", ["--name", name], lambda: check_heldout(name, tau))
        require(file_record(SELECTION) == selection["selection"], "Train selection changed during evaluation")
        self.record.update(status="evaluation_complete", stage="finished", tau_star=selection["tau_star"])


def main():
    if STATUS.exists():
        raise FileExistsError(STATUS)
    with (RUN / "pipeline.pid").open("x") as pid_file:
        pid_file.write(f"{os.getpid()}\n")
    def stop(signum, frame):
        raise InterruptedError(f"Pipeline received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    controller = Pipeline()
    controller.save()
    try:
        controller.run()
    except BaseException as exc:
        controller.record.update(status="failed", first_error=str(exc), traceback=traceback.format_exc())
        if controller.record["stages"] and controller.record["stages"][-1]["status"] != "complete":
            controller.record["stages"][-1].update(status="failed", ended_utc=now())
        raise
    finally:
        controller.record.update(ended_utc=now(), stage_pid=None)
        controller.save()


if __name__ == "__main__":
    main()
