#!/usr/bin/env python3
"""Freeze Task 05 external inventory and final resource evidence, CPU only.

Use all (default), inventory, or resources after B, C and their watcher exit.
All commands require completed work, inactive recorded Task 05 processes, GPU 1
release and the root-disk limit. Existing checkpoints, logs and source results
are never modified. Outputs live only in project eval/sft_run, outside the three
inventoried trees; rerunning cannot add its own outputs to the inventory.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
sys.path.insert(0, str(PROJECT))
from scripts.checksums_task05 import FINAL_ROOTS, external_inventory
from src.training_control import checkpoint_steps, validate_training_counts

MODELS = ("base", "checkpoint-500", "checkpoint-1000", "checkpoint-1500", "checkpoint-2000", "checkpoint-2500")
PID_NAMES = {"worker_started.json", "worker_config.json", "worker_result.json", "launch.json",
             "continuation_started.json", "continuation_status.json", "orchestrator_started.json",
             "orchestrator_resumed.json", "train_run_config.json", "model_rollout_run.json",
             "batch_bench.json", "batch_bench_progress.json", "batch8_initialization_failure.json", "training_stop.json"}
LAUNCH_NAME = re.compile(r"(?:pipeline|continuation|evaluation)_launch(?:_[A-Za-z0-9_]+)?\.json\Z")
TASK_COMMANDS = {"train_sft_worker.py", "bench_batch.py", "wait_train_then_eval.py", "run_sft_pipeline.sh",
                 "train_sft.sh", "run_heldout_model.sh", "rollout_model.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def stat_identity(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def file_record(path):
    """Hash one unchanged regular file and count actual newline-terminated rows."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Not a regular unlinked file: {path}")
    before, digest, lines, last = stat_identity(path), hashlib.sha256(), 0, b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
            if path.suffix == ".jsonl":
                lines += block.count(b"\n")
                last = block[-1:]
    require(stat_identity(path) == before, f"File changed while hashing: {path}")
    result = {"path": str(path), "bytes": before[2], "sha256": digest.hexdigest()}
    if path.suffix == ".jsonl":
        require(not before[2] or last == b"\n", f"Incomplete final JSONL row: {path}")
        result["lines"] = lines
    return result


def freeze_inventory(external=EXTERNAL):
    """Exhaustive deterministic listing; detect mutations across the entire scan."""
    external = Path(external)
    roots = [external / name for name in FINAL_ROOTS]
    started = time.monotonic()
    files = external_inventory(roots)
    require(all(any(path.is_relative_to(root) for path in files) for root in roots), "An external family is empty")
    before = {path: stat_identity(path) for path in files}
    records = [file_record(path) for path in sorted(files)]
    require(external_inventory(roots) == files, "External file set changed during inventory")
    require(all(stat_identity(path) == identity for path, identity in before.items()),
            "External artifacts changed during inventory")
    return {"schema_version": 1, "complete": True, "created_utc": utc(),
            "roots": list(map(str, roots)), "files": records, "file_count": len(records),
            "total_bytes": sum(item["bytes"] for item in records),
            "elapsed_seconds": time.monotonic() - started, "jsonl_counts_are_actual": True,
            "file_set_and_metadata_stable": True, "symlinks_allowed": False}


def completion_evidence(project=PROJECT, external=EXTERNAL):
    """Do not freeze active/incomplete work or an explicit active failure marker."""
    project, external = Path(project), Path(external)
    out, run = project / "eval/sft_run", external / "runs/sft_20k_20261117"
    require(not (out / "training_stop.json").exists(), "Active training_stop marker prevents finalization")
    paths = [out / name for name in ("preflight.json", "batch_bench.json", "train_run_config.json",
                                    "cpu_baseline_results.json", "model_rollout_run.json", "model_inference_preflight.json")]
    paths += [run / "worker_result.json", run / "continuation_status.json"]
    values = {path.name: load(path) for path in paths}
    batch = values["batch_bench.json"]
    require(batch["status"] == "ok" and sorted(item["batch_size"] for item in batch["trials"]) == [1, 4, 8],
            "All three final batch configurations must be complete")
    for item in batch["trials"]:
        require(item["status"] == "ok" and item["optimizer_steps"] == 30 and item["consumed_samples"] == 480
                and item.get("oom") is False and item.get("nonfinite") is False, "Incomplete or failed batch trial")
    config, worker = values["train_run_config.json"], values["worker_result.json"]
    require(config["status"] == "ok" and worker["status"] == "ok", "Formal training not complete")
    for runtime in (config["runtime"], worker):
        validate_training_counts(runtime["optimizer_steps"], runtime["consumed_samples"], runtime["epoch"])
        checkpoint_steps(runtime["checkpoints"])
        require(runtime["oom"] is False and runtime["nonfinite"] is False, "Formal training recorded OOM/nonfinite")
    require(config["runtime"] == worker, "Formal worker result differs from training config runtime")
    expected_checkpoints = {run / f"checkpoints/checkpoint-{step}" for step in (500, 1000, 1500, 2000, 2500)}
    require({Path(path) for path in worker["checkpoints"]} == expected_checkpoints, "Unexpected checkpoint output paths")
    for directory in expected_checkpoints:
        for name in ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json"):
            path = directory / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                    f"Required checkpoint artifact is missing: {path}")
    model = values["model_rollout_run.json"]
    expected = values["model_inference_preflight.json"]
    require(expected["status"] == "ok" and expected["expected_trajectories_per_strategy"] == 1494
            and expected["expected_outputs_per_strategy"] > 0, "Saved-step inference preflight is incomplete")
    require(model["status"] == "complete" and set(model["strategies"]) == set(MODELS), "Model evaluation is incomplete")
    require(all(item["status"] == "complete" and item["n_trajectories"] == 1494
                and item["n_steps"] == expected["expected_outputs_per_strategy"]
                for item in model["strategies"].values()), "Incomplete model strategy")
    cpu = values["cpu_baseline_results.json"]
    require(set(cpu["strategies"]) == {"no_compress", "latest", "bm25", "oracle"}
            and all(item["combined"]["n_tasks"] == 1494 for item in cpu["strategies"].values()),
            "CPU baselines are incomplete")
    watcher = values["continuation_status.json"]
    require(watcher["status"] == "evaluation_complete" and watcher["exit_code"] == 0,
            "Continuation watcher has not completed successfully")
    return {"passed": True, "records": [file_record(path) for path in paths],
            "formal_optimizer_steps": 2500, "formal_sample_exposures": 40000, "formal_epochs": 2,
            "checkpoint_count": 5, "model_strategies": len(MODELS), "cpu_strategies": 4}


def pid_source_paths(project=PROJECT, external=EXTERNAL):
    """Select lifecycle files, avoiding model tokenizer vocabulary keys like pid."""
    directories = [Path(project) / "eval/sft_run", *(Path(external) / name for name in FINAL_ROOTS)]
    return sorted({path for directory in directories for path in directory.rglob("*.json")
                   if path.name in PID_NAMES or LAUNCH_NAME.fullmatch(path.name)})


def parse_utc(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def collect_pid_identities(paths):
    identities = defaultdict(list)
    for path in paths:
        path = Path(path)
        data = load(path)
        observed = [parse_utc(data.get(name)) for name in ("utc", "started_at_utc", "created_utc")]
        observed.append(parse_utc(data.get("initial_gpu", {}).get("utc")))
        observed = [value for value in observed if value is not None]
        when = min(observed) if observed else path.stat().st_mtime
        basis = "recorded_utc" if observed else "record_file_mtime_upper_bound"

        def walk(value, position=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    location = f"{position}.{key}" if position else key
                    if (key == "pid" or key.endswith("_pid")) and key not in {"parent_pid", "ppid"}:
                        require(type(item) is int and item > 0, f"Invalid PID at {path}:{location}")
                        identities[item].append({"path": str(path), "field": location,
                                                 "observed_at_unix": when, "time_basis": basis})
                    elif isinstance(item, (dict, list)):
                        walk(item, location)
            elif isinstance(value, list):
                for number, item in enumerate(value):
                    walk(item, f"{position}[{number}]")
        walk(data)
    require(bool(identities), "No recorded Task 05 PID identities found")
    return dict(identities)


def process_snapshot(pid, proc_root=Path("/proc"), clock_ticks=None):
    """Read only this exact PID; never signal or use process-name matching tools."""
    proc_root = Path(proc_root)
    path = proc_root / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        state, start_ticks = fields[0], int(fields[19])
        boot = next(int(line.split()[1]) for line in (proc_root / "stat").read_text().splitlines()
                    if line.startswith("btime "))
        tokens = (path / "cmdline").read_bytes().split(b"\0")
        command = [token.decode(errors="replace") for token in tokens if token]
        return {"pid": pid, "exists": True, "state": state, "start_ticks": start_ticks,
                "start_unix": boot + start_ticks / (clock_ticks or os.sysconf("SC_CLK_TCK")), "cmdline": command}
    except FileNotFoundError:
        return {"pid": pid, "exists": False}


def classify_process(identity, snapshot):
    """Distinguish exited/zombie, active Task 05, and independently proven PID reuse."""
    result = {**snapshot, "recorded_identity": identity}
    if not snapshot["exists"]:
        return {**result, "active": False, "classification": "exited"}
    if snapshot["state"] in ("Z", "X"):
        return {**result, "active": False, "classification": "zombie_or_dead"}
    task_command = any(Path(token).name in TASK_COMMANDS for token in snapshot["cmdline"])
    observed = min(row["observed_at_unix"] for row in identity)
    newer_identity = snapshot["start_unix"] > observed + 1.0
    if newer_identity and not task_command:
        return {**result, "active": False, "classification": "pid_reused_by_unrelated_process",
                "reuse_proof": "current /proc start time is after historical PID observation; cmdline is unrelated"}
    return {**result, "active": True,
            "classification": "active_task05" if task_command else "unresolved_live_historical_pid",
            "identity_note": "No signal sent; unresolved identities conservatively block finalization"}


def verify_processes(project=PROJECT, external=EXTERNAL, proc_root=Path("/proc")):
    paths = pid_source_paths(project, external)
    identities = collect_pid_identities(paths)
    checks = [classify_process(identities[pid], process_snapshot(pid, proc_root)) for pid in sorted(identities)]
    active = [item for item in checks if item["active"]]
    return {"passed": not active, "recorded_pid_count": len(checks), "checks": checks,
            "active": active, "sources": [file_record(path) for path in paths]}


def verify_gpu_history(project=PROJECT, external=EXTERNAL):
    """Use actual visible-device settings, UUIDs and recorded resource samples.

This is a provenance check of the recorded stages, not a claim of continuous
system-wide GPU process tracing. It includes the zero-step A8 initialization.
"""
    project, external = Path(project), Path(external)
    out = project / "eval/sft_run"
    baseline = load(out / "preflight.json")
    require(baseline["gpu"]["physical_index"] == 1, "Task 05 baseline does not identify physical GPU 1")
    expected_uuid, stages, evidence = baseline["gpu"]["uuid"], [], []
    workers = sorted((external / "runs").glob("sft_20k_20261117/worker_config.json"))
    workers += sorted((external / "runs/batch_bench_20261117").rglob("worker_config.json"))
    required = {external / f"runs/batch_bench_20261117/batch_{batch}/worker_config.json" for batch in (1, 4, 8)}
    required.add(external / "runs/sft_20k_20261117/worker_config.json")
    require(required <= set(workers), "Missing a measured worker GPU configuration")
    for path in workers:
        config = load(path)
        started_path = path.parent / "worker_started.json"
        started = load(started_path) if started_path.exists() else {}
        initial = config.get("initial_gpu", started.get("initial_gpu", {}))
        require(config.get("cuda_visible_devices") == "1", f"Worker visible GPU not proven: {path}")
        require(initial.get("index") == 1 and initial.get("uuid") == expected_uuid,
                f"Worker initial physical GPU mismatch: {path}")
        if "gpu_uuid" in config:
            require(config["gpu_uuid"] == expected_uuid, f"Worker UUID changed: {path}")
        resources_path = path.parent / "resources.jsonl"
        count = 0
        if resources_path.exists():
            with resources_path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if "monitor_error" in row:
                        raise ValueError(f"Recorded worker monitor error: {resources_path}")
                    require(row.get("index") == 1 and row.get("uuid") == expected_uuid,
                            f"Worker monitor recorded another GPU: {resources_path}")
                    count += 1
            evidence.append(file_record(resources_path))
        require(count > 0, f"Worker GPU monitor has no recorded samples: {path}")
        evidence.append(file_record(path))
        if started_path.exists():
            evidence.append(file_record(started_path))
        stages.append({"worker_config": str(path), "cuda_visible_devices": "1", "gpu_uuid": expected_uuid,
                       "resource_samples": count, "archived_initialization": "initialization_attempts" in path.parts})
    model_path = out / "model_rollout_run.json"
    model = load(model_path)
    require(model.get("cuda_visible_devices") == "1" and model.get("gpu_before", {}).get("index") == 1,
            "Model evaluation GPU visibility/initial device mismatch")
    monitor_path, count = external / "eval_rollouts/model_resources.jsonl", 0
    with monitor_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row.get("gpu", {}).get("index") == 1, "Model monitor recorded another GPU")
            count += 1
    require(count > 0, "Model monitor has no GPU samples")
    evidence += [file_record(model_path), file_record(monitor_path)]
    return {"passed": True, "gpu_index": 1, "gpu_uuid": expected_uuid, "worker_stages": stages,
            "model_resource_samples": count, "model_cuda_visible_devices": "1", "sources": evidence,
            "evidence_scope": "all recorded worker visibility/physical UUID and GPU monitor samples, plus model visibility/device samples; no continuous system-wide process-tracing claim"}


def current_machine_snapshot():
    gpu = subprocess.check_output(["nvidia-smi", "-i", "1",
        "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    values = [value.strip() for value in gpu.strip().split(",")]
    require(len(values) == 4, "Unexpected nvidia-smi GPU response")
    gpu_row = {"index": int(values[0]), "uuid": values[1], "memory_used_mib": int(values[2]),
               "utilization_percent": int(values[3])}
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                                      "--format=csv,noheader,nounits"], text=True)
    processes = []
    for line in output.splitlines():
        if not line.strip() or "No running processes found" in line:
            continue
        fields = [value.strip() for value in line.split(",")]
        require(len(fields) == 3 and fields[0].isdigit(), "Unexpected nvidia-smi compute process response")
        processes.append({"pid": int(fields[0]), "gpu_uuid": fields[1], "used_memory_mib": fields[2]})
    df = subprocess.check_output(["df", "-B1", "--output=source,size,used,avail,pcent,target", "/"], text=True)
    parts = df.splitlines()[-1].split()
    return {"utc": utc(), "gpu1": gpu_row, "compute_processes": processes, "df_raw": df,
            "root_used_bytes": int(parts[2]), "root_available_bytes": int(parts[3])}


def resource_record(preflight, processes, history, machine):
    """Pure acceptance gate; callers only write its successful return value."""
    require(processes["passed"] and not processes["active"], "Recorded Task 05 process/watcher is still active")
    require(history["passed"], "Recorded GPU 1 provenance failed")
    gpu = machine["gpu1"]
    require(gpu["index"] == 1 and gpu["uuid"] == preflight["gpu"]["uuid"] == history["gpu_uuid"],
            "Current GPU 1 identity differs from preflight")
    require(gpu["memory_used_mib"] < 2048, "GPU 1 is not released below 2 GiB")
    delta = machine["root_used_bytes"] - preflight["root_df_used_bytes"]
    require(preflight["root_final_growth_limit_bytes"] == 300000000 and delta <= 300000000,
            "Final root growth exceeds 300 MB")
    task_pids = {item["pid"] for item in processes["checks"]
                 if item["classification"] != "pid_reused_by_unrelated_process"}
    remaining = [item for item in machine["compute_processes"] if item["pid"] in task_pids]
    require(not remaining, "An exact recorded Task 05 PID still appears in GPU compute processes")
    return {"schema_version": 1, "status": "complete", "utc": machine["utc"],
            "root_final_used_bytes": machine["root_used_bytes"], "root_baseline_used_bytes": preflight["root_df_used_bytes"],
            "root_delta_bytes": delta, "root_limit_bytes": 300000000, "root_limit_passed": True,
            "df": machine["df_raw"], "gpu1_after": gpu, "gpu_released": True, "only_gpu1": True,
            "task05_gpu_processes_remaining": [], "recorded_task05_processes_alive": [],
            "process_checks": processes, "gpu_identity_evidence": history,
            "current_compute_processes": machine["compute_processes"],
            "formal_optimizer_steps": 2500, "formal_sample_exposures": 40000,
            "checkpoint_count": 5, "signals_sent": 0}


def atomic_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finalize(command, project=PROJECT, external=EXTERNAL):
    project, external = Path(project), Path(external)
    require(command in ("all", "inventory", "resources"), "Unknown finalization command")
    out = project / "eval/sft_run"
    completion = completion_evidence(project, external)
    processes = verify_processes(project, external)
    require(processes["passed"], "Recorded Task 05 PID/watcher still active")
    history = verify_gpu_history(project, external)
    preflight = load(out / "preflight.json")
    resource_record(preflight, processes, history, current_machine_snapshot())
    inventory = freeze_inventory(external) if command in ("all", "inventory") else None
    # Hashing can take minutes: recheck lifecycle files, exact process identities,
    # current GPU and disk before publishing either success record.
    require(completion_evidence(project, external) == completion, "Completion evidence changed during finalization")
    fresh_processes = verify_processes(project, external)
    require(fresh_processes["sources"] == processes["sources"], "Process evidence changed during finalization")
    resources = resource_record(preflight, fresh_processes, history, current_machine_snapshot())
    resources["completion_evidence"] = completion
    resources["finalizer_sha256"] = file_record(__file__)["sha256"]
    if inventory is not None:
        inventory["completion_evidence"] = completion
        inventory["finalizer_sha256"] = resources["finalizer_sha256"]
        atomic_write(out / "external_artifact_manifest.json", inventory)
    if command in ("all", "resources"):
        atomic_write(out / "final_resource_check.json", resources)
    return {"status": "complete", "command": command, "external_files": inventory["file_count"] if inventory else None,
            "recorded_pids": fresh_processes["recorded_pid_count"], "root_delta_bytes": resources["root_delta_bytes"],
            "gpu1_memory_used_mib": resources["gpu1_after"]["memory_used_mib"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("all", "inventory", "resources"), default="all")
    args = parser.parse_args(argv)
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("run only on 5880 formal project with CUDA_VISIBLE_DEVICES='' (CPU only)")
    try:
        result = finalize(args.command)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "failed", "command": args.command, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
