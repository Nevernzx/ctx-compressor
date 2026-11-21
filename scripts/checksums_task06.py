#!/usr/bin/env python3
"""Verify historical artifacts, index current delivery, and measure task-owned overlay files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import checksums_task04 as hashes
from scripts import checksums_task05 as v1

OUT = PROJECT / "eval/sft_run_v2"
EXTERNAL = Path("/root/userfolder/ctx-compressor")
DOCS = {"README.md", "NOTES.md", "HANDOFF.md"}
EXTERNAL_DIRS = ("data/traj_train_v2", "data/sft_v2", "runs/sft_20k_v2_20261118", "eval_rollouts_v2")


def read(path):
    return json.loads(path.read_text())


def atomic(path, content):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content)
    temp.replace(path)


def protect_history(full_external=False):
    previous = hashes.read_sums(OUT / "TASK05_SHA256SUMS")
    baselines = read(OUT / "preflight_v2.json")["task05_document_baselines"]
    if hashes.digest(OUT / "TASK05_SHA256SUMS") != baselines["SHA256SUMS"]["sha256"]:
        raise ValueError("Archived Task 05 SHA256SUMS changed")
    for name, expected in previous.items():
        path = PROJECT / name
        if name in DOCS:
            with path.open("rb") as stream:
                actual = hashlib.sha256(stream.read(baselines[name]["bytes"])).hexdigest()
        else:
            actual = hashes.digest(path)
        if actual != expected:
            raise ValueError(f"Task 05 file/prefix mismatch: {name}")
    legacy = v1.verify_legacy()
    record = {"task05_files": len(previous), "protected_prefixes": sorted(DOCS),
              "historical_tasks": list(legacy["tasks"]) + ["TASK_05"]}
    if full_external:
        record["task05_external_files"] = v1.verify_final_artifacts(slow=True)["file_count"]
        record["task04_external_files"] = hashes.verify_external(slow=True)["file_count"]
    return record


def task06_files():
    # Other experiments may be active on this checkout. Enumerate only v2 deliverables.
    files = {path for path in OUT.rglob("*") if path.is_file() and not path.is_symlink()}
    patterns = {"scripts": ("*sft*v2*.py", "*sft*v2*.sh", "*heldout*v2*.py", "*heldout*v2*.sh",
                            "checksums_task06.py", "audit_final_rollout_v2.py"),
                "src": ("oracle_noise_v2.py",),
                "tests": ("test_oracle_noise_v2.py", "test_sft_v2_training.py", "test_heldout_v2.py"),
                "specs": ("TASK_06_*.md",)}
    for directory, globs in patterns.items():
        for pattern in globs:
            files.update(p for p in (PROJECT / directory).glob(pattern) if p.is_file() and not p.is_symlink())
    return files


def disk_usage():
    used = subprocess.check_output(["df", "-B1", "--output=used", "/"], text=True)
    return int(used.splitlines()[-1])


def resource_measurement():
    marker = OUT / "start.marker"
    root_dev = Path("/").stat().st_dev
    marker_ns = marker.stat().st_mtime_ns
    # -xdev excludes all external data, model weights, logs and configured caches.
    found = subprocess.run(["find", "/", "-xdev", "(", "-name", ".git", "-o", "-name", "grpo_prep",
                            ")", "-prune", "-o", "-type", "f", "-newer", str(marker), "-print0"],
                           check=True, capture_output=True)
    paths = {Path(os.fsdecode(name)) for name in found.stdout.split(b"\0") if name}
    candidates = task06_files()
    candidates.update(PROJECT / name for name in DOCS | {"SHA256SUMS"}
                      if (PROJECT / name).stat().st_mtime_ns >= marker_ns)
    candidates.add(marker)
    for path in paths:
        if any("task06" in part.lower() or "sft_20k_v2" in part.lower()
               or "eval_rollouts_v2" in part.lower() for part in path.parts):
            candidates.add(path)
    # Count full regular-file sizes rather than the byte delta of edited files.
    rows = []
    for path in sorted(candidates):
        if path.is_symlink() or not path.is_file() or path.stat().st_dev != root_dev:
            continue
        rows.append({"path": str(path), "bytes": path.stat().st_size})
    total = sum(x["bytes"] for x in rows)
    preflight = read(OUT / "preflight_v2.json")
    raw = disk_usage()
    return {"utc": datetime.now(timezone.utc).isoformat(), "marker": str(marker),
            "method": "find / -xdev -type f -newer marker; Task 06 delivery paths, shared documents, task-named paths; distinct other-task output excluded",
            "overlay_files": rows, "task_overlay_bytes": total, "limit_bytes": 300000000,
            "task_overlay_within_limit": total <= 300000000, "raw_df_start_bytes": preflight["root_baseline_used_bytes"],
            "raw_df_current_bytes": raw, "raw_df_delta_bytes": raw - preflight["root_baseline_used_bytes"],
            "raw_df_used_for_acceptance": False,
            "other_recent_regular_paths_n": len(paths - candidates)}


def external_inventory():
    # Only called after the detached pipeline has exited. Large payloads remain on 5880.
    rows = []
    for relative in EXTERNAL_DIRS:
        directory = EXTERNAL / relative
        if not directory.is_dir():
            raise ValueError(f"missing output directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"unexpected symlink: {path}")
            if path.is_file():
                st = path.stat()
                h = hashes.digest(path)
                end = path.stat()
                if (st.st_size, st.st_mtime_ns) != (end.st_size, end.st_mtime_ns):
                    raise ValueError(f"file changed while hashing: {path}")
                rows.append({"path": str(path), "bytes": st.st_size, "sha256": h})
    return {"files": rows, "file_count": len(rows), "external_root": str(EXTERNAL)}


def final_resources():
    run = EXTERNAL / "runs/sft_20k_v2_20261118"
    status = read(run / "continuation_status.json")
    worker = read(run / "worker_result.json")
    rollout = read(OUT / "model_rollout_run.json")
    monitor = [json.loads(line) for line in (run / "resources.jsonl").read_bytes().splitlines()]
    polls = [json.loads(line) for line in (run / "card_polls.jsonl").read_bytes().splitlines()]
    uuid = worker["initial_gpu"]["uuid"]
    if (status["status"] != "evaluation_complete" or status["child_exit_code"] != 0
            or worker["oom"] or worker["nonfinite"] or not rollout["engine_shutdown"]["completed"]
            or rollout["cuda_visible_devices"] != "1" or rollout["gpu_before"]["index"] != 1
            or not all(row["index"] == 1 and row["uuid"] == uuid for row in monitor + polls)):
        raise ValueError("Incomplete training, evaluation, or GPU identity proof")
    def query(arguments):
        return subprocess.check_output(["nvidia-smi", "-i", "1", arguments, "--format=csv,noheader"], text=True).strip()
    sample = query("--query-gpu=index,uuid,memory.used,utilization.gpu")
    compute = query("--query-compute-apps=pid")
    job_pids = [status[key] for key in ("waiter_pid", "training_pid", "evaluating_pid")]
    live = [pid for pid in job_pids if str(pid) in compute.splitlines()]
    if live or sample.split(", ")[1] != uuid:
        raise ValueError("GPU 1 release check failed")
    durations = [(datetime.fromisoformat(b["utc"]) - datetime.fromisoformat(a["utc"])).total_seconds()
                 for a, b in zip(monitor, monitor[1:])]
    record = resource_measurement()
    record["raw_df_end_text"] = subprocess.check_output(["df", "-B1", "/"], text=True).strip()
    record["raw_df_start_source"] = str(OUT / "preflight_v2.json")
    record["gpu"] = {"physical_index": 1, "uuid": uuid, "post_run_readonly_query": sample,
                     "observed_compute_pids": compute.splitlines(), "task_compute_pids_present": live,
                     "pipeline_exit_code": status["child_exit_code"], "engine_shutdown": rollout["engine_shutdown"],
                     "monitor_sample_count": len(monitor), "waiter_poll_count": len(polls),
                     "max_monitor_gap_seconds": max(durations), "cpu_closeout": True,
                     "torch_peak_allocated_bytes": worker["peak_allocated_bytes"],
                     "torch_peak_reserved_bytes": worker["peak_reserved_bytes"],
                     "eval_peak_card_used_mib": rollout["peak_gpu_memory_used_mib"]}
    # Count this receipt itself. Its numeric length stabilizes after a few iterations.
    path = OUT / "final_resource_v2.json"
    rows = record["overlay_files"]
    other = sorted((r for r in rows if r["path"] != str(path)), key=lambda r: r["path"])
    fixed_bytes = sum(r["bytes"] for r in other)
    size = 0
    while True:
        record["overlay_files"] = sorted([*other, {"path": str(path), "bytes": size}], key=lambda r: r["path"])
        record["task_overlay_bytes"] = fixed_bytes + size
        record["task_overlay_within_limit"] = record["task_overlay_bytes"] <= record["limit_bytes"]
        content = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        actual = len(content.encode())
        if size == actual:
            break
        size = actual
    atomic(path, content)
    if not record["task_overlay_within_limit"]:
        raise ValueError("Task 06 exceeds the overlay file budget")
    return {"task_overlay_bytes": record["task_overlay_bytes"], "overlay_files": len(other) + 1,
            "raw_start": record["raw_df_start_bytes"], "raw_end": record["raw_df_current_bytes"],
            "gpu": sample, "compute": compute.splitlines()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("history", "write", "check", "resources", "final-resources", "external"))
    parser.add_argument("--verify-old-external", action="store_true")
    args = parser.parse_args()
    if PROJECT != Path("/root/workplace/ctx-compressor"):
        parser.error("run on the formal 5880 project")
    if args.command == "final-resources":
        return print(json.dumps(final_resources(), ensure_ascii=False))
    if args.command == "resources":
        return print(json.dumps(resource_measurement(), ensure_ascii=False))
    if args.command == "external":
        status = read(EXTERNAL / "runs/sft_20k_v2_20261118/continuation_status.json")
        if status.get("status") not in {"completed", "complete", "evaluation_complete", "finished"}:
            raise ValueError("external files can still be changing; wait for pipeline completion")
        inventory = external_inventory()
        atomic(OUT / "external_manifest_v2.json", json.dumps(inventory, indent=2) + "\n")
        return print(json.dumps({"file_count": inventory["file_count"]}))
    result = protect_history(args.verify_old_external)
    if args.command in ("write", "check"):
        dest = PROJECT / "SHA256SUMS"
        # Preserve unrelated entries when multiple tasks share the top-level manifest.
        existing = hashes.read_sums(dest)
        legacy = hashes.read_sums(OUT / "TASK05_SHA256SUMS")
        names = set(legacy) | {p.relative_to(PROJECT).as_posix() for p in task06_files()}
        actual = {name: hashes.digest(PROJECT / name) for name in sorted(names)}
        if args.command == "write":
            merged = {**existing, **actual}
            atomic(dest, "".join(f"{merged[name]}  {name}\n" for name in sorted(merged)))
        else:
            hashes.check_current(actual, {name: existing[name] for name in names if name in existing})
        result["project_files"] = len(actual)
        result["other_manifest_entries_preserved"] = len(existing.keys() - names)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
