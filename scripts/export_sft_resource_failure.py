#!/usr/bin/env python3
"""Export completed Task 05 artifacts when only the raw root-disk gate fails.

The strict finalizer remains unchanged. This command requires all its completion,
process and GPU gates before and after inventory. It publishes failed acceptance
evidence, never success, and returns 1 even when both exports are written.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import finalize_sft_artifacts as final

EXTERNAL = final.EXTERNAL
ROOT_LIMIT = 300000000
ROOT_FAILURE = "Final root growth exceeds 300 MB"
require = final.require


def failed_resource_record(preflight, processes, history, machine):
    """Require every non-root gate before accepting the sole failed root gate."""
    require(processes["passed"] and not processes["active"],
            "Recorded Task 05 process/watcher is still active")
    require(history["passed"], "Recorded GPU 1 provenance failed")
    gpu = machine["gpu1"]
    require(preflight["gpu"]["physical_index"] == history["gpu_index"] == gpu["index"] == 1
            and gpu["uuid"] == preflight["gpu"]["uuid"] == history["gpu_uuid"],
            "Current GPU 1 identity differs from preflight/history")
    require(type(gpu["memory_used_mib"]) is int and 0 <= gpu["memory_used_mib"] < 2048,
            "GPU 1 is not released below 2 GiB")
    task_pids = {item["pid"] for item in processes["checks"]
                 if item["classification"] != "pid_reused_by_unrelated_process"}
    remaining = [item for item in machine["compute_processes"] if item["pid"] in task_pids]
    require(not remaining, "An exact recorded Task 05 PID still appears in GPU compute processes")
    require(type(preflight["root_final_growth_limit_bytes"]) is int
            and preflight["root_final_growth_limit_bytes"] == ROOT_LIMIT,
            "Original root growth limit must remain 300000000 bytes")
    require(type(machine["root_used_bytes"]) is int and type(preflight["root_df_used_bytes"]) is int,
            "Raw root filesystem measurements must be integer bytes")
    delta = machine["root_used_bytes"] - preflight["root_df_used_bytes"]
    require(delta > ROOT_LIMIT,
            "Root growth is within 300 MB; use scripts/finalize_sft_artifacts.py instead")
    return {"schema_version": 1, "status": "failed", "task_acceptance_status": "failed",
            "failure_reasons": [ROOT_FAILURE], "non_root_gates_passed": True,
            "utc": machine["utc"], "root_final_used_bytes": machine["root_used_bytes"],
            "root_baseline_used_bytes": preflight["root_df_used_bytes"], "root_delta_bytes": delta,
            "root_limit_bytes": ROOT_LIMIT, "root_limit_passed": False,
            "unknown_growth_subtracted": False, "df": machine["df_raw"],
            "gpu1_after": gpu, "gpu_released": True, "only_gpu1": True,
            "task05_gpu_processes_remaining": [], "recorded_task05_processes_alive": [],
            "process_checks": processes, "gpu_identity_evidence": history,
            "current_compute_processes": machine["compute_processes"],
            "formal_optimizer_steps": 2500, "formal_sample_exposures": 40000,
            "checkpoint_count": 5, "signals_sent": 0}


def export_failed_acceptance(project=PROJECT, external=EXTERNAL):
    """Freeze files only for completed work with the sole root-disk failure."""
    project, external = Path(project), Path(external)
    out = project / "eval/sft_run"
    completion = final.completion_evidence(project, external)
    processes = final.verify_processes(project, external)
    history = final.verify_gpu_history(project, external)
    preflight = final.load(out / "preflight.json")
    before_machine = final.current_machine_snapshot()
    failed_resource_record(preflight, processes, history, before_machine)

    inventory = final.freeze_inventory(external)

    # Reuse and repeat all non-root checks after hashing, including GPU history.
    require(final.completion_evidence(project, external) == completion,
            "Completion evidence changed during failure export")
    fresh_processes = final.verify_processes(project, external)
    require(fresh_processes["sources"] == processes["sources"],
            "Process evidence changed during failure export")
    fresh_history = final.verify_gpu_history(project, external)
    require(fresh_history == history, "GPU history changed during failure export")
    fresh_preflight = final.load(out / "preflight.json")
    require(fresh_preflight == preflight, "Preflight baseline changed during failure export")
    resources = failed_resource_record(fresh_preflight, fresh_processes, fresh_history,
                                       final.current_machine_snapshot())
    exporter = final.file_record(__file__)
    strict_finalizer = final.file_record(final.__file__)
    provenance = {"completion_evidence": completion, "exporter_sha256": exporter["sha256"],
                  "strict_finalizer_sha256": strict_finalizer["sha256"]}
    resources.update(provenance)
    resources["pre_inventory_machine_snapshot"] = before_machine
    resources["non_root_gates_verified_before_and_after_inventory"] = True
    inventory.update(provenance)
    inventory.update(task_acceptance_status="failed", resource_acceptance_status="failed",
                     failure_reasons=[ROOT_FAILURE],
                     complete_meaning="Exhaustive stable file inventory only; task acceptance failed.")
    # Publish the failed receipt first: a later I/O failure cannot leave a new
    # inventory paired with a stale successful resource record.
    final.atomic_write(out / "final_resource_check.json", resources)
    final.atomic_write(out / "external_artifact_manifest.json", inventory)
    return {"status": "failed", "task_acceptance_status": "failed", "export_complete": True,
            "failure_reasons": [ROOT_FAILURE], "external_files": inventory["file_count"],
            "root_delta_bytes": resources["root_delta_bytes"], "root_limit_bytes": ROOT_LIMIT,
            "gpu1_memory_used_mib": resources["gpu1_after"]["memory_used_mib"],
            "recorded_pids": fresh_processes["recorded_pid_count"],
            "resource_file": str(out / "final_resource_check.json"),
            "inventory_file": str(out / "external_artifact_manifest.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("run only on 5880 formal project with CUDA_VISIBLE_DEVICES='' (CPU only)")
    try:
        result = export_failed_acceptance()
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "failed", "export_complete": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 1  # Exporting failed acceptance evidence never signals task success.


if __name__ == "__main__":
    raise SystemExit(main())
