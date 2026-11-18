"""Completed measurements remain deliverable without weakening resource gates."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import export_sft_resource_failure as export
from scripts import finalize_sft_artifacts as final
from scripts import sft_run_report as report
from scripts.checksums_task05 import verify_final_artifacts
from test_finalize_sft_artifacts import fixture, inactive, machine, save
from test_sft_run_report import strategy


@pytest.fixture(autouse=True)
def forbid_real_machine_queries(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Fixture tests must never query real GPU/process state")
    monkeypatch.setattr(final.subprocess, "check_output", blocked)
    monkeypatch.setattr(final, "process_snapshot", lambda pid, *args: {"pid": pid, "exists": False})


def over_limit_machine():
    return {**machine(), "root_used_bytes": 1300000001}


def prepare(tmp_path, monkeypatch):
    project, external, baseline = fixture(tmp_path)
    snapshot = over_limit_machine()
    monkeypatch.setattr(final, "current_machine_snapshot", lambda: deepcopy(snapshot))
    return project, external, baseline, snapshot


def outputs(project):
    out = project / "eval/sft_run"
    return out / "final_resource_check.json", out / "external_artifact_manifest.json"


@pytest.mark.parametrize("mutation", ["active_pid", "history_failed", "wrong_uuid", "wrong_index",
                                      "gpu_busy", "remaining_gpu_pid", "changed_limit"])
def test_excess_root_never_masks_another_failed_gate(tmp_path, mutation):
    project, external, baseline = fixture(tmp_path)
    history, processes, snapshot = final.verify_gpu_history(project, external), inactive(), over_limit_machine()
    if mutation == "active_pid":
        processes.update(passed=False, active=[{"pid": 200}])
    elif mutation == "history_failed":
        history["passed"] = False
    elif mutation == "wrong_uuid":
        snapshot["gpu1"]["uuid"] = "GPU-other"
    elif mutation == "wrong_index":
        snapshot["gpu1"]["index"] = 0
    elif mutation == "gpu_busy":
        snapshot["gpu1"]["memory_used_mib"] = 2048
    elif mutation == "remaining_gpu_pid":
        snapshot["compute_processes"] = [{"pid": 200, "gpu_uuid": "GPU-correct"}]
    else:
        baseline["root_final_growth_limit_bytes"] += 1
    with pytest.raises(ValueError):
        export.failed_resource_record(baseline, processes, history, snapshot)


@pytest.mark.parametrize("delta", [299999999, 300000000])
def test_root_within_original_limit_refuses_failure_export_before_inventory(tmp_path, monkeypatch, delta):
    project, external, baseline, snapshot = prepare(tmp_path, monkeypatch)
    snapshot["root_used_bytes"] = baseline["root_df_used_bytes"] + delta
    def forbidden(*args):
        raise AssertionError("Inventory must not start when root gate passes")
    monkeypatch.setattr(final, "freeze_inventory", forbidden)
    with pytest.raises(ValueError, match="use scripts/finalize_sft_artifacts.py"):
        export.export_failed_acceptance(project, external)
    assert not any(path.exists() for path in outputs(project))


def test_failed_export_preserves_raw_values_and_passes_inventory_integrity(tmp_path, monkeypatch):
    project, external, _, snapshot = prepare(tmp_path, monkeypatch)
    first = export.export_failed_acceptance(project, external)
    resources, inventory = [final.load(path) for path in outputs(project)]
    assert first["status"] == resources["status"] == "failed" and first["export_complete"]
    assert resources["root_delta_bytes"] == 300000001 and resources["root_limit_bytes"] == 300000000
    assert resources["root_limit_passed"] is False and resources["unknown_growth_subtracted"] is False
    assert resources["df"] == snapshot["df_raw"] and resources["signals_sent"] == 0
    assert resources["non_root_gates_verified_before_and_after_inventory"] is True
    assert resources["failure_reasons"] == [export.ROOT_FAILURE]
    assert resources["gpu_released"] and resources["only_gpu1"] and not resources["task05_gpu_processes_remaining"]
    assert inventory["complete"] and inventory["task_acceptance_status"] == "failed"
    assert verify_final_artifacts(project, external, slow=True)["coverage_complete"]
    second = export.export_failed_acceptance(project, external)
    assert second["status"] == "failed" and final.load(outputs(project)[1])["files"] == inventory["files"]


@pytest.mark.parametrize("mutation", ["root_recovers", "gpu_busy", "wrong_uuid", "remaining_gpu_pid",
                                      "pid_active", "watcher_changes", "history_changes", "pid_sources_change"])
def test_all_gates_are_rechecked_after_inventory_before_any_publication(tmp_path, monkeypatch, mutation):
    project, external, _, snapshot = prepare(tmp_path, monkeypatch)
    real_inventory = final.freeze_inventory
    def change_after_hashing(root):
        inventory = real_inventory(root)
        if mutation == "root_recovers":
            snapshot["root_used_bytes"] = 1300000000
        elif mutation == "gpu_busy":
            snapshot["gpu1"]["memory_used_mib"] = 2048
        elif mutation == "wrong_uuid":
            snapshot["gpu1"]["uuid"] = "GPU-other"
        elif mutation == "remaining_gpu_pid":
            snapshot["compute_processes"] = [{"pid": 200}]
        elif mutation == "pid_active":
            monkeypatch.setattr(final, "process_snapshot", lambda pid, *args: {
                "pid": pid, "exists": True, "state": "S", "start_unix": 0,
                "cmdline": ["python", "train_sft_worker.py"]})
        elif mutation == "watcher_changes":
            save(external / "runs/sft_20k_20261117/continuation_status.json", {"status": "evaluating", "exit_code": 0})
        elif mutation == "history_changes":
            path = external / "eval_rollouts/model_resources.jsonl"
            value = final.load(path)
            value["new_metadata"] = "late modification"
            save(path, value)  # Valid GPU index still cannot hide changed history.
        else:
            save(external / "runs/sft_20k_20261117/evaluation_launch_late.json", {"pid": 333})
        return inventory
    monkeypatch.setattr(final, "freeze_inventory", change_after_hashing)
    with pytest.raises(ValueError):
        export.export_failed_acceptance(project, external)
    assert not any(path.exists() for path in outputs(project))


@pytest.mark.parametrize("mutation", ["incomplete_train", "incomplete_model", "watcher_running", "wrong_history"])
def test_incomplete_work_and_gpu_history_block_before_inventory(tmp_path, monkeypatch, mutation):
    project, external, _, _ = prepare(tmp_path, monkeypatch)
    if mutation == "incomplete_train":
        path = external / "runs/sft_20k_20261117/worker_result.json"
        value = final.load(path); value["status"] = "running"; save(path, value)
    elif mutation == "incomplete_model":
        path = project / "eval/sft_run/model_rollout_run.json"
        value = final.load(path); value["status"] = "running"; save(path, value)
    elif mutation == "watcher_running":
        save(external / "runs/sft_20k_20261117/continuation_status.json", {"status": "evaluating", "exit_code": 0})
    else:
        path = external / "runs/batch_bench_20261117/initialization_attempts/batch_8_missing_length/resources.jsonl"
        value = final.load(path); value["uuid"] = "GPU-other"; save(path, value)
    def forbidden(*args):
        raise AssertionError("Inventory must not start for another failed gate")
    monkeypatch.setattr(final, "freeze_inventory", forbidden)
    with pytest.raises(ValueError):
        export.export_failed_acceptance(project, external)
    assert not any(path.exists() for path in outputs(project))


def test_proven_unrelated_reused_pid_remains_excluded(tmp_path):
    project, external, baseline = fixture(tmp_path)
    processes, snapshot = inactive(), over_limit_machine()
    processes["checks"][0]["classification"] = "pid_reused_by_unrelated_process"
    snapshot["compute_processes"] = [{"pid": 200, "gpu_uuid": "GPU-unrelated"}]
    value = export.failed_resource_record(baseline, processes, final.verify_gpu_history(project, external), snapshot)
    assert value["task05_gpu_processes_remaining"] == [] and value["status"] == "failed"


def test_report_keeps_checkpoint_selection_when_resource_export_is_failed(tmp_path, monkeypatch):
    project, external, _, _ = prepare(tmp_path, monkeypatch)
    export.export_failed_acceptance(project, external)
    report_dir = tmp_path / "report"
    save(report_dir / "final_resource_check.json", final.load(outputs(project)[0]))
    for step in report.STEPS:
        save(report_dir / f"checkpoint-{step}_results.json", strategy(f"checkpoint-{step}"))
    result = report.build_report_data(report_dir)
    assert result["status"] == "failed" and result["selected_checkpoint"]["checkpoint_step"] == 500
    assert len(result["strategies"]) == 5 and any("300MB" in reason for reason in result["failure_reasons"])


@pytest.mark.parametrize("succeeds", [True, False])
def test_cli_always_returns_one_and_distinguishes_export_completion(monkeypatch, capsys, succeeds):
    monkeypatch.setattr(export, "PROJECT", Path("/root/workplace/ctx-compressor"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    def run():
        if not succeeds:
            raise ValueError("Root growth is within 300 MB; use strict finalizer")
        return {"status": "failed", "task_acceptance_status": "failed", "export_complete": True}
    monkeypatch.setattr(export, "export_failed_acceptance", run)
    assert export.main([]) == 1
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "failed" and value["export_complete"] is succeeds
