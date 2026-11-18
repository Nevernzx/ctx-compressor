"""Finalization never publishes success while work/resources remain active."""
import json
from pathlib import Path

import pytest

from scripts import finalize_sft_artifacts as final
from scripts.checksums_task05 import verify_final_artifacts


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def save(path, value):
    return write(path, json.dumps(value) + "\n")


def fixture(tmp_path):
    project, external = tmp_path / "project", tmp_path / "external"
    out, train = project / "eval/sft_run", external / "runs/sft_20k_20261117"
    bench, rollout = external / "runs/batch_bench_20261117", external / "eval_rollouts"
    gpu = {"index": 1, "uuid": "GPU-correct", "memory_used_mib": 375, "utc": "2026-11-17T12:00:00+00:00"}
    baseline = {"root_df_used_bytes": 1000000000, "root_final_growth_limit_bytes": 300000000,
                "gpu": {"physical_index": 1, "uuid": gpu["uuid"]}}
    save(out / "preflight.json", baseline)
    trials = [{"batch_size": size, "status": "ok", "optimizer_steps": 30, "consumed_samples": 480,
               "oom": False, "nonfinite": False} for size in (1, 4, 8)]
    save(out / "batch_bench.json", {"status": "ok", "trials": trials})
    runtime = {"pid": 200, "status": "ok", "optimizer_steps": 2500, "consumed_samples": 40000, "epoch": 2.0,
               "checkpoints": [str(train / f"checkpoints/checkpoint-{step}") for step in (500, 1000, 1500, 2000, 2500)],
               "oom": False, "nonfinite": False}
    save(out / "train_run_config.json", {"status": "ok", "runtime": runtime})
    save(train / "worker_result.json", runtime)
    save(train / "continuation_status.json", {"status": "evaluation_complete", "exit_code": 0,
        "pid": 300, "training_pid": 200, "watcher_pid": 400})
    for path in runtime["checkpoints"]:
        write(Path(path) / "adapter_model.safetensors", "weights")
        save(Path(path) / "adapter_config.json", {"r": 16})
        save(Path(path) / "trainer_state.json", {"global_step": int(path.rsplit("-", 1)[1])})
    for directory, pid in [(train, 200), *[(bench / f"batch_{size}", size + 100) for size in (1, 4, 8)],
                           (bench / "initialization_attempts/batch_8_missing_length", 188)]:
        save(directory / "worker_config.json", {"pid": pid, "cuda_visible_devices": "1", "gpu_uuid": gpu["uuid"], "initial_gpu": gpu})
        save(directory / "worker_started.json", {"pid": pid, "ppid": 99999, "initial_gpu": gpu,
            "utc": "2026-11-17T12:00:01+00:00"})
        save(directory / "resources.jsonl", {**gpu, "pid": pid, "optimizer_step": 0})
    save(out / "cpu_baseline_results.json", {"strategies": {name: {"combined": {"n_tasks": 1494}}
         for name in ("no_compress", "latest", "bm25", "oracle")}})
    save(out / "model_inference_preflight.json", {"status": "ok", "expected_trajectories_per_strategy": 1494,
                                                 "expected_outputs_per_strategy": 13956})
    model = {"pid": 300, "status": "complete", "cuda_visible_devices": "1", "gpu_before": gpu,
        "started_at_utc": "2026-11-17T12:00:00+00:00",
        "strategies": {name: {"status": "complete", "n_trajectories": 1494, "n_steps": 13956} for name in final.MODELS}}
    save(out / "model_rollout_run.json", model)
    save(rollout / "model_resources.jsonl", {"utc": "2026-11-17T12:00:01+00:00", "gpu": gpu})
    for name in final.MODELS:
        write(rollout / name / "steps.jsonl", "{}\n{}\n")
    sample_files = {}
    for key, name in (("train", "train_20k.jsonl"), ("eval_loss", "heldout_eval_loss_1k.jsonl"), ("sample_ids", "sample_ids.json")):
        sample_files[key] = final.file_record(write(train / name, "{}\n"))
    save(out / "sample_manifest.json", {"files": sample_files})
    return project, external, baseline


def inactive():
    return {"passed": True, "recorded_pid_count": 1, "active": [], "sources": [],
            "checks": [{"pid": 200, "classification": "zombie_or_dead", "active": False}]}


def machine():
    return {"utc": "2026-11-17T13:00:00+00:00", "gpu1": {"index": 1, "uuid": "GPU-correct", "memory_used_mib": 375},
            "compute_processes": [], "root_used_bytes": 1020000000, "df_raw": "measured df"}


def test_complete_fixture_includes_archived_zero_step_gpu_history(tmp_path):
    project, external, _ = fixture(tmp_path)
    assert final.completion_evidence(project, external)["passed"]
    history = final.verify_gpu_history(project, external)
    assert history["passed"] and len(history["worker_stages"]) == 5
    assert sum(item["archived_initialization"] for item in history["worker_stages"]) == 1


@pytest.mark.parametrize("mutation", ["train_incomplete", "model_incomplete", "wrong_step_count", "watcher_running", "stop_marker", "missing_adapter"])
def test_incomplete_work_never_reaches_finalization(tmp_path, mutation):
    project, external, _ = fixture(tmp_path)
    out, train = project / "eval/sft_run", external / "runs/sft_20k_20261117"
    if mutation == "train_incomplete":
        path = train / "worker_result.json"
        value = final.load(path); value["consumed_samples"] = 39999; save(path, value)
    elif mutation in ("model_incomplete", "wrong_step_count"):
        path = out / "model_rollout_run.json"
        value = final.load(path)
        if mutation == "model_incomplete":
            value["status"] = "running"
        else:
            value["strategies"]["base"]["n_steps"] -= 1
        save(path, value)
    elif mutation == "watcher_running":
        save(train / "continuation_status.json", {"status": "evaluating", "exit_code": 0})
    elif mutation == "missing_adapter":
        (train / "checkpoints/checkpoint-500/adapter_model.safetensors").unlink()
    else:
        save(out / "training_stop.json", {"error": "stop"})
    with pytest.raises(ValueError):
        final.completion_evidence(project, external)


def test_inventory_matches_final_checker_and_is_independent_of_project_outputs(tmp_path):
    project, external, _ = fixture(tmp_path)
    manifest = final.freeze_inventory(external)
    save(project / "eval/sft_run/external_artifact_manifest.json", manifest)
    check = verify_final_artifacts(project, external, slow=True)
    assert check["passed"] and check["coverage_complete"]
    second = final.freeze_inventory(external)
    assert second["files"] == manifest["files"] and second["total_bytes"] == manifest["total_bytes"]
    assert all("lines" in item for item in manifest["files"] if item["path"].endswith(".jsonl"))


@pytest.mark.parametrize("kind", ["symlink", "partial", "unterminated_jsonl"])
def test_inventory_refuses_unstable_or_unsafe_outputs(tmp_path, kind):
    _, external, _ = fixture(tmp_path)
    root = external / "eval_rollouts"
    if kind == "symlink":
        (root / "link").symlink_to(root / "model_resources.jsonl")
    elif kind == "partial":
        write(root / "partial.tmp", "unfinished")
    else:
        write(root / "bad.jsonl", "{}")
    with pytest.raises(ValueError):
        final.freeze_inventory(external)


def test_inventory_detects_a_file_added_during_hashing(tmp_path, monkeypatch):
    _, external, _ = fixture(tmp_path)
    original, injected = final.file_record, []
    def mutate(path):
        result = original(path)
        if not injected:
            write(external / "eval_rollouts/new.log", "late write")
            injected.append(True)
        return result
    monkeypatch.setattr(final, "file_record", mutate)
    with pytest.raises(ValueError, match="file set changed"):
        final.freeze_inventory(external)


def test_pid_collection_ignores_parent_and_tokenizer_vocabulary(tmp_path):
    project, external, _ = fixture(tmp_path)
    train = external / "runs/sft_20k_20261117"
    save(train / "pipeline_launch_initial.json", {"pid": 500, "initial_pipeline_pid": 499,
         "utc": "2026-11-17T12:00:00+00:00", "parent_pid": 99999, "session_id": 99998})
    save(train / "tokenizer.json", {"vocab": {"pid": 12345}})
    paths = final.pid_source_paths(project, external)
    identities = final.collect_pid_identities(paths)
    assert 499 in identities and 500 in identities
    assert 99999 not in identities and 99998 not in identities and 12345 not in identities
    assert any("initialization_attempts" in row["path"] for row in identities[188])


def test_proc_parser_handles_parentheses_and_zombies(tmp_path):
    proc = tmp_path / "proc"
    write(proc / "stat", "cpu 0\nbtime 1000\n")
    # Fields after comm: state is index0; starttime is index19 (Linux field22).
    write(proc / "88/stat", "88 (name with ) parentheses) Z " + " ".join(["0"] * 18 + ["1234", "0"]))
    (proc / "88/cmdline").write_bytes(b"python\0scripts/train_sft_worker.py\0train\0")
    result = final.process_snapshot(88, proc, clock_ticks=100)
    assert result["state"] == "Z" and result["start_unix"] == 1012.34
    classified = final.classify_process([{"observed_at_unix": 1013}], result)
    assert not classified["active"] and classified["classification"] == "zombie_or_dead"
    assert not final.process_snapshot(99, proc)["exists"]


def test_pid_reuse_requires_new_start_time_and_unrelated_cmdline():
    identity = [{"observed_at_unix": 1000, "path": "record"}]
    current = {"pid": 10, "exists": True, "state": "S", "start_unix": 2000, "cmdline": ["unrelated-program"]}
    result = final.classify_process(identity, current)
    assert not result["active"] and result["classification"] == "pid_reused_by_unrelated_process"
    current["cmdline"] = ["python", "/root/workplace/ctx-compressor/scripts/rollout_model.py", "run"]
    assert final.classify_process(identity, current)["active"]
    current.update(start_unix=999, cmdline=["unknown-wrapper"])
    assert final.classify_process(identity, current)["classification"] == "unresolved_live_historical_pid"


@pytest.mark.parametrize("mutation", ["visible_gpu", "archive_wrong_uuid", "model_wrong_gpu", "missing_monitor"])
def test_gpu_history_never_infers_missing_or_wrong_evidence(tmp_path, mutation):
    project, external, _ = fixture(tmp_path)
    if mutation == "visible_gpu":
        path = external / "runs/batch_bench_20261117/batch_1/worker_config.json"
        value = final.load(path); value["cuda_visible_devices"] = "0"; save(path, value)
    elif mutation == "archive_wrong_uuid":
        path = external / "runs/batch_bench_20261117/initialization_attempts/batch_8_missing_length/resources.jsonl"
        value = final.load(path); value["uuid"] = "wrong"; save(path, value)
    elif mutation == "model_wrong_gpu":
        path = external / "eval_rollouts/model_resources.jsonl"
        value = final.load(path); value["gpu"]["index"] = 0; save(path, value)
    else:
        (external / "runs/batch_bench_20261117/batch_4/resources.jsonl").write_text("")
    with pytest.raises(ValueError):
        final.verify_gpu_history(project, external)


@pytest.mark.parametrize("mutation", ["active_process", "gpu_not_free", "disk_over_limit", "wrong_uuid", "remaining_gpu_pid"])
def test_resource_gate_refuses_false_success(tmp_path, mutation):
    project, external, baseline = fixture(tmp_path)
    processes, snapshot = inactive(), machine()
    history = final.verify_gpu_history(project, external)
    if mutation == "active_process":
        processes.update(passed=False, active=[{"pid": 200}])
    elif mutation == "gpu_not_free":
        snapshot["gpu1"]["memory_used_mib"] = 2048
    elif mutation == "disk_over_limit":
        snapshot["root_used_bytes"] = 1300000001
    elif mutation == "wrong_uuid":
        snapshot["gpu1"]["uuid"] = "changed"
    else:
        snapshot["compute_processes"] = [{"pid": 200, "gpu_uuid": "GPU-correct"}]
    with pytest.raises(ValueError):
        final.resource_record(baseline, processes, history, snapshot)


def test_finalize_is_repeatable_and_failure_does_not_write_success_files(tmp_path, monkeypatch):
    project, external, _ = fixture(tmp_path)
    monkeypatch.setattr(final, "verify_processes", lambda *args: inactive())
    monkeypatch.setattr(final, "current_machine_snapshot", machine)
    first = final.finalize("all", project, external)
    manifest = final.load(project / "eval/sft_run/external_artifact_manifest.json")
    assert first["status"] == "complete" and first["root_delta_bytes"] == 20000000
    second = final.finalize("all", project, external)
    assert second["external_files"] == first["external_files"]
    assert final.load(project / "eval/sft_run/external_artifact_manifest.json")["files"] == manifest["files"]
    other_project, other_external, _ = fixture(tmp_path / "blocked")
    monkeypatch.setattr(final, "verify_processes", lambda *args: {**inactive(), "passed": False, "active": [{"pid": 200}]})
    with pytest.raises(ValueError, match="still active"):
        final.finalize("all", other_project, other_external)
    assert not (other_project / "eval/sft_run/final_resource_check.json").exists()
    assert not (other_project / "eval/sft_run/external_artifact_manifest.json").exists()


def test_gpu_process_with_reused_unrelated_pid_does_not_implicate_task05(tmp_path):
    project, external, baseline = fixture(tmp_path)
    processes, snapshot = inactive(), machine()
    processes["checks"][0]["classification"] = "pid_reused_by_unrelated_process"
    snapshot["compute_processes"] = [{"pid": 200, "gpu_uuid": "some-other-gpu"}]
    value = final.resource_record(baseline, processes, final.verify_gpu_history(project, external), snapshot)
    assert value["only_gpu1"] and value["task05_gpu_processes_remaining"] == []


@pytest.mark.parametrize("command", ["inventory", "resources"])
def test_separate_finalization_commands_only_publish_the_requested_artifact(tmp_path, monkeypatch, command):
    project, external, _ = fixture(tmp_path)
    monkeypatch.setattr(final, "verify_processes", lambda *args: inactive())
    monkeypatch.setattr(final, "current_machine_snapshot", machine)
    assert final.finalize(command, project, external)["status"] == "complete"
    out = project / "eval/sft_run"
    assert (out / "external_artifact_manifest.json").exists() is (command == "inventory")
    assert (out / "final_resource_check.json").exists() is (command == "resources")
