"""Exercise the detached controllers' GPU gates and monitoring on CPU."""
from contextlib import contextmanager
from functools import partial
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import Mock

from scripts import agent_v1_smoke as smoke
from scripts import run_sft_v3_pipeline as sft
from scripts import sft_v3_wait_gpu as gate
from scripts import test_v1_pipeline as locked


def check_write_paths(monkeypatch, work):
    path_open, path_mkdir, fd_open, replace = Path.open, Path.mkdir, os.open, os.replace

    def inside(path):
        path = Path(path).resolve()
        assert work in path.parents, f"controller write outside {work}: {path}"

    def open_path(path, mode="r", *args, **kwargs):
        if any(mark in mode for mark in "wax+"):
            inside(path)
        return path_open(path, mode, *args, **kwargs)

    def mkdir(path, *args, **kwargs):
        inside(path)
        return path_mkdir(path, *args, **kwargs)

    def open_fd(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            inside(path)
        return fd_open(path, flags, *args, **kwargs)

    def rename(source, target, *args, **kwargs):
        inside(source)
        inside(target)
        return replace(source, target, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(os, "open", open_fd)
    monkeypatch.setattr(os, "replace", rename)


def fast_gate():
    elapsed = 0

    def sleep(seconds):
        nonlocal elapsed
        elapsed += seconds

    return partial(gate.wait_for_gpu, clock=lambda: elapsed, sleep=sleep)


def gpu_sample():
    return {"index": 1, "uuid": smoke.old.GPU_UUID, "memory_used_mib": 375, "utilization_pct": 0}


def detached_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(os, "getsid", lambda _: os.getpid())


def test_task20_controller_gate_and_monitor_stay_in_agent_v1(monkeypatch, tmp_path):
    work = tmp_path / "agent_v1"
    work.mkdir()
    legacy = tmp_path / "dagger_v3"
    legacy.mkdir()
    old_log = legacy / "resource_monitor.jsonl"
    old_bytes = b'{"signed":true}\n'
    old_log.write_bytes(old_bytes)
    monkeypatch.setattr(sft, "RUN", legacy)
    for attribute, name in {
        "PID": "pipeline.pid", "STATUS": "status.json", "LOGS": "logs",
        "RUNTIME": "runtime", "LOCK": "gpu.lock", "TASK19_STATUS": "task19_proof.json",
        "SMOKE": "smoke", "SMOKE_CONFIG": "smoke_config.json", "SMOKE_REPORT": "smoke_results.json",
        "P_RECEIPT": "phase_p.json",
    }.items():
        monkeypatch.setattr(smoke, attribute, work / name)
    monkeypatch.setattr(smoke, "WORK", work)
    smoke.P_RECEIPT.write_text("{}")
    smoke.TASK19_STATUS.write_text('{"status":"evaluation_complete"}')
    monkeypatch.setattr(smoke, "phase_p", lambda **_: {})
    monkeypatch.setattr(smoke, "new_env", lambda: {})
    detached_cpu(monkeypatch)
    monkeypatch.setattr(smoke.signal, "signal", lambda *_: None)
    monkeypatch.setattr(smoke.old, "gpu", gpu_sample)
    monkeypatch.setattr(gate, "wait_for_gpu", fast_gate())

    @contextmanager
    def child(_argv, env, _out):
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        proc = Mock(pid=12345)
        proc.poll.side_effect = [None, 0]
        yield proc
        smoke.SMOKE_REPORT.write_text(json.dumps({
            "status": "smoke_complete", "awaiting_main_approval": True,
            "n_smoke_episodes": 3 * smoke.data.N_SMOKE * len(smoke.agent.ROWS),
        }))
        smoke.SMOKE_CONFIG.write_text('{"status":"frozen_after_smoke"}')

    monkeypatch.setattr(smoke.old, "child_process", child)
    with monkeypatch.context() as scoped:
        check_write_paths(scoped, work)
        smoke.detached()
    assert old_log.read_bytes() == old_bytes
    samples = [json.loads(row) for row in (work / "resource_monitor.jsonl").read_text().splitlines()]
    assert [s["consecutive_free"] for s in samples if s["pid"] is None] == [1, 2, 3]
    assert [s["alive"] for s in samples if s["pid"] == 12345] == [True, False]
    assert len((work / "pipeline_monitor.jsonl").read_text().splitlines()) == 2
    assert json.loads(smoke.STATUS.read_bytes())["stage"] == "waiting_main_approval"


def test_task19_controller_routes_gate_and_samples_to_its_stage(monkeypatch, tmp_path):
    work = tmp_path / "test_v1"
    work.mkdir()
    receipt = {"priority": "task19_before_task20"}
    for attribute, path in {
        "W": work, "OUT": work, "PID": work / "pipeline.pid", "STATUS": work / "status.json",
        "PHASE_P": work / "phase_p.json", "GPU_LOCK": work / "gpu.lock",
    }.items():
        monkeypatch.setattr(locked, attribute, path)
    monkeypatch.setattr(locked.assets, "PRE", work / "assets.json")
    locked.PHASE_P.write_text(json.dumps(receipt))
    locked.assets.PRE.write_text("{}")
    detached_cpu(monkeypatch)
    monkeypatch.setattr(locked, "phase_p", lambda: receipt)
    monkeypatch.setattr(locked, "environment", lambda _name, gpu: {"CUDA_VISIBLE_DEVICES": "1" if gpu else ""})
    monkeypatch.setattr(locked.r17.reader.original, "root_df", lambda: {})
    monkeypatch.setattr(locked.r17.reader, "gpu", gpu_sample)
    monkeypatch.setattr(locked.r17, "wait_for_gpu", partial(fast_gate(), reader=gpu_sample))

    real_stage = locked.r17.stage

    def first_stage(name, script, args, _outputs, **kwargs):
        if name == "G0":
            return real_stage(name, script, args, [], **kwargs)
        return {"stage": name}

    def child(_argv, *, env, **_):
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        proc = Mock(pid=12345)
        proc.wait.side_effect = [subprocess.TimeoutExpired("CPU fixture", locked.r17.POLL), 0]
        return proc

    monkeypatch.setattr(locked.r17, "stage", first_stage)
    monkeypatch.setattr(locked.r17.subprocess, "Popen", child)
    with monkeypatch.context() as scoped:
        check_write_paths(scoped, work)
        locked.worker()
    run = work / "stages/G0"
    assert json.loads((run / "receipt_gate.json").read_bytes())["gpu"] == gpu_sample()
    assert json.loads((run / "samples.jsonl").read_bytes())["child_pid"] == 12345
    events = [json.loads(row) for row in (work / "events.jsonl").read_text().splitlines()]
    assert [event["consecutive_free"] for event in events if event["status"] == "waiting_gpu"] == [1, 2, 3]
    assert json.loads(locked.STATUS.read_bytes())["status"] == "evaluation_complete"

