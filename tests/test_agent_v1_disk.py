"""CPU contracts for Task 20 root sampling and isolated smoke retries."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts import agent_v1_disk as disk, agent_v1_engines as engines, agent_v1_smoke as smoke


def test_root_usage_baseline_and_engine_sample_agree(monkeypatch, tmp_path):
    snapshot = shutil.disk_usage("/")

    def same_mount(path):
        assert path == "/"
        return snapshot

    monkeypatch.setattr(disk.shutil, "disk_usage", same_mount)
    monkeypatch.setattr(engines.pipeline, "gpu", lambda: {"memory_used_mib": 375})
    runtime = engines.RuntimeMeasurements(tmp_path / "gpu.jsonl")
    baseline = disk.root_usage().used
    runtime.gpu("agent_loaded")
    observed = json.loads(runtime.path.read_text())
    assert abs(baseline - observed["root_used_bytes"]) < 1_000_000
    assert observed["root_used_bytes"] == snapshot.used == runtime.root_usage[0]
    assert observed["root_free_bytes"] == snapshot.free


def test_root_trace_captures_engine_boundaries_and_transient_peak(monkeypatch, tmp_path):
    probe = disk.RootTelemetry(tmp_path)
    root = iter([120_000_000_000, 120_402_345_984, 120_000_004_096])
    monkeypatch.setattr(disk, "root_usage", lambda: SimpleNamespace(used=next(root), free=10_000_000_000))
    monkeypatch.setattr(disk, "allocated_bytes", lambda paths, timeout: {path: 4096 for path in paths})
    monkeypatch.setattr(disk, "writable_root_files", lambda *_: [])
    probe.workplace = {"directory": "/root/workplace", "bytes": 8192,
                       "started_utc": "2026-11-25T00:00:00Z", "ended_utc": "2026-11-25T00:00:06Z"}
    probe.active_engine = "agent"
    with probe.engine_start("agent"):
        probe.sample("periodic", probe.active_engine)
    series = list(map(json.loads, probe.trace.read_text().splitlines()))
    assert [row["boundary"] for row in series] == ["before_engine", "periodic", "after_engine"]
    assert all(row["engine"] == "agent" and row["workplace"]["bytes"] == 8192 for row in series)
    assert all(set(row["directories"]) == set(disk.FAST_DIRS) for row in series)
    probe.policy.finish(series[-1])
    proof = probe.policy.report()
    assert proof["peak_growth_bytes"] == 402_345_984
    assert proof["status"] == "passed" and proof["sustained_growth_bytes"] == 4096
    assert len(proof["transient_unattributed_peaks"]) == 1
    peak = proof["transient_unattributed_peaks"][0]
    assert peak["engine"] == "agent" and peak["global_growth_bytes"] == 402_345_984
    assert peak["utc"] == series[1]["utc"] and peak["returned_utc"] == series[-1]["utc"]
    probe.policy.check()


def measurement(*, growth=0, tmp=0, cache=0, workplace=0, fds=(), free=10_000_000_000):
    return {"root_used_bytes": 120_000_000_000 + growth, "root_free_bytes": free,
            "utc": "2026-11-25T12:00:00Z", "engine": "r4", "boundary": "periodic",
            "directories": {"/tmp": 5_000_000_000 + tmp, "/root/.cache": 4_000_000_000 + cache},
            "workplace": {"bytes": 30_000_000_000 + workplace},
            "root_writable_fds": list(fds)}


def test_sustained_root_growth_is_checked_when_finishing():
    policy = disk.RootPolicy()
    policy.observe(measurement())
    closing = measurement(growth=disk.ROOT_LIMIT_BYTES + 1)
    policy.observe(closing)
    policy.check()
    policy.finish(closing)
    assert policy.report()["violation"]["rule"] == "a"
    with pytest.raises(disk.RootLimitError, match="rule a"):
        policy.check()


@pytest.mark.parametrize("key", ["tmp", "cache", "workplace"])
def test_attributed_growth_in_any_directory_is_a_stop_sample(key):
    policy = disk.RootPolicy()
    policy.observe(measurement())
    policy.observe(measurement(**{key: disk.ROOT_LIMIT_BYTES + 1}))
    with pytest.raises(disk.RootLimitError, match="rule b"):
        policy.check()


def test_attributed_bytes_deduplicate_inodes_and_overlapping_paths():
    policy = disk.RootPolicy()
    policy.observe(measurement(fds=[{"inode": 1, "path": "/tmp/output", "allocated_bytes": 120_000_000}]))
    linked = {"inode": 1, "path": "/tmp/output", "allocated_bytes": 320_000_000}
    second = {"inode": 2, "path": "/root/anonymous (deleted)", "allocated_bytes": 110_000_000}
    current = measurement(tmp=200_000_000, fds=[linked, dict(linked), second])
    policy.observe(current)
    assert current["attributed_growth_bytes"] == 310_000_000
    with pytest.raises(disk.RootLimitError, match="rule b"):
        policy.check()


def test_available_root_bytes_threshold_is_strict():
    policy = disk.RootPolicy()
    policy.observe(measurement(free=disk.MIN_ROOT_FREE_BYTES))
    policy.check()
    policy.observe(measurement(free=disk.MIN_ROOT_FREE_BYTES - 1))
    with pytest.raises(disk.RootLimitError, match="rule c"):
        policy.check()


def test_retry_leaves_original_output_directories_in_place(monkeypatch, tmp_path):
    work = tmp_path / "agent_v1"
    small = tmp_path / "eval/agent_v1"
    previous = [work / "smoke/episodes.jsonl", work / "try2/smoke/episodes.jsonl"]
    for path in previous:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"prior recorded attempt\n")
    monkeypatch.setattr(smoke.data, "WORK", work)
    monkeypatch.setattr(smoke.data, "SMALL", small)
    monkeypatch.setattr(smoke, "EXTERNAL", tmp_path)
    for key in ("ATTEMPT", "WORK", "PID", "STATUS", "LOGS", "RUNTIME", "SMOKE", "PROMPTS", "P_RECEIPT",
                "MONITOR_TESTS", "SMOKE_REPORT", "SMOKE_CONFIG", "LOG"):
        monkeypatch.setattr(smoke, key, getattr(smoke, key))
    smoke.select_attempt("try3")
    assert smoke.STATUS == work / "try3/status.json"
    assert smoke.SMOKE == work / "try3/smoke"
    assert smoke.SMOKE_CONFIG == small / "try3/smoke_run_config.json"
    assert smoke.PROMPTS == small / "try3/prompts.json"
    assert smoke.RUNTIME == tmp_path / "rt/t20/try3"
    assert smoke.invocation(Path("python"), "run-smoke")[-2:] == ["--attempt", "try3"]
    assert all(path.read_bytes() == b"prior recorded attempt\n" for path in previous)
