"""CPU fixtures for transient root-disk events and persistent monitor errors."""
import json
from types import SimpleNamespace

import pytest

from scripts import rollout_model as runner


BASELINE = {"root_df_used_bytes": 1_000_000, "root_final_growth_limit_bytes": 300_000_000}
LIMIT = BASELINE["root_final_growth_limit_bytes"]


def disk(delta):
    used = BASELINE["root_df_used_bytes"] + delta
    return {"used_bytes": used, "available_bytes": 20_000_000_000,
            "df_raw": f"Used Avail\n{used} 20000000000\n"}


class BoundedStop:
    """Run exactly n monitor iterations without sleeping or starting threads."""
    def __init__(self, iterations):
        self.iterations = iterations
        self.waits = 0

    def is_set(self):
        return self.waits >= self.iterations

    def wait(self, seconds):
        assert seconds == 20
        self.waits += 1


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(runner, "gpu_info", lambda: {"index": 1, "memory_used_mib": 123})


def test_raw_df_is_retained_without_altering_measurement(monkeypatch):
    raw = "        Used        Avail\n931440451584 23000000000\n"
    def query(command, **kwargs):
        assert command == ["df", "-B1", "--output=used,avail", "/"]
        assert kwargs == {"text": True}
        return raw
    monkeypatch.setattr(runner.subprocess, "check_output", query)
    assert runner.disk_info() == {"used_bytes": 931440451584, "available_bytes": 23000000000,
                                  "df_raw": raw}


def test_exceed_recover_and_later_samples_keep_monitoring(tmp_path, monkeypatch, fake_gpu):
    sequence = iter([disk(10), disk(LIMIT + 1), disk(LIMIT + 2), disk(LIMIT), disk(-20)])
    monkeypatch.setattr(runner, "disk_info", lambda: next(sequence))
    path = tmp_path / "resources.jsonl"
    monitor = runner.Monitor(path, BASELINE)
    monitor.stop = BoundedStop(4)
    monitor.run()
    monitor.check()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 4 and monitor.stop.waits == 4
    assert [r["root_delta_bytes"] for r in rows] == [LIMIT + 1, LIMIT + 2, LIMIT, -20]
    assert [r["root_limit_event"] for r in rows] == ["enter", None, "recover", None]
    assert [r["root_over_limit"] for r in rows] == [True, True, False, False]
    assert all(r["root_df"]["df_raw"] for r in rows)
    summary = monitor.diagnostics()
    assert summary["baseline_used_bytes"] == BASELINE["root_df_used_bytes"]
    assert summary["peak_delta_bytes"] == LIMIT + 2
    assert summary["last_delta_bytes"] == -20 and not summary["last_over_limit"]
    assert summary["over_limit_sample_count"] == 2
    assert summary["sample_count_including_startup_and_shutdown"] == 5
    assert [r["root_limit_event"] for r in summary["threshold_events"]] == ["enter", "recover"]
    assert summary["monitor_error"] is None


def test_startup_over_limit_and_shutdown_transition_are_diagnostics(tmp_path, monkeypatch, fake_gpu):
    sequence = iter([disk(LIMIT + 7), disk(42)])
    monkeypatch.setattr(runner, "disk_info", lambda: next(sequence))
    monitor = runner.Monitor(tmp_path / "resources.jsonl", BASELINE)
    assert monitor.startup_root["root_over_limit"]
    assert monitor.startup_root["root_limit_event"] == "enter"
    assert monitor.startup_root["root_df"] == disk(LIMIT + 7)
    monitor.stop = BoundedStop(1)
    monitor.run()
    monitor.check()
    final = monitor.record_disk(disk(LIMIT + 9), phase="shutdown")
    assert final["root_over_limit"] and final["root_limit_event"] == "enter"
    summary = monitor.diagnostics()
    assert [(r["phase"], r["root_limit_event"]) for r in summary["threshold_events"]] == [
        ("startup", "enter"), ("monitor", "recover"), ("shutdown", "enter")]
    assert summary["peak_delta_bytes"] == LIMIT + 9
    assert summary["sample_count_including_startup_and_shutdown"] == 3
    assert summary["monitor_error"] is None


def test_disk_read_failure_remains_a_monitor_error(tmp_path, monkeypatch, fake_gpu):
    monkeypatch.setattr(runner, "disk_info", lambda: disk(0))
    monitor = runner.Monitor(tmp_path / "resources.jsonl", BASELINE)
    error = OSError("df could not read the filesystem")
    def broken_read():
        raise error
    monkeypatch.setattr(runner, "disk_info", broken_read)
    monitor.stop = BoundedStop(1)
    monitor.run()
    assert monitor.error is error and monitor.stop.waits == 0
    with pytest.raises(RuntimeError, match="resource monitor failed") as failure:
        monitor.check()
    assert failure.value.__cause__ is error
    assert monitor.diagnostics()["monitor_error"] == repr(error)


class JoinThread:
    def __init__(self, monitor, *, timeout=False, late_error=None):
        self.monitor, self.timeout, self.late_error = monitor, timeout, late_error
        self.alive = True
        self.joins = []

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        self.joins.append(timeout)
        self.alive = self.timeout
        self.monitor.error = self.late_error


@pytest.fixture
def cleanup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "disk_info", lambda: disk(5))
    monitor = runner.Monitor(tmp_path / "resources.jsonl", BASELINE)
    monitor.thread = JoinThread(monitor)
    calls = []
    core = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=core))
    monkeypatch.setattr(runner.gc, "collect", lambda: calls.append("gc"))
    path = tmp_path / "run.json"
    manifest = {"status": "complete"}
    return SimpleNamespace(monitor=monitor, calls=calls, core=core, llm=llm,
                           path=path, manifest=manifest, started=runner.time.monotonic())


def finish(case):
    runner.finish_run(case.manifest, case.monitor, case.llm, case.path, case.started)


def test_success_published_only_after_thread_exit_and_engine_shutdown(cleanup_case, monkeypatch):
    case = cleanup_case
    monkeypatch.setattr(runner, "disk_info", lambda: disk(LIMIT + 3))
    real_write = runner.write_json
    def after_cleanup(path, value):
        assert case.calls == ["shutdown", "gc"]
        assert not case.monitor.thread.is_alive()
        real_write(path, value)
    monkeypatch.setattr(runner, "write_json", after_cleanup)
    finish(case)
    saved = json.loads(case.path.read_text())
    assert saved["status"] == "complete"
    assert saved["monitor_thread_stopped"] and saved["engine_shutdown"]["completed"]
    assert saved["root_disk_diagnostics_complete"]
    assert saved["root_shutdown_diagnostics"]["root_delta_bytes"] == LIMIT + 3
    assert saved["root_shutdown_diagnostics"]["root_over_limit"]
    assert case.monitor.stop.is_set() and case.monitor.thread.joins == [10]


def test_join_timeout_fails_without_sampling_live_monitor_and_releases_engine(cleanup_case, monkeypatch):
    case = cleanup_case
    case.monitor.thread = JoinThread(case.monitor, timeout=True)
    def unexpected_read():
        pytest.fail("must not sample a monitor that failed to exit")
    monkeypatch.setattr(runner, "disk_info", unexpected_read)
    with pytest.raises(RuntimeError, match="did not stop within 10 seconds"):
        finish(case)
    saved = json.loads(case.path.read_text())
    assert saved["status"] == "failed" and not saved["monitor_thread_stopped"]
    assert not saved["root_disk_diagnostics_complete"]
    assert saved["cleanup_errors"][0]["stage"] == "monitor_stop"
    assert case.calls == ["shutdown", "gc"]


def test_error_arriving_during_join_cannot_publish_complete(cleanup_case):
    case = cleanup_case
    error = OSError("late monitor flush failure")
    case.monitor.thread = JoinThread(case.monitor, late_error=error)
    with pytest.raises(RuntimeError, match="resource monitor failed") as failure:
        finish(case)
    assert failure.value.__cause__ is error
    saved = json.loads(case.path.read_text())
    assert saved["status"] == "failed" and saved["monitor_thread_stopped"]
    assert saved["cleanup_errors"][0]["stage"] == "monitor_error_check"
    assert case.calls == ["shutdown", "gc"]


@pytest.mark.parametrize("operation", ["read", "record", "diagnostics"])
def test_final_disk_failures_always_release_engine_and_persist_failure(cleanup_case, monkeypatch, operation):
    case = cleanup_case
    error = OSError(f"final disk {operation} failed")
    def fail(*args, **kwargs):
        raise error
    if operation == "read":
        monkeypatch.setattr(runner, "disk_info", fail)
    else:
        monkeypatch.setattr(case.monitor, "record_disk" if operation == "record" else "diagnostics", fail)
    with pytest.raises(OSError) as failure:
        finish(case)
    assert failure.value is error
    saved = json.loads(case.path.read_text())
    assert saved["status"] == "failed" and not saved["root_disk_diagnostics_complete"]
    assert saved["cleanup_errors"][0]["stage"] == f"final_disk_{operation}"
    assert case.calls == ["shutdown", "gc"]
    assert saved["engine_shutdown"]["completed"]


@pytest.mark.parametrize("permanent", [False, True])
def test_final_manifest_write_failure_occurs_after_cleanup_and_never_returns_success(cleanup_case, monkeypatch, permanent):
    case = cleanup_case
    real_write = runner.write_json
    real_write(case.path, {"status": "running"})
    error = OSError("manifest write failed")
    attempts = []
    def broken_write(path, value):
        assert case.calls == ["shutdown", "gc"]
        attempts.append(value["status"])
        if permanent or len(attempts) == 1:
            raise error
        real_write(path, value)
    monkeypatch.setattr(runner, "write_json", broken_write)
    with pytest.raises(OSError) as failure:
        finish(case)
    assert failure.value is error
    assert case.manifest["status"] == "failed" and attempts == ["complete", "failed"]
    assert json.loads(case.path.read_text())["status"] == ("running" if permanent else "failed")
    assert case.manifest["cleanup_errors"][0]["stage"] == "terminal_manifest_write"


def test_engine_shutdown_error_is_failed_and_gc_still_runs(cleanup_case):
    case = cleanup_case
    error = OSError("engine shutdown failed")
    def broken_shutdown():
        case.calls.append("shutdown")
        raise error
    case.core.shutdown = broken_shutdown
    with pytest.raises(OSError) as failure:
        finish(case)
    assert failure.value is error and case.calls == ["shutdown", "gc"]
    saved = json.loads(case.path.read_text())
    assert saved["status"] == "failed" and not saved["engine_shutdown"]["completed"]
    assert saved["cleanup_errors"][0]["stage"] == "engine_shutdown"


def test_cleanup_failure_preserves_an_active_inference_exception(cleanup_case):
    case = cleanup_case
    case.monitor.thread = JoinThread(case.monitor, timeout=True)
    original = ValueError("original inference failure")
    case.manifest.update(status="failed", error=repr(original))
    with pytest.raises(ValueError, match="original inference failure") as failure:
        try:
            raise original
        finally:
            finish(case)
    assert failure.value is original
    saved = json.loads(case.path.read_text())
    assert saved["status"] == "failed" and saved["error"] == repr(original)
    assert saved["cleanup_errors"][0]["stage"] == "monitor_stop"
    assert case.calls == ["shutdown", "gc"]


@pytest.mark.parametrize("operation", ["open", "write", "flush"])
def test_log_io_failures_remain_monitor_errors(operation, monkeypatch, fake_gpu):
    monkeypatch.setattr(runner, "disk_info", lambda: disk(LIMIT + 1))
    error = OSError(f"log {operation} failed")
    class BrokenLog:
        def open(self, mode):
            assert mode == "x"
            if operation == "open":
                raise error
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, value):
            if operation == "write":
                raise error

        def flush(self):
            if operation == "flush":
                raise error

    monitor = runner.Monitor(BrokenLog(), BASELINE)
    monitor.stop = BoundedStop(1)
    monitor.run()
    assert monitor.error is error
    with pytest.raises(RuntimeError, match="resource monitor failed") as failure:
        monitor.check()
    assert failure.value.__cause__ is error
