"""Exercise the detached coordinator's one-shot and failure receipts without a GPU."""
import os
from pathlib import Path
import sys

import pytest

from scripts import run_formal_grpo_pipeline as pipeline


def test_stage_failure_preserves_first_error_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOGS", tmp_path)
    monkeypatch.setattr(pipeline, "RUN", tmp_path)
    monkeypatch.setattr(pipeline, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(pipeline, "gpu", lambda: {"index": 1, "memory_used_mib": 122})
    controller = pipeline.Pipeline()
    args = [sys.executable, "-B", "-c",
            "import os; assert os.environ['CUDA_VISIBLE_DEVICES']==''; raise ValueError('first error')"]
    with pytest.raises(RuntimeError, match="first error"):
        controller.stage("D", args, False)
    dumped = pipeline.read(pipeline.STATUS)
    assert dumped["last_monitor"]["alive"] is False
    assert dumped["stages"][0]["exit_code"] != 0
    assert (tmp_path / "D.log").exists()


def test_busy_card_prevents_stage_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "gpu", lambda: {"memory_used_mib": 2048})
    monkeypatch.setattr(pipeline, "STATUS", tmp_path / "status.json")
    p = pipeline.Pipeline()
    with pytest.raises(ValueError, match="busy"):
        p.stage("F0", ["unused"], True)
    assert p.record["stage_pid"] is None


def test_every_saved_checkpoint_is_evaluated(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "TRAIN", tmp_path)
    paths = [tmp_path / f"checkpoint-{step}" for step in (125, 200)]
    for path in paths:
        path.mkdir()
        (path / "adapter_model.safetensors").write_bytes(b"test")
    receipt = {"status": "ok", "checkpoints": list(map(str, paths)), "checkpoint_steps": [125, 200]}
    assert pipeline.checkpoint_paths(receipt) == paths
    receipt.update(checkpoints=receipt["checkpoints"][:1], checkpoint_steps=[125])
    with pytest.raises(ValueError, match="absent"):
        pipeline.checkpoint_paths(receipt)


@pytest.fixture
def failed_c_run(tmp_path, monkeypatch):
    root = tmp_path / "formal"
    logs = root / "logs"
    logs.mkdir(parents=True)
    states = root / "states" / "states.jsonl"
    states.parent.mkdir()
    states.write_bytes(b"closed-loop state\\n")
    raw = tmp_path / "eval_rollouts_grpo_v2" / pipeline.LABEL / "steps.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"frozen step\\n")
    pipeline.put_new(states.parent / "state_checks.json", {"state_file": pipeline.file_record(states)})
    status = root / "pipeline_status.json"
    for key, path in {"ROOT": tmp_path, "RUN": root, "LOGS": logs, "STATUS": status,
                      "GROUPS": root / "filter/groups.jsonl", "DATA": root / "data/train.jsonl",
                      "STATES": states, "TRAIN": root / "train"}.items():
        monkeypatch.setattr(pipeline, key, path)
    pid = 99999999
    stages = [{"stage": s, "exit_code": code, "status": state, "pid": pid}
              for s, code, state in [("F0", 0, "complete"), ("B", 0, "complete"), ("C", 1, "failed")]]
    previous = {"status": "failed", "stage": "C", "stage_pid": None, "ended_utc": "before",
                "pipeline_pid": pid, "first_error": "Unexpected vLLM top_k default", "started_utc": "earlier",
                "stages": stages}
    pipeline.put_new(status, previous)
    (root / "pipeline.pid").write_text(str(pid))
    (root / "pipeline_monitor.jsonl").write_text("old monitor")
    (logs / "C.log").write_text("old error")
    observed = []
    def check_eval(name):
        observed.append(name)
        return {"steps_artifact": pipeline.file_record(raw), "combined": {"n_tasks": 1494, "n_steps": 13956}}
    def check_states():
        observed.append("B")
        return 22158
    monkeypatch.setattr(pipeline, "check_eval", check_eval)
    monkeypatch.setattr(pipeline, "check_states", check_states)
    return observed


def test_resume_c_validates_receipts_and_preserves_failed_files(failed_c_run, monkeypatch):
    files = pipeline.archive_sources()
    before = {p: p.read_bytes() for p in files}
    receipt, prior = pipeline.prepare_resume_c()
    assert failed_c_run == [pipeline.LABEL, "B"]
    assert receipt["n_states"] == 22158 and [s["stage"] for s in prior] == ["F0", "B"]
    assert all(not p.exists() and pipeline.archived(p).read_bytes() == data for p, data in before.items())
    monkeypatch.setattr(pipeline, "check_a", lambda: None)
    controller = pipeline.Pipeline()
    started = []
    def stop_after_dispatch(stage, args, on_gpu):
        started.append(stage)
        raise InterruptedError("test dispatch")
    monkeypatch.setattr(controller, "stage", stop_after_dispatch)
    with pytest.raises(InterruptedError, match="test dispatch"):
        controller.run(receipt)
    assert started == ["C"]


def test_resume_c_rejects_damaged_states_before_archival(failed_c_run):
    with pipeline.STATES.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="B states file hash"):
        pipeline.prepare_resume_c()
    assert pipeline.STATUS.is_file() and (pipeline.LOGS / "C.log").is_file()
    assert not pipeline.archived(pipeline.STATUS).exists()


def test_resume_c_does_not_replace_existing_filter_or_archive(failed_c_run):
    same_name = pipeline.archived(pipeline.LOGS / "C.log")
    same_name.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="ambiguous"):
        pipeline.prepare_resume_c()
    assert same_name.read_bytes() == b"preserve"
