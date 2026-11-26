"""Guard both pipeline entry points against a missing or failed two-step smoke."""
import json
import sys
from unittest.mock import Mock

import pytest
from scripts import grpo_r2_start as start
from scripts import run_grpo_r2_pipeline as driver


def test_detach_refuses_before_creating_launcher_or_child(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["run_grpo_r2_pipeline.py", "--detach"])
    monkeypatch.setattr(sys, "prefix", str(driver.ROOT / "envs/grpo"))
    monkeypatch.setattr(driver, "PROJECT", driver.Path("/root/workplace/ctx-compressor"))
    monkeypatch.setattr(driver, "RUN", tmp_path)
    monkeypatch.setattr(driver, "START", tmp_path/"start_model.json")
    monkeypatch.setattr(driver, "OUT", tmp_path)
    driver.START.write_text("{}")
    (tmp_path/"phase_a_tests.json").write_text("{}")
    monkeypatch.setattr(driver, "require_new_run", lambda: None)
    for key, value in {"CUDA_VISIBLE_DEVICES":"1", "CUDA_DEVICE_ORDER":"PCI_BUS_ID", "HF_HUB_OFFLINE":"1"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(driver, "phase_a_evidence", Mock(side_effect=ValueError("smoke failed")))
    child = Mock()
    monkeypatch.setattr(driver.subprocess, "Popen", child)
    with pytest.raises(ValueError, match="smoke failed"):
        driver.main()
    child.assert_not_called()
    assert not (tmp_path/"launcher.log").exists()


def test_independent_child_rechecks_before_running_f0(monkeypatch):
    monkeypatch.setattr(driver, "phase_a_evidence", Mock(side_effect=ValueError("smoke changed")))
    process = driver.Pipeline()
    evaluate = Mock()
    monkeypatch.setattr(process, "evaluate", evaluate)
    with pytest.raises(ValueError, match="smoke changed"):
        process.run()
    evaluate.assert_not_called()


def test_phase_a_includes_verified_smoke_receipt(monkeypatch, tmp_path):
    from scripts import smoke_grpo_r2 as smoke
    root = tmp_path/"grpo_r2"
    merged = root/"merged_start"
    merged.mkdir(parents=True)
    out = tmp_path/"eval"
    out.mkdir()
    monkeypatch.setattr(start, "RUN", root)
    monkeypatch.setattr(start, "MERGED", merged)
    monkeypatch.setattr(start, "OUT", out)
    monkeypatch.setattr(start, "START", out/"start_model.json")
    origin = {"path":"/sft-v3", "adapter_sha256":"fixture-identity"}
    monkeypatch.setattr(start, "resolve_start", lambda: origin)
    monkeypatch.setattr(start, "compare_fixture", lambda *args: {"n_reference":30})
    start.START.write_text(json.dumps(origin))
    weight = merged/"weight.safetensors"
    weight.write_bytes(b"w")
    (merged/"merge_record.json").write_text(json.dumps({"base":str(start.BASE),"adapter":origin["path"],
        "adapter_sha256":origin["adapter_sha256"],"weights":{weight.name:start.digest(weight)}}))
    for tag, base, adapter in [("ref_adapter", start.BASE, origin["path"]), ("ref_merged", merged, None)]:
        (root/tag).mkdir()
        (root/tag/"greedy_reference.json").write_text(json.dumps({"base_model":str(base),"inference_adapter":adapter}))
    (out/"phase_a_tests.json").write_text(json.dumps({"status":"passed","tests":691,"passed":691,"failures":0,
        "errors":0,"skipped":0}))
    (out/"phase_p_tests.json").write_text('{"tests": 691}')
    dest = root/"smoke_train/smoke_checks.json"
    dest.parent.mkdir()
    with pytest.raises(FileNotFoundError):
        monkeypatch.setattr(smoke, "verify_smoke", lambda: json.loads(dest.read_bytes()))
        start.phase_a_evidence()
    good = {"status":"ok","optimizer_steps":2,"n_prompts":8,"reward_recheck":{"mismatches":0,"advantage_mismatches":0}}
    dest.write_text(json.dumps(good))
    assert start.phase_a_evidence()["smoke_training"] == start.file_record(dest)
    good["reward_recheck"]["advantage_mismatches"] = 1
    dest.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="smoke verification"):
        start.phase_a_evidence()
