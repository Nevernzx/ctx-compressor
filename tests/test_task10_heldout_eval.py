"""Task 10 E diagnostics use effective valid actions and the exact Claude step population."""
import json
from pathlib import Path

import pytest

from scripts import eval_sft_v3 as e


def test_multigold_drop_precision_and_claude_population():
    before = {11, 12, 13}
    r = {"mode": "reselect", "step": 3, "gold": 15, "kept_before": [11, 12, 19],
         "keep_after": [12, 13, 15, 18], "action_valid": True}
    c = e.extra_counts(r, before)
    assert (c["multiadd_contains_current_gold"], c["multiadd_contains_current_gold_n"]) == (1, 1)
    assert (c["multiadd_fragment_gold"], c["multiadd_fragment_gold_n"]) == (2, 3)
    assert (c["drop_precision"], c["drop_precision_n"]) == (1, 2)
    assert (c["claude_nongold_kept_drop_old_gold"], c["claude_nongold_kept_drop_old_gold_n"]) == (1, 1)
    assert c["claude_allgold_kept_drop_old_gold_n"] == 0
    # The current PID contributes to gold at action time, but not to previous gold for Claude's split.
    r.update(gold=19, keep_after=[12])
    c = e.extra_counts(r, before)
    assert (c["drop_precision"], c["drop_precision_n"]) == (0, 2)
    assert c["claude_nongold_kept_drop_old_gold"] == 1


def test_claude_split_excludes_invalid_append_and_step_zero():
    r = {"mode": "reselect", "step": 0, "gold": 8, "kept_before": [], "keep_after": [6, 7], "action_valid": True}
    assert e.extra_counts(r, set())["claude_allgold_kept_drop_old_gold_n"] == 0
    r.update(step=1, mode="append")
    assert e.extra_counts(r, {3})["claude_allgold_kept_drop_old_gold_n"] == 0
    r.update(mode="reselect")
    c = e.extra_counts(r, {3})
    assert (c["claude_allgold_kept_drop_old_gold"], c["claude_allgold_kept_drop_old_gold_n"]) == (0, 1)
    assert (c["multiadd_contains_current_gold"], c["multiadd_contains_current_gold_n"]) == (0, 1)
    r.update(action_valid=False, keep_after=[])
    assert not e.extra_counts(r, {3})


def test_selection_checks_complete_grid_and_adapter_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    raw, small = tmp_path / "raw", tmp_path / "small"
    monkeypatch.setattr(e.cfg, "OUT", small)
    monkeypatch.setattr(e, "RAW", raw)
    monkeypatch.setattr(e, "validate_strategy", lambda v, name: v)
    adapters = {}
    data = zip(e.cfg.EVAL, (8, 3, 3, 3), (0.99, 0.80, 0.88, 0.88), strict=True)
    for step, loss, recall in data:
        name = f"checkpoint-{step}"
        directory = small / name
        directory.mkdir(parents=True)
        steps = raw / name / "steps.jsonl"
        steps.parent.mkdir(parents=True)
        steps.write_text("record\n")
        adapter = tmp_path / "train" / name
        adapter.mkdir(parents=True)
        weight = adapter / "adapter_model.safetensors"
        weight.write_text(name)
        adapters[step] = adapter, e.v1.file_record(weight)
        metrics = {"strategy": name, "checkpoint_step": step, "combined": {"n_tasks": 1494, "n_steps": 13956,
                   "evidence_loss": {"count": loss, "n": 1494}, "final_recall": {"mean": recall, "n": 1494}},
                   "settings": {"n_tasks": 249}, "configs": {str(i): {"metrics": {"n_tasks": 249}} for i in range(6)},
                   "steps_artifact": e.v1.file_record(steps)}
        outputs = {f"{name}_results.json": metrics, "diagnostics.json": {}, "extra.json": {}}
        for filename, value in outputs.items():
            (directory / filename).write_text(json.dumps(value))
        run = {"status": "complete", "checkpoint_step": step, "adapter": str(adapter),
               "adapter_weight": adapters[step][1], "engine_shutdown": {"completed": True}, "step0": {"n": 1494},
               "artifacts": {f: e.v1.file_record(directory / f) for f in outputs}}
        (directory / "run.json").write_text(json.dumps(run))
    monkeypatch.setattr(e, "check_adapter", adapters.__getitem__)
    e.select()
    result = json.loads((small / "selected_checkpoint.json").read_text())
    assert result["checkpoint_step"] == 3000
    assert Path(result["path"]) == adapters[3000][0]
    assert result["adapter_sha256"] == adapters[3000][1]["sha256"]
    with pytest.raises(ValueError, match="existing checkpoint selection"):
        e.select()

