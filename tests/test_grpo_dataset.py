"""Task 07 (Claude): GRPO dataset builder is balanced, seeded, train-only, exclusion-aware, no-overwrite."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/make_grpo_dataset.py"
MODES, TIERS = ("reselect", "append"), ("tight", "mid", "loose")


def _states(path, split="train", per_cell=5):
    with path.open("w") as f:
        for m in MODES:
            for t in TIERS:
                for k in range(per_cell):
                    f.write(json.dumps({"split": split, "mode": m, "budget_tier": t, "step": k,
                                        "trajectory_id": f"x|{m}|{t}|{k}",
                                        "messages": [{"role": "user", "content": f"{m}-{t}-{k}"}],
                                        "reward_state_json": "{}"}) + "\n")


def _run(*args):
    return subprocess.run([sys.executable, "-B", str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_balanced_and_seeded(tmp_path):
    states = tmp_path / "s.jsonl"
    _states(states)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert _run("--states", states, "--per-cell", 3, "--out", a).returncode == 0
    assert _run("--states", states, "--per-cell", 3, "--out", b).returncode == 0
    rows = [json.loads(l) for l in a.open()]
    assert a.read_text() == b.read_text()
    assert len(rows) == 18
    cells = {}
    for r in rows:
        cells[r["cell"]] = cells.get(r["cell"], 0) + 1
    assert set(cells.values()) == {3} and len(cells) == 6
    assert all(set(r) == {"messages", "reward_state_json", "trajectory_id", "state_step", "cell"} for r in rows)


def test_exclusion_and_overwrite(tmp_path):
    states = tmp_path / "s.jsonl"
    _states(states, per_cell=2)
    probe = tmp_path / "probe.jsonl"
    probe.write_text(json.dumps({"trajectory_id": "x|append|mid|0", "step": 0}) + "\n")
    out = tmp_path / "o.jsonl"
    res = _run("--states", states, "--per-cell", 2, "--exclude", probe, "--out", out)
    assert res.returncode != 0  # the append|mid cell now has only one state left
    res = _run("--states", states, "--per-cell", 1, "--exclude", probe, "--out", out)
    assert res.returncode == 0
    assert all(r["trajectory_id"] != "x|append|mid|0" for r in map(json.loads, out.open()))
    assert _run("--states", states, "--per-cell", 1, "--out", out).returncode != 0


def test_train_only(tmp_path):
    states = tmp_path / "s.jsonl"
    _states(states, split="heldout")
    assert _run("--states", states, "--per-cell", 1, "--out", tmp_path / "o.jsonl").returncode != 0
