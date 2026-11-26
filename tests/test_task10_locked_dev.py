"""Regression checks for the CPU-only Task 10 lock and historical retrieval comparison."""
import json
import random

import pytest

from scripts import build_task10_locked_dev as lock


def test_locked_group_order_exclusion_and_remainder():
    valid = [{"id": str(i)} for i in range(12)]
    groups, tail = lock.grouped_ids(valid, {"1", "8"})
    expected = ["0", "2", "3", "4", "5", "6", "7", "9", "10", "11"]
    random.Random(20261122).shuffle(expected)
    assert groups == [expected[:4], expected[4:8]]
    assert tail == expected[8:]
    with pytest.raises(ValueError, match="duplicate"):
        lock.grouped_ids(valid + [valid[0]], set())


def test_historical_returned_is_full_ordered_retrieval(tmp_path, monkeypatch):
    cdir = tmp_path / "eval/feasibility"
    ddir = tmp_path / "data/feasibility"
    cdir.mkdir(parents=True)
    ddir.mkdir(parents=True)
    (cdir / "run_config.json").write_text(json.dumps({"n_tasks": 1, "seed": 89}))
    step = {"new": [31, 27], "gold": 31, "question_index": 0, "hop_index": 0}
    test_task = {"task": 0, "steps": [step]}
    (ddir / "tasks_dev150.jsonl").write_text(lock.compact(test_task) + "\n")
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    monkeypatch.setattr(lock, "build_task", lambda *args: test_task)
    configs = ["no_compress"] + [f"{p}|{m}|{t}" for p in ("bm25", "latest", "oracle")
                                 for m in ("append", "reselect") for t in lock.BUDGETS]
    all_records = [{"task": 0, "step": 0, "question_index": 0, "hop_index": 0, "gold": 31,
                    "new": [31], "returned": [31, 27], "config": c} for c in configs]
    file = ddir / "steps.jsonl"
    file.write_text("".join(json.dumps(r) + "\n" for r in all_records))
    valid = [{"id": i} for i in range(4)]
    assert lock.reproduce(valid, None, None)["historical_rows"] == 19
    all_records[-1]["returned"].reverse()
    file.write_text("".join(json.dumps(r) + "\n" for r in all_records))
    with pytest.raises(ValueError, match="historical steps mismatch"):
        lock.reproduce(valid, None, None)

