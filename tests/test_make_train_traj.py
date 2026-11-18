"""Task 04 data split, streaming schema, and end-to-end rollout checks."""

from collections import Counter, defaultdict
import hashlib
import json
import random

import numpy as np
import pytest

from scripts.make_train_traj import (BUDGETS, SEED, JsonlWriter, filter_examples,
                                    make_task, restore_state_keys, split_examples,
                                    summarize_statistics, update_statistics)
from src.build_traj import oracle_action, simulate_task


def examples(n=83):
    return [{"id": f"example-{i}", "question": f"Question {i}"} for i in range(n)]


def test_split_uses_id_draw_and_original_partition_order_with_exact_remainders():
    source = examples()
    tasks, metadata = split_examples(source)
    heldout = random.Random(SEED).sample([e["id"] for e in source], 4)
    assert metadata["heldout_ids"] == heldout
    used = set()
    for split in ("train", "heldout"):
        pool = [ex for ex in source if (ex["id"] in heldout) == (split == "heldout")]
        expected = random.Random(SEED).sample(pool, len(pool) // 4 * 4)
        actual = [ex for group in tasks[split] for ex in group]
        assert actual == expected
        ids = {ex["id"] for ex in actual}
        assert not ids & used
        used.update(ids)
        discarded = [ex["id"] for ex in pool if ex["id"] not in ids]
        assert metadata["partitions"][split]["discarded_ids"] == discarded
    assert metadata["partitions"]["train"]["discarded_questions"] == 3
    assert metadata["partitions"]["heldout"]["tasks"] == 1


def test_small_partition_drops_all_unfillable_questions():
    tasks, metadata = split_examples(examples(47))
    assert len(tasks["train"]) == 11
    assert tasks["heldout"] == []
    assert metadata["partitions"]["train"]["discarded_questions"] == 1
    assert metadata["partitions"]["heldout"]["discarded_questions"] == 2


def test_duplicate_question_ids_cannot_leak_across_partitions():
    with pytest.raises(ValueError, match="duplicate"):
        split_examples(examples() + examples(1))


class FixedIndex:
    def retrieve(self, queries, k, **kwargs):
        return np.array([list(range(k))]), np.array([[1.0] * k])


def task_fixture():
    corpus = [{"title": f"Title {i}", "text": f"Title {i}\nBody {i}",
               "paragraph_text": f"Body {i}", "ntok": 25} for i in range(24)]
    keys = {(p["title"], p["paragraph_text"]): i for i, p in enumerate(corpus)}
    source = []
    for i in range(4):
        source.append({
            "id": f"task-example-{i}", "question": f"Question {i}",
            "paragraphs": [{"idx": i, "title": corpus[i]["title"],
                            "paragraph_text": corpus[i]["paragraph_text"]}],
            "question_decomposition": [
                {"question": f"Question hop {i}", "answer": f"Answer {i}",
                 "paragraph_support_idx": i},
                {"question": "Where was #1 born?", "answer": "Place",
                 "paragraph_support_idx": i}],
        })
    return source, corpus, FixedIndex(), keys


def test_task_reuses_exact_simulate_steps_and_clean_targets_and_json_keys():
    source, corpus, index, keys = task_fixture()
    task = make_task(7, source, "train", corpus, index, keys)
    assert task["steps"] == simulate_task(source, index, keys, top_n=20, seed=7)
    assert task["tier"] == random.Random(SEED + 7).choice(["tight", "mid", "loose"])
    assert task["budget"] == BUDGETS[task["tier"]]
    decoded = json.loads(json.dumps(task))
    for mode in ("reselect", "append"):
        assert len(decoded["rollouts"][mode]) == 8
        for serialized in decoded["rollouts"][mode]:
            assert all(isinstance(k, str) for k in serialized["ntok"])
            restored = restore_state_keys(serialized)
            assert restored["sft_target"] == oracle_action(restored)
            assert all(type(k) is int for k in restored["pid_by_lid"])
    assert any(s["action"] != s["sft_target"] for s in task["rollouts"]["reselect"])
    assert any(s["sft_target"]["drop"] for s in task["rollouts"]["reselect"])
    assert all(s["action"] == s["sft_target"] for s in task["rollouts"]["append"])
    counters = defaultdict(Counter)
    update_statistics(task, counters)
    summary = summarize_statistics(counters)
    assert summary["samples"] == 16
    assert summary["natural_hit"] == {"count": 8, "n": 8, "rate": 1.0}


def test_filter_keeps_literal_reference_and_reports_broken_evidence():
    source, _, _, keys = task_fixture()
    source[0]["question_decomposition"][0]["question"] = "Who wrote #9 Dream?"
    source[1]["question_decomposition"][0]["paragraph_support_idx"] = 999
    valid, report = filter_examples(source, keys)
    assert len(valid) == 3
    assert report["rejected"] == 1
    assert report["literal_out_of_range_reference_count"] == 1
    assert report["literal_out_of_range_reference_examples"][0]["id"] == source[0]["id"]


def test_streaming_writer_hashes_exact_bytes_and_refuses_overwrite(tmp_path):
    path = tmp_path / "records.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"text": "中文", "task": 0})
        writer.write({"text": "second", "task": 1})
        assert not path.exists()
    data = path.read_bytes()
    record = writer.record()
    assert record["lines"] == data.count(b"\n") == 2
    assert record["bytes"] == len(data)
    assert record["sha256"] == hashlib.sha256(data).hexdigest()
    with pytest.raises(FileExistsError):
        with JsonlWriter(path):
            pass


def test_failed_stream_keeps_partial_and_never_publishes_complete_file(tmp_path):
    path = tmp_path / "failed.jsonl"
    with pytest.raises(RuntimeError):
        with JsonlWriter(path) as writer:
            writer.write({"task": 0})
            raise RuntimeError("synthetic failure")
    assert not path.exists()
    assert path.with_suffix(".jsonl.partial").read_text() == '{"task":0}\n'
