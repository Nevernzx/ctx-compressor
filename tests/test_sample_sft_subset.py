"""Task 05 deterministic coverage, heldout independence and source preservation."""
from collections import Counter
import hashlib
import json
import random

import pytest

from scripts import sample_sft_subset as sample


def rows(sizes=(1, 2, 5), split="train"):
    output = []
    for task, size in enumerate(sizes):
        for mode in sample.MODES:
            for step in range(size):
                meta = {"task": task, "step": step, "mode": mode, "split": split,
                        "tier": sample.TIERS[task % 3], "n_input_tokens": 10,
                        "n_target_tokens": 2, "total_tokens": 12, "loss_tokens": 2}
                output.append({"source_line_1based": len(output) + 1, "meta": meta})
    return output


def test_training_coverage_short_tasks_and_mode_balance():
    population = rows()
    chosen = sample.select_training(population, per_mode=6)
    counts = Counter((item["meta"]["mode"], item["meta"]["task"])
                     for item in population if item["source_line_1based"] in chosen)
    assert len(chosen) == len(set(chosen)) == 12
    assert chosen == sorted(chosen)
    for mode in sample.MODES:
        assert counts[mode, 0] == 1
        assert counts[mode, 1] == 2
        assert counts[mode, 2] == 3


def test_sampling_does_not_depend_on_iteration_order_or_global_rng():
    population = rows((4, 4, 4))
    expected = sample.select_training(population, per_mode=8)
    random.seed(12345)
    random.shuffle(population)
    assert sample.select_training(population, per_mode=8) == expected
    assert sample.select_training(population, per_mode=8, seed=5) != expected


def test_random_protocol_uses_one_rng_modes_tasks_and_source_order():
    population = rows((4, 4, 4))
    rng, expected = random.Random(sample.SEED), set()
    for mode in ("reselect", "append"):
        selected = set()
        for task in range(3):
            group = [item["source_line_1based"] for item in population
                     if item["meta"]["mode"] == mode and item["meta"]["task"] == task]
            selected.update(rng.sample(group, 2))
        remaining = [item["source_line_1based"] for item in population
                     if item["meta"]["mode"] == mode and item["source_line_1based"] not in selected]
        selected.update(rng.sample(remaining, 2))
        expected.update(selected)
    assert sample.select_training(population, per_mode=8) == sorted(expected)


@pytest.mark.parametrize("size", [4, 9])
def test_infeasible_size_fails_instead_of_dropping_task_coverage(size):
    with pytest.raises(ValueError, match="coverage/population"):
        sample.select_training(rows(), per_mode=size)


def test_eval_loss_has_independent_uniform_rng():
    population = rows((4, 4, 4), "heldout")
    expected = sorted(random.Random(sample.SEED).sample(list(range(1, 25)), 7))
    sample.select_training(rows((5, 5, 5)), per_mode=8)
    assert sample.select_eval_loss(population, count=7) == expected
    assert sample.select_eval_loss(list(reversed(population)), count=7) == expected


@pytest.mark.parametrize("field,value", [("split", "dev"), ("mode", "unknown"),
                                        ("task", True), ("step", -1), ("tier", "bad")])
def test_invalid_metadata_rejected(field, value):
    population = rows()
    population[0]["meta"][field] = value
    with pytest.raises(ValueError):
        sample.select_training(population, per_mode=6)


def test_duplicate_identity_and_missing_mode_task_rejected():
    population = rows()
    duplicate = {"source_line_1based": 17, "meta": dict(population[0]["meta"])}
    with pytest.raises(ValueError, match="Duplicate sample"):
        sample.select_training(population + [duplicate], per_mode=6)
    with pytest.raises(ValueError, match="coverage"):
        sample.select_training(population[1:], per_mode=6)


def source_file(tmp_path):
    population = rows()
    raw = [(json.dumps({"messages": [{"role": "user", "content": "原文"}], "meta": item["meta"]},
                       ensure_ascii=False, separators=(",", ":")) + "\n").encode() for item in population]
    path = tmp_path / "source.jsonl"
    path.write_bytes(b"".join(raw))
    return raw, {"path": str(path), "bytes": path.stat().st_size, "lines": len(raw),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_full_source_verification_and_byte_preserving_copy(tmp_path):
    raw, record = source_file(tmp_path)
    population = sample.scan_source(record, "train", expected_tasks=3)
    selected = sample.select_training(population, per_mode=6)
    target = tmp_path / "selected.jsonl"
    manifest = sample.copy_selected(record, selected, target)
    assert target.read_bytes() == b"".join(raw[i - 1] for i in selected)
    assert manifest["sha256"] == sample.digest(target)
    assert manifest["lines"] == 12
    assert sample.statistics([item for item in population if item["source_line_1based"] in selected])["by_tier"] == {
        "tight": 2, "mid": 4, "loose": 6}
    with pytest.raises(ValueError, match="overwrite"):
        sample.copy_selected(record, selected, target)


def test_full_hash_detects_same_size_source_mutation_even_unselected_row(tmp_path):
    _, record = source_file(tmp_path)
    path = tmp_path / "source.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"n_input_tokens":10', b'"n_input_tokens":11', 1))
    with pytest.raises(ValueError, match="SHA256"):
        sample.scan_source(record, "train", expected_tasks=3)
    target = tmp_path / "selected.jsonl"
    with pytest.raises(ValueError, match="Source changed"):
        sample.copy_selected(record, [2, 3], target)
    assert not target.exists()


def test_expected_task_coverage_and_eval_size_are_enforced():
    with pytest.raises(ValueError, match="task ID coverage"):
        sample.validate_metadata(rows(), "train", expected_tasks=4)
    with pytest.raises(ValueError, match="eval-loss count"):
        sample.select_eval_loss(rows(split="heldout"), count=17)
