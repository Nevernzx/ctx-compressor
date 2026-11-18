"""Test timing arithmetic and evidence boundaries without model imports."""

import hashlib
import json

import pytest

from scripts.sft_prep_report import (benchmark_summary, estimate_plans,
                                     scan_split, test_counts as read_test_counts)


def measured_benchmark():
    return {"status": "ok", "consumed_tokens": 8000000, "selected_total_tokens": 8000000,
            "train_seconds": 4000.0, "wall_seconds": 4150.0,
            "consumed_samples": 1600, "optimizer_steps": 100, "oom": False,
            "loss_curve": [{"step": i+1, "loss": 2.0 - i/100} for i in range(100)],
            "longest_probe": {"seconds": 50.0, "sequence_tokens": 8390,
                              "oom": False, "finite_gradients": True,
                              "optimizer_states_resident": True, "optimizer_steps": 0}}


def test_estimates_use_full_train_length_one_startup_and_all_five_plans():
    plans = estimate_plans(88562, 6000.0, measured_benchmark())
    assert [(p["samples"], p["epochs"]) for p in plans] == [
        (88562, 1), (88562, 2), (88562, 3), (20000, 2), (40000, 2)]
    for p in plans:
        assert p["sample_exposures"] == p["samples"] * p["epochs"]
        assert p["one_time_startup_seconds"] == 100.0
        assert p["wall_seconds"] == 100.0 + p["sample_exposures"] * 6000 / 2000
        assert p["gpu1_occupancy_seconds"] == p["wall_seconds"]
        assert p["mean_length_source_n"] == 88562
    assert plans[1]["wall_seconds"] != 2 * plans[0]["wall_seconds"]


def test_negative_startup_is_clamped_and_incomplete_benchmark_not_extrapolated():
    b = measured_benchmark()
    b["wall_seconds"] = 4001
    assert estimate_plans(88562, 6000, b)[0]["one_time_startup_seconds"] == 0
    b["optimizer_steps"] = 99
    assert estimate_plans(88562, 6000, b) == []
    b = measured_benchmark(); b["train_seconds"] = 0
    assert estimate_plans(88562, 6000, b) == []


def test_benchmark_recomputes_speed_and_loss_and_checks_post_training_probe():
    b = measured_benchmark()
    b["tokens_per_second"] = 999999  # Saved summary cannot replace measured numerator/time.
    b["first_10_loss_mean"] = -1
    summary = benchmark_summary(b, {"observed_max_tokens": 8390})
    assert summary["tokens_per_second"] == 2000
    assert summary["first_10_loss_mean"] == pytest.approx(1.955)
    assert summary["last_10_loss_mean"] == pytest.approx(1.055)
    assert all(summary["acceptance_checks"].values())
    b["longest_probe"]["optimizer_steps"] = 1
    assert not benchmark_summary(b, {"observed_max_tokens": 8390})["acceptance_checks"]["longest_no_extra_optimizer_step"]


def test_split_quantiles_pool_actual_rows_and_verify_bytes(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = []
    for length in (100, 200, 400, 800):
        rows.append({"meta": {"split": "train", "n_input_tokens": length-10,
                              "n_target_tokens": 10, "loss_tokens": 10, "total_tokens": length}})
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(data)
    record = {"path": str(path), "lines": 4, "sha256": hashlib.sha256(data).hexdigest()}
    summary = scan_split(record, "train")
    d = summary["lengths"]["total_tokens"]
    assert summary["samples"] == d["n"] == 4
    assert d["mean"] == 375
    assert d["p50"] == 300
    assert d["max"] == 800
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        scan_split(record, "train")


def test_full_test_xml_counts_cases_and_failures_not_nested_suite_totals(tmp_path):
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite tests="999"><testcase name="a"/>'
                    '<testcase name="b"><failure/></testcase><testcase name="c"><skipped/>'
                    '</testcase></testsuite></testsuites>')
    assert read_test_counts(path) == {"n": 3, "failures": 1, "errors": 0, "skipped": 1, "passed": False}
