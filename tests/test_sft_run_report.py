"""Task 05 report arithmetic, heldout-only selection, and pending evidence gates."""
from copy import deepcopy
import csv
import hashlib
import io
import json

import pytest

from scripts import sft_run_report as report


def metric(n, losses=10, recall=.9):
    steps = n * 4
    rate = lambda count, total: {"count": count, "n": total, "rate": count / total}
    value = {"n_tasks": n, "n_steps": steps,
        "step_recall": {"micro": {"n": steps, "mean": .8}, "task_macro": {"n": n, "mean": .8}},
        "final_recall": {"n": n, "mean": recall}, "evidence_loss": rate(losses, n),
        "evidence_complete_questions": rate(n * 3, n * 4),
        "occupancy": {"n": n, "mean": .7, "max": 1.0},
        "requested_budget_occupancy": {"n": n, "mean": .7, "max": 1.0},
        "invalid_outputs": {name: rate(0, steps) for name in (*report.INVALID_REASONS, "any_invalid")},
        "output_limit": rate(0, steps),
        "cost": {name: {"n": n, "mean": 123.0, "relative_to_no_compress": .3} for name in ("plain", "cached")}}
    for family, keys in (("hop", ("1", "2", "3", "4")), ("question_position", ("1", "2", "3", "4")),
                          ("mechanism", ("never_kept", "dropped"))):
        value["evidence_loss"]["by_" + family] = {
            key: {**rate(losses, n), "lost_evidence_count": losses, "evidence_n": n} for key in keys}
    return value


def strategy(name, losses=10, recall=.9):
    return {"schema_version": 1, "strategy": name,
        "checkpoint_step": int(name.rsplit("-", 1)[1]) if name.startswith("checkpoint-") else None,
        "settings": {"split": "train-heldout", "saved_retrieval_steps": True,
                     "n_tasks": 249, "n_trajectories": 1494, "budgets": report.BUDGETS, "modes": list(report.MODES)},
        "configs": {f"{name}|{mode}|{tier}": {"mode": mode, "budget_tier": tier, "budget": budget,
                     "metrics": metric(249, losses, recall)} for mode in report.MODES for tier, budget in report.BUDGETS.items()},
        "combined": metric(1494, losses * 6, recall)}


def test_missing_loss_measurements_remain_empty_and_csv_preserves_sample_denominators():
    config = {"runtime": {"loss_curve": [{"step": 1, "loss": 2.0, "epoch": .001},
                                        {"step": 500, "loss": .5, "learning_rate": .0001}],
                          "eval_curve": [{"step": 500, "eval_loss": .7}]}}
    rows = report.loss_rows(config)
    assert len(rows) == 2 and "eval_loss" not in rows[0]
    assert rows[1]["train_n_exposures"] == 16 and rows[1]["eval_n_samples"] == 1000
    parsed = list(csv.DictReader(io.StringIO(report.loss_csv(rows))))
    assert parsed[0]["eval_loss"] == "" and parsed[1]["eval_loss"] == "0.7"
    assert report.loss_rows({}) == []


@pytest.mark.parametrize("points", [[{"step": 1, "loss": float("nan")}],
                                    [{"step": 1, "loss": 1}, {"step": 1, "loss": 2}],
                                    [{"step": 2501, "loss": 1}]])
def test_loss_curve_rejects_nonfinite_duplicate_and_out_of_range_points(points):
    with pytest.raises(ValueError):
        report.loss_rows({"runtime": {"loss_curve": points}})


def test_complete_strategy_validation_preserves_original_metrics():
    value = strategy("bm25")
    before = deepcopy(value)
    assert report.validate_strategy(value, "bm25") is value
    assert value == before


@pytest.mark.parametrize("mutation", ["dev", "missing_cell", "wrong_n", "pooled_loss", "pooled_recall",
                                     "nonfinite", "overlapping_invalid", "wrong_checkpoint"])
def test_incomplete_or_inconsistent_evaluation_results_rejected(mutation):
    value = strategy("checkpoint-500")
    if mutation == "dev":
        value["settings"]["split"] = "dev"
    elif mutation == "missing_cell":
        value["configs"].pop(next(iter(value["configs"])))
    elif mutation == "wrong_n":
        next(iter(value["configs"].values()))["metrics"]["final_recall"]["n"] = 248
    elif mutation == "pooled_loss":
        value["combined"]["evidence_loss"]["count"] += 1
        value["combined"]["evidence_loss"]["rate"] = 61 / 1494
    elif mutation == "pooled_recall":
        value["combined"]["final_recall"]["mean"] = .91
    elif mutation == "nonfinite":
        value["combined"]["final_recall"]["mean"] = float("nan")
    elif mutation == "overlapping_invalid":
        value["combined"]["invalid_outputs"]["invalid_json"].update(count=1, rate=1 / (1494 * 4))
    elif mutation == "wrong_checkpoint":
        value["checkpoint_step"] = 1000
    with pytest.raises(ValueError):
        report.validate_strategy(value, "checkpoint-500")


def test_no_provisional_checkpoint_selection_and_ties_use_recall_then_step():
    results = {f"checkpoint-{step}": strategy(f"checkpoint-{step}") for step in report.STEPS[:-1]}
    results["base"] = strategy("base", losses=0, recall=1)
    assert report.select_complete_checkpoints(results) is None
    results["checkpoint-2500"] = strategy("checkpoint-2500", losses=11, recall=.99)
    selected = report.select_complete_checkpoints(results)
    assert selected["checkpoint_step"] == 500 and selected["eval_loss_used"] is False
    results["checkpoint-1000"] = strategy("checkpoint-1000", losses=10, recall=.95)
    assert report.select_complete_checkpoints(results)["checkpoint_step"] == 1000


def sample():
    return {"statistics": {"train": {"tokens": {"total_tokens": {"sum": 60000000}}}}}


def benchmark():
    trials = [{"batch_size": batch, "gradient_accumulation_steps": 16 // batch,
        "group_by_length": batch == 8,
        "status": "ok", "optimizer_steps": 30, "consumed_samples": 480, "consumed_tokens": 1200000,
        "train_seconds": seconds, "tokens_per_second": 1200000 / seconds,
        "selection_peak_bytes": 10000000000,
        "longest_probe": {"oom": False, "finite_gradients": True, "optimizer_steps": 0}}
        for batch, seconds in ((1, 1000), (4, 500), (8, 800))]
    return {"trials": trials, "selected": {"batch_size": 4, "gradient_accumulation_steps": 4,
                                          "tokens_per_second": 2400, "group_by_length": False},
        "group_by_length": "varies_by_trial", "group_by_length_by_batch": {"1": False, "4": False, "8": True},
        "formal_estimate": {"sample_exposures": 40000, "sequence_tokens": 120000000,
                            "train_loop_seconds": 50000, "train_loop_hours": 50000 / 3600}}


def test_benchmark_estimate_uses_exact_subset_tokens_and_real_measured_speed():
    result = report.batch_summary(benchmark(), sample())
    assert result["selection_verified"] and result["estimate_source_n"] == 20000
    value = benchmark()
    value["formal_estimate"]["sequence_tokens"] = 88562 * 3000 * 2
    with pytest.raises(ValueError, match="exact frozen"):
        report.batch_summary(value, sample())
    value = benchmark()
    value["trials"][0]["tokens_per_second"] = 999999
    with pytest.raises(ValueError, match="throughput"):
        report.batch_summary(value, sample())


def test_benchmark_rejects_stored_winner_that_violates_policy():
    value = benchmark()
    value["selected"]["batch_size"] = 8
    with pytest.raises(ValueError, match="Wrong batch"):
        report.batch_summary(value, sample())


def test_grouping_is_a_measured_configuration_dimension_not_a_uniform_batch_setting(tmp_path):
    value = benchmark()
    summary = report.batch_summary(value, sample())
    assert summary["group_by_length_by_batch"] == {"1": False, "4": False, "8": True}
    result = report.build_report_data(tmp_path)
    result["batch_benchmark"] = summary
    text = report.render_report(result)
    assert "1×16 | False" in text and "4×4 | False" in text and "8×2 | True" in text
    assert "完整配置" in text and "按长度分组会改变样本顺序" in text
    value["selected"]["group_by_length"] = True
    with pytest.raises(ValueError, match="differs from measured trial"):
        report.batch_summary(value, sample())
    value = benchmark()
    value["group_by_length_by_batch"]["8"] = False
    with pytest.raises(ValueError, match="metadata mismatch"):
        report.batch_summary(value, sample())


def formal_config():
    return {"lora_rank": 16, "lora_alpha": 32, "lora_dropout": .05, "target_modules": "all-linear",
        "learning_rate": 1e-4, "lr_scheduler_type": "cosine", "warmup_ratio": .03,
        "max_length": 8392, "truncation_strategy": "raise", "seed": 20261118,
        "bf16": True, "gradient_checkpointing": True, "template": "qwen3", "enable_thinking": False,
        "loss_scale": "last_round", "epochs_planned": 2, "effective_batch": 16, "batch_size": 4,
        "group_by_length": False,
        "gradient_accumulation_steps": 4, "cuda_visible_devices": "1", "checkpoint_selection_uses_eval_loss": False,
        "runtime": {"status": "ok", "optimizer_steps": 2500, "consumed_samples": 40000,
            "epoch": 2.0, "consumed_tokens": 120000000, "oom": False, "nonfinite": False,
            "wall_seconds": 5000, "train_seconds": 4900,
            "checkpoints": [f"/external/checkpoint-{step}" for step in report.STEPS],
            "loss_curve": [{"step": step, "loss": .9} for step in range(1, 2501)],
            "eval_curve": [{"step": step, "eval_loss": .8} for step in report.STEPS]}}


def test_formal_training_completion_requires_all_exposures_losses_and_checkpoints():
    value = formal_config()
    result = report.training_summary(value, sample())
    assert result["complete"] and result["measured_train_loss_n"] == 2500 and result["measured_eval_loss_n"] == 5
    value["runtime"]["consumed_samples"] = 39999
    with pytest.raises(ValueError, match="40,000"):
        report.training_summary(value, sample())
    value = formal_config()
    value["runtime"]["eval_curve"].pop()
    with pytest.raises(ValueError, match="five fixed"):
        report.training_summary(value, sample())
    value = formal_config()
    value["max_length"] = 8192
    with pytest.raises(ValueError, match="hyperparameters"):
        report.training_summary(value, sample())


def test_missing_all_measurements_is_pending_without_invented_numbers(tmp_path):
    result = report.build_report_data(tmp_path)
    assert result["status"] == "pending" and result["selected_checkpoint"] is None
    assert result["strategies"] == {} and result["loss_curve"]["rows"] == []
    text = report.render_report(result)
    assert "NA（五个完整结果未齐）" in text
    assert "训练成功" not in text and "训练失败" not in text


def test_existing_cpu_results_are_reported_while_models_are_pending(tmp_path):
    value = strategy("bm25")
    (tmp_path / "bm25_results.json").write_text(json.dumps(value))
    result = report.build_report_data(tmp_path)
    assert result["status"] == "pending" and result["strategies"]["bm25"] == value
    text = report.render_report(result)
    assert "90.000% (n=249)" in text and "10/249" in text and "0/996" in text
    assert "123.000 (n=249)" in text
    assert "never_kept" in text and "question_position" in text


def test_test_xml_counts_actual_cases_not_declared_suite_total(tmp_path):
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite tests="999">' + '<testcase/>' * 264 + '</testsuite></testsuites>')
    assert report.read_test_counts(path)["passed"]
    path.write_text('<testsuite tests="264"><testcase><failure/></testcase></testsuite>')
    result = report.read_test_counts(path)
    assert not result["passed"] and result["n"] == 1 and result["failures"] == 1


def test_complete_report_requires_all_evidence_and_current_plot(tmp_path):
    def save(name, value):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))

    sample_value = sample()
    # Only fields consumed by assembly are needed; rendering has separate tests.
    save("sample_manifest", sample_value)
    save("batch_bench", benchmark())
    config = formal_config()
    save("train_run_config", config)
    strategies = {name: strategy(name) for name in report.STRATEGIES}
    external_records = []
    for name, value in strategies.items():
        value["steps_artifact"] = {"path": f"/external/eval_rollouts/{name}/steps.jsonl", "bytes": 123,
                                   "sha256": "a" * 64}
        external_records.append({**value["steps_artifact"], "lines": value["combined"]["n_steps"]})
        save(name + "_results", value)
    save("cpu_baseline_results", {"strategies": {name: strategies[name] for name in report.BASELINES}})
    save("model_rollout_run", {"status": "complete", "cuda_visible_devices": "1",
        "generation": {"temperature": 0, "max_tokens": 256},
        "strategies": {name: {"status": "complete", "n_steps": strategies[name]["combined"]["n_steps"],
            "actual_training_prefix_check": {"exact_token_ids_match": True,
                                             "runtime_echoed_exact_prompt_ids": True}}
            for name in report.MODELS}})
    save("model_template_reference", {"status": "ok", "actual_training_prefix_equals_inference_ids": True})
    save("final_resource_check", {"root_limit_passed": True, "gpu_released": True, "only_gpu1": True,
        "root_delta_bytes": 1000000, "root_limit_bytes": 300000000,
        "gpu1_after": {"index": 1}, "task05_gpu_processes_remaining": []})
    save("external_artifact_manifest", {"schema_version": 1, "complete": True, "files": external_records})
    (tmp_path / "tests.xml").write_text('<testsuite>' + '<testcase/>' * 264 + '</testsuite>')
    csv_text = report.loss_csv(report.loss_rows(config))
    (tmp_path / "loss_curve.svg").write_text('<svg/>')
    save("loss_curve_plot", {"csv_sha256": hashlib.sha256(csv_text.encode()).hexdigest(),
         "svg_sha256": report.sha256(tmp_path / "loss_curve.svg"), "train_points": 2500, "eval_points": 5})
    result = report.build_report_data(tmp_path)
    assert result["status"] == "complete" and result["failure_reasons"] == [] and result["missing_inputs"] == []
    assert result["selected_checkpoint"]["checkpoint_step"] == 500
    config["group_by_length"] = True
    save("train_run_config", config)
    mismatch = report.build_report_data(tmp_path)
    assert mismatch["status"] == "failed" and any("未继承" in reason for reason in mismatch["failure_reasons"])
    config["group_by_length"] = False
    config["runtime"]["loss_curve"][0]["loss"] = .91
    save("train_run_config", config)
    stale = report.build_report_data(tmp_path)
    assert stale["status"] == "pending" and "loss_curve.svg_matching_current_csv" in stale["missing_inputs"]


def test_claimed_resource_pass_cannot_hide_excess_disk_or_wrong_gpu(tmp_path):
    (tmp_path / "final_resource_check.json").write_text(json.dumps({"root_limit_passed": True,
        "gpu_released": True, "only_gpu1": True, "root_delta_bytes": 300000001,
        "root_limit_bytes": 300000000, "gpu1_after": {"index": 0}, "task05_gpu_processes_remaining": []}))
    result = report.build_report_data(tmp_path)
    assert result["status"] == "failed"
    assert any("300MB" in reason for reason in result["failure_reasons"])
    assert any("GPU1" in reason for reason in result["failure_reasons"])
