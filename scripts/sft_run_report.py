#!/usr/bin/env python3
"""Build Task 05 evidence-only reports; missing measurements remain pending.

Run `build` with the existing ctxc Python. Run `plot` with the existing train
Python, then `build` again to bind the SVG to the current CSV hash. Both commands
are CPU only. Only Task 05 report/curve outputs are replaced; source artifacts
are read-only. No rollout, training, model loading, download or API is performed.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.heldout_eval import BUDGETS, INVALID_REASONS, MODES, select_checkpoint
from src.training_control import choose_batch, checkpoint_steps, validate_training_counts

STEPS = (500, 1000, 1500, 2000, 2500)
BASELINES = ("no_compress", "latest", "bm25", "oracle")
MODELS = ("base", *(f"checkpoint-{step}" for step in STEPS))
STRATEGIES = (*BASELINES, *MODELS)
CSV_FIELDS = ("step", "train_loss", "train_n_exposures", "eval_loss", "eval_n_samples",
              "learning_rate", "epoch", "elapsed_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    return isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def loss_rows(config):
    """Join measured losses by optimizer step without filling missing values."""
    runtime, joined = config.get("runtime", {}), {}
    for key, field, count_field, count in (("loss_curve", "loss", "train_n_exposures", 16),
                                          ("eval_curve", "eval_loss", "eval_n_samples", 1000)):
        seen = set()
        for point in runtime.get(key, []):
            step = point["step"]
            require(type(step) is int and 0 < step <= 2500 and step not in seen, "Invalid/duplicate loss step")
            value = point[field]
            require(isinstance(value, (int, float)) and math.isfinite(value), "Nonfinite recorded loss")
            seen.add(step)
            row = joined.setdefault(step, {"step": step})
            row["train_loss" if field == "loss" else "eval_loss"] = value
            row[count_field] = count
            for name in ("learning_rate", "epoch", "elapsed_seconds"):
                if point.get(name) is not None:
                    require(isinstance(point[name], (int, float)) and math.isfinite(point[name]),
                            f"Nonfinite loss-curve metadata: {name}")
                    row[name] = point[name]
    return [joined[step] for step in sorted(joined)]


def loss_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def validate_strategy(value, expected_name):
    """Reject incomplete/dev grids or denominator drift before reporting/ranking."""
    require(value["strategy"] == expected_name, "Strategy filename/identity mismatch")
    settings, combined = value["settings"], value["combined"]
    require(settings["split"] == "train-heldout" and settings["saved_retrieval_steps"] is True,
            "Only saved train-heldout retrieval steps are allowed")
    require(settings["n_tasks"] == 249 and settings["n_trajectories"] == 1494
            and settings["budgets"] == BUDGETS and set(settings["modes"]) == set(MODES), "Incomplete evaluation grid")
    expected = {f"{expected_name}|{mode}|{tier}" for mode in MODES for tier in BUDGETS}
    require(set(value["configs"]) == expected, "Expected exactly six mode/budget cells")
    group_metrics = []
    for mode in MODES:
        for tier, budget in BUDGETS.items():
            cell = value["configs"][f"{expected_name}|{mode}|{tier}"]
            require(cell["mode"] == mode and cell["budget_tier"] == tier and cell["budget"] == budget,
                    "Mode/budget metadata mismatch")
            require(cell["metrics"]["n_tasks"] == 249, "Each evaluation cell requires 249 tasks")
            group_metrics.append(cell["metrics"])
    for metric, expected_n in [(m, 249) for m in group_metrics] + [(combined, 1494)]:
        n_steps = metric["n_steps"]
        require(type(n_steps) is int and n_steps > 0, "Missing evaluation step denominator")
        for key in ("final_recall", "evidence_loss", "occupancy"):
            require(metric[key]["n"] == expected_n, f"Wrong {key} denominator")
        require(metric["step_recall"]["micro"]["n"] == n_steps
                and metric["step_recall"]["task_macro"]["n"] == expected_n, "Wrong recall denominator")
        require(metric["evidence_complete_questions"]["n"] == expected_n * 4, "Wrong question denominator")
        for rate in [metric["evidence_loss"], metric["evidence_complete_questions"],
                     *metric["invalid_outputs"].values(), metric["output_limit"]]:
            require(type(rate["n"]) is int and rate["n"] > 0
                    and type(rate["count"]) is int and 0 <= rate["count"] <= rate["n"]
                    and close(rate["rate"], rate["count"] / rate["n"]), "Invalid rate/count denominator")
        require(set(metric["invalid_outputs"]) == {*INVALID_REASONS, "any_invalid"}, "Missing invalid-output category")
        require(all(item["n"] == n_steps for item in metric["invalid_outputs"].values())
                and metric["output_limit"]["n"] == n_steps, "Wrong invalid-output denominator")
        require(sum(metric["invalid_outputs"][name]["count"] for name in INVALID_REASONS)
                == metric["invalid_outputs"]["any_invalid"]["count"], "Invalid-output categories are not exclusive")
        for field in (metric["final_recall"], metric["step_recall"]["micro"], metric["step_recall"]["task_macro"]):
            require(isinstance(field["mean"], (int, float)) and math.isfinite(field["mean"])
                    and 0 <= field["mean"] <= 1, "Invalid recall mean")
        for key in ("plain", "cached"):
            cost = metric["cost"][key]
            require(cost["n"] == expected_n and math.isfinite(cost["mean"]) and cost["mean"] >= 0,
                    "Invalid task cost")
    require(combined["n_tasks"] == 1494 and combined["n_steps"] == sum(m["n_steps"] for m in group_metrics),
            "Combined grid denominator mismatch")
    require(combined["evidence_loss"]["count"] == sum(m["evidence_loss"]["count"] for m in group_metrics),
            "Combined evidence loss differs from six cells")
    require(close(combined["final_recall"]["mean"], sum(m["final_recall"]["mean"] * m["n_tasks"]
                                                       for m in group_metrics) / 1494),
            "Combined final recall differs from six cells")
    if expected_name.startswith("checkpoint-"):
        require(value.get("checkpoint_step") == int(expected_name.rsplit("-", 1)[1]), "Wrong checkpoint step")
    return value


def select_complete_checkpoints(strategies):
    """Do not select a provisional winner while any planned checkpoint is absent."""
    names = [f"checkpoint-{step}" for step in STEPS]
    if not all(name in strategies for name in names):
        return None
    winner = select_checkpoint([strategies[name] for name in names])
    return {"strategy": winner["strategy"], "checkpoint_step": winner["checkpoint_step"],
            "evidence_loss": winner["combined"]["evidence_loss"],
            "final_recall": winner["combined"]["final_recall"],
            "n_checkpoints": 5, "n_cells": 6, "n_trajectories_per_checkpoint": 1494,
            "rule": "pooled evidence-loss count ascending, pooled final recall descending, checkpoint step ascending",
            "eval_loss_used": False}


def batch_summary(batch, sample):
    trials = []
    for trial in batch["trials"]:
        copy = dict(trial)
        grouped = trial.get("group_by_length", batch.get("group_by_length"))
        require(type(grouped) is bool, "Each benchmark configuration must declare actual group_by_length")
        copy["group_by_length"] = grouped
        if trial.get("status") == "ok":
            require(trial["train_seconds"] > 0, "Invalid benchmark timing")
            copy["tokens_per_second"] = trial["consumed_tokens"] / trial["train_seconds"]
            require(close(trial["tokens_per_second"], copy["tokens_per_second"]), "Benchmark throughput mismatch")
        trials.append(copy)
    require(sorted(trial["batch_size"] for trial in trials) == [1, 4, 8], "Three fresh batch trials required")
    require(all(trial["gradient_accumulation_steps"] * trial["batch_size"] == 16 for trial in trials),
            "Benchmark effective batch changed")
    winner = choose_batch(trials)
    require(batch["selected"]["batch_size"] == winner["batch_size"]
            and close(batch["selected"]["tokens_per_second"], winner["tokens_per_second"]), "Wrong batch selection")
    require(batch["selected"].get("group_by_length") is winner["group_by_length"],
            "Selected configuration group_by_length differs from measured trial")
    flags = {str(trial["batch_size"]): trial["group_by_length"] for trial in trials}
    if "group_by_length_by_batch" in batch:
        require(batch["group_by_length_by_batch"] == flags, "Benchmark group_by_length metadata mismatch")
    tokens = sample["statistics"]["train"]["tokens"]["total_tokens"]["sum"] * 2
    estimate = batch["formal_estimate"]
    require(estimate["sample_exposures"] == 40000 and estimate["sequence_tokens"] == tokens
            and close(estimate["train_loop_seconds"], tokens / winner["tokens_per_second"]),
            "Estimate must use exact frozen 20k token sum times two")
    return {**batch, "trials": trials, "selection_verified": True,
            "estimate_source_n": 20000, "benchmark_trial_n": 3,
            "group_by_length_by_batch": flags,
            "comparison_unit": "complete configurations: batch_size, accumulation and group_by_length"}


def training_summary(config, sample):
    runtime = config.get("runtime", {})
    rows = loss_rows(config)
    result = {"config": config, "runtime": runtime, "measured_train_loss_n": sum("train_loss" in r for r in rows),
              "measured_eval_loss_n": sum("eval_loss" in r for r in rows), "complete": False}
    if runtime.get("status") != "ok":
        return result
    validate_training_counts(runtime["optimizer_steps"], runtime["consumed_samples"], runtime["epoch"])
    checkpoint_steps(runtime["checkpoints"])
    require([row["step"] for row in rows if "train_loss" in row] == list(range(1, 2501)), "Expected 2500 train loss logs")
    require([row["step"] for row in rows if "eval_loss" in row] == list(STEPS), "Expected five fixed eval losses")
    require(runtime["oom"] is False and runtime["nonfinite"] is False, "Formal training recorded OOM/nonfinite")
    require(runtime["consumed_tokens"] == sample["statistics"]["train"]["tokens"]["total_tokens"]["sum"] * 2,
            "Formal training exposure token count changed")
    expected = {"lora_rank": 16, "lora_alpha": 32, "lora_dropout": .05, "target_modules": "all-linear",
                "learning_rate": 1e-4, "lr_scheduler_type": "cosine", "warmup_ratio": .03,
                "max_length": 8392, "truncation_strategy": "raise", "seed": 20261118,
                "bf16": True, "gradient_checkpointing": True, "template": "qwen3",
                "enable_thinking": False, "loss_scale": "last_round", "epochs_planned": 2,
                "effective_batch": 16, "cuda_visible_devices": "1", "checkpoint_selection_uses_eval_loss": False}
    require(all(config.get(key) == value for key, value in expected.items()), "Formal hyperparameters/protocol changed")
    require(config["batch_size"] * config["gradient_accumulation_steps"] == 16, "Formal effective batch changed")
    require(type(config.get("group_by_length")) is bool, "Formal group_by_length must be explicit")
    require(runtime["wall_seconds"] > 0 and runtime["train_seconds"] > 0, "Missing measured formal runtime")
    result["complete"] = True
    return result


def read_test_counts(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    counts = {"n": len(cases), **{name: sum(case.find(tag) is not None for case in cases)
        for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))}}
    counts["passed"] = counts["n"] >= 264 and not any(counts[k] for k in ("failures", "errors", "skipped"))
    return counts


def build_report_data(directory):
    directory = Path(directory)
    sources, missing, failures = {}, [], []

    def load(name, required=True):
        path = directory / (name + ".json")
        if not path.exists():
            if required:
                missing.append(path.name)
            return {}
        sources[name] = {"path": str(path), "sha256": sha256(path)}
        return json.loads(path.read_text())

    sample = load("sample_manifest")
    raw_batch, config = load("batch_bench"), load("train_run_config")
    cpu, model_run = load("cpu_baseline_results"), load("model_rollout_run")
    resources, external = load("final_resource_check"), load("external_artifact_manifest")
    proof = load("model_template_reference")
    wall_estimate = load("wall_time_estimate_update", required=False)
    batch, training = {}, {}
    for label, raw, operation in (("batch", raw_batch, lambda: batch_summary(raw_batch, sample)),
                                   ("training", config, lambda: training_summary(config, sample))):
        if raw and sample:
            try:
                value = operation()
                if label == "batch":
                    batch = value
                else:
                    training = value
            except (ValueError, KeyError, TypeError) as error:
                failures.append(f"{label}: {error}")
    if config:
        if config.get("status") == "failed" or config.get("runtime", {}).get("status") == "failed":
            failures.append("正式训练进程已记录failed")
        elif not training.get("complete"):
            missing.append("formal_training_completion")
        if batch:
            selected_config = batch["selected"]
            if (config.get("batch_size") != selected_config["batch_size"]
                    or config.get("group_by_length") is not selected_config["group_by_length"]):
                failures.append("正式训练未继承选定完整配置的batch/group_by_length")
    if model_run:
        if model_run.get("status") == "failed":
            failures.append("模型闭环推理已记录failed")
        elif model_run.get("status") != "complete":
            missing.append("model_rollout_completion")
    strategies = {}
    for name in STRATEGIES:
        result = load(name + "_results")
        if not result:
            continue
        try:
            validate_strategy(result, name)
            if name in BASELINES and cpu:
                require(cpu["strategies"][name] == result, "CPU receipt differs from strategy result")
            if name in MODELS and model_run:
                progress = model_run.get("strategies", {}).get(name, {})
                require(progress.get("status") == "complete", "Strategy result lacks completion receipt")
                check = progress["actual_training_prefix_check"]
                require(check["exact_token_ids_match"] is True and check["runtime_echoed_exact_prompt_ids"] is True,
                        "Actual generation/training token IDs did not match")
                require(progress["n_steps"] == result["combined"]["n_steps"], "Inference output count mismatch")
            strategies[name] = result
        except (ValueError, KeyError, TypeError) as error:
            failures.append(f"{name}: {error}")
    if model_run.get("status") == "complete":
        if model_run.get("cuda_visible_devices") != "1" or model_run.get("generation", {}).get("temperature") != 0:
            failures.append("推理GPU或greedy设置不符")
        if model_run.get("generation", {}).get("max_tokens") != 256:
            failures.append("推理生成上限不是256")
    if proof and (proof.get("status") != "ok" or proof.get("actual_training_prefix_equals_inference_ids") is not True):
        failures.append("训练/推理模板逐token核对未通过")
    selected = select_complete_checkpoints(strategies)
    if selected is None:
        missing.append("five_complete_checkpoint_metrics")
    if resources:
        for key in ("root_limit_passed", "gpu_released", "only_gpu1"):
            if resources.get(key) is not True:
                failures.append("资源验收未通过：" + key)
        if resources.get("task05_gpu_processes_remaining"):
            failures.append("仍有Task05 GPU进程")
        if (type(resources.get("root_delta_bytes")) is not int
                or resources.get("root_limit_bytes") != 300000000
                or resources["root_delta_bytes"] > 300000000):
            failures.append("根分区实测增量未满足300MB上限")
        if resources.get("gpu1_after", {}).get("index") != 1:
            failures.append("缺少GPU1终态记录")
        if "task05_gpu_processes_remaining" not in resources:
            failures.append("缺少Task05剩余GPU进程清单")
    if external and external.get("complete") is not True:
        missing.append("complete_external_artifact_inventory")
    elif external:
        if external.get("schema_version") != 1 or not external.get("files"):
            failures.append("最终外置产物清单为空或schema不符")
        external_files = {item["path"]: item for item in external.get("files", [])}
        for name, value in strategies.items():
            artifact = value.get("steps_artifact", value.get("run", {}).get("steps_file"))
            if not artifact:
                failures.append(f"{name}: 缺少逐步rollout产物记录")
                continue
            item = external_files.get(artifact["path"])
            if (item is None or any(item.get(key) != artifact[key] for key in ("bytes", "sha256"))
                    or item.get("lines") != value["combined"]["n_steps"]):
                failures.append(f"{name}: 最终清单未覆盖逐步rollout及行数")
    tests, test_path = None, directory / "tests.xml"
    if test_path.exists():
        sources["tests"] = {"path": str(test_path), "sha256": sha256(test_path)}
        tests = read_test_counts(test_path)
        if not tests["passed"]:
            failures.append("全量测试缺少原264项或存在失败/错误/跳过")
    else:
        missing.append(test_path.name)
    try:
        curve = loss_rows(config)
    except (ValueError, KeyError, TypeError) as error:
        failures.append("loss_curve: " + str(error))
        curve = []
    csv_text = loss_csv(curve)
    csv_hash = hashlib.sha256(csv_text.encode()).hexdigest()
    plot = load("loss_curve_plot", required=False)
    svg = directory / "loss_curve.svg"
    if not plot or not svg.exists() or plot.get("csv_sha256") != csv_hash or plot.get("svg_sha256") != sha256(svg):
        missing.append("loss_curve.svg_matching_current_csv")
    elif plot.get("train_points") != sum("train_loss" in row for row in curve) or plot.get("eval_points") != sum("eval_loss" in row for row in curve):
        failures.append("loss图点数不匹配CSV")
    else:
        sources["loss_curve_svg"] = {"path": str(svg), "sha256": sha256(svg)}
    environment_path = directory.parent / "sft_prep/environment_metadata.json"
    environment = {}
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        sources["frozen_train_environment"] = {"path": str(environment_path), "sha256": sha256(environment_path)}
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if failures else "pending" if missing else "complete",
            "status_meaning": "acceptance evidence completeness only; no claim about training effectiveness",
            "missing_inputs": sorted(set(missing)), "failure_reasons": failures, "sources": sources,
            "sampling": sample, "batch_benchmark": batch, "training": training,
            "wall_time_estimate_update": wall_estimate,
            "strategies": strategies, "selected_checkpoint": selected, "model_rollout": model_run,
            "template_proof": {key: proof.get(key) for key in ("status", "identity", "versions", "masked_label_range",
                                "supervised_label_range", "empty_think_tokens", "actual_training_prefix_equals_inference_ids")},
            "resources": resources, "tests": tests, "environment": environment,
            "external_artifact_manifest": external, "loss_curve": {"rows": curve, "csv_sha256": csv_hash,
                "eval_loss_used_for_checkpoint_selection": False, "train_n_per_point": 16, "eval_n_per_point": 1000}}


def number(value, digits=3):
    return "NA" if value is None else str(value) if type(value) is int else f"{value:.{digits}f}"


def rate(value):
    return "NA" if not value or value.get("rate") is None else f"{100 * value['rate']:.3f}% ({value['count']}/{value['n']})"


def mean(value, percent=False):
    if not value or value.get("mean") is None:
        return "NA"
    return f"{number(value['mean'] * (100 if percent else 1))}{'%' if percent else ''} (n={value['n']})"


def markdown_table(headers, rows):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(map(escape, headers)) + " |", "|" + "---|" * len(headers),
            *["| " + " | ".join(map(escape, row)) + " |" for row in rows], ""]


def metric_cells(result):
    for name in STRATEGIES:
        if name in result["strategies"]:
            for mode in MODES:
                for tier in BUDGETS:
                    cell = result["strategies"][name]["configs"][f"{name}|{mode}|{tier}"]
                    yield name, mode, tier, cell["metrics"]


def render_report(result):
    lines = ["# 任务05：LoRA SFT 与留出集闭环评测", "",
             f"验收记录状态：**{result['status']}**。状态只表示所需实测与验收材料是否齐备。", "",
             "评测复用 train 中按题留出的249个任务（每任务4题），两种模式×三档预算形成每策略1494条轨迹；"
             "不读取dev任务，不计算阅读模型F1。单步召回基于当前可得证据；末态召回/证据丢失基于该任务完整去重证据。"
             "合并主指标的n=1494为任务×模式×预算，基础独立任务n=249。", ""]
    if result["missing_inputs"]:
        lines += ["待完成：" + "、".join(result["missing_inputs"]) + "。缺失结果记NA；未齐五个checkpoint时不选择最优项。", ""]
    if result["failure_reasons"]:
        lines += ["验收记录问题：" + "；".join(result["failure_reasons"]) + "。", ""]
    lines += ["## 抽样与固定loss子集", ""]
    sample = result["sampling"]
    if sample:
        lines += ["训练使用单个random.Random(20261118)，模式依次reselect、append；任务ID升序，任务内按step升序；"
                  "各任务先随机取min(2,样本数)，再从该模式未选源行中均匀补齐到10000。"
                  "最终保存源行顺序及原始字节，原始meta与源行号在外置sample_ids.json。", ""]
        rows = []
        for split in ("train", "eval_loss"):
            for mode in MODES:
                value = sample["statistics"][split]["by_mode"][mode]
                rows.append([split, mode, value["samples"], value["tasks"], value["min_samples_per_task"],
                             value["max_samples_per_task"], *(value["by_tier"][tier] for tier in BUDGETS)])
        lines += markdown_table(["子集", "模式", "样本n", "覆盖任务n", "每任务min", "每任务max", "tight n", "mid n", "loose n"], rows)
        lines += ["eval-loss用单独重置的random.Random(20261118)从全部heldout SFT行均匀抽1000条，固定用于五次loss计算；"
                  "不参与checkpoint选择。两种模式覆盖的任务数允许不同。原数据源SHA在抽样前和复制时全量校验。", ""]
    lines += ["## A：batch吞吐、显存与预计时间", ""]
    batch = result["batch_benchmark"]
    if batch:
        rows = []
        for value in batch["trials"]:
            probe = value.get("longest_probe", {})
            rows.append([f"{value['batch_size']}×{value['gradient_accumulation_steps']}", value["group_by_length"], value["status"],
                         value.get("optimizer_steps"), value.get("consumed_samples"), value.get("consumed_tokens"),
                         number(value.get("train_seconds")), number(value.get("tokens_per_second")),
                         number(value.get("selection_peak_bytes", 0) / 1e9), probe.get("samples", "NA"),
                         probe.get("sequence_lengths", "NA"), probe.get("oom", value.get("oom")),
                         probe.get("finite_gradients", "NA")])
        lines += markdown_table(["batch×累积", "group_by_length", "状态", "optimizer步n", "样本n", "真实token n", "训练秒", "token/s",
                                 "峰值GB(n=1 trial)", "最长batch样本n", "序列长度", "最长OOM", "梯度有限"], rows)
        chosen, estimate = batch["selected"], batch["formal_estimate"]
        lines += [f"选定batch={chosen['batch_size']}、累积={chosen['gradient_accumulation_steps']}、"
                  f"group_by_length={chosen['group_by_length']}，"
                  f"吞吐{number(chosen['tokens_per_second'])} token/s（n=480样本、30 optimizer步）。"
                  "选择规则：在最长batch无OOM且显存≤40 GB的配置中取最高token/s；显存取torch reserved峰值+初始背景与"
                  "整卡采样峰值的较大者。最长batch在Adam状态常驻时前后向，额外optimizer步n=0。", "",
                  "三组比较单位为表中完整配置：batch、累积与group_by_length。使用同一批n=480样本、"
                  "相同seed=20261118与每组n=30 optimizer步；按长度分组会改变样本顺序，速度/显存差异不能全部归因于batch大小。"
                  "正式训练继承选定配置的group_by_length。", "",
                  f"按已抽训练子集的真实总长×2（n=20000唯一样本、40000次暴露、{estimate['sequence_tokens']} token），"
                  f"预计训练循环{number(estimate['train_loop_seconds'])}秒 / {number(estimate['train_loop_hours'])}小时"
                  "（速度依据n=1个选定trial）。CPU编码/模型启动、五次验证与保存另计；正式实测墙钟见下表。", ""]
    else:
        lines += ["吞吐、峰值显存、最长batch与配置选择：NA（待实测）。", ""]
    update = result.get("wall_time_estimate_update")
    if update:
        basis = update["basis"]
        startup, evaluation, first = basis["startup"], basis["first_eval"], basis["first_500_steps"]
        lines += ["### 首次500步后的B阶段墙钟外推", "",
                  f"实测startup={number(startup['seconds'])}秒（n={startup['n_runs']}次）；"
                  f"首次eval={number(evaluation['seconds'])}秒（n={evaluation['n_runs']}次、固定n={evaluation['n_samples']}样本，"
                  f"eval loss={number(evaluation['loss'], 8)}）；"
                  f"首500步训练={number(first['train_seconds'])}秒（n={first['optimizer_steps_n']} optimizer步、"
                  f"{first['sample_exposures_n']}次样本暴露）。首500步elapsed在首次eval前记录，不包含该次eval或checkpoint保存。", ""]
        labels = {"A_token_throughput": "A的token吞吐外推", "first_500_training_steps": "首500步训练外推"}
        rows = [[labels.get(item["method"], item["method"]), item["formula"],
                 number(item["wall_seconds"]), number(item["wall_hours"], 4)] for item in update["estimates"]]
        lines += markdown_table(["估计依据", "公式（秒）", "预计秒", "预计小时"], rows)
        lines += ["两条估计均加入一次实测startup及5×首次eval耗时；A的速度依据n=1个trial、30步、480样本，"
                  "第二条的训练速度依据首500步/8000次暴露。两种结果是不同依据的透明外推，不是置信区间。"
                  "checkpoint保存耗时尚未估计，C阶段闭环评测不在此B阶段时间内；原A训练循环估计保持不变。"
                  "数据与来源快照见[墙钟外推更新](wall_time_estimate_update.json)。", ""]
    lines += ["## B：正式训练与loss", ""]
    training = result["training"]
    if training:
        config, runtime = training["config"], training["runtime"]
        lines += ["超参：LoRA r=16/alpha=32/dropout=0.05、all-linear；lr=1e-4、cosine、warmup=3%；"
                  "bf16、gradient checkpointing、max_length=8392、truncation_strategy=raise；"
                  "Qwen3模板enable_thinking=False，assistant-only loss（含空think前缀与终止符）；seed=20261118。", "",
                  f"实际batch={config.get('batch_size')}、累积={config.get('gradient_accumulation_steps')}、"
                  f"group_by_length={config.get('group_by_length')}（继承A选定配置，n=1次正式训练）；"
                  f"版本：`{json.dumps(config.get('versions', result['environment'].get('packages', {})), ensure_ascii=False)}`。", ""]
        rows = [["optimizer步", runtime.get("optimizer_steps"), "计划n=2500"],
                ["epoch", runtime.get("epoch"), "计划n=2轮"], ["样本暴露", runtime.get("consumed_samples"), "计划n=40000"],
                ["完整序列token", runtime.get("consumed_tokens"), f"n={runtime.get('consumed_samples', 'NA')}次样本暴露"],
                ["训练循环秒（含eval/save）", number(runtime.get("train_seconds")), "n=1次正式训练"],
                ["进程工作墙钟秒（含启动/编码）", number(runtime.get("wall_seconds")), "n=1次正式训练"],
                ["显存峰值bytes", runtime.get("selection_peak_bytes"), f"n={runtime.get('resource_samples', 'NA')}次资源采样"],
                ["OOM", runtime.get("oom"), "n=1次正式训练"], ["非有限值", runtime.get("nonfinite"), "n=1次正式训练"],
                ["train loss记录", training["measured_train_loss_n"], "计划n=2500；每点n=16次样本暴露"],
                ["eval loss记录", training["measured_eval_loss_n"], "计划n=5；每点固定n=1000样本"]]
        lines += markdown_table(["项目", "实测", "n/口径"], rows)
        checkpoints = runtime.get("checkpoints", [])
        lines += [f"保存checkpoint n={len(checkpoints)}：" + "、".join(f"`{path}`" for path in checkpoints) + "。", ""]
    else:
        lines += ["正式训练实际耗时、loss、epoch与checkpoint：NA（待完成）。", ""]
    if result["loss_curve"]["rows"]:
        lines += ["[Loss数据](loss_curve.csv)；train逐optimizer步原始记录，每点n=16次样本暴露；"
                  "heldout每500步测一次，每点固定n=1000样本。曲线未平滑，eval loss不参与checkpoint排序。", ""]
        if "loss_curve_svg" in result["sources"]:
            lines += ["![训练与固定heldout loss](loss_curve.svg)", ""]
    lines += ["## C：每个checkpoint的合并主指标", ""]
    rows = []
    for step in STEPS:
        value = result["strategies"].get(f"checkpoint-{step}")
        rows.append([step, rate(value["combined"]["evidence_loss"]) if value else "NA",
                     mean(value["combined"]["final_recall"], True) if value else "NA"])
    lines += markdown_table(["checkpoint步", "合并证据丢失率(count/n)", "合并任务末召回"], rows)
    selected = result["selected_checkpoint"]
    lines += [f"选中：`{selected['strategy']}`（候选n=5；每候选n=1494条轨迹）。" if selected else "选中checkpoint：NA（五个完整结果未齐）。",
              "排序依次为六格合并的证据丢失率最小、合并任务末召回最大、checkpoint步数最早；eval loss不参与。", "",
              "## 中档预算2814：选中模型、原始模型与基线", ""]
    rows = []
    names = [("选中模型", selected["strategy"] if selected else None), ("原始模型", "base"), ("bm25", "bm25"), ("oracle", "oracle")]
    for label, name in names:
        for mode in MODES:
            value = result["strategies"].get(name)
            metric = value["configs"][f"{name}|{mode}|mid"]["metrics"] if value else None
            rows.append([label, mode, mean(metric["final_recall"], True) if metric else "NA",
                         rate(metric["evidence_loss"]) if metric else "NA",
                         *[rate(metric["invalid_outputs"][key]) if metric else "NA" for key in (*INVALID_REASONS, "any_invalid")]])
    lines += markdown_table(["策略", "模式", "任务末召回", "证据丢失率", "非法JSON", "候选编号非法", "append drop", "超预算", "任一非法"], rows)
    model = result["model_rollout"]
    lines += ["## 推理协议与成本口径", "",
              "每步使用模型自身保留状态；严格解析完整JSON并交给原apply_action。四类非法输出互斥，按"
              "非法JSON→候选编号非法→append drop非空→超预算的优先级计数，非法动作本步保持原内容。"
              "greedy、max_tokens=256；跨轨迹按步批处理；无输入截断。", ""]
    if model:
        lines += ["后端：" + model.get("lora_loading", "NA") + "。配置：`" + json.dumps(model.get("generation", {})) + "`。", ""]
        check = model.get("engine_tokenizer_check", model.get("template_check", {}))
        lines += [f"实际训练/推理token前缀核对：{check.get('exact_token_ids_match', 'NA')}，"
                  f"prefix={check.get('prompt_tokens', 'NA')} token，masked={check.get('masked_input_tokens', 'NA')} token，"
                  f"空think={check.get('empty_think_tokens', 'NA')} token（n=1条固定样本；各策略首条实际生成另核对）。"
                  f"样本：`{json.dumps(check.get('identity'), ensure_ascii=False)}`；见[模板证据](model_template_reference.json)。", ""]
    lines += ["成本单位为基础输入价格折算token；不开缓存与开缓存分别报告，不跨任务共享缓存。"
              "缓存读取/写入倍率0.05/1.25；相对成本分母为同任务/模式/预算的no_compress。"
              "no_compress是非约束参考线，表内请求预算占用可大于1；其执行预算单独记录。", "",
              "## 全策略×模式×预算：质量与占用", ""]
    rows = []
    for name, mode, tier, m in metric_cells(result):
        rows.append([name, mode, f"{tier}/{BUDGETS[tier]}", mean(m["step_recall"]["micro"], True),
                     mean(m["step_recall"]["task_macro"], True), mean(m["final_recall"], True), rate(m["evidence_loss"]),
                     rate(m["evidence_complete_questions"]), mean(m["occupancy"]), mean(m["requested_budget_occupancy"])])
    lines += markdown_table(["策略", "模式", "预算", "单步召回(按步)", "单步召回(按任务)", "任务末召回", "证据丢失", "证据完整题", "有效预算占用", "请求预算占用"], rows)
    lines += ["## 全策略：成本、非法输出与生成上限", ""]
    rows = []
    for name, mode, tier, m in metric_cells(result):
        rows.append([name, mode, tier, mean(m["cost"]["plain"]), mean(m["cost"]["cached"]),
                     f"{number(m['cost']['plain']['relative_to_no_compress'])}/{number(m['cost']['cached']['relative_to_no_compress'])} (各n={m['n_tasks']})",
                     *[rate(m["invalid_outputs"][key]) for key in (*INVALID_REASONS, "any_invalid")], rate(m["output_limit"])])
    lines += markdown_table(["策略", "模式", "预算", "不开缓存成本", "开缓存成本", "相对no_compress(无/有缓存)",
                             "非法JSON", "候选编号非法", "append drop", "超预算", "任一非法", "达到生成上限"], rows)
    lines += ["## 证据丢失分组（全部策略与预算）", "",
              "跳位与题位按证据出现次数计数，同一pid在不同位置可重复计入；每组任务丢失仍是该任务是否至少丢一段。"
              "丢失方式按每任务去重gold pid计数，仅将末态缺失证据分为never_kept和dropped。", ""]
    rows = []
    for name, mode, tier, m in metric_cells(result):
        for family in ("hop", "question_position", "mechanism"):
            for key, bucket in m["evidence_loss"]["by_" + family].items():
                rows.append([name, mode, tier, family, key, rate(bucket),
                             f"{bucket['lost_evidence_count']}/{bucket['evidence_n']} (证据n={bucket['evidence_n']})"])
    lines += markdown_table(["策略", "模式", "预算", "分组", "位置/方式", "任务丢失率", "丢失证据数/证据数"], rows)
    lines += ["## 资源、测试与外置文件", ""]
    resource, tests = result["resources"], result["tests"]
    if resource:
        lines += [f"根分区增量={resource.get('root_delta_bytes')} bytes，限额={resource.get('root_limit_bytes')} bytes，"
                  f"通过={resource.get('root_limit_passed')}（n=1次基线到终态比较）。"
                  f"GPU释放={resource.get('gpu_released')}、只用GPU1={resource.get('only_gpu1')}、"
                  f"剩余任务进程数={len(resource.get('task05_gpu_processes_remaining', []))}（n=1次终态检查）。", "",
                  "GPU1终态：`" + json.dumps(resource.get("gpu1_after"), ensure_ascii=False) + "`。", ""]
    else:
        lines += ["最终磁盘/GPU检查：pending。", ""]
    if tests:
        lines += [f"测试n={tests['n']}，失败={tests['failures']}、错误={tests['errors']}、跳过={tests['skipped']}"
                  f"（同一n），通过={tests['passed']}；原264项包含在全量测试中。", ""]
    inventory = result["external_artifact_manifest"]
    if inventory:
        lines += [f"外置清单含n={len(inventory.get('files', []))}个文件（checkpoint、训练/benchmark日志及逐步rollout），"
                  f"complete={inventory.get('complete')}；路径、bytes、SHA256和JSONL行数见"
                  "[外置产物清单](external_artifact_manifest.json)。全量哈希与穷举文件覆盖由checksums_task05.py check --slow核验。", ""]
    rows = [[name, f"`{item['path']}`", item.get("lines", "NA"), item["bytes"], f"`{item['sha256']}`"]
            for name, item in sample.get("files", {}).items()]
    lines += markdown_table(["子集文件", "路径", "行数n", "bytes(n=1文件)", "SHA256"], rows)
    lines += ["完整结构化指标保留于[sft_run_results.json](sft_run_results.json)，所有均值/比例保留原有n。"
              "报告仅列实测数值与checkpoint选择结果。", ""]
    return "\n".join(lines)


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plot_curve(directory):
    """One axes, raw measured training points and fixed heldout markers, no smoothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    path = directory / "loss_curve.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    train = [(int(row["step"]), float(row["train_loss"])) for row in rows if row["train_loss"]]
    heldout = [(int(row["step"]), float(row["eval_loss"])) for row in rows if row["eval_loss"]]
    require(all(math.isfinite(value) for _, value in train + heldout), "Nonfinite plotted loss")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "axes.linewidth": .7}):
        figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
        if train:
            axis.plot(*zip(*train), color="#2878A0", linewidth=.65, alpha=.8,
                      label=f"Train loss ({len(train)} points; n=16 exposures/point)")
        if heldout:
            axis.plot(*zip(*heldout), color="#D06B32", linewidth=1.2, marker="o", markersize=4,
                      label=f"Fixed heldout loss ({len(heldout)} points; n=1,000/point)")
        if not train and not heldout:
            axis.text(.5, .5, "Pending: no measured loss records", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.legend(frameon=False, loc="best", fontsize=8)
        axis.set(xlabel="Optimizer step", ylabel="Assistant loss", title="Task 05: measured training and heldout loss")
        axis.grid(axis="y", color="#E2E2E2", linewidth=.5)
        axis.set_axisbelow(True)
        figure.savefig(directory / "loss_curve.svg", format="svg", metadata={"Date": None})
        plt.close(figure)
    receipt = {"schema_version": 1, "csv_sha256": sha256(path), "svg_sha256": sha256(directory / "loss_curve.svg"),
               "train_points": len(train), "eval_points": len(heldout), "smoothing": False,
               "eval_loss_used_for_checkpoint_selection": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "matplotlib_version": matplotlib.__version__, "source_sha256": sha256(__file__)}
    write_json(directory / "loss_curve_plot.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("build", "plot"), default="build")
    args = parser.parse_args(argv)
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("run on 5880 formal project with CUDA_VISIBLE_DEVICES='' (CPU only)")
    directory = PROJECT / "eval/sft_run"
    if args.command == "plot":
        print(json.dumps(plot_curve(directory), ensure_ascii=False))
        return 0
    result = build_report_data(directory)
    (directory / "loss_curve.csv").write_text(loss_csv(result["loss_curve"]["rows"]))
    write_json(directory / "sft_run_results.json", result)
    (directory / "sft_run_report.md").write_text(render_report(result))
    print(json.dumps({"status": result["status"], "missing_inputs": result["missing_inputs"],
                      "failure_reasons": result["failure_reasons"], "selected_checkpoint": result["selected_checkpoint"]},
                     ensure_ascii=False))
    return 1 if result["status"] == "failed" else 2 if result["status"] == "pending" else 0


if __name__ == "__main__":
    raise SystemExit(main())
