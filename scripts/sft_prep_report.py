#!/usr/bin/env python3
"""Build the Task 04 Chinese report from measured artifacts, CPU only.

Missing benchmark/resources remain pending; length overflow or failed acceptance
is a failure. Rerunning replaces only this report and its companion result JSON.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.sft_protocol import length_distribution

INPUTS = ("reference_fix_checks", "trajectory_stats", "sft_data_stats", "model_manifest",
          "environment_metadata", "bench_run_config", "bench_results", "final_resource_check")
LENGTHS = ("n_input_tokens", "n_target_tokens", "total_tokens", "loss_tokens")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scan_split(record, split):
    """Compute exact pooled split quantiles from saved metadata; verify input bytes."""
    histograms = {field: Counter() for field in LENGTHS}
    digest, count = hashlib.sha256(), 0
    with Path(record["path"]).open("rb") as stream:
        for line in stream:
            digest.update(line)
            meta = json.loads(line)["meta"]
            if meta["split"] != split:
                raise ValueError(f"Wrong split in {record['path']}")
            if meta["n_input_tokens"] + meta["n_target_tokens"] != meta["total_tokens"]:
                raise ValueError("Input + target token identity failed")
            for field in LENGTHS:
                if type(meta[field]) is not int or meta[field] <= 0:
                    raise ValueError(f"Invalid saved length: {field}")
                histograms[field][meta[field]] += 1
            count += 1
    if count != record["lines"] or digest.hexdigest() != record["sha256"]:
        raise ValueError(f"SFT line count or SHA256 mismatch: {split}")
    return {"samples": count,
            "lengths": {field: length_distribution(hist) for field, hist in histograms.items()},
            "source_sha256_verified": True}


def estimate_plans(train_samples, train_mean_tokens, benchmark):
    """Token-scaled exposures plus one measured startup; never select a plan."""
    tokens, seconds = benchmark.get("consumed_tokens"), benchmark.get("train_seconds")
    if not tokens or not seconds or not math.isfinite(seconds) or seconds <= 0:
        return []
    if benchmark.get("optimizer_steps") != 100 or benchmark.get("consumed_samples") != 1600:
        return []
    probe = benchmark.get("longest_probe", {})
    wall, probe_seconds = benchmark.get("wall_seconds"), probe.get("seconds")
    if wall is None or probe_seconds is None:
        return []
    startup = max(0.0, wall - seconds - probe_seconds)
    speed = tokens / seconds
    if not (math.isfinite(speed) and speed > 0 and math.isfinite(startup)):
        return []
    plans = [("全训练集", train_samples, epoch) for epoch in (1, 2, 3)]
    plans.extend([("抽样 20k", 20000, 2), ("抽样 40k", 40000, 2)])
    result = []
    for label, sample_n, epochs in plans:
        exposures = sample_n * epochs
        training = exposures * train_mean_tokens / speed
        predicted_wall = startup + training
        result.append({"plan": label, "samples": sample_n, "epochs": epochs,
                       "sample_exposures": exposures, "mean_tokens_per_sample": train_mean_tokens,
                       "mean_length_source_n": train_samples, "estimated_tokens": exposures * train_mean_tokens,
                       "measured_tokens_per_second": speed, "benchmark_n": 1600,
                       "one_time_startup_seconds": startup, "training_seconds": training,
                       "wall_seconds": predicted_wall, "wall_hours": predicted_wall / 3600,
                       "gpu1_occupancy_seconds": predicted_wall, "gpu1_occupancy_hours": predicted_wall / 3600})
    return result


def test_counts(path):
    if not path.exists():
        return None
    cases = list(ET.parse(path).getroot().iter("testcase"))
    counts = {"n": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        for tag, key in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
            counts[key] += int(case.find(tag) is not None)
    counts["passed"] = bool(cases) and not any(counts[key] for key in ("failures", "errors", "skipped"))
    return counts


def benchmark_summary(bench, sft):
    summary = dict(bench)
    if bench.get("train_seconds", 0) > 0:
        summary["tokens_per_second"] = bench.get("consumed_tokens", 0) / bench["train_seconds"]
        if bench.get("consumed_samples", 0):
            summary["seconds_per_sample"] = bench["train_seconds"] / bench["consumed_samples"]
    losses = [row["loss"] for row in bench.get("loss_curve", [])]
    if len(losses) >= 10 and all(math.isfinite(x) for x in losses):
        summary["first_10_loss_mean"] = statistics.mean(losses[:10])
        summary["last_10_loss_mean"] = statistics.mean(losses[-10:])
        summary["loss_decreased"] = summary["last_10_loss_mean"] < summary["first_10_loss_mean"]
    probe = bench.get("longest_probe", {})
    summary["acceptance_checks"] = {
        "status_ok": bench.get("status") == "ok",
        "exact_optimizer_steps": bench.get("optimizer_steps") == 100,
        "exact_samples": bench.get("consumed_samples") == 1600,
        "actual_tokens_match_selected": bench.get("consumed_tokens", 0) > 0 and
                                        bench.get("consumed_tokens") == bench.get("selected_total_tokens"),
        "loss_logs_100": len(losses) == 100,
        "loss_decreased": summary.get("loss_decreased") is True,
        "no_oom": bench.get("oom") is False,
        "longest_measured": probe.get("sequence_tokens") == sft.get("observed_max_tokens") and bool(probe),
        "longest_no_oom": probe.get("oom") is False,
        "longest_finite_gradients": probe.get("finite_gradients") is True,
        "longest_optimizer_states_resident": probe.get("optimizer_states_resident") is True,
        "longest_no_extra_optimizer_step": probe.get("optimizer_steps") == 0,
    }
    return summary


def build_report_data(directory):
    sources, values, missing = {}, {}, []
    for name in INPUTS:
        path = directory / (name + ".json")
        if path.exists():
            values[name] = json.loads(path.read_text())
            sources[name] = {"path": str(path), "sha256": sha256(path)}
        else:
            missing.append(path.name)
    test_path = directory / "tests.xml"
    tests = test_counts(test_path)
    if tests is None:
        missing.append(test_path.name)
    else:
        sources["tests"] = {"path": str(test_path), "sha256": sha256(test_path)}
    for suffix in ("svg", "csv"):
        path = directory / ("loss_curve." + suffix)
        if path.exists():
            sources["loss_curve_" + suffix] = {"path": str(path), "sha256": sha256(path)}
        else:
            missing.append(path.name)
    failures = []
    ref, traj, sft = (values.get(name, {}) for name in
                      ("reference_fix_checks", "trajectory_stats", "sft_data_stats"))
    model, resources = values.get("model_manifest", {}), values.get("final_resource_check", {})
    if ref and not ref.get("passed"):
        failures.append("引用修复或旧产物核验未通过")
    if tests is not None and not tests["passed"]:
        failures.append("完整测试存在失败、错误或跳过")
    if model and model.get("status") != "verified":
        failures.append("模型逐文件校验未全部完成")
    splits = {}
    if sft:
        for split in ("train", "heldout"):
            splits[split] = scan_split(sft["files"][split], split)
            splits[split]["tasks"] = sft["splits"][split]["tasks"]
            splits[split]["mode_tier"] = sft["splits"][split]["mode_tier"]
        if not sft.get("benchmark_allowed") or sft["observed_max_tokens"] > 16384 or sft["max_length"] > 16384:
            failures.append("完整样本长度超过 16384；停止基准，不截断、不丢弃样本")
        if sft["validation"]["truncated_samples"] or sft["validation"]["dropped_samples"]:
            failures.append("SFT 存在截断或丢弃的状态")
    bench = values.get("bench_results", {})
    summary = benchmark_summary(bench, sft) if bench else {}
    if summary:
        failures.extend("基准验收未通过：" + key for key, ok in summary["acceptance_checks"].items() if not ok)
    if resources:
        for key in ("root_limit_passed", "gpu_released", "only_gpu1"):
            if resources.get(key) is not True:
                failures.append("资源验收未通过：" + key)
        if resources.get("task04_gpu_processes_remaining"):
            failures.append("仍有任务04 GPU进程")
    plans = []
    if sft and sft.get("benchmark_allowed") and summary and all(summary["acceptance_checks"].values()):
        train = splits["train"]
        plans = estimate_plans(train["samples"], train["lengths"]["total_tokens"]["mean"], bench)
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fail" if failures else "pending" if missing else "ok",
        "missing_inputs": missing, "failure_reasons": failures, "sources": sources,
        "reference_fix": ref, "trajectory": traj, "sft": sft,
        "split_summary": splits, "environment": values.get("environment_metadata", {}),
        "model": {key: model.get(key) for key in ("model_id", "model_root", "status",
                                                   "verified_file_count", "total_expected_bytes", "tokenizer_comparison")},
        "bench_run_config": values.get("bench_run_config", {}), "benchmark": summary,
        "resources": resources, "tests": tests, "time_estimates": plans,
        "estimate_assumptions": {
            "formula": "sample_exposures * full_train_mean_sequence_tokens / measured_tokens_per_second + startup_seconds",
            "startup": "max(0, benchmark.wall_seconds - benchmark.train_seconds - longest_probe.seconds)",
            "tokens": "actual full sequence tokens, including masked input; excludes padding",
            "sampled_length": "20k and 40k estimates use full train mean length; subsets are not selected",
            "gpu_occupancy": "conservative wall-time upper bound while single GPU 1 is exclusively reserved",
            "excluded": ["formal checkpoint saving", "checkpoint evaluation", "new data preparation", "model downloads"],
            "initial_condition": "training data and local model are already ready", "plan_selected": None},
    }


def number(value, digits=2):
    if value is None:
        return "NA"
    return str(value) if type(value) is int else f"{value:.{digits}f}"


def fraction(value):
    if not value or value.get("rate") is None:
        return "NA"
    count = value.get("count", value.get("hits"))
    return f"{100*value['rate']:.3f}% ({count}/{value['n']})"


def render_report(result):
    lines = ["# 任务04：SFT 数据、环境与吞吐准备报告", "",
             f"状态：**{result['status']}**。所有 token 长度均来自实际 ms-swift 训练编码；未正式训练、未选择训练样本量方案。", ""]
    if result["missing_inputs"]:
        lines += ["尚待完成：" + "、".join(result["missing_inputs"]) + "；相应结果为 NA。", ""]
    if result["failure_reasons"]:
        lines += ["停止或未通过原因：" + "；".join(result["failure_reasons"]) + "。", ""]
    ref, traj, sft = result["reference_fix"], result["trajectory"], result["sft"]
    lines += ["## 引用修复与数据划分", ""]
    if ref:
        train, dev, saved = ref["train"], ref["all_dev"], ref["task02_saved_queries"]
        lines += [f"保留超出已完成跳数的原始 `#k`：恢复 {train['newly_accepted']} 题（n={train['questions']} train题），"
                  f"原始出现 {train['preserved_reference_occurrences']} 次；受影响 ID：`{'`、`'.join(train['affected_example_ids'])}`。",
                  f"dev 查询差异 {dev['mismatch_count']}（n={dev['questions']} 题/{dev['hops']} hops）；"
                  f"保存轨迹查询差异 {saved['mismatch_count']}（n={saved['tasks']} 任务/{saved['queries']} queries）。", ""]
    if traj:
        lines += [f"语料由全部 {traj['corpus']['built_from_all_train_questions']} 题构建，去重后 {traj['corpus']['paragraphs']} 段；"
                  "训练与留出共用该 train 语料库。按题留出 5%，向下取整；两部分独立拼四题任务。", "",
                  "| 分区 | 题池 n | 使用题数 | 丢弃余数 | 任务 n | 状态 n | 自然命中（命中/检索步数） |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for split in ("train", "heldout"):
            p, st = traj["partitions"][split], traj["statistics"][split]
            lines.append(f"| {split} | {p['pool_questions']} | {p['used_questions']} | {p['discarded_questions']} | "
                         f"{st['tasks']} | {st['samples']} | {fraction(st['natural_hit'])} |")
        lines += ["", "丢弃题 ID：" + "; ".join(f"{part}: {', '.join(p['discarded_ids']) or '无'}" for part, p in traj["partitions"].items()) + "。", "",
                  "预算 tight/mid/loose=1876/2814/5628；题划分与拼任务 seed=20261118。"
                  "每部分任务从 0 编号，检索/noise seed=task；tier 由 Random(20261118+task) 抽取。"
                  "同任务两模式共用检索，reselect 使用 oracle+noise 状态、append 使用纯 oracle；全部 SFT 目标为纯 oracle。", ""]
    lines += ["## 长度与 SFT 目标统计", "",
              "下表长度为完整训练序列 token（屏蔽输入+监督目标），每行 n 为状态/样本数。", "",
              "| 分区 | n | 输入均值 | 目标均值 | 总长均值 | P50 | P90 | P99 | max |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summaries = {**result["split_summary"]}
    if sft:
        summaries["all"] = sft["overall"]
    for split, group in summaries.items():
        ls = group["lengths"]; d = ls["total_tokens"]
        lines.append(f"| {split} | {group['samples']} | {number(ls['n_input_tokens']['mean'])} | "
                     f"{number(ls['n_target_tokens']['mean'])} | " + " | ".join(number(d[key]) for key in ("mean", "p50", "p90", "p99", "max")) + " |")
    lines += ["", "| 分区 | mode | tier | n | P50 | P90 | P99 | max | add为空 | drop非空 | 预算约束 |",
              "|---|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    grouped = {split: group["mode_tier"] for split, group in result["split_summary"].items()}
    if sft:
        grouped["all"] = sft["mode_tier"]
    for split, groups in grouped.items():
        for mode in ("reselect", "append"):
            for tier in ("tight", "mid", "loose"):
                g = groups[f"{mode}|{tier}"]; d = g["lengths"]["total_tokens"]
                lines.append(f"| {split} | {mode} | {tier} | {g['samples']} | " +
                             " | ".join(number(d[key]) for key in ("p50", "p90", "p99", "max")) + " | " +
                             " | ".join(fraction(g[key]) for key in ("target_add_empty", "target_drop_nonempty", "budget_binding")) + " |")
    if sft:
        v = sft["validation"]
        lines += ["", f"max_length={sft['max_length']}，观测最大={sft['observed_max_tokens']}（n={sft['overall']['samples']}）；"
                  f"截断 {v['truncated_samples']}、丢弃 {v['dropped_samples']}（同一 n）。"
                  "预算约束比例指纯 oracle 动作不能保留全部当前可得证据的步数比例。", "",
                  "模板为 Qwen3、enable_thinking=False；仅 assistant 内容计 loss。"
                  "assistant 内容前有空 `<think>\\n\\n</think>\\n\\n`，4 token 计 loss；目标 JSON 与末尾 2 token 终止符计 loss。"
                  "用户内容和 assistant 角色头全部屏蔽。推理 prefix 与训练 token 前缀一致；"
                  f"已检查 {v['template_prefix_checks']} 条（n={v['template_prefix_checks']}），见 [完整模板与 labels 核验](template_check.md)。", ""]
    lines += ["## 环境与模型", ""]
    env, model = result["environment"], result["model"]
    if env:
        lines += ["环境：`" + env["prefix"] + "`；包版本：`" + json.dumps(env["packages"], ensure_ascii=False) + "`；attention=sdpa。", ""]
    if model.get("status"):
        tok = model.get("tokenizer_comparison") or {}
        lines += [f"模型 {model['model_id']}：状态 {model['status']}，逐文件校验 {model['verified_file_count']} 个"
                  f"（n={model['verified_file_count']}），共 {model['total_expected_bytes']} bytes。"
                  f"tokenizer.json 与任务01{'逐字节一致' if tok.get('identical') else '未确认一致'}；ntok影响：{tok.get('ntok_impact', 'NA')}。", ""]
    lines += ["## 吞吐与最长样本实测", ""]
    config, b = result["bench_run_config"], result["benchmark"]
    if config:
        lines += [f"LoRA r={config['lora_rank']}、alpha={config['lora_alpha']}、dropout={config['lora_dropout']}、"
                  f"target={config['target_modules']}；bf16、gradient checkpointing、batch={config['per_device_train_batch_size']}、"
                  f"累积={config['gradient_accumulation_steps']}、lr={config['learning_rate']}、cosine、warmup={config['warmup_ratio']}；"
                  f"packing={config['packing']}，只用物理 GPU 1。抽样 {config['samples']} 条（每模式 n={config['per_mode']}）。", ""]
    if b:
        n = b.get("consumed_samples", 0)
        lines += [f"状态 {b.get('status')}；optimizer steps={b.get('optimizer_steps')}（计划 n=100），消费样本 {n}（计划 n=1600）。"
                  f"实际完整序列 token={b.get('consumed_tokens')}（n={n} 样本）；训练 {number(b.get('train_seconds'))} s，"
                  f"吞吐 {number(b.get('tokens_per_second'))} token/s，{number(b.get('seconds_per_sample'), 4)} s/样本（同一 n）。",
                  f"loss 首10步均值={number(b.get('first_10_loss_mean'), 6)}，末10步均值={number(b.get('last_10_loss_mean'), 6)}"
                  f"（各 n=10），下降={b.get('loss_decreased')}；OOM={b.get('oom')}（n=1 次基准）。", "",
                  f"训练 torch allocated/reserved 峰值={number(b.get('peak_allocated_bytes'))}/{number(b.get('peak_reserved_bytes'))} bytes（n=1 次基准）；"
                  f"全卡监控 memory.used 峰值={number(b.get('monitor_peak_memory_used_mib'))} MiB（n={b.get('monitor_samples', 0)} 个监控点，含最长样本探测）。", ""]
        probe = b.get("longest_probe", {})
        if probe:
            lines += [f"最长样本 {probe.get('sequence_tokens')} token（n=1）；在100个optimizer step后、优化器状态驻留={probe.get('optimizer_states_resident')} 时"
                      f"做一次 forward/backward；额外 optimizer step={probe.get('optimizer_steps')}，耗时={number(probe.get('seconds'))} s，"
                      f"OOM={probe.get('oom')}，梯度有限={probe.get('finite_gradients')}。"
                      f"torch allocated/reserved 峰值={probe.get('peak_allocated_bytes')}/{probe.get('peak_reserved_bytes')} bytes（n=1 次探测）。", ""]
        else:
            lines += ["最长样本探测：NA；未完成。", ""]
    else:
        lines += ["吞吐、loss、峰值显存及最长样本探测：NA（尚无完成的基准结果）。", ""]
    loss_n = len(b.get("loss_curve", []))
    if "loss_curve_svg" in result["sources"]:
        lines += [f"原始 optimizer-step loss 曲线（n={loss_n} 个日志点）：", "",
                  "![基准原始 loss 曲线](loss_curve.svg)", ""]
    else:
        lines += ["loss 曲线图：pending，等待 `loss_curve.svg`。", ""]
    if "loss_curve_csv" in result["sources"]:
        lines += [f"[下载原始 loss CSV](loss_curve.csv)（n={loss_n} 个日志点）。", ""]
    else:
        lines += ["loss CSV：pending，等待 `loss_curve.csv`。", ""]
    lines += ["## 训练样本量估时（仅列方案）", "",
              "以全训练集平均完整序列长度 × 样本曝光次数 ÷ 实测 token/s 估算训练时间；"
              "每个方案加一次启动开销 max(0, wall_seconds−train_seconds−longest_probe.seconds)。"
              "20k/40k 子集尚未抽取，长度使用全训练均值。单卡独占预留时将 GPU 1 占用时长保守取为墙钟时长。"
              "假设数据和模型已就绪、无评估、无保存；正式 checkpoint 保存、checkpoint 评估、新数据准备和下载开销均未包含。", "",
              "| 方案 | 样本 n | epoch | 曝光 n | 训练均长来源 n | 一次启动 s | 预计墙钟 h | GPU 1 占用 h |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for plan in result["time_estimates"]:
        lines.append(f"| {plan['plan']} | {plan['samples']} | {plan['epochs']} | {plan['sample_exposures']} | "
                     f"{plan['mean_length_source_n']} | {number(plan['one_time_startup_seconds'])} | "
                     f"{number(plan['wall_hours'], 3)} | {number(plan['gpu1_occupancy_hours'], 3)} |")
    if not result["time_estimates"]:
        lines.append("| 全量×1/2/3 epoch、20k×2、40k×2 | NA | NA | NA | NA | NA | NA | NA |")
    lines += ["", "## 测试、资源与产物", ""]
    tests = result["tests"]
    lines += [f"完整测试：{json.dumps(tests, ensure_ascii=False) if tests is not None else 'NA，等待 tests.xml'}。", ""]
    resource = result["resources"]
    if resource:
        lines += [f"根目录增量={resource.get('root_delta_bytes')} bytes，限额={resource.get('root_limit_bytes')} bytes，"
                  f"通过={resource.get('root_limit_passed')}（n=1 次基线到终态比较）；"
                  f"GPU已释放={resource.get('gpu_released')}，仅GPU1={resource.get('only_gpu1')}，"
                  f"剩余任务GPU进程数={len(resource.get('task04_gpu_processes_remaining', []))}（n=1 次终态检查）。",
                  "GPU1终态：`" + json.dumps(resource.get("gpu1_after"), ensure_ascii=False) + "`。", ""]
    else:
        lines += ["最终磁盘增量/GPU释放检查：pending。", ""]
    lines += ["数据和模型均保存在 userfolder；正式目录保存代码、小结果及拆分ID。文件行数、字节与 SHA256：", "",
              "| 类型 | 路径 | 行数 n | bytes | SHA256 |", "|---|---|---:|---:|---|"]
    for kind, files in (("trajectory", traj.get("files", {})), ("sft", sft.get("files", {}))):
        for split in ("train", "heldout"):
            if split in files:
                item = files[split]
                lines.append(f"| {kind}/{split} | `{item['path']}` | {item['lines']} | {item['bytes']} | `{item['sha256']}` |")
    lines += ["", "报告证据文件：" + "、".join(f"[{name}]({Path(item['path']).name})" for name, item in result["sources"].items()) + "。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "eval/sft_prep")
    args = parser.parse_args()
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("run on 5880 with CUDA_VISIBLE_DEVICES='' in the existing ctxc environment")
    if args.output_dir.resolve() != PROJECT / "eval/sft_prep":
        parser.error("reports must use the dedicated eval/sft_prep directory")
    result = build_report_data(args.output_dir)
    (args.output_dir / "sft_prep_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "sft_prep_report.md").write_text(render_report(result))
    print(json.dumps({"status": result["status"], "missing_inputs": result["missing_inputs"],
                      "failure_reasons": result["failure_reasons"], "estimates": result["time_estimates"]}, ensure_ascii=False))
    return 1 if result["status"] == "fail" else 2 if result["status"] == "pending" else 0


if __name__ == "__main__":
    raise SystemExit(main())
