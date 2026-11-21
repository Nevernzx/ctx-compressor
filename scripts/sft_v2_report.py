#!/usr/bin/env python3
"""Build Chinese Task 06 reports from saved data and actual pipeline measurements."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.sft_run_report import (
    INVALID_REASONS, MODES, STEPS, loss_csv, loss_rows, markdown_table, mean,
    rate, require, select_complete_checkpoints, sha256, training_summary, validate_strategy, write_json,
)

OUT = PROJECT / "eval/sft_run_v2"
OLD = PROJECT / "eval/sft_run"
RUN = Path("/root/userfolder/ctx-compressor/runs/sft_20k_v2_20261118")
MID = "mid"
CAUSES = ("invalid_output", "new_gold_not_added", "old_gold_dropped", "both", "none")


def read(path):
    return json.loads(path.read_text())


def build():
    data = read(OUT / "data_v2_stats.json")
    if not data["all_checks_passed"]:
        raise ValueError("v2 data consistency failed")
    refs = {"v1-2000": "checkpoint-2000", "base": "base", "bm25": "bm25",
            "oracle": "oracle", "no_compress": "no_compress", "latest": "latest"}
    ref_metrics = {}
    for label, name in refs.items():
        path = OLD / f"{name}_results.json"
        ref_metrics[label] = {"path": str(path), "sha256": sha256(path),
                              "metrics": validate_strategy(read(path), name)}
    current = {}
    for step in STEPS:
        name = f"checkpoint-{step}"
        path = OUT / f"{name}_results.json"
        if path.is_file():
            current[name] = validate_strategy(read(path), name)
    status_path = RUN / "continuation_status.json"
    status = read(status_path) if status_path.exists() else {"status": "prepared"}
    config = read(OUT / "train_run_config.json") if (OUT / "train_run_config.json").exists() else {}
    worker_file = RUN / "worker_result.json"
    runtime = read(worker_file) if worker_file.exists() else config.get("runtime", {})
    rows = loss_rows({"runtime": runtime})
    if rows:
        (OUT / "loss_curve.csv").write_text(loss_csv(rows))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "pipeline": status, "v2_checkpoints": current, "v1_references": ref_metrics,
              "selected_checkpoint": select_complete_checkpoints(current), "data": data,
              "train_seconds": runtime.get("train_seconds"),
              "train_points": sum("train_loss" in row for row in rows),
              "eval_points": sum("eval_loss" in row for row in rows),
              "diagnostics": {}, "config_diff": None}
    for version in ("v1", "v2"):
        path = OUT / f"diagnostics_{version}.json"
        if path.is_file():
            result["diagnostics"][version] = {
                "path": str(path), "sha256": sha256(path), "strategies": read(path)["strategies"]}
    if (OUT / "config_diff_vs_task05.json").exists():
        result["config_diff"] = read(OUT / "config_diff_vs_task05.json")
    resolved = RUN / "resolved_args_diff.json"
    if resolved.exists():
        result["actual_swift_arguments_comparison"] = read(resolved)
    if status["status"] == "evaluation_complete":
        complete_results(result, config, runtime)
    return result


def complete_results(result, config, runtime):
    sample = read(OUT / "sample_manifest.json")
    preflight = read(OUT / "heldout_v2_preflight.json")
    run = read(OUT / "model_rollout_run.json")
    original_run = read(OLD / "model_rollout_run.json")
    names = {f"checkpoint-{step}" for step in STEPS}
    require(training_summary({**config, "runtime": runtime}, sample)["complete"],
            "Completed pipeline has incomplete training")
    require(sample["ordered_id_lists_and_source_lines_identical"] and result["data"]["all_checks_passed"],
            "Training data or frozen subset verification failed")
    require(not result["config_diff"]["nonpath_hyperparameter_differences"]
            and not result["actual_swift_arguments_comparison"]["nonpath_differences"], "Configuration drift")
    require(sha256(OLD / "model_rollout_run.json") == preflight["task05_run_sha256"]
            and sha256(OUT / "heldout_v2_preflight.json") == run["preflight"]["sha256"], "Preflight source changed")
    require(run["status"] == "complete" and run["engine_shutdown"]["completed"]
            and run["cuda_visible_devices"] == "1" and runtime["initial_gpu"]["index"] == 1, "Unexpected run state")
    for key in ("engine_parameters", "generation", "lora_loading", "input_files"):
        require(run[key] == original_run[key], f"Task 05 evaluation protocol differs: {key}")
    require(set(run["strategies"]) == set(result["v2_checkpoints"]) == names, "Checkpoint grid differs")
    require(preflight["step0"]["n"] == 1494 and preflight["step0"]["user_utf8_byte_identical"]
            and preflight["step0"]["all_token_ids_identical"], "First-step prompts changed")
    for name, entry in run["strategies"].items():
        expected = result["v2_checkpoints"][name]["combined"]
        require(entry["status"] == "complete" and entry["matched_task05_step0_inputs"] == 1494
                and entry["n_steps"] == expected["n_steps"] == 13956
                and entry["n_trajectories"] == expected["n_tasks"] == 1494, f"Incomplete rollout: {name}")
    for version, required in (("v1", names | {"base", "bm25", "oracle", "latest", "no_compress"}), ("v2", names)):
        diagnostics = result["diagnostics"][version]["strategies"]
        require(set(diagnostics) == required, f"Missing diagnostic strategy: {version}")
        require(all(d["combined"]["n_trajectories"] == 1494 and d["combined"]["n_steps"] == 13956
                    for d in diagnostics.values()), f"Incomplete diagnostic denominators: {version}")
    winner = result["selected_checkpoint"]
    record = next(item for item in run["checkpoint_inputs"] if item["strategy"] == winner["strategy"])
    for file in record["files"]:
        require(file["sha256"] == sha256(Path(file["path"])), "Selected adapter changed")
    winner.update(path=record["path"], files=record["files"])
    result["training"] = {key: runtime[key] for key in (
        "status", "optimizer_steps", "consumed_samples", "consumed_tokens", "epoch", "oom", "nonfinite",
        "startup_seconds", "train_seconds", "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes")}
    result["sample"] = {"validation": sample["validation"], "ordered_ids_identical":
                        sample["ordered_id_lists_and_source_lines_identical"], "seed": sample["seed"]}
    result["heldout_protocol"] = {"preflight": preflight, "source": run["preflight"], "generation": run["generation"],
                                 "lora_loading": run["lora_loading"], "engine_parameters": run["engine_parameters"],
                                 "engine_shutdown": run["engine_shutdown"], "cuda_visible_devices": run["cuda_visible_devices"]}
    result["curve"] = {"source": str(RUN / "worker_result.json"), "source_sha256": sha256(RUN / "worker_result.json"),
                       "csv_sha256": sha256(OUT / "loss_curve.csv"), "train_n_per_point": 16, "eval_n_per_point": 1000}


def show_rate(metric):
    return rate(metric) if metric.get("n") else "不适用 (n=0)"


def render(result):
    status = result["pipeline"]
    lines = ["# 任务06　SFT v2 留出集报告", "",
             f"后台状态为 \x60{status['status']}\x60，已保存 {len(result['v2_checkpoints'])}/5 个 checkpoint 的闭环指标。",
             "", "## 训练数据", ""]
    for split, check in result["data"]["checks"].items():
        lines.append(f"- {split} 共 {check['sft_rows_v1']:,} 条 v1 和 {check['sft_rows_v2']:,} 条 v2。"
                     f"逐字节核对了全部 {check['append_rows_byte_identical']:,} 条 append、"
                     f"{check['reselect_step0_rows_byte_identical']:,} 条 reselect 第 0 步样本。")
        lines.append(f"  用旧版默认策略重放 {check['default_replay_tasks_byte_identical']:,} 个任务"
                     f"（n={check['default_replay_tasks_byte_identical']:,}），全都与旧轨迹逐字节相同。"
                     f"两版起始状态逐一一致（n={check['step0_states_identical']:,}）。")
    lines.extend(["", "重选的噪声段数由每步均匀抽取 1–3 段改为 0–2 段；种子仍取任务索引。"
                  "下表只统计 step ≥ 1，kept 指执行动作前的保留区，空集合也计入全为 gold。", ""])
    rows = []
    for split, versions in result["data"]["statistics"].items():
        for mode in MODES:
            for tier in ("tight", "mid", "loose", "all"):
                for version in ("v1", "v2"):
                    m = versions[version][f"{mode}|{tier}"]
                    rows.append([split, mode, tier, version, rate(m["target_drop_nonempty"]),
                                 mean(m["target_drop_count"]), mean(m["kept_fragments"]),
                                 rate(m["kept_entirely_gold"])])
    lines.extend(markdown_table(["划分", "模式", "预算", "版本", "目标 drop 非空", "平均 drop 数",
                                 "平均 kept 数", "kept 全为 gold"], rows))
    lines.extend(["## 训练与等待", ""])
    waited = status.get("wait_seconds")
    if waited is not None:
        lines.append(f"本次等卡 {waited:.1f} 秒（n=1 次后台运行）。")
    if result["train_seconds"] is not None:
        lines.append(f"训练循环 {result['train_seconds']:.1f} 秒（n=1 次训练）。")
    if result.get("training"):
        t, sample = result["training"], result["sample"]["validation"]
        lines.extend(["", f"训练抽样 n={sample['train']['rows']:,}，固定验证损失抽样 n={sample['eval_loss']['rows']:,}。"
                      f"两者与 v1 的样本 ID 及顺序全部一致；只追加样本也逐字节相同"
                      f"（训练 n={sample['train']['append']:,}，验证 n={sample['eval_loss']['append']:,}）。",
                      "", f"训练完成 {t['epoch']:g} 轮（n={t['consumed_samples']:,} 次样本暴露，"
                      f"n={t['optimizer_steps']:,} 次优化更新）。启动至训练 worker 退出 {t['wall_seconds']:.1f} 秒"
                      "（n=1 次）。训练记录的非有限值或显存不足事件为 0（n=1 次训练）。"])
    lines.extend(["", f"已读取 {result['train_points']} 个训练 loss 点（每点 n=16 次样本暴露）和 "
                  f"{result['eval_points']} 个验证点（每点 n=1000）。",
                  "", "逐字段配置对照见 [config_diff_vs_task05.json](config_diff_vs_task05.json)。", ""])
    if result["config_diff"]:
        diff = result["config_diff"]
        lines.append(f"配置和显式 Swift 参数分别逐字段比对"
                     f"（n={len(diff['compared_config_fields'])} 和 n={len(diff['explicit_swift_fields'])}）。"
                     f"下表列出两层的全部差异，路径之外的超参数差异为 {len(diff['nonpath_hyperparameter_differences'])}。")
        changes = []
        for section, key in (("训练配置", "differences"), ("显式 Swift 参数", "explicit_swift_argument_differences")):
            for field, pair in diff[key].items():
                values = [pair[version] for version in ("task05", "task06")]
                changes.append([section, field] + [f"`{value[0] if isinstance(value, list) else value}`" for value in values])
        lines.extend(["", *markdown_table(["层级", "字段", "v1", "v2"], changes)])
    if result.get("actual_swift_arguments_comparison"):
        args = result["actual_swift_arguments_comparison"]
        fields = "、".join(f"`{field}`" for field in args["changed_path_fields"])
        lines.append(f"Swift 实际解析参数中，路径变化 {len(args['changed_path_fields'])} 项"
                     f"（n={args['compared_fields']} 个字段），分别为 {fields}；"
                     f"其余差异 {len(args['nonpath_differences'])} 项（n={args['compared_fields']}）。")
    if result["train_points"]:
        lines.extend(["", "[Loss 数据](loss_curve.csv) · [曲线](loss_curve.svg)", ""])
    rows = []
    for name, strategy in result["v2_checkpoints"].items():
        m = strategy["combined"]
        rows.append([name, rate(m["evidence_loss"]), mean(m["final_recall"], percent=True)])
    if result.get("heldout_protocol"):
        proof = result["heldout_protocol"]["preflight"]
        data = next(v for p, v in proof["input_files"].items() if p.endswith("/heldout_tasks.jsonl"))
        lines.extend(["", "## 留出集和生成协议", "",
                      f"同一批训练留出题组合成 n={proof['n_tasks']} 个任务，保存的检索步数 n={proof['n_saved_steps']:,}。"
                      f"两种模式乘三档预算得到每个 checkpoint 的 n={proof['step0']['n']:,} 条轨迹，"
                      f"共 n={proof['expected_outputs_per_strategy']:,} 个动作。任务05的留出输入 SHA-256 为 `{data['sha256']}`。"
                      f"从 v1、v2 两份轨迹提取的任务、问题和检索序列有相同的 SHA-256 "
                      f"`{proof['shared_task_questions_retrieval_sha256']}`。",
                      "", f"首步用户文本逐字节、提示 token ID 逐项一致（每个 checkpoint n={proof['step0']['n']:,}）。"
                      "沿用任务05的 vLLM 原生 LoRA 加载、贪心解码和每步 256 个输出 token 上限。"
                      "依次判定 JSON、候选编号、只追加模式删除和超预算错误；遇到任一错误，本步保留区不变。"
                      "基础模型与规则策略均来自任务05保存的闭环结果。"
                      " [输入及首步核对](heldout_v2_preflight.json) · [推理记录](model_rollout_run.json)", ""])
    lines.extend(["", "## 合并证据丢失率与 checkpoint 选择", ""])
    if rows:
        lines.extend(markdown_table(["v2 checkpoint", "证据丢失率", "任务末召回"], rows))
    sel = result["selected_checkpoint"]
    if sel is None:
        lines.append("五个 checkpoint 的全部闭环结果到齐后按合并证据丢失率选优。")
    else:
        lines.append("将六个模式与预算分组合并，按轨迹末丢失至少一段证据的比例升序选点。若相同，"
                     "依次取轨迹末平均召回较高者、训练步数较小者。")
        lines.append(f"选中 \x60{RUN / 'checkpoints' / sel['strategy']}\x60（n=5 个候选，每个 n=1494 条轨迹）。")
    lines.extend(["", "## 中档预算 2814 的对照", ""])
    metrics = {label: obj["metrics"] for label, obj in result["v1_references"].items()
               if label in ("v1-2000", "base", "bm25", "oracle")}
    if sel is not None:
        metrics = {"v2-" + str(sel["checkpoint_step"]): result["v2_checkpoints"][sel["strategy"]], **metrics}
    lines.extend(["召回先在每条轨迹内计算，再对轨迹取平均。单步召回同时列出按步和先按任务取平均两种口径。"
                  "证据丢失指轨迹结束时至少缺一段证据。缓存成本为每条轨迹含读缓存、写缓存的平均输入成本，"
                  "单位为基础输入 token 等价量。", ""])
    rows, diagnostic_rows, error_rows = [], [], []
    for label, strategy in metrics.items():
        name = strategy["strategy"]
        version = "v2" if label.startswith("v2-") else "v1"
        diagnostics = result["diagnostics"].get(version, {}).get("strategies", {}).get(name)
        for mode in MODES:
            m = strategy["configs"][f"{name}|{mode}|{MID}"]["metrics"]
            rows.append([label, mode, mean(m["final_recall"], True), rate(m["evidence_loss"]),
                         mean(m["step_recall"]["micro"], True), mean(m["step_recall"]["task_macro"], True),
                         mean(m["cost"]["cached"]), *[rate(m["invalid_outputs"][key]) for key in INVALID_REASONS]])
            if diagnostics:
                cohort = diagnostics["configs"][f"{mode}|{MID}"]
                diagnostic_rows.append([label, mode, *[show_rate(cohort[k]) for k in
                                        ("new_gold_add", "kept_gold_survival", "all_gold_valid_drop")]])
                error_rows.append([label, mode, *[show_rate(cohort["first_error"][k]) for k in CAUSES]])
    lines.extend(markdown_table(["策略", "模式", "末召回", "证据丢失", "单步召回（步平均）", "单步召回（任务平均）",
                                 "缓存成本", "非法 JSON", "非法编号", "只追加非法 drop", "超预算"], rows))
    if diagnostic_rows:
        lines.extend(markdown_table(["策略", "模式", "新 gold 加入率", "旧 gold 存活率",
                                     "全 gold 且有效时仍删除"], diagnostic_rows))
        lines.extend(markdown_table(["策略", "模式", "首错非法", "首错未加新 gold", "首错误删旧 gold",
                                     "首错两者兼有", "无错误"], error_rows))
    lines.extend(["新证据加入率按本次返回且此前未保留的可用证据段计，旧证据存活率按检索前已确认、"
                  "已保留的证据段计。仍删除的分母为第 1 步起动作有效、且保留区全为已知证据的重选步；"
                  "只追加模式的后两项不适用。首错按轨迹首次出现的无效动作、漏加、误删或同时漏加误删分类。"
                  "同一步先判无效；无错误轨迹单列。", ""])
    if sel is not None and "v2" in result["diagnostics"]:
        lines.extend(["## 所有 SFT checkpoint 的诊断", "",
                      "下表合并全部模式和预算。旧证据存活、全证据时仍删除均只统计重选。"
                      "分模式、分预算和逐步位置的完整分子、分母见表后两份诊断文件。", ""])
        pooled = []
        for version in ("v1", "v2"):
            for step in STEPS:
                name = f"checkpoint-{step}"
                d = result["diagnostics"][version]["strategies"][name]["combined"]
                pooled.append([version, name, *[show_rate(d[k]) for k in
                              ("new_gold_add", "kept_gold_survival", "all_gold_valid_drop")],
                              *[show_rate(d["first_error"][k]) for k in CAUSES]])
        lines.extend(markdown_table(["版本", "checkpoint", "新证据加入", "旧证据存活", "全证据时仍删除",
                                     "首错非法", "首错漏加", "首错误删", "首错两者兼有", "无错"], pooled))
    lines.append("[v1 逐步诊断](diagnostics_v1.json) · [v2 逐步诊断](diagnostics_v2.json) · "
                 "[任务05基础模型](../sft_run/base_results.json) · [BM25](../sft_run/bm25_results.json) · "
                 "[oracle](../sft_run/oracle_results.json)")
    return "\n".join(lines) + "\n"


def plot():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "ctx-compressor-sft-v2"
    matplotlib.rcParams["svg.fonttype"] = "none"
    from matplotlib import pyplot as plt
    import csv
    with (OUT / "loss_curve.csv").open(newline="") as f:
        data = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    for field, marker, label in [("train_loss", None, "Train"), ("eval_loss", "o", "Fixed held-out")]:
        points = [(int(row["step"]), float(row[field])) for row in data if row[field]]
        if points:
            ax.plot(*zip(*points), marker=marker, lw=1, label=f"{label} (n={len(points)} points)")
    ax.set(xlabel="Optimizer step", ylabel="Assistant loss", title="Task 06 · SFT v2")
    ax.legend()
    fig.savefig(OUT / "loss_curve.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "plot"), default="build", nargs="?")
    args = parser.parse_args()
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("run on 5880 with CUDA_VISIBLE_DEVICES=''")
    if args.action == "plot":
        plot()
    else:
        result = build()
        write_json(OUT / "sft_v2_results.json", result)
        (OUT / "sft_v2_report.md").write_text(render(result))
        if result["selected_checkpoint"] is not None:
            write_json(OUT / "selected_checkpoint.json", result["selected_checkpoint"])
        print(json.dumps({"pipeline": result["pipeline"].get("status"), "v2_results": len(result["v2_checkpoints"])}))
