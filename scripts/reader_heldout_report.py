#!/usr/bin/env python3
"""Validate held-out reader predictions and report paired F1/EM with the published costs."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
sys.path.insert(0, str(PROJECT))
from scripts.reader_report import token_distribution
from src.qa_metrics import score_answer
from src.reader import parse_answer

R2 = "grpo_r2:checkpoint-125"
V3 = "sft_v3:checkpoint-3000"
H1 = "h1:tau_star"
STRATEGIES = (R2, V3, "sft_v2:checkpoint-2000", "grpo_r1:checkpoint-500", H1,
              "bm25", "latest", "oracle", "no_compress", "no_context")
MODES = ("append", "reselect")
BUDGETS = {"tight": 1876, "mid": 2814, "loose": 5628}
CELLS = tuple(f"{mode}|{tier}" for mode in MODES for tier in BUDGETS)
DISPLAY = ("GRPO r2 · 125", "SFT v3 · 3000", "SFT v2 · 2000", "GRPO r1 · 500", "H1",
           "BM25", "Latest", "Oracle", "No compression", "No context")
MATCH = ("strategy", "mode", "tier", "example_id", "question_hops", "evidence_complete",
         "trajectory_lost", "prompt_sha1", "prompt_tokens")
IDENTITY = ("config", "task", "question_index")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def input_rows(path, provenance):
    """Fingerprint the exact byte stream being decoded in this pass."""
    digest = sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            yield json.loads(line)
    provenance[str(path)] = digest.hexdigest()


@dataclass(frozen=True)
class Score:
    task: int
    index: int
    example: str
    hops: int
    complete: bool
    lost: bool | None
    tokens: int
    em: float
    f1: float


def index_requests(rows):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in IDENTITY)
        require(key not in out, f"duplicate preflight question: {key}")
        out[key] = (tuple(row[k] for k in MATCH), row["answer"], row["answer_aliases"])
    return out


def collect(rows, sources, strategies, *, only=None):
    pending = sources.copy() if sources is not None else None
    data = {strategy: defaultdict(dict) for strategy in strategies}
    for row in rows:
        strategy = row["strategy"]
        if only is not None and strategy != only:
            continue
        require(strategy in data, f"unknown strategy {strategy}")
        key = tuple(row[k] for k in IDENTITY)
        if pending is not None:
            require(key in pending, f"unexpected/duplicate prediction: {key}")
            metadata, answer, aliases = pending.pop(key)
            require(metadata == tuple(row[k] for k in MATCH), f"preflight metadata mismatch: {key}")
            require((row.get("answer", answer), row.get("answer_aliases", aliases)) == (answer, aliases),
                    f"reference mismatch: {key}")
        else:
            answer, aliases = row["answer"], row["answer_aliases"]
        parsed = parse_answer(row["raw_output"])
        require(parsed == row["pred"], f"raw output parse mismatch: {key}")
        metrics = score_answer(parsed, answer, aliases)
        for name, value in metrics.items():
            require(type(row[name]) in (float, int) and math.isclose(value, row[name], rel_tol=0, abs_tol=1e-12),
                    f"{name} differs from official score: {key}")
        mode, tier = row["mode"], row["tier"]
        if mode is None and tier is None:
            require(strategy in ("no_context", "no_compress"), f"missing cell: {key}")
            cell = "shared"
        else:
            require(mode in MODES and tier in BUDGETS, f"bad mode/tier: {key}")
            cell = f"{mode}|{tier}"
        expected_config = strategy if cell == "shared" else f"{strategy}|{cell}"
        require(row["config"] == expected_config, f"wrong configuration: {key}")
        t, i = row["task"], row["question_index"]
        require(type(t) is int and type(i) is int and 0 <= i < 4 and row["question_hops"] in (2, 3, 4),
                f"bad question key/hops: {key}")
        require(type(row["evidence_complete"]) is bool and
                (type(row["trajectory_lost"]) is bool or strategy == "no_context" and row["trajectory_lost"] is None),
                f"bad evidence flag: {key}")
        require(type(row["prompt_tokens"]) is int and 0 < row["prompt_tokens"] <= 40960 - 48
                and row["input_truncated"] is False and row["finish_reason"] in ("length", "stop")
                and 0 <= row["output_tokens"] == len(row["output_token_ids"]) <= 48, f"incomplete output: {key}")
        scores = data[strategy][cell]
        require((t, i) not in scores, f"duplicate question: {key}")
        scores[t, i] = Score(t, i, row["example_id"], row["question_hops"], row["evidence_complete"],
                             row["trajectory_lost"], row["prompt_tokens"], metrics["em"], metrics["f1"])
    require(pending is None or not pending, f"missing {len(pending or {})} predictions")
    return data


def audit_matrix(data, no_compress, n_tasks=249):
    identity = None
    for strategy, cells in data.items():
        shared = strategy == "no_context" or strategy == "no_compress" and no_compress["all_identical"]
        require(set(cells) == ({"shared"} if shared else set(CELLS)), f"incomplete cells for {strategy}")
        for cell, rows in cells.items():
            require(len(rows) == n_tasks * 4 and set(Counter(s.task for s in rows.values()).values()) == {4},
                    f"missing questions: {strategy}|{cell}")
            keys = {k: (s.example, s.hops) for k, s in rows.items()}
            if identity is None:
                identity = keys
            require(keys == identity, f"examples differ across policies/cells: {strategy}|{cell}")
            by_task = defaultdict(list)
            for s in rows.values():
                by_task[s.task].append(s)
            if strategy != "no_context":
                for task, scores in by_task.items():
                    require(len({s.lost for s in scores}) == 1, f"inconsistent trajectory loss: {task}")
                    require(scores[0].lost == any(not s.complete for s in scores),
                            f"final question gold vs trajectory loss disagreement: {strategy}|{cell}|{task}")


def average(scores):
    n = len(scores)
    return {"n": n, "em": math.fsum(s.em for s in scores) / n if n else None,
            "f1": math.fsum(s.f1 for s in scores) / n if n else None}


def summarize_group(scores):
    scores = list(scores)
    return {**average(scores), "n_tasks": len({s.task for s in scores}),
            "by_evidence_complete": {str(b).lower(): average([s for s in scores if s.complete == b])
                                     for b in (True, False)},
            "by_trajectory_lost": {str(b).lower(): average([s for s in scores if s.lost == b])
                                   for b in (True, False)},
            "by_hops": {str(h): average([s for s in scores if s.hops == h]) for h in (2, 3, 4)},
            "by_position": {str(i+1): average([s for s in scores if s.index == i]) for i in range(4)},
            "prompt_tokens": token_distribution([s.tokens for s in scores])}


def paired(left, right, data):
    keys = [(cell, key) for cell in CELLS for key in data[left][cell]]
    require(set(keys) == {(cell, k) for cell, rows in data[right].items() for k in rows}, "paired key mismatch")
    counts = Counter(left_higher=0, equal=0, right_higher=0)
    matrix = [[0, 0], [0, 0]]
    x, y = [], []
    for cell, key in keys:
        a, b = data[left][cell][key], data[right][cell][key]
        require((a.example, a.hops) == (b.example, b.hops), "paired examples disagree")
        x.append(a)
        y.append(b)
        counts["left_higher" if a.f1 > b.f1 else "right_higher" if a.f1 < b.f1 else "equal"] += 1
        matrix[1-int(a.em)][1-int(b.em)] += 1
    xs, ys = average(x), average(y)
    return {"left": left, "right": right, "n": len(keys), "left_scores": xs, "right_scores": ys,
            "f1_difference_pp": 100 * (xs["f1"] - ys["f1"]), "em_difference_pp": 100 * (xs["em"] - ys["em"]),
            "f1_counts": dict(counts), "em_matrix_correct_then_incorrect": matrix}


def group_cost(preflight, strategy, cell, n_tasks):
    if strategy == "no_context":
        return {"n": n_tasks, "cached_mean": 0., "recomputed_cached_mean": 0.}
    by_cell = preflight["costs"][strategy]
    cells = list(CELLS) if cell == "shared" else [cell]
    costs = [by_cell[c] for c in cells]
    for cost in costs:
        require(cost["n"] == n_tasks and math.isfinite(cost["cached_mean"]) and cost["cached_mean"] >= 0 and
                math.isclose(cost["cached_mean"], cost["recomputed_cached_mean"], abs_tol=1e-6, rel_tol=0),
                f"published vs recomputed cached cost mismatch: {strategy}|{cell}")
    if len(costs) == 1:
        return costs[0]
    return {"n": sum(c["n"] for c in costs), "cached_mean": math.fsum(c["cached_mean"] for c in costs) / len(costs),
            "by_cell": {cell: by_cell[cell] for cell in cells}}


def make_results(data, preflight, extra=None, n_tasks=249):
    nc = preflight.get("no_compress")
    audit_matrix(data, nc, n_tasks)
    names = [extra] if extra else STRATEGIES
    result = {"input_split": "train-held-out", "units": {"scores": "fraction", "deltas": "percentage points",
               "costs": "cached equivalent input tokens per trajectory"}, "strategies": {},
              "comparisons": [], "no_compress": nc, "n_tasks": n_tasks, "plots": {}}
    for name in names:
        cells = data[name]
        groups = {}
        for cell, rows in cells.items():
            g = summarize_group(rows.values())
            g["cost"] = group_cost(preflight, name, cell, n_tasks)
            groups[cell] = g
        result["strategies"][name] = {"pooled": summarize_group(s for rows in cells.values() for s in rows.values()),
                                      "cells": groups}
    result["n_predictions"] = sum(s["pooled"]["n"] for s in result["strategies"].values())
    result["comparisons"] = [paired(extra, R2, data)] if extra else [paired(R2, name, data) for name in (V3, H1)]
    return result


def render_markdown(result, extra=None):
    from scripts.reader_report import _table
    def pct(value):
        return "—" if value is None else f"{100*value:.3f}%"
    def fgroup(stats):
        return f"{pct(stats['f1'])}（n={stats['n']:,}）"
    def heading(text):
        return ["", "## " + text, ""]
    n = result["n_predictions"]
    nt = result["n_tasks"]
    lines = ["# MuSiQue 留出集阅读模型评测" if not extra else "# GRPO 第三轮所选模型的阅读评测", "",
             f"Qwen3-8B 根据压缩器在轨迹末步保留的原文，分别回答每个任务的四道题。"
             f"本次实际评分 {n:,} 条（n={nt:,} 个不同任务，{nt*4:,} 道不同题）。"
             "同一问题在不同模式与预算下重复出现。合并分数对各格等权，每个预测均按任务03协议取标准答案和别名中的最高 EM 与 F1。"]
    if not extra:
        nc = result["no_compress"]
        lines += ["", f"不压缩基线以 {nc['reference_cell']} 为参照，逐任务比较末步段号及其顺序。"
                  f"有差异的任务数为 {nc['differing_tasks']:,}/{nc['tasks']:,}，有差异的配置数为 {nc['differing_cells']}/6。"
                  + ("六种配置共用一组读者预测。" if nc["all_identical"] else
                     f"六种配置分别运行。逐格与参照不同的任务数为 "
                     + "、".join(f"{cell} {count}/{nt}" for cell, count in nc["by_cell"].items()) + "。")]
        smoke = result["smoke"]
        lines += ["", f"用同一引擎重跑任务03的一致性小样，原始输出不同的提示共 "
                  f"{smoke['raw_output_mismatches']}/{smoke['n']} 条。"]
    lines += heading("合并 F1 与 EM")
    series = result["strategies"]
    lines += [_table(["策略", "实测题次 n", "F1", "EM"], [
        [name, values["pooled"]["n"], pct(values["pooled"]["f1"]), pct(values["pooled"]["em"])]
        for name, values in series.items()])]
    lines += heading("预注册的逐题配对比较")
    for cmp in result["comparisons"]:
        a, b, n = cmp["left"], cmp["right"], cmp["n"]
        counts = cmp["f1_counts"]
        lines += [f"### {a} 与 {b}", "",
                  f"左列模型减右列模型，F1 差 {cmp['f1_difference_pp']:+.3f} 个百分点，"
                  f"EM 差 {cmp['em_difference_pp']:+.3f} 个百分点（各 n={n:,} 对）。"
                  f"逐题 F1 左高 {counts['left_higher']:,}、相等 {counts['equal']:,}、右高 "
                  f"{counts['right_higher']:,}（n={n:,} 对）。", "",
                  _table([f"EM 配对（n={n:,}）", f"{b} 正确", f"{b} 错误"], [
                      [f"{a} 正确", *cmp["em_matrix_correct_then_incorrect"][0]],
                      [f"{a} 错误", *cmp["em_matrix_correct_then_incorrect"][1]]]), ""]
    lines += heading("按模式与预算")
    lines += [_table(["策略","模式 / 档","题次 n","F1","EM","缓存成本均值","成本轨迹 n"], [
        [name, cell.replace("|", " / ") if cell != "shared" else "各档共用",
         group["n"], pct(group["f1"]), pct(group["em"]), f"{group['cost']['cached_mean']:,.2f}", group["cost"]["n"]]
        for name, values in series.items() for cell, group in values["cells"].items()]),
        "", "成本单位为每条压缩轨迹的等价输入 token。每格成本按已有策略结果计算，并与归档轨迹重算值核对。"
        "无上下文行的段落缓存成本取零。"]
    if not extra:
        lines += ["", "![预算与阅读 F1](budget_f1.svg)", "", "![轨迹缓存成本与阅读 F1](cost_f1.svg)"]
    specs = [
        ("证据齐全与轨迹丢失", [("该题证据齐全", "by_evidence_complete", "true"),
                            ("该题证据不齐", "by_evidence_complete", "false"),
                            ("轨迹丢证据", "by_trajectory_lost", "true"),
                            ("轨迹无丢失", "by_trajectory_lost", "false")]),
        ("题目跳数", [(f"{i} 跳", "by_hops", str(i)) for i in (2, 3, 4)]),
        ("题目在任务中的位置", [(f"第 {i} 题", "by_position", str(i)) for i in range(1, 5)])]
    for title, cols in specs:
        lines += heading(title)
        if title.startswith("证据"):
            lines += ["题级齐全指该题的全部支撑段都在最终上下文中。轨迹级丢失指最终上下文缺少该任务任一证据，"
                      "同一轨迹的四题一同分组。表中列出平均 F1 和实际题次，空组用破折号。", ""]
        lines += [_table(["策略"] + [h for h, key, val in cols], [
            [name] + [fgroup(values["pooled"][key][val]) for h, key, val in cols] for name, values in series.items()])]
    lines += heading("每题提示 token")
    lines += ["长度为套用任务03对话模板后的实际输入 token 数。分位数沿用任务03的线性插值。", "",
              _table(["策略", "题次 n", "均值", "P50", "P90", "最大值"], [
                  [name, dist["n"], f"{dist['mean']:,.2f}", f"{dist['p50']:,.1f}", f"{dist['p90']:,.1f}",
                   dist["max"]] for name, values in series.items() for dist in [values["pooled"]["prompt_tokens"]]])]
    if not extra:
        lines += ["", "[运行配置](run_config.json) · [逐格结果与输入指纹](reader_heldout_results.json)"]
    else:
        lines += ["", "[逐格结果与输入指纹](reader_f1_results.json)"]
    return "\n".join(lines) + "\n"


def svg_charts(result):
    """Render both figures from identical per-cell means, using CPU Matplotlib."""
    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "held-out-reader-v1"})
    colors = ("#005e85", "#c85522", "#7155a1", "#4b7b59", "#cc638a",
              "#82713a", "#98999a", "#282832", "#a28d72", "#5b6972")
    markers = ("o", "s", "D", "v", "^", "X", "P", "*", "h", "+")
    based = result["strategies"]
    ref_names = ("no_compress", "no_context")
    outputs = {}
    upper = min(100, max(10, 5 * math.ceil((max(g["f1"]*100 for s in based.values() for g in s["cells"].values()) + 3)/5)))
    max_cost = max(g["cost"]["cached_mean"] for s in based.values() for g in s["cells"].values())
    for axis in ("budget", "cost"):
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.7), sharey=True)
        fig.subplots_adjust(left=.087, right=.977, bottom=.313, top=.875, wspace=.14)
        handles = []
        for ax, mode in zip(axs, MODES):
            ax.set_title(mode.capitalize(), loc="left", weight="bold", pad=10)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#dedfe0", linewidth=.55)
            ax.set_ylim(0, upper)
            if axis == "budget":
                ax.set_xticks(list(BUDGETS.values()), [f"{n:,}" for n in BUDGETS.values()])
                ax.set_xlim(1500, 6000)
                ax.set_xlabel("Passage budget (tokens)")
            else:
                ax.set_xscale("symlog", linthresh=1500, linscale=.35, base=10)
                ticks = [v for v in (0,1500,3000,10000,30000,100000) if v <= max_cost*1.15]
                ax.set_xlim(-90, max_cost*1.15)
                ax.set_xticks(ticks)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
                ax.set_xlabel("Cached cost (equivalent input tokens / trajectory)")
            for j, (name, label) in enumerate(zip(STRATEGIES, DISPLAY)):
                s = based[name]
                color = colors[j]
                marker = markers[j]
                ref = name in ref_names
                if axis == "budget" and ref:
                    linestyle = "--" if name == "no_compress" else ":"
                    ax.axhline(s["pooled"]["f1"]*100, color=color, linestyle=linestyle, linewidth=1.3)
                    if mode == MODES[0]:
                        handles.append(Line2D([], [], label=label, color=color, linestyle=linestyle))
                    if "shared" in s["cells"]:
                        continue
                elif mode == MODES[0]:
                    handles.append(Line2D([], [], label=label, color=color, marker=marker,
                        linestyle="-" if axis == "budget" else "", markersize=6))
                cells = s["cells"]
                if "shared" in cells:
                    g = cells["shared"]
                    for tier in BUDGETS:
                        cell = f"{mode}|{tier}"
                        cost = g["cost"].get("by_cell", {}).get(cell, g["cost"])
                        ax.plot(cost["cached_mean"], g["f1"]*100, marker=marker, color=color,
                                markersize=6, linestyle="none")
                else:
                    ordered = [cells[f"{mode}|{tier}"] for tier in BUDGETS]
                    xs = list(BUDGETS.values()) if axis == "budget" else [g["cost"]["cached_mean"] for g in ordered]
                    ax.plot(xs, [g["f1"]*100 for g in ordered], color=color, marker=marker,
                            markersize=6 if name==R2 else 4.5, alpha=1 if name==R2 else .85,
                            linewidth=2 if name==R2 else 1.2, zorder=5 if name==R2 else 3,
                            linestyle="-" if axis=="budget" else "none")
        axs[0].set_ylabel("Answer F1 (%)")
        fig.text(.087, .96, "Reader F1 by budget" if axis=="budget" else "Reader F1 by cached cost",
                 fontsize=14, weight="bold", va="top")
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52,.102), ncol=5,
                   frameon=False, columnspacing=1.4, handletextpad=.5)
        caption = f"Each cell: n = {result['n_tasks']*4:,} question responses from {result['n_tasks']:,} tasks."
        fig.text(.087,.073,caption,color="#40474b",fontsize=8)
        foot = ("References show pooled means; shared predictions are counted once."
                if axis=="budget" else "Each marker represents one budget cell; identical coordinates overlap. Cost axis uses symmetric log scaling.")
        fig.text(.087,.039,foot,color="#40474b",fontsize=8)
        fig.canvas.draw()
        rects = [a.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) for a in axs]
        require(all(abs(getattr(rects[0], p)-getattr(rects[1], p))*72 < 1.5
                    for p in ("width", "height", "y0")), "misaligned plot areas")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", metadata={"Date":None})
        outputs[f"{axis}_f1.svg"] = buffer.getvalue()
        plt.close(fig)
    return outputs


def check_signed_prediction(path):
    manifest = PROJECT / "eval/reader_heldout/TASK12_SHA256SUMS"
    hashes = {name: digest for digest, name in
              (line.split("  ", 1) for line in manifest.read_text().splitlines())}
    digest = hashes.get(str(path))
    require(digest is not None, "task12 predictions absent from signed manifest")
    return digest


def publish(output_dir, payloads):
    for name in payloads:
        if (output_dir / name).exists():
            raise FileExistsError(output_dir / name)
    output_dir.mkdir(exist_ok=True, parents=True)
    for name, body in payloads.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(body)


def encoded(data):
    return (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--extra-strategy")
    parser.add_argument("--run-config", type=Path)
    args = parser.parse_args()
    extra = args.extra_strategy
    output = args.output_dir.resolve()
    target = PROJECT / ("eval/grpo_r3" if extra else "eval/reader_heldout")
    require(PROJECT == Path("/root/workplace/ctx-compressor") and output == target,
            f"expected formal output directory {target}")
    require(bool(args.baseline_predictions) == bool(extra), "extra comparison requires its signed task12 predictions")
    for path in (args.preflight, args.predictions, args.baseline_predictions):
        if path:
            require(path.resolve().is_relative_to(EXTERNAL / "reader_heldout") and path.is_file(), f"unexpected reader artifact: {path}")
    if extra:
        require(extra.startswith("grpo_r3:checkpoint-"), "expected a third-round checkpoint")
    preflight = args.preflight.read_bytes()
    hashes = {str(args.preflight): sha256(preflight).hexdigest()}
    verified = json.loads(preflight)
    require(verified["status"] == "passed" and verified["input_truncations"] == 0, "incomplete preflight")
    requests = args.preflight.with_name("requests.jsonl")
    sources = index_requests(input_rows(requests, hashes))
    require(len(sources) == verified["requests"]["n"] and hashes[str(requests)] == verified["requests"]["sha256"],
            "frozen preflight requests changed")
    data = collect(input_rows(args.predictions, hashes), sources, [extra] if extra else STRATEGIES)
    if extra:
        expected_digest = check_signed_prediction(args.baseline_predictions.resolve())
        data.update(collect(input_rows(args.baseline_predictions, hashes), None, [R2], only=R2))
        require(hashes[str(args.baseline_predictions)] == expected_digest, "task12 baseline prediction hash mismatch")
    result = make_results(data, verified, extra)
    if not extra:
        # Keep statistics and SVG generation on CPU. The project's plotting environment
        # has Matplotlib, while the frozen task03 reader environment does not.
        runfile = args.run_config or output / "run_config.json"
        raw = runfile.read_bytes()
        hashes[str(runfile)] = sha256(raw).hexdigest()
        result["smoke"] = json.loads(raw)["smoke"]
        smoke = result["smoke"]
        require(smoke["n"] == 40 and type(smoke["raw_output_mismatches"]) is int and
                0 <= smoke["raw_output_mismatches"] <= 40, "incomplete task03 smoke comparison")
    result["input_sha256"] = hashes
    stem = "reader_f1" if extra else "reader_heldout"
    payloads = {stem+"_results.json": encoded(result), stem+"_report.md": render_markdown(result, extra).encode()}
    if not extra:
        import base64
        env = {**os.environ, "CUDA_VISIBLE_DEVICES":"", "PYTHONDONTWRITEBYTECODE":"1",
               "MPLBACKEND":"Agg","MPLCONFIGDIR":str(EXTERNAL/"reader_heldout"/"plot_cache")}
        child = subprocess.run([str(EXTERNAL/"envs/train/bin/python"), "-B", str(Path(__file__).resolve()), "--plot-stdin"],
                               input=encoded(result), capture_output=True, check=False, env=env)
        if child.returncode:
            raise RuntimeError(child.stderr.decode(errors="replace"))
        payloads.update({name: base64.b64decode(value) for name, value in json.loads(child.stdout).items()})
    publish(output, payloads)
    print(json.dumps({"n": result["n_predictions"], "pairs": result["comparisons"]}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["--plot-stdin"]:
        import base64
        print(json.dumps({n: base64.b64encode(raw).decode() for n, raw in svg_charts(json.load(sys.stdin)).items()}))
    else:
        main()
