#!/usr/bin/env python3
"""Validate the formal training CSV against its logs and plot its four GRPO metrics."""
import argparse
from collections import deque
import csv
import io
import json
import os
from pathlib import Path
import sys
from xml.etree import ElementTree

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import check_formal_grpo_training as training

PANELS = (
    ("reward", "Mean reward", "#2166AC"),
    ("reward_std", "Within-group reward SD", "#128276"),
    ("frac_reward_zero_std", "Groups with zero reward SD", "#9B5B13"),
    ("kl", "KL divergence", "#74549A"),
)


def validated_curve(directory):
    receipt = json.loads((directory / "training_checks.json").read_bytes())
    if receipt["status"] != "ok" or receipt["optimizer_steps"] != 500:
        raise ValueError("Incomplete formal training receipt")
    train = training.ROOT / "train"
    rows = training.curve(train, receipt["optimizer_steps"])
    csv_path = directory / "training_curve.csv"
    encoded = io.StringIO(newline="")
    writer = csv.DictWriter(encoded, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    if encoded.getvalue().encode() != csv_path.read_bytes() or training.sha256(csv_path) != receipt["curve_sha256"]:
        raise ValueError("The delivered CSV differs from the optimizer and Swift logs")

    data = Path(receipt["resolved_config"]["dataset"][0])
    n, states = training.dataset_states(data)
    if n != receipt["n_prompts"] or training.sha256(data) != receipt["data_sha256"]:
        raise ValueError("Training dataset or prompt count changed")
    log_path = train / "completions.jsonl"
    rewards = training.reward_recheck(log_path, states, len(rows))
    if json.loads(json.dumps(rewards)) != receipt["reward_recheck"]:
        raise ValueError("Reward recheck disagrees with the formal training receipt")
    with log_path.open("rb") as stream:
        last = deque(stream, maxlen=2)
    repeats = rewards["extra_logged_completions"]
    if len(last) != 2 or last[0] != last[1] or repeats != len(json.loads(last[1])["completion"]):
        raise ValueError("Extra terminal reward entries are not an exact repeat of the final logged group")
    sources = (train / "logging.jsonl", train / "optimizer_steps.jsonl", log_path, csv_path)
    return rows, {
        "status": "passed",
        "csv_byte_identical_to_reconstructed_log_table": True,
        "verified_steps": len(rows),
        "verified_columns": list(rows[0]),
        "plotted_columns": [key for key, _, _ in PANELS],
        "n_prompts": n,
        "rewards_checked": rewards["n"],
        "reward_mismatches": rewards["mismatches"],
        "unmatched_prompts": rewards["unmatched_prompts"],
        "training_completions": rewards["n_planned_completions"],
        "terminal_repeated_completions": repeats,
        "terminal_table_byte_identical": True,
        "source_sha256": {str(path): training.sha256(path) for path in sources},
    }


def render(rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    steps = [r["step"] for r in rows]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "svg.hashsalt": "ctx-compressor-grpo-formal", "path.simplify": False}):
        fig, axes = plt.subplots(2, 2, sharex=True, figsize=(11.0, 6.2), layout="constrained")
        fig.suptitle(f"GRPO training metrics  (n = {len(rows):,} optimizer steps)",
                     fontsize=15, fontweight="semibold")
        for ax, (field, label, color) in zip(axes.flat, PANELS):
            values = [r[field] for r in rows]
            line, = ax.plot(steps, values, color=color, linewidth=0.85)
            line.set_gid("trace-" + field)
            ax.set_title(label, loc="left", fontweight="semibold", fontsize=11)
            ax.set_xlim(steps[0], steps[-1])
            ax.set_xticks([1, 100, 200, 300, 400, 500])
            if field == "frac_reward_zero_std":
                ax.set_ylim(-0.05, 1.05)
                ax.set_yticks([0, .25, .5, .75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#8D979E")
            ax.tick_params(axis="both", length=3, colors="#364248", labelsize=9)
            ax.grid(axis="y", linewidth=0.55, color="#D4DCE0")
            ax.set_axisbelow(True)
        fig.supxlabel("Optimizer step", fontsize=11)
        output = io.BytesIO()
        fig.savefig(output, format="svg", metadata={"Date": None, "Title": "Formal GRPO training metrics"})
        plt.close(fig)

    data = output.getvalue()
    root = ElementTree.fromstring(data)
    ns = {"s": "http://www.w3.org/2000/svg"}
    for key, _, _ in PANELS:
        path = root.find(f".//s:g[@id='trace-{key}']/s:path", ns)
        if path is None or path.attrib["d"].split().count("L") != len(rows) - 1:
            raise ValueError(f"Incomplete SVG trace for {key}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=training.RESULTS)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Run with CUDA_VISIBLE_DEVICES=''")
    svg = args.output_dir / "training_curve.svg"
    proof = args.output_dir / "training_curve_validation.json"
    if svg.exists() or proof.exists():
        raise FileExistsError("Choose a new destination to regenerate the curve")
    rows, checked = validated_curve(training.RESULTS)
    content = render(rows)
    with svg.open("xb") as target:
        target.write(content)
    checked["svg_sha256"] = training.sha256(svg)
    with proof.open("x") as target:
        json.dump(checked, target, indent=2)
        target.write("\n")
    print(json.dumps({"steps": len(rows), "reward_checks": checked["rewards_checked"],
                      "mismatches": checked["reward_mismatches"], "columns": len(rows[0])}))


if __name__ == "__main__":
    main()
