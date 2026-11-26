"""Train and audit reward v3 through the same Swift entry point as Task 11."""
import argparse
import csv
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

from scripts import check_grpo_r2_training as audit
from scripts import grpo_r4_common as c
from scripts import run_grpo_r2_training as original
from scripts.grpo_r4_swift_plugin import COMPONENTS, ORM_NAME, PLUGIN_PATH, RewardComponents, take_components
from scripts.run_formal_grpo_pipeline import gpu
from src.grpo_reward_v3 import MU, NONGOLD_PRICE, score_completion_v3

SMOKE_TRAIN = c.SMOKE_DATA.parent / "train"
SMOKE_RECEIPT = c.SMOKE_DATA.parent / "smoke_checks.json"
TRAIN_RECEIPT = c.SMALL / "training_checks.json"
CURVE = c.SMALL / "training_curve.csv"
TASK11_SETTINGS = c.R2 / "training_checks.json"


def frozen_args(data, out, n, smoke):
    old = original.signed_json("launch_args.json", original.CLI_SHA)["argv"]
    args = original.swift_argv(old, c.BASE, data, out, n, lr=1.5e-5, scale_rewards="none",
                               seed=c.SEED, smoke=smoke)
    args[0] = "scripts/grpo_r4_training.py"
    args[args.index("--external_plugins") + 1] = PLUGIN_PATH
    args[args.index("--reward_funcs") + 1] = ORM_NAME
    return args


def verify_config(saved, _baseline, _reference, data, n, *, smoke_out=None):
    c.previous_file(TASK11_SETTINGS)
    expected = c.read(TASK11_SETTINGS)["resolved_config"].copy()
    out = SMOKE_TRAIN if smoke_out else c.TRAIN
    c.require(smoke_out is None or smoke_out == SMOKE_TRAIN, "Unexpected smoke output")
    expected.update(dataset=[str(data)], output_dir=str(out), max_steps=n // 4,
                    external_plugins=[PLUGIN_PATH], reward_funcs=[ORM_NAME])
    differences = {k: (value, saved.get(k)) for k, value in expected.items()
                   if k not in saved or saved[k] != value}
    c.require(not differences, f"Resolved Task 11 settings differ: {differences}")
    c.require(expected["model"] == str(c.BASE) and expected["seed"] == expected["data_seed"] == c.SEED,
              "Wrong training base or generation seed")
    return frozen_args(data, out, n, smoke_out is not None), expected


def reward_recheck(log, users, steps):
    old_rows = audit.old.json_rows

    def rows(path):
        for row in old_rows(path):
            if path == log:
                row = {**row, "CtxStepRewardV2": row["CtxStepRewardV3"]}
            yield row

    # The existing check is on reward values, ordered groups of eight, and
    # advantage without std scaling. Only feed it this experiment's column and scorer.
    with patch.object(audit.old, "json_rows", rows), patch.multiple(
        audit, RewardComponents=RewardComponents, score_raw=score_completion_v3
    ):
        receipt, measurements = _task11_recheck(log, users, steps)
    return {**receipt, "reward_version": 3, "non_gold_price": NONGOLD_PRICE}, measurements


_task11_recheck = audit.reward_recheck


def join_curve(train, steps, measurements):
    curve = audit.old.curve(train, steps)
    for row, logged in zip(curve, audit.old.json_rows(train / "optimizer_steps.jsonl"), strict=True):
        step = row["step"]
        c.require(logged["reward_component_n"] == 32, "Expected 32 new generation rewards at every optimizer step")
        for k in COMPONENTS:
            c.require(math.isclose(logged[k], measurements[step][k], rel_tol=0, abs_tol=1e-10),
                      f"Plugin component disagrees with independent completion score at step {step}: {k}")
            row[k] = logged[k]
        row["reward_component_n"] = logged["reward_component_n"]
        reward = (-row["mean_n_lost"] - NONGOLD_PRICE * row["mean_n_nongold_after"]
                  - MU * row["mean_used_per_1000"] - row["invalid_rate"])
        c.require(math.isclose(reward, row["reward"], rel_tol=0, abs_tol=audit.SCORE_TOLERANCE),
                  f"Swift reward does not reconcile to components at step {step}")
    return curve


def inputs_for(smoke):
    receipt = c.read(c.SMALL / "dataset_stats.json")
    path, out, n = (c.SMOKE_DATA, SMOKE_TRAIN, 8) if smoke else (c.DATA, c.TRAIN, receipt["n"])
    c.require(receipt["status"] == "complete" and c.file_record(path) == receipt["smoke_data" if smoke else "output"],
              "Training source differs from Phase P")
    original.validate_count(n, smoke)
    return path, out, n


def run_training(smoke):
    c.require(Path(sys.prefix).resolve() == c.ROOT / "envs/grpo"
              and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
              and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
              and os.environ.get("HF_HUB_OFFLINE") == "1", "Expected offline Swift on GPU 1")
    data, out, n = inputs_for(smoke)
    c.expect_new(out)
    if not smoke:
        evidence = c.read(SMOKE_RECEIPT)
        c.require(evidence["status"] == "ok" and evidence["optimizer_steps"] == 2
                  and evidence["reward_recheck"]["mismatches"] == evidence["reward_recheck"]["advantage_mismatches"] == 0,
                  "Training smoke did not pass")
    c.require(gpu()["memory_used_mib"] < 2048, "Physical GPU 1 occupied after admission")
    out.mkdir()
    with patch.object(original, "take_components", take_components):
        original.train(out, frozen_args(data, out, n, smoke))


def validate(smoke):
    data, out, n = inputs_for(smoke)
    target = SMOKE_RECEIPT if smoke else TRAIN_RECEIPT
    c.expect_new(target, *([] if smoke else [CURVE]))
    with patch.multiple(audit, reward_recheck=reward_recheck, join_curve=join_curve, verify_config=verify_config):
        curve, checks = audit.validate(out, data, smoke=smoke)
    c.require(checks["optimizer_steps"] == n // 4 and checks["n_prompts"] == n and
              checks["reward_recheck"]["mismatches"] == checks["reward_recheck"]["advantage_mismatches"] == 0,
              "Reward or advantage audit failed")
    if smoke:
        checks["step_metrics"] = curve
    else:
        c.require(checks["checkpoint_steps"] == list(c.CHECKPOINT_ORDER), "Expected all four Task 18 checkpoints")
        with CURVE.open("x", newline="") as out_file:
            writer = csv.DictWriter(out_file, list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        checks["curve"] = c.file_record(CURVE)
    c.write_json_exclusive(target, {**checks, "reward_version": 3, "non_gold_price": NONGOLD_PRICE})
    print(f"{'Smoke' if smoke else 'Training'} audit: {len(curve)} steps, {checks['reward_recheck']['n']} completions")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("smoke", "train", "check-smoke", "check"))
    args = cli.parse_args()
    if args.action in ("smoke", "train"):
        run_training(args.action == "smoke")
    else:
        validate(args.action == "check-smoke")


if __name__ == "__main__":
    main()

