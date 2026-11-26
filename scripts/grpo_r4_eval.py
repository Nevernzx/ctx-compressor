"""Replay the Task 11 held-out evaluator and score the v3 objective on executed actions."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import math
from pathlib import Path
from unittest.mock import patch

from scripts import eval_grpo_r2 as old
from scripts import grpo_r4_common as c
from scripts import grpo_r4_training as training
from scripts.build_grpo_states import lines, rate
from scripts.sft_run_report import validate_strategy
from src.grpo_reward_v3 import MU, NONGOLD_PRICE, score_completion_v3
from src import reader_heldout_inputs as inputs

STEPS = 13956
TRAJECTORIES = 1494
SELECTED = c.SMALL / "selected_checkpoint.json"
BASELINES = c.SMALL / "historical_reward_v3.json"


def aggregate_reward(counts):
    n = counts["n"]
    return {
        "mu": MU, "non_gold_price": NONGOLD_PRICE, "n_steps": n,
        "reward_units_1_over_2000": -counts["penalty_units"],
        "reward": old.mean(-counts["penalty_units"] / 2000, n),
        "n_lost": old.mean(counts["lost"], n),
        "n_nongold_after": old.mean(counts["nongold"], n),
        "used_per_1000": old.mean(counts["tokens"] / 1000, n),
        "invalid": rate(counts["invalid"], n),
    }


def score_trace(path, gold_by_task, metrics):
    per_cell = defaultdict(Counter)
    per_final = defaultdict(Counter)
    finals = {}
    for row in lines(path):
        name = f"{row['mode']}|{row['budget_tier']}"
        key = (name, row["task"])
        task_gold = gold_by_task[row["task"]]
        step = row["step"]
        state = row["state"]
        c.require(step == finals.get(key, -1) + 1 and step < len(task_gold)
                  and row["gold"] == task_gold[step], "Held-out step order or task gold differs")
        finals[key] = step
        gold_set = set(task_gold)
        score = score_completion_v3(row["raw_output"], {**state, "task_gold_pids": sorted(gold_set)})
        kept_pids = [state["pid_by_lid"][str(lid)] for lid in score["keep_after"]]
        c.require(kept_pids == row["keep_after"] and score["keep_after"] == state["keep_after"]
                  and score["used_after"] == row["used_after"] and score["invalid_reason"] == row["invalid_reason"],
                  "v3 executor and archived held-out action disagree")
        lost, nongold, ntok, invalid = (score[k] for k in ("n_lost", "n_nongold_after", "used_after", "invalid_penalty"))
        penalty_units = 2000 * (lost + invalid) + 320 * nongold + ntok
        c.require(math.isclose(-penalty_units / 2000, score["reward"], rel_tol=0, abs_tol=1e-12)
                  and nongold == sum(pid not in gold_set for pid in row["keep_after"]), "v3 component math differs")
        per_cell[name].update(n=1, lost=lost, nongold=nongold, tokens=ntok, invalid=invalid, penalty_units=penalty_units)
        if step == len(task_gold) - 1:
            per_final[name].update(n=1, total=len(kept_pids), nongold=nongold)
    config = {"|".join((r["mode"], r["budget_tier"])): r["metrics"] for r in metrics["configs"].values()}
    c.require(len(finals) == TRAJECTORIES and set(config) == set(per_final) == set(per_cell)
              and all(last + 1 == len(gold_by_task[task]) for (_, task), last in finals.items()),
              "Incomplete held-out trace")
    config["pooled"] = metrics["combined"]
    per_cell["pooled"] = sum(per_cell.values(), Counter())
    per_final["pooled"] = sum(per_final.values(), Counter())
    c.require(per_cell["pooled"]["n"] == STEPS, "Held-out steps differ")
    for cell, proof in config.items():
        c.require(per_cell[cell]["n"] == proof["n_steps"] and per_final[cell]["n"] == proof["n_tasks"],
                  "Reward/terminal denominator differs from rollout")
    return {
        cell: {"heldout_reward_v3": aggregate_reward(per_cell[cell]),
               "final_context": {"segments": old.mean(per_final[cell]["total"], per_final[cell]["n"]),
                                 "nongold": old.mean(per_final[cell]["nongold"], per_final[cell]["n"])}}
        for cell in config
    }


def gold():
    return c.task_gold(c.HELDOUT, "heldout", 249, c.HELDOUT_SHA)


def run_eval(step):
    name = f"checkpoint-{step}"
    receipt = c.read(training.TRAIN_RECEIPT)
    c.require(receipt["status"] == "ok" and step in receipt["checkpoint_steps"]
              and c.digest(c.TRAIN / name / "adapter_model.safetensors") == receipt["adapter_sha256"][name],
              "Adapter was not verified in E")
    task_gold, gold_file = gold()
    old_augment = old.augment_objective

    def augment(extra, path, reference=None):
        c.require(reference is None and path == c.RAW / name / "steps.jsonl", "Unexpected F0 or rollout destination")
        old_augment(extra, path)
        metrics = c.read(path.with_name("metrics.json"))
        priced = score_trace(path, task_gold, metrics)
        for cell, results in priced.items():
            dest = extra["combined"] if cell == "pooled" else extra["configs"][cell]
            dest.update(results)
        extra["task_gold_source_v3"] = gold_file
        return extra

    # Preserve Task 11's replay, including its byte/token step-0 checks.
    with patch.multiple(old, RUN=c.RUN, SMALL=c.SMALL, RAW=c.RAW, augment_objective=augment):
        old.run(name)


def check_eval(step):
    name = f"checkpoint-{step}"
    folder = c.SMALL / name
    run = c.read(folder / "run.json")
    c.require(run["status"] == "complete" and run["engine_shutdown"]["completed"]
              and run["model"] == str(c.BASE) and run["adapter"] == str(c.TRAIN / name)
              and run["step0"]["n"] == TRAJECTORIES
              and run["step0"]["token_id_mismatches"] == run["step0"]["user_byte_mismatches"] == 0
              and run["max_tokens"] == 256 and run["temperature"] == 0, f"{name}: held-out run did not pass")
    for path in (folder / f"{name}_results.json", folder / "extra.json", folder / "diagnostics.json"):
        c.require(c.file_record(path) == run["artifacts"][path.name], f"{name}: signed small artifact changed")
    metrics = validate_strategy(c.read(folder / f"{name}_results.json"), name)
    c.require(metrics["combined"]["n_steps"] == STEPS and metrics["combined"]["n_tasks"] == TRAJECTORIES
              and metrics["steps_artifact"] == c.file_record(c.RAW / name / "steps.jsonl"), "Incomplete or changed F rollout")
    extra = c.read(folder / "extra.json")
    c.require(extra["combined"]["heldout_reward_v3"]["n_steps"] == STEPS
              and all(row["heldout_reward_v3"]["n_steps"] == 2326 for row in extra["configs"].values()),
              "Incomplete v3 reward audit")
    return run, metrics, extra


def select_checkpoint(candidates):
    def order(entry):
        score = entry["heldout_reward_v3"]
        return (-Fraction(score["reward_units_1_over_2000"], score["n_steps"]),
                entry["evidence_loss"]["count"], entry["checkpoint_step"])
    if not candidates:
        raise ValueError("No audited round-four checkpoint")
    return min(candidates, key=order)


def historical():
    assets = inputs.historical_sources(inputs.sums(c.PROJECT / "SHA256SUMS"))
    wanted = (c.R2_POLICY, c.V3_POLICY)
    return [next(source for source in assets if source.name == name) for name in wanted]


def baseline_rewards():
    c.expect_new(BASELINES)
    task_gold, gold_file = gold()
    sources = {}
    for asset in historical():
        inputs.verify(asset.steps, asset.steps_sha256)
        metrics = inputs.checked_json(asset.metrics, asset.metrics_sha256)
        c.require(metrics["steps_artifact"] == c.file_record(asset.steps), "Signed baseline metrics/steps disagree")
        sources[asset.name] = {"steps": metrics["steps_artifact"], "metrics": c.file_record(asset.metrics),
                               "heldout": score_trace(asset.steps, task_gold, metrics)}
    c.write_json_exclusive(BASELINES, {"status": "complete", "task_gold_source": gold_file, "strategies": sources})


def finalize_f():
    c.expect_new(SELECTED)
    choices = []
    for step in c.CHECKPOINT_ORDER:
        run, metrics, extra = check_eval(step)
        choices.append({
            "strategy": f"checkpoint-{step}", "checkpoint_step": step, "path": run["adapter"],
            "adapter_sha256": run["adapter_weight"]["sha256"],
            "heldout_reward_v3": extra["combined"]["heldout_reward_v3"],
            "final_context": extra["combined"]["final_context"],
            "evidence_loss": metrics["combined"]["evidence_loss"], "final_recall": metrics["combined"]["final_recall"],
            "run": c.file_record(c.SMALL / f"checkpoint-{step}" / "run.json"),
        })
    winner = select_checkpoint(choices)
    c.write_json_exclusive(SELECTED, {**winner, "n_candidates": len(choices), "candidates": choices,
                                     "rule": "Maximum pooled held-out reward v3, then fewer loss trajectories, then earliest step."})
    print(f"F selected {winner['strategy']} by reward v3")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("eval", "select", "baselines"))
    cli.add_argument("--step", type=int, choices=c.CHECKPOINT_ORDER)
    args = cli.parse_args()
    if args.action == "eval":
        c.require(args.step in c.CHECKPOINT_ORDER, "Expected registered checkpoint step")
        run_eval(args.step)
    elif args.action == "select":
        finalize_f()
    else:
        baseline_rewards()

