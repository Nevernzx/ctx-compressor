#!/usr/bin/env python3
"""Offline replay of the Task 09 threshold grid and all three held-out policies."""
from collections import Counter, defaultdict
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.audit_final_rollout_v2 import REASONS, apply, close, digest, load, require
from scripts.audit_formal_grpo_closeout import compare_sft, rows
from src.heldout_eval import parse_action

EXT = Path("/root/userfolder/ctx-compressor")
OUT = PROJECT / "eval/hedge_v2"
TRAIN_STATES = EXT / "grpo_v2/states/states.jsonl"
TRAIN_SCORES = EXT / "hedge_v2/train/state_scores.jsonl"
CELLS = tuple(f"{mode}|{tier}" for mode in ("append", "reselect") for tier in ("loose", "mid", "tight"))
GRID = (0., 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01)
BUCKETS = ("0", "1", "2", "3+")


def confidence(state, saved):
    ids = list(dict.fromkeys(i for i in state["new_ids"] if i not in state["kept_ids"]))
    scores = {int(k): v for k, v in saved.items()}
    require(set(ids) == set(scores) and all(math.isfinite(v) for v in scores.values()), "candidate score coverage")
    ranked = sorted(ids, key=scores.__getitem__, reverse=True)
    if not ranked:
        return ranked, None
    top = scores[ranked[0]]
    return ranked, 1 / math.fsum(math.exp(v-top) for v in scores.values())


def execute(state, greedy, ranking, confidence_value, tau):
    reason, keep = apply(greedy, state)
    used = sum(state["ntok"][str(i)] for i in keep)
    executed, extra, extra_tokens = greedy, None, 0
    if reason is None and confidence_value is not None and confidence_value < tau:
        action = parse_action(greedy)
        candidate = next((i for i in ranking if i not in action["add"]), None)
        if candidate is not None:
            size = state["ntok"][str(candidate)]
            if used + size <= state["budget"]:
                extra, extra_tokens = candidate, size
                action = {"add": [*action["add"], candidate], "drop": action.get("drop", [])}
                executed = json.dumps(action, separators=(",", ":"))
                reason, keep = apply(executed, state)
                used += size
                require(reason is None and extra in keep, "invalid hedge independently constructed")
    gold = set(state["gold"])
    step_recall = len(gold.intersection(keep))/len(gold) if gold else 1.0
    reward = -1.0 if reason is not None else step_recall + (0.3*(1-used/state["budget"]) if step_recall == 1.0 else 0.0)
    require(used <= state["budget"], "post-action budget exceeded")
    return {"raw_output": executed, "keep_after": keep, "extra_id": extra, "hedged": extra is not None,
            "extra_tokens": extra_tokens, "reward": reward}, reason


def probability_check(p, saved):
    require((p is None and saved is None) or (p is not None and saved is not None and close(p, saved)), "top confidence")


def check_rate(num, den, expected):
    require((num, den) == (expected["count"], expected["n"]), "rate count or denominator")
    require(close(num / den, expected["rate"]) if den else expected["rate"] is None, "rate")


def check_mean(total, den, expected):
    require(den == expected["n"] and close(total / den, expected["mean"]), "mean or denominator")
    if "sum" in expected:
        require(close(total, expected["sum"]), "recorded sum")


def train_audit():
    result = load(OUT / "train_selection.json")
    source = Path(result["train_task_file"]["path"])
    for path, name in ((TRAIN_STATES, "state_file"), (TRAIN_SCORES, "score_file"), (source, "train_task_file")):
        require(digest(path) == result[name]["sha256"], name)
    require(TRAIN_STATES.is_relative_to(EXT / "grpo_v2") and tuple(result["grid"]) == GRID
            and result["split"] == "train" and result["lambda"] == 0.3, "train protocol")
    tasks = {r["task"]: (r["steps"], r["example_ids"]) for r in rows(source) if r["split"] == "train"}
    seen, candidates = set(), 0
    totals = defaultdict(Counter)
    for original, row in zip(rows(TRAIN_STATES), rows(TRAIN_SCORES), strict=True):
        require(original["split"] == row["split"] == "train" and all(original[k] == row[k]
                for k in ("task", "trajectory_id", "step", "prompt_sha256")), "train step identity")
        identity = (row["task"], row["cell"], row["step"])
        require(row["cell"] == original["mode"]+"|"+original["budget_tier"] and identity not in seen
                and row["greedy_raw"] == original["greedy_output"], "train source, identity or greedy action")
        seen.add(identity)
        state = json.loads(original["reward_state_json"])
        ranked, p = confidence(state, row["scores"])
        probability_check(p, row["p1"])
        candidates += len(ranked)
        pid = tasks[row["task"]][0][row["step"]]["gold"]
        ids = json.loads(original["pid_by_lid_json"])
        current = next((i for i in ranked if ids[str(i)] == pid), None)
        require(current == row["current_gold_new_lid"], "training current gold")
        require(set(row["grid_results"]) == {str(t) for t in GRID}, "saved threshold grid")
        for tau in GRID:
            actual, reason = execute(state, row["greedy_raw"], ranked, p, tau)
            require(actual == row["grid_results"][str(tau)], "train action, extra tokens or reward")
            if tau == 0:
                require(reason == original["greedy_invalid_reason"] and actual["keep_after"] == original["greedy_keep_after"],
                        "training greedy replay")
            c = totals[row["cell"], tau]
            c["n"] += 1
            c["reward"] += actual["reward"]
            c["extra_tokens"] += actual["extra_tokens"]
            c["hedged"] += actual["hedged"]
            c["hit"] += actual["hedged"] and actual["extra_id"] in state["gold"]
            c["current_hit"] += actual["hedged"] and actual["extra_id"] == current
            c["current_n"] += current is not None
            c["miss"] += current is not None and current not in actual["keep_after"]
    task_ids = {task for task, _, _ in seen}
    expected = {(task, cell, step) for task in task_ids for cell in CELLS for step in range(len(tasks[task][0]))}
    require(seen == expected and len(task_ids) == 400 and len(seen) == 22158, "full training coverage")
    means = {}
    for cell in (*CELLS, "pooled"):
        official = result["pooled"] if cell == "pooled" else result["cells"][cell]
        for tau in GRID:
            c = totals[cell, tau] if cell != "pooled" else sum((totals[k, tau] for k in CELLS), Counter())
            n = c["n"]
            require(n == (22158 if cell == "pooled" else 3693) == official[str(tau)]["n"], "training cell size")
            ref = official[str(tau)]
            check_mean(c["reward"], n, ref["mean_reward"])
            check_mean(c["extra_tokens"], n, ref["mean_extra_tokens"])
            for key, numerator, denominator in (("hedge_rate", c["hedged"], n), ("hedge_hit_rate", c["hit"], c["hedged"]),
                    ("hedge_current_gold_hit_rate", c["current_hit"], c["hedged"]),
                    ("current_gold_miss_rate", c["miss"], c["current_n"])):
                check_rate(numerator, denominator, ref[key])
            if cell == "pooled":
                means[tau] = c["reward"]/n
    selected = min(GRID, key=lambda tau: (-means[tau], tau))
    require(selected == result["tau_star"], "independent threshold choice")
    train_examples = {q for task in task_ids for q in tasks[task][1]}
    return {"n_tasks": len(task_ids), "n_trajectories": len(task_ids)*len(CELLS), "n_states": len(seen),
            "n_scored_candidates": candidates, "n_thresholds": len(GRID), "n_replayed_actions": len(seen)*len(GRID),
            "n_train_question_ids": len(train_examples), "recomputed_pooled_mean_reward": means,
            "tau_star": selected, "state_sha256": result["state_file"]["sha256"],
            "score_sha256": result["score_file"]["sha256"]}, train_examples


def check_eval_summary(c, metrics, extra, same_state):
    n, steps, valid = c["n"], c["steps"], c["valid"]
    check_rate(c["loss"], n, metrics["evidence_loss"])
    for key, denominator, ref in (("final", n, metrics["final_recall"]), ("occupancy", n, metrics["occupancy"]),
            ("cached", n, metrics["cost"]["cached"]), ("micro", steps, metrics["step_recall"]["micro"]),
            ("macro", n, metrics["step_recall"]["task_macro"]), ("added", valid, extra["mean_added_per_valid"]),
            ("tokens", steps, extra["mean_kept_tokens_per_step"]),
            ("extra_tokens", steps, extra["mean_extra_tokens_per_step"])):
        check_mean(c[key], denominator, ref)
    for reason in REASONS:
        check_rate(c[reason], steps, metrics["invalid_outputs"][reason])
    check_rate(steps-valid, steps, metrics["invalid_outputs"]["any_invalid"])
    check_rate(c["hedged"], steps, extra["hedge_rate"])
    check_rate(c["hit"], c["hedged"], extra["hedge_hit_rate"])
    for bucket in BUCKETS:
        check_rate(c["all_"+bucket], steps, extra["executed_add_counts_all_steps"][bucket])
        check_rate(c["valid_"+bucket], valid, extra["add_counts"][bucket])
    for kind in ("same","different"):
        check_rate(c["hit_"+kind], c["gold_"+kind], extra["new_gold_by_title"][kind])
    if same_state:
        check_rate(c["diff"], c["same"], extra["raw_diff_identical_state"])
    return {"n_trajectories": n, "n_steps": steps, "n_valid": valid, "loss_count": c["loss"],
            "final_recall": c["final"]/n, "mean_cached_cost": c["cached"]/n, "mean_occupancy": c["occupancy"]/n,
            "n_hedges": c["hedged"], "n_hedge_hits": c["hit"],
            **({"n_raw_different_on_identical_states": c["diff"], "n_identical_states": c["same"]} if same_state else {})}


def replay(name, tau, tasks, titles):
    path = EXT / "eval_rollouts_hedge_v2" / name / "steps.jsonl"
    metrics = load(OUT/name/"metrics.json")
    extra = load(OUT/name/"extra.json")
    require(digest(path) == metrics["steps_artifact"]["sha256"] and metrics == load(path.with_name("metrics.json")),
            "heldout raw and metrics receipts")
    cells = defaultdict(Counter)
    previous, positions, cumulative = {}, Counter(), Counter()
    revealed = defaultdict(set)
    paired = name == "nohedge"
    old = rows(EXT / "eval_rollouts_v2/checkpoint-2000/steps.jsonl") if paired else iter(())
    for r, baseline in zip_longest(rows(path), old):
        require(r is not None and ((baseline is not None) == paired), "SFT/H0 row count")
        key = (r["task"], r["mode"], r["budget_tier"])
        t, mode, tier = key
        step = r["step"]
        require(positions[key] == step and previous.get(key, []) == r["kept_before"] and r["strategy"] == name,
                "trajectory continuity")
        source = tasks[t][step]
        require(r["returned"] == source["new"] and r["gold"] == source["gold"], "fixed retrieval")
        c = cells[mode+"|"+tier]
        if paired:
            same, different = compare_sft(r, baseline)
            c["same"] += same
            c["diff"] += different
        state, h = r["state"], r["hedge"]
        ranked, p = confidence(state, h["scores"])
        probability_check(p, h["p1"])
        require(tau == h["tau"], "run threshold")
        actual, reason = execute(state, r["greedy_raw_output"], ranked, p, tau)
        require(r["raw_output"] == actual["raw_output"] == r["executed_raw_output"] and all(
                actual[k] == h[k] for k in ("extra_id","extra_tokens","hedged"))
                and actual["reward"] == h["step_reward"], "heldout hedge or reward")
        mapping = state["pid_by_lid"]
        kept = [mapping[str(i)] for i in actual["keep_after"]]
        require(kept == r["keep_after"] and r["invalid_reason"] == reason and r["action_valid"] == (reason is None),
                "heldout action")
        c["hedged"] += actual["hedged"]
        hit = actual["hedged"] and actual["extra_id"] in state["gold"]
        require(hit == h["extra_is_gold"], "heldout hedge hit")
        c["hit"] += hit
        c["extra_tokens"] += actual["extra_tokens"]
        before, after = set(r["kept_before"]), set(kept)
        added = len(after-before)
        bucket = BUCKETS[min(3,added)]
        c["all_"+bucket] += 1
        if reason is None:
            c["valid"] += 1
            c["valid_"+bucket] += 1
            c["added"] += added
        else:
            c[reason] += 1
        revealed[key].add(source["gold"])
        returned = set(r["returned"])
        available = revealed[key] & (before | returned)
        require(set(r["new"]) == returned-before and {mapping[str(i)] for i in state["gold"]} == available,
                "available gold")
        recall = len(available & after)/len(available) if available else 1.
        require(close(recall,r["recall"]), "step recall")
        for gold in available-before:
            kind = "same" if any(other != gold and titles[other] == titles[gold] for other in returned) else "different"
            c["gold_"+kind] += 1
            c["hit_"+kind] += gold in after
        by_pid = {mapping[k]:v for k,v in state["ntok"].items()}
        used = sum(by_pid[pid] for pid in kept)
        cached = 0
        for left,right in zip(r["kept_before"],kept):
            if left != right:
                break
            cached += by_pid[left]
        require(used == r["used_after"] and used <= r["effective_budget"], "token budget")
        c["cached"] += 0.05*cached + 1.25*(used-cached)
        c["tokens"] += used
        c["micro"] += recall
        c["steps"] += 1
        cumulative[key] += recall
        positions[key] += 1
        previous[key] = kept
        if positions[key] == len(tasks[t]):
            c["n"] += 1
            c["final"] += len(after & revealed[key])/len(revealed[key])
            c["loss"] += bool(revealed[key]-after)
            c["occupancy"] += used/r["effective_budget"]
            c["macro"] += cumulative[key]/positions[key]
    require(len(positions) == 1494 and all(pos == len(tasks[t]) for (t,_,_),pos in positions.items())
            and set(cells) == set(CELLS), "heldout coverage")
    checked = {}
    for cell,c in cells.items():
        require(c["n"] == 249 and c["steps"] == 2326, "heldout cell count")
        checked[cell] = check_eval_summary(c,metrics["configs"][name+"|"+cell]["metrics"],extra["configs"][cell],paired)
    return {"raw_sha256": metrics["steps_artifact"]["sha256"], "tau": tau, "cells": checked,
            "pooled": check_eval_summary(sum(cells.values(),Counter()),metrics["combined"],extra["combined"],paired)}


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA_VISIBLE_DEVICES empty")
    train, train_ids = train_audit()
    held = {r["task"]:r for r in rows(EXT/"data/traj_train/heldout_tasks.jsonl") if r["split"] == "heldout"}
    held_ids = {q for r in held.values() for q in r["example_ids"]}
    require(len(held_ids) == 996 and len(train_ids) == 1600 and not held_ids & train_ids,
            "global training/held-out question overlap")
    tasks = {t:r["steps"] for t,r in held.items()}
    titles = {r["pid"]:r["title"] for r in rows(EXT/"data/traj_train/corpus_train.jsonl")}
    evaluated = {name:replay(name,tau,tasks,titles) for name,tau in (
        ("nohedge",0.),("tau_star",train["tau_star"]),("always",1.01))}
    result = {"status": "ok", "training": train, "n_heldout_tasks":len(tasks), "n_heldout_question_ids":len(held_ids),
              "overlapping_question_ids":0, "heldout":evaluated, "n_heldout_actions":sum(v["pooled"]["n_steps"] for v in evaluated.values())}
    target = OUT/"final_audit.json"
    payload = json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if target.exists():
        require(target.read_text() == payload, "prior audit differs")
    else:
        with target.open("x") as f:
            f.write(payload)
    print(json.dumps({k:v["pooled"] for k,v in evaluated.items()}))


if __name__ == "__main__":
    main()

