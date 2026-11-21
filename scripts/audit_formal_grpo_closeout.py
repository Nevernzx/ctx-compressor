#!/usr/bin/env python3
"""Offline cross-check of Task 08 held-out rollouts, filter groups and sampled train rows."""
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.audit_final_rollout_v2 import REASONS, apply, close, digest, load, require
from src.grpo_reward import score_completion

EXT = Path("/root/userfolder/ctx-compressor")
OUT = PROJECT / "eval/grpo_v2"
FORMAL = EXT / "grpo_v2"
F0 = "merged_sft_v2"
MODELS = [F0, *[f"checkpoint-{i}" for i in (125, 250, 375, 500)]]
STATE_FIELDS = ("questions", "mode", "budget", "kept_ids", "new_ids", "ntok", "pid_by_lid", "gold")


def rows(path):
    with path.open("rb") as stream:
        for line in stream:
            yield json.loads(line)


def compare_sft(row, old):
    require(all(row[k] == old[k] for k in ("mode", "budget_tier", "task", "step", "gold", "returned")),
            "F0 and SFT retrieval records are misaligned")
    same_prompt = all(row["generation_metadata"][k] == old["generation_metadata"][k]
                      for k in ("prompt_token_ids", "user_text_sha256"))
    same = same_prompt and all(row["state"][k] == old["state"][k] for k in STATE_FIELDS)
    require(row["step"] or same, "F0/SFT initial state mismatch")
    return same, same and row["raw_output"] != old["raw_output"]


def compare_counts(numerator, denominator, recorded):
    require((numerator, denominator) == (recorded["count"], recorded["n"]), "count or denominator mismatch")
    if denominator:
        require(close(numerator / denominator, recorded["rate"]), "rate mismatch")


def summarize(count, official, extra, diag, name):
    n, steps, valid = (count[k] for k in ("n", "steps", "valid"))
    result = {"n_trajectories": n, "n_steps": steps, "n_valid": valid, "evidence_loss_count": count["loss"],
              "sum_added": count["added"], "mean_added_per_valid": count["added"] / valid}
    require(n == official["n_tasks"] and steps == official["n_steps"] == extra["n_steps"]
            and valid == extra["n_valid"], "evaluation dimensions")
    compare_counts(count["loss"], n, official["evidence_loss"])
    measurements = [(key, denom, record) for key, denom, record in (
        ("final_recall", n, official["final_recall"]), ("micro_step_recall", steps, official["step_recall"]["micro"]),
        ("macro_step_recall", n, official["step_recall"]["task_macro"]), ("occupancy", n, official["occupancy"]),
        ("cached_cost", n, official["cost"]["cached"]), ("added", valid, extra["mean_added_per_valid"]))]
    for key, denom, record in measurements:
        mean = count[key] / denom
        require(denom == record["n"] and close(mean, record["mean"]), key)
        if key != "added":
            result[key] = mean
    for key in REASONS:
        compare_counts(count[key], steps, official["invalid_outputs"][key])
    for key in ("new_gold_add", "kept_gold_survival", "all_gold_among_valid_reselect", "all_gold_valid_drop"):
        compare_counts(count[key], count[key + "_n"], diag[key])
    for key, record in diag["first_error"].items():
        compare_counts(count["first_" + key], n, record)
    for bucket, record in extra["add_counts"].items():
        compare_counts(count["add_" + bucket], valid, record)
    for key, strata in (("title_", "new_gold_by_title"), ("current_", "current_gold_valid_new_by_title")):
        for label in ("same", "different"):
            compare_counts(count[key + label], count[key + label + "_n"], extra[strata][label])
    if name == F0:
        compare_counts(count["raw_differences"], count["same_state"], extra["raw_diff_identical_state"])
        result["raw_differences"] = count["raw_differences"]
        result["same_state"] = count["same_state"]
    return result


def replay(name, tasks, titles):
    raw = EXT / "eval_rollouts_grpo_v2" / name
    metrics = load(OUT / name / f"{name}_results.json")
    extra = load(OUT / name / "extra.json")
    diag = load(OUT / name / "diagnostics.json")
    require(metrics == load(raw / "metrics.json"), "large and tracked evaluation metrics disagree")
    cells, previous, next_step = defaultdict(Counter), {}, Counter()
    seen, step_sums, ever_failed = defaultdict(set), Counter(), set()
    sha = hashlib.sha256()
    with ExitStack() as resources:
        source = resources.enter_context((raw / "steps.jsonl").open("rb"))
        old = None
        if name == F0:
            old = resources.enter_context((EXT / "eval_rollouts_v2/checkpoint-2000/steps.jsonl").open("rb"))
        for line in source:
            sha.update(line)
            row = json.loads(line)
            task, mode, tier, pos = (row[k] for k in ("task", "mode", "budget_tier", "step"))
            key = (task, mode, tier)
            c = cells[f"{mode}|{tier}"]
            c["steps"] += 1
            require(row["strategy"] == name and next_step[key] == pos, "trajectory order")
            require(previous.get(key, []) == row["kept_before"], "trajectory continuity")
            saved = tasks[task][pos]
            require((saved["new"], saved["gold"]) == (row["returned"], row["gold"]), "held-out source")
            if old is not None:
                same, changed = compare_sft(row, json.loads(next(old)))
                c["same_state"] += same
                c["raw_differences"] += changed
            state = row["state"]
            gen = row["generation_metadata"]
            require(not row["output_limit_hit"] and gen["runtime_prompt_ids_match"] and not gen["input_truncated"],
                    "held-out generation limits")
            before, prev = row["kept_before"], set(row["kept_before"])
            reason, local_after = apply(row["raw_output"], state)
            mapping = state["pid_by_lid"]
            after = [mapping[str(i)] for i in local_after]
            now = set(after)
            valid = reason is None
            require(reason == row["invalid_reason"] and valid == row["action_valid"]
                    and after == row["keep_after"], "action replay disagrees")
            old_gold = prev & seen[key]
            gold = row["gold"]
            seen[key].add(gold)
            returned, fresh = set(row["returned"]), set(row["new"])
            require(fresh == returned - prev, "new IDs")
            available = seen[key] & (prev | returned)
            arrived = available - prev
            recall = len(now & available) / len(available) if available else 1
            require(close(recall, row["recall"]), "step recall")
            c["micro_step_recall"] += recall
            step_sums[key] += recall
            if valid:
                added = len(now - prev)
                c["valid"] += 1
                c["added"] += added
                c["add_" + ("3+" if added >= 3 else str(added))] += 1
            else:
                c[reason] += 1
            c["new_gold_add"] += len(now & arrived)
            c["new_gold_add_n"] += len(arrived)
            for pid in arrived:
                kind = "same" if any(p != pid and titles[p] == titles[pid] for p in returned) else "different"
                c["title_" + kind] += pid in now
                c["title_" + kind + "_n"] += 1
            if valid and gold in fresh:
                kind = "same" if any(p != gold and titles[p] == titles[gold] for p in fresh) else "different"
                c["current_" + kind] += gold in now
                c["current_" + kind + "_n"] += 1
            if mode == "reselect":
                c["kept_gold_survival"] += len(now & old_gold)
                c["kept_gold_survival_n"] += len(old_gold)
                if pos and valid:
                    c["all_gold_among_valid_reselect_n"] += 1
                    if prev == old_gold:
                        c["all_gold_among_valid_reselect"] += 1
                        c["all_gold_valid_drop_n"] += 1
                        c["all_gold_valid_drop"] += bool(prev - now)
            miss, dropped = bool(arrived - now), bool(old_gold - now)
            if key not in ever_failed and (reason or miss or dropped):
                cause = "invalid_output" if reason else ("both" if miss and dropped else
                        "new_gold_not_added" if miss else "old_gold_dropped")
                c["first_" + cause] += 1
                ever_failed.add(key)
            token_counts = {mapping[k]: v for k, v in state["ntok"].items()}
            prefix = 0
            for p, q in zip(before, after):
                if p != q:
                    break
                prefix += token_counts[p]
            used = sum(token_counts[p] for p in now)
            require(used == row["used_after"], "executed budget size")
            c["cached_cost"] += 0.05 * prefix + 1.25 * (used - prefix)
            previous[key] = after
            next_step[key] += 1
            if pos + 1 == len(tasks[task]):
                c["n"] += 1
                c["loss"] += bool(seen[key] - now)
                c["final_recall"] += len(seen[key] & now) / len(seen[key])
                c["macro_step_recall"] += step_sums[key] / next_step[key]
                c["occupancy"] += used / row["effective_budget"]
                c["first_none"] += key not in ever_failed
        require(old is None or next(old, None) is None, "extra SFT records")
    require(sha.hexdigest() == metrics["steps_artifact"]["sha256"], "evaluation raw checksum")
    require(len(next_step) == 1494 and all(v == len(tasks[k[0]]) for k, v in next_step.items()), "coverage")
    pooled = Counter()
    checked = {}
    for cell, count in cells.items():
        pooled.update(count)
        checked[cell] = summarize(count, metrics["configs"][name + "|" + cell]["metrics"],
                                  extra["configs"][cell], diag["configs"][cell], name)
    return {"raw_sha256": sha.hexdigest(), "cells": checked,
            "pooled": summarize(pooled, metrics["combined"], extra["combined"], diag["combined"], name)}


def filter_and_dataset(heldout_example_ids):
    states = FORMAL / "states/states.jsonl"
    groups = FORMAL / "filter/groups.jsonl"
    train = FORMAL / "data/train.jsonl"
    probe = load(groups.parent / "sampling_probe.json")
    stats = load(train.parent / "dataset_stats.json")
    require(probe["informative_threshold"] == 1e-9 and probe["ddof"] == 0 and probe["lambda"] == 0.3, "filter setup")
    for path, reference in ((states, probe["state_file"]), (groups, probe["group_file"]), (train, stats["output"])):
        require(digest(path) == reference["sha256"], "state, group or train identity")
    exported = {(r["trajectory_id"], r["state_step"]): r for r in rows(train)}
    train_keys = [(r["trajectory_id"], r["state_step"]) for r in rows(train)]
    require(len(train_keys) == len(exported) == 2000, "unique exported IDs")
    lookup, state_meta = {}, {}
    available = defaultdict(list)
    total = Counter()
    for group in rows(groups):
        key = group["trajectory_id"], group["step"]
        require(key not in lookup and group["split"] == "train", "filter split or repeat")
        sd = statistics.pstdev(group["rewards"])
        require(len(group["rewards"]) == 8 and sd == group["population_std"] and
                (sd > 1e-9) == group["informative"], "eight-sample filter threshold")
        lookup[key] = group
    used_ids = {group["task"] for group in lookup.values()}
    train_tasks = {r["task"]: r["example_ids"] for r in rows(EXT / "data/traj_train_v2/train_tasks.jsonl")
                   if r["split"] == "train" and r["task"] in used_ids}
    train_examples = {example for examples in train_tasks.values() for example in examples}
    require(set(train_tasks) == used_ids and len(train_examples) == 1600 and len(heldout_example_ids) == 996
            and train_examples.isdisjoint(heldout_example_ids), "train and held-out example identities")
    for state in rows(states):
        key = state["trajectory_id"], state["step"]
        group = lookup[key]
        require(key not in state_meta and state["split"] == "train" and
                all(state[k] == group[k] for k in ("task", "mode", "budget_tier", "prompt_sha256")), "B/C identity")
        packed = json.loads(state["reward_state_json"])
        require(packed["prompt_token_ids_tail"] == state["prompt_token_ids"][-24:], "reward token suffix")
        state_meta[key] = packed, state["greedy_output"]
        cell = group["cell"]
        require(cell == state["mode"] + "|" + state["budget_tier"], "cell key")
        total[cell] += 1
        if group["informative"]:
            available[cell].append(key)
        if key in exported:
            out = exported[key]
            require(group["informative"] and out["cell"] == cell and out["messages"] == state["messages"]
                    == [{"role": "user", "content": state["user_text"]}]
                    and out["reward_state_json"] == state["reward_state_json"], "selected row identity")

    require(state_meta.keys() == lookup.keys() and len(lookup) == 22158, "B/C coverage")
    seen = set()
    for cell, record in probe["raw_samples_by_cell"].items():
        raw_path = Path(record["path"])
        require(digest(raw_path) == record["sha256"], "raw C stream checksum")
        for row in rows(raw_path):
            key = row["trajectory_id"], row["step"]
            group = lookup[key]
            packed, greedy = state_meta[key]
            require(key not in seen and group["cell"] == cell and
                    row["prompt_sha256"] == group["prompt_sha256"], "raw C identity")
            seen.add(key)
            require(score_completion(greedy, packed)["reward"] == group["greedy_reward"] == row["greedy_reward"],
                    "greedy reward")
            samples = sorted(row["samples"], key=lambda s: s["index"])
            require([s["index"] for s in samples] == list(range(8)), "raw C sample indices")
            for i, sample in enumerate(samples):
                actual = score_completion(sample["completion"], packed)
                count = None if actual["action"] is None else len(set(actual["action"]["add"]))
                require(actual["reward"] == sample["reward"] == group["rewards"][i]
                        and actual["invalid_reason"] == sample["invalid_reason"] == group["invalid_reasons"][i]
                        and count == group["add_counts"][i], "reward, invalid reason or add count mismatch")
    require(seen == lookup.keys(), "raw C samples cover all states")
    cells = sorted(total)
    n = min(2000, sum(map(len, available.values())))
    base, extra = divmod(n, len(cells))
    quotas = {cell: min(len(available[cell]), base + (i < extra)) for i, cell in enumerate(cells)}
    while sum(quotas.values()) < n:
        for cell in cells:
            if sum(quotas.values()) < n and quotas[cell] < len(available[cell]):
                quotas[cell] += 1
    rng = random.Random(20261121)
    expected = [key for cell in cells for key in rng.sample(available[cell], quotas[cell])]
    rng.shuffle(expected)
    require(expected == train_keys, "seeded balanced sampling or ordering")
    for cell in cells:
        compare_counts(len(available[cell]), total[cell], probe["cells"][cell]["informative"])
    compare_counts(sum(map(len, available.values())), len(lookup), probe["pooled"]["informative"])
    require(stats["per_cell_counts"] == quotas and stats["per_cell_available"] ==
            {c: len(available[c]) for c in cells}, "D summary")
    return {"n_train_example_ids": len(train_examples), "n_heldout_example_ids": len(heldout_example_ids),
            "overlapping_example_ids": 0, "n_train_states": len(lookup), "n_scores_recomputed": 8 * len(lookup),
            "n_greedy_scores_recomputed": len(lookup), "n_informative": sum(map(len, available.values())),
            "sampling_seed": 20261121, "n_train_rows": n, "same_seeded_order": True,
            "cells": {c: {"group_n": total[c], "informative_n": len(available[c]), "selected_n": quotas[c]}
                      for c in cells}}


def main():
    source = EXT / "data/traj_train"
    heldout = [r for r in rows(source / "heldout_tasks.jsonl") if r["split"] == "heldout"]
    tasks = {r["task"]: r["steps"] for r in heldout}
    heldout_examples = {example for r in heldout for example in r["example_ids"]}
    del heldout
    require(len(tasks) == 249 and sum(map(len, tasks.values())) == 2326, "shared source dimensions")
    titles = {r["pid"]: r["title"] for r in rows(source / "corpus_train.jsonl")}
    evaluated = {name: replay(name, tasks, titles) for name in MODELS}
    best = min(MODELS[1:], key=lambda k: (evaluated[k]["pooled"]["evidence_loss_count"],
               -evaluated[k]["pooled"]["final_recall"], int(k.split("-")[-1])))
    require(best == load(OUT / "selected_checkpoint.json")["strategy"], "checkpoint choice")
    result = {"status": "ok", "heldout_tasks": len(tasks), "n_runs": len(evaluated),
              "n_step_records": sum(r["pooled"]["n_steps"] for r in evaluated.values()),
              "selected_checkpoint": best, "evaluation": evaluated, "filter_and_dataset": filter_and_dataset(heldout_examples)}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target = OUT / "final_audit.json"
    if target.exists():
        require(text == target.read_text(), "audit differs from its previous verified output")
    else:
        with target.open("x") as stream:
            stream.write(text)
    print(best, result["n_step_records"], result["filter_and_dataset"]["n_scores_recomputed"])


if __name__ == "__main__":
    main()
