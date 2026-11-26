"""Re-score the complete frozen Task 11 C archive and export the registered Swift dataset."""
import argparse
from collections import Counter
import json
import math
import os
import random
from types import SimpleNamespace

from scripts import grpo_r4_common as c
from scripts import make_formal_grpo_dataset as balanced
from scripts import probe_formal_grpo as groups
from scripts.build_formal_grpo_states import COLUMNS
from scripts.build_grpo_states import compact, lines, rate
from scripts.grpo_r4_swift_plugin import CtxStepRewardV3
from scripts.probe_grpo_r2 import summarize
from src.grpo_reward_v2 import score_completion_v2
from src.grpo_reward_v3 import MU, NONGOLD_PRICE, score_completion_v3

POOL = c.R2EXT / "states/states.jsonl"
OLD_C = c.R2EXT / "filter"
NEW_C = c.RUN / "filter"
GROUPS = NEW_C / "groups.jsonl"
RECEIPT_C = c.SMALL / "sampling_probe.json"
RECEIPT_D = c.SMALL / "dataset_stats.json"


def state_with_gold(row, gold_by_step):
    state = json.loads(row["reward_state_json"])
    mapping = {int(lid): pid for lid, pid in json.loads(row["pid_by_lid_json"]).items()}
    lids = set(state["kept_ids"]) | set(state["new_ids"])
    c.require(set(mapping) >= lids and len({mapping[lid] for lid in lids}) == len(lids)
              and mapping.keys() >= set(state["gold"]), "Invalid local-to-corpus PID mapping")
    step = row["step"]
    task_gold = set(gold_by_step)
    c.require(0 <= step < len(gold_by_step) and gold_by_step[step] in {mapping[k] for k in state["gold"]}
              and {mapping[k] for k in state["gold"]} <= task_gold, "B state disagrees with task gold sequence")
    return {**state, "pid_by_lid": mapping, "task_gold_pids": sorted(task_gold)}


def score_and_check(raw, state, recorded):
    v2 = score_completion_v2(raw, state, MU)
    v3 = score_completion_v3(raw, state)
    n = v3["n_nongold_after"]
    c.require(v2["reward"] == recorded, "Archived reward v2 mismatch")
    # These coefficients are integral in 1/2000 reward units. Also verify the
    # equality in floating point to absorb only binary representation error.
    delta_units = round(2000 * v3["reward"]) - round(2000 * v2["reward"])
    c.require(v3["reward"] == v2["reward"] - NONGOLD_PRICE * n and delta_units == -320 * n
              and math.isclose(v3["reward"] - v2["reward"], -NONGOLD_PRICE * n, rel_tol=0, abs_tol=1e-12),
              "Unexpected difference between reward v2 and v3")
    return v3


def score_group(row, raw, prior, state):
    c.require(all(row[k] == raw[k] == prior[k] for k in COLUMNS)
              and prior["cell"] == groups.cell(row), "B/C identity or ordering differs")
    g = score_and_check(row["greedy_output"], state, prior["greedy_reward"])
    c.require(g["keep_after"] == row["greedy_keep_after"]
              and g["invalid_reason"] == row["greedy_invalid_reason"], "B greedy action differs")
    samples = sorted(raw["samples"], key=lambda value: value["index"])
    c.require([s["index"] for s in samples] == list(range(8)) and len(prior["rewards"]) == 8,
              "Missing or duplicate completion")
    scored = []
    for sample, reference in zip(samples, prior["rewards"], strict=True):
        c.require(len(sample["token_ids"]) <= 256, "Recorded continuation exceeds token cap")
        scored.append({"index": sample["index"], **score_and_check(sample["completion"], state, reference)})
    actual = groups.group_record({**raw, "samples": scored, "greedy_reward": g["reward"]})
    parts = [{k: s[k] for k in ("n_lost", "n_nongold_after", "used_after", "invalid_penalty")} for s in scored]
    earlier_parts = [{"n_lost": s["n_lost"], "used_after": s["used_after"], "invalid": s["invalid_penalty"]}
                     for s in scored]
    gold_varies = len({(p["n_lost"], p["invalid_penalty"]) for p in parts}) > 1
    nongold_varies = len({p["n_nongold_after"] for p in parts}) > 1
    tokens_vary = len({p["used_after"] for p in parts}) > 1
    c.require(gold_varies == prior["gold_varying"] and (not gold_varies and tokens_vary) == prior["token_only_varying"]
              and earlier_parts == prior["reward_parts"] and actual["invalid_reasons"] == prior["invalid_reasons"]
              and actual["add_counts"] == prior["add_counts"], "Archived v2 components differ")
    return {**actual, "state_id": row["state_id"], "reward_v2": prior["rewards"], "greedy_reward_v2": prior["greedy_reward"],
            "reward_parts": parts, "gold_varying": gold_varies, "nongold_varying": nongold_varies,
            "nongold_only_varying": not gold_varies and nongold_varies,
            "token_only_varying": not gold_varies and not nongold_varies and tokens_vary}


def summary(rows, seconds):
    result = summarize(rows, seconds)
    n = len(rows)
    result["nongold_varying"] = rate(sum(r["nongold_varying"] for r in rows), n)
    result["nongold_only_varying"] = rate(sum(r["nongold_only_varying"] for r in rows), n)
    return result


def rescore():
    c.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Phase P requires CPU")
    c.expect_new(NEW_C, RECEIPT_C)
    base = c.verify_model()
    state_file = c.previous_file(POOL)
    state_record = c.R2EXT / "states/state_checks.json"
    old_config = OLD_C / "sampling_probe.json"
    old_groups = OLD_C / "groups.jsonl"
    sources = {str(path): c.previous_file(path) for path in (state_record, old_config, old_groups)}
    states, prior_config = c.read(state_record), c.read(old_config)
    c.require(states["scope"] == prior_config["split"] == "train" and states["n_tasks"] == 400
              and states["state_file"] == state_file == prior_config["state_file"]
              and states["n_states"] == prior_config["n_states"] == 22494
              and states["available_checks_pass"] and states["heldout_overlap"] == states["locked_test_overlap"] == 0
              and all(states[k] == 0 for k in ("gold_mismatches", "reencode_mismatches", "rerender_mismatches"))
              and prior_config["seed"] == c.SEED and prior_config["mu"] == MU
              and prior_config["status"] == "complete", "Task 11 B/C input receipts disagree")
    task_gold, task_source = c.task_gold(c.TRAIN_TASKS, "train", 4735, c.TRAIN_SHA)
    offsets = groups.index_states(POOL)
    c.require(sum(map(len, offsets.values())) == states["n_states"], "Wrong archived state count")
    NEW_C.mkdir()
    import time
    began = time.monotonic()
    cell_rows, elapsed, seen = {}, {}, set()
    with GROUPS.open("x") as out, (NEW_C / "informative.jsonl").open("x") as subset, old_groups.open() as v2:
        for cell in groups.CELLS:
            start = time.monotonic()
            sample_path = OLD_C / (cell.replace("|", "_") + "_samples.jsonl")
            recorded = c.previous_file(sample_path)
            c.require(recorded == prior_config["raw_samples_by_cell"][cell], "C sample manifest differs")
            sources[str(sample_path)] = recorded
            scored = []
            for row, raw in zip(groups.IndexedRows(POOL, offsets[cell]), lines(sample_path), strict=True):
                c.require(row["task"] in states["task_ids"] and row["state_id"] == f"{row['trajectory_id']}:{row['step']}"
                          and row["state_id"] not in seen, "Unexpected or duplicate Task 11 train state")
                seen.add(row["state_id"])
                group = score_group(row, raw, json.loads(next(v2)), state_with_gold(row, task_gold[row["task"]]))
                line = compact(group) + "\n"
                out.write(line)
                if group["informative"]:
                    subset.write(line)
                scored.append(group)
            cell_rows[cell] = scored
            elapsed[cell] = time.monotonic() - start
            print(f"C {cell}: {len(scored)} archived groups verified", flush=True)
        c.require(not next(v2, None), "Unused archived group")
    all_rows = [row for cell in groups.CELLS for row in cell_rows[cell]]
    counts = summary(all_rows, time.monotonic() - began)
    c.require(len(all_rows) == len(seen) == states["n_states"], "Incomplete C archive coverage")
    n_samples = 8 * len(all_rows)
    old = {"pooled": prior_config["pooled"], "cells": prior_config["cells"]}
    result = {
        "status": "complete", "split": "train", "reused_task11_samples": True,
        "n_states": len(all_rows), "n_samples": n_samples, "n_informative": counts["informative"]["count"],
        "mu": MU, "non_gold_price": NONGOLD_PRICE, "informative_threshold": groups.THRESHOLD,
        "source_generation": prior_config["generation"], "state_file": state_file, "task_gold_source": task_source,
        "task11_source_files": sources, "merged_model_files": base,
        "group_file": c.file_record(GROUPS), "informative_file": c.file_record(NEW_C / "informative.jsonl"),
        "pooled": counts, "cells": {cell: summary(rows, elapsed[cell]) for cell, rows in cell_rows.items()},
        "task11_c": old,
        "parity": {"v2_reward_checks": n_samples, "v2_reward_mismatches": 0, "v3_delta_checks": n_samples,
                   "v3_delta_mismatches": 0, "greedy_checks": len(all_rows), "greedy_mismatches": 0},
        "definitions": {
            "gold_varying": "group varies in n_lost or invalid",
            "nongold_varying": "group varies in n_nongold_after, potentially alongside other terms",
            "nongold_only_varying": "the nongold count varies while n_lost and invalid agree; tokens may vary",
            "token_only_varying": "only retained token cost varies under v3; Task 11 defined this without nongold count",
            "zero_std": "population standard deviation exactly zero; informative requires > 1e-9",
        },
    }
    c.write_json_exclusive(RECEIPT_C, result)
    print(f"C pooled: zero std {counts['zero_std_groups']['count']}/{len(all_rows)}, informative {result['n_informative']}", flush=True)


def select(candidates, target=c.TARGET):
    counts = balanced.quotas({cell: len(candidates[cell]) for cell in groups.CELLS}, target)
    rng = random.Random(c.DATA_SEED)
    picked = []
    for cell in groups.CELLS:
        picked.extend(rng.sample(candidates[cell], counts[cell]))
    rng.shuffle(picked)
    return picked, counts


def make_dataset():
    c.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Phase P requires CPU")
    c.expect_new(c.DATA.parent, c.SMOKE_DATA.parent, RECEIPT_D)
    receipt = c.read(RECEIPT_C)
    c.require(receipt["status"] == "complete" and receipt["state_file"] == c.file_record(POOL)
              and receipt["group_file"] == c.file_record(GROUPS), "C identity differs")
    n = receipt["n_informative"]
    if n < c.MIN_SIGNAL:
        c.write_json_exclusive(RECEIPT_D, {"status": "stopped_insufficient_signal", "n_informative": n,
                                          "minimum_informative": c.MIN_SIGNAL})
        raise SystemExit("NEED: Task 18 D has fewer than 500 informative states")
    task_gold, task_record = c.task_gold(c.TRAIN_TASKS, "train", 4735, c.TRAIN_SHA)
    c.require(task_record == receipt["task_gold_source"], "Task gold source changed")
    prior_data = c.R2EXT / "data/train.jsonl"
    prior_data_receipt = c.previous_file(prior_data)
    prior_ids = {f"{r['trajectory_id']}:{r['state_step']}" for r in lines(prior_data)}
    c.require(len(prior_ids) == 2000, "Task 11 D coverage differs")
    informative = {r["state_id"]: r for r in lines(NEW_C / "informative.jsonl")}
    c.require(len(informative) == n and all(r["population_std"] > groups.THRESHOLD for r in informative.values())
              and receipt["informative_file"] == c.file_record(NEW_C / "informative.jsonl"), "C subset differs")
    candidates = {cell: [] for cell in groups.CELLS}
    with POOL.open("rb") as inputs:
        while True:
            offset = inputs.tell()
            line = inputs.readline()
            if not line:
                break
            row = json.loads(line)
            identity = row["state_id"]
            if identity in informative:
                archived = informative[identity]
                c.require(all(archived[k] == row[k] for k in COLUMNS), "C/B state mismatch at D")
                candidates[groups.cell(row)].append((identity, offset))
    c.require(sum(map(len, candidates.values())) == n, "Informative states missing from B")
    picked, counts = select(candidates)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(c.BASE / "tokenizer.json"))
    orm = CtxStepRewardV3(SimpleNamespace(backend_tokenizer=tok))
    strata = Counter()
    per_cell = {cell: Counter() for cell in groups.CELLS}
    smoke_ids = []
    c.DATA.parent.mkdir()
    c.SMOKE_DATA.parent.mkdir()
    with POOL.open("rb") as inputs, c.DATA.open("x") as out, c.SMOKE_DATA.open("x") as smoke:
        for i, (identity, offset) in enumerate(picked):
            inputs.seek(offset)
            row = json.loads(inputs.readline())
            sample = informative[identity]
            source = row["messages"]
            c.require(row["split"] == "train" and row["state_id"] == identity and
                      source == [{"role": "user", "content": row["user_text"]}], "Training prompt differs from B")
            state = state_with_gold(row, task_gold[row["task"]])
            c.require(state["prompt_token_ids_tail"] == row["prompt_token_ids"][-24:], "Training prompt suffix differs")
            packed = compact(state)
            scores = orm([row["greedy_output"]], [packed], [row["greedy_output_token_ids"]])
            c.require(scores == [sample["greedy_reward"]], "D plugin greedy reward differs from C")
            exported = {"messages": source, "reward_state_json": packed, "trajectory_id": row["trajectory_id"],
                        "state_step": row["step"], "cell": sample["cell"]}
            encoded = compact(exported) + "\n"
            c.require(json.loads(encoded)["messages"][0]["content"].encode() == row["user_text"].encode(),
                      "D serialized user text differs")
            out.write(encoded)
            if i < 8:
                smoke.write(encoded)
                smoke_ids.append(identity)
            label = ("gold_varying" if sample["gold_varying"] else "nongold_only_varying"
                     if sample["nongold_varying"] else "token_only_varying")
            c.require(label != "token_only_varying" or sample["token_only_varying"], "Unclassified informative state")
            strata[label] += 1
            per_cell[sample["cell"]][label] += 1
    c.require(len(picked) == min(n, c.TARGET) and len(smoke_ids) == 8, "D coverage differs")
    keys = ("gold_varying", "nongold_only_varying", "token_only_varying")
    c.write_json_exclusive(RECEIPT_D, {
        "status": "complete", "n": len(picked), "n_informative": n, "target_n": c.TARGET,
        "per_cell_counts": counts, "per_cell_available": {cell: len(rows) for cell, rows in candidates.items()},
        "composition": {k: rate(strata[k], len(picked)) for k in keys},
        "per_cell_composition": {cell: {k: rate(v[k], counts[cell]) for k in keys} for cell, v in per_cell.items()},
        "seed": c.DATA_SEED, "heldout_rows": 0, "user_text_mismatches": 0, "plugin_greedy_mismatches": 0,
        "rows_checked": len(picked), "plugin_greedy_checked": len(picked),
        "same_state_id_as_task11_D": rate(sum(identity in prior_ids for identity, _ in picked), len(picked)),
        "source_task11_dataset": prior_data_receipt, "source_states": receipt["state_file"],
        "source_groups": receipt["group_file"], "source_informative": receipt["informative_file"],
        "output": c.file_record(c.DATA), "smoke_data": c.file_record(c.SMOKE_DATA), "smoke_state_ids": smoke_ids,
    })
    print(f"D selected {len(picked)} from {n} informative states; cell counts {counts}; {dict(strata)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("rescore", "dataset", "prepare"))
    args = parser.parse_args()
    if args.action in ("rescore", "prepare"):
        rescore()
    if args.action in ("dataset", "prepare"):
        make_dataset()


if __name__ == "__main__":
    main()

