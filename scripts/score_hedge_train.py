#!/usr/bin/env python3
"""Score the frozen Task 08 training population and select the Task 09 threshold."""
from collections import Counter
from itertools import islice
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.build_formal_grpo_states import gpu_gate
from scripts.hedge_action import TAU_GRID, candidate_ids, execute_hedge
from scripts.heldout_eval import compact, file_record, write_json
from scripts.prepare_formal_grpo import ADAPTER, BASE, SHA, check_start

EXT = Path("/root/userfolder/ctx-compressor")
STATES = EXT / "grpo_v2/states/states.jsonl"
TRAIN = EXT / "data/traj_train_v2/train_tasks.jsonl"
RAW = EXT / "hedge_v2/train/state_scores.jsonl"
RESULT = PROJECT / "eval/hedge_v2/train_selection.json"
CELLS = tuple(sorted(f"{m}|{t}" for m in ("append", "reselect") for t in ("tight", "mid", "loose")))
STATE_SHA = "33fdea6942278c99a815fc931f1be7354c1ace3ecb78a486e58baafc716e4eb8"
CHECKS_SHA = "216e77f5edbdf6f7bb1af2aff2a25ee24771063089e86e9215f84af58ffd3b42"


def train_inputs():
    check_start()
    receipt_path = STATES.with_name("state_checks.json")
    if file_record(receipt_path)["sha256"] != CHECKS_SHA:
        raise ValueError("Task 08 train-state receipt hash changed")
    proof = json.loads(receipt_path.read_bytes())
    state_file = file_record(STATES)
    if (state_file["sha256"] != STATE_SHA or proof["state_file"] != state_file
            or proof["scope"] != "train" or not proof["available_checks_pass"]
            or proof["heldout_overlap"] != 0 or proof["n_states"] != 22158
            or proof["n_trajectories"] != 2400 or proof["n_tasks"] != 400
            or proof["source_adapter_sha256"] != SHA or proof["base_model"] != str(BASE)
            or proof["inference_adapter"] != str(ADAPTER)):
        raise ValueError("Frozen state provenance or population changed")
    selected = set(proof["task_ids"])
    gold_pids = {}
    with TRAIN.open("rb") as stream:
        for line in stream:
            task = json.loads(line)
            if task["split"] != "train":
                raise ValueError("Non-training task in train task file")
            key = task["task"]
            if key in selected:
                if key in gold_pids:
                    raise ValueError(f"Duplicate train task {key}")
                gold_pids[key] = [step["gold"] for step in task["steps"]]
    if set(gold_pids) != selected or len(selected) != 400:
        raise ValueError("Selected training tasks are incomplete")
    return state_file, gold_pids


def local_current_gold(row, state, gold_pids):
    """Return the local ID if the current step's gold is newly available."""
    global_id = gold_pids[row["task"]][row["step"]]
    mapping = json.loads(row["pid_by_lid_json"])
    candidates = [i for i in candidate_ids(state) if mapping[str(i)] == global_id]
    if len(candidates) > 1 or (candidates and candidates[0] not in state["gold"]):
        raise ValueError("Local gold IDs do not match the frozen train task")
    return candidates[0] if candidates else None


class Aggregate:
    def __init__(self):
        self.counts = Counter()
        self.rewards = []

    def add(self, evaluated, gold, current):
        counts = self.counts
        self.rewards.append(evaluated["reward"])
        counts["hedged"] += evaluated["hedged"]
        counts["hit"] += evaluated["hedged"] and evaluated["extra_id"] in gold
        counts["current_hit"] += evaluated["hedged"] and evaluated["extra_id"] == current
        counts["extra_tokens"] += evaluated["extra_tokens"]
        if current is not None:
            counts["current_gold_new"] += 1
            counts["miss"] += current not in evaluated["keep_after"]

    def result(self):
        c = self.counts
        n = len(self.rewards)
        if not n:
            raise ValueError("No train states in cell")
        def rate(k, denominator):
            return {"count": c[k], "n": denominator, "rate": c[k] / denominator if denominator else None}
        total_reward = math.fsum(self.rewards)
        return {"n": n, "hedge_rate": rate("hedged", n),
                "hedge_hit_rate": rate("hit", c["hedged"]),
                "hedge_current_gold_hit_rate": rate("current_hit", c["hedged"]),
                "current_gold_miss_rate": rate("miss", c["current_gold_new"]),
                "mean_extra_tokens": {"sum": c["extra_tokens"], "n": n, "mean": c["extra_tokens"] / n},
                "mean_reward": {"sum": total_reward, "n": n, "mean": total_reward / n}}


def select_tau(pooled):
    if set(pooled) != {str(t) for t in TAU_GRID}:
        raise ValueError("Incomplete tau grid")
    return max(TAU_GRID, key=lambda t: pooled[str(t)]["mean_reward"]["mean"])


def run():
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Train-state scoring requires the fixed 5880 project and GPU 1")
    if RESULT.exists() or RAW.exists():
        raise FileExistsError("Train-state output exists")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state_file, gold_pids = train_inputs()
    from scripts.hedge_scorer import create_engine, score_candidates
    gpu_gate(1)
    llm, adapter, engine = create_engine()
    try:
        score_population(llm, adapter, engine, state_file, gold_pids, started, score_candidates)
    finally:
        llm.llm_engine.engine_core.shutdown()


def score_population(llm, adapter, engine, state_file, gold_pids, started, score_candidates):
    tables = {s: {tau: Aggregate() for tau in TAU_GRID} for s in ("pooled", *CELLS)}
    seen = set()
    trajectory_keys = set()
    cell_counts = Counter()
    with STATES.open("rb") as source, RAW.open("x", encoding="utf-8") as sink:
        batches = iter(lambda: list(islice(source, 16)), [])
        for batch_index, lines in enumerate(batches, 1):
            batch = [json.loads(line) for line in lines]
            states = []
            for row in batch:
                cell = f"{row['mode']}|{row['budget_tier']}"
                key = row["trajectory_id"], row["step"]
                if (row["split"] != "train" or cell not in CELLS or row["task"] not in gold_pids
                        or key in seen or row["source_adapter_sha256"] != SHA
                        or not 0 <= row["step"] < len(gold_pids[row["task"]])
                        or row["trajectory_id"] != f"checkpoint-2000|{cell}|{row['task']}"):
                    raise ValueError(f"Invalid train identity {key}")
                state = json.loads(row["reward_state_json"])
                if state["mode"] != row["mode"] or state["budget"] != row["budget"]:
                    raise ValueError(f"State metadata and prompt disagree for {key}")
                seen.add(key)
                trajectory_keys.add(key[0])
                cell_counts[cell] += 1
                states.append(state)
            scores = score_candidates(llm, adapter, [r["prompt_token_ids"] for r in batch],
                                      [candidate_ids(s) for s in states])
            if len(scores) != len(batch):
                raise ValueError("Scorer returned a partial batch")
            for row, state, ranked_scores in zip(batch, states, scores):
                cell = f"{row['mode']}|{row['budget_tier']}"
                current = local_current_gold(row, state, gold_pids)
                details = {}
                for tau in TAU_GRID:
                    result = execute_hedge(state, row["greedy_output"], ranked_scores, tau)
                    if tau == 0:
                        if (result["keep_after"] != row["greedy_keep_after"]
                                or result["invalid_reason"] != row["greedy_invalid_reason"] or result["hedged"]):
                            raise ValueError("Frozen greedy replay mismatch")
                    tables["pooled"][tau].add(result, state["gold"], current)
                    tables[cell][tau].add(result, state["gold"], current)
                    details[str(tau)] = {key: result[key] for key in
                        ("raw_output", "extra_id", "extra_tokens", "hedged", "keep_after", "reward")}
                record = {k: row[k] for k in ("split","task","trajectory_id","step","prompt_sha256")}
                sink.write(compact({**record, "cell": cell, "current_gold_new_lid": current,
                    "scores": ranked_scores, "p1": result["p1"], "greedy_raw": row["greedy_output"],
                    "grid_results": details}) + "\n")
            if batch_index == 1 or batch_index % 20 == 0:
                sink.flush()
                print(f"train-only scored states: {len(seen)}/22158", flush=True)
    if (len(seen) != 22158 or len(trajectory_keys) != 2400
            or cell_counts != {c: 3693 for c in CELLS}
            or sum(map(len, gold_pids.values())) * len(CELLS) != len(seen)):
        raise ValueError("Full unique train-state coverage failed")
    summary = {s: {str(t): a.result() for t,a in table.items()} for s,table in tables.items()}
    pooled = summary.pop("pooled")
    selected = select_tau(pooled)
    write_json(RESULT, {"status": "complete", "split": "train", "n_tasks": len(gold_pids),
        "n_states": len(seen), "n_trajectories": len(trajectory_keys), "heldout_overlap": 0,
        "heldout_overlap_source": str(STATES.with_name("state_checks.json")), "grid": TAU_GRID,
        "tau_star": selected, "selection": "highest pooled mean step reward; ties choose smaller tau",
        "lambda": 0.3, "metrics": {"hedge_hit": "extra is any previously revealed or current gold",
            "current_gold_miss": "current gold newly available but absent after the executed action, conditional on newly available",
            "mean_extra_tokens": "tokens added to greedy's keep_after, averaged over all states",
            "score_ties": "equal log scores retain prompt candidate order"},
        "pooled": pooled, "cells": summary, "state_file": state_file, "train_task_file": file_record(TRAIN),
        "score_file": file_record(RAW), "base_model": str(BASE), "adapter": str(ADAPTER),
        "adapter_sha256": SHA, "engine_parameters": engine, "elapsed_seconds": time.monotonic()-started})
    print(f"selected tau={selected} from {len(seen)} train states", flush=True)


if __name__ == "__main__":
    run()

