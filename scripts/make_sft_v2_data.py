#!/usr/bin/env python3
"""Build Task 06 from the saved Task 04 training and held-out retrievals on CPU."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.make_sft_data import bounded_results, initialize_worker
from scripts.make_train_traj import (EXTERNAL, JsonlWriter, compact, file_record,
                                     ratio, utc, write_json)
from src.build_traj import rollout
from src.oracle_noise_v2 import make_oracle_noise_policy


DATA = EXTERNAL / "data"
ROWS = {"train": 88562, "heldout": 4652}
STEP_INPUT_FIELDS = ("questions", "mode", "budget", "kept_ids", "new_ids",
                     "ntok", "text", "gold", "pid_by_lid", "sft_target")


def json_line(record):
    return (compact(record) + "\n").encode("utf-8")


def require(passed, message):
    if not passed:
        raise ValueError(message)


def measure(counters, states):
    for state in states[1:]:
        drop = state["sft_target"]["drop"]
        kept = state["kept_ids"]
        counters["n"] += 1
        counters["drop_nonempty"] += bool(drop)
        counters["drop_count"] += len(drop)
        counters["kept_count"] += len(kept)
        counters["all_gold"] += set(kept).issubset(state["gold"])


def summary(counters):
    n = counters["n"]
    return {
        "target_drop_nonempty": ratio(counters["drop_nonempty"], n),
        "target_drop_count": {"sum": counters["drop_count"], "n": n, "mean": counters["drop_count"] / n},
        "kept_fragments": {"sum": counters["kept_count"], "n": n, "mean": counters["kept_count"] / n},
        "kept_entirely_gold": ratio(counters["all_gold"], n),
    }


def step0_check(old, new, returned):
    # JSON serialization normalizes the int keys in memory to v1's on-disk keys.
    for state in (old, new):
        require(state["kept_ids"] == [], "nonempty initial state")
        mapping = {int(k): v for k, v in state["pid_by_lid"].items()}
        require([mapping[i] for i in state["new_ids"]] == returned,
                "step-0 passage order differs from the recorded retrieval")
    require(json_line({k: old[k] for k in STEP_INPUT_FIELDS}) ==
            json_line({k: new[k] for k in STEP_INPUT_FIELDS}), "step-0 state mismatch")


def make_trajectories(split, corpus, expected, counts):
    old_path = DATA / "traj_train" / f"{split}_tasks.jsonl"
    new_path = DATA / "traj_train_v2" / old_path.name
    digest = hashlib.sha256()
    retrieval_hash = hashlib.sha256()
    metrics = {"v1": defaultdict(Counter), "v2": defaultdict(Counter)}
    with old_path.open("rb") as source, JsonlWriter(new_path) as writer:
        for index, line in enumerate(source):
            digest.update(line)
            old = json.loads(line)
            require(old["split"] == split and old["task"] == old["noise_seed"] == index,
                    f"unexpected task or seed in {split} {index}")
            args = (old["questions"], old["steps"], corpus, old["budget"], "reselect")
            legacy = rollout(*args, policy=make_oracle_noise_policy(index))
            rolls = old["rollouts"]
            require(json_line({**old, "rollouts": {**rolls, "reselect": legacy}}) == line,
                    f"default policy failed byte replay at {split} {index}")
            counts["default_replay_tasks_byte_identical"] += 1
            updated = rollout(*args, policy=make_oracle_noise_policy(index, (0, 2)))
            new = {**old, "rollouts": {**rolls, "reselect": updated}}
            for mode in ("reselect", "append"):
                old_states, new_states = rolls[mode], new["rollouts"][mode]
                step0_check(old_states[0], new_states[0], old["steps"][0]["new"])
                counts["step0_states_identical"] += 1
                cohort = f"{mode}|{old['tier']}"
                for version, states in (("v1", old_states), ("v2", new_states)):
                    measure(metrics[version][cohort], states)
            retrieval_hash.update(json_line(old["steps"]))
            writer.write(new)
            if (index + 1) % 1000 == 0:
                print(f"{split} trajectories {index + 1}", flush=True)
        require(digest.hexdigest() == expected["sha256"] and writer.lines == expected["lines"],
                "frozen v1 trajectory input checksum or count mismatch")
    for version in metrics.values():
        for mode in ("reselect", "append"):
            version[f"{mode}|all"] = sum(
                (version[f"{mode}|{tier}"] for tier in ("tight", "mid", "loose")), Counter())
    statistics = {version: {cohort: summary(items) for cohort, items in groups.items()}
                  for version, groups in metrics.items()}
    return writer.record(), retrieval_hash.hexdigest(), statistics


def make_sft(split, trajectory, original, retrieval_sha, pool, workers, counts):
    digest = hashlib.sha256()
    steps_hash = hashlib.sha256()
    traj_hash = hashlib.sha256()
    maximum = 0

    def tasks():
        with Path(trajectory["path"]).open("rb") as stream:
            for line in stream:
                traj_hash.update(line)
                task = json.loads(line)
                steps_hash.update(json_line(task["steps"]))
                yield task, False

    # The frozen encoder regenerates every row, including the append and step-0
    # rows, making the byte comparisons independent of the archived SFT bytes.
    path = DATA / "sft_v2" / f"{split}.jsonl"
    with Path(original["path"]).open("rb") as old, JsonlWriter(path) as writer:
        for index, (rows, _) in enumerate(bounded_results(pool, tasks(), 2 * workers)):
            for row in rows:
                line = old.readline()
                digest.update(line)
                meta = row["meta"]
                if meta["mode"] == "append" or meta["step"] == 0:
                    require(json_line(row) == line, f"row differs at {split} {meta}")
                    key = ("append_rows_byte_identical" if meta["mode"] == "append"
                           else "reselect_step0_rows_byte_identical")
                    counts[key] += 1
                maximum = max(maximum, meta["total_tokens"])
                writer.write(row)
            if (index + 1) % 500 == 0:
                print(f"{split} SFT {index + 1} tasks, {writer.lines} rows", flush=True)
        require(not old.readline() and digest.hexdigest() == original["sha256"],
                "v1 SFT source count or SHA mismatch")
        require(writer.lines == ROWS[split] == original["lines"], "v1/v2 SFT row count mismatch")
        require(trajectory["sha256"] == traj_hash.hexdigest(), "v2 trajectory changed during encoding")
        require(steps_hash.hexdigest() == retrieval_sha, "saved v2 retrieval differs from v1")
    counts["sft_rows_v1"] = counts["sft_rows_v2"] = writer.lines
    return writer.record(), maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    require(PROJECT == Path("/root/workplace/ctx-compressor") and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "" and
            Path(sys.prefix) == EXTERNAL / "envs/train", "use existing 5880 train environment with GPU hidden")
    started, clock = utc(), time.monotonic()
    stats_path = PROJECT / "eval/sft_run_v2/data_v2_stats.json"
    targets = [stats_path, *(DATA / "sft_v2" / f"{s}.jsonl" for s in ROWS),
               *(DATA / "traj_train_v2" / f"{s}_tasks.jsonl" for s in ROWS)]
    for target in targets:
        require(not target.exists() and not target.with_suffix(target.suffix + ".partial").exists(),
                f"output already exists: {target}")
    original_traj = json.loads((PROJECT / "eval/sft_prep/trajectory_stats.json").read_text())["files"]
    original_sft = json.loads((PROJECT / "eval/sft_prep/sft_data_stats.json").read_text())["files"]
    task05 = json.loads((PROJECT / "eval/sft_run/cpu_baseline_results.json").read_text())["run"]["input_files"]
    for record in task05:
        key = {"heldout_tasks.jsonl": "heldout", "corpus_train.jsonl": "corpus"}[Path(record["path"]).name]
        require(record["sha256"] == original_traj[key]["sha256"], "Task 04/05 source SHA disagreement")
    corpus_path = DATA / "traj_train/corpus_train.jsonl"
    with corpus_path.open("rb") as stream:
        corpus = [json.loads(line) for line in stream]
    corpus_record = file_record(corpus_path, len(corpus))
    require(corpus_record == original_traj["corpus"] and
            all(i == p["pid"] for i, p in enumerate(corpus)), "frozen training corpus mismatch")
    checks = {split: Counter() for split in ROWS}
    trajectories, retrievals, metrics = {}, {}, {}
    for split in ROWS:
        trajectories[split], retrievals[split], metrics[split] = make_trajectories(
            split, corpus, original_traj[split], checks[split])
        require(checks[split]["default_replay_tasks_byte_identical"] >= 200,
                f"insufficient default replays for {split}")
    del corpus
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    files, max_tokens = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker,
                             initargs=(str(EXTERNAL / "models/Qwen3-1.7B"),)) as pool:
        for split in ROWS:
            files[split], max_tokens[split] = make_sft(split, trajectories[split], original_sft[split],
                                                      retrievals[split], pool, args.workers, checks[split])
    report = {
        "schema_version": 1, "started_utc": started, "finished_utc": utc(),
        "seconds": time.monotonic() - clock, "cpu_only": True, "workers": args.workers,
        "noise_range_v1": [1, 3], "noise_range_v2": [0, 2], "noise_seed": "split-local task index",
        "files": files, "trajectory_files": trajectories, "inputs": {
            "sft": original_sft, "trajectories": {split: original_traj[split] for split in ROWS},
            "corpus": corpus_record, "task05_eval_inputs": task05},
        "statistics": metrics, "checks": checks, "all_checks_passed": True,
        "step_scope": "steps >= 1; kept before action; empty kept is a subset of gold",
        "shared_retrieval_steps_sha256": retrievals,
        "retrieval_hash_encoding": "one compact JSON array of saved steps plus LF per task, in source order; v1 and v2 compared",
        "max_total_tokens": max_tokens,
    }
    write_json(stats_path, report)
    print(compact({"completed": True, "checks": checks, "files": files}), flush=True)


if __name__ == "__main__":
    main()
