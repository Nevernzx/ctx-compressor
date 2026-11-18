#!/usr/bin/env python3
"""TASK_04: stream CPU train/heldout trajectories, using only existing assets.

Each JSONL line is one task with shared retrieval steps and two rollout lists.
JSON object keys in state.ntok, state.text and state.pid_by_lid must be restored
to integers before calling build_traj functions (see restore_state_keys).
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
import resource
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
SEED = 20261118
BUDGETS = {"tight": 1876, "mid": 2814, "loose": 5628}
MODES = ("reselect", "append")
STATE_INTEGER_KEY_FIELDS = ("ntok", "text", "pid_by_lid")
sys.path.insert(0, str(PROJECT))

from src.build_traj import (apply_action, build_corpus, hop_queries,
                            make_oracle_noise_policy, oracle_action, rollout,
                            sample_tasks, simulate_task)


def utc():
    return datetime.now(timezone.utc).isoformat()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_record(path, lines=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "lines": lines, "sha256": digest.hexdigest()}


class JsonlWriter:
    """Only a single serialized task is in memory; hash the exact written bytes."""

    def __init__(self, path):
        self.path = path
        self.partial = path.with_suffix(path.suffix + ".partial")
        self.lines = 0
        self.digest = hashlib.sha256()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.stream = self.partial.open("xb")
        return self

    def write(self, value):
        data = (compact(value) + "\n").encode("utf-8")
        self.stream.write(data)
        self.digest.update(data)
        self.lines += 1

    def flush(self):
        self.stream.flush()

    def __exit__(self, exc_type, exc, tb):
        self.stream.flush()
        if exc_type is None:
            os.fsync(self.stream.fileno())
        self.stream.close()
        if exc_type is None:
            self.partial.rename(self.path)

    def record(self):
        return {"path": str(self.path), "bytes": self.path.stat().st_size,
                "lines": self.lines, "sha256": self.digest.hexdigest()}


def restore_state_keys(state):
    """Return a shallow copy with the three JSON-stringified ID maps restored."""
    return {**state, **{field: {int(k): v for k, v in state[field].items()}
                       for field in STATE_INTEGER_KEY_FIELDS}}


def split_examples(examples, seed=SEED):
    """Sample heldout IDs, then retain source order within both input pools."""
    ids = [ex["id"] for ex in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate example ids")
    heldout_ids = random.Random(seed).sample(ids, len(ids) // 20)
    heldout_set = set(heldout_ids)
    partitions = {
        "train": [ex for ex in examples if ex["id"] not in heldout_set],
        "heldout": [ex for ex in examples if ex["id"] in heldout_set],
    }
    tasks, details = {}, {}
    for split, pool in partitions.items():
        tasks[split] = sample_tasks(pool, len(pool) // 4, seed=seed)
        selected = {ex["id"] for task in tasks[split] for ex in task}
        details[split] = {
            "pool_questions": len(pool), "tasks": len(tasks[split]),
            "used_questions": len(selected),
            "discarded_ids": [ex["id"] for ex in pool if ex["id"] not in selected],
            "discarded_questions": len(pool) - len(selected),
        }
    metadata = {
        "schema_version": 1, "seed": seed, "heldout_fraction": 0.05,
        "heldout_rounding": "floor (len(valid_examples) // 20)",
        "valid_questions": len(ids), "heldout_ids": heldout_ids,
        "heldout_ids_order": "random.Random(seed).sample draw order",
        "train_ids": [ex["id"] for ex in partitions["train"]],
        "sample_tasks_input_order": "original dataset order for each partition",
        "sample_tasks_seed": seed, "questions_per_task": 4,
        "partitions": details,
    }
    return tasks, metadata


def filter_examples(examples, key2pid):
    valid, rejected = [], []
    literal_examples, literal_occurrences = [], 0
    for ex in examples:
        try:
            hop_queries(ex, key2pid)
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"id": ex.get("id"), "reason": str(error)})
            continue
        valid.append(ex)
        occurrences = []
        for hop_index, hop in enumerate(ex["question_decomposition"]):
            for match in re.finditer(r"#(\d+)", hop["question"]):
                if not 1 <= int(match.group(1)) <= hop_index:
                    occurrences.append({"hop_index": hop_index,
                                        "reference": match.group(0),
                                        "question": hop["question"]})
        if occurrences:
            literal_occurrences += len(occurrences)
            literal_examples.append({"id": ex["id"], "occurrences": occurrences})
    return valid, {
        "total": len(examples), "valid": len(valid), "rejected": len(rejected),
        "rejected_examples": rejected,
        "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
        "literal_out_of_range_reference_count": literal_occurrences,
        "literal_out_of_range_reference_examples": literal_examples,
    }


def make_task(task_index, examples, split, corpus, index, key2pid, seed=SEED):
    tier = random.Random(seed + task_index).choice(list(BUDGETS))
    budget = BUDGETS[tier]
    steps = simulate_task(examples, index, key2pid, top_n=20, seed=task_index)
    questions = [ex["question"] for ex in examples]
    records = {}
    for mode in MODES:
        policy = make_oracle_noise_policy(task_index) if mode == "reselect" else oracle_action
        records[mode] = rollout(questions, steps, corpus, budget, mode, policy=policy)
    return {
        "schema_version": 1, "split": split, "task": task_index,
        "example_ids": [ex["id"] for ex in examples], "questions": questions,
        "tier": tier, "budget": budget, "retrieval_seed": task_index,
        "noise_seed": task_index, "steps": steps, "rollouts": records,
    }


def ratio(numerator, denominator):
    return {"count": numerator, "n": denominator,
            "rate": numerator / denominator if denominator else None}


def update_statistics(task, counters):
    """Validate the real trajectories and collect bounded-size scalar counts."""
    retrieval = counters["retrieval"]
    retrieval["tasks"] += 1
    for step in task["steps"]:
        if len(step["new"]) != 20 or len(set(step["new"])) != 20 or step["gold"] not in step["new"]:
            raise AssertionError("retrieval must contain 20 unique paragraphs including gold")
        retrieval["steps"] += 1
        retrieval["natural_hits"] += int(step["natural_hit"])
        retrieval[f"hop_{step['hop_index'] + 1}_steps"] += 1
        retrieval[f"hop_{step['hop_index'] + 1}_natural_hits"] += int(step["natural_hit"])
    for mode, states in task["rollouts"].items():
        if len(states) != len(task["steps"]):
            raise AssertionError("rollout does not match retrieval steps")
        c = counters[f"{mode}|{task['tier']}"]
        c["tasks"] += 1
        previous = []
        for state in states:
            target_keep = apply_action(state, state["sft_target"])
            if target_keep is None or state["sft_target"] != oracle_action(state):
                raise AssertionError("SFT target must be the legal pure oracle")
            if not state["action_valid"] or state["used_after"] > state["budget"]:
                raise AssertionError("invalid or over-budget rollout action")
            if state["kept_ids"] != previous:
                raise AssertionError("closed-loop state was not carried forward")
            if mode == "append" and (state["action"] != state["sft_target"] or
                                     state["keep_after"][:len(previous)] != previous):
                raise AssertionError("append must use pure oracle and preserve prefix")
            previous = state["keep_after"]
            c["samples"] += 1
            c["target_add_empty"] += not bool(state["sft_target"]["add"])
            c["target_drop_nonempty"] += bool(state["sft_target"]["drop"])
            c["budget_binding"] += not set(state["gold"]).issubset(target_keep)
            c["available_gold"] += len(state["gold"])
            c["target_retained_gold"] += len(set(state["gold"]) & set(target_keep))
            c["noisy_actions"] += state["action"] != state["sft_target"]


def summarize_statistics(counters):
    r = counters["retrieval"]
    return {
        "tasks": r["tasks"], "retrieval_steps": r["steps"],
        "samples": 2 * r["steps"],
        "natural_hit": ratio(r["natural_hits"], r["steps"]),
        "natural_hit_by_hop": {
            str(hop): ratio(r[f"hop_{hop}_natural_hits"], r[f"hop_{hop}_steps"])
            for hop in range(1, 5)},
        "mode_tier": {f"{mode}|{tier}": {
            **dict(counters[f"{mode}|{tier}"]),
            **{name: ratio(counters[f"{mode}|{tier}"][name],
                           counters[f"{mode}|{tier}"]["samples"])
               for name in ("target_add_empty", "target_drop_nonempty", "budget_binding")}
        } for mode in MODES for tier in BUDGETS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=EXTERNAL / "data/traj_train")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if PROJECT != Path("/root/workplace/ctx-compressor"):
        parser.error("execute only on 5880 in /root/workplace/ctx-compressor")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("source reader_environment.sh, then set CUDA_VISIBLE_DEVICES='' for CPU execution")
    if not args.data_dir.resolve().is_relative_to(EXTERNAL) or args.progress_every <= 0:
        parser.error("data-dir must be inside userfolder and progress-every must be positive")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    args.data_dir = args.data_dir.resolve()
    stats_path = PROJECT / "eval/sft_prep/trajectory_stats.json"
    split_path = PROJECT / "data/splits/train_heldout_ids.json"
    external_split = EXTERNAL / "data/splits/train_heldout_ids.json"
    if any(p.exists() for p in (args.data_dir, stats_path, split_path, external_split)):
        parser.error("refusing to overwrite existing trajectories, splits, or statistics")
    args.data_dir.mkdir(parents=True)
    started, started_utc = time.monotonic(), utc()
    timings, outputs = {}, {}
    train_path = PROJECT / "data/raw/musique/musique_ans_v1.0_train.jsonl"
    tokenizer_path = PROJECT / "data/raw/qwen3_tokenizer/tokenizer.json"
    print(f"{utc()} building full train corpus; CPU only, seed={SEED}", flush=True)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with train_path.open(encoding="utf-8") as stream:
        examples = [json.loads(line) for line in stream if line.strip()]
    stage = time.monotonic()
    corpus, index, key2pid = build_corpus(
        examples, lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))
    timings["build_corpus_seconds"] = time.monotonic() - stage
    valid, filtering = filter_examples(examples, key2pid)
    # The repaired song title must not silently remain rejected under old code.
    if any("unresolved_reference" in item["reason"] for item in filtering["rejected_examples"]):
        raise RuntimeError("hop_queries literal-reference fix is required before generation")
    tasks, split_metadata = split_examples(valid)
    split_metadata["filtering"] = filtering
    write_json(external_split, split_metadata)
    write_json(split_path, split_metadata)
    outputs["split_ids"] = file_record(external_split)
    stage = time.monotonic()
    with JsonlWriter(args.data_dir / "corpus_train.jsonl") as writer:
        for pid, paragraph in enumerate(corpus):
            writer.write({"pid": pid, **paragraph})
    outputs["corpus"] = writer.record()
    index_dir = args.data_dir / "bm25_train"
    index.save(str(index_dir), show_progress=False)
    outputs["index"] = [file_record(p) for p in sorted(index_dir.rglob("*")) if p.is_file()]
    timings["save_corpus_and_index_seconds"] = time.monotonic() - stage
    print(f"{utc()} corpus={len(corpus)} valid={len(valid)} rejected={len(examples)-len(valid)} "
          f"tasks={ {split: len(items) for split, items in tasks.items()} }", flush=True)
    statistics = {}
    for split, selected in tasks.items():
        stage = time.monotonic()
        counters = defaultdict(Counter)
        with JsonlWriter(args.data_dir / f"{split}_tasks.jsonl") as writer:
            for task_index, group in enumerate(selected):
                task = make_task(task_index, group, split, corpus, index, key2pid)
                update_statistics(task, counters)
                writer.write(task)
                del task
                if (task_index + 1) % args.progress_every == 0 or task_index + 1 == len(selected):
                    writer.flush()
                    elapsed = time.monotonic() - stage
                    print(f"{utc()} {split} {task_index+1}/{len(selected)} tasks; "
                          f"states={2*counters['retrieval']['steps']}; "
                          f"elapsed={elapsed:.1f}s; s/task={elapsed/(task_index+1):.3f}", flush=True)
        outputs[split] = writer.record()
        statistics[split] = summarize_statistics(counters)
        timings[f"{split}_generation_seconds"] = time.monotonic() - stage
    manifest = {
        "schema_version": 1, "started_at_utc": started_utc, "finished_at_utc": utc(),
        "elapsed_seconds": time.monotonic() - started, "timings_seconds": timings,
        "settings": {"seed": SEED, "top_n": 20, "questions_per_task": 4,
                     "budgets": BUDGETS, "task_index_base": 0,
                     "retrieval_seed": "task index, reset to zero within each split",
                     "tier": "random.Random(20261118 + task).choice(['tight','mid','loose'])",
                     "reselect_policy": "make_oracle_noise_policy(seed=task)",
                     "append_policy": "oracle_action", "sft_target": "oracle_action in both modes"},
        "schema": {"task_record": "one task per JSONL line; shared steps; rollouts[mode] contains full states",
                   "integer_key_state_fields": list(STATE_INTEGER_KEY_FIELDS),
                   "restore_integer_keys": "{field: {int(k): v for k, v in state[field].items()}}",
                   "corpus_key2pid": "{(row['title'], row['paragraph_text']): row['pid']} from corpus_train.jsonl",
                   "binary_file_lines": "null means not a JSONL record file"},
        "source": {"train": file_record(train_path, len(examples)),
                   "tokenizer": file_record(tokenizer_path),
                   "build_traj": file_record(PROJECT / "src/build_traj.py"),
                   "generator": file_record(Path(__file__).resolve())},
        "corpus": {"paragraphs": len(corpus), "built_from_all_train_questions": len(examples),
                   "includes_heldout_paragraphs": True},
        "filtering": filtering, "partitions": split_metadata["partitions"],
        "statistics": statistics, "files": outputs,
        "validation": {"all_states_checked": True, "pure_oracle_targets": True,
                       "legal_targets_and_actions": True, "append_prefix_preserved": True,
                       "retrieval_shared_between_modes": True,
                       "no_downloads_installs_api_or_gpu": True, "dev_inputs_not_read": True},
        "runtime": {"python": sys.version, "cpu_processes": 1,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "maximum_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "packages": {p: version(p) for p in ("bm25s", "numpy", "tokenizers")}},
    }
    write_json(args.data_dir / "manifest.json", manifest)
    write_json(stats_path, manifest)
    print(compact({"completed": True, "statistics": statistics,
                   "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
