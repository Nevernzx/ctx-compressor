#!/usr/bin/env python3
"""Run the frozen Task 07 train-state generator on the formal v2 inputs."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import os
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_grpo_states as old

ROOT = old.EXTERNAL / "grpo_v2"
TRAIN = old.EXTERNAL / "data/traj_train_v2/train_tasks.jsonl"
HELDOUT = TRAIN.with_name("heldout_tasks.jsonl")
FIXTURE = old.PREP / "step0_fixture/prompts.jsonl"
LABEL = "SFT v2 checkpoint-2000"
COLUMNS = ("split", "trajectory_id", "task", "mode", "budget_tier", "step", "prompt_sha256")


def formal_path(path):
    resolved = Path(path).resolve()
    if ROOT not in resolved.parents:
        raise ValueError(f"Use a path below {ROOT}: {path}")
    return resolved


def gpu_gate(index):
    if index != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Formal GPU jobs require physical GPU 1 and PCI_BUS_ID")
    memory = int(subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=memory.used",
                                          "--format=csv,noheader,nounits"], text=True).strip())
    if memory >= 2048:
        raise RuntimeError(f"GPU 1 occupied: {memory} MiB before job")


def train_overlap(train, heldout, selected):
    def ids(path, split, keep=None):
        result = set()
        matched = set()
        for r in old.lines(path):
            if r["split"] != split:
                raise ValueError(f"Unexpected split: {path}")
            if keep is None or r["task"] in keep:
                matched.add(r["task"])
                result.update(r["example_ids"])
        if keep is not None and matched != keep:
            raise ValueError("Could not verify all selected train task IDs")
        return result
    train_ids = ids(train, "train", set(selected))
    heldout_ids = ids(heldout, "heldout")
    overlap = train_ids & heldout_ids
    if overlap:
        raise ValueError(f"{len(overlap)} selected question IDs occur in held-out")
    return {"train_example_ids": len(train_ids), "heldout_example_ids": len(heldout_ids),
            "heldout_overlap": len(overlap), "field": "example_ids"}


def state_checks(path, facts):
    result = dict(facts)
    result.update(train_overlap(TRAIN, HELDOUT, result["task_ids"]))
    lengths = defaultdict(Counter)
    keys = set()
    for row in old.lines(path):
        key = (row["trajectory_id"], row["step"])
        if key in keys or row["split"] != "train":
            raise ValueError("Duplicate or non-training state")
        keys.add(key)
        lengths[f"{row['mode']}|{row['budget_tier']}"][row["prompt_tokens"]] += 1
    if len(keys) != result["n_states"]:
        raise ValueError("State count discrepancy")
    for cell, counts in lengths.items():
        result["cells"][cell]["prompt_lengths"] = {**old.length_distribution(counts), "min": min(counts),
            "exceeds_8392": sum(n for k,n in counts.items() if k > 8392),
            "prompt_plus_256_exceeds_8392": sum(n for k,n in counts.items() if k + 256 > 8392)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare","reference","build"), default="build")
    p.add_argument("--base-model", type=Path, default=old.DEFAULT_BASE)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--source-adapter", type=Path)
    p.add_argument("--model-label", default=LABEL)
    p.add_argument("--task-file", type=Path, default=TRAIN)
    p.add_argument("--n-tasks", type=int, default=400)
    p.add_argument("--seed", type=int, default=old.SEED)
    p.add_argument("--gpu", type=int, default=1, choices=(1,))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompts", type=Path, default=FIXTURE)
    p.add_argument("--compare", type=Path)
    args = p.parse_args()
    if args.task_file.resolve() != TRAIN:
        p.error("Only the v2 train task file is allowed")
    if args.stage == "reference":
        if args.prompts.resolve() != FIXTURE or sum(1 for _ in old.lines(args.prompts)) != 30:
            p.error("Reference must use the 30 saved Task 07 prompts")
        if args.compare:
            formal_path(args.compare)
    args.output_dir = formal_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    original_writer = old.write_json
    original_make_row = old.make_row
    def make_row(*a):
        row = original_make_row(*a)
        return {**row, "user_text": row["messages"][0]["content"]}
    def write(path, result):
        if path.name == "state_checks.json":
            result = state_checks(args.output_dir/"states.jsonl", result)
        if args.stage != "prepare":
            result["cuda_visible_devices"] = str(args.gpu)
        original_writer(path, result)
    # The frozen generator exposes its train path and engine guard as module globals.
    # Bind only these run-local settings and result annotations to the formal job.
    with patch.multiple(old, TRAIN_TASKS=TRAIN, check_gpu=gpu_gate, write_json=write, make_row=make_row):
        {"prepare":old.prompt_proof, "reference":old.reference, "build":old.build}[args.stage](args)


if __name__ == "__main__":
    main()
