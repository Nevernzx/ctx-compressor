"""Frozen paths, provenance checks and dated inputs of GRPO round four."""
from pathlib import Path
import os
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.grpo_r3_config import previous_file, previous_manifest
from scripts.grpo_r2_start import ROOT, read
from scripts.rollout_model import file_record
from scripts.sample_sft_subset import digest, require, write_json_exclusive
from src.reader_heldout_inputs import sums

BASE = ROOT / "grpo_r2/merged_start"
R2EXT = BASE.parent
R2 = PROJECT / "eval/grpo_r2"
RUN = ROOT / "grpo_r4"
SMALL = PROJECT / "eval/grpo_r4"
RAW = ROOT / "eval_rollouts_grpo_r4"
READER = ROOT / "reader_t18"
LOCK = ROOT / "rt/gpu1_t17_t18.lock"
DATA = RUN / "data/train.jsonl"
SMOKE_DATA = RUN / "smoke_train/train.jsonl"
TRAIN = RUN / "train"
TRAIN_TASKS = ROOT / "data/traj_train/train_tasks.jsonl"
HELDOUT = TRAIN_TASKS.with_name("heldout_tasks.jsonl")
TRAIN_SHA = "ae96199b83b264bd926312e1e91e3b2af753b20117a6cb88161c8f3b23033e56"
HELDOUT_SHA = "5a6c8ce2c2c738ff2a0e3a15e95d43e8b6ecde7806d7104e5472cf74dddaafac"
R2_POLICY = "grpo_r2:checkpoint-125"
V3_POLICY = "sft_v3:checkpoint-3000"
SEED = 20261123
DATA_SEED = 20261124
TARGET = 2000
MIN_SIGNAL = 500
CHECKPOINT_ORDER = (125, 250, 375, 500)


def expect_new(*paths):
    for path in paths:
        require(path.resolve() == path and not os.path.lexists(path), f"Existing or redirected output: {path}")


def verify_model():
    manifest = previous_manifest()
    paths = {Path(name) for name in manifest if name.startswith(str(BASE) + "/")}
    require(paths and set(BASE.iterdir()) == paths, "Task 11 signed merged model inventory differs")
    records = {path.name: previous_file(path) for path in sorted(paths)}
    merge = read(BASE / "merge_record.json")
    start = read(R2 / "start_model.json")
    previous_file(R2 / "start_model.json")
    require(start["strategy"] == "checkpoint-3000" and start["version"] == "v3" and
            merge["adapter"] == start["path"] and merge["adapter_sha256"] == start["adapter_sha256"] and
            records["model.safetensors"]["sha256"] == merge["weights"]["model.safetensors"],
            "Task 11's merged SFT v3 start does not agree with its receipts")
    return records


def signature(path, manifest_path):
    index = sums(PROJECT / "SHA256SUMS")
    signed = sums(manifest_path)
    require(digest(manifest_path) == index[str(manifest_path.relative_to(PROJECT))], "Historical manifest changed")
    name = str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path)
    out = file_record(path)
    require(path.resolve() == path and out["sha256"] == signed[name], f"Signature differs: {path}")
    return out


def task_gold(path, split, count, expected_sha):
    proof = file_record(path)
    require(path.resolve() == path and proof["sha256"] == expected_sha, f"Wrong {split} task source")
    import json
    out = {}
    with path.open() as stream:
        for line in stream:
            task = json.loads(line)
            tid = task["task"]
            require(tid not in out and task["split"] == split, "Unexpected split or duplicate task")
            gold_by_step = [step["gold"] for step in task["steps"]]
            require(gold_by_step and all(type(pid) is int and pid >= 0 for pid in gold_by_step), "Bad gold PID")
            out[tid] = gold_by_step
    require(set(out) == set(range(count)), f"Incomplete {split} task index")
    return out, proof

