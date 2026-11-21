"""Verify the formal SFT starting point and immutable train split; perform the frozen CPU merge."""
import argparse
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import merge_sft_lora

ROOT = Path("/root/userfolder/ctx-compressor")
RUN = ROOT / "grpo_v2"
OUT = PROJECT / "eval/grpo_v2"
BASE = ROOT / "models/Qwen3-1.7B"
ADAPTER = ROOT / "runs/sft_20k_v2_20261118/checkpoints/checkpoint-2000"
SHA = "7253eae08072782971fa8b7874e97fc3e76dee4ebd11ced5c014147090cf1f9c"
MERGED = RUN / "merged_v2_ckpt2000"


def check_start():
    record = json.loads((PROJECT / "eval/sft_run_v2/selected_checkpoint.json").read_bytes())
    actual = merge_sft_lora.sha256(ADAPTER / "adapter_model.safetensors")
    if record["path"] != str(ADAPTER) or actual != SHA:
        raise ValueError("Task 06 selected adapter or weights have changed")


def digest_row(digest, row):
    digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def audit_inputs():
    old = ROOT / "data/traj_train/train_tasks.jsonl"
    new = ROOT / "data/traj_train_v2/train_tasks.jsonl"
    heldout = ROOT / "data/traj_train/heldout_tasks.jsonl"
    fixture = ROOT / "grpo_prep/step0_fixture/prompts.jsonl"
    d1, d2 = hashlib.sha256(), hashlib.sha256()
    task_ids, questions = set(), set()
    n_steps = 0
    with old.open("rb") as a, new.open("rb") as b:
        for i, (x, y) in enumerate(zip_longest(a, b), 1):
            if x is None or y is None:
                raise ValueError("Train task counts differ")
            first, second = json.loads(x), json.loads(y)
            for rec in (first, second):
                rec.pop("rollouts")
            if first != second or first["split"] != "train" or first["task"] in task_ids:
                raise ValueError(f"Task input fields or unique task ID differ at row {i}")
            task_ids.add(first["task"])
            questions.update(first["example_ids"])
            n_steps += len(first["steps"])
            digest_row(d1, first)
            digest_row(d2, second)
    with heldout.open("rb") as f:
        held = [json.loads(line)["example_ids"] for line in f]
    held_ids = set().union(*held)
    with fixture.open("rb") as f:
        rows = [json.loads(line) for line in f]
    if (len(task_ids) != 4735 or len(held) != 249 or questions & held_ids or len(rows) != 30
            or len({r["task"] for r in rows}) != 30
            or any(r["task"] not in task_ids or r["split"] != "train" or r["step"] != 0 for r in rows)):
        raise ValueError("Split isolation or fixed step-0 fixture failed")
    data = {"n_train_tasks": len(task_ids), "n_train_steps": n_steps, "n_heldout_tasks": len(held),
            "n_train_question_ids": len(questions), "n_heldout_question_ids": len(held_ids),
            "overlapping_question_ids": 0, "compared_fields": sorted(first), "excluded_fields": ["rollouts"],
            "mismatches": 0, "old_projection_sha256": d1.hexdigest(), "v2_projection_sha256": d2.hexdigest(),
            "fixture": str(fixture), "fixture_sha256": merge_sft_lora.sha256(fixture), "fixture_n": len(rows),
            "source_adapter_sha256": SHA}
    with (OUT / "input_checks.json").open("x") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["merge", "inputs"])
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("CPU only")
    check_start()
    if args.command == "inputs":
        audit_inputs()
        return
    # Reuse the frozen Task 07 merge verbatim with the formal run's output boundary.
    merge_sft_lora.PREP = RUN
    sys.argv = [__file__, "--base", str(BASE), "--adapter", str(ADAPTER), "--out", str(MERGED)]
    merge_sft_lora.main()


if __name__ == "__main__":
    main()
