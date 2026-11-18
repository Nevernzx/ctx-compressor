#!/usr/bin/env python3
"""Freeze Task 05 training and loss-only heldout subsets, without model imports.

Training uses one Random(seed): mode order reselect, append; task IDs ascending;
within-task candidates by step then source line. Take min(2, n) per task, then
uniformly sample the remaining source-order rows to reach 10,000 per mode.
Heldout uses a separate Random(seed) and samples 1,000 rows uniformly. Both
outputs preserve original source line order and every selected byte.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
SEED = 20261118
MODES = ("reselect", "append")
TIERS = ("tight", "mid", "loose")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def metadata_identity(item):
    meta = item["meta"]
    return meta["task"], meta["step"], meta["mode"]


def validate_metadata(items, split, expected_tasks=None):
    seen, lines, tasks, tiers = set(), set(), defaultdict(set), {}
    for item in items:
        meta = item["meta"]
        require(meta["split"] == split, "Source split mismatch")
        require(meta["mode"] in MODES and meta["tier"] in TIERS, "Unknown mode/tier")
        require(all(type(meta[key]) is int and meta[key] >= 0 for key in ("task", "step")),
                "task/step must be nonnegative integers")
        line = item["source_line_1based"]
        require(type(line) is int and line > 0 and line not in lines, "Duplicate/invalid source line")
        identity = metadata_identity(item)
        require(identity not in seen, f"Duplicate sample identity: {identity}")
        seen.add(identity)
        lines.add(line)
        tasks[meta["mode"]].add(meta["task"])
        old_tier = tiers.setdefault(meta["task"], meta["tier"])
        require(old_tier == meta["tier"], "Task tier changed within source")
    require(all(tasks[mode] for mode in MODES), "Both modes must be present")
    require(tasks[MODES[0]] == tasks[MODES[1]], "Modes have different task coverage")
    if expected_tasks is not None:
        require(tasks[MODES[0]] == set(range(expected_tasks)), "Unexpected task ID coverage")


def select_training(items, per_mode=10000, seed=SEED, minimum_per_task=2):
    """Return source line numbers; never depend on input dictionary/set ordering."""
    require(type(per_mode) is int and per_mode > 0, "per_mode must be positive")
    require(type(minimum_per_task) is int and minimum_per_task >= 1, "minimum must be positive")
    validate_metadata(items, "train")
    rng, selected = random.Random(seed), set()
    for mode in MODES:
        candidates = sorted((item for item in items if item["meta"]["mode"] == mode),
                            key=lambda item: item["source_line_1based"])
        groups = defaultdict(list)
        for item in candidates:
            groups[item["meta"]["task"]].append(item)
        mandatory = sum(min(minimum_per_task, len(group)) for group in groups.values())
        require(mandatory <= per_mode <= len(candidates),
                f"{mode}: requested size cannot satisfy coverage/population")
        mode_selected = set()
        for task in sorted(groups):
            group = sorted(groups[task], key=lambda item: (item["meta"]["step"], item["source_line_1based"]))
            # random.sample is called even when n <= minimum; its RNG consumption
            # is part of the explicit deterministic protocol.
            mode_selected.update(item["source_line_1based"]
                                 for item in rng.sample(group, min(minimum_per_task, len(group))))
        remaining = [item["source_line_1based"] for item in candidates
                     if item["source_line_1based"] not in mode_selected]
        mode_selected.update(rng.sample(remaining, per_mode - len(mode_selected)))
        selected.update(mode_selected)
    return sorted(selected)


def select_eval_loss(items, count=1000, seed=SEED):
    """Independent uniform draw: diagnostic loss only, never checkpoint selection."""
    validate_metadata(items, "heldout")
    require(type(count) is int and 0 < count <= len(items), "Invalid eval-loss count")
    lines = sorted(item["source_line_1based"] for item in items)
    return sorted(random.Random(seed).sample(lines, count))


def statistics(items):
    counts = Counter((item["meta"]["mode"], item["meta"]["tier"]) for item in items)
    task_counts = Counter((item["meta"]["mode"], item["meta"]["task"]) for item in items)
    return {"samples": len(items), "unique_tasks": len({item["meta"]["task"] for item in items}),
            "by_tier": {tier: sum(counts[mode, tier] for mode in MODES) for tier in TIERS},
            "by_mode": {mode: {"samples": sum(counts[mode, tier] for tier in TIERS),
                "tasks": sum(m == mode for m, _ in task_counts),
                "min_samples_per_task": min((n for (m, _), n in task_counts.items() if m == mode), default=0),
                "max_samples_per_task": max((n for (m, _), n in task_counts.items() if m == mode), default=0),
                "by_tier": {tier: counts[mode, tier] for tier in TIERS}} for mode in MODES},
            "tokens": {key: {"sum": sum(item["meta"][key] for item in items),
                              "max": max((item["meta"][key] for item in items), default=0)}
                       for key in ("n_input_tokens", "n_target_tokens", "total_tokens", "loss_tokens")}}


def scan_source(record, split, expected_tasks):
    """Read every source byte and retain metadata only; reject before writing."""
    source = Path(record["path"])
    require(source.stat().st_size == record["bytes"], "Source byte size differs from Task04")
    items, hasher = [], hashlib.sha256()
    with source.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            hasher.update(line)
            require(line.endswith(b"\n"), "Source JSONL is not newline terminated")
            row = json.loads(line)
            require(isinstance(row.get("messages"), list), "Missing original messages")
            meta = row["meta"]
            require(type(meta["total_tokens"]) is int and 0 < meta["total_tokens"] <= 8392,
                    "Source violates frozen max_length")
            items.append({"source_line_1based": line_number, "meta": meta})
    require(len(items) == record["lines"] and hasher.hexdigest() == record["sha256"],
            f"Full source SHA256/line count mismatch: {source}")
    validate_metadata(items, split, expected_tasks)
    return items


def copy_selected(record, lines, destination):
    """Copy original bytes and rehash the complete source before publishing."""
    destination = Path(destination)
    partial = destination.with_suffix(destination.suffix + ".partial")
    require(not destination.exists() and not partial.exists(), f"Refusing overwrite: {destination}")
    selected = set(lines)
    require(len(selected) == len(lines) and selected and min(selected) >= 1, "Invalid selected lines")
    source_hash, output_hash, source_lines, count = hashlib.sha256(), hashlib.sha256(), 0, 0
    with Path(record["path"]).open("rb") as source, partial.open("xb") as target:
        for source_lines, line in enumerate(source, 1):
            source_hash.update(line)
            if source_lines in selected:
                target.write(line)
                output_hash.update(line)
                count += 1
        target.flush()
        os.fsync(target.fileno())
    require(source_hash.hexdigest() == record["sha256"] and source_lines == record["lines"],
            "Source changed during subset creation")
    require(count == len(selected), "Requested source rows missing")
    partial.rename(destination)
    return {"path": str(destination), "bytes": destination.stat().st_size,
            "lines": count, "sha256": output_hash.hexdigest()}


def write_json_exclusive(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=EXTERNAL / "runs/sft_20k_20261117")
    args = parser.parse_args(argv)
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("execute on 5880 in the formal project with CUDA_VISIBLE_DEVICES=''")
    if Path(sys.prefix) != EXTERNAL / "envs/train":
        parser.error("use the existing dedicated train environment")
    if not args.run_dir.resolve().is_relative_to(EXTERNAL / "runs"):
        parser.error("run directory must remain under external runs")
    start = time.monotonic()
    source_stats = PROJECT / "eval/sft_prep/sft_data_stats.json"
    source_manifest = json.loads(source_stats.read_text())
    files = {"train": args.run_dir / "train_20k.jsonl",
             "eval_loss": args.run_dir / "heldout_eval_loss_1k.jsonl",
             "sample_ids": args.run_dir / "sample_ids.json"}
    result_path = PROJECT / "eval/sft_run/sample_manifest.json"
    for path in [*files.values(), result_path]:
        require(not path.exists() and not path.with_suffix(path.suffix + ".partial").exists(),
                f"Refusing to overwrite existing output: {path}")
    inputs, selected, selection_lines = {}, {}, {}
    for split, count in (("train", 4735), ("heldout", 249)):
        record = source_manifest["files"][split]
        require(Path(record["path"]) == EXTERNAL / f"data/sft/{split}.jsonl", "Unexpected source path")
        items = scan_source(record, split, count)
        lines = select_training(items) if split == "train" else select_eval_loss(items)
        line_set = set(lines)
        selected[split] = [item for item in items if item["source_line_1based"] in line_set]
        inputs[split] = {**record, "full_sha256_verified": True, "task_count": count}
        selection_lines[split] = lines
        print(f"{split}: source SHA256 verified, selected {len(lines)} / {len(items)} rows", flush=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, key in (("train", "train"), ("heldout", "eval_loss")):
        outputs[key] = copy_selected(inputs[split], selection_lines[split], files[key])
        inputs[split]["full_sha256_verified_after_copy"] = True
    protocol = {
        "rng": "random.Random", "seed": SEED, "training_rng_instances": 1,
        "mode_order": list(MODES), "task_order": "ascending numeric task ID",
        "within_task_order": "ascending (step, source_line_1based)",
        "minimum_per_task": 2, "training_samples_per_mode": 10000,
        "task_draw": "random.sample(group, min(2, len(group))), including groups with <= 2 rows",
        "fill_draw": "uniform random.sample from unselected rows in ascending source line order",
        "output_order": "ascending source line; source bytes unchanged",
        "eval_loss": {"seed": SEED, "separate_rng": True, "samples": 1000,
                      "draw": "uniform random.sample from all heldout rows in ascending source line order",
                      "used_for_checkpoint_selection": False,
                      "purpose": "fixed diagnostic loss curve only"}}
    ids = {"schema_version": 1, "protocol": protocol, "sources": inputs,
           "train": selected["train"], "heldout_eval_loss": selected["heldout"]}
    outputs["sample_ids"] = write_json_exclusive(files["sample_ids"], ids)
    manifest = {"schema_version": 1, "task": "TASK_05", "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(args.run_dir), "protocol": protocol, "sources": inputs,
                "source_manifest": {"path": str(source_stats), "sha256": digest(source_stats)},
                "files": outputs,
                "statistics": {"train": statistics(selected["train"]),
                               "eval_loss": statistics(selected["heldout"])},
                "validation": {"original_jsonl_bytes_preserved": True,
                    "source_sha256_verified_twice": True, "models_loaded": False,
                    "cuda_visible_devices": "", "training_tasks_per_mode": 4735,
                    "eval_loss_used_for_checkpoint_selection": False},
                "elapsed_seconds": time.monotonic() - start,
                "python_version": sys.version, "code_sha256": {"scripts/sample_sft_subset.py": digest(__file__)}}
    write_json_exclusive(result_path, manifest)
    print(json.dumps({"manifest": str(result_path), "files": outputs,
                      "statistics": manifest["statistics"], "elapsed_seconds": manifest["elapsed_seconds"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
