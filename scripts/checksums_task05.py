#!/usr/bin/env python3
"""Task 05 integrity checks, inheriting four immutable task generations.

write|check [--slow] requires the final complete external artifact manifest.
--verify-previous checks only Task 01--04 preservation. --verify-sampling checks
the frozen subsets before the final run is complete. --slow hashes all external
payloads, counts JSONL lines, and reproduces the deterministic subset selection.
This script never mutates archived baselines or external data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
sys.path.insert(0, str(ROOT))
from scripts import checksums_task04 as previous
from scripts import sample_sft_subset as sampling

require = previous.require
digest = previous.digest
FINAL_ROOTS = ("runs/sft_20k_20261117", "runs/batch_bench_20261117", "eval_rollouts")


def validate_subset_ids(items, split, expected_tasks):
    """Heldout uniform sampling need not cover the same tasks in both modes."""
    if split == "train":
        sampling.validate_metadata(items, split, expected_tasks)
        return
    identities, lines, tiers = set(), set(), {}
    for item in items:
        meta = item["meta"]
        require(meta["split"] == "heldout" and meta["mode"] in sampling.MODES
                and meta["tier"] in sampling.TIERS, "Invalid heldout subset metadata")
        require(type(meta["task"]) is int and 0 <= meta["task"] < expected_tasks
                and type(meta["step"]) is int and meta["step"] >= 0, "Invalid heldout task/step")
        identity, line = sampling.metadata_identity(item), item["source_line_1based"]
        require(type(line) is int and line > 0 and line not in lines
                and identity not in identities, "Duplicate/invalid heldout sample ID")
        require(tiers.setdefault(meta["task"], meta["tier"]) == meta["tier"], "Heldout task tier changed")
        identities.add(identity)
        lines.add(line)


def verify_legacy(root=ROOT):
    """Earlier authorized edits are frozen in full by the Task 04 baseline."""
    root = Path(root)
    inherited = previous.verify_legacy(root)
    entries = previous.read_sums(root / "specs/TASK04_SHA256SUMS")
    docs = json.loads((root / "specs/TASK04_document_baselines.json").read_text())
    require(set(docs) == previous.DOCUMENTS, "TASK_04: unexpected document baseline set")
    for number in (1, 2, 3):
        old = previous.read_sums(root / f"specs/TASK{number:02d}_SHA256SUMS")
        require(set(old) <= set(entries), "TASK_04 baseline omitted an older protected path")
    protected = 0
    for name, expected in entries.items():
        path = root / name
        require(path.is_file(), f"TASK_04: missing legacy file: {name}")
        if name in docs:
            size = docs[name]["size_bytes"]
            require(type(size) is int and size >= 0, f"TASK_04: invalid document prefix size: {name}")
            with path.open("rb") as stream:
                prefix = stream.read(size)
            require(len(prefix) == size and hashlib.sha256(prefix).hexdigest()
                    == expected == docs[name]["sha256"],
                    f"TASK_04: document prefix changed or shortened: {name}")
        else:
            require(digest(path) == expected, f"TASK_04: immutable artifact changed: {name}")
            protected += name.startswith(("data/", "eval/"))
    inherited["tasks"]["TASK_04"] = {"manifest_entries": len(entries),
        "append_only_documents": len(docs), "authorized_function_changes": 0,
        "protected_data_and_eval_files": protected}
    inherited["task04_function_edits_now_fully_frozen"] = sorted(previous.AUTHORIZED)
    return inherited


def verify_sampling(root=ROOT, external=EXTERNAL, slow=False):
    root, external = Path(root), Path(external)
    manifest = json.loads((root / "eval/sft_run/sample_manifest.json").read_text())
    run_dir = external / "runs/sft_20k_20261117"
    require(Path(manifest["run_dir"]) == run_dir, "Unexpected subset run directory")
    source_path = root / "eval/sft_prep/sft_data_stats.json"
    source = json.loads(source_path.read_text())
    require(manifest["source_manifest"] == {"path": str(source_path), "sha256": digest(source_path)},
            "Subset source manifest differs from Task 04")
    protocol = manifest["protocol"]
    require(protocol["seed"] == sampling.SEED and protocol["rng"] == "random.Random"
            and protocol["training_rng_instances"] == 1 and protocol["mode_order"] == list(sampling.MODES)
            and protocol["minimum_per_task"] == 2 and protocol["training_samples_per_mode"] == 10000,
            "Training sampling protocol changed")
    evaluation = protocol["eval_loss"]
    require(evaluation["seed"] == sampling.SEED and evaluation["separate_rng"] is True
            and evaluation["samples"] == 1000 and evaluation["used_for_checkpoint_selection"] is False,
            "Loss-only heldout protocol changed")
    records, checked = manifest["files"], []
    require(set(records) == {"train", "eval_loss", "sample_ids"}, "Unexpected subset file set")
    for key, name, lines in (("train", "train_20k.jsonl", 20000),
                             ("eval_loss", "heldout_eval_loss_1k.jsonl", 1000),
                             ("sample_ids", "sample_ids.json", None)):
        require(Path(records[key]["path"]) == run_dir / name, "Unexpected subset filename")
        require(records[key].get("lines") == lines, "Unexpected subset line count")
        checked.append(previous.validate_external_file(records[key], run_dir, slow))
    ids = json.loads(Path(records["sample_ids"]["path"]).read_text())
    require(ids["protocol"] == protocol and ids["sources"] == manifest["sources"],
            "Sample IDs disagree with subset manifest")
    for split, id_key, file_key, tasks, rows in (("train", "train", "train", 4735, 20000),
                                               ("heldout", "heldout_eval_loss", "eval_loss", 249, 1000)):
        source_record = source["files"][split]
        declared = manifest["sources"][split]
        require(all(declared[k] == source_record[k] for k in ("path", "bytes", "lines", "sha256")),
                f"Subset {split} provenance differs from Task 04")
        require(Path(source_record["path"]) == external / f"data/sft/{split}.jsonl",
                "Unexpected source SFT path")
        require(declared["full_sha256_verified"] is True
                and declared["full_sha256_verified_after_copy"] is True
                and declared["task_count"] == tasks, "Source validation record incomplete")
        chosen = ids[id_key]
        require(len(chosen) == rows, "Wrong number of sample IDs")
        validate_subset_ids(chosen, split, tasks)
        lines = [item["source_line_1based"] for item in chosen]
        require(lines == sorted(lines) and max(lines) <= source_record["lines"], "Sample IDs not in source order")
        require(sampling.statistics(chosen) == manifest["statistics"][file_key], "Subset statistics mismatch")
        if split == "train":
            require(all(manifest["statistics"]["train"]["by_mode"][mode]["samples"] == 10000
                        and manifest["statistics"]["train"]["by_mode"][mode]["tasks"] == tasks
                        and manifest["statistics"]["train"]["by_mode"][mode]["min_samples_per_task"] >= 2
                        for mode in sampling.MODES), "Training subset task coverage is incomplete")
        if slow:
            population = sampling.scan_source(source_record, split, tasks)
            expected_lines = (sampling.select_training(population) if split == "train"
                              else sampling.select_eval_loss(population))
            line_set = set(expected_lines)
            expected_ids = [item for item in population if item["source_line_1based"] in line_set]
            require(chosen == expected_ids, "Subset selection or original metadata is not reproducible")
            selected_hash = hashlib.sha256()
            with Path(source_record["path"]).open("rb") as stream:
                for number, line in enumerate(stream, 1):
                    if number in line_set:
                        selected_hash.update(line)
            require(selected_hash.hexdigest() == records[file_key]["sha256"],
                    "Subset is not an exact original-byte selection")
    return {"passed": True, "mode": "full_sha256_and_reproducible_sampling" if slow else "metadata_and_sizes_only",
            "files": checked, "file_count": len(checked), "selection_reproduced": slow,
            "eval_loss_used_for_checkpoint_selection": False}


def external_inventory(roots):
    """Inventory all regular files and reject symlinks or unfinished writes."""
    files = set()
    for root in roots:
        root = Path(root)
        require(root.is_dir() and not root.is_symlink(), f"Missing/linked external root: {root}")
        for path in root.rglob("*"):
            require(not path.is_symlink(), f"Symlink in external artifacts: {path}")
            if path.is_file():
                require(not path.name.endswith((".partial", ".part", ".tmp")),
                        f"Unfinished external artifact: {path}")
                files.add(path)
    return files


def verify_final_artifacts(root=ROOT, external=EXTERNAL, slow=False):
    root, external = Path(root), Path(external)
    manifest_path = root / "eval/sft_run/external_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema_version"] == 1 and manifest["complete"] is True,
            "External artifact manifest is not final/complete")
    allowed = {external / relative for relative in FINAL_ROOTS}
    declared_roots = [Path(value) for value in manifest["roots"]]
    require(len(declared_roots) == len(allowed) and set(declared_roots) == allowed,
            "External artifact roots differ from the three Task 05 families")
    require(all(path.resolve() == path for path in declared_roots), "External roots must not resolve elsewhere")
    actual, expected, checked = external_inventory(declared_roots), set(), []
    for record in manifest["files"]:
        path = Path(record["path"])
        require(path not in expected, f"Duplicate external artifact record: {path}")
        family = [parent for parent in allowed if path.is_absolute() and path.is_relative_to(parent)]
        require(len(family) == 1, f"Unexpected external artifact path: {path}")
        if path.suffix == ".jsonl":
            require(type(record.get("lines")) is int, f"JSONL record lacks line count: {path}")
        expected.add(path)
        checked.append(previous.validate_external_file(record, family[0], slow))
    require(expected == actual,
            f"External artifact coverage mismatch: missing={sorted(map(str, expected - actual))}, "
            f"unrecorded={sorted(map(str, actual - expected))}")
    require(bool(expected), "Empty external artifact manifest")
    sample = json.loads((root / "eval/sft_run/sample_manifest.json").read_text())
    by_path = {record["path"]: record for record in manifest["files"]}
    for record in sample["files"].values():
        frozen = by_path.get(record["path"], {})
        require(all(frozen.get(key) == value for key, value in record.items()),
                "Final external manifest disagrees with frozen subset")
    return {"passed": True, "mode": "sha256_and_actual_lines" if slow else "metadata_and_sizes_only",
            "file_count": len(checked), "total_bytes": sum(path.stat().st_size for path in actual),
            "coverage_complete": True, "files": checked}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("write", "check"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify-previous", action="store_true")
    group.add_argument("--verify-sampling", action="store_true")
    parser.add_argument("--slow", action="store_true")
    args = parser.parse_args(argv)
    if (args.verify_previous or args.verify_sampling) and args.action:
        parser.error("read-only subset checks cannot be combined with write/check")
    legacy = verify_legacy()
    if args.verify_previous:
        print(json.dumps({"legacy": legacy}, ensure_ascii=False, indent=2))
        return 0
    sample = verify_sampling(slow=args.slow)
    if args.verify_sampling:
        print(json.dumps({"legacy": legacy, "sampling": sample}, ensure_ascii=False, indent=2))
        return 0
    inherited_external = previous.verify_external(slow=args.slow)
    final_external = verify_final_artifacts(slow=args.slow)
    actual = previous.current_sums()
    target, action = ROOT / "SHA256SUMS", args.action or "check"
    if action == "write":
        target.write_text("".join(f"{checksum}  {name}\n" for name, checksum in actual.items()), encoding="utf-8")
    else:
        previous.check_current(actual, previous.read_sums(target))
    print(json.dumps({"action": action, "passed": True, "project_files": len(actual),
                      "legacy": legacy, "sampling": sample, "inherited_external": inherited_external,
                      "final_external": final_external}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
