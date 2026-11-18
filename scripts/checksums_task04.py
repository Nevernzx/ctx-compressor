#!/usr/bin/env python3
"""Task 04 delivery hashes and read-only preservation checks (standard library).

Usage: checksums_task04.py write|check [--slow]
Legacy aliases: no action means write; --verify means check;
--verify-previous checks only the three archived task baselines.
External payload SHA256 and actual JSONL line counts are checked only with
--slow. Fast mode validates manifest metadata, file placement and byte sizes.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
DOCUMENTS = {"README.md", "NOTES.md", "HANDOFF.md"}
AUTHORIZED = {
    "src/build_traj.py": ("hop_queries", "hop_queries"),
    "tests/test_build_traj.py": (
        "test_invalid_or_forward_reference_is_not_silently_retrieved",
        "test_out_of_range_reference_is_preserved"),
}
SKIP_DIRS = {".git", ".conda", ".cache", ".tmp", ".venv", "envs", "logs", "runs",
             "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
             ".ipynb_checkpoints"}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    require(isinstance(name, str) and name and "\n" not in name and "\r" not in name,
            f"Invalid relative filename: {name!r}")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            f"Unsafe relative filename: {name!r}")
    return path


def read_sums(path):
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        require("  " in line, f"Malformed checksum entry in {path}: {line!r}")
        checksum, name = line.split("  ", 1)
        require(bool(SHA256_RE.fullmatch(checksum)), f"Invalid SHA256: {name}")
        safe_relative(name)
        require(name not in entries, f"Duplicate checksum entry: {name}")
        entries[name] = checksum
    require(bool(entries), f"Empty checksum file: {path}")
    return entries


def outside_function(path, function_name):
    source = Path(path).read_bytes()
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name == function_name]
    require(len(nodes) == 1, f"Expected one function {function_name} in {path}")
    node, lines = nodes[0], source.splitlines(keepends=True)
    return b"".join(lines[:node.lineno - 1] + lines[node.end_lineno:])


def check_authorized_scope(root, baselines):
    """Authenticate old files against all baselines, then compare function exterior."""
    results = []
    for name, (old_function, new_function) in AUTHORIZED.items():
        before = root / "specs/task03_pre_task04" / (name + ".txt")
        after = root / name
        old_sha = digest(before)
        for task, baseline in baselines.items():
            require(baseline.get(name) == old_sha,
                    f"{task}: archived authorized-file snapshot changed: {name}")
        require(outside_function(before, old_function) == outside_function(after, new_function),
                f"Unauthorized change outside {new_function}: {name}")
        results.append({"path": name, "authorized_function": new_function,
                        "snapshot_sha256": old_sha, "current_sha256": digest(after),
                        "outside_function_byte_identical": True})
    return results


def verify_legacy(root=ROOT):
    root = Path(root)
    baselines = {f"TASK_{task:02d}": read_sums(root / f"specs/TASK{task:02d}_SHA256SUMS")
                 for task in (1, 2, 3)}
    scope = check_authorized_scope(root, baselines)
    cache, reports, protected = {}, {}, set()
    for task, entries in baselines.items():
        number = task[-2:]
        docs = json.loads((root / f"specs/TASK{number}_document_baselines.json").read_text())
        require(set(docs) == DOCUMENTS, f"{task}: unexpected document baseline set")
        for name, expected in entries.items():
            path = root / name
            require(path.is_file(), f"{task}: missing legacy file: {name}")
            if name in AUTHORIZED:
                continue  # Snapshot hash + function-exterior comparison already checked.
            if name in docs:
                size = docs[name]["size_bytes"]
                require(type(size) is int and size >= 0, f"Invalid document prefix size: {name}")
                with path.open("rb") as stream:
                    prefix = stream.read(size)
                require(len(prefix) == size and hashlib.sha256(prefix).hexdigest()
                        == expected == docs[name]["sha256"],
                        f"{task}: document prefix changed or shortened: {name}")
            else:
                if name not in cache:
                    cache[name] = digest(path)
                require(cache[name] == expected, f"{task}: immutable artifact changed: {name}")
                if name.startswith(("data/", "eval/")):
                    protected.add(name)
        reports[task] = {"manifest_entries": len(entries), "append_only_documents": len(docs),
                         "authorized_function_changes": len(AUTHORIZED)}
    return {"passed": True, "tasks": reports, "authorized_scope": scope,
            "protected_data_and_eval_files": len(protected),
            "retrieval_runs": 0, "model_imports": 0}


def artifact_files(root=ROOT):
    """Include all regular delivery files; do not traverse environment/log/link trees."""
    root = Path(root).resolve()
    files = []
    for parent, directories, names in os.walk(root, followlinks=False):
        parent = Path(parent)
        directories[:] = sorted(name for name in directories
                                if name not in SKIP_DIRS and not (parent / name).is_symlink())
        for name in names:
            path = parent / name
            if path == root / "SHA256SUMS" or name.endswith((".pyc", ".pyo")):
                continue
            if path.is_symlink() and not path.resolve().is_relative_to(root):
                continue
            if path.is_file():
                files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def current_sums(root=ROOT):
    root = Path(root).resolve()
    return {path.relative_to(root).as_posix(): digest(path) for path in artifact_files(root)}


def check_current(actual, expected):
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    changed = sorted(name for name in actual.keys() & expected.keys() if actual[name] != expected[name])
    require(not (missing or extra or changed),
            f"SHA256SUMS mismatch: missing={missing}, extra={extra}, changed={changed}")


def validate_external_file(record, allowed_root, slow=False):
    """Validate one manifest record; never follow a manifest path outside its family."""
    path = Path(record["path"])
    allowed_root = Path(allowed_root).resolve()
    require(path.is_absolute() and path.resolve().is_relative_to(allowed_root),
            f"External manifest path outside {allowed_root}: {path}")
    expected_size = record.get("bytes", record.get("size_bytes"))
    require(type(expected_size) is int and expected_size >= 0, f"Invalid byte size: {path}")
    require(bool(SHA256_RE.fullmatch(record.get("sha256", ""))), f"Invalid SHA256: {path}")
    lines = record.get("lines")
    require(lines is None or (type(lines) is int and lines >= 0), f"Invalid line count: {path}")
    require(path.is_file() and path.stat().st_size == expected_size,
            f"External file missing or wrong byte size: {path}")
    if slow:
        h, actual_lines = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
                if lines is not None:
                    actual_lines += block.count(b"\n")
        require(h.hexdigest() == record["sha256"], f"External SHA256 mismatch: {path}")
        require(lines is None or actual_lines == lines, f"External JSONL line count mismatch: {path}")
    return {"path": str(path), "bytes": expected_size, "declared_lines": lines,
            "sha256_verified": slow, "actual_lines_verified": slow and lines is not None}


def verify_external(root=ROOT, external=EXTERNAL, slow=False):
    root, external = Path(root), Path(external)
    result = {"mode": "sha256_and_actual_lines" if slow else "metadata_and_sizes_only", "files": []}
    traj_path = root / "eval/sft_prep/trajectory_stats.json"
    traj = json.loads(traj_path.read_text())
    require(traj_path.read_bytes() == (external / "data/traj_train/manifest.json").read_bytes(),
            "External trajectory manifest differs from protected project copy")
    records = traj["files"]
    for split in ("train", "heldout"):
        item = records[split]
        require(Path(item["path"]) == external / f"data/traj_train/{split}_tasks.jsonl",
                f"Unexpected {split} trajectory filename")
        require(item["lines"] == traj["statistics"][split]["tasks"], "Trajectory task count mismatch")
        result["files"].append(validate_external_file(item, external / "data/traj_train", slow))
    require(records["corpus"]["lines"] == traj["corpus"]["paragraphs"], "Corpus line count mismatch")
    for item in [records["corpus"], *records["index"]]:
        result["files"].append(validate_external_file(item, external / "data/traj_train", slow))
    split_record = records["split_ids"]
    require(Path(split_record["path"]) == external / "data/splits/train_heldout_ids.json",
            "Unexpected split IDs filename")
    result["files"].append(validate_external_file(split_record, external / "data/splits", slow))
    require((root / "data/splits/train_heldout_ids.json").read_bytes() == Path(split_record["path"]).read_bytes(),
            "External split IDs differ from protected project copy")
    sft = json.loads((root / "eval/sft_prep/sft_data_stats.json").read_text())
    for split in ("train", "heldout"):
        item = sft["files"][split]
        require(Path(item["path"]) == external / f"data/sft/{split}.jsonl", f"Unexpected {split} SFT filename")
        require(item["lines"] == traj["statistics"][split]["samples"], "SFT line count differs from trajectory states")
        provenance = sft["trajectory_inputs"][split]
        require(all(provenance[k] == records[split][k] for k in ("path", "bytes", "lines", "sha256")),
                f"SFT input manifest differs from original {split} trajectories")
        result["files"].append(validate_external_file(item, external / "data/sft", slow))
    model = json.loads((root / "eval/sft_prep/model_manifest.json").read_text())
    model_root = Path(model["model_root"])
    require(model_root == external / "models/Qwen3-1.7B" and model["status"] == "verified",
            "Policy model snapshot is not complete and verified")
    expected_names, total_bytes = set(), 0
    for item in model["files"]:
        name = item["path"]
        safe_relative(name)
        require(name not in expected_names, f"Duplicate model file: {name}")
        require(item["status"] == "verified" and item["actual_sha256"] == item["sha256"],
                f"Model file was not verified against source hash: {name}")
        expected_names.add(name)
        total_bytes += item["size_bytes"]
        record = {**item, "path": str(model_root / name)}
        result["files"].append(validate_external_file(record, model_root, slow))
    actual_names = {p.relative_to(model_root).as_posix() for p in model_root.rglob("*") if p.is_file()}
    require(bool(expected_names) and actual_names == expected_names, "Model file coverage differs from frozen manifest")
    require(total_bytes == model["total_expected_bytes"], "Model expected byte total mismatch")
    result.update(passed=True, file_count=len(result["files"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("write", "check"))
    parser.add_argument("--verify", action="store_true", help="Legacy alias for check")
    parser.add_argument("--verify-previous", action="store_true", help="Only protect Task 01/02/03 baselines")
    parser.add_argument("--slow", action="store_true", help="Hash external payloads and count JSONL lines")
    args = parser.parse_args(argv)
    if args.verify and args.action == "write":
        parser.error("write conflicts with --verify")
    if args.verify_previous and (args.action or args.verify or args.slow):
        parser.error("--verify-previous is a standalone read-only check")
    action = "check" if args.verify else args.action or "write"
    legacy = verify_legacy()
    if args.verify_previous:
        print(json.dumps({"legacy": legacy}, ensure_ascii=False, indent=2))
        return 0
    external = verify_external(slow=args.slow)
    actual = current_sums()
    target = ROOT / "SHA256SUMS"
    if action == "check":
        check_current(actual, read_sums(target))
    else:
        target.write_text("".join(f"{checksum}  {name}\n" for name, checksum in actual.items()), encoding="utf-8")
    print(json.dumps({"action": action, "passed": True, "project_files": len(actual),
                      "legacy": legacy, "external": external}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
