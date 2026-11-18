#!/usr/bin/env python3
"""Inventory the isolated Task05 compiler dependency; not task acceptance.

Reads only the fixed toolchain tree, task source receipts and process FD metadata.
Never follows tree symlinks, invokes the compiler, queries GPU state, or scans C
outputs. The sole write is a new small project manifest; existing files stay put.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[1]
TOOLCHAIN = Path("/root/userfolder/ctx-compressor/toolchains/gcc13_20261118")
OUTPUT = PROJECT / "eval/sft_run/toolchain_full_artifact_manifest.json"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metadata(path):
    value = path.lstat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns, value.st_nlink,
            value.st_uid, value.st_gid, value.st_blocks)


def tree_snapshot(root):
    """Record every entry including empty directories, without following links."""
    require(root.is_dir() and not root.is_symlink() and root.resolve() == root,
            "The fixed toolchain root must be a real directory")
    snapshot, pending = {".": ("directory", metadata(root), None)}, [root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            fields = metadata(path)
            mode = fields[2]
            target = None
            if stat.S_ISLNK(mode):
                kind, target = "symlink", os.readlink(path)
            elif stat.S_ISDIR(mode):
                kind = "directory"
                pending.append(path)
            elif stat.S_ISREG(mode):
                kind = "regular"
            else:
                raise ValueError(f"Unsupported toolchain entry type: {path}")
            snapshot[path.relative_to(root).as_posix()] = (kind, fields, target)
    return snapshot


def writable_toolchain_fds(root):
    """Inspect targets/flags only; never read unrelated files or send signals."""
    found, checked = [], 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        checked += 1
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                    if not (target == str(root) or target.startswith(str(root) + "/")):
                        continue
                    info = (proc / "fdinfo" / fd.name).read_text()
                    flags = int(next(line.split()[1] for line in info.splitlines()
                                     if line.startswith("flags:")), 8)
                    if flags & os.O_ACCMODE:
                        found.append({"pid": int(proc.name), "fd": int(fd.name),
                                      "path": target, "flags_octal": oct(flags)})
                except FileNotFoundError:
                    pass  # A closed FD or exited process is not an active writer.
        except FileNotFoundError:
            pass
    # Permission errors deliberately propagate: no unsupported claim of coverage.
    return {"visible_pids_checked": checked, "writable_toolchain_fds": found,
            "scope": "Visible /proc FD path and flags snapshots, not continuous writer tracing"}


def main():
    require(PROJECT == Path("/root/workplace/ctx-compressor")
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Run on 5880 formal project with CUDA_VISIBLE_DEVICES='' (CPU only)")
    require(not OUTPUT.exists(), "Refusing to overwrite an existing toolchain manifest")
    started = time.monotonic()
    source_hash = digest(__file__)
    test_receipt = PROJECT / "eval/sft_run/final_tests_receipt.json"
    receipt_hash = digest(test_receipt)
    receipt = json.loads(test_receipt.read_text())
    mismatches = [name for name, expected in receipt["source_sha256"].items()
                  if digest(PROJECT / name) != expected]
    require(not mismatches, f"Historical test source changed: {mismatches}")
    before_fds = writable_toolchain_fds(TOOLCHAIN)
    require(not before_fds["writable_toolchain_fds"], "An active FD writes the toolchain tree")
    before = tree_snapshot(TOOLCHAIN)
    files, links, directories, inode_hashes, inode_metadata = [], [], [], {}, {}
    for name, (kind, fields, target) in sorted(before.items()):
        if kind == "directory":
            directories.append(name)
        elif kind == "symlink":
            links.append({"path": name, "target": target})
        else:
            inode = fields[:2]
            if inode not in inode_hashes:
                path = TOOLCHAIN / name
                require(metadata(path) == fields, f"File changed before hashing: {path}")
                inode_hashes[inode] = digest(path)
                require(metadata(path) == fields, f"File changed while hashing: {path}")
                inode_metadata[inode] = fields
            else:
                require(inode_metadata[inode] == fields, f"Inconsistent hard-link metadata: {name}")
            files.append({"path": name, "bytes": fields[3], "sha256": inode_hashes[inode]})
    after = tree_snapshot(TOOLCHAIN)
    require(before == after, "Toolchain entry set, metadata or symlink targets changed during inventory")
    after_fds = writable_toolchain_fds(TOOLCHAIN)
    require(not after_fds["writable_toolchain_fds"], "An active FD writes the toolchain after hashing")
    require(digest(__file__) == source_hash and digest(test_receipt) == receipt_hash,
            "Audit source or historical test receipt changed during inventory")
    du_text = subprocess.check_output(["du", "-sx", "-B1", str(TOOLCHAIN)], text=True)
    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "additional_dependency_inventory_complete", "complete": True,
        "complete_meaning": "All regular paths and symlink targets in this fixed dependency tree; not task acceptance.",
        "task_acceptance_status": "not_evaluated", "root": str(TOOLCHAIN),
        "path_encoding": "POSIX paths relative to root; symlink targets preserved verbatim; links never followed",
        "files": files, "symlinks": links, "directories": directories,
        "counts": {"regular_paths": len(files), "symlinks": len(links),
                   "directories_including_root": len(directories),
                   "unique_regular_inodes": len(inode_hashes)},
        "size_accounting": {
            "regular_path_logical_bytes": sum(row["bytes"] for row in files),
            "unique_regular_inode_logical_bytes": sum(row[3] for row in inode_metadata.values()),
            "unique_regular_inode_allocated_bytes": sum(row[9] * 512 for row in inode_metadata.values()),
            "du_allocated_bytes": int(du_text.split()[0]), "du_raw": du_text,
            "interpretation": "Path logical bytes count hard-linked payloads repeatedly. Unique-inode allocation uses st_blocks*512; du also counts directory/link blocks. None is new root-filesystem growth or a replacement for raw final df."},
        "verification": {"all_entry_sets_and_metadata_stable": True,
                         "symlink_targets_stable": True, "before_fd_check": before_fds,
                         "after_fd_check": after_fds, "elapsed_seconds": time.monotonic() - started,
                         "hash_read_bytes_unique_inodes": sum(row[3] for row in inode_metadata.values())},
        "source": {"path": str(Path(__file__).resolve()), "sha256": source_hash},
        "historical_test_relationship": {
            "receipt": {"path": str(test_receipt), "sha256": receipt_hash},
            "historical_tests": receipt["tests"], "historical_source_count": len(receipt["source_sha256"]),
            "historical_sources_still_match": True, "new_script_in_historical_snapshot": False,
            "new_script_pytest_claim": "Not included in the historical 462-test run; no new tests claimed.",
            "new_script_validation": "This successful real-tree execution, before/after entry/metadata stability, per-file hashes and writer-FD checks."},
        "scope": {"fixed_dependency_root_only": True, "finalizer_three_roots_changed": False,
                  "C_outputs_scanned": False, "gpu_queried_or_initialized": False,
                  "compiler_executed": False, "dependencies_installed": False,
                  "existing_files_modified": False, "signals_sent": 0},
        "requirement_assessment": {
            "task05_section5": "Task05 requires external large artifacts with path and SHA256 but does not explicitly name a compiler dependency tree. This supplemental tree inventory makes the later isolated build dependency auditable.",
            "prior74_scope": "The original74-entry manifest covers19 downloaded archives plus selected critical binaries/metadata/proof. It is retained unchanged and is not described as whole-tree coverage.",
            "acceptance": "This additional dependency inventory neither changes the three formal external roots nor establishes terminal GPU/disk or overall task acceptance."}}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "task_acceptance_status": "not_evaluated",
                      "path": str(OUTPUT), "bytes": OUTPUT.stat().st_size, "sha256": digest(OUTPUT),
                      "counts": result["counts"], "size_accounting": result["size_accounting"],
                      "historical_sources_unchanged": len(receipt["source_sha256"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
