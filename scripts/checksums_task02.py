"""Extend TASK_01 hash coverage while preserving its immutable artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

from checksums import artifact_files, digest

ROOT = Path(__file__).resolve().parents[1]


def verify_task01():
    old_entries = {}
    for line in (ROOT / "specs/TASK01_SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        old_entries[name] = checksum
    append_only = json.loads((ROOT / "specs/TASK01_document_baselines.json").read_text())
    for name, checksum in old_entries.items():
        path = ROOT / name
        if name in append_only:
            spec = append_only[name]
            with path.open("rb") as stream:
                prefix = stream.read(spec["size_bytes"])
            assert hashlib.sha256(prefix).hexdigest() == spec["sha256"] == checksum, f"TASK_01 document prefix changed: {name}"
        else:
            assert digest(path) == checksum, f"TASK_01 artifact changed: {name}"
    print(f"TASK_01 preserved: {len(old_entries) - len(append_only)} immutable files and {len(append_only)} original document prefixes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-task01", action="store_true")
    args = parser.parse_args()
    verify_task01()
    if args.verify_task01:
        return
    files = set(artifact_files(ROOT))
    files.update(p for p in (ROOT / "data/feasibility").rglob("*") if p.is_file())
    actual = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}
    target = ROOT / "SHA256SUMS"
    if args.verify:
        expected = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in target.read_text().splitlines()}
        assert set(actual) == set(expected), f"File coverage mismatch: {set(actual) ^ set(expected)}"
        assert actual == expected, f"Changed files: {[p for p in actual if actual[p] != expected[p]]}"
        print(f"TASK_02 SHA256SUMS verified: {len(actual)} files, including data/feasibility")
    else:
        target.write_text("".join(f"{checksum}  {name}\n" for name, checksum in actual.items()))
        print(f"TASK_02 SHA256SUMS generated: {len(actual)} files")


if __name__ == "__main__":
    main()
