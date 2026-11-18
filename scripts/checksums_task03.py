"""Extend delivery hashes while preserving TASK_01/02 immutable artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

from checksums import artifact_files, digest
from checksums_task02 import verify_task01

ROOT = Path(__file__).resolve().parents[1]


def verify_task02():
    entries = {}
    for line in (ROOT / "specs/TASK02_SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        assert name not in entries, f"Duplicate archived entry: {name}"
        entries[name] = checksum
    prefixes = json.loads((ROOT / "specs/TASK02_document_baselines.json").read_text())
    assert set(prefixes) == {"README.md", "NOTES.md", "HANDOFF.md"}
    for name, checksum in entries.items():
        path = ROOT / name
        if name in prefixes:
            baseline = prefixes[name]
            with path.open("rb") as stream:
                prefix = stream.read(baseline["size_bytes"])
            assert len(prefix) == baseline["size_bytes"], f"TASK_02 document shortened: {name}"
            assert hashlib.sha256(prefix).hexdigest() == baseline["sha256"] == checksum, f"TASK_02 document prefix changed: {name}"
        else:
            assert digest(path) == checksum, f"TASK_02 immutable artifact changed: {name}"
    print(f"TASK_02 preserved: {len(entries) - len(prefixes)} immutable files and {len(prefixes)} original document prefixes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-previous", action="store_true")
    args = parser.parse_args()
    verify_task01()
    verify_task02()
    if args.verify_previous:
        return
    files = set(artifact_files(ROOT))
    for directory in ("data/feasibility", "data/reader"):
        files.update(path for path in (ROOT / directory).rglob("*") if path.is_file())
    actual = {path.relative_to(ROOT).as_posix(): digest(path)
              for path in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())}
    target = ROOT / "SHA256SUMS"
    if args.verify:
        expected = {line.split("  ", 1)[1]: line.split("  ", 1)[0]
                    for line in target.read_text().splitlines()}
        assert set(actual) == set(expected), f"File coverage mismatch: {sorted(set(actual) ^ set(expected))}"
        assert actual == expected, f"Changed files: {[name for name in actual if actual[name] != expected[name]]}"
        print(f"TASK_03 SHA256SUMS verified: {len(actual)} files, including data/reader; external model weights excluded")
    else:
        target.write_text("".join(f"{checksum}  {name}\n" for name, checksum in actual.items()))
        print(f"TASK_03 SHA256SUMS generated: {len(actual)} files, including data/reader; external model weights excluded")


if __name__ == "__main__":
    main()
