"""Integrity checks reject drift while permitting the two Task 04 function edits."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.checksums_task04 import (AUTHORIZED, DOCUMENTS, artifact_files,
                                     check_current, digest, read_sums,
                                     validate_external_file, verify_external,
                                     verify_legacy)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def json_file(path, value):
    return write(path, json.dumps(value) + "\n")


def record(path, lines=None):
    return {"path": str(path), "bytes": path.stat().st_size,
            "lines": lines, "sha256": digest(path)}


def legacy_fixture(root):
    baseline_files = {}
    for name, (old_name, new_name) in AUTHORIZED.items():
        old = f'"""Preserved module."""\n\ndef {old_name}():\n    return 1\n\nOTHER = 3\n'
        new = f'"""Preserved module."""\n\ndef {new_name}():\n    return 2\n\nOTHER = 3\n'
        path = write(root / "specs/task03_pre_task04" / (name + ".txt"), old)
        baseline_files[name] = digest(path)
        write(root / name, new)
    for name in ("data/raw/train.jsonl", "eval/reader/results.json", "scripts/old.py"):
        baseline_files[name] = digest(write(root / name, "unchanged\n"))
    for task in (1, 2, 3):
        entries, docs = dict(baseline_files), {}
        for name in DOCUMENTS:
            text = (name + "\n") * task
            data = text.encode()
            entries[name] = hashlib.sha256(data).hexdigest()
            docs[name] = {"sha256": entries[name], "size_bytes": len(data)}
            write(root / name, text + "Task 04 append only.\n")
        write(root / f"specs/TASK{task:02d}_SHA256SUMS",
              "".join(f"{value}  {name}\n" for name, value in sorted(entries.items())))
        json_file(root / f"specs/TASK{task:02d}_document_baselines.json", docs)


def test_all_legacy_baselines_accept_scoped_changes_and_document_appends(tmp_path):
    legacy_fixture(tmp_path)
    result = verify_legacy(tmp_path)
    assert result["passed"]
    assert result["protected_data_and_eval_files"] == 2
    assert len(result["authorized_scope"]) == 2
    assert set(result["tasks"]) == {"TASK_01", "TASK_02", "TASK_03"}
    assert result["retrieval_runs"] == result["model_imports"] == 0


@pytest.mark.parametrize("name", list(AUTHORIZED))
def test_function_exterior_changes_are_rejected(tmp_path, name):
    legacy_fixture(tmp_path)
    path = tmp_path / name
    path.write_text(path.read_text().replace("OTHER = 3", "OTHER = 4"))
    with pytest.raises(ValueError, match="Unauthorized change outside"):
        verify_legacy(tmp_path)


def test_corrupted_snapshot_cannot_authorize_a_change(tmp_path):
    legacy_fixture(tmp_path)
    path = tmp_path / "specs/task03_pre_task04/src/build_traj.py.txt"
    path.write_text(path.read_text().replace("return 1", "return 9"))
    with pytest.raises(ValueError, match="snapshot changed"):
        verify_legacy(tmp_path)


@pytest.mark.parametrize("name", ["data/raw/train.jsonl", "eval/reader/results.json", "scripts/old.py"])
def test_all_legacy_data_eval_and_code_remain_immutable(tmp_path, name):
    legacy_fixture(tmp_path)
    write(tmp_path / name, "mutated\n")
    with pytest.raises(ValueError, match="immutable artifact changed"):
        verify_legacy(tmp_path)


@pytest.mark.parametrize("mutation", ["replacement", ""])
def test_document_rewrite_or_truncation_is_rejected(tmp_path, mutation):
    legacy_fixture(tmp_path)
    write(tmp_path / "README.md", mutation)
    with pytest.raises(ValueError, match="document prefix changed or shortened"):
        verify_legacy(tmp_path)


@pytest.mark.parametrize("lines", [
    f"{'a'*64}  same\n{'b'*64}  same\n",
    f"{'a'*64}  ../escape\n",
    f"{'a'*64}  /absolute\n",
    "broken  file\n",
])
def test_malformed_or_duplicate_checksum_entries_are_rejected(tmp_path, lines):
    path = write(tmp_path / "sums", lines)
    with pytest.raises(ValueError):
        read_sums(path)


def test_inventory_covers_new_formal_directories_and_excludes_runtime_and_external_links(tmp_path):
    root = tmp_path / "project"
    for name in (".gitignore", "README.md", "data/splits/ids.json", "new_formal/report.txt"):
        write(root / name, "keep")
    for name in ("SHA256SUMS", ".conda/env/bin/python", "logs/train.log", "eval/__pycache__/a.pyc"):
        write(root / name, "skip")
    outside = write(tmp_path / "outside.bin", "external model")
    (root / "weight-link").symlink_to(outside)
    (root / "outside-dir").symlink_to(outside.parent, target_is_directory=True)
    assert {p.relative_to(root).as_posix() for p in artifact_files(root)} == {
        ".gitignore", "README.md", "data/splits/ids.json", "new_formal/report.txt"}


def test_current_manifest_detects_missing_added_and_changed_files():
    with pytest.raises(ValueError) as error:
        check_current({"changed": "new", "added": "new"}, {"changed": "old", "missing": "old"})
    assert all(name in str(error.value) for name in ("changed", "added", "missing"))


def test_fast_check_does_not_claim_sha_verified_and_slow_detects_same_size_corruption(tmp_path):
    path = write(tmp_path / "rows.jsonl", "{}\n{}\n")
    expected = record(path, 2)
    assert validate_external_file(expected, tmp_path, slow=True)["actual_lines_verified"]
    path.write_text("[]\n[]\n")
    result = validate_external_file(expected, tmp_path)
    assert not result["sha256_verified"] and not result["actual_lines_verified"]
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_external_file(expected, tmp_path, slow=True)
    expected = record(path, 3)
    with pytest.raises(ValueError, match="line count mismatch"):
        validate_external_file(expected, tmp_path, slow=True)


def test_manifest_path_cannot_escape_via_symlink(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = write(tmp_path / "outside.jsonl", "{}\n")
    link = allowed / "rows.jsonl"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        validate_external_file(record(link, 1), allowed)


def external_fixture(root, external):
    trajectories = external / "data/traj_train"
    entries = {split: record(write(trajectories / f"{split}_tasks.jsonl", "{}\n"), 1)
               for split in ("train", "heldout")}
    entries["corpus"] = record(write(trajectories / "corpus_train.jsonl", "{}\n"), 1)
    entries["index"] = [record(write(trajectories / "bm25_train/index.npy", "dummy index"))]
    entries["split_ids"] = record(json_file(external / "data/splits/train_heldout_ids.json", {}))
    json_file(root / "data/splits/train_heldout_ids.json", {})
    trajectory = {"files": entries, "corpus": {"paragraphs": 1},
                  "statistics": {split: {"tasks": 1, "samples": 2} for split in ("train", "heldout")}}
    json_file(root / "eval/sft_prep/trajectory_stats.json", trajectory)
    json_file(trajectories / "manifest.json", trajectory)
    sft = {"files": {split: record(write(external / f"data/sft/{split}.jsonl", "{}\n{}\n"), 2)
                     for split in ("train", "heldout")},
           "trajectory_inputs": {split: entries[split] for split in ("train", "heldout")}}
    json_file(root / "eval/sft_prep/sft_data_stats.json", sft)
    model_root = external / "models/Qwen3-1.7B"
    path = write(model_root / "model.safetensors", "synthetic model")
    model = {"model_root": str(model_root), "status": "verified", "total_expected_bytes": path.stat().st_size,
             "files": [{"path": path.name, "status": "verified", "size_bytes": path.stat().st_size,
                        "sha256": digest(path), "actual_sha256": digest(path)}]}
    json_file(root / "eval/sft_prep/model_manifest.json", model)


def test_all_external_manifest_families_validate_and_detect_wrong_sft_provenance(tmp_path):
    root, external = tmp_path / "project", tmp_path / "external"
    external_fixture(root, external)
    result = verify_external(root, external, slow=True)
    assert result["passed"] and result["file_count"] == 8
    assert all(item["sha256_verified"] for item in result["files"])
    path = root / "eval/sft_prep/sft_data_stats.json"
    value = json.loads(path.read_text())
    value["trajectory_inputs"]["train"]["sha256"] = "0" * 64
    json_file(path, value)
    with pytest.raises(ValueError, match="input manifest differs"):
        verify_external(root, external)


def test_external_model_must_be_complete_and_extra_model_files_are_detected(tmp_path):
    root, external = tmp_path / "project", tmp_path / "external"
    external_fixture(root, external)
    write(external / "models/Qwen3-1.7B/partial.part", "unfinished")
    with pytest.raises(ValueError, match="coverage differs"):
        verify_external(root, external)
