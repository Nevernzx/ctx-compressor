"""Four-generation immutability and exhaustive external delivery coverage."""
import hashlib
import json

import pytest

from scripts import checksums_task05 as checks
from test_checksums_task04 import legacy_fixture, write, json_file, record


def test_uniform_heldout_subset_allows_different_task_coverage_between_modes():
    chosen = [{"source_line_1based": 1, "meta": {"task": 0, "step": 0,
               "mode": "reselect", "tier": "mid", "split": "heldout"}},
              {"source_line_1based": 2, "meta": {"task": 1, "step": 0,
               "mode": "append", "tier": "loose", "split": "heldout"}}]
    checks.validate_subset_ids(chosen, "heldout", 249)
    with pytest.raises(ValueError, match="Duplicate"):
        checks.validate_subset_ids(chosen + [chosen[0]], "heldout", 249)


def four_task_fixture(root):
    legacy_fixture(root)
    write(root / "src/sft_protocol.py", "task04 original\n")
    write(root / "eval/sft_prep/sft_data_stats.json", "{}\n")
    entries = checks.previous.current_sums(root)
    docs = {name: {"size_bytes": (root / name).stat().st_size, "sha256": entries[name]}
            for name in checks.previous.DOCUMENTS}
    json_file(root / "specs/TASK04_document_baselines.json", docs)
    write(root / "specs/TASK04_SHA256SUMS",
          "".join(f"{sha}  {name}\n" for name, sha in entries.items()))


def test_four_generations_preserved_and_new_documents_can_append(tmp_path):
    four_task_fixture(tmp_path)
    for name in checks.previous.DOCUMENTS:
        with (tmp_path / name).open("a") as stream:
            stream.write("Task 05 addition.\n")
    result = checks.verify_legacy(tmp_path)
    assert set(result["tasks"]) == {"TASK_01", "TASK_02", "TASK_03", "TASK_04"}
    assert result["tasks"]["TASK_04"]["authorized_function_changes"] == 0


@pytest.mark.parametrize("name", ["src/build_traj.py", "tests/test_build_traj.py",
                                  "src/sft_protocol.py", "eval/sft_prep/sft_data_stats.json"])
def test_task04_functions_and_artifacts_now_fully_frozen(tmp_path, name):
    four_task_fixture(tmp_path)
    path = tmp_path / name
    path.write_text(path.read_text().replace("return 2", "return 999") if name in checks.previous.AUTHORIZED
                    else "changed\n")
    with pytest.raises(ValueError, match="TASK_04: immutable artifact changed"):
        checks.verify_legacy(tmp_path)


def test_task04_document_append_cannot_be_rewritten(tmp_path):
    four_task_fixture(tmp_path)
    path = tmp_path / "README.md"
    path.write_text(path.read_text().replace("Task 04", "Task XX"))
    with pytest.raises(ValueError, match="TASK_04: document prefix"):
        checks.verify_legacy(tmp_path)


def test_task04_archive_cannot_omit_earlier_protected_path(tmp_path):
    four_task_fixture(tmp_path)
    path = tmp_path / "specs/TASK04_SHA256SUMS"
    path.write_text("\n".join(line for line in path.read_text().splitlines()
                               if not line.endswith("  scripts/old.py")) + "\n")
    with pytest.raises(ValueError, match="omitted"):
        checks.verify_legacy(tmp_path)


def external_fixture(root, external):
    roots = [external / relative for relative in checks.FINAL_ROOTS]
    for path in roots:
        path.mkdir(parents=True)
    train = record(write(roots[0] / "train_20k.jsonl", "{}\n{}\n"), 2)
    loss = record(write(roots[0] / "heldout_eval_loss_1k.jsonl", "{}\n"), 1)
    ids = record(write(roots[0] / "sample_ids.json", "{}\n"))
    records = [train, loss, ids,
               record(write(roots[0] / "checkpoints/checkpoint-500/adapter.safetensors", "weights")),
               record(write(roots[1] / "benchmark.log", "complete\n")),
               record(write(roots[2] / "original/steps.jsonl", "{}\n"), 1)]
    sample = {"files": {"train": train, "eval_loss": loss, "sample_ids": ids}}
    json_file(root / "eval/sft_run/sample_manifest.json", sample)
    manifest = {"schema_version": 1, "complete": True, "roots": list(map(str, roots)), "files": records}
    json_file(root / "eval/sft_run/external_artifact_manifest.json", manifest)
    return roots, manifest


def test_final_artifacts_have_exhaustive_coverage_and_full_slow_hashes(tmp_path):
    root, external = tmp_path / "project", tmp_path / "external"
    external_fixture(root, external)
    result = checks.verify_final_artifacts(root, external, slow=True)
    assert result["passed"] and result["file_count"] == 6 and result["coverage_complete"]
    assert all(item["sha256_verified"] for item in result["files"])


@pytest.mark.parametrize("mutation", ["unrecorded", "missing", "changed", "wrong_lines", "duplicate",
                                        "incomplete", "root_escape", "symlink", "partial"])
def test_final_external_drift_is_detected(tmp_path, mutation):
    root, external = tmp_path / "project", tmp_path / "external"
    roots, manifest = external_fixture(root, external)
    if mutation == "unrecorded":
        write(roots[0] / "extra.bin", "unknown")
    elif mutation == "missing":
        (roots[1] / "benchmark.log").unlink()
    elif mutation == "changed":
        (roots[0] / "train_20k.jsonl").write_text("[]\n[]\n")
    elif mutation == "wrong_lines":
        manifest["files"][-1]["lines"] = 2
    elif mutation == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif mutation == "incomplete":
        manifest["complete"] = False
    elif mutation == "root_escape":
        manifest["roots"][0] = str(external)
    elif mutation == "symlink":
        (roots[0] / "link").symlink_to(roots[1] / "benchmark.log")
    elif mutation == "partial":
        write(roots[0] / "adapter.partial", "unfinished")
    json_file(root / "eval/sft_run/external_artifact_manifest.json", manifest)
    with pytest.raises(ValueError):
        checks.verify_final_artifacts(root, external, slow=True)


def test_final_manifest_cannot_change_frozen_subset_metadata(tmp_path):
    root, external = tmp_path / "project", tmp_path / "external"
    _, manifest = external_fixture(root, external)
    manifest["files"][0]["sha256"] = hashlib.sha256(b"different").hexdigest()
    json_file(root / "eval/sft_run/external_artifact_manifest.json", manifest)
    with pytest.raises(ValueError, match="disagrees with frozen subset"):
        checks.verify_final_artifacts(root, external, slow=False)
