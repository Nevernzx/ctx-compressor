#!/usr/bin/env python3
"""Audit Task 04 hop references without retrieval, tokenization, or model calls.

Run only on 5880 after sourcing scripts/reader_environment.sh, then setting
CUDA_VISIBLE_DEVICES='' and invoking the existing ctxc Python interpreter.
Reads legacy assets in place and writes only eval/sft_prep/reference_fix_checks.json.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET


PROJECT = Path(__file__).resolve().parents[1]
SNAPSHOT = PROJECT / "specs/task03_pre_task04"
sys.path.insert(0, str(PROJECT))
from src.build_traj import hop_queries


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def jsonlines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def manifest():
    return {name: digest for digest, name in (
        line.split("  ", 1) for line in
        (PROJECT / "specs/TASK03_SHA256SUMS").read_text().splitlines() if line)}


def function_parts(path, name):
    source = path.read_bytes()
    parsed = ast.parse(source)
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    return node, b"".join(lines[:node.lineno - 1] + lines[node.end_lineno:])


def check_authorized_scope(baseline):
    changed = []
    for rel, old_name, new_name in (
        ("src/build_traj.py", "hop_queries", "hop_queries"),
        ("tests/test_build_traj.py", "test_invalid_or_forward_reference_is_not_silently_retrieved",
         "test_out_of_range_reference_is_preserved"),
    ):
        before = SNAPSHOT / (rel + ".txt")
        after = PROJECT / rel
        _, old_outside = function_parts(before, old_name)
        _, new_outside = function_parts(after, new_name)
        changed.append({"path": rel, "authorized_function": new_name,
                        "snapshot_sha256": sha256(before), "baseline_sha256": baseline[rel],
                        "new_sha256": sha256(after),
                        "snapshot_matches_baseline": sha256(before) == baseline[rel],
                        "outside_function_byte_identical": old_outside == new_outside})
    return {"passed": all(x["snapshot_matches_baseline"] and
                           x["outside_function_byte_identical"] for x in changed),
            "files": changed}


def load_old_hop_queries():
    node, _ = function_parts(SNAPSHOT / "src/build_traj.py.txt", "hop_queries")
    namespace = {"re": re}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(SNAPSHOT / "src/build_traj.py.txt"), "exec"), namespace)
    return namespace["hop_queries"]


def check_legacy_files(baseline):
    doc_baselines = json.loads((PROJECT / "specs/TASK03_document_baselines.json").read_text())
    files = []
    for rel, expected in baseline.items():
        path = PROJECT / rel
        actual = sha256(path)
        entry = {"path": rel, "baseline_sha256": expected, "actual_sha256": actual,
                 "bytes": path.stat().st_size, "byte_identical": expected == actual}
        if rel in ("src/build_traj.py", "tests/test_build_traj.py"):
            entry["status"] = "authorized_function_change"
        elif rel in doc_baselines:
            saved = doc_baselines[rel]
            with path.open("rb") as stream:
                prefix = stream.read(saved["size_bytes"])
            prefix_ok = (len(prefix) == saved["size_bytes"] and
                         hashlib.sha256(prefix).hexdigest() == expected == saved["sha256"])
            entry.update(prefix_byte_identical=prefix_ok, prefix_bytes=saved["size_bytes"],
                         status="unchanged_or_append_only_document" if prefix_ok else "mismatch")
        else:
            entry["status"] = "unchanged" if actual == expected else "mismatch"
        files.append(entry)
    protected = [x for x in files if x["path"].startswith(("data/", "eval/"))]
    return {"passed": all(x["status"] != "mismatch" for x in files),
            "manifest_entries": len(files),
            "exactly_unchanged_files": sum(x["byte_identical"] for x in files),
            "protected_data_and_eval_files": len(protected),
            "protected_data_and_eval_all_byte_identical": all(x["byte_identical"] for x in protected),
            "files": files}


def call(function, example, keys, **kwargs):
    try:
        return function(example, keys, **kwargs), None
    except (KeyError, ValueError, TypeError) as error:
        return None, {"type": type(error).__name__, "reason": str(error)}


def check_dev(old_function):
    path = PROJECT / "data/raw/musique/musique_ans_v1.0_dev.jsonl"
    examples = list(jsonlines(path))
    keys = {}
    for example in examples:
        for p in example["paragraphs"]:
            key = (p["title"], p["paragraph_text"])
            keys.setdefault(key, len(keys))
    old_digest, new_digest = hashlib.sha256(), hashlib.sha256()
    old_nbytes = new_nbytes = nhops = 0
    mismatches, new_by_id, stats = [], {}, {}
    for example in examples:
        old, old_error = call(old_function, example, keys)
        new, new_error = call(hop_queries, example, keys, stats=stats)
        before = compact_bytes({"id": example["id"], "query_pid": old, "error": old_error}) + b"\n"
        after = compact_bytes({"id": example["id"], "query_pid": new, "error": new_error}) + b"\n"
        old_digest.update(before)
        new_digest.update(after)
        old_nbytes += len(before)
        new_nbytes += len(after)
        if before != after or old_error or new_error:
            mismatches.append({"id": example["id"], "old": old, "new": new,
                               "old_error": old_error, "new_error": new_error})
        if new is not None:
            nhops += len(new)
            new_by_id[example["id"]] = new
    report = {"passed": not mismatches, "questions": len(examples), "hops": nhops,
              "corpus_paragraphs": len(keys), "mismatch_count": len(mismatches),
              "mismatches": mismatches, "old_sha256": old_digest.hexdigest(),
              "new_sha256": new_digest.hexdigest(), "old_serialized_bytes": old_nbytes,
              "new_serialized_bytes": new_nbytes,
              "serialization": "Per raw dev line: UTF-8 compact JSON {id,query_pid,error} + LF; compare every line before hashing",
              "preserved_original_reference_occurrences": stats.get("preserved_reference_count", 0)}
    return report, new_by_id


def check_saved_tasks(new_by_id):
    path = PROJECT / "data/feasibility/tasks_dev150.jsonl"
    expected_digest, saved_digest = hashlib.sha256(), hashlib.sha256()
    ntasks = nquestions = nqueries = 0
    example_ids, mismatches = set(), []
    for task in jsonlines(path):
        ntasks += 1
        nquestions += len(task["example_ids"])
        expected = []
        for qi, qid in enumerate(task["example_ids"]):
            example_ids.add(qid)
            for hi, (query, pid) in enumerate(new_by_id[qid]):
                expected.append([qid, qi, hi, query, pid])
        saved = [[step["example_id"], step["question_index"], step["hop_index"],
                  step["query"], step["gold"]] for step in task["steps"]]
        before, after = compact_bytes(expected) + b"\n", compact_bytes(saved) + b"\n"
        expected_digest.update(before)
        saved_digest.update(after)
        nqueries += len(saved)
        if before != after:
            mismatches.append({"task": task["task"], "expected": expected, "saved": saved})
    return {"passed": not mismatches, "tasks": ntasks, "questions": nquestions,
            "unique_questions": len(example_ids), "queries": nqueries,
            "mismatch_count": len(mismatches), "mismatches": mismatches,
            "new_function_sha256": expected_digest.hexdigest(),
            "saved_sequence_sha256": saved_digest.hexdigest(),
            "serialization": "Per task: UTF-8 compact JSON ordered [example_id,question_index,hop_index,query,gold] rows + LF",
            "saved_file_sha256": sha256(path)}


def check_train(old_function):
    stats, accepted, rejected, changed, affected, errors = {}, Counter(), [], [], [], []
    total = hops = 0
    for example in jsonlines(PROJECT / "data/raw/musique/musique_ans_v1.0_train.jsonl"):
        total += 1
        keys = {(p["title"], p["paragraph_text"]): i
                for i, p in enumerate(example["paragraphs"])}
        old, old_error = call(old_function, example, keys)
        prior_count = stats.get("preserved_reference_count", 0)
        new, new_error = call(hop_queries, example, keys, stats=stats)
        if old_error is None:
            accepted["old"] += 1
        else:
            rejected.append({"id": example["id"], **old_error})
        if new_error is None:
            accepted["new"] += 1
            hops += len(new)
        else:
            errors.append({"id": example["id"], **new_error})
        if old_error is None and compact_bytes(old) != compact_bytes(new):
            changed.append({"id": example["id"], "old": old, "new": new})
        if stats.get("preserved_reference_count", 0) > prior_count or old_error != new_error:
            events = stats.get("preserved_references", [])[prior_count:]
            affected.append({"id": example["id"], "old_error": old_error,
                             "new_error": new_error, "query_pid": new,
                             "original_reference_occurrences": len(events)})
            for event in events:
                event["resolved_query"] = new[event["hop_index"]][0] if new else None
    occurrence_ids = sorted({event["example_id"] for event in stats.get("preserved_references", [])})
    return {"passed": not errors and not changed, "questions": total, "hops": hops,
            "old_accepted": accepted["old"], "new_accepted": accepted["new"],
            "newly_accepted": accepted["new"] - accepted["old"],
            "old_rejected": rejected, "new_rejected": errors,
            "previously_accepted_changed": changed, "affected_examples": affected,
            "affected_example_ids": sorted(x["id"] for x in affected),
            "preserved_reference_example_ids": occurrence_ids,
            "preserved_reference_occurrences": stats.get("preserved_reference_count", 0),
            "reference_counts": dict(Counter(x["reference"] for x in stats.get("preserved_references", []))),
            "occurrences": stats.get("preserved_references", []),
            "counting_scope": "Original hop question occurrences only; inserted answer text is not rescanned"}


def check_test_results(xml_path):
    before = list(ET.parse(PROJECT / "eval/reader/tests.xml").getroot().iter("testcase"))
    after = list(ET.parse(xml_path).getroot().iter("testcase"))
    failed = [node.attrib for node in after if any(
        child.tag in ("failure", "error", "skipped") for child in node)]
    current_names = {(node.get("classname"), node.get("name")) for node in after}
    missing = []
    for node in before:
        name = node.get("name").replace(
            "test_invalid_or_forward_reference_is_not_silently_retrieved",
            "test_out_of_range_reference_is_preserved")
        key = (node.get("classname"), name)
        if key not in current_names:
            missing.append(list(key))
    return {"passed": not failed and not missing, "baseline_tests": len(before),
            "current_tests": len(after), "added_tests": len(after) - len(before),
            "baseline_cases_missing": missing, "failed_or_skipped": failed,
            "xml_path": str(xml_path.relative_to(PROJECT)), "xml_sha256": sha256(xml_path),
            "authorized_test_rename": "3 parameterized out-of-range cases now assert preservation and count"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-xml", type=Path,
                        default=PROJECT / "eval/sft_prep/reference_unit_tests.xml")
    args = parser.parse_args()
    if PROJECT != Path("/root/workplace/ctx-compressor"):
        parser.error("execute on 5880 in /root/workplace/ctx-compressor only")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("CUDA_VISIBLE_DEVICES='' is required")
    if Path(sys.prefix).name != "ctxc":
        parser.error("use the existing ctxc interpreter")
    start = time.monotonic()
    output = PROJECT / "eval/sft_prep/reference_fix_checks.json"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.executable, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "retrieval_runs": 0, "model_runs": 0, "legacy_results_written": False,
              "script_sha256": sha256(Path(__file__))}
    try:
        baseline = manifest()
        report["authorized_change_scope"] = check_authorized_scope(baseline)
        if not report["authorized_change_scope"]["passed"]:
            raise RuntimeError("source/test modification exceeds authorized function scope")
        old_function = load_old_hop_queries()
        report["all_dev"], new_by_id = check_dev(old_function)
        if not report["all_dev"]["passed"]:
            raise RuntimeError("dev query/pid bytes changed: stop")
        report["task02_saved_queries"] = check_saved_tasks(new_by_id)
        if not report["task02_saved_queries"]["passed"]:
            raise RuntimeError("saved task02 query/pid bytes differ: stop")
        print("Dev query/pid checks passed; scanning train references.", flush=True)
        report["train"] = check_train(old_function)
        report["unit_tests"] = check_test_results(args.tests_xml.resolve())
        print("Train references checked; hashing every Task 03 manifest entry in place.", flush=True)
        report["legacy_integrity"] = check_legacy_files(baseline)
        report["passed"] = all(value["passed"] for value in report.values()
                               if isinstance(value, dict) and "passed" in value)
    except Exception as error:
        report.update(passed=False, error={"type": type(error).__name__, "reason": str(error)})
    report["elapsed_seconds"] = time.monotonic() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(output),
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
