#!/usr/bin/env python3
"""Run Task 05 CPU baselines on saved train-heldout steps, without retrieval."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
sys.path.insert(0, str(PROJECT))

from src.build_traj import oracle_action
from src.heldout_eval import HeldoutReplay, load_heldout_assets
from src.policies import bm25_policy, latest_policy, no_compress_policy, score_query


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=EXTERNAL / "data/traj_train")
    parser.add_argument("--rollout-dir", type=Path, default=EXTERNAL / "eval_rollouts")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "eval/sft_run")
    args = parser.parse_args()
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("run only on 5880 with CUDA_VISIBLE_DEVICES='' (CPU only)")
    if args.data_dir.resolve() != EXTERNAL / "data/traj_train":
        parser.error("only the preserved Task 04 train corpus and heldout task directory is allowed")
    if not args.rollout_dir.resolve().is_relative_to(EXTERNAL / "eval_rollouts"):
        parser.error("large rollouts must remain below external eval_rollouts")
    if not args.output_dir.resolve().is_relative_to(PROJECT / "eval/sft_run"):
        parser.error("small results must remain below eval/sft_run")
    policies = ("no_compress", "latest", "bm25", "oracle")
    destinations = [args.rollout_dir / policy / "steps.jsonl" for policy in policies]
    destinations += [args.output_dir / f"{policy}_results.json" for policy in policies]
    destinations += [args.output_dir / "cpu_baseline_results.json"]
    if any(path.exists() for path in destinations):
        parser.error("refusing to overwrite an existing baseline result")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    started, started_utc = time.monotonic(), datetime.now(timezone.utc).isoformat()
    tasks, corpus = load_heldout_assets(args.data_dir)
    print(f"loaded {len(tasks)} heldout tasks and {len(corpus)} train paragraphs", flush=True)
    import bm25s
    index = bm25s.BM25.load(str(args.data_dir / "bm25_train"), load_corpus=False)
    if index.scores["num_docs"] != len(corpus):
        raise ValueError("train index and corpus disagree")
    # Scores are computed once per task with the existing corpus-global IDF.
    scores = {task["task"]: score_query(index, " ".join(task["questions"])) for task in tasks}
    bm25_policies = {task_id: bm25_policy(index, scores=values) for task_id, values in scores.items()}
    results = {}
    for strategy in policies:
        stage = time.monotonic()
        replay = HeldoutReplay(tasks, corpus, strategy)
        rollout_path = args.rollout_dir / strategy / "steps.jsonl"
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with rollout_path.open("x", encoding="utf-8") as stream:
            while not replay.completed():
                batch = replay.next_batch()
                outputs = []
                for item in batch:
                    state = item["state"]
                    if strategy == "no_compress":
                        action = no_compress_policy(state)
                    elif strategy == "latest":
                        action = latest_policy()(state)
                    elif strategy == "bm25":
                        action = bm25_policies[item["task"]](state)
                    else:
                        action = oracle_action(state)
                    outputs.append({"raw_output": compact(action), "generation_backend": "cpu_policy"})
                records = replay.advance_batch(outputs)
                for record in records:
                    if strategy == "bm25":
                        state = record["state"]
                        record["bm25_scores"] = {str(pid): float(scores[record["task"]][pid])
                                                 for pid in state["pid_by_lid"].values()}
                    stream.write(compact(record) + "\n")
                count += len(records)
                stream.flush()
                print(f"{strategy}: step={batch[0]['step'] + 1}, active={len(batch)}, rows={count}", flush=True)
        result = replay.aggregate(baseline=results.get("no_compress"))
        result["run"] = {"elapsed_seconds": time.monotonic() - stage,
                         "steps_file": {**file_record(rollout_path), "lines": count}}
        assert result["combined"]["n_tasks"] == 1494
        assert result["combined"]["action_validity"]["invalid_count"] == 0
        assert result["combined"]["action_validity"]["over_budget_count"] == 0
        if strategy == "no_compress":
            assert result["combined"]["final_recall"]["mean"] == 1.0
            assert result["combined"]["evidence_loss"]["count"] == 0
        write_json(args.output_dir / f"{strategy}_results.json", result)
        results[strategy] = result
        del replay
    receipt = {"schema_version": 1, "strategies": results,
        "run": {"started_at_utc": started_utc, "elapsed_seconds": time.monotonic() - started,
                "maximum_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "input_files": [file_record(args.data_dir / name) for name in
                                ("heldout_tasks.jsonl", "corpus_train.jsonl")],
                "index_files": [file_record(path) for path in sorted((args.data_dir / "bm25_train").rglob("*"))
                                if path.is_file()],
                "code_files": [file_record(path) for path in
                               (Path(__file__).resolve(), PROJECT / "src/heldout_eval.py")],
                "saved_steps_only": True, "dev_inputs_read": False,
                "no_downloads_installs_api_or_gpu": True}}
    write_json(args.output_dir / "cpu_baseline_results.json", receipt)
    print(compact({"completed": True, "elapsed_seconds": receipt["run"]["elapsed_seconds"],
                   "strategies": {name: result["combined"]["evidence_loss"]
                                  for name, result in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
