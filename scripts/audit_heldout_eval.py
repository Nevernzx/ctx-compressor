#!/usr/bin/env python3
"""Independently reconstruct CPU baseline states and Task 02 metrics from disk."""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
sys.path.insert(0, str(PROJECT))
from src.build_traj import apply_action, make_sample, oracle_action
from src.feasibility import aggregate_tasks, summarize_task
from src.heldout_eval import BUDGETS, MODES, load_heldout_assets
from src.policies import bm25_policy, latest_policy, no_compress_policy, score_query
from scripts.heldout_eval import file_record, write_json


def main():
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("only execute CPU audit on 5880")
    started = time.monotonic()
    tasks, corpus = load_heldout_assets(EXTERNAL / "data/traj_train")
    tasks = {task["task"]: task for task in tasks}
    ntok = {i: p["ntok"] for i, p in enumerate(corpus)}
    import bm25s
    index = bm25s.BM25.load(str(EXTERNAL / "data/traj_train/bm25_train"), load_corpus=False)
    scores = {task_id: score_query(index, " ".join(task["questions"])) for task_id, task in tasks.items()}
    policies = {task_id: bm25_policy(index, scores=values) for task_id, values in scores.items()}
    source_receipt = json.loads((PROJECT / "eval/sft_run/cpu_baseline_results.json").read_text())
    audit = {}
    no_compress = {}
    for strategy in ("no_compress", "latest", "bm25", "oracle"):
        grouped = defaultdict(list)
        rolling = {}
        path = EXTERNAL / "eval_rollouts" / strategy / "steps.jsonl"
        evidence = file_record(path)
        assert evidence == {key: value for key, value in
            source_receipt["strategies"][strategy]["run"]["steps_file"].items() if key != "lines"}
        result = json.loads((PROJECT / "eval/sft_run" / f"{strategy}_results.json").read_text())
        records = 0
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                task = tasks[record["task"]]
                key = (record["mode"], record["budget_tier"], record["task"])
                entry = rolling.setdefault(key, {"kept": [], "lid": {}, "gold_upto": set(), "samples": []})
                step = task["steps"][record["step"]]
                assert record["step"] == len(entry["samples"])
                assert record["budget"] == BUDGETS[key[1]]
                assert record["returned"] == step["new"]
                entry["gold_upto"].add(step["gold"])
                state = make_sample(task["questions"], entry["kept"], step["new"], entry["gold_upto"],
                                    corpus, record["effective_budget"], key[0], entry["lid"])
                sample = record["state"]
                for field in ("pid_by_lid", "ntok"):
                    sample[field] = {int(k): v for k, v in sample[field].items()}
                assert {k: sample[k] for k in state if k != "text"} == {k: v for k, v in state.items() if k != "text"}
                action = json.loads(record["raw_output"])
                policy = (no_compress_policy if strategy == "no_compress" else
                          latest_policy() if strategy == "latest" else
                          policies[record["task"]] if strategy == "bm25" else oracle_action)
                assert action == policy(state)
                keep = apply_action(state, action)
                assert keep == sample["keep_after"] and keep is not None
                assert record["parsed_action"] == action
                assert record["action_valid"] and record["invalid_reason"] is None
                assert record["kept_before"] == entry["kept"]
                pids = [state["pid_by_lid"][i] for i in keep]
                assert pids == record["keep_after"]
                assert sum(ntok[p] for p in pids) == record["used_after"]
                gold = set(state["gold"])
                recall = len(gold.intersection(keep)) / len(gold) if gold else 1.0
                assert recall == sample["recall"] == record["recall"]
                entry["kept"] = pids
                entry["samples"].append(sample)
                entry["budget"] = record["effective_budget"]
                if strategy == "bm25":
                    assert record["bm25_scores"] == {str(pid): float(scores[record["task"]][pid])
                                                     for pid in state["pid_by_lid"].values()}
                records += 1
        assert set(rolling) == {(mode, tier, task) for mode in MODES for tier in BUDGETS for task in tasks}
        for (mode, tier, task_id), entry in rolling.items():
            task = tasks[task_id]
            question_gold = [set() for _ in task["questions"]]
            for step in task["steps"]:
                question_gold[step["question_index"]].add(step["gold"])
            grouped[(mode, tier)].append(summarize_task(entry["samples"], task["steps"], question_gold,
                                                       ntok, entry["budget"]))
        for (mode, tier), metrics in grouped.items():
            aggregate = aggregate_tasks(metrics, baseline=no_compress.get((mode, tier)))
            if strategy == "no_compress":
                no_compress[(mode, tier)] = aggregate
            saved = result["configs"][f"{strategy}|{mode}|{tier}"]["metrics"]
            assert {key: saved[key] for key in aggregate} == aggregate
        assert records == result["combined"]["n_steps"] == 13956
        assert sum(m["evidence_loss"] for metrics in grouped.values() for m in metrics) == result["combined"]["evidence_loss"]["count"]
        audit[strategy] = {"n_trajectories": len(rolling), "n_step_records": records,
                           "sha256": evidence["sha256"], "verified": True}
        print(f"{strategy}: all {records} recorded steps independently reconstructed", flush=True)
    receipt = {"schema_version": 1, "strategies": audit, "elapsed_seconds": time.monotonic() - started,
               "checks": ["saved heldout/train inputs only", "full six-cell task coverage",
                          "saved retrieval steps", "closed-loop local IDs and retained contexts",
                          "original CPU policy action equality", "train-index BM25 scores",
                          "existing apply_action", "available-gold step recalls", "Task 02 aggregates",
                          "task-final loss count", "external rollout sha256"],
               "no_gpu_api_downloads_installs_or_dev_inputs": True}
    write_json(PROJECT / "eval/sft_run/cpu_baseline_audit.json", receipt)


if __name__ == "__main__":
    main()
