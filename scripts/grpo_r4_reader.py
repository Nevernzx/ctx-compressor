"""Construct and score the registered six-policy, cascade-off Task 18 reader batches."""
import argparse
from collections import Counter
from itertools import groupby
import math
import os
from pathlib import Path
from unittest.mock import patch

from scripts import grpo_r4_common as c
from scripts import grpo_r4_eval as evaluation
from scripts import reader_diag_metrics as diag
from scripts import reader_heldout_report as report
from scripts import reader_heldout_v2 as engine
from scripts import run_reader_heldout as reader
from scripts.rollout_model import file_record
from src import reader_heldout_inputs as inputs

NAMES = tuple(f"grpo_r4:checkpoint-{step}" for step in c.CHECKPOINT_ORDER) + (c.R2_POLICY, c.V3_POLICY)
N_TASKS = 249
BATCH = 144
N_PREDICTIONS = N_TASKS * BATCH
PREFLIGHT = c.READER / "input_preflight.json"
REQUESTS = c.READER / "requests.jsonl"
FINALS = c.READER / "final_context_counts.json"
READER_CONFIG = c.SMALL / "reader_run_config.json"
RESULT = c.SMALL / "reader_f1_results.json"
SHORT = c.ROOT / "rt/g18"
M12 = c.PROJECT / "eval/reader_heldout/TASK12_SHA256SUMS"
M16 = c.PROJECT / "eval/reader_heldout_v2/TASK16_SHA256SUMS"
HIST_PRED = c.ROOT / "reader_heldout_v2/predictions.jsonl"


def batches(stream):
    seen = set()
    for task, rows in groupby(stream, key=lambda r: r["task"]):
        batch = list(rows)
        expected = [(name, mode, tier, qi) for name in NAMES for mode, tier in inputs.CELLS for qi in range(4)]
        got = [(r["strategy"], r["mode"], r["tier"], r["question_index"]) for r in batch]
        c.require(task not in seen and got == expected, f"Unexpected 144-request batch order at task {task}")
        for r in batch:
            c.require(r["prompt_tokens"] == len(r["token_ids"]) > 0, "Prompt token identity differs")
            inputs.check_length(r["prompt_tokens"], (task, r["config"], r["question_index"]))
        seen.add(task)
        yield task, batch


def new_sources():
    chosen = c.read(evaluation.SELECTED)
    c.require(len(chosen["candidates"]) == 4 and chosen["strategy"] == evaluation.select_checkpoint(chosen["candidates"])["strategy"],
              "F selection changed")
    sources = []
    for step in c.CHECKPOINT_ORDER:
        name = f"checkpoint-{step}"
        run, metrics, _ = evaluation.check_eval(step)
        signature = run["artifacts"][name + "_results.json"]
        raw = metrics["steps_artifact"]
        sources.append(inputs.Source("grpo_r4:" + name, Path(signature["path"]), signature["sha256"],
                                     Path(raw["path"]), raw["sha256"], "Task18 F checked receipt"))
    return sources + evaluation.historical()


def prepare():
    c.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "G input generation requires CPU")
    c.expect_new(c.READER)
    c.READER.mkdir()
    aliases = {path: c.signature(path, M12) for path in (engine.R12, reader.STANDARD / "input_preflight/smoke.jsonl")}
    baseline = c.signature(HIST_PRED, M16)
    old_cfg_path = c.PROJECT / "eval/reader_heldout_v2/run_config.json"
    old_cfg_record = c.signature(old_cfg_path, M16)
    c.require(c.read(old_cfg_path)["effective_disable_cascade_attn"] is True, "Task 16 reader reference lacks cascade-off proof")
    historical_tokens = {
        tuple(r[k] for k in report.IDENTITY): (r["prompt_sha1"], r["token_ids"])
        for r in inputs.rows(engine.R12) if r["strategy"] in (c.R2_POLICY, c.V3_POLICY)
    }
    c.require(len(historical_tokens) == 11952, "Unexpected Task 12 reference coverage")
    index = inputs.sums(c.PROJECT / "SHA256SUMS")
    corpus, tasks, task_sources = inputs.load_task_inputs(index)
    sources = new_sources()
    c.require(tuple(s.name for s in sources) == NAMES, "Wrong source strategy order")
    ntok = [p["ntok"] for p in corpus]
    finals, costs, final_stats, trace_files = {}, {}, {}, {}
    for source in sources:
        kept, costs[source.name] = inputs.replay(source, tasks, ntok)
        finals[source.name] = kept
        final_stats[source.name] = {
            f"{mode}|{tier}|{tid}": {"segments": len(pids), "nongold": sum(p not in tasks[tid]["all_gold"] for p in pids)}
            for ((mode, tier), tid), pids in kept.items()
        }
        trace_files[source.name] = {"metrics": file_record(source.metrics), "steps": file_record(source.steps)}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(c.ROOT / "models/Qwen3-8B"), local_files_only=True, trust_remote_code=False)

    def stream():
        source = inputs.make_requests(sources, finals, tasks, corpus, tok, False)
        count = 0
        for task, batch in batches(source):
            c.require(task == count, "G changed the task order")
            for row in batch:
                if row["strategy"] in (c.R2_POLICY, c.V3_POLICY):
                    key = tuple(row[k] for k in report.IDENTITY)
                    c.require(historical_tokens.pop(key) == (row["prompt_sha1"], row["token_ids"]),
                              "r2/v3 request differs from the signed historical prompt")
                yield row
            count += 1
        c.require(count == N_TASKS and not historical_tokens, "Missing G task or historical comparison request")

    requests = inputs.export_requests(REQUESTS, stream())
    c.require(requests["n"] == N_PREDICTIONS and set(requests["n_by_config"].values()) == {996},
              "Reader request denominator differs")
    c.write_json_exclusive(FINALS, final_stats)
    c.write_json_exclusive(PREFLIGHT, {
        "status": "passed", "requests": requests, "batch_size": BATCH, "n_tasks": N_TASKS, "strategies": list(NAMES),
        "historical_prompt_matches": 11952, "historical_prompt_mismatches": 0, "input_truncations": 0,
        "historical_requests": aliases[engine.R12], "smoke": aliases[reader.STANDARD / "input_preflight/smoke.jsonl"],
        "task16_predictions": baseline, "task16_reader_config": old_cfg_record, "source_tasks": task_sources,
        "trace_files": trace_files, "final_context_file": file_record(FINALS), "costs": costs,
        "df_opening": reader.original.root_df()
    })
    print(f"G CPU preflight: {N_TASKS} tasks, {N_PREDICTIONS} requests, {requests['max_prompt_tokens']} max prompt tokens")


def preflight():
    info = c.read(PREFLIGHT)
    rq = info["requests"]
    c.require(info["status"] == "passed" and info["n_tasks"] == N_TASKS and rq["n"] == N_PREDICTIONS
              and rq["path"] == str(REQUESTS) and info["batch_size"] == BATCH
              and info["historical_prompt_mismatches"] == 0 and info["historical_prompt_matches"] == 11952
              and info["input_truncations"] == 0 and info["strategies"] == list(NAMES), "G input proof incomplete")
    inputs.verify(REQUESTS, rq["sha256"])
    c.require(file_record(FINALS) == info["final_context_file"], "G final contexts changed")
    return info


def run():
    info = preflight()
    c.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
              "Task 18 reader requires GPU 1")
    c.expect_new(c.READER / "ipc_check.json", READER_CONFIG)
    with patch.multiple(engine, WORK=c.READER, PRE=PREFLIGHT, SHORT=SHORT):
        check = engine.deployable_ipc()
        c.write_json_exclusive(c.READER / "ipc_check.json", check)
        print(f"IPC {check['path_bytes']} path bytes / {check['uri_bytes']} URI bytes (<100)", flush=True)
        result = engine.run_engine(c.READER, READER_CONFIG, batches(inputs.rows(REQUESTS)), N_TASKS, N_PREDICTIONS,
                                   provenance={"path": str(PREFLIGHT), "sha256": c.digest(PREFLIGHT),
                                               "batch_size": BATCH, "strategies": info["strategies"]})
    c.require(result["effective_disable_cascade_attn"] is True and result["engine_shutdown"],
              "Reader settings or engine shutdown receipt is missing")
    reader.publish(c.READER, {"status": "complete", "stage": "finished", "predictions": N_PREDICTIONS})


def per_cell_pair(left, right, data):
    pooled = report.paired(left, right, data)
    cells = {}
    for cell in report.CELLS:
        a, b = data[left][cell], data[right][cell]
        x, y = report.average(list(a.values())), report.average(list(b.values()))
        cells[cell] = {"n": x["n"], "left_scores": x, "right_scores": y,
                       "f1_difference_pp": 100 * (x["f1"] - y["f1"]),
                       "em_difference_pp": 100 * (x["em"] - y["em"])}
    return {**pooled, "cells": cells}


def decomposition(left, data, context):
    right = c.R2_POLICY
    transitions = {tag: [] for tag in ("false_to_false", "false_to_true", "true_to_false", "true_to_true")}
    bins = {tag: [] for tag in ("<=-1", "0", "1", ">=2")}
    for cell in report.CELLS:
        for key, a in data[right][cell].items():
            b = data[left][cell][key]
            difference = b.f1 - a.f1
            label = f"{str(a.complete).lower()}_to_{str(b.complete).lower()}"
            transitions[label].append(difference)
            if a.complete and b.complete:
                variant = f"{cell}|{key[0]}"
                d = context[left][variant]["nongold"] - context[right][variant]["nongold"]
                bins["<=-1" if d < 0 else ">=2" if d >= 2 else str(d)].append((difference, b.tokens - a.tokens))
    total = sum(map(len, transitions.values()))
    c.require(total == 5976, "Missing r4 vs r2 comparison requests")
    return {
        "n": total,
        "evidence_complete_transitions_r2_to_r4": {
            tag: {"n": len(d), "mean_f1_difference_pp": 100 * math.fsum(d) / len(d) if d else None,
                  "contribution_f1_pp": 100 * math.fsum(d) / total} for tag, d in transitions.items()
        },
        "both_complete_by_added_nongold_count_r4_minus_r2": {
            tag: {"n": len(d), "mean_f1_difference_pp": 100 * math.fsum(v for v, _ in d) / len(d) if d else None,
                  "mean_prompt_token_difference": math.fsum(t for _, t in d) / len(d) if d else None}
            for tag, d in bins.items()
        },
    }


def score():
    c.expect_new(RESULT)
    proof = preflight()
    cfg = c.read(READER_CONFIG)
    rp = c.read(c.READER / "receipt_inference.json")
    predicted = c.READER / "predictions.jsonl"
    c.require(cfg["status"] == "complete" and cfg["engine_shutdown"]
              and cfg["effective_disable_cascade_attn"] is True and cfg["predictions"] == N_PREDICTIONS
              and cfg["predictions_sha256"] == rp["predictions_sha256"] == c.digest(predicted)
              and rp["run_config_sha256"] == c.digest(READER_CONFIG), "Incomplete G receipt")
    token_ids = {}
    raw_output = {}
    raw_by_name = {name: [] for name in NAMES}

    def requested():
        for p in inputs.rows(REQUESTS):
            token_ids[tuple(p[k] for k in report.IDENTITY)] = diag.signature(p)
            yield p

    def predictions():
        for row, key in zip(inputs.rows(predicted), token_ids, strict=True):
            c.require(tuple(row[k] for k in report.IDENTITY) == key, "Prediction order differs from G batch plan")
            raw_output[key] = row["raw_output"]
            raw_by_name[row["strategy"]].append((token_ids[key], row["raw_output"]))
            yield row

    sources = report.index_requests(requested())
    scores = report.collect(predictions(), sources, NAMES)
    report.audit_matrix(scores, None, N_TASKS)
    stats = {}
    for name, per_cell in scores.items():
        flat = [s for rows in per_cell.values() for s in rows.values()]
        stats[name] = {
            "pooled": report.summarize_group(flat),
            "cells": {key: report.summarize_group(rows.values()) for key, rows in per_cell.items()},
            "evidence_complete_count": sum(s.complete for s in flat),
            "n": len(flat),
        }
    c.require(proof["task16_predictions"] == file_record(HIST_PRED), "Task 16 reference changed")
    deltas = {name: Counter() for name in (c.R2_POLICY, c.V3_POLICY)}
    for row in inputs.rows(HIST_PRED):
        if row["strategy"] in deltas:
            key = tuple(row[k] for k in report.IDENTITY)
            c.require(sources[key][0] == tuple(row[k] for k in report.MATCH), "Task 16 comparison inputs differ")
            deltas[row["strategy"]].update(count=raw_output[key] != row["raw_output"], n=1)
    c.require(all(r["n"] == 5976 for r in deltas.values()), "Task 16 comparison needs 5976 per strategy")
    selection = c.read(evaluation.SELECTED)
    name = "grpo_r4:" + selection["strategy"]
    c.require(name in NAMES[:4], "G model selection must come from F")
    pair = [per_cell_pair(name, other, scores) for other in (c.R2_POLICY, c.V3_POLICY)]
    c.write_json_exclusive(RESULT, {
        "status": "complete", "selected": name, "n_predictions": N_PREDICTIONS, "n_tasks": N_TASKS,
        "batch_size": BATCH, "same_batch_order": list(NAMES), "strategies": stats, "comparisons": pair,
        "same_run_inconsistent": {
            "all": diag.inconsistent(item for rows in raw_by_name.values() for item in rows),
            "strategies": {s: diag.inconsistent(rows) for s, rows in raw_by_name.items()},
        },
        "different_raw_vs_task16": dict(deltas), "smoke": cfg["smoke"],
        "decomposition_r4_minus_r2": decomposition(name, scores, c.read(FINALS)),
        "source_predictions": file_record(predicted), "run_config": file_record(READER_CONFIG),
    })
    print(f"G verified {N_PREDICTIONS} saved predictions including two same-batch comparisons")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("prepare", "run", "score"))
    args = cli.parse_args()
    {"prepare": prepare, "run": run, "score": score}[args.action]()

