#!/usr/bin/env python3
"""Independently audit completed Task 05 model rollouts on CPU.

This verifier does not invoke the replay state machine or its action parser.
It reconstructs each state from saved retrieval and its own preceding result,
then reuses only the original Task 01 executor and Task 02 metric definitions.
Missing/in-progress outputs produce a separate pending receipt, never a winner.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
DATA = EXTERNAL / "data/traj_train"
MODEL = EXTERNAL / "models/Qwen3-1.7B"
RESULTS = PROJECT / "eval/sft_run"
ROLLOUTS = EXTERNAL / "eval_rollouts"
CHECKPOINT_STEPS = (500, 1000, 1500, 2000, 2500)
STRATEGIES = ("base", *(f"checkpoint-{step}" for step in CHECKPOINT_STEPS))
sys.path.insert(0, str(PROJECT))

from src.build_traj import apply_action, make_sample, render_input
from src.feasibility import aggregate_tasks, summarize_task
from src.heldout_eval import BUDGETS, MODES, load_heldout_assets, select_checkpoint

REASONS = ("invalid_json", "invalid_candidate_id", "append_drop", "over_budget")
MAX_TOKENS = 256


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def equal(actual, expected, path):
    """Compare every metric field exactly, including JSON scalar types."""
    if type(actual) is not type(expected):
        raise AuditError(f"{path}: type {type(actual).__name__} != {type(expected).__name__}")
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f"{path}: object fields differ")
        for key in expected:
            equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{path}: list length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            equal(left, right, f"{path}[{index}]")
    else:
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-JSON constant: {value}")


def strict_json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def independently_execute(state, raw):
    """Independent parse/classification; invalid actions explicitly become no-op."""
    parsed = None
    try:
        if not isinstance(raw, str):
            raise ValueError("raw output must be text")
        candidate = strict_json(raw)
        if not isinstance(candidate, dict) or "add" not in candidate:
            raise ValueError("action object with add required")
        add, drop = candidate["add"], candidate.get("drop", [])
        if not isinstance(add, list) or not isinstance(drop, list):
            raise ValueError("add/drop must be arrays")
        if any(type(item) is not int for item in add + drop):
            raise ValueError("fragment IDs must be integers, excluding booleans")
        parsed = candidate
    except (ValueError, TypeError, RecursionError):
        pass
    reason, executed = None, None
    if parsed is None:
        reason = "invalid_json"
    else:
        add, drop = set(parsed["add"]), set(parsed.get("drop", []))
        executed = apply_action(state, parsed)
        if add - set(state["new_ids"]) or drop - set(state["kept_ids"]):
            reason = "invalid_candidate_id"
        elif state["mode"] == "append" and drop:
            reason = "append_drop"
        else:
            attempted = [i for i in state["kept_ids"] if i not in drop]
            attempted.extend(i for i in state["new_ids"] if i in add)
            if sum(state["ntok"][i] for i in attempted) > state["budget"]:
                reason = "over_budget"
            else:
                equal(executed, attempted, "original apply_action")
        if reason:
            require(executed is None, "original apply_action accepted independently invalid action")
    keep = list(state["kept_ids"]) if reason else executed
    gold = set(state["gold"])
    return {"action": parsed, "parsed_action": parsed, "action_valid": reason is None,
            "invalid_reason": reason, "keep_after": keep,
            "used_after": sum(state["ntok"][i] for i in keep),
            "recall": len(gold.intersection(keep)) / len(gold) if gold else 1.0}


def ratio(count, n):
    return {"count": count, "n": n, "rate": count / n if n else None}


def recompute_metrics(rows, baseline):
    """Original Task 02 aggregate plus independently accumulated model counters."""
    result = aggregate_tasks(rows, baseline=baseline)
    n = result["n_steps"]
    result["invalid_outputs"] = {reason: ratio(sum(row["invalid"][reason] for row in rows), n)
                                 for reason in REASONS}
    result["invalid_outputs"]["any_invalid"] = ratio(sum(row["invalid_action_count"] for row in rows), n)
    result["output_limit"] = ratio(sum(row["output_limit_hits"] for row in rows), n)
    result["requested_budget_occupancy"] = {"n": len(rows),
        "mean": sum(row["final_occupancy"] for row in rows) / len(rows),
        "max": max(row["final_occupancy"] for row in rows)}
    result["generation"] = {"n": n, "output_tokens": sum(row["generated_tokens"] for row in rows),
                            "prompt_tokens": sum(row["prompt_tokens"] for row in rows)}
    return result


def audit_strategy(records, tasks, corpus, saved, baseline, *, tokenizer=None,
                   prompt_check="first", max_model_len=40960):
    """Audit a record iterable; flexible task count permits synthetic CPU tests."""
    require(prompt_check in ("first", "all"), "prompt_check must be first or all")
    require(tokenizer is not None, "a CPU tokenizer is required for prompt re-encoding")
    strategy = saved["strategy"]
    require(strategy in STRATEGIES, "unexpected model strategy")
    expected_step = None if strategy == "base" else int(strategy.rsplit("-", 1)[1])
    equal(saved["checkpoint_step"], expected_step, "checkpoint_step")
    require(all(task["split"] == "heldout" for task in tasks), "only train-heldout tasks allowed")
    require(len({task["task"] for task in tasks}) == len(tasks), "duplicate task identity")
    equal(saved["settings"]["n_tasks"], len(tasks), "settings.n_tasks")
    equal(saved["settings"]["n_trajectories"], len(tasks) * 6, "settings.n_trajectories")
    equal(saved["settings"]["budgets"], BUDGETS, "settings.budgets")
    equal(saved["settings"]["modes"], list(MODES), "settings.modes")
    equal(saved["settings"]["split"], "train-heldout", "settings.split")
    equal(saved["settings"]["saved_retrieval_steps"], True, "settings.saved_retrieval_steps")
    equal(saved["settings"]["invalid_reason_priority"], list(REASONS), "settings.invalid_reason_priority")
    equal(saved["settings"]["max_tokens"], MAX_TOKENS, "settings.max_tokens")
    equal(saved["settings"]["r_read"], 0.05, "settings.r_read")
    equal(saved["settings"]["r_write"], 1.25, "settings.r_write")
    trajectories = {}
    for mode in MODES:
        for tier in BUDGETS:
            for task in tasks:
                trajectories[(mode, tier, task["task"])] = {
                    "task": task, "kept": [], "lid": {}, "gold": set(), "samples": [],
                    "invalid": Counter(), "output_limit_hits": 0, "generated_tokens": 0, "prompt_tokens": 0}
    ntok = {pid: row["ntok"] for pid, row in enumerate(corpus)}
    records = iter(records)
    count, reencoded = 0, 0
    # This expected order independently enforces whole-active-step batches.
    for step_index in range(max(len(task["steps"]) for task in tasks)):
        for (mode, tier, task_id), trajectory in trajectories.items():
            task = trajectory["task"]
            if step_index >= len(task["steps"]):
                continue
            identity = f"{strategy}|{mode}|{tier}|{task_id}:{step_index}"
            try:
                record = next(records)
            except StopIteration as error:
                raise AuditError(f"missing record: {identity}") from error
            expected = {"schema_version": 1, "strategy": strategy, "task": task_id,
                        "mode": mode, "budget_tier": tier, "budget": BUDGETS[tier],
                        "effective_budget": BUDGETS[tier], "step": step_index,
                        "trajectory_id": f"{strategy}|{mode}|{tier}|{task_id}",
                        "config": f"{strategy}|{mode}|{tier}"}
            for key, value in expected.items():
                equal(record[key], value, f"{identity}.{key}")
            step = task["steps"][step_index]
            trajectory["gold"].add(step["gold"])
            state = make_sample(task["questions"], trajectory["kept"], step["new"], trajectory["gold"],
                                corpus, BUDGETS[tier], mode, trajectory["lid"])
            sample = dict(record["state"])
            for key in ("ntok", "pid_by_lid"):
                original = sample[key]
                sample[key] = {int(k): v for k, v in original.items()}
                require(len(sample[key]) == len(original), f"{identity}: aliased integer mapping keys")
            require(isinstance(record["raw_output"], str), f"{identity}: raw_output must remain text")
            executed = independently_execute(state, record["raw_output"])
            expected_sample = {**{key: value for key, value in state.items() if key != "text"}, **executed}
            equal(sample, expected_sample, f"{identity}.state")
            pids = [state["pid_by_lid"][i] for i in executed["keep_after"]]
            fields = {"kept_before": trajectory["kept"], "new": [state["pid_by_lid"][i] for i in state["new_ids"]],
                      "returned": step["new"], "gold": step["gold"], "question_index": step["question_index"],
                      "hop_index": step["hop_index"], "keep_after": pids,
                      **{key: executed[key] for key in ("parsed_action", "invalid_reason", "action_valid", "used_after", "recall")}}
            for key, value in fields.items():
                equal(record[key], value, f"{identity}.{key}")
            if mode == "append":
                equal(pids[:len(trajectory["kept"])], trajectory["kept"], f"{identity}.append_prefix")

            generation = record["generation_metadata"]
            for key, count_key in (("prompt_token_ids", "prompt_tokens"), ("output_token_ids", "generated_tokens")):
                ids = generation[key]
                require(isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids), f"{identity}.{key}: invalid IDs")
                equal(generation[count_key], len(ids), f"{identity}.{count_key}")
            require(0 < generation["prompt_tokens"] <= max_model_len - MAX_TOKENS, f"{identity}: prompt overflow")
            require(generation["generated_tokens"] <= MAX_TOKENS, f"{identity}: output token overflow")
            equal(generation["input_truncated"], False, f"{identity}.input_truncated")
            equal(generation["runtime_prompt_ids_match"], True, f"{identity}.runtime_prompt_ids_match")
            equal(generation["prompt_sha256"], digest_json(generation["prompt_token_ids"]), f"{identity}.prompt_sha256")
            user_text = render_input(state)
            equal(generation["user_text_sha256"], hashlib.sha256(user_text.encode()).hexdigest(), f"{identity}.user_text_sha256")
            if prompt_check == "all" or count == 0:
                ids = tokenizer.apply_chat_template([{"role": "user", "content": user_text}], tokenize=True,
                                                     add_generation_prompt=True, enable_thinking=False)
                equal(generation["prompt_token_ids"], list(ids), f"{identity}.reencoded_prompt_ids")
                reencoded += 1
            limit_hit = generation["generated_tokens"] == MAX_TOKENS or generation["finish_reason"] == "length"
            equal(record["output_limit_hit"], limit_hit, f"{identity}.output_limit_hit")
            equal(generation["output_limit_hit"], limit_hit, f"{identity}.generation.output_limit_hit")
            trajectory["kept"] = pids
            trajectory["samples"].append({"pid_by_lid": state["pid_by_lid"], "keep_after": executed["keep_after"],
                                          "recall": executed["recall"], "action_valid": executed["action_valid"]})
            if executed["invalid_reason"]:
                trajectory["invalid"][executed["invalid_reason"]] += 1
            trajectory["output_limit_hits"] += limit_hit
            trajectory["generated_tokens"] += generation["generated_tokens"]
            trajectory["prompt_tokens"] += generation["prompt_tokens"]
            count += 1
    require(next(records, None) is None, "extra records after all expected trajectories completed")
    configs, all_summaries = {}, []
    for mode in MODES:
        for tier, budget in BUDGETS.items():
            summaries = []
            for task in tasks:
                trajectory = trajectories[(mode, tier, task["task"])]
                question_gold = [set() for _ in task["questions"]]
                for step in task["steps"]:
                    question_gold[step["question_index"]].add(step["gold"])
                summary = summarize_task(trajectory["samples"], task["steps"], question_gold, ntok, budget)
                summary.update({key: trajectory[key] for key in
                                ("invalid", "output_limit_hits", "generated_tokens", "prompt_tokens")})
                summaries.append(summary)
            name = f"{strategy}|{mode}|{tier}"
            configs[name] = {"policy": strategy, "strategy": strategy, "mode": mode, "budget_tier": tier,
                             "budget": budget, "budget_binding": True,
                             "metrics": recompute_metrics(summaries, baseline["configs"][f"no_compress|{mode}|{tier}"]["metrics"])}
            all_summaries.extend(summaries)
    combined = recompute_metrics(all_summaries, baseline["combined"])
    equal(saved["configs"], configs, f"{strategy}.configs")
    equal(saved["combined"], combined, f"{strategy}.combined")
    return {"strategy": strategy, "checkpoint_step": expected_step, "status": "verified",
            "n_trajectories": len(trajectories), "n_step_records": count,
            "prompt_reencoding": {"policy": prompt_check, "n": reencoded, "all_recorded_id_hashes_checked": count,
                                  "all_visible_prompt_text_hashes_checked": count},
            "configs_sha256": digest_json(configs), "combined": combined}


def completion_status(results_dir, rollout_root):
    """Cheap check before reading large inputs; missing results never imply zero."""
    missing = []
    for name in STRATEGIES:
        for path in (rollout_root / name / "steps.jsonl", rollout_root / name / "metrics.json",
                     results_dir / f"{name}_results.json"):
            if not path.is_file():
                missing.append(str(path))
    run_path = results_dir / "model_rollout_run.json"
    run = json.loads(run_path.read_text()) if run_path.is_file() else {}
    if run.get("status") != "complete":
        missing.append("model_rollout_run.status=complete")
    for name in STRATEGIES:
        if run.get("strategies", {}).get(name, {}).get("status") != "complete":
            missing.append(f"model_rollout_run.strategies.{name}.status=complete")
    return {"status": "pending" if missing else "ready", "missing_inputs": missing,
            "model_run_status": run.get("status"), "selected_checkpoint": None}


def independent_selection(audits):
    candidates = [row for row in audits if row["strategy"].startswith("checkpoint-")]
    require({row["checkpoint_step"] for row in candidates} == set(CHECKPOINT_STEPS)
            and len(candidates) == 5, "all five distinct checkpoints required for selection")
    require(all(row["n_trajectories"] == 1494 for row in candidates), "selection requires 1494 audited trajectories per checkpoint")
    ordered = sorted(candidates, key=lambda row: (row["combined"]["evidence_loss"]["count"],
                      -row["combined"]["final_recall"]["mean"], row["checkpoint_step"]))
    return {"strategy": ordered[0]["strategy"], "checkpoint_step": ordered[0]["checkpoint_step"],
            "rule": "pooled final evidence-loss count ascending, pooled final recall descending, checkpoint step ascending",
            "n_checkpoints": 5, "n_trajectories_per_checkpoint": 1494,
            "ranking": [{"strategy": row["strategy"], "checkpoint_step": row["checkpoint_step"],
                         "evidence_loss": row["combined"]["evidence_loss"],
                         "final_recall": row["combined"]["final_recall"]} for row in ordered]}


class TrackedJsonl:
    def __init__(self, path):
        self.path, self.record = Path(path), None

    def __iter__(self):
        digest, size, lines = hashlib.sha256(), 0, 0
        with self.path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                size += len(line)
                lines += 1
                yield strict_json(line)
        self.record = {"path": str(self.path), "bytes": size, "sha256": digest.hexdigest()}


def write_new(path, result):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-check", choices=("first", "all"), default="first")
    args = parser.parse_args()
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("run only on 5880 with CUDA_VISIBLE_DEVICES='' (CPU only)")
    if Path(sys.prefix).resolve() != EXTERNAL / "envs/reader":
        parser.error("use the existing reader environment for CPU tokenizer verification")
    if any(os.environ.get(key) for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")):
        parser.error("proxies must be unset")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        parser.error("offline flags required")
    final_path = RESULTS / "model_rollout_audit.json"
    if final_path.exists():
        parser.error("refusing to overwrite existing final audit")
    started, stamp = time.monotonic(), datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    pending = completion_status(RESULTS, ROLLOUTS)
    if pending["status"] == "pending":
        pending.update(schema_version=1, created_at_utc=stamp, audited_models=0,
                       note="No model result or checkpoint winner inferred from missing outputs.")
        path = RESULTS / f"model_rollout_audit.pending.{stamp}.json"
        write_new(path, pending)
        print(compact({**pending, "receipt": str(path)}))
        return 3
    run = strict_json((RESULTS / "model_rollout_run.json").read_text())
    expected_inputs = [DATA / "heldout_tasks.jsonl", DATA / "corpus_train.jsonl",
                       MODEL / "tokenizer.json", MODEL / "tokenizer_config.json", MODEL / "config.json"]
    inputs = {str(path): file_record(path) for path in expected_inputs}
    equal(run["input_files"], inputs, "model run input files")
    tasks, corpus = load_heldout_assets(DATA, expected_tasks=249)
    require(sum(len(task["steps"]) for task in tasks) * 6 == 13956, "expected 13956 saved-step evaluations")
    baseline = strict_json((RESULTS / "no_compress_results.json").read_text())
    equal(run["no_compress_baseline"], file_record(RESULTS / "no_compress_results.json"), "no_compress source checksum")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    native = strict_json((MODEL / "config.json").read_text())["max_position_embeddings"]
    verified, saved_results = [], []
    for name in STRATEGIES:
        small = RESULTS / f"{name}_results.json"
        external = ROLLOUTS / name / "metrics.json"
        saved = strict_json(small.read_text())
        equal(saved, strict_json(external.read_text()), f"{name}.external_metrics_copy")
        stream = TrackedJsonl(ROLLOUTS / name / "steps.jsonl")
        checked = audit_strategy(stream, tasks, corpus, saved, baseline, tokenizer=tokenizer,
                                 prompt_check=args.prompt_check, max_model_len=native)
        equal(saved["steps_artifact"], stream.record, f"{name}.steps_artifact")
        equal(run["strategies"][name]["metrics"], file_record(external), f"{name}.run_metrics_artifact")
        checked.update(steps_artifact=stream.record, metrics_artifact=file_record(small))
        verified.append(checked)
        saved_results.append(saved)
        print(compact({"strategy": name, "status": "verified", "n_steps": checked["n_step_records"]}), flush=True)
    selection = independent_selection(verified)
    reference_winner = select_checkpoint([row for row in saved_results if row["checkpoint_step"] is not None])
    equal(selection["strategy"], reference_winner["strategy"], "checkpoint_selection_reference")
    published = RESULTS / "sft_run_results.json"
    report_check = {"status": "not_yet_available"}
    if published.is_file():
        final_report = strict_json(published.read_text())
        picked = final_report.get("selected_checkpoint")
        if picked is not None:
            equal(picked["strategy"], selection["strategy"], "published checkpoint winner")
            report_check = {"status": "matches", "report": file_record(published)}
        else:
            report_check = {"status": "report_pending", "report": file_record(published)}
    import torch
    require(not torch.cuda.is_initialized(), "audit must never initialize CUDA")
    result = {"schema_version": 1, "status": "ok", "created_at_utc": stamp,
              "elapsed_seconds": time.monotonic() - started, "strategies": verified,
              "n_strategies": 6, "n_trajectories": sum(row["n_trajectories"] for row in verified),
              "n_step_records": sum(row["n_step_records"] for row in verified),
              "selected_checkpoint": selection, "selection_matches_shared_helper": True,
              "published_report_selection": report_check, "input_files": inputs,
              "run_manifest": file_record(RESULTS / "model_rollout_run.json"),
              "no_compress_baseline": file_record(RESULTS / "no_compress_results.json"),
              "source": file_record(Path(__file__).resolve()),
              "checks": {"independent_state_chain": True, "raw_json_reparsed": True,
                         "four_invalid_reasons_and_noop": True, "all_metric_fields_equal": True,
                         "stable_local_ids_and_corpus_pids": True, "whole_active_step_batches": True,
                         "train_corpus_saved_heldout_steps_only": True, "teacher_rollout_states_used": False,
                         "dev_inputs_read": False, "model_outputs_modified": False,
                         "cuda_initialized": False, "api_downloads_installs": False}}
    write_new(final_path, result)
    print(compact({"status": "ok", "receipt": str(final_path), "selected_checkpoint": selection["strategy"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
