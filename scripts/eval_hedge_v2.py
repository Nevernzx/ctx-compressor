#!/usr/bin/env python3
"""Task 09: replay a fixed train-selected hedge rule on Task 06's held-out grid."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import eval_formal_grpo as prior
from scripts import heldout_diagnostics_v2 as diagnostics
from scripts import heldout_model_v2 as sft
from scripts.hedge_action import TAU_GRID, candidate_ids, execute_hedge
from scripts.sft_run_report import validate_strategy
from src.heldout_eval import HeldoutReplay, MODES, BUDGETS, load_heldout_assets
from src.model_inference import MAX_NEW_TOKENS, SEED, compact, generation_record, guard_prompt, inference_prompt

v1 = sft.v1
SMALL = PROJECT / "eval/hedge_v2"
RAW = v1.EXTERNAL / "eval_rollouts_hedge_v2"
SELECTION = SMALL / "train_selection.json"
NAMES = ("nohedge", "tau_star", "always")
ADAPTER = sft.TRAIN / prior.SFT_NAME


def run_tau(name, selection):
    if selection.get("status") != "complete" or type(selection.get("tau_star")) not in (int, float) or selection["tau_star"] not in TAU_GRID:
        raise ValueError("A completed selection on the registered train grid is required before any held-out run")
    return {"nohedge": 0., "tau_star": float(selection["tau_star"]), "always": 1.01}[name]


def wrap_batch(replay, items, generated, scores, tau):
    """Advance using executed text; keep the native decoding and its token IDs available separately."""
    if len(items) != len(generated) or len(scores) != len(items):
        raise ValueError("Incomplete greedy or teacher-forcing batch")
    decisions, inputs = [], []
    for item, output, candidates in zip(items, generated, scores):
        decision = execute_hedge(item["state"], output["raw_output"], candidates, tau)
        decisions.append(decision)
        inputs.append({**output, "raw_output": decision["raw_output"], "output_token_ids_source": "greedy_raw_output"})
    rows = replay.advance_batch(inputs)
    for row, item, decision, output, candidates in zip(rows, items, decisions, generated, scores):
        for field in ("parsed_action", "action_valid", "invalid_reason", "keep_after", "used_after", "recall"):
            if decision[field] != row["state"][field]:
                raise ValueError(f"Reward and closed-loop executor disagree on {field}")
        if row["used_after"] > row["effective_budget"]:
            raise ValueError("Executed keep exceeds budget")
        hedged, extra = decision["hedged"], decision["extra_id"]
        if (not hedged and row["raw_output"] != output["raw_output"]) or (hedged and not row["action_valid"]):
            raise ValueError("Inconsistent hedge action or invalid fallback")
        row["greedy_raw_output"] = output["raw_output"]
        row["executed_raw_output"] = row["raw_output"]
        row["hedge"] = {"tau": tau, "p1": decision["p1"], "scores": candidates, "hedged": hedged,
                        "extra_id": extra, "extra_tokens": decision["extra_tokens"],
                        "extra_is_gold": hedged and extra in item["state"]["gold"],
                        "step_reward": decision["reward"]}
    return rows


def observations(row):
    h = row["hedge"]
    count = len(set(row["keep_after"]) - set(row["kept_before"]))
    return Counter(n=1, hedged=int(h["hedged"]), hit=int(h["extra_is_gold"]), kept_tokens=row["used_after"],
                   extra_tokens=h["extra_tokens"], **{"count_" + prior.BUCKETS[min(count, 3)]: 1})


def summarize(counter):
    n = counter["n"]
    if not n or sum(counter["count_" + k] for k in prior.BUCKETS) != n:
        raise ValueError("Missing steps in hedge observations")
    return {"hedge_rate": diagnostics.rate(counter["hedged"], n),
            "hedge_hit_rate": diagnostics.rate(counter["hit"], counter["hedged"]),
            "mean_kept_tokens_per_step": {"sum": counter["kept_tokens"], "n": n, "mean": counter["kept_tokens"] / n},
            "mean_extra_tokens_per_step": {"sum": counter["extra_tokens"], "n": n, "mean": counter["extra_tokens"] / n},
            "executed_add_counts_all_steps": {key: diagnostics.rate(counter["count_"+key], n) for key in prior.BUCKETS}}


def replay_run(llm, adapter, sampling, score_candidates, name, tau, tasks, corpus, proof, baseline,
               raw, run_file, manifest, monitor):
    replay = HeldoutReplay(tasks, corpus, name)
    tokenizer = llm.get_tokenizer()
    counters = defaultdict(Counter)
    manifest.update(status="running", n_steps=0, batches=[])
    v1.write_json(run_file, manifest)
    with (raw / "steps.jsonl").open("x", encoding="utf-8") as stream:
        while not replay.completed():
            monitor.check()
            items = replay.next_batch()
            prompts = [inference_prompt(tokenizer, item["state"]) for item in items]
            for item, prompt in zip(items, prompts):
                guard_prompt(prompt, proof["engine_parameters"]["max_model_len"],
                             identity=(item["trajectory_id"], item["step"]))
            position = items[0]["step"]
            if position == 0:
                digest = hashlib.sha256()
                for item, prompt in zip(items, prompts):
                    digest.update(sft.prompt_digest(sft.grid_id(item), prompt))
                if len(items) != proof["step0"]["n"] or digest.hexdigest() != proof["step0"]["identity_and_hashes_sha256"]:
                    raise ValueError("Live vLLM tokenizer differs from the Task 06 first-step grid")
            start = time.monotonic()
            outputs = llm.generate([{"prompt_token_ids": p["prompt_token_ids"]} for p in prompts],
                                   sampling_params=sampling, lora_request=adapter, use_tqdm=False)
            if len(outputs) != len(items):
                raise ValueError("Incomplete vLLM greedy batch")
            generated = [generation_record(p, out) for p, out in zip(prompts, outputs)]
            scores = []
            for i in range(0, len(items), 32):
                monitor.check()
                scores.extend(score_candidates(llm, adapter, [p["prompt_token_ids"] for p in prompts[i:i+32]],
                                              [candidate_ids(x["state"]) for x in items[i:i+32]]))
            rows = wrap_batch(replay, items, generated, scores, tau)
            for row in rows:
                counters[f'{row["mode"]}|{row["budget_tier"]}'].update(observations(row))
                stream.write(compact(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            manifest["n_steps"] += len(items)
            manifest["batches"].append({"step": position, "n": len(items), "seconds": time.monotonic() - start})
            v1.write_json(run_file, manifest)
            print(f'{v1.utc()} {name} step={position} outputs={manifest["n_steps"]}', flush=True)
    metrics = replay.aggregate(baseline=baseline)
    metrics.update(checkpoint_step=2000, steps_artifact=v1.file_record(raw / "steps.jsonl"))
    return metrics, counters


def run(name):
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("The Task 09 held-out replay requires physical GPU 1 on 5880")
    if Path(sys.prefix).resolve() not in {v1.EXTERNAL / "envs" / e for e in ("reader", "grpo")}:
        raise RuntimeError("Use the existing reader or grpo environment")
    selection = sft.load_json(SELECTION)
    tau = run_tau(name, selection)
    if (selection["split"] != "train" or selection["n_states"] != 22158 or selection["n_tasks"] != 400
            or selection["heldout_overlap"] != 0 or selection["adapter_sha256"] != prior.SFT_ADAPTER_SHA):
        raise ValueError("Unexpected train selection provenance")
    small, raw = SMALL / name, RAW / name
    if any(path.exists() or path.resolve() != path for path in (small, raw)):
        raise FileExistsError("Existing, redirected or previously attempted evaluation")
    small.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    run_file = small / "run.json"
    monitor = sft.Monitor(raw / "resources.jsonl", {})
    llm = None
    started = time.monotonic()
    manifest = {"status": "initializing", "name": name, "tau": tau, "pid": os.getpid(), "gpu": 1,
                "started_at_utc": v1.utc(), "selection": v1.file_record(SELECTION), "model": str(v1.MODEL),
                "adapter": str(ADAPTER), "schema_version": 1,
                "score_checks": v1.file_record(SMALL / "score_checks.json"),
                "score_implementation": [v1.file_record(PROJECT / "scripts" / path) for path in
                                         ("hedge_action.py", "hedge_scorer.py", "hedge_forced_tokens.py", "hedge_raw_logits.py")]}
    try:
        v1.write_json(run_file, manifest)
        tasks, corpus = load_heldout_assets(v1.DATA)
        proof = prior.load_signed(sft.PREFLIGHT, prior.SFT_PREFLIGHT_SHA)
        sft_metrics = prior.load_signed(prior.SFT_RESULTS / "checkpoint-2000_results.json", prior.SFT_METRICS_SHA)
        prev_steps = Path(sft_metrics["steps_artifact"]["path"])
        if v1.sha256(prev_steps) != sft_metrics["steps_artifact"]["sha256"]:
            raise ValueError("The Task 06 saved rollout differs from its frozen checksum")
        for key, val in sft.validate_inputs(tasks).items():
            if json.loads(json.dumps(val)) != proof[key]:
                raise ValueError(f"Frozen input or inference setting changed: {key}")
        weights = v1.file_record(ADAPTER / "adapter_model.safetensors")
        if weights["sha256"] != prior.SFT_ADAPTER_SHA:
            raise ValueError("The LoRA weights differ from Task 06 checkpoint-2000")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(v1.MODEL), local_files_only=True, trust_remote_code=False)
        proof["step0"] = prior.compare_step0(tasks, corpus, tokenizer, prev_steps, proof["step0"]["identity_and_hashes_sha256"])
        manifest.update(heldout_input=proof["input_files"][str(v1.DATA / "heldout_tasks.jsonl")],
                        step0=proof["step0"], adapter_weight=weights, seed=SEED, temperature=0, max_tokens=MAX_NEW_TOKENS)
        gpu = v1.gpu_info()
        if gpu["index"] != 1 or gpu["memory_used_mib"] >= 2048:
            raise RuntimeError(f"GPU 1 admission gate failed: {gpu}")
        manifest["gpu_before"] = gpu
        monitor.thread.start()
        from scripts.hedge_scorer import create_engine, score_candidates
        from vllm import SamplingParams
        llm, adapter, params = create_engine()
        if any(params[k] != v for k, v in proof["engine_parameters"].items()):
            raise ValueError("Teacher-forcing engine changed a frozen Task 06 parameter")
        if Path(adapter.lora_path).resolve() != ADAPTER.resolve():
            raise ValueError("Incorrect adapter on live scoring and greedy requests")
        manifest["engine_parameters"] = proof["engine_parameters"]
        manifest["engine_tokenizer_check"] = v1.check_reference(sft.V1_ARGS, llm.get_tokenizer(), tasks, corpus)
        sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, seed=SEED)
        metrics, counts = replay_run(llm, adapter, sampling, score_candidates, name, tau, tasks, corpus, proof,
                                     sft.load_json(v1.RESULTS / "no_compress_results.json"), raw, run_file, manifest, monitor)
        validate_strategy(metrics, name)
        v1.write_json(raw / "metrics.json", metrics)
        v1.write_json(small / "metrics.json", metrics)
        steps = raw / "steps.jsonl"
        checks = diagnostics.summarize_strategy(name, steps, metrics, {t["task"]:t["steps"] for t in tasks}, metrics["steps_artifact"]["sha256"])
        v1.write_json(small / "diagnostics.json", {"definitions": diagnostics.DEFINITIONS, **checks})
        extra = prior.collect_extra(steps, corpus, prev_steps if name == "nohedge" else None)
        for cell in prior.CELL_KEYS:
            group = extra["configs"][cell]
            observed = checks["configs"][cell]["new_gold_add"]
            strata = group["new_gold_by_title"].values()
            if (sum(s["count"] for s in strata), sum(s["n"] for s in strata)) != (observed["count"], observed["n"]):
                raise ValueError(f"Task 06 eligible gold and title strata disagree in {cell}")
            group.update(summarize(counts[cell]))
        for mode in MODES:
            extra["modes"][mode].update(summarize(sum((counts[f"{mode}|{tier}"] for tier in BUDGETS), Counter())))
        extra["combined"].update(summarize(sum(counts.values(), Counter())))
        extra["definitions"].update(hedge_rate="Executed hedges / all steps, including invalid greedy outputs.",
                                    hedge_hit_rate="Executed extras in the state's available gold / executed hedges.",
                                    mean_kept_tokens_per_step="Total retained tokens after all steps / all steps.",
                                    executed_add_counts_all_steps="Newly retained distinct PIDs over all steps, including invalid no-ops.")
        if name == "nohedge":
            extra["metric_deltas_vs_sft_v2"] = prior.metric_deltas(metrics, sft_metrics)
        if tau == 0 and extra["combined"]["hedge_rate"]["count"] != 0:
            raise ValueError("Zero-threshold policy hedged")
        if (extra["combined"]["n_steps"] != 13956 or metrics["combined"]["n_tasks"] != 1494
                or any(extra["configs"][cell]["n_steps"] != 2326 for cell in prior.CELL_KEYS)):
            raise ValueError("Incomplete Task 09 evaluation grid")
        v1.write_json(small / "extra.json", extra)
        monitor.check()
        manifest.update(status="complete", checks={"n_steps":13956, "n_trajectories":1494, "n_cells":6,
                        "step0_matched":proof["step0"]["n"]}, metrics=v1.file_record(small / "metrics.json"))
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        v1.finish_run(manifest, monitor, llm, run_file, started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=NAMES, required=True)
    run(parser.parse_args().name)
