#!/usr/bin/env python3
"""Measure reward variation from eight sampled generations of each stratified train-state prompt."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.build_grpo_states import (BUDGETS, DEFAULT_BASE, LABEL, MAX_NEW_TOKENS, MODES, PREP, SEED,
                                     compact, create_engine, file_record, guard_prompt, lines, rate,
                                     source_info, write_json)
from src.heldout_eval import INVALID_REASONS

def stratified(states, per_cell, seed):
    cells = defaultdict(list)
    for row in states:
        if row["split"] != "train" or row["mode"] not in MODES or row["budget_tier"] not in BUDGETS:
            raise ValueError("Unrecognized training stratum")
        cells[(row["mode"], row["budget_tier"])].append(row)
    rng = random.Random(seed)
    return [s for mode in MODES for tier in BUDGETS for s in rng.sample(cells[mode, tier], per_cell)]


def summarize(groups):
    rewards = [s["reward"] for g in groups for s in g["samples"]]
    reasons = Counter(s["invalid_reason"] for g in groups for s in g["samples"])
    dispersions = [statistics.pstdev(s["reward"] for s in g["samples"]) for g in groups]
    gains = [max(s["reward"] for s in g["samples"]) - g["greedy_reward"] for g in groups]
    return {"n_groups": len(groups), "n_completions": len(rewards), "mean_reward": statistics.mean(rewards),
            "invalid": rate(len(rewards) - reasons[None], len(rewards)),
            "invalid_by_category": {k:rate(reasons[k], len(rewards)) for k in INVALID_REASONS},
            "zero_std_groups": rate(dispersions.count(0), len(groups)),
            "mean_within_group_std": statistics.mean(dispersions),
            "mean_best_minus_greedy": statistics.mean(gains),
            "some_sample_beats_greedy": rate(sum(g > 0 for g in gains), len(groups))}


def run(args):
    from src.grpo_reward import score_completion
    from vllm import SamplingParams
    source = source_info(args)
    selected = stratified(lines(args.states), args.per_cell, args.seed)
    n = len(selected)
    for row in selected:
        if any(row[k] != source[k] for k in ("base_model","source_adapter","source_adapter_sha256","inference_adapter")):
            raise ValueError("Sampling model does not match the greedy state policy")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.output_dir / "probe_outputs.jsonl"
    partial = raw.with_name("probe_outputs.partial.jsonl")
    if raw.exists() or partial.exists():
        raise FileExistsError("Previous probe records exist")
    llm, adapter, engine = create_engine(args)
    sampling = SamplingParams(n=args.generations, temperature=1., top_p=1., max_tokens=MAX_NEW_TOKENS, seed=args.seed)
    for r in selected:
        guard_prompt(r, engine["max_model_len"], identity=(r["trajectory_id"],r["step"]))
    started = time.monotonic()
    compact_groups = []
    with partial.open("x") as stream:
        # Keep each prompt group intact and make raw samples durable as each batch finishes.
        for pos in range(0, n, 16):
            batch = selected[pos:pos+16]
            outputs = llm.generate([{"prompt_token_ids": r["prompt_token_ids"]} for r in batch], sampling,
                                   lora_request=adapter, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("Incomplete probe batch")
            for row, output in zip(batch, outputs):
                if row["prompt_token_ids"] != list(output.prompt_token_ids) or len(output.outputs) != args.generations:
                    raise RuntimeError("Probe prompt changed or sample group incomplete")
                state = json.loads(row["reward_state_json"])
                greedy = score_completion(row["greedy_output"], state, lam=args.lam)
                if greedy["invalid_reason"] != row["greedy_invalid_reason"] or greedy["keep_after"] != row["greedy_keep_after"]:
                    raise ValueError("Plugin disagrees with the closed-loop greedy result")
                identity = {k:row[k] for k in ("split","trajectory_id","task","mode","budget_tier","step","prompt_sha256")}
                samples = []
                for o in output.outputs:
                    scored = score_completion(o.text, state, lam=args.lam)
                    samples.append({"index":o.index, "completion":o.text, "token_ids":list(o.token_ids),
                                    "finish_reason":o.finish_reason, **scored})
                group = {**identity, "greedy_reward":greedy["reward"], "samples":samples}
                stream.write(compact({**group, **source}) + "\n")
                compact_groups.append({**identity, "greedy_reward":greedy["reward"],
                    "samples":[{k:s[k] for k in ("reward","invalid_reason")} for s in samples]})
            stream.flush()
            print(f"generated {min(pos+16,n)}/{n} train prompt groups", flush=True)
    partial.rename(raw)
    grouped = defaultdict(list)
    for group in compact_groups:
        grouped[f"{group['mode']}|{group['budget_tier']}"].append(group)
    write_json(args.output_dir / "sampling_probe.json", {**source, "split":"train", "seed":args.seed,
        "lambda":args.lam, "generation":{"n":args.generations, "temperature":1, "top_p":1, "max_tokens":MAX_NEW_TOKENS},
        "sampling_order":{"modes":list(MODES),"budget_tiers":list(BUDGETS),"states_per_cell":args.per_cell},
        "engine":engine,"population_state_file":file_record(args.states),"raw_samples":file_record(raw),
        "within_group_std":"population standard deviation over rewards (ddof=0)",
        "pooled":summarize(compact_groups),"cells":{k:summarize(v) for k,v in grouped.items()},
        "elapsed_excluding_engine_startup_seconds":time.monotonic()-started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--source-adapter", type=Path)
    parser.add_argument("--model-label", default=LABEL)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-cell", type=int, default=32)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--lam", type=float, default=.3)
    args = parser.parse_args()
    if not (args.states.resolve().is_relative_to(PREP) and args.output_dir.resolve().is_relative_to(PREP)):
        parser.error("Use the training state pool and an output directory under grpo_prep")
    if args.generations < 2:
        parser.error("At least two generations per group are required")
    run(args)


if __name__ == "__main__":
    main()
