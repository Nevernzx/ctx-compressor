#!/usr/bin/env python3
"""Produce training prompts by replaying a parameterized local policy on saved train tasks."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.rollout_model import file_record, sha256, write_json
from src.build_traj import render_input
from src.heldout_eval import BUDGETS, MODES, HeldoutReplay
from src.model_inference import (MAX_NEW_TOKENS, SEED as EVAL_SEED, compact, engine_parameters,
                                 generation_record, guard_prompt, inference_prompt)
from src.sft_protocol import length_distribution

EXTERNAL = Path("/root/userfolder/ctx-compressor")
TRAIN_TASKS = EXTERNAL / "data/traj_train/train_tasks.jsonl"
TRAIN_CORPUS = TRAIN_TASKS.with_name("corpus_train.jsonl")
SFT = EXTERNAL / "data/sft_v2/train.jsonl"
DEFAULT_BASE = EXTERNAL / "models/Qwen3-1.7B"
STANDIN = EXTERNAL / "runs/sft_20k_20261117/checkpoints/checkpoint-2000"
PREP = EXTERNAL / "grpo_prep"
SEED = 20261120
LABEL = "替身 v1 checkpoint-2000，非正式结果"
REWARD_FIELDS = ("mode", "budget", "kept_ids", "new_ids", "ntok", "gold")


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def require_train_file(path):
    if path.resolve() != TRAIN_TASKS.resolve():
        raise ValueError("Only the frozen train_tasks.jsonl input is permitted for this experiment")


def load_assets(path, n, seed):
    require_train_file(path)
    population = []
    for row in lines(path):
        if row["split"] != "train":
            raise ValueError("Non-training row in task source")
        population.append({k: row[k] for k in ("task", "split", "tier", "budget", "questions", "steps")})
    if len({t["task"] for t in population}) != len(population):
        raise ValueError("Duplicate train task IDs")
    tasks = random.Random(seed).sample(population, n)
    corpus = list(lines(TRAIN_CORPUS))
    if any(p["pid"] != i for i, p in enumerate(corpus)):
        raise ValueError("Frozen corpus ordering changed")
    return tasks, corpus


def source_info(args):
    source = args.source_adapter or args.adapter
    if source is None:
        raise ValueError("Pass --source-adapter to identify the LoRA merged into the base")
    return {"base_model": str(args.base_model.resolve()),
            "inference_adapter": str(args.adapter.resolve()) if args.adapter else None,
            "source_adapter": str(source.resolve()),
            "source_adapter_sha256": sha256(source / "adapter_model.safetensors"),
            "model_output_label": args.model_label}


def check_gpu(gpu):
    if gpu != 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU jobs must use CUDA_VISIBLE_DEVICES=0 on 5880")


def create_engine(args):
    check_gpu(args.gpu)
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    native = json.loads((args.base_model / "config.json").read_text())
    params = engine_parameters(args.base_model, native)
    llm = LLM(**params)
    adapter = LoRARequest(args.adapter.name, 1, str(args.adapter.resolve())) if args.adapter else None
    return llm, adapter, params


def greedy(llm, adapter, prompts, max_model_len):
    from vllm import SamplingParams
    for p in prompts:
        guard_prompt(p, max_model_len)
    samples = llm.generate([{"prompt_token_ids": p["prompt_token_ids"]} for p in prompts],
                           SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, seed=EVAL_SEED),
                           lora_request=adapter, use_tqdm=False)
    if len(samples) != len(prompts):
        raise RuntimeError("Incomplete generation batch")
    return [generation_record(p, s) for p, s in zip(prompts, samples)]


def read_sft_references(selected):
    refs = {}
    selected = set(selected)
    for row in lines(SFT):
        meta = row["meta"]
        if meta["split"] != "train":
            raise ValueError("Non-training SFT row")
        if meta["task"] not in selected or meta["step"] != 0:
            continue
        key = (meta["task"], meta["mode"])
        if key in refs:
            raise ValueError(f"Multiple SFT step-0 rows for {key}")
        refs[key] = (meta["tier"], row["messages"][0]["content"])
    if set(refs) != {(task, mode) for task in selected for mode in MODES}:
        raise ValueError("Missing original SFT row")
    return refs


def check_sft(text, reference_tier, reference, tier):
    original = BUDGETS[reference_tier]
    block = f"\n预算：{original} token\n已用 token：0\n\n已保留："
    if reference.count(block) != 1:
        raise ValueError("Expected exactly one original SFT budget header")
    expected = reference.replace(block, block.replace(str(original), str(BUDGETS[tier]), 1), 1)
    return {"direct": tier == reference_tier, "matches": expected.encode() == text.encode()}


def audit_step_zero(rows, refs):
    counts = Counter()
    for r in rows:
        task, mode, tier = r["task"], r["mode"], r["budget_tier"]
        ref_tier, ref_text = refs[(task, mode)]
        result = check_sft(r["messages"][0]["content"], ref_tier, ref_text, tier)
        name = "direct" if result["direct"] else "budget_substituted"
        counts[f"{name}_n"] += 1
        counts[f"{name}_mismatches"] += not result["matches"]
    same_tier = sum(refs[t, MODES[0]][0] == refs[t, MODES[1]][0] for t in {r["task"] for r in rows})
    return {**counts, "trajectories": len(rows), "tasks_with_same_sft_tier_in_both_modes": same_tier,
            "literal_check_unavailable_n": counts["budget_substituted_n"],
            "definition": "Only the native tier has an SFT row. For other tiers replace exactly the unique budget header in its original SFT text; all other UTF-8 bytes are compared."}


def prompt_proof(args):
    from transformers import AutoTokenizer
    tasks, corpus = load_assets(args.task_file, args.n_tasks, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True)
    replay = HeldoutReplay(tasks, corpus, "step0")
    refs = read_sft_references(t["task"] for t in tasks)
    rows = []
    for item in replay.next_batch():
        state = item["state"]
        row = {k: item[k] for k in ("task", "mode", "budget_tier", "step")}
        row["messages"] = [{"role": "user", "content": render_input(state)}]
        row["split"] = "train"
        rows.append({**row, **inference_prompt(tokenizer, state)})
    audit = audit_step_zero(rows, refs)
    # Distinct train tasks, original SFT tiers, alternating modes, fixed stable input batch.
    keyed = {(r["task"], r["mode"], r["budget_tier"]): r for r in rows}
    fixture = []
    for index, task in enumerate(tasks):
        mode = MODES[index % len(MODES)]
        fixture.append(keyed[task["task"], mode, task["tier"]])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prompts.jsonl").open("x") as out:
        for row in fixture:
            out.write(compact(row) + "\n")
    write_json(args.output_dir / "step0_preflight.json",
               {"step0_sft": audit, "tasks": [t["task"] for t in tasks], "seed": args.seed,
                "prompt_file": file_record(args.output_dir / "prompts.jsonl"),
                "n_fixed_step0_prompts": len(fixture), "split": "train",
                "train_source": str(args.task_file), "sft_source": str(SFT)})
    if audit["direct_mismatches"] or audit["budget_substituted_mismatches"]:
        raise AssertionError(audit)


def reference(args):
    rows = list(lines(args.prompts))
    if len(rows) < 20 or any(r["split"] != "train" or r["step"] != 0 for r in rows):
        raise ValueError("At least 20 training step-0 prompts are required")
    llm, adapter, params = create_engine(args)
    generated = greedy(llm, adapter, rows, params["max_model_len"])
    result = [{k: row[k] for k in ("task", "mode", "budget_tier", "prompt_sha256",
                                     "raw_output", "output_token_ids")} for row in generated]
    details = {"n": len(result), "rows": result, "engine": params, **source_info(args),
               "environment": sys.prefix, "prompt_file": file_record(args.prompts)}
    if args.compare:
        other = json.loads(args.compare.read_text())
        if (len(other["rows"]) != len(result)
                or any(x["prompt_sha256"] != y["prompt_sha256"] for x, y in zip(result, other["rows"]))):
            raise ValueError("Different reference prompts")
        details["against"] = file_record(args.compare)
        details["text_mismatches"] = sum(x["raw_output"] != y["raw_output"] for x,y in zip(result, other["rows"]))
        details["token_id_mismatches"] = sum(x["output_token_ids"] != y["output_token_ids"] for x,y in zip(result, other["rows"]))
    write_json(args.output_dir / "greedy_reference.json", details)


def rate(count, n):
    return {"count": count, "n": n, "rate": count / n if n else None}


def state_summary(c):
    return {"n": c["n"], **{k: rate(c[k], c["n"]) for k in (
        "invalid", "kept_non_gold", "lost_gold", "prior_gold_missing_from_kept")}}


def make_row(item, output, record, source):
    state = item["state"]
    return {"split": "train", **{k: item[k] for k in ("task", "mode", "budget_tier", "budget", "step", "trajectory_id")},
            **source, "messages": [{"role": "user", "content": render_input(state)}],
            "prompt_token_ids": output["prompt_token_ids"], "prompt_sha256": output["prompt_sha256"],
            "prompt_tokens": output["prompt_tokens"], "reward_state_json": compact({**{k: state[k] for k in REWARD_FIELDS}, "prompt_token_ids_tail": output["prompt_token_ids"][-24:]}),
            "pid_by_lid_json": compact(state["pid_by_lid"]), "greedy_output": record["raw_output"],
            "greedy_output_token_ids": output["output_token_ids"], "greedy_invalid_reason": record["invalid_reason"],
            "greedy_keep_after": record["state"]["keep_after"]}


def verify_rows(path, tasks, corpus, tokenizer):
    task_map = {t["task"]: t for t in tasks}
    refs = read_sft_references(task_map)
    step0 = []
    counts = Counter()
    lengths = Counter()
    groups = defaultdict(Counter)
    for row in lines(path):
        state = json.loads(row["reward_state_json"])
        mapping = {int(k): v for k,v in json.loads(row["pid_by_lid_json"]).items()}
        task, step = task_map[row["task"]], row["step"]
        prior = {r["gold"] for r in task["steps"][:step]}
        seen = prior | {task["steps"][step]["gold"]}
        before = {mapping[i] for i in state["kept_ids"]}
        candidates = state["kept_ids"] + state["new_ids"]
        expected = [i for i in candidates if mapping[i] in seen]
        counts["gold_mismatches"] += expected != state["gold"]
        # Rebuild the visible state from the frozen corpus, independent of the rendered user text.
        restored = {**state, "questions": task["questions"], "ntok": {i:corpus[p]["ntok"] for i,p in mapping.items()},
                    "text": {i:corpus[p]["text"] for i,p in mapping.items()}}
        actual = inference_prompt(tokenizer, restored)
        counts["reencode_mismatches"] += actual["prompt_token_ids"] != row["prompt_token_ids"]
        counts["rerender_mismatches"] += render_input(restored) != row["messages"][0]["content"]
        if step == 0:
            step0.append(row)
        lengths[row["prompt_tokens"]] += 1
        counts["n"] += 1
        counts["exceeds_8392"] += row["prompt_tokens"] > 8392
        counts["prompt_plus_256_exceeds_8392"] += row["prompt_tokens"] + MAX_NEW_TOKENS > 8392
        group = groups[f"{row['mode']}|{row['budget_tier']}"]
        group["n"] += 1
        group["invalid"] += row["greedy_invalid_reason"] is not None
        group["kept_non_gold"] += bool(before - seen)
        group["lost_gold"] += bool(seen - set(mapping.values()))
        group["prior_gold_missing_from_kept"] += bool(prior - before)
    expected_count = len(MODES) * len(BUDGETS) * sum(len(t["steps"]) for t in tasks)
    if counts["n"] != expected_count:
        raise AssertionError((counts["n"], expected_count))
    if len(step0) != len(tasks) * 6 or len({r["trajectory_id"] for r in step0}) != len(step0):
        raise AssertionError("Incomplete step-0 trajectory set")
    pooled = sum(groups.values(), Counter())
    return {"n_tasks": len(tasks), "n_trajectories": len(step0), "n_states": counts["n"], "scope": "train",
            "step0_sft": audit_step_zero(step0, refs), "gold_recompute_n": counts["n"],
            "gold_mismatches": counts["gold_mismatches"], "prompt_reencode_n": counts["n"],
            "reencode_mismatches": counts["reencode_mismatches"], "rerender_mismatches": counts["rerender_mismatches"],
            "prompt_lengths": {**length_distribution(lengths), "min": min(lengths), "max_length": 8392,
                               "exceeds_8392": counts["exceeds_8392"], "prompt_plus_256_exceeds_8392": counts["prompt_plus_256_exceeds_8392"]},
            "cells": {k: state_summary(v) for k,v in groups.items()}, "pooled": state_summary(pooled),
            "metric_definitions": {"kept_non_gold": "any retained paragraph outside step-wise seen gold, before action",
                "lost_gold": "any gold seen through current step unavailable from kept plus new before action",
                "prior_gold_missing_from_kept": "any gold seen before this step absent from retained area, even if retrieved again now"}}


def build(args):
    wall = time.monotonic()
    tasks, corpus = load_assets(args.task_file, args.n_tasks, args.seed)
    source = source_info(args)
    policy = args.adapter.name if args.adapter else args.base_model.name
    replay = HeldoutReplay(tasks, corpus, policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "states.jsonl"
    partial = out.with_name("states.partial.jsonl")
    if out.exists():
        raise FileExistsError(out)
    llm, adapter, params = create_engine(args)
    tok = llm.get_tokenizer()
    init_time = time.monotonic() - wall
    rollout = time.monotonic()
    generation_s = 0.
    with partial.open("x") as stream:
        while not replay.completed():
            items = replay.next_batch()
            prompts = [inference_prompt(tok, item["state"]) for item in items]
            now = time.monotonic()
            outputs = greedy(llm, adapter, prompts, params["max_model_len"])
            generation_s += time.monotonic() - now
            for item, output, rec in zip(items, outputs, replay.advance_batch(outputs)):
                stream.write(compact(make_row(item, output, rec, source)) + "\n")
            stream.flush()
            print(f"step {items[0]['step']} finished ({len(items)} active trajectories)", flush=True)
    rollout_seconds = time.monotonic() - rollout
    summary = verify_rows(partial, tasks, corpus, tok)
    partial.rename(out)
    summary.update(source, task_ids=[t["task"] for t in tasks], engine=params, seed=args.seed, cuda_visible_devices="0",
                   engine_and_data_startup_seconds=init_time, rollout_seconds=rollout_seconds,
                   vllm_generate_seconds=generation_s, wall_including_checks_seconds=time.monotonic()-wall,
                   state_file=file_record(out))
    audit = summary["step0_sft"]
    summary["available_checks_pass"] = not (summary["gold_mismatches"] or summary["reencode_mismatches"]
        or summary["rerender_mismatches"] or audit["direct_mismatches"] or audit["budget_substituted_mismatches"])
    write_json(args.output_dir / "state_checks.json", summary)
    if not summary["available_checks_pass"]:
        raise AssertionError("State verification mismatches; see state_checks.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "build", "reference"), default="build")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, help="LoRA on this base; omit for a merged model alone")
    parser.add_argument("--source-adapter", type=Path, help="Adapter provenance for a merged model")
    parser.add_argument("--model-label", default=LABEL)
    parser.add_argument("--task-file", type=Path, default=TRAIN_TASKS)
    parser.add_argument("--n-tasks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, help="Fixture emitted by --stage prepare")
    parser.add_argument("--compare", type=Path, help="Baseline greedy_reference.json")
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(PREP):
        parser.error(f"--output-dir must be below {PREP}")
    if args.stage == "reference" and (not args.prompts or not args.prompts.resolve().is_relative_to(PREP)):
        parser.error("--prompts must reference the train fixture saved under grpo_prep")
    {"prepare": prompt_proof, "build": build, "reference": reference}[args.stage](args)


if __name__ == "__main__":
    main()
