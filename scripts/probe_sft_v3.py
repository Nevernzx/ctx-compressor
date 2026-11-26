#!/usr/bin/env python3
"""Run the Task 07 sampler on matched out-of-sample DAgger states for Task 10."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import build_grpo_states as state_ops
from scripts import probe_grpo_sampling as probe
from scripts.build_formal_grpo_states import gpu_gate
from scripts.offline_reward_v2 import MU_GRID
from scripts.prepare_formal_grpo import ADAPTER, BASE, SHA, check_start
from scripts.probe_formal_grpo import COLUMNS, group_record
from scripts.sample_sft_subset import require, write_json_exclusive
from src.grpo_reward import STATE_FIELDS, score_completion
from src.grpo_reward_v2 import score_completion_v2
from src.model_inference import compact, sha256_json

EXT = BASE.parents[1]
OUT = PROJECT / "eval/sft_run_v3"
ROLLS = EXT / "dagger_v3/rollouts/steps.jsonl"
TRAIN_IDS = EXT / "data/sft_v3/sample_ids.json"
STATE_SEED = 20261123
GENERATION_SEED = 20261120
MODES = ("reselect", "append")


def load(path):
    return json.loads(path.read_bytes())


def sample_states(path, excluded, per_mode=150):
    """Sample uniformly within each mode, retaining only row offsets while scanning."""
    offsets = defaultdict(list)
    seen, digest = set(), hashlib.sha256()
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            digest.update(line)
            r = json.loads(line)
            identity = r["state_id"]
            require(r["split"] == "train" and r["mode"] in MODES and
                    identity == f"{r['trajectory_id']}:{r['step']}" and identity not in seen,
                    "Non-training or duplicate DAgger identity")
            seen.add(identity)
            if identity not in excluded:
                offsets[r["mode"]].append(offset)
        require(excluded <= seen, "Unknown SFT training state in exclusions")
        rng = random.Random(STATE_SEED)
        chosen = []
        for mode in MODES:
            for offset in rng.sample(offsets[mode], per_mode):
                f.seek(offset)
                chosen.append(json.loads(f.readline()))
    return chosen, {"sha256":digest.hexdigest(),"bytes":path.stat().st_size,"path":str(path)}, len(seen)


def probe_state(row, source):
    generation = row["generation_metadata"]
    require(sha256_json(generation["prompt_token_ids"]) == generation["prompt_sha256"]
            and len(generation["prompt_token_ids"]) == generation["prompt_tokens"]
            and hashlib.sha256(row["user_text"].encode()).hexdigest() == generation["user_text_sha256"],
            "DAgger prompt hash or length mismatch")
    state = {k:row["state"][k] for k in STATE_FIELDS}
    greedy = score_completion(row["greedy_raw_output"], state)
    require(greedy["keep_after"] == row["greedy_keep_after"] and state["mode"] == row["mode"]
            and row["hedge"]["tau"] == .99, "DAgger state or pre-hedge baseline differs")
    return {**{k:row[k] for k in COLUMNS if k != "prompt_sha256"},
            **{k:generation[k] for k in ("prompt_token_ids","prompt_tokens","prompt_sha256")},
            "state_id":row["state_id"],"reward_state_json":compact(state),
            "greedy_output":row["greedy_raw_output"],"greedy_keep_after":greedy["keep_after"],
            "greedy_invalid_reason":greedy["invalid_reason"], **source}


def group_stats(raw, row):
    require(all(raw[k] == row[k] for k in COLUMNS), "Probe output identity changed")
    parsed = group_record(raw)
    counts = parsed["add_counts"]
    samples = raw["samples"]
    require(all(len(s["token_ids"]) <= 256 for s in samples), "Completion exceeded registered limit")
    requested = set()
    executed = set()
    zero_v2 = {}
    state = json.loads(row["reward_state_json"])
    for s in samples:
        a = s["action"]
        requested.add(None if a is None else (frozenset(a["add"]), frozenset(a.get("drop", []))))
        executed.add(frozenset(s["keep_after"]))
    for mu in MU_GRID:
        values = []
        for s in samples:
            priced = score_completion_v2(s["completion"], state, mu)
            require(all(priced[k] == s[k] for k in ("keep_after","invalid_reason","used_after")),
                    "v1/v2 parser or execution disagreement")
            values.append(priced["reward"])
        zero_v2[str(mu)] = int(len(set(values)) == 1)
    return Counter(n=1, completions=len(samples), parsed=sum(c is not None for c in counts),
                   multi_add=sum(c is not None and c >= 2 for c in counts),
                   valid_multi_add=sum(c is not None and c >= 2 and reason is None
                                       for c, reason in zip(counts, parsed["invalid_reasons"])),
                   invalid=sum(r is not None for r in parsed["invalid_reasons"]),
                   distinct=len(requested), distinct_executed=len(executed),
                   zero_v1=int(parsed["population_std"] == 0), **zero_v2)


def summarize(c):
    groups, samples = c["n"], c["completions"]
    require(samples == groups * 8 and groups > 0, "Incomplete sampling groups")
    def mean(key):
        return {"sum":c[key],"n":groups,"mean":c[key]/groups}
    return {"n_groups":groups,"n_completions":samples,
            "multi_add_rate":probe.rate(c["multi_add"],samples),
            "multi_add_rate_among_parsed":probe.rate(c["multi_add"],c["parsed"]),
            "executed_multi_add_rate":probe.rate(c["valid_multi_add"],samples),
            "invalid_rate":probe.rate(c["invalid"],samples),
            "mean_distinct_actions":mean("distinct"),"mean_distinct_executed_actions":mean("distinct_executed"),
            "zero_std_v1":probe.rate(c["zero_v1"],groups),
            "zero_std_v2":{str(mu):probe.rate(c[str(mu)],groups) for mu in MU_GRID}}


def adapter_identity(model):
    check_start()
    if model == "v2":
        return ADAPTER, SHA
    decision = load(OUT / "selected_checkpoint.json")
    path = Path(decision["path"])
    step = decision["checkpoint_step"]
    require(path.resolve() == path and path.is_relative_to(EXT/"runs") and
            decision["strategy"] == path.name == f"checkpoint-{step}" and step in (1000,2000,3000,3750),
            "Selected v3 adapter identity is inconsistent")
    actual = state_ops.sha256(path / "adapter_model.safetensors")
    require(decision["adapter_sha256"] == actual, "Selected v3 adapter bytes differ")
    return path, actual


def run(model):
    require(PROJECT == Path("/root/workplace/ctx-compressor") and Path(sys.prefix).resolve() == EXT/"envs/grpo",
            "Use the pinned 5880 grpo environment")
    gpu_gate(1)
    from vllm import SamplingParams
    sampler = SamplingParams(n=8,temperature=1.,top_p=1.,max_tokens=256,seed=GENERATION_SEED)
    require(type(sampler.top_k) is int and sampler.top_k in (0,-1), "Unrecognized sampling top_k")
    receipt = OUT / f"probe_{model}.json"
    target = EXT / f"dagger_v3/probe/{model}"
    require(not receipt.exists() and not target.exists(), "Task 10 probe outputs exist")
    adapter, adapter_sha = adapter_identity(model)
    ids_record = state_ops.file_record(TRAIN_IDS)
    manifest = load(OUT / "sample_manifest.json")
    require(manifest["files"]["sample_ids"] == ids_record and manifest["selected_by_mode"] == dict.fromkeys(MODES,5000),
            "SFT training exclusion list changed after D")
    excluded = load(TRAIN_IDS)["dagger_state_ids"]
    require(len(excluded) == len(set(excluded)) == 10000, "Expected 10,000 unique sampled training states")
    selected, input_record, n_total = sample_states(ROLLS, set(excluded))
    rollout_proof = load(OUT / "dagger_rollouts.json")
    require(rollout_proof["status"] == "complete" and rollout_proof["steps_file"] == input_record
            and rollout_proof["n_steps"] == n_total == 22158 and rollout_proof["adapter_sha256"] == SHA,
            "Input state population differs from C receipt")
    ids = [r["state_id"] for r in selected]
    per_mode = Counter(r["mode"] for r in selected)
    overlap = len(set(ids).intersection(excluded))
    require(len(ids) == len(set(ids)) == 300 and per_mode == dict.fromkeys(MODES,150) and overlap == 0,
            "Probe sample size or overlap failed")
    ids_sha = sha256_json(ids)
    if model == "v2":
        v3 = load(OUT / "probe_v3.json")
        require(v3["selected_ids_sha256"] == ids_sha and v3["status"] == "complete"
                and v3["population_file"] == input_record and v3["training_ids_file"] == ids_record,
                "Comparison requires exactly the same training exclusions and prompt identities")
    args = argparse.Namespace(states=target/"sampled_states.jsonl", output_dir=target/"sampler",
                              base_model=BASE, adapter=adapter, source_adapter=ADAPTER, gpu=1,
                              model_label=f"SFT {model} {adapter.name}",seed=GENERATION_SEED,
                              generations=8,per_cell=None,lam=.3)
    # Both policies use the recorded v2 greedy baseline from the DAgger trajectory.
    # source_adapter identifies that baseline; inference_adapter identifies the actual sampler.
    source = {**state_ops.source_info(args), "sampling_adapter_sha256":adapter_sha,
              "greedy_baseline":"SFT v2 checkpoint-2000 before the Task 09 hedge wrapper"}
    chosen = [probe_state(r,source) for r in selected]
    target.mkdir(parents=True)
    with args.states.open("x") as f:
        for row in chosen:
            f.write(compact(row)+"\n")
    def write_sampler_summary(path, result):
        result["sampling_order"] = {"modes":MODES, "states_per_mode":150, "state_selection_seed":STATE_SEED}
        result.update(source)
        write_json_exclusive(path,result)
    with patch.object(state_ops,"check_gpu",gpu_gate):
        llm, lora, engine = state_ops.create_engine(args)
    try:
        with patch.multiple(probe, stratified=lambda *unused: chosen, source_info=lambda _:source,
                            create_engine=lambda _:(llm,lora,engine), write_json=write_sampler_summary):
            probe.run(args)
    finally:
        llm.llm_engine.engine_core.shutdown()
    raw_path = args.output_dir/"probe_outputs.jsonl"
    raw = list(state_ops.lines(raw_path))
    require(len(raw) == len(chosen), "Missing probe groups")
    counts = defaultdict(Counter)
    for group, row in zip(raw,chosen):
        require(all(group[k] == source[k] for k in source), "Sampler provenance mismatch")
        stats = group_stats(group,row)
        counts["pooled"].update(stats)
        counts[row["mode"]].update(stats)
    tables = {scope:summarize(c) for scope,c in counts.items()}
    old = load(args.output_dir/"sampling_probe.json")
    require(old["pooled"]["zero_std_groups"] == tables["pooled"]["zero_std_v1"]
            and old["pooled"]["invalid"] == tables["pooled"]["invalid_rate"]
            and old["raw_samples"] == state_ops.file_record(raw_path), "Task 07 sampling summary differs")
    summary = {"status":"complete", "model":model, "n_groups":len(raw),"n_completions":len(raw)*8,
               "adapter":str(adapter),"adapter_sha256":adapter_sha,"training_overlap":overlap,
               "excluded_train_n":len(excluded),"per_mode":dict(per_mode),"selected_ids_sha256":ids_sha,
               "n_dagger_states":n_total,"population_file":input_record,"training_ids_file":ids_record,
               "sampled_state_file":state_ops.file_record(args.states),"raw_samples":old["raw_samples"],
               "raw_sampler_receipt":state_ops.file_record(args.output_dir/"sampling_probe.json"),
               "state_seed":STATE_SEED,"generation":{**old["generation"],"seed":GENERATION_SEED,
               "top_k_requested":-1,"top_k_actual":sampler.top_k,"top_k_interpretation":"0/-1 均为不截断。"},
               "lambda":0.3,"mu_grid":MU_GRID,"engine":engine,"tables":tables,
               "definitions":{"multi_add_rate":"可解析且请求至少两个不同的新增编号，占全部生成样本。",
               "mean_distinct_actions":"每组不同的请求动作数，新增和丢弃各按编号集合比较；无法解析的动作合为一类。",
               "mean_distinct_executed_actions":"每组执行后的不同保留集合数，无效动作执行为空操作。",
               "zero_std":"每组八条生成的总体标准差精确等于零。","greedy_baseline":source["greedy_baseline"]}}
    write_json_exclusive(receipt,summary)
    print(f"{model}: groups={len(raw)}, samples={len(raw)*8}, overlap={overlap}, ids_sha={ids_sha}",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("v3","v2"), required=True)
    run(p.parse_args().model)

