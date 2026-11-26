"""Filter every verified Task 11 B state using eight samples under reward v2."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import build_grpo_states as states
from scripts import probe_formal_grpo as prior
from scripts.build_formal_grpo_states import COLUMNS, gpu_gate
from scripts.sample_sft_subset import require, write_json_exclusive
from src.grpo_reward_v2 import score_completion_v2

ROOT = states.EXTERNAL / "grpo_r2"
POOL = ROOT / "states/states.jsonl"
FILTER = ROOT / "filter"
START = PROJECT / "eval/grpo_r2/start_model.json"
SEED = 20261123
MU = .5
CELLS = prior.CELLS


def load(path):
    return json.loads(Path(path).read_bytes())


def checked_file(record, expected):
    require(expected.resolve() == expected and states.file_record(expected) == record,
            f"Changed or redirected artifact {expected}")
    return expected


def score_group(row, raw):
    """Apply the frozen eval parser to the actual greedy and sampled continuations."""
    require(all(row[k] == raw[k] for k in COLUMNS), "Sampling and B state identities differ")
    state = json.loads(row["reward_state_json"])
    greedy = score_completion_v2(row["greedy_output"], state, MU)
    require(greedy["keep_after"] == row["greedy_keep_after"] and
            greedy["invalid_reason"] == row["greedy_invalid_reason"], "Greedy execution differs from B")
    samples = sorted(raw["samples"], key=lambda s: s["index"])
    require([s["index"] for s in samples] == list(range(8)), "Missing or duplicate generation index")
    scored = []
    parts = []
    for s in samples:
        require(len(s["token_ids"]) <= 256, "Completion length exceeds 256")
        result = score_completion_v2(s["completion"], state, MU)
        if "keep_after" in s:
            require(all(s[k] == result[k] for k in ("keep_after", "invalid_reason", "used_after")),
                    "Historical sample execution differs from the frozen eval parser")
        scored.append({"index":s["index"], **result})
        parts.append({"n_lost":result["n_lost"], "used_after":result["used_after"],
                      "invalid":int(not result["action_valid"])})
    group = prior.group_record({**raw, "samples":scored, "greedy_reward":greedy["reward"]})
    gold_varies = len({(p["n_lost"],p["invalid"]) for p in parts}) > 1
    return {**group, "reward_parts":parts, "gold_varying":gold_varies,
            "token_only_varying":not gold_varies and len({p["used_after"] for p in parts}) > 1}


def summarize(groups, elapsed):
    # The Task 10 probe clocks all six interleaved strata together. It has no individual cell clock.
    stats = prior.describe(groups, elapsed if elapsed is not None else 1.0)
    if elapsed is None:
        stats.update(elapsed_seconds=None, states_per_second=None)
    n = len(groups)
    stats.update(gold_varying=states.rate(sum(g["gold_varying"] for g in groups), n),
                 token_only_varying=states.rate(sum(g["token_only_varying"] for g in groups), n))
    counts = [c for g in groups for c in g["add_counts"]]
    stats["multi_add_rate"] = states.rate(sum(c is not None and c >= 2 for c in counts),len(counts))
    stats["completions_per_second"] = (8*n/elapsed) if elapsed is not None else None
    return stats


def tables(groups_by_cell, cell_times, pooled_seconds):
    require(set(groups_by_cell) == set(CELLS), "Incomplete six-cell sampling grid")
    return {"cells": {c:summarize(groups_by_cell[c],cell_times.get(c)) for c in CELLS},
            "pooled": summarize([g for c in CELLS for g in groups_by_cell[c]], pooled_seconds)}


def archived_groups(rows, raw_file, groups_by_cell):
    for state, raw in zip(rows, states.lines(raw_file), strict=True):
        group = score_group(state, raw)
        groups_by_cell[group["cell"]].append(group)


def task08_reference():
    folder = states.EXTERNAL / "grpo_v2/filter"
    receipt = folder / "sampling_probe.json"
    published = load(receipt)
    state_file = checked_file(published["state_file"], states.EXTERNAL / "grpo_v2/states/states.jsonl")
    offsets = prior.index_states(state_file)
    groups = defaultdict(list)
    cell_times = {}
    for c in CELLS:
        samples = checked_file(published["raw_samples_by_cell"][c], folder / c.replace("|","_") / "probe_outputs.jsonl")
        archived_groups(prior.IndexedRows(state_file,offsets[c]),samples,groups)
        cell_times[c] = published["cells"][c]["elapsed_seconds"]
    priced = tables(groups,cell_times,sum(cell_times.values()))
    require(priced["pooled"]["n_groups"] == published["n_states"] and
            priced["pooled"]["invalid"] == published["pooled"]["invalid"], "Task 08 reference count mismatch")
    return {"receipt":states.file_record(receipt), "sample_policy":published["model_output_label"],
            "greedy_baseline":published["model_output_label"], "task08_registered_reward_v1":
            {"pooled":published["pooled"], "cells":published["cells"]}, "reward_v2":priced}


def task10_reference(start):
    version = start["version"]
    require(version in ("v2","v3"), "Unknown start checkpoint family")
    receipt = PROJECT / f"eval/sft_run_v3/probe_{version}.json"
    published = load(receipt)
    require(published["status"] == "complete" and published["model"] == version and
            published["adapter_sha256"] == start["adapter_sha256"] and published["n_groups"] == 300,
            "Task 10 F sampling model differs from the Task 11 start")
    folder = states.EXTERNAL / f"dagger_v3/probe/{version}"
    sampled = checked_file(published["sampled_state_file"], folder/"sampled_states.jsonl")
    raw = checked_file(published["raw_samples"],folder/"sampler/probe_outputs.jsonl")
    t = load(checked_file(published["raw_sampler_receipt"],folder/"sampler/sampling_probe.json"))
    groups = defaultdict(list)
    archived_groups(states.lines(sampled),raw,groups)
    priced = tables(groups,{},t["elapsed_excluding_engine_startup_seconds"])
    current, cited = priced["pooled"],published["tables"]["pooled"]
    require(current["zero_std_groups"] == cited["zero_std_v2"][str(MU)] and
            current["multi_add_rate"] == cited["multi_add_rate"] and current["invalid"] == cited["invalid_rate"],
            "Task 10 archived samples differ from the published probe")
    return {"receipt":states.file_record(receipt), "sample_policy":published["adapter"],
            "greedy_baseline":published["definitions"]["greedy_baseline"], "reward_v2":priced}


def differences(current, reference):
    """Differences of point estimates with both explicit denominators."""
    def group(a,b):
        def delta(x,y):
            return {"delta":x["rate"]-y["rate"] if x["n"] and y["n"] else None,
                    "r2_n":x["n"],"reference_n":y["n"]}
        result = {k:delta(a[k],b[k]) for k in ("zero_std_groups","informative","gold_varying",
                  "token_only_varying","some_sample_beats_greedy","multi_add_rate","invalid")}
        for k in ("sampled_add_counts_parsed","invalid_by_category"):
            result[k] = {name:delta(a[k][name],v) for name,v in b[k].items()}
        result["states_per_second"] = {"delta":a["states_per_second"]-b["states_per_second"]
            if b["states_per_second"] is not None else None, "r2_n":a["n_groups"], "reference_n":b["n_groups"]}
        return result
    return {"pooled":group(current["pooled"],reference["pooled"]),
            "cells":{k:group(current["cells"][k],reference["cells"][k]) for k in CELLS}}


def sample_cell(rows, llm, adapter, engine, sampling, source, out_raw, out_groups, out_informative):
    groups = []
    tic = time.monotonic()
    with out_raw.open("x",encoding="utf-8") as stream:
        for pos in range(0,len(rows),16):
            batch = rows[pos:pos+16]
            for row in batch:
                require(all(row[k] == source[k] for k in ("base_model","inference_adapter","source_adapter",
                                                          "source_adapter_sha256")), "B model identity differs")
                states.guard_prompt(row,engine["max_model_len"],identity=prior.state_key(row))
            outputs = llm.generate([{"prompt_token_ids":r["prompt_token_ids"]} for r in batch],sampling,
                                   lora_request=adapter,use_tqdm=False)
            for row,output in zip(batch,outputs,strict=True):
                require(list(output.prompt_token_ids) == row["prompt_token_ids"],"Sampler changed prompt IDs")
                raw = {**{k:row[k] for k in COLUMNS}, "samples":[{"index":s.index, "completion":s.text,
                    "token_ids":list(s.token_ids), "finish_reason":s.finish_reason} for s in output.outputs]}
                g = score_group(row,raw)
                stream.write(states.compact(raw)+"\n")
                line = states.compact(g)+"\n"
                out_groups.write(line)
                if g["informative"]:
                    out_informative.write(line)
                groups.append(g)
            stream.flush()
            out_groups.flush()
            out_informative.flush()
            if pos % 320 == 0:
                print(f"{prior.cell(batch[0])}: sampled {pos+len(batch)}/{len(rows)}",flush=True)
    return groups,time.monotonic()-tic


def run():
    from vllm import SamplingParams
    start = load(START)
    adapter_path = Path(start["path"])
    require(adapter_path.resolve() == adapter_path and adapter_path.is_relative_to(states.EXTERNAL/"runs")
            and adapter_path.name == start["strategy"],"Unexpected start adapter")
    args = argparse.Namespace(base_model=states.DEFAULT_BASE,adapter=adapter_path,source_adapter=adapter_path,
                              model_label=start["strategy"],gpu=1)
    source = states.source_info(args)
    proof = load(POOL.parent/"state_checks.json")
    require(checked_file(proof["state_file"],POOL) == POOL and proof["scope"] == "train"
            and proof["available_checks_pass"] and proof["heldout_overlap"] == proof["locked_test_overlap"] == 0
            and all(proof[k] == 0 for k in ("gold_mismatches","reencode_mismatches","rerender_mismatches"))
            and source["source_adapter_sha256"] == start["adapter_sha256"] == proof["source_adapter_sha256"],
            "B state integrity or sampling adapter mismatch")
    offsets = prior.index_states(POOL)
    require(sum(map(len,offsets.values())) == proof["n_states"],"B state count mismatch")
    sampling = SamplingParams(n=8, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=256, seed=SEED)
    require(sampling.top_k in (-1,0),"Top-k truncation must be disabled")
    require(FILTER.resolve() == FILTER and not os.path.lexists(FILTER),"Filter output already exists")
    FILTER.mkdir()
    raw_files, groups, times = {},{},{}
    with patch.object(states,"check_gpu",gpu_gate):
        llm,adapter,engine = states.create_engine(args)
    try:
        with (FILTER/"groups.jsonl").open("x") as out, (FILTER/"informative.jsonl").open("x") as informative:
            for c in CELLS:
                rows = prior.IndexedRows(POOL,offsets[c])
                raw = FILTER/(c.replace("|","_")+"_samples.jsonl")
                groups[c],times[c] = sample_cell(rows,llm,adapter,engine,sampling,source,raw,out,informative)
                raw_files[c] = states.file_record(raw)
    finally:
        llm.llm_engine.engine_core.shutdown()
    stats = tables(groups,times,sum(times.values()))
    require(stats["pooled"]["n_groups"] == proof["n_states"],"Incomplete sampling pass")
    references = {"task08_c":task08_reference(),"task10_f":task10_reference(start)}
    for ref in references.values():
        ref["delta_r2_minus_reference_v2"] = differences(stats,ref["reward_v2"])
    result = {"status":"complete","n_states":proof["n_states"],"n_informative":stats["pooled"]["informative"]["count"],
        "seed":SEED,"mu":MU,"ddof":0,"split":"train","informative_threshold":prior.THRESHOLD,
        "generation":{"n":8,"temperature":1.0,"top_p":1.0,"top_k":-1,"top_k_actual":sampling.top_k,
                      "max_tokens":256,"seed":SEED},"state_file":proof["state_file"],"start_receipt":states.file_record(START),
        "group_file":states.file_record(FILTER/"groups.jsonl"),"informative_file":states.file_record(FILTER/"informative.jsonl"),
        "raw_samples_by_cell":raw_files,"engine":engine,**source,**stats,"references":references,
        "comparison_basis":"Task 08 original v1 aggregates are retained. Deltas use archived continuations rescored at mu=0.5, on each experiment's own states. Task 10 F uses its recorded pre-hedge SFT v2 greedy baseline, including for v3 samples.",
        "timing":"Generation and per-group scoring/writes, excluding engine initialization and historical comparisons. Task 10 F interleaves strata and times only the pooled run."}
    write_json_exclusive(FILTER/"sampling_probe.json",result)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()

