#!/usr/bin/env python3
"""Run the Task 07 sampler over every verified state and retain the 8-way filter."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import probe_grpo_sampling as probe
from scripts import build_grpo_states as state_ops
from scripts.build_formal_grpo_states import COLUMNS, LABEL, formal_path, gpu_gate
from src.heldout_eval import INVALID_REASONS

CELLS = tuple(sorted(f"{m}|{t}" for m in probe.MODES for t in probe.BUDGETS))
THRESHOLD = 1e-9
BUCKETS = ("0", "1", "2", ">=3")


def cell(row):
    name = f"{row['mode']}|{row['budget_tier']}"
    if row["split"] != "train" or name not in CELLS:
        raise ValueError("Expected a train state in one of the six configured cells")
    return name


def state_key(row):
    return row["trajectory_id"], row["step"]


class IndexedRows:
    """Disk-backed sequence with the slicing interface used by the existing batch loop."""
    def __init__(self, path, offsets):
        self.path = path
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets)

    def __iter__(self):
        with self.path.open("rb") as stream:
            for offset in self.offsets:
                stream.seek(offset)
                yield json.loads(stream.readline())

    def __getitem__(self, sl):
        return list(IndexedRows(self.path, self.offsets[sl]))


def index_states(path):
    offsets = defaultdict(list)
    keys = set()
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            key = state_key(row)
            if key in keys:
                raise ValueError(f"Duplicate state {key}")
            keys.add(key)
            offsets[cell(row)].append(offset)
    if set(offsets) != set(CELLS):
        raise ValueError("State pool lacks a configured cell")
    return offsets


def group_record(raw):
    samples = sorted(raw["samples"], key=lambda s:s["index"])
    if [s["index"] for s in samples] != list(range(8)):
        raise ValueError("Missing or duplicate generation index")
    rewards = [s["reward"] for s in samples]
    reasons = [s["invalid_reason"] for s in samples]
    if any(v not in (None, *INVALID_REASONS) for v in reasons):
        raise ValueError("Unknown invalid reason")
    # Count distinct requested IDs in the parsed output. Unparseable JSON has no add list.
    counts = [len(set(s["action"]["add"])) if s["action"] is not None else None for s in samples]
    sd = statistics.pstdev(rewards)
    return {**{k:raw[k] for k in COLUMNS}, "cell":cell(raw), "rewards":rewards,
            "invalid_reasons":reasons, "add_counts":counts, "population_std":sd,
            "informative":sd > THRESHOLD, "greedy_reward":raw["greedy_reward"]}


def describe(groups, elapsed):
    compact = [{"samples":[{"reward":r, "invalid_reason":i} for r,i in zip(g["rewards"],g["invalid_reasons"])],
                "greedy_reward":g["greedy_reward"]} for g in groups]
    summary = probe.summarize(compact)
    counts = [c for g in groups for c in g["add_counts"]]
    parsed = Counter(">=3" if c >= 3 else str(c) for c in counts if c is not None)
    executed = Counter()
    for g in groups:
        for c, invalid in zip(g["add_counts"],g["invalid_reasons"]):
            count = 0 if invalid else c
            executed[">=3" if count >= 3 else str(count)] += 1
    informative = sum(g["informative"] for g in groups)
    return {**summary,"informative":probe.rate(informative,len(groups)),
            "std_le_threshold":probe.rate(len(groups)-informative,len(groups)),
            "sampled_add_counts_parsed":{k:probe.rate(parsed[k],sum(parsed.values())) for k in BUCKETS},
            "unparseable_add_count":probe.rate(counts.count(None),len(counts)),
            "executed_add_counts_invalid_noop":{k:probe.rate(executed[k],len(counts)) for k in BUCKETS},
            "elapsed_seconds":elapsed,"states_per_second":len(groups)/elapsed}


def run(args):
    from vllm import SamplingParams
    # Task 07's sampling call specifies n, temperature, top_p, max_tokens and seed.
    # Its pinned vLLM default for the remaining top_k parameter is validated here.
    top_k_actual = SamplingParams(n=8, temperature=1., top_p=1., max_tokens=256, seed=args.seed).top_k
    if type(top_k_actual) is not int or top_k_actual not in (0, -1):
        raise ValueError(f"Unexpected vLLM top_k default {top_k_actual!r}; expected 0 or -1")
    formal_path(args.states)
    out = formal_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    proof = json.loads((args.states.parent/"state_checks.json").read_text())
    frozen = probe.file_record(args.states)
    if proof["state_file"] != frozen or not proof["available_checks_pass"] or proof["heldout_overlap"] != 0:
        raise ValueError("State verification receipt is missing or inconsistent")
    offsets = index_states(args.states)
    source = probe.source_info(args)
    with patch.object(state_ops,"check_gpu",gpu_gate):
        llm, adapter, engine = state_ops.create_engine(args)
    def fixed_engine(unused):
        return llm, adapter, engine
    record_file = probe.file_record
    def cached_record(path):
        return frozen if path == args.states else record_file(path)
    all_groups = []
    summaries = {}
    shard_sources = {}
    elapsed_total = 0.
    with (out/"groups.jsonl").open("x", encoding="utf-8") as stream:
        for name in CELLS:
            rows = IndexedRows(args.states, offsets[name])
            target = out/name.replace("|","_")
            def select_cell(*unused):
                return rows
            arg_cell = argparse.Namespace(**vars(args))
            arg_cell.output_dir = target
            arg_cell.per_cell = len(rows)
            with patch.multiple(probe, stratified=select_cell, create_engine=fixed_engine, file_record=cached_record):
                probe.run(arg_cell)
            measured = json.loads((target/"sampling_probe.json").read_text())
            secs = measured["elapsed_excluding_engine_startup_seconds"]
            elapsed_total += secs
            compact_groups = []
            for row in state_ops.lines(target/"probe_outputs.jsonl"):
                group = group_record(row)
                if group["cell"] != name:
                    raise ValueError("Mixed sampling strata")
                compact_groups.append(group)
                stream.write(state_ops.compact(group)+"\n")
            if len(compact_groups) != len(rows):
                raise ValueError("Incomplete state population")
            stream.flush()
            summaries[name] = describe(compact_groups,secs)
            all_groups.extend(compact_groups)
            shard_sources[name] = measured["raw_samples"]
    if len(all_groups) != proof["n_states"]:
        raise ValueError("Filter did not cover the full state population")
    probe.write_json(out/"sampling_probe.json",{**source,"state_file":frozen,"split":"train",
        "n_states":len(all_groups),"generation":{"n":8,"temperature":1.0,"top_p":1.0,"top_k":-1,
        "top_k_actual":top_k_actual,
        "top_k_interpretation":"vLLM 0.11.0 accepts 0 and -1 to disable top-k truncation. "
                               "The registered top_k is -1. The Task 07 sampler uses the runtime default.",
        "max_tokens":256,"seed":args.seed},"lambda":args.lam,"ddof":0,"informative_threshold":THRESHOLD,
        "group_file":record_file(out/"groups.jsonl"),"raw_samples_by_cell":shard_sources,
        "timing":"stage sampling, excluding engine initialization and state checks",
        "add_count_definition":"Number of distinct requested add IDs when parseable; malformed JSON is null. Executed count treats invalid actions as no-ops.",
        "engine":engine,"pooled":describe(all_groups,elapsed_total),"cells":summaries})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all-states",action="store_true",required=True)
    p.add_argument("--states",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--base-model",type=Path,default=probe.DEFAULT_BASE)
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--source-adapter",type=Path)
    p.add_argument("--model-label",default=LABEL)
    p.add_argument("--gpu",type=int,choices=(1,),default=1)
    p.add_argument("--seed",type=int,default=20261120,choices=(20261120,))
    p.add_argument("--generations",type=int,default=8,choices=(8,))
    p.add_argument("--lam",type=float,default=0.3,choices=(0.3,))
    run(p.parse_args())


if __name__ == "__main__":
    main()
