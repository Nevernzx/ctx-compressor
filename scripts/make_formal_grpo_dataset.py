#!/usr/bin/env python3
"""Prepare the formal balanced Swift dataset using the frozen Task 07 exporter."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_grpo_states as state_ops
from scripts import make_grpo_dataset as legacy
from scripts.build_formal_grpo_states import formal_path
from scripts.probe_formal_grpo import CELLS, THRESHOLD, cell, state_key


def quotas(available, target):
    target = min(target, sum(available.values()))
    minimum, remainder = divmod(target, len(CELLS))
    picked = {c: min(available[c], minimum + (idx < remainder)) for idx,c in enumerate(CELLS)}
    remaining = target - sum(picked.values())
    while remaining:
        for c in CELLS:
            if remaining and picked[c] < available[c]:
                picked[c] += 1
                remaining -= 1
    return picked


def filter_signals(path):
    signals = {}
    available = Counter({c:0 for c in CELLS})
    for group in state_ops.lines(path):
        key = state_key(group)
        name = cell(group)
        rewards = group["rewards"]
        if key in signals or group["cell"] != name:
            raise ValueError("Duplicate or inconsistent filter identity")
        if any(len(group[k]) != 8 for k in ("rewards","invalid_reasons","add_counts")):
            raise ValueError("Incomplete reward group")
        if not all(math.isfinite(r) for r in rewards):
            raise ValueError("Nonfinite sampled reward")
        sd = statistics.pstdev(rewards)
        informative = sd > THRESHOLD
        if sd != group["population_std"] or informative is not group["informative"]:
            raise ValueError("Filter threshold disagrees with recomputed population std")
        signals[key] = (name, group["prompt_sha256"], informative)
        available[name] += informative
    return signals, available


def candidate_lines(path, signals):
    seen = set()
    for row in state_ops.lines(path):
        key = state_key(row)
        name = cell(row)
        if key in seen or key not in signals or signals[key][:2] != (name,row["prompt_sha256"]):
            raise ValueError("B and C state identities do not match one-to-one")
        seen.add(key)
        if signals[key][2]:
            message = row["messages"]
            if message != [{"role":"user","content":row["user_text"]}]:
                raise ValueError("State prompt differs from B user_text")
            packed = json.loads(row["reward_state_json"])
            if packed["prompt_token_ids_tail"] != row["prompt_token_ids"][-24:]:
                raise ValueError("Incorrect reward token suffix")
            yield state_ops.compact({k:row[k] for k in
                ("messages","reward_state_json","trajectory_id","step","mode","budget_tier","split")})+"\n"
    if seen != signals.keys():
        raise ValueError("Filter does not cover exactly the B population")


def build(states, informative, output, target, seed):
    if output.exists():
        raise FileExistsError(output)
    signals, available = filter_signals(informative)
    counts = quotas(available,target)

    # Task 07 uses one Random for sorted-cell selection and the final shuffle.
    # Change only the requested per-cell sample size; reuse its Swift row writer.
    class QuotaRandom(random.Random):
        def sample(self, population, k):
            return super().sample(population, counts[cell(population[0])])

    def train_source(path):
        if Path(path) != states:
            raise ValueError("Expected the verified train state stream")
        return candidate_lines(states, signals)

    cli = ["make_grpo_dataset.py","--states",str(states),"--out",str(output),
           "--per-cell","0","--seed",str(seed)]
    with patch.object(sys,"argv",cli), patch.object(legacy,"open",train_source,create=True), \
         patch.object(legacy,"random",SimpleNamespace(Random=QuotaRandom)):
        legacy.main()
    selected = {}
    actual_counts = Counter({c:0 for c in CELLS})
    for row in state_ops.lines(output):
        key = row["trajectory_id"],row["state_step"]
        if key in selected or key not in signals or signals[key][2] is not True or signals[key][0] != row["cell"]:
            raise ValueError("Invalid exported identity")
        selected[key] = row
        actual_counts[row["cell"]] += 1
    if dict(actual_counts) != counts:
        raise ValueError("Exported cell allocation differs from plan")
    checked = set()
    for raw in state_ops.lines(states):
        key = state_key(raw)
        if key in selected:
            out = selected[key]
            if (raw["split"] != "train" or out["messages"] != raw["messages"]
                    or out["messages"][0]["content"].encode() != raw["user_text"].encode()
                    or out["reward_state_json"] != raw["reward_state_json"]):
                raise ValueError("Exported row differs from its B train state")
            checked.add(key)
    if checked != selected.keys():
        raise ValueError("Missing exported state in B")
    return {"n":len(selected),"target_n":target,"n_states":len(signals),"n_informative":sum(available.values()),
            "per_cell_available":dict(available),"per_cell_counts":counts,"seed":seed,
            "heldout_rows":0,"user_text_mismatches":0,"rows_checked":len(checked),
            "shortfall_from_target":target-len(selected),"informative_std_threshold":THRESHOLD,
            "output":state_ops.file_record(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states",type=Path,required=True)
    p.add_argument("--informative",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,choices=(20261121,),default=20261121)
    p.add_argument("--n-prompts",type=int,default=2000,choices=(2000,))
    args=p.parse_args()
    states,groups,out = [formal_path(k) for k in (args.states,args.informative,args.out)]
    out.parent.mkdir(parents=True,exist_ok=True)
    summary = out.parent/"dataset_stats.json"
    if summary.exists() or out.exists():
        raise FileExistsError(out.parent)
    proof=json.loads((groups.parent/"sampling_probe.json").read_text())
    if proof["state_file"] != state_ops.file_record(states) or proof["group_file"] != state_ops.file_record(groups):
        raise ValueError("Filter and training states do not match the verified source hashes")
    result=build(states,groups,out,args.n_prompts,args.seed)
    if result["n_informative"] != proof["pooled"]["informative"]["count"] or result["n_states"] != proof["n_states"]:
        raise ValueError("Filter and dataset coverage disagree")
    state_ops.write_json(summary,{**result,"source_states":proof["state_file"],"source_groups":proof["group_file"]})


if __name__ == "__main__":
    main()
