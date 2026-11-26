"""Export the Task 08 balanced dataset from informative reward-v2 groups, verifying the training ORM."""
import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import make_formal_grpo_dataset as prior
from scripts import probe_grpo_r2 as filtering
from scripts import build_grpo_states as states
from scripts.sample_sft_subset import require, write_json_exclusive

TARGET = 2000
SEED = 20261124
MIN_SIGNAL = 500


def greedy_rewards(groups_path, informative_path):
    """Verify that the informative stream is exactly the ordered subset of the full stream."""
    greedy = {}
    with groups_path.open("rb") as population, informative_path.open("rb") as subset:
        for line in population:
            row = json.loads(line)
            if row["informative"]:
                require(subset.readline() == line,"Informative stream differs from the corresponding full group")
                key = prior.state_key(row)
                require(key not in greedy,"Duplicate informative state")
                greedy[key] = row["greedy_reward"]
        require(not subset.readline(),"Extra records in the informative stream")
    return greedy


def verify_plugin(dataset, pool, greedy, orm):
    selected = {(r["trajectory_id"],r["state_step"]):r for r in states.lines(dataset)}
    checked = 0
    for row in states.lines(pool):
        key = prior.state_key(row)
        if key in selected:
            exported = selected[key]
            require(row["split"] == "train" and exported["reward_state_json"] == row["reward_state_json"]
                    and exported["messages"] == row["messages"]
                    and exported["messages"][0]["content"].encode() == row["user_text"].encode(),
                    f"Exported row differs from the recorded B state {key}")
            actual = orm([row["greedy_output"]], [exported["reward_state_json"]], [row["greedy_output_token_ids"]])
            require(actual == [greedy[key]], f"Plugin reward differs from C greedy reward for {key}")
            checked += 1
    require(checked == len(selected),"Missing B state for training ORM")
    return checked


def make_orm():
    from tokenizers import Tokenizer
    from types import SimpleNamespace
    from src.grpo_r2_rewards import CtxStepRewardV2
    backend = Tokenizer.from_file(str(states.DEFAULT_BASE / "tokenizer.json"))
    return CtxStepRewardV2(SimpleNamespace(backend_tokenizer=backend))


def build(pool, groups, informative, destination, orm, target=TARGET):
    greedy = greedy_rewards(groups,informative)
    summary = prior.build(pool,groups,destination,target,SEED)
    n = summary["n"]
    require(summary["n_informative"] == len(greedy) and n == min(target, len(greedy)),
            "Balanced export count differs from informative population")
    checked = verify_plugin(destination,pool,greedy,orm)
    require(checked == n,"Plugin parity did not cover the exported dataset")
    return {**summary, "status":"complete" if n >= MIN_SIGNAL else "stopped_insufficient_signal",
            "minimum_informative":MIN_SIGNAL,"mu":filtering.MU,"plugin_greedy_checked":checked,
            "plugin_greedy_mismatches":0,"plugin":"CtxStepRewardV2"}


def run():
    proof = filtering.load(filtering.FILTER/"sampling_probe.json")
    require(proof["status"] == "complete" and proof["split"] == "train" and proof["mu"] == filtering.MU
            and proof["informative_threshold"] == filtering.prior.THRESHOLD,"Wrong filter or reward")
    pool = filtering.checked_file(proof["state_file"],filtering.POOL)
    groups = filtering.checked_file(proof["group_file"],filtering.FILTER/"groups.jsonl")
    informative = filtering.checked_file(proof["informative_file"],filtering.FILTER/"informative.jsonl")
    data = filtering.ROOT/"data"
    require(data.resolve() == data and not os.path.lexists(data),"Dataset destination exists or is redirected")
    data.mkdir()
    info = build(pool,groups,informative,data/"train.jsonl",make_orm())
    require(info["n_states"] == proof["n_states"] and
            info["n_informative"] == proof["n_informative"] == proof["pooled"]["informative"]["count"],
            "C/D coverage mismatch")
    write_json_exclusive(data/"dataset_stats.json",{**info,"source_states":proof["state_file"],
        "source_groups":proof["group_file"],"source_informative":proof["informative_file"]})


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()

