"""Round 2 on-policy states from the selected SFT adapter, on new training tasks."""
from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import build_formal_grpo_states as formal
from scripts import sft_v3_config as cfg
from scripts.sample_sft_subset import require, write_json_exclusive
from src.grpo_reward_v2 import score_completion_v2

old = formal.old
RUN = cfg.ROOT / "grpo_r2"
SMALL = PROJECT / "eval/grpo_r2"
R2_SEED = 20261123
FROZEN_08 = cfg.ROOT / "grpo_v2/states/state_checks.json"
FROZEN_10 = cfg.OUT / "dagger_rollouts.json"


def start_model():
    decision = cfg.read(SMALL / "start_model.json")
    name, version = decision["strategy"], decision["version"]
    if version == "v2":
        require(name == "checkpoint-2000", "unexpected v2 starting checkpoint")
        path = cfg.v2.RUN / "checkpoints" / name
    elif version == "v3":
        step = int(name.removeprefix("checkpoint-"))
        require(step in cfg.EVAL and name == f"checkpoint-{step}", "unexpected v3 checkpoint")
        path = cfg.checkpoint(step)
    else:
        raise ValueError(f"unexpected SFT start version {version}")
    require(decision["path"] == str(path) and decision["adapter_sha256"] ==
            old.sha256(path / "adapter_model.safetensors"), "starting adapter identity mismatch")
    return decision


def select_tasks(population, previous08, previous10, n=400):
    require(all(t["split"] == "train" for t in population), "non-training task in round 2 population")
    ids = {t["task"] for t in population}
    require(len(population) == len(ids), "duplicate training task ID")
    excluded = set(previous08) | set(previous10)
    require(excluded <= ids, "previous training task missing from population")
    eligible = [t for t in population if t["task"] not in excluded]
    return random.Random(R2_SEED).sample(eligible, n)


def check_question_ids(tasks, train_ids, dev_ids, heldout_ids):
    ids = [q for t in tasks for q in t["example_ids"]]
    qs = set(ids)
    require(len(ids) == len(qs) == 4 * len(tasks) and qs <= train_ids, "non-training or duplicate question")
    require(qs.isdisjoint(heldout_ids), "held-out question overlap")
    # The locked split was constructed entirely from this dev source. Testing against its
    # superset establishes zero overlap without opening the locked tasks or retrieval steps.
    require(qs.isdisjoint(dev_ids), "raw dev question overlap, including potential locked test questions")
    return {"train_example_ids": len(qs), "heldout_example_ids": len(heldout_ids),
            "raw_dev_ids_compared": len(dev_ids), "heldout_overlap": 0, "locked_test_overlap": 0,
            "locked_test_proof": "All selected raw example IDs are disjoint from the entire MuSiQue dev source."}


def select_assets():
    reference08 = formal.old.file_record(FROZEN_08)
    pre = cfg.read(cfg.OUT / "dagger_precheck.json")
    ten = cfg.read(FROZEN_10)
    require(pre["source_receipt"] == reference08 and ten["precheck"] == old.file_record(cfg.OUT / "dagger_precheck.json")
            and ten["status"] == "complete" and pre["train_file"] == old.file_record(formal.TRAIN)
            and pre["corpus_file"] == old.file_record(old.TRAIN_CORPUS), "historical training sources changed")
    old08, old10 = cfg.read(FROZEN_08)["task_ids"], ten["task_ids"]
    require(len(set(old08)) == len(set(old10)) == 400 and old10 == pre["task_ids"], "historical task IDs incomplete")
    fields = ("task", "split", "example_ids", "tier", "budget", "questions", "steps")
    population = [{k:t[k] for k in fields} for t in old.lines(formal.TRAIN)]
    require(len(population) == 4735, "training population changed")
    tasks = select_tasks(population, old08, old10)
    heldout = list(old.lines(formal.HELDOUT))
    require(len(heldout) == 249 and all(t["split"] == "heldout" for t in heldout), "invalid held-out source")
    raw = PROJECT / "data/raw/musique"
    raw_records = {}
    raw_ids = {}
    hashes = cfg.read(PROJECT / "eval/feasibility/run_config.json")["dataset_sha256"]
    for split in ("train", "dev"):
        path = raw / f"musique_ans_v1.0_{split}.jsonl"
        raw_records[split] = old.file_record(path)
        require(raw_records[split]["sha256"] == hashes[str(path.relative_to(PROJECT))], "raw dataset changed")
        raw_ids[split] = {q["id"] for q in old.lines(path)}
    overlap = check_question_ids(tasks, raw_ids["train"], raw_ids["dev"], {q for t in heldout for q in t["example_ids"]})
    ids = {t["task"] for t in tasks}
    facts = {**overlap, "raw_sources":raw_records, "task08_source":reference08, "task10_source":old.file_record(FROZEN_10),
             "task08_overlap":len(ids.intersection(old08)), "task10_overlap":len(ids.intersection(old10)),
             "train_file":pre["train_file"], "corpus_file":pre["corpus_file"]}
    require(facts["task08_overlap"] == facts["task10_overlap"] == 0, "training task overlap")
    corpus = list(old.lines(old.TRAIN_CORPUS))
    require(all(p["pid"] == i for i,p in enumerate(corpus)), "training corpus ordering changed")
    return tasks, corpus, facts


def rollout_stats(path, expected_n):
    adds, lengths = defaultdict(Counter), defaultdict(Counter)
    seen = set()
    for row in old.lines(path):
        key = (row["trajectory_id"], row["step"])
        require(key not in seen and row["split"] == "train", "invalid training state identity")
        seen.add(key)
        state = json.loads(row["reward_state_json"])
        action = score_completion_v2(row["greedy_output"], state, .5)
        require(action["keep_after"] == row["greedy_keep_after"] and action["invalid_reason"] ==
                row["greedy_invalid_reason"], "greedy parser mismatch in B")
        n_added = len(set(action["keep_after"]) - set(state["kept_ids"]))
        bucket = str(n_added) if n_added < 3 else "3+"
        for cell in ("pooled", f"{row['mode']}|{row['budget_tier']}"):
            adds[cell][bucket] += 1
            lengths[cell][row["prompt_tokens"]] += 1
    require(len(seen) == expected_n, "incorrect train-state denominator")
    return {cell: {"add_counts":{k:old.rate(counts[k], sum(counts.values())) for k in ("0","1","2","3+")},
                   "prompt_lengths":{**old.length_distribution(lengths[cell]),"min":min(lengths[cell]),
                                     "exceeds_8392":sum(n for k,n in lengths[cell].items() if k > 8392)}}
            for cell, counts in adds.items()}


def run():
    require(Path(sys.prefix).resolve() == cfg.ROOT / "envs/grpo", "use the existing GRPO environment")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
            "physical GPU 1 required")
    decision = start_model()
    tasks, corpus, provenance = select_assets()
    folder = RUN / "states"
    require(folder == folder.resolve() and not os.path.lexists(folder), "B output already exists")
    folder.mkdir(parents=True)
    args = Namespace(task_file=formal.TRAIN, n_tasks=len(tasks), seed=R2_SEED, base_model=old.DEFAULT_BASE,
                     adapter=Path(decision["path"]), source_adapter=None, model_label=f"SFT {decision['version']} {decision['strategy']}",
                     output_dir=folder, gpu=1)
    original_make = old.make_row

    def make_row(item, output, executed, source):
        row = original_make(item, output, executed, source)
        return {**row, "user_text":row["messages"][0]["content"], "state_id":f"{row['trajectory_id']}:{row['step']}"}

    def write(path, summary):
        require(path == folder / "state_checks.json" and summary["available_checks_pass"]
                and summary["source_adapter_sha256"] == decision["adapter_sha256"]
                and summary["n_trajectories"] == 2400 and len(summary["cells"]) == 6, "incomplete B checks")
        stats = rollout_stats(folder / "states.jsonl", summary["n_states"])
        for cell in summary["cells"]:
            summary["cells"][cell].update(stats[cell])
        summary["pooled"].update(stats["pooled"])
        summary["metric_definitions"]["add_counts"] = "All greedy steps, distinct executed additions; an invalid no-op adds zero."
        write_json_exclusive(path, {**summary, **provenance, "cuda_visible_devices":"1","status":"complete"})

    with patch.multiple(old, load_assets=lambda *args:(tasks,corpus), make_row=make_row,
                        check_gpu=formal.gpu_gate, write_json=write):
        old.build(args)


if __name__ == "__main__":
    ArgumentParser(description=__doc__).parse_args()
    run()

