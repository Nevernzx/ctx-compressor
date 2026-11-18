#!/usr/bin/env python3
"""Independently review saved acceptance records against raw dev; read-only.

Execute on 5880. This does not import the production trajectory/audit helpers,
download data, build a new retrieval index, write artifacts, or run a model.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from tokenizers import Tokenizer


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=project)
    args = parser.parse_args()
    base = args.project
    with (base / "data/raw/musique/musique_ans_v1.0_dev.jsonl").open(encoding="utf-8") as stream:
        dev = [json.loads(line) for line in stream if line.strip()]
    by_id = {example["id"]: example for example in dev}
    assert len(by_id) == len(dev), "duplicate dev question IDs"
    keys, corpus = {}, []
    for example in dev:
        for paragraph in example["paragraphs"]:
            key = (paragraph["title"], paragraph["paragraph_text"])
            if key not in keys:
                keys[key] = len(corpus)
                corpus.append(paragraph)
    tokenizer = Tokenizer.from_file(str(base / "data/raw/qwen3_tokenizer/tokenizer.json"))
    with (base / "data/traj/acceptance.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]

    seen, previous, gold_seen, token_counts = {}, {}, {}, {}
    task_questions, task_hops = defaultdict(list), defaultdict(set)
    mode_counts = defaultdict(int)
    for record in records:
        retrieval, state = record["retrieval"], record["state"]
        trajectory = (record["task"], state["mode"])
        example = by_id[retrieval["example_id"]]
        hop_index = retrieval["hop_index"]
        hop = example["question_decomposition"][hop_index]

        def replace_reference(match):
            reference = int(match.group(1))
            assert 1 <= reference <= hop_index, "not a previous-hop reference"
            return example["question_decomposition"][reference - 1]["answer"]

        query = re.sub(r"#(\d+)", replace_reference, hop["question"])
        assert query == retrieval["query"] and not re.search(r"#\d", query)
        assert hop["question"] == retrieval["original_query"]
        support = next(p for p in example["paragraphs"] if p["idx"] == hop["paragraph_support_idx"])
        assert keys[(support["title"], support["paragraph_text"]) ] == retrieval["gold"]

        mapping = {int(key): value for key, value in state["pid_by_lid"].items()}
        prior_keep = previous.get(trajectory, [])
        assert state["kept_ids"] == prior_keep
        assert len(retrieval["new"]) == 20 and len(set(retrieval["new"])) == 20
        assert retrieval["gold"] in retrieval["new"]
        prior_ids = seen.setdefault(trajectory, {})
        assert len(set(mapping.values())) == len(mapping)
        assert set(mapping) == set(state["kept_ids"] + state["new_ids"])
        for local_id, paragraph_id in mapping.items():
            assert prior_ids.get(paragraph_id, local_id) == local_id
            prior_ids[paragraph_id] = local_id
            paragraph = corpus[paragraph_id]
            text = paragraph["title"] + "\n" + paragraph["paragraph_text"]
            assert state["text"][str(local_id)] == text
            if paragraph_id not in token_counts:
                token_counts[paragraph_id] = len(tokenizer.encode(text, add_special_tokens=False).ids)
            assert state["ntok"][str(local_id)] == token_counts[paragraph_id]

        kept_paragraphs = [mapping[i] for i in state["kept_ids"]]
        assert [mapping[i] for i in state["new_ids"]] == [p for p in retrieval["new"] if p not in kept_paragraphs]
        available_gold = gold_seen.setdefault(trajectory, set())
        available_gold.add(retrieval["gold"])
        assert set(state["gold"]) == {i for i, pid in mapping.items() if pid in available_gold}

        action = state["sft_target"]
        assert all(type(i) is int for i in action["add"] + action["drop"])
        assert set(action["add"]) <= set(state["new_ids"])
        assert set(action["drop"]) <= set(state["kept_ids"])
        expected_keep = [i for i in state["kept_ids"] if i not in action["drop"]]
        expected_keep += [i for i in state["new_ids"] if i in action["add"]]
        assert state["keep_after"] == expected_keep
        used = sum(state["ntok"][str(i)] for i in expected_keep)
        assert used == state["used_after"] and used <= state["budget"] == 3000
        gold = set(state["gold"])
        recall = len(gold & set(expected_keep)) / len(gold) if gold else 1.0
        assert recall == state["recall"]
        if state["mode"] == "append":
            assert not action["drop"]
            assert expected_keep[:len(prior_keep)] == prior_keep
        previous[trajectory] = expected_keep
        if hop_index == 0:
            task_questions[trajectory].append(example["id"])
        task_hops[record["task"]].add(len(example["question_decomposition"]))
        mode_counts[state["mode"]] += 1

    assert len(task_questions) == 6
    all_questions = []
    for task in (1, 2, 3):
        ids = task_questions[(task, "reselect")]
        assert len(ids) == 4 and ids == task_questions[(task, "append")]
        all_questions.extend(ids)
    assert len(set(all_questions)) == 12
    expected_records = 2 * sum(len(by_id[i]["question_decomposition"]) for i in all_questions)
    assert len(records) == expected_records == 58
    print(json.dumps({
        "result": "PASS", "records_independently_checked": len(records),
        "unique_candidate_fragments_retokenized": len(token_counts),
        "dev_corpus_size": len(corpus), "distinct_questions": len(set(all_questions)),
        "records_per_mode": dict(mode_counts),
        "task_hop_coverage": {str(task): sorted(hops) for task, hops in task_hops.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
