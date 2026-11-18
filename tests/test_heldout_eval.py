"""Task 05 closed-loop protocol and exact Task 02 metric reuse checks."""
from copy import deepcopy
import json

import pytest

from src.build_traj import oracle_action, rollout
from src.feasibility import aggregate_tasks, summarize_task
from src.heldout_eval import (HeldoutReplay, apply_model_output, load_heldout_assets,
                              parse_action, select_checkpoint)
from src.policies import latest_policy, no_compress_policy


@pytest.fixture
def corpus():
    return [{"pid": i, "text": f"paragraph {i}", "ntok": length}
            for i, length in enumerate([2, 3, 4, 2, 5, 1])]


@pytest.fixture
def tasks():
    return [{"task": i, "split": "heldout", "questions": ["a", "b", "c", "d"],
             "example_ids": [f"{i}_{j}" for j in range(4)],
             "steps": [{"new": new, "gold": gold, "question_index": q, "hop_index": 0}
                       for q, (new, gold) in enumerate([([0, 1], 0), ([2, 3], 2),
                                                        ([0, 4], 4), ([3, 5], 5)])]}
            for i in range(2)]


@pytest.fixture
def state():
    return {"kept_ids": [0], "new_ids": [1, 2], "ntok": {0: 2, 1: 3, 2: 5},
            "mode": "reselect", "budget": 5, "gold": [0, 1]}


@pytest.mark.parametrize("raw", ["", "```json\n{\"add\":[]}\n```", '{"add":[]} trailing',
    '[]', '{}', '{"add":true}', '{"add":[true]}', '{"add":[1.0]}', '{"add":[],"drop":{}}',
    '{"add":[],"add":[1]}', '{"add":[],"other":NaN}', '{"add":[],"other":Infinity}'])
def test_strict_json_rejects_invalid_schema_without_repair(raw, state):
    result = apply_model_output(state, raw)
    assert result["invalid_reason"] == "invalid_json"
    assert result["keep_after"] == state["kept_ids"]


def test_existing_action_grammar_is_preserved(state):
    raw = ' {"add":[1,1], "annotation":"ignored by existing executor"} '
    assert parse_action(raw)["add"] == [1, 1]
    result = apply_model_output(state, raw)
    assert result["action_valid"] and result["keep_after"] == [0, 1]


@pytest.mark.parametrize("mode,raw,reason", [
    ("append", '{"add":[99],"drop":[0]}', "invalid_candidate_id"),
    ("append", '{"add":[2],"drop":[0]}', "append_drop"),
    ("reselect", '{"add":[2],"drop":[]}', "over_budget"),
    ("reselect", '{"add":[0],"drop":[]}', "invalid_candidate_id"),
    ("reselect", '{"add":[],"drop":[1]}', "invalid_candidate_id"),
])
def test_invalid_categories_are_exclusive_and_noop(state, mode, raw, reason):
    state["mode"] = mode
    result = apply_model_output(state, raw)
    assert result["invalid_reason"] == reason
    assert result["keep_after"] == [0] and result["used_after"] == 2


@pytest.mark.parametrize("mode", ["append", "reselect"])
@pytest.mark.parametrize("policy", [oracle_action, latest_policy()])
def test_replay_and_aggregation_match_original_task02(tasks, corpus, mode, policy):
    budget = 7
    replay = HeldoutReplay(tasks, corpus, "policy", budgets={"test": budget}, modes=(mode,))
    records = []
    while not replay.completed():
        batch = replay.next_batch()
        assert replay.next_batch() == batch
        assert len({item["step"] for item in batch}) == 1
        records.extend(replay.advance_batch([{"raw_output": json.dumps(policy(item["state"]))}
                                             for item in batch]))
    expected = []
    for task in tasks:
        samples = rollout(task["questions"], task["steps"], corpus, budget, mode, policy=policy)
        expected.append(summarize_task(samples, task["steps"], [{step["gold"]} for step in task["steps"]],
                                       {i: p["ntok"] for i, p in enumerate(corpus)}, budget))
        actual = [record for record in records if record["task"] == task["task"]]
        assert [record["state"]["keep_after"] for record in actual] == [s["keep_after"] for s in samples]
        assert all("text" not in record["state"] for record in actual)
    result = replay.aggregate()["configs"][f"policy|{mode}|test"]["metrics"]
    original = aggregate_tasks(expected)
    assert {key: result[key] for key in original} == original
    assert result["invalid_outputs"]["any_invalid"] == {"count": 0, "n": 8, "rate": 0.0}


def test_invalid_step_is_carried_forward_and_final_loss_not_available_recall(tasks, corpus):
    replay = HeldoutReplay(tasks[:1], corpus, "model", budgets={"test": 20}, modes=("append",))
    records = []
    while not replay.completed():
        item = replay.next_batch()[0]
        raw = 'bad JSON' if item["step"] == 0 else json.dumps(oracle_action(item["state"]))
        records.extend(replay.advance_batch([{"raw_output": raw, "generated_tokens": 256,
                                               "finish_reason": "length", "prompt_sha256": "abc"}]))
    # Gold 0 reappears at step 2, so create a permanent omission at step 1 instead.
    assert records[1]["state"]["kept_ids"] == []
    assert records[0]["invalid_reason"] == "invalid_json"
    result = replay.aggregate()["combined"]
    assert result["output_limit"] == {"count": 4, "n": 4, "rate": 1.0}
    assert result["invalid_outputs"]["invalid_json"]["count"] == 1
    assert records[0]["generation_metadata"]["prompt_sha256"] == "abc"
    changed = deepcopy(tasks[:1])
    changed[0]["steps"][2]["new"] = [1, 4]
    replay = HeldoutReplay(changed, corpus, "model", budgets={"test": 20}, modes=("append",))
    while not replay.completed():
        item = replay.next_batch()[0]
        raw = 'bad JSON' if item["step"] == 0 else json.dumps(oracle_action(item["state"]))
        replay.advance_batch([{"raw_output": raw}])
    result = replay.aggregate()["combined"]
    assert result["step_recall"]["micro"]["mean"] == 0.75
    assert result["final_recall"]["mean"] == 0.75
    assert result["evidence_loss"]["rate"] == 1.0
    assert result["evidence_loss"]["by_mechanism"]["never_kept"]["lost_evidence_count"] == 1


def test_nonbinding_reference_keeps_all_and_records_requested_budget(tasks, corpus):
    replay = HeldoutReplay(tasks, corpus, "no_compress", budgets={"tiny": 1})
    while not replay.completed():
        batch = replay.next_batch()
        assert all(item["budget"] == 1 and item["effective_budget"] > 1 for item in batch)
        replay.advance_batch([{"raw_output": json.dumps(no_compress_policy(item["state"]))} for item in batch])
    result = replay.aggregate()["combined"]
    assert result["final_recall"]["mean"] == 1.0
    assert result["requested_budget_occupancy"]["mean"] == 17.0
    assert result["action_validity"]["over_budget_count"] == 0


def test_batch_validation_is_atomic_and_incomplete_aggregation_rejected(tasks, corpus):
    replay = HeldoutReplay(tasks, corpus, "model", budgets={"test": 7}, modes=("append",))
    with pytest.raises(RuntimeError):
        replay.aggregate()
    with pytest.raises(RuntimeError):
        replay.advance_batch([])
    batch = replay.next_batch()
    with pytest.raises(ValueError):
        replay.advance_batch([{"raw_output": '{}'}])
    with pytest.raises(ValueError):
        replay.advance_batch([{"raw_output": '{}'}, {"raw_output": '{}', "generated_tokens": -1}])
    assert replay.next_batch() == batch
    assert all(t["step"] == 0 for t in replay.trajectories)


def test_assets_discard_teacher_states_and_reject_wrong_split(tmp_path, tasks, corpus):
    (tmp_path / "corpus_train.jsonl").write_text("\n".join(json.dumps(row) for row in corpus))
    for task in tasks:
        task["rollouts"] = {"reselect": ["must not use teacher input"]}
    path = tmp_path / "heldout_tasks.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in tasks))
    loaded, loaded_corpus = load_heldout_assets(tmp_path, expected_tasks=2)
    assert all("rollouts" not in task for task in loaded)
    assert loaded_corpus == corpus
    tasks[0]["split"] = "dev"
    path.write_text("\n".join(json.dumps(row) for row in tasks))
    with pytest.raises(ValueError, match="heldout train"):
        load_heldout_assets(tmp_path, expected_tasks=2)


def _checkpoint(step, loss, recall):
    return {"checkpoint_step": step, "settings": {"n_tasks": 249},
            "configs": {str(i): {"metrics": {"n_tasks": 249}} for i in range(6)},
            "combined": {"n_tasks": 1494, "evidence_loss": {"count": loss},
                         "final_recall": {"mean": recall}}}


def test_checkpoint_selection_uses_complete_grid_loss_recall_earliest_step():
    first = _checkpoint(500, 20, 0.98)
    later = _checkpoint(1000, 20, 0.98)
    bad_loss = _checkpoint(1500, 21, 0.999)
    bad_recall = _checkpoint(2000, 20, 0.97)
    assert select_checkpoint([later, bad_loss, bad_recall, first]) is first
    incomplete = _checkpoint(2500, 0, 1.0)
    incomplete["configs"]["0"]["metrics"]["n_tasks"] = 248
    with pytest.raises(ValueError, match="249 tasks"):
        select_checkpoint([incomplete])
