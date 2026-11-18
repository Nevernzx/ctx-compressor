import json

import pytest

from src.reward import step_reward


@pytest.fixture
def state():
    return {
        "mode": "reselect", "budget": 100,
        "kept_ids": [0], "new_ids": [1, 2],
        "ntok": {0: 20, 1: 30, 2: 90}, "gold": [0, 1],
    }


@pytest.mark.parametrize("completion", [
    "not json", '{"add":', '{}', '{"drop":[]}',
    '{"add":["1"]}', '{"add":[1.0]}', '{"add":[true]}',
    '{"add":null}', '{"add":{}}', '[]', 'null',
    '{"add":[],"drop":[false]}', '{"add":[],"drop":"bad"}',
])
def test_invalid_json_or_schema_is_negative_one(state, completion):
    assert step_reward(completion, state) == -1.0


@pytest.mark.parametrize("action", [
    {"add": [99], "drop": []},
    {"add": [0], "drop": []},
    {"add": [], "drop": [1]},
    {"add": [], "drop": [99]},
])
def test_add_and_drop_membership(state, action):
    assert step_reward(json.dumps(action), state) == -1.0


def test_append_rejects_nonempty_drop(state):
    state["mode"] = "append"
    assert step_reward('{"add":[1],"drop":[0]}', state) == -1.0


def test_over_budget_is_negative_one(state):
    assert step_reward('{"add":[2],"drop":[]}', state) == -1.0


def test_incomplete_recall_gets_no_saving_bonus(state):
    for lam in (0, 0.3, 10):
        assert step_reward('{"add":[],"drop":[]}', state, lam) == 0.5


def test_full_recall_saving_bonus(state):
    assert step_reward('{"add":[1],"drop":[]}', state, 0.3) == pytest.approx(1.15)


def test_full_budget_has_zero_saving_bonus(state):
    state["budget"] = 50
    assert step_reward('{"add":[1]}', state, 0.3) == 1.0


def test_empty_gold_is_full_recall(state):
    state["gold"] = []
    assert step_reward('{"add":[],"drop":[0]}', state, 0.3) == pytest.approx(1.3)


def test_duplicate_add_ids_have_set_semantics(state):
    # PLAN explicitly uses a set: duplicate IDs are neither double charged nor rejected.
    assert step_reward('{"add":[1,1,1],"drop":[]}', state) == pytest.approx(1.15)


def test_missing_drop_defaults_to_empty(state):
    assert step_reward('{"add":[1]}', state) == pytest.approx(1.15)
