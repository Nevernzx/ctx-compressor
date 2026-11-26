import pytest

from src.grpo_reward import score_completion
from src.grpo_reward_v2 import score_completion_v2


@pytest.fixture
def state():
    return {"mode": "append", "kept_ids": [7], "new_ids": [8, 9, 2], "budget": 100,
            "ntok": {"7": 20, "8": 30, "9": 10, "2": 90}, "gold": [7, 8, 9]}


@pytest.mark.parametrize("raw,reason", [
    ("broken", "invalid_json"), ('{"add":[17]}', "invalid_candidate_id"),
    ('{"add":[],"drop":[7]}', "append_drop"), ('{"add":[2]}', "over_budget")])
def test_invalid_noop_retains_token_and_loss_charges(state, raw, reason):
    r = score_completion_v2(raw, state, 0.5)
    assert r["invalid_reason"] == reason and r["keep_after"] == [7]
    assert (r["invalid_penalty"], r["n_lost"], r["used_after"]) == (1, 2, 20)
    assert r["reward"] == pytest.approx(-3.01)


@pytest.mark.parametrize("raw,lost", [('{"add":[8,9]}',0),('{"add":[8]}',1),('{"add":[]}',2)])
@pytest.mark.parametrize("mu", [0,0.1,0.25,0.5,1.0])
def test_valid_parity_and_price(state, raw, lost, mu):
    original = score_completion(raw, state)
    r = score_completion_v2(raw, state, mu)
    assert r["keep_after"] == original["keep_after"] and r["used_after"] == original["used_after"]
    assert r["action_valid"] and r["n_lost"] == lost and r["invalid_penalty"] == 0
    assert r["reward"] == pytest.approx(-lost - mu * original["used_after"] / 1000)


def test_drop_old_gold_and_exclude_unavailable_gold(state):
    state.update(mode="reselect", gold=[7,8,9,12])
    raw = '{"add":[8,9],"drop":[7]}'
    r = score_completion_v2(raw, state, .25)
    assert r["keep_after"] == score_completion(raw,state)["keep_after"] == [8,9]
    assert (r["n_available_gold"], r["n_lost"]) == (3,1)
    assert r["token_cost"] == .01
    assert r["reward"] == -1.01


@pytest.mark.parametrize("mu", [float("nan"), float("inf"), -0.1])
def test_invalid_mu(state, mu):
    with pytest.raises(ValueError, match="mu"):
        score_completion_v2('{"add":[]}',state,mu)

