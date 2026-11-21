"""Reward grammar, branches and Swift decoded-display boundary regression tests."""

import json
import sys

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel

from src.grpo_reward import decode_rollout, rollout_rewards, score_completion
from src.reward import step_reward
from src.sft_protocol import EMPTY_THINK


@pytest.fixture
def state():
    return {"mode": "append", "budget": 100, "kept_ids": [7], "new_ids": [8, 2],
            "ntok": {"7": 20, "8": 30, "2": 90}, "gold": [7, 8]}


@pytest.mark.parametrize("raw, reason", [
    ('{"add":[8]} trailing', "invalid_json"),
    ('{"add":[99],"drop":[7]}', "invalid_candidate_id"),
    ('{"add":[2],"drop":[7]}', "append_drop"),
    ('{"add":[2]}', "over_budget"),
    (EMPTY_THINK + '{"add":[8]}', "invalid_json"),
    ('{"add":[8]}<|im_end|>', "invalid_json"),
])
def test_invalid_priority_noop_and_sentinel(state, raw, reason):
    result = score_completion(raw, state)
    assert (result["reward"], result["invalid_reason"], result["keep_after"]) == (-1.0, reason, [7])


@pytest.mark.parametrize("lam", [0.0, 0.2, 0.3, 0.5])
def test_legal_action_recall_and_budget_bonus(state, lam):
    full = score_completion(' \n {"add":[8,8]}\t', state, lam)
    assert full["invalid_reason"] is None and full["used_after"] == 50
    assert full["reward"] == 1.0 + lam * 0.5
    partial = score_completion('{"add":[]}', state, lam)
    assert partial["reward"] == partial["recall"] == 0.5


def test_empty_gold_full_recall(state):
    state["gold"] = []
    detail = score_completion('{"add":[]}', state)
    assert detail["recall"] == 1.0 and detail["reward"] == pytest.approx(1.24)


@pytest.mark.parametrize("old_valid", [
    b'{"add":[]}', bytearray(b'{"add":[]}'),
    '{"add":[99],"add":[]}', '{"add":[],"extra":{"a":1,"a":2}}',
    '{"add":[],"extra":NaN}', '{"add":[],"extra":Infinity}', '{"add":[],"extra":-Infinity}',
])
def test_legacy_permissive_json_is_rejected_by_eval_wrapper(state, old_valid):
    int_state = {**state, "ntok": {int(k): v for k, v in state["ntok"].items()}}
    assert step_reward(old_valid, int_state) == 0.5
    assert score_completion(old_valid, state)["invalid_reason"] == "invalid_json"


def test_excessive_json_depth_is_invalid(state):
    depth = sys.getrecursionlimit() + 50
    raw = '{"add":[],"extra":' + "[" * depth + "1" + "]" * depth + "}"
    with pytest.raises(RecursionError):
        step_reward(raw, state)
    assert score_completion(raw, state)["reward"] == -1.0


def test_transport_prefix_and_eos_are_distinct_from_generated_text(state):
    good = '{"add":[8]}'
    vocab = {s: i for i, s in enumerate(("p", good, EMPTY_THINK, "<|im_end|>", "junk"))}
    tokenizer = Tokenizer(WordLevel(vocab))
    tokenizer.decoder = Fuse()
    tokenizer.add_special_tokens(["<|im_end|>"])
    tail = [0, 0, 0, 0]
    packet = json.dumps({**state, "prompt_token_ids_tail": tail})
    # The display string is intentionally identical across rows. Only generated
    # tokens distinguish renderer-added think from a generated second think.
    tokens = [[1, 3], [2, 1, 3], [1, 4, 3]]
    assert [decode_rollout(tokenizer, tail, ids) for ids in tokens] == [
        good, EMPTY_THINK + good, good + "junk"]
    assert rollout_rewards(tokenizer, [EMPTY_THINK + good] * 3, [packet] * 3, tokens, 0.3) == [
        1.15, -1.0, -1.0]
    with pytest.raises(ValueError, match="aligned"):
        rollout_rewards(tokenizer, [good], [], [], 0.3)


def test_lambda_must_keep_invalid_sentinel_unique(state):
    for value in (-2, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="lambda"):
            score_completion('{"add":[]}', state, value)
