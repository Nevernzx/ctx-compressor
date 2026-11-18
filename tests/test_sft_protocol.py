"""CPU-only protocol invariants independent of Swift/torch installation."""
from collections import Counter
from copy import deepcopy

import pytest

from src.build_traj import oracle_action, render_input, render_target
from src.sft_protocol import (ASSISTANT_END, EMPTY_THINK, choose_max_length,
                              encode_sft_record, encoding_counts, length_distribution,
                              make_messages, restore_state_keys, sample_flags)


def state():
    value = {"questions": ["Find the birthplace"], "mode": "reselect", "budget": 7,
             "kept_ids": [0], "new_ids": [1, 2],
             "ntok": {0: 3, 1: 4, 2: 6},
             "text": {0: "Zero", 1: "One", 2: "Two"},
             "pid_by_lid": {0: 100, 1: 101, 2: 102}, "gold": [1, 2]}
    value["sft_target"] = oracle_action(value)
    return value


def test_messages_match_fixed_renderers_and_oracle_metadata_never_enters_user():
    value = state()
    messages = make_messages(value)
    assert messages == [{"role": "user", "content": render_input(value)},
                        {"role": "assistant", "content": render_target(value)}]
    altered = deepcopy(value)
    altered.update(gold=[0], action={"drop": [2]}, keep_after=[2], recall=0.123,
                   answer="SECRET_ANSWER", sft_target={"add": [], "drop": []})
    assert make_messages(altered)[0] == messages[0]
    assert "SECRET_ANSWER" not in messages[0]["content"]
    assert make_messages(altered)[1] != messages[1]


def test_restored_json_keys_recover_visible_state_and_target():
    import json
    original = state()
    restored = restore_state_keys(json.loads(json.dumps(original)))
    assert restored == original
    assert make_messages(restored) == make_messages(original)


@pytest.mark.parametrize("labels", [[1, 2, 3], [-100, -100, -100], [-100, 7, 3], [-100, 2, -100]])
def test_invalid_or_leaking_loss_masks_are_rejected(labels):
    with pytest.raises(ValueError):
        encoding_counts([1, 2, 3], labels)


def test_mask_counts_partition_entire_sequence():
    assert encoding_counts([1, 2, 3, 4], [-100, -100, 3, 4]) == {
        "n_input_tokens": 2, "n_target_tokens": 2, "total_tokens": 4, "loss_tokens": 2}


def test_binding_means_oracle_cannot_keep_all_available_evidence():
    value = state()
    assert sample_flags(value) == {"target_add_empty": False, "target_drop_nonempty": True, "budget_binding": True}
    value["budget"] = 10
    value["sft_target"] = oracle_action(value)
    assert sample_flags(value)["budget_binding"] is False
    value["sft_target"] = {"add": [], "drop": []}
    with pytest.raises(ValueError, match="pure oracle"):
        sample_flags(value)


def test_histogram_percentiles_match_linear_interpolation_without_expansion():
    result = length_distribution(Counter({1: 1, 3: 2, 9: 1}))
    assert result["n"] == 4 and result["p50"] == 3 and result["max"] == 9
    assert result["p90"] == pytest.approx(7.2)
    assert result["p99"] == pytest.approx(8.82)
    assert result["mean"] == 4
    assert length_distribution(Counter())["n"] == 0


def test_length_alignment_does_not_clip_over_limit_samples():
    assert choose_max_length(16381) == 16384
    assert choose_max_length(16385) == 16392


class FakeTokenizer:
    def decode(self, ids, **kwargs):
        return EMPTY_THINK + '{"add":[],"drop":[]}' + ASSISTANT_END


class FakeTemplate:
    mode = "train"
    truncation_strategy = "raise"
    tokenizer = FakeTokenizer()

    def encode(self, value):
        value["messages"][0]["content"] = "mutated internally"
        return {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}


def test_encoding_preserves_input_messages_and_validates_assistant_only_text():
    messages = [{"role": "user", "content": "original"},
                {"role": "assistant", "content": '{"add":[],"drop":[]}'}]
    before = deepcopy(messages)
    assert encode_sft_record(FakeTemplate(), messages)["total_tokens"] == 3
    assert messages == before
    messages[1]["content"] = "different target"
    with pytest.raises(ValueError, match="assistant-only"):
        encode_sft_record(FakeTemplate(), messages)
