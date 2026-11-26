import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import grpo_r4_common as c
from scripts import grpo_r4_eval as evaluation
from scripts import grpo_r4_prepare as prep
from scripts import grpo_r4_reader as reader
from scripts import grpo_r4_training as training
from scripts import grpo_r4_swift_plugin as plugin
from src.grpo_reward_v2 import score_completion_v2
from src.grpo_reward_v3 import score_completion_v3
from src.reader_heldout_inputs import CELLS


def state():
    return {
        "mode": "reselect", "budget": 520, "kept_ids": [0], "new_ids": [1, 2, 3],
        "ntok": {0: 100, 1: 120, 2: 140, 3: 160}, "gold": [1],
        "pid_by_lid": {"0": 1000, "1": 32, "2": 43, "3": 1500},
        "task_gold_pids": [32, 43], "prompt_token_ids_tail": [4, 5, 6, 7],
    }


@pytest.mark.parametrize("raw,changes,reason,kept,nongold,value", [
    ('{"add":[1],"drop":[]}', {}, None, [0, 1], 1, -.27),
    ('{"add":[1,2],"drop":[0]}', {}, None, [1, 2], 0, -.13),
    ('{"add":[1,2,3],"drop":[0]}', {}, None, [1, 2, 3], 1, -.37),
    ("invalid", {}, "invalid_json", [0], 1, -2.21),
    ('{"add":[99],"drop":[]}', {}, "invalid_candidate_id", [0], 1, -2.21),
    ('{"add":[1],"drop":[]}', {"budget": 180}, "over_budget", [0], 1, -2.21),
    ('{"add":[1],"drop":[0]}', {"mode": "append"}, "append_drop", [0], 1, -2.21),
])
def test_v3_executes_the_same_action_then_charges_only_true_nongold(raw, changes, reason, kept, nongold, value):
    inputs = {**state(), **changes}
    v2 = score_completion_v2(raw, inputs, .5)
    v3 = prep.score_and_check(raw, inputs, v2["reward"])
    assert v3["invalid_reason"] == reason
    assert v3["keep_after"] == kept
    assert v3["n_nongold_after"] == nongold
    assert v3["reward"] == pytest.approx(value, abs=1e-12)
    assert all(v3[k] == v2[k] for k in v2 if k != "reward")


def test_v2_archive_reward_mismatch_stops():
    with pytest.raises(ValueError, match="Archived reward v2 mismatch"):
        prep.score_and_check('{"add":[1]}', state(), 0)


def test_task_gold_uses_future_steps_and_the_corpus_id_mapping():
    raw_state = state()
    mapping = raw_state.pop("pid_by_lid")
    raw_state.pop("task_gold_pids")
    row = {"step": 0, "reward_state_json": json.dumps(raw_state), "pid_by_lid_json": json.dumps(mapping)}
    joined = prep.state_with_gold(row, [32, 43, 43])
    assert joined["task_gold_pids"] == [32, 43]
    assert joined["pid_by_lid"][2] == 43
    assert score_completion_v3('{"add":[2],"drop":[0]}', joined)["n_nongold_after"] == 0
    with pytest.raises(ValueError, match="gold sequence"):
        prep.state_with_gold(row, [500, 43])


def test_d_uses_balanced_quotas_and_same_rng():
    names = prep.groups.CELLS
    candidates = {cell: [(cell, j) for j in range(400)] for cell in names}
    chosen, counts = prep.select(candidates)
    assert chosen == prep.select(candidates)[0]
    assert len(set(chosen)) == 2000
    assert list(counts.values()) == [334, 334, 333, 333, 333, 333]
    candidates[names[0]] = []
    chosen, counts = prep.select(candidates)
    assert len(set(chosen)) == 2000
    assert list(counts.values()) == [0, 400, 400, 400, 400, 400]


def test_plugin_consumes_32_components_for_exact_optimizer_step(monkeypatch):
    monkeypatch.setattr(plugin, "decode_rollout", lambda *_: '{"add":[1]}')
    fn = plugin.CtxStepRewardV3(SimpleNamespace(backend_tokenizer=None))
    trainer = SimpleNamespace(global_step=0)
    for _ in range(4):
        assert fn(["ignored"] * 8, [json.dumps(state())] * 8, [[]] * 8, trainer) == [-.27] * 8
    trainer.global_step = 1
    step = plugin.take_components(trainer)
    assert step["reward_component_n"] == 32
    assert step["mean_n_nongold_after"] == step["mean_add_count"] == 1
    assert step["mean_used_per_1000"] == .22 and step["invalid_rate"] == 0
    with pytest.raises(ValueError):
        plugin.take_components(trainer)


def test_logged_v3_rewards_and_no_std_advantages(tmp_path):
    raw = ['{"add":[1]}', '{"add":[1],"drop":[0]}'] * 4
    scores = [score_completion_v3(x, state())["reward"] for x in raw]
    mean = sum(scores) / len(scores)
    user = "question fixture"
    prompt = training.audit.old.USER_START + user + training.audit.old.RESPONSE_SUFFIX
    batch = {"step": [1] * 32, "prompt": [prompt] * 32, "completion": [x + "<|im_end|>" for x in raw] * 4,
             "CtxStepRewardV3": scores * 4, "advantages": [x - mean for x in scores] * 4}
    file = tmp_path / "completions.jsonl"
    file.write_text(json.dumps(batch) + "\n")
    receipt, measurements = training.reward_recheck(file, {user: state()}, 1)
    assert receipt["mismatches"] == receipt["advantage_mismatches"] == 0 and receipt["n"] == 32
    assert measurements[1]["mean_n_nongold_after"] == .5
    batch["advantages"][0] -= .03
    file.write_text(json.dumps(batch) + "\n")
    with pytest.raises(ValueError, match="advantage mismatches"):
        training.reward_recheck(file, {user: state()}, 1)


def test_f_prefers_objective_then_loss_then_earliest():
    def candidate(step, score, loss):
        return {"checkpoint_step": step, "heldout_reward_v3": {"reward_units_1_over_2000": score, "n_steps": 1},
                "evidence_loss": {"count": loss}}
    rows = [candidate(125, -10, 1), candidate(500, -9, 4), candidate(375, -9, 4), candidate(250, -9, 5)]
    assert evaluation.select_checkpoint(rows)["checkpoint_step"] == 375


def test_g_order_and_refuse_reordered_batch():
    expected = [
        {"strategy": n, "mode": m, "tier": t, "question_index": q,
         "prompt_tokens": 1, "token_ids": [1], "config": f"{n}|{m}|{t}", "task": 0}
        for n in reader.NAMES for m, t in CELLS for q in range(4)
    ]
    assert list(reader.batches(iter(expected))) == [(0, expected)]
    expected[0], expected[24] = expected[24], expected[0]
    with pytest.raises(ValueError, match="batch order"):
        list(reader.batches(iter(expected)))


def test_refuse_overwrite_existing(tmp_path):
    p = tmp_path / "already.json"
    p.write_text("keep")
    with pytest.raises(ValueError, match="Existing or redirected"):
        c.expect_new(p)
    assert p.read_text() == "keep"

