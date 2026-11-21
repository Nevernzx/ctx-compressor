"""CPU-only tests for formal GRPO telemetry and reward-log validation."""
import json
import math

import pytest

from scripts.check_formal_grpo_training import (
    USER_START, RESPONSE_SUFFIX, curve, dataset_states, reward_recheck, stored_checkpoints,
)
from scripts.run_formal_swift_grpo import check_logged_finite


def lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def state():
    return {"mode": "append", "budget": 10, "kept_ids": [], "new_ids": [2],
            "ntok": {"2": 5}, "gold": [2]}


def reward_batch(n=32):
    return {"step": ["1"] * n, "prompt": [USER_START + "u" + RESPONSE_SUFFIX] * n,
            "completion": ['{"add":[2],"drop":[]}<|im_end|>'] * n, "CtxStepReward": [1.15] * n}


def test_terminal_table_counted_in_full(tmp_path):
    batch = reward_batch()
    p = lines(tmp_path / "log", [batch, batch])
    result = reward_recheck(p, {"u": state()}, 1)
    assert result["n"] == 64 and result["extra_logged_completions"] == 32
    assert result["mismatches"] == 0


@pytest.mark.parametrize("change", ["misaligned", "nan", "short", "wrong_prompt", "wrong_reward"])
def test_corrupt_log_rejected(tmp_path, change):
    batch = reward_batch(31 if change == "short" else 32)
    if change == "misaligned":
        batch["completion"].append("")
    elif change == "nan":
        batch["CtxStepReward"][0] = float("nan")
    elif change == "wrong_prompt":
        batch["prompt"][0] = USER_START + "other" + RESPONSE_SUFFIX
    elif change == "wrong_reward":
        batch["CtxStepReward"][0] = 0
    with pytest.raises(ValueError):
        reward_recheck(lines(tmp_path / "log", [batch]), {"u": state()}, 1)


def test_ambiguous_reward_state_rejected(tmp_path):
    st = state()
    rows = [{"messages":[{"role":"user","content":"same"}], "reward_state_json":json.dumps(s)}
            for s in [st, {**st, "gold":[]}]]
    with pytest.raises(ValueError, match="ambiguous"):
        dataset_states(lines(tmp_path / "data", rows))


@pytest.mark.parametrize("key, val", [("loss", math.nan), ("grad_norm", math.inf), ("grad_norm", None)])
def test_finite_guard(key,val):
    logs = {"loss": 0., "grad_norm": 4., key: val}
    with pytest.raises(FloatingPointError, match=key):
        check_logged_finite(logs, 1)


def test_missing_optimizer_step(tmp_path):
    lines(tmp_path / "optimizer_steps.jsonl", [])
    lines(tmp_path / "logging.jsonl", [{"loss": 0.}])
    with pytest.raises(ValueError, match="optimizer step"):
        curve(tmp_path, 1)


def test_missing_periodic_adapter(tmp_path):
    (tmp_path / "checkpoint-250").mkdir()
    with pytest.raises(ValueError, match="saved checkpoint"):
        stored_checkpoints(tmp_path, 250)

