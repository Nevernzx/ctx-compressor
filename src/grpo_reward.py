"""Step rewards on the exact continuation text used in Task 05 evaluation."""

import json
import math

from src.heldout_eval import apply_model_output
from src.reward import step_reward


STATE_FIELDS = ("mode", "budget", "kept_ids", "new_ids", "ntok", "gold")


def score_completion(raw_output, state, lam=0.3):
    """Classify with eval first; score valid actions using the frozen PLAN formula."""
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("reward lambda must be finite and nonnegative")
    state = {**state, "ntok": {int(k): v for k, v in state["ntok"].items()}}
    executed = apply_model_output(state, raw_output)
    reward = step_reward(raw_output, state, lam) if executed["action_valid"] else -1.0
    return {**executed, "reward": reward}


def decode_rollout(tokenizer, prompt_tail, generated_ids):
    """Match vLLM 0.11's Qwen fast DecodeStream with Task 05 sampling defaults.

    Qwen3's fixed non-thinking generation suffix ends in complete UTF-8. vLLM
    primes DecodeStream with the final four tokens in this case. Its two EOS
    tokens are special tokens, so skip_special_tokens removes either one.
    The stream also preserves its incremental handling of incomplete UTF-8.
    """
    from tokenizers.decoders import DecodeStream

    if len(prompt_tail) < 4 or "�" in tokenizer.decode(prompt_tail[-4:]):
        raise ValueError("rollout prompt must have the verified Qwen3 generation suffix")
    stream = DecodeStream(skip_special_tokens=True)
    for token in prompt_tail[-4:]:
        stream.step(tokenizer, token)
    pieces = []
    for token in generated_ids:
        try:
            piece = stream.step(tokenizer, token)
        except Exception as exc:
            # Same recovery as vLLM's FastIncrementalDetokenizer._protected_step.
            if not str(exc).startswith("Invalid prefix encountered"):
                raise
            stream = DecodeStream(skip_special_tokens=True)
            piece = stream.step(tokenizer, token)
        if piece:
            pieces.append(piece)
    return "".join(pieces)


def rollout_rewards(tokenizer, completions, reward_state_json, response_token_ids, lam):
    """Swift's template prepends a display prefix; raw token IDs omit it."""
    if not (len(completions) == len(reward_state_json) == len(response_token_ids)):
        raise ValueError("reward inputs and generated tokens must be aligned")
    rewards = []
    for packed, tokens in zip(reward_state_json, response_token_ids):
        state = json.loads(packed)
        raw = decode_rollout(tokenizer, state["prompt_token_ids_tail"], tokens)
        rewards.append(score_completion(raw, state, lam)["reward"])
    return rewards
