"""Swift registration and optimizer-step telemetry for Task 18's reward."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.grpo_r2_rewards import COMPONENTS as V2_COMPONENTS, RewardComponents as V2Components
from src.grpo_reward import decode_rollout
from src.grpo_reward_v3 import score_completion_v3

PLUGIN_PATH = "scripts/grpo_r4_swift_plugin.py"
ORM_NAME = "ctx_step_reward_v3"
COMPONENT_ATTR = "_ctx_grpo_r4_components"
COMPONENTS = (*V2_COMPONENTS, "mean_n_nongold_after")


class RewardComponents(V2Components):
    def __init__(self):
        super().__init__()
        self.nongold = 0

    def observe(self, score):
        super().observe(score)
        self.nongold += score["n_nongold_after"]

    def result(self):
        values = super().result()
        return {**values, "mean_n_nongold_after": self.nongold / self.n}


def take_components(trainer_state):
    pending = getattr(trainer_state, COMPONENT_ATTR, None)
    if pending is None or pending[0] != trainer_state.global_step or pending[1].n != 32:
        raise ValueError(f"Missing or misaligned reward components at step {trainer_state.global_step}")
    delattr(trainer_state, COMPONENT_ATTR)
    return pending[1].result()


class CtxStepRewardV3:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer.backend_tokenizer

    def __call__(self, completions, reward_state_json, response_token_ids, trainer_state=None, **kwargs):
        scores = []
        for _, packed, tokens in zip(completions, reward_state_json, response_token_ids, strict=True):
            state = json.loads(packed)
            raw = decode_rollout(self.tokenizer, state["prompt_token_ids_tail"], tokens)
            scores.append(score_completion_v3(raw, state))
        if trainer_state is not None:
            step = trainer_state.global_step + 1
            pending = getattr(trainer_state, COMPONENT_ATTR, None)
            if pending is None:
                pending = step, RewardComponents()
                setattr(trainer_state, COMPONENT_ATTR, pending)
            if pending[0] != step:
                raise ValueError("Unconsumed reward components from a prior optimizer step")
            for score in scores:
                pending[1].observe(score)
        return [score["reward"] for score in scores]


# Swift imports the plugin by file in the worker; CPU audit can import the
# package without loading Swift or any CUDA runtime.
if __name__ != "scripts.grpo_r4_swift_plugin":
    from swift.plugin import orms
    orms[ORM_NAME] = CtxStepRewardV3

