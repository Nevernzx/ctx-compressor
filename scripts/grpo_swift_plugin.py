"""ms-swift 3.12.6 external reward plugin for closed-loop compression."""

import os
from pathlib import Path
import sys

from swift.plugin import ORM, orms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.grpo_reward import rollout_rewards


class CtxStepReward(ORM):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer.backend_tokenizer
        self.lam = float(os.environ.get("CTX_GRPO_LAMBDA", "0.3"))

    def __call__(self, completions, reward_state_json, response_token_ids, **kwargs):
        return rollout_rewards(
            self.tokenizer, completions, reward_state_json, response_token_ids, self.lam)


orms["ctx_step_reward"] = CtxStepReward
