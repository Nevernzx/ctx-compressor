"""Register the round-two synchronous reward with the installed Swift plugin loader."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swift.plugin import orms
from src.grpo_r2_rewards import CtxStepRewardV2

orms['ctx_step_reward_v2'] = CtxStepRewardV2

