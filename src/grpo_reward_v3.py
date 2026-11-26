"""Task 18 reward on the frozen action executor with a task-wide non-gold charge."""
from src.grpo_reward_v2 import score_completion_v2

MU = 0.5
NONGOLD_PRICE = 0.16


def score_completion_v3(raw, state):
    """Price the executed retained set, including when the action is a no-op.

    The frozen parser uses local IDs. A rollout reward state supplies the
    mapping to corpus PIDs, the namespace of task_gold_pids.
    """
    scored = score_completion_v2(raw, state, MU)
    gold_pids = set(state["task_gold_pids"])
    kept_ids = scored["keep_after"]
    if "pid_by_lid" in state:
        pid_by_lid = {int(lid): pid for lid, pid in state["pid_by_lid"].items()}
        kept_ids = [pid_by_lid[lid] for lid in kept_ids]
    n_nongold_after = sum(pid not in gold_pids for pid in kept_ids)
    cost = NONGOLD_PRICE * n_nongold_after
    return {**scored, "n_nongold_after": n_nongold_after, "nongold_cost": cost,
            "reward": scored["reward"] - cost}

