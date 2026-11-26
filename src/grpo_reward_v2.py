"""Task 10 evidence-loss reward on the same parsed action used by held-out evaluation."""
import math

from src.heldout_eval import apply_model_output


def score_completion_v2(raw, state, mu):
    """Charge for every missing available gold, all retained tokens and an invalid action."""
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("mu must be finite and nonnegative")
    ntok = {int(key): value for key, value in state["ntok"].items()}
    visible = set(state["kept_ids"]) | set(state["new_ids"])
    available_gold = visible.intersection(state["gold"])
    executed = apply_model_output({**state, "ntok": ntok, "gold": sorted(available_gold)}, raw)
    n_lost = len(available_gold.difference(executed["keep_after"]))
    penalty = int(not executed["action_valid"])
    token_cost = mu * executed["used_after"] / 1000
    return {**executed, "n_available_gold": len(available_gold), "n_lost": n_lost,
            "invalid_penalty": penalty, "token_cost": token_cost,
            "mu": mu, "reward": -n_lost - token_cost - penalty}

