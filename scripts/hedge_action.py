"""Execute the Task 09 confidence hedge with the frozen evaluation parser and reward."""
import json
import math

from src.grpo_reward import score_completion


TAU_GRID = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01)


def candidate_ids(state):
    """Distinct new local IDs, in prompt order, excluding previously retained IDs."""
    kept = set(state["kept_ids"])
    return list(dict.fromkeys(i for i in state["new_ids"] if i not in kept))


def ranked_confidence(candidates, scores):
    if set(candidates) != set(scores) or not all(math.isfinite(s) for s in scores.values()):
        raise ValueError("Missing, extra, or nonfinite candidate log scores")
    if not candidates:
        return [], None
    # Equal scores retain the order in the eval prompt.
    ranked = sorted(candidates, key=scores.__getitem__, reverse=True)
    maximum = scores[ranked[0]]
    return ranked, 1.0 / math.fsum(math.exp(scores[i] - maximum) for i in ranked)


def execute_hedge(state, greedy_raw, scores, tau):
    """Return executed raw text, evaluation, reward, and per-step hedge metadata."""
    if tau not in TAU_GRID or (scores is None and tau != 0):
        raise ValueError("Use a registered tau; positive tau requires a score for each candidate")
    state = {**state, "ntok": {int(k): v for k, v in state["ntok"].items()}}
    greedy = score_completion(greedy_raw, state, lam=0.3)
    ranked, p1 = ([], None) if scores is None else ranked_confidence(candidate_ids(state), scores)
    result = {**greedy, "raw_output": greedy_raw, "greedy_raw_output": greedy_raw, "hedged": False,
              "extra_id": None, "p1": p1, "extra_tokens": 0}
    if not greedy["action_valid"] or p1 is None or p1 >= tau:
        return result
    action = greedy["action"]
    extra = next((i for i in ranked if i not in action["add"]), None)
    if extra is None or greedy["used_after"] + state["ntok"][extra] > state["budget"]:
        return result
    raw = json.dumps({"add": [*action["add"], extra], "drop": action.get("drop", [])},
                     separators=(",", ":"))
    executed = score_completion(raw, state, lam=0.3)
    if (not executed["action_valid"] or extra not in executed["keep_after"]
            or executed["used_after"] != greedy["used_after"] + state["ntok"][extra]):
        raise ValueError("Hedged action failed the evaluation parser or token accounting")
    return {**result, **executed, "raw_output": raw, "hedged": True, "extra_id": extra,
            "extra_tokens": state["ntok"][extra]}

