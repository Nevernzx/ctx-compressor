"""PLAN v2 evidence-recall reward shared by both compression modes."""

import json

from .build_traj import apply_action


def step_reward(completion, s, lam=0.3):
    """Return -1 for invalid output; give a budget bonus only at full recall."""
    try:
        act = json.loads(completion)
        if not isinstance(act, dict) or "add" not in act:
            return -1.0
        if not isinstance(act["add"], list) or not isinstance(act.get("drop", []), list):
            return -1.0
        ids = act["add"] + act.get("drop", [])
        if not all(type(i) is int for i in ids):
            return -1.0
    except (ValueError, TypeError):
        return -1.0
    keep = apply_action(s, act)
    if keep is None:
        return -1.0
    gold = set(s["gold"])
    recall = len(gold & set(keep)) / len(gold) if gold else 1.0
    used = sum(s["ntok"][i] for i in keep)
    bonus = lam * (1 - used / s["budget"]) if recall == 1.0 else 0.0
    return recall + bonus
