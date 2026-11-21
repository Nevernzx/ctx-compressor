"""Task 06 state-generation policy. The frozen v1 module stays reproducible."""

import random

from src.build_traj import apply_action, oracle_action


def make_oracle_noise_policy(seed, noise_range=(1, 3)):
    """Draw the noise count after shuffling new non-evidence, as in v1."""
    low, high = noise_range
    if type(low) is not int or type(high) is not int or not 0 <= low <= high:
        raise ValueError("noise range must be a nonnegative ordered integer pair")
    rng = random.Random(seed)

    def policy(state):
        action = oracle_action(state)
        if state["mode"] != "reselect":
            return action
        keep = apply_action(state, action)
        used = sum(state["ntok"][i] for i in keep)
        gold = set(state["gold"])
        candidates = [i for i in state["new_ids"] if i not in gold]
        rng.shuffle(candidates)
        want = rng.randint(low, high)
        added = 0
        for i in candidates:
            if added == want:
                break
            size = state["ntok"][i]
            if used + size <= state["budget"]:
                action["add"].append(i)
                used += size
                added += 1
        action["add"].sort()
        return action

    return policy
