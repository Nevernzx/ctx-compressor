import json

from src.build_traj import make_oracle_noise_policy as legacy_policy
from src.build_traj import rollout
from src.oracle_noise_v2 import make_oracle_noise_policy


def run(policy, mode="reselect"):
    corpus = [{"ntok": 1, "text": str(i)} for i in range(48)]
    steps = [{"gold": start, "new": list(range(start, start + 8))}
             for start in range(0, 48, 8)]
    return rollout(["q"], steps, corpus, 40, mode, policy=policy)


def test_default_policy_matches_legacy_serialized_closed_loop_across_steps():
    for seed in range(10):
        expected = json.dumps(run(legacy_policy(seed)), separators=(",", ":"))
        actual = json.dumps(run(make_oracle_noise_policy(seed)), separators=(",", ":"))
        assert expected.encode() == actual.encode()


def test_new_range_draws_zero_one_and_two_keeps_target_clean_and_leaves_append_pure():
    counts = set()
    for seed in range(5):
        for state in run(make_oracle_noise_policy(seed, (0, 2))):
            counts.add(len(set(state["action"]["add"]) - set(state["sft_target"]["add"])))
        for state in run(make_oracle_noise_policy(seed, (0, 2)), "append"):
            assert state["action"] == state["sft_target"]
    assert counts == {0, 1, 2}
