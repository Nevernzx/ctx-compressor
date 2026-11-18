"""Hand-calculated TASK_02 metrics, independent of the real experiment data."""
from copy import deepcopy

import pytest

from src.feasibility import (
    aggregate_tasks,
    contains_answer,
    distribution,
    normalize_answer,
    summarize_task,
)


def example_task():
    # The same gold pid 20 occurs at hop 2 of both questions. Pid 10 is
    # retained then dropped; pid 30 is never retained. PLAN step recall
    # becomes 1 at the end even though task-level recall is only 1/3.
    steps = [
        {"question_index": 0, "hop_index": 0, "gold": 10},
        {"question_index": 0, "hop_index": 1, "gold": 20},
        {"question_index": 1, "hop_index": 0, "gold": 30},
        {"question_index": 1, "hop_index": 1, "gold": 20},
    ]
    samples = [
        {"pid_by_lid": {0: 10}, "keep_after": [0], "recall": 1., "action_valid": True},
        {"pid_by_lid": {0: 10, 1: 20}, "keep_after": [0, 1], "recall": 1., "action_valid": True},
        {"pid_by_lid": {1: 20, 2: 30}, "keep_after": [1], "recall": 1 / 3, "action_valid": True},
        {"pid_by_lid": {1: 20}, "keep_after": [1], "recall": 1., "action_valid": True},
    ]
    return samples, steps, [{10, 20}, {20, 30}], {10: 10, 20: 20, 30: 30}, 40


def test_distribution_uses_linear_percentiles_and_n():
    out = distribution([0, 10, 20, 30])
    assert out["n"] == 4
    assert {key: value for key, value in out.items() if key != "n"} == pytest.approx(
        {"mean": 15., "p10": 3., "p50": 15., "p90": 27., "p99": 29.7, "max": 30.}
    )
    assert distribution([7])["p99"] == pytest.approx(7.)


def test_empty_distribution_has_explicit_missing_values():
    assert distribution([]) == {"n": 0, "mean": None, "p10": None,
                                "p50": None, "p90": None, "p99": None, "max": None}
    with pytest.raises(ValueError):
        distribution([1, float("nan")])


def test_squad_normalization_matches_punctuation_deletion_order():
    assert normalize_answer("  The U.S.A., an Apple! A cat. ") == "usa apple cat"
    # SQuAD deletes ASCII punctuation, it does not replace hyphens by spaces.
    assert normalize_answer("New-York") == "newyork"
    assert normalize_answer("a an the") == ""


@pytest.mark.parametrize(("text", "answer", "expected"), [
    ("The New York, USA.", "new york", True),
    ("The New Yorker", "New York", False),
    ("earth and artwork", "art", False),
    ("art, science", "Art", True),
    ("UK-based", "UK", False),
    ("in the UK.", "UK", True),
    ("19901 and 1990", "1990", True),
    ("19901", "1990", False),
    ("anything", "the", False),
    ("anything", "!!!", False),
    ("anything", "", False),
])
def test_answer_matching_has_word_boundaries(text, answer, expected):
    assert contains_answer(text, answer) is expected


def test_final_all_gold_recall_differs_from_plan_step_recall():
    out = summarize_task(*example_task())
    assert out["step_recall_mean"] == pytest.approx(5 / 6)
    assert out["final_recall"] == pytest.approx(1 / 3)
    assert out["all_gold_count"] == 3
    assert out["retained_gold_count"] == 1
    assert out["evidence_loss"] is True
    assert out["complete_question_count"] == 0
    assert out["final_used"] == 20
    assert out["final_occupancy"] == .5
    assert out["cost_plain"] == 80
    # 12.5 + (0.05*10 + 1.25*20) + 1.25*20 + 0.05*20.
    assert out["cost_cached"] == 64
    assert out["invalid_action_count"] == out["over_budget_count"] == 0


def test_loss_counts_actual_eligible_tasks_and_gold_occurrences():
    task = summarize_task(*example_task())
    out = aggregate_tasks([task])
    loss = out["evidence_loss"]
    assert (loss["n"], loss["count"], loss["rate"]) == (1, 1, 1)
    assert loss["by_hop"]["1"] == {"n": 1, "count": 1, "rate": 1,
                                     "evidence_n": 2, "lost_evidence_count": 2}
    # Pid 20 is shared by two hops: two occurrences, neither lost.
    assert loss["by_hop"]["2"]["evidence_n"] == 2
    assert loss["by_hop"]["2"]["lost_evidence_count"] == 0
    # This synthetic task has no hop 3/4: denominator zero, not all tasks.
    assert loss["by_hop"]["3"] == {"n": 0, "count": 0, "rate": None,
                                     "evidence_n": 0, "lost_evidence_count": 0}
    assert loss["by_question_position"]["1"]["lost_evidence_count"] == 1
    assert loss["by_question_position"]["2"]["lost_evidence_count"] == 1
    assert loss["by_question_position"]["3"]["n"] == 0
    assert task["never_kept_pids"] == [30]
    assert task["dropped_pids"] == [10]
    for mechanism in ("never_kept", "dropped"):
        assert loss["by_mechanism"][mechanism] == {
            "n": 1, "count": 1, "rate": 1., "evidence_n": 3,
            "lost_evidence_count": 1,
        }


def test_retrieved_dropped_evidence_is_not_missing_if_kept_again():
    args = list(example_task())
    args[0][-1] = {"pid_by_lid": {0: 10, 1: 20, 2: 30},
                   "keep_after": [1, 0, 2], "recall": 1, "action_valid": True}
    args[-1] = 60
    out = summarize_task(*args)
    assert out["evidence_loss"] is False
    assert out["never_kept_pids"] == out["dropped_pids"] == []
    assert out["complete_question_count"] == 2
    assert out["final_recall"] == 1


def test_shared_missing_pid_counts_every_hop_but_once_in_task_gold():
    args = list(example_task())
    args[0][-1] = {"pid_by_lid": {0: 10, 1: 20},
                   "keep_after": [0], "recall": .5, "action_valid": True}
    task = summarize_task(*args)
    assert task["all_gold_count"] == 3
    assert task["missing_pids"] == [20, 30]
    # Gold pid 20 occurs in hop 2 of both questions, so both are missing.
    assert task["loss_by_hop"]["2"]["lost_evidence_count"] == 2
    assert task["loss_by_hop"]["2"]["count"] == 1
    # Mechanism accounting deduplicates pid 20 inside the task.
    assert task["loss_by_mechanism"]["dropped"]["lost_evidence_count"] == 1


def test_cost_uses_corpus_pids_from_every_state_not_local_numbers():
    args = list(example_task())
    # Renumbering the inputs cannot change corpus-prefix cache accounting.
    args[0][2]["pid_by_lid"] = {0: 20, 1: 30}
    args[0][2]["keep_after"] = [0]
    args[0][3]["pid_by_lid"] = {99: 20}
    args[0][3]["keep_after"] = [99]
    out = summarize_task(*args)
    assert out["cost_plain"] == 80
    assert out["cost_cached"] == 64


def test_cost_resets_between_tasks_and_ratios_use_task_means():
    one = summarize_task(*example_task())
    baseline = aggregate_tasks([one, one])
    out = aggregate_tasks([one, one], baseline=baseline)
    assert out["cost"]["plain"] == {"n": 2, "mean": 80., "relative_to_no_compress": 1.}
    assert out["cost"]["cached"] == {"n": 2, "mean": 64., "relative_to_no_compress": 1.}
    assert baseline["cost"]["plain"]["relative_to_no_compress"] is None
    with pytest.raises(ValueError, match="same number"):
        aggregate_tasks([one], baseline=baseline)


def test_step_micro_and_task_macro_use_different_denominators():
    one = summarize_task(*example_task())
    other = deepcopy(one)
    other.update(n_steps=1, step_recall_sum=0., step_recall_mean=0., final_recall=0.,
                 final_occupancy=.25, cost_plain=20., cost_cached=25.,
                 complete_question_count=1, invalid_action_count=1, over_budget_count=2)
    out = aggregate_tasks([one, other])
    assert out["step_recall"]["micro"] == {"n": 5, "mean": pytest.approx(2 / 3)}
    assert out["step_recall"]["task_macro"] == {"n": 2, "mean": pytest.approx(5 / 12)}
    assert out["final_recall"] == {"n": 2, "mean": pytest.approx(1 / 6)}
    assert out["evidence_complete_questions"] == {"n": 4, "count": 1, "rate": .25}
    assert out["occupancy"] == {"n": 2, "mean": .375, "max": .5}
    assert out["action_validity"] == {"n": 5, "invalid_count": 1, "over_budget_count": 2}
    assert out["cost"]["plain"]["mean"] == 50


def test_invalid_and_over_budget_steps_are_counted_not_discarded():
    args = list(example_task())
    args[0][0]["action_valid"] = False
    args[-1] = 25  # The second state keeps 30 tokens.
    out = summarize_task(*args)
    assert out["n_steps"] == 4
    assert out["invalid_action_count"] == 1
    assert out["over_budget_count"] == 1


def test_empty_aggregation_retains_zero_denominators():
    out = aggregate_tasks([])
    assert out["n_tasks"] == out["n_steps"] == 0
    assert out["step_recall"]["micro"] == {"n": 0, "mean": None}
    assert out["final_recall"]["mean"] is None
    assert out["evidence_loss"]["rate"] is None
    assert out["occupancy"]["max"] is None
    assert out["cost"]["cached"]["mean"] is None


def test_gold_mismatch_and_bad_shape_are_rejected():
    args = list(example_task())
    args[2] = [{10, 20}, {30, 40}]
    with pytest.raises(ValueError, match="gold sets differ"):
        summarize_task(*args)
    args = list(example_task())
    args[0] = args[0][:-1]
    with pytest.raises(ValueError, match="per step"):
        summarize_task(*args)
