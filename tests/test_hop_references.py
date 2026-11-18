"""Regression coverage for literal #k text and completed-hop references."""
import copy

import pytest

from src.build_traj import hop_queries


def example(questions, answers=None):
    answers = answers or [f"answer-{i + 1}" for i in range(len(questions))]
    return {"id": "reference-regression", "paragraphs": [
        {"idx": 17, "title": "Evidence", "paragraph_text": "Evidence body."}],
        "question_decomposition": [
            {"id": 100 + i, "question": q, "answer": answer,
             "paragraph_support_idx": 17}
            for i, (q, answer) in enumerate(zip(questions, answers))]}


KEYS = {("Evidence", "Evidence body."): 31}


def test_dream_title_preserved_and_first_answer_resolved():
    ex = example(["#9 Dream >> performer", "Where was #1 born?"],
                 ["John Lennon", "Liverpool"])
    stats = {}
    assert hop_queries(ex, KEYS, stats=stats) == [
        ("#9 Dream >> performer", 31), ("Where was John Lennon born?", 31)]
    assert stats == {"preserved_reference_count": 1, "preserved_references": [{
        "example_id": "reference-regression", "hop_index": 0, "reference": "#9",
        "reference_index": 9, "completed_hops": 0, "span": [0, 2],
        "original_query": "#9 Dream >> performer"}]}


@pytest.mark.parametrize("literal", ["#0", "#1", "#9", "#123"])
def test_first_hop_has_no_completed_answers(literal):
    assert hop_queries(example([literal + " stays"]), KEYS) == [(literal + " stays", 31)]


def test_mixed_and_repeated_references_are_counted_by_occurrence():
    ex = example(["Start", "#1 / #9 / #0 / #9", "#2 / #1 / #3"])
    stats = {}
    assert hop_queries(ex, KEYS, stats=stats) == [
        ("Start", 31), ("answer-1 / #9 / #0 / #9", 31),
        ("answer-2 / answer-1 / #3", 31)]
    assert stats["preserved_reference_count"] == 4
    assert [x["reference"] for x in stats["preserved_references"]] == ["#9", "#0", "#9", "#3"]
    assert [x["hop_index"] for x in stats["preserved_references"]] == [1, 1, 1, 2]


@pytest.mark.parametrize("answer", ["#9 Dream", "literal #1", "#0 and #123"])
def test_literal_references_in_answers_are_inserted_once(answer):
    ex = example(["Which song?", "Who performs #1?"], [answer, "John Lennon"])
    stats = {}
    assert hop_queries(ex, KEYS, stats=stats)[1] == (f"Who performs {answer}?", 31)
    assert stats.get("preserved_reference_count", 0) == 0
    assert not stats.get("preserved_references")


def test_stats_accumulate_without_mutating_inputs():
    ex = example(["#9 Dream", "#1 >> author"])
    before = copy.deepcopy(ex)
    keys_before = dict(KEYS)
    stats = {"other": "untouched"}
    first = hop_queries(ex, KEYS, stats=stats)
    assert hop_queries(ex, KEYS, stats=stats) == first
    assert stats["preserved_reference_count"] == len(stats["preserved_references"]) == 2
    assert stats["other"] == "untouched"
    assert ex == before
    assert KEYS == keys_before


def test_stats_do_not_change_existing_two_argument_interface():
    ex = example(["Start", "#1 and #9"])
    assert hop_queries(ex, KEYS) == hop_queries(ex, KEYS, stats={})
