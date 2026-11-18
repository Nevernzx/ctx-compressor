import ast
from pathlib import Path

import pytest

from src.reader import parse_answer, render_passages, render_user_message


def test_context_user_message_matches_task_template_byte_for_byte():
    passages = "First title\nFirst body.\n\nSecond title\nSecond body."
    question = "Where was Ada born?"
    expected = (
        "Answer the question based on the passages below. Reply with only the final answer "
        "(a short phrase such as a name, date, place or number), with no explanation.\n\n"
        "Passages:\nFirst title\nFirst body.\n\nSecond title\nSecond body.\n\n"
        "Question: Where was Ada born?"
    )
    assert render_user_message(passages, question) == expected
    assert not render_user_message(passages, question).endswith("\n")


def test_no_context_message_matches_task_template_and_does_not_include_passages():
    expected = (
        "Answer the question. Reply with only the final answer (a short phrase such as a "
        "name, date, place or number), with no explanation.\n\nQuestion: Where was Ada born?"
    )
    assert render_user_message("THIS MUST NOT APPEAR", "Where was Ada born?", no_context=True) == expected


def test_template_does_not_strip_or_interpret_question_and_passages():
    assert render_user_message("{braces}\n", " question? ").endswith(
        "Passages:\n{braces}\n\n\nQuestion:  question? "
    )


def test_empty_context_remains_an_explicit_empty_passages_block():
    assert "\n\nPassages:\n\n\nQuestion: q" in render_user_message("", "q")
    assert "Passages:" not in render_user_message("", "q", no_context=True)


def test_passages_use_original_fields_and_preserve_arrival_order():
    corpus = [
        {"title": "Title 0", "paragraph_text": " body 0 ", "text": "do not use this"},
        {"title": "Title 1", "paragraph_text": "body 1\nline 2", "text": "do not use this either"},
    ]
    assert render_passages(corpus, [1, 0]) == "Title 1\nbody 1\nline 2\n\nTitle 0\n body 0 "
    assert render_passages(corpus, [0, 0]) == "Title 0\n body 0 \n\nTitle 0\n body 0 "
    assert render_passages(corpus, []) == ""


@pytest.mark.parametrize("raw,expected", [
    ("Paris", "Paris"),
    (" \n Paris. \n", "Paris"),
    ("Answer: Paris.", "Paris"),
    ("aNsWeR: \n\n Paris.\nExplanation follows.", "Paris"),
    ("<think>reason\nmore reasoning</think>\nAnswer: Paris.", "Paris"),
    ("<think>one</think><think>two</think>\nParis", "Paris"),
    (" \n<think>reason</think>\n\n", ""),
    ("", ""),
    (" \n\t ", ""),
    ("Answer:\n \n", ""),
    ('"Paris."', "Paris"),
    ('"Paris".', "Paris"),
    ("'New York'.", "New York"),
    ("“New York.”", "New York"),
    ("‘New York’.", "New York"),
    ("Paris。\nSecond line", "Paris"),
    ("U.S.A.", "U.S.A"),
    ("St. Louis.", "St. Louis"),
    ("O'Connor.", "O'Connor"),
    ("Paris\nbecause it is France's capital", "Paris"),
    ("Answer: \n\n'Paris'\nAnswer: London", "Paris"),
    ("...Paris.", "...Paris"),
])
def test_parse_answer_exact_sequence(raw, expected):
    assert parse_answer(raw) == expected


def test_incomplete_thinking_block_is_not_hidden():
    assert parse_answer("<think>unfinished reasoning\nParis") == "<think>unfinished reasoning"
    assert parse_answer("</think>Paris") == "</think>Paris"


def test_answer_prefix_only_removed_at_its_specified_stage_and_position():
    assert parse_answer('\"Answer: Paris\"') == "Answer: Paris"
    assert parse_answer("The Answer: Paris.") == "The Answer: Paris"
    assert parse_answer("Answer : Paris.") == "Answer : Paris"


def test_reader_modules_have_only_standard_library_imports():
    project = Path(__file__).resolve().parents[1]
    for filename in ("reader.py", "qa_metrics.py"):
        tree = ast.parse((project / "src" / filename).read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module.split(".")[0])
        assert set(names) <= {"re", "string", "collections"}
