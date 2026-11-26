"""CPU-only identity, subset and negative controls for Task 14 input preparation."""
import hashlib
import random

import pytest

from scripts import deepseek_api_inputs as inp


def test_registered_subset_and_smoke_composition():
    tasks = list(range(249))
    s60, s30 = inp.subsets(reversed(tasks))
    assert s60 == random.Random(20261127).sample(tasks, 60)
    assert s30 == s60[:30] and s60[:2] == [9, 151]
    smoke = list(inp.configurations(s60, s30, "smoke"))
    assert smoke == [("oracle", "reselect", "mid", 9), ("oracle", "reselect", "mid", 151),
                     ("no_compress", None, None, 9)]
    full = list(inp.configurations(s60, s30, "full"))
    assert len(full) == len(set(full)) == 450
    assert set(smoke) <= set(full)
    assert sum(key[0] == "no_context" for key in full) == 60
    assert sum(key[0] == "no_compress" for key in full) == 30


def toy():
    corpus = [{"title": "文", "paragraph_text": "One\ntwo"},
              {"title": "Other", "paragraph_text": "More"},
              {"title": "Source", "paragraph_text": "unknown"}]
    metadata = [dict(example_id=str(i), question=f"Who is {i}?", answer="New York City",
                     answer_aliases=["NYC"], question_hops=2, gold={i % 2}) for i in range(4)]
    return corpus, {9: {"metadata": metadata, "all_gold": {0, 1}}}


def test_render_and_task_level_non_gold_count():
    corpus, tasks = toy()
    row, *others = inp.question_rows(inp.R2, "append", 9, [2, 0], tasks, corpus)
    message = inp.t12.render_user_message("Source\nunknown\n\n文\nOne\ntwo", "Who is 0?")
    assert row["messages"] == [{"role": "user", "content": message}]
    assert row["prompt_sha1"] == hashlib.sha1(message.encode()).hexdigest()
    assert row["evidence_complete"] and row["trajectory_lost"]
    assert others[0]["evidence_complete"] is False
    assert all(x["non_gold_count"] == 1 for x in [row, *others])
    closed = inp.question_rows("no_context", None, 9, [], tasks, corpus)
    assert closed[0]["messages"][0]["content"] == inp.t12.render_user_message("", "Who is 0?", no_context=True)
    inputs = inp.PreparedInputs([9], [9], tasks, corpus, {}, {})
    assert [r["question_index"] for r in inputs.reader_rows(9, "reselect", [0, 1])] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="outside registered subset"):
        inputs.reader_rows(8, "reselect", [])


def test_prompt_identity_and_rescoring_enforced():
    corpus, tasks = toy()
    r = inp.question_rows("oracle", "append", 9, [1, 0], tasks, corpus)[0]
    prediction = dict(r, input_truncated=False, raw_output="Answer: NYC.", pred="NYC",
                      prompt_tokens=80, em=1.0, f1=1.0)
    assert inp.compare_qwen(r, prediction)["qwen_f1"] == 1
    changed = inp.question_rows("oracle", "append", 9, [0, 1], tasks, corpus)[0]
    with pytest.raises(ValueError, match="prompt_sha1"):
        inp.compare_qwen(changed, prediction)
    with pytest.raises(ValueError, match="identity mismatch"):
        inp.compare_qwen(r, dict(prediction, task=90))
    with pytest.raises(ValueError, match="scoring mismatch"):
        inp.compare_qwen(r, dict(prediction, f1=0.0))

