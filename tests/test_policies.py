"""Task 02 policies: hand values, information isolation and real-dev BM25."""
import copy
import json
from pathlib import Path
import random

import bm25s
import numpy as np
import pytest
from tokenizers import Tokenizer

from src.build_traj import apply_action, build_corpus, hop_queries, rollout
from src.policies import bm25_policy, latest_policy, no_compress_policy, score_query


def state(mode="reselect"):
    # IDs are deliberately unrelated to arrival order.
    return {"questions": ["alpha", "beta", "gamma", "delta"], "mode": mode,
            "budget": 8, "kept_ids": [40, 2], "new_ids": [19, 7, 1],
            "ntok": {40: 2, 2: 3, 19: 4, 7: 3, 1: 9},
            "text": {i: str(i) for i in (40, 2, 19, 7, 1)},
            "pid_by_lid": {40: 0, 2: 1, 19: 2, 7: 3, 1: 4},
            "gold": [40, 19], "sft_target": {"add": [19], "drop": []}}


def test_latest_reselect_skips_oversized_and_continues_first_fit():
    s = state()
    # Reverse order: 1(9) skipped; 7(3)+19(4) fit; both older fragments miss.
    action = latest_policy("reselect")(s)
    assert action == {"add": [19, 7], "drop": [40, 2]}
    assert apply_action(s, action) == [19, 7]


def test_latest_append_keeps_old_and_fills_only_remaining_space():
    s = state("append")
    assert latest_policy()(s) == {"add": [7], "drop": []}
    assert apply_action(s, latest_policy()(s)) == [40, 2, 7]


def test_bm25_first_fit_uses_full_corpus_pid_scores():
    s = state()
    action = bm25_policy(None, scores=[7, 6, 9, 8, 10])(s)
    # pid 4 (local 1) is highest but 9 tokens exceeds B; next 4+3 fit.
    assert action == {"add": [19, 7], "drop": [40, 2]}


def test_bm25_same_score_prefers_newest_presentation_not_highest_id():
    s = state()
    s["ntok"] = {i: 4 for i in s["ntok"]}
    action = bm25_policy(None, scores=np.ones(5))(s)
    assert action == {"add": [7, 1], "drop": [40, 2]}
    assert apply_action(s, action) == [7, 1]


def test_bm25_append_keeps_old_even_when_old_scores_are_zero():
    s = state("append")
    action = bm25_policy(None, "append", scores=[0, 0, 9, 8, 10])(s)
    assert action == {"add": [7], "drop": []}


@pytest.mark.parametrize("factory", [latest_policy, lambda: bm25_policy(None, scores=[5, 4, 3, 2, 1])])
@pytest.mark.parametrize("mode", ["reselect", "append"])
def test_changing_gold_and_oracle_does_not_change_action(factory, mode):
    s = state(mode)
    policy = factory()
    before = policy(s)
    for gold in ([], [1, 7, 19, 2, 40], [19, 40]):
        s["gold"] = gold
        s["sft_target"] = {"add": [999999], "drop": [40, 2]}
        s["recall"] = -987
        assert policy(s) == before


class VisibleState(dict):
    allowed = {"questions", "mode", "budget", "kept_ids", "new_ids", "text", "ntok", "pid_by_lid"}

    def __getitem__(self, key):
        assert key in self.allowed, f"Forbidden state read: {key}"
        return super().__getitem__(key)

    def get(self, key, *args):
        assert key in self.allowed, f"Forbidden state read: {key}"
        return super().get(key, *args)

    def __iter__(self):
        raise AssertionError("Reading the entire state is not allowed")

    def items(self):
        raise AssertionError("Reading the entire state is not allowed")


@pytest.mark.parametrize("mode", ["reselect", "append"])
def test_policies_work_with_only_whitelisted_fields(mode):
    s = state(mode)
    visible = VisibleState({key: value for key, value in s.items() if key in VisibleState.allowed})
    assert latest_policy()(visible) == latest_policy()(s)
    policy = bm25_policy(None, scores=[5, 4, 3, 2, 1])
    assert policy(visible) == policy(s)


@pytest.mark.parametrize("mode", ["reselect", "append"])
@pytest.mark.parametrize("seed", range(5))
def test_baselines_are_legal_on_random_feasible_states(mode, seed):
    rng = random.Random(seed)
    for _ in range(150):
        ids = rng.sample(range(1000), rng.randint(1, 50))
        split = rng.randrange(len(ids) + 1)
        lengths = {i: rng.randint(1, 300) for i in ids}
        kept, new = ids[:split], ids[split:]
        budget = max(1, sum(lengths[i] for i in kept) + rng.randint(0, 500))
        s = {"mode": mode, "budget": budget, "kept_ids": kept, "new_ids": new,
             "ntok": lengths, "pid_by_lid": dict(zip(ids, range(len(ids))))}
        scores = [rng.choice([0.0, 0.0, 1.0, 2.0, 3.0]) for _ in ids]
        for policy in (latest_policy(), bm25_policy(None, scores=scores)):
            action = policy(s)
            after = apply_action(s, action)
            assert after is not None
            assert sum(lengths[i] for i in after) <= budget
            if mode == "append":
                assert not action["drop"] and after[:len(kept)] == kept


def test_no_compress_keeps_everything_through_rollout():
    corpus = [{"text": str(i), "ntok": i + 1} for i in range(5)]
    steps = [{"new": [0, 1, 2], "gold": 1}, {"new": [1, 3, 4], "gold": 4}]
    outcomes = []
    for mode in ("reselect", "append"):
        samples = rollout(["q"], steps, corpus, 16, mode, no_compress_policy)
        assert all(s["action_valid"] and s["recall"] == 1 for s in samples)
        outcomes.append(samples[-1]["keep_after"])
    assert outcomes == [list(range(5)), list(range(5))]


@pytest.mark.parametrize("factory", [latest_policy, lambda mode: bm25_policy(None, mode, scores=np.ones(5))])
def test_mode_mismatch_is_not_silently_overridden(factory):
    with pytest.raises(ValueError, match="mode does not match"):
        factory("append")(state("reselect"))


def small_index():
    documents = ["alpha alpha beta", "beta gamma", "gamma delta"]
    index = bm25s.BM25()
    index.index(bm25s.tokenize(documents, show_progress=False), show_progress=False)
    return index


def test_score_query_matches_retrieve_with_repeated_and_unknown_terms():
    index = small_index()
    for query in ("alpha alpha gamma", "no_such_token_qzx", "the a an", ""):
        scores = score_query(index, query)
        ids, retrieved = index.retrieve(bm25s.tokenize([query], show_progress=False), k=3, show_progress=False)
        np.testing.assert_allclose(scores[ids[0]], retrieved[0], rtol=0, atol=0)
        np.testing.assert_allclose(np.sort(scores)[::-1], retrieved[0], rtol=0, atol=0)


def test_bm25_joins_task_questions_and_caches_by_query(monkeypatch):
    calls = []

    def score(index, query):
        calls.append(query)
        return np.arange(5, dtype=float)

    monkeypatch.setattr("src.policies.score_query", score)
    s = state()
    visible = VisibleState({key: value for key, value in s.items() if key in VisibleState.allowed})
    policy = bm25_policy(object())
    policy(visible)
    policy(visible)
    visible["questions"] = ["a new task", "question two"]
    policy(visible)
    assert calls == ["alpha beta gamma delta", "a new task question two"]


@pytest.fixture(scope="module")
def real_dev_index_and_queries():
    project = Path(__file__).resolve().parents[1]
    dev_file = project / "data/raw/musique/musique_ans_v1.0_dev.jsonl"
    tokenizer_file = project / "data/raw/qwen3_tokenizer/tokenizer.json"
    if not dev_file.is_file() or not tokenizer_file.is_file():
        pytest.skip("real dev/tokenizer integration check runs on 5880 only")
    with dev_file.open(encoding="utf-8") as stream:
        examples = [json.loads(line) for line in stream if line.strip()]
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    _, index, keys = build_corpus(examples, lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))
    queries = [query for example in examples for query, _ in hop_queries(example, keys)]
    return index, random.Random(20261117).sample(queries, 20)


@pytest.mark.parametrize("query_number", range(20))
def test_real_dev_twenty_hop_queries_match_full_corpus_retrieval(real_dev_index_and_queries, query_number):
    index, queries = real_dev_index_and_queries
    query = queries[query_number]
    scores = score_query(index, query)
    ranked = np.argsort(-scores, kind="stable")[:20]
    retrieved_ids, retrieved_scores = index.retrieve(
        bm25s.tokenize([query], show_progress=False), k=20, show_progress=False)
    retrieved_ids = retrieved_ids[0]
    assert len(set(retrieved_ids)) == 20
    np.testing.assert_allclose(scores[retrieved_ids], retrieved_scores[0], rtol=0, atol=0)
    # At a tied boundary, either ID is valid; the score multiset must be exact.
    np.testing.assert_allclose(scores[ranked], retrieved_scores[0], rtol=0, atol=0)
    assert set(np.flatnonzero(scores > scores[ranked[-1]])).issubset(set(retrieved_ids))
