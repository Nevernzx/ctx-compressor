import numpy as np
import pytest

from src.policies import bm25_policy
from src import reranker_v1 as r
from scripts import reranker_v1_rollouts as rollout


def test_frozen_train_split_is_disjoint_and_val100_is_prefix():
    fit, val, val100 = r.split_task_ids(range(4735))
    assert len(fit) == 4535 and len(set(val)) == 200 and len(set(val100)) == 100
    assert set(fit).isdisjoint(val) and set(fit) | set(val) == set(range(4735))
    assert val100 == val[:100]
    assert r.split_task_ids(reversed(range(4735))) == (fit, val, val100)


def test_task_gold_labels_cover_future_and_keep_first_candidate_occurrence():
    steps = [{"new": [3, 7, 8], "gold": 7}, {"new": [7, 8, 2], "gold": 8}]
    assert r.candidates(steps) == [(3, 0), (7, 1), (8, 1), (2, 0)]


def test_train_draw_keeps_all_gold_and_samples_negatives_without_replacement():
    rows = {0: [(0, 1), (1, 1), *[(n, 0) for n in range(2, 24)]],
            1: [(101, 1), (102, 0), (103, 0)]}
    draw = r.sample_pair_rows(rows, 7)
    assert set(draw) >= {0, 1, 101, 102, 103}
    assert len(draw) == len(set(draw)) == 19
    assert len(r.sample_pair_rows(rows, 3)) == 11


class CharacterTokenizer:
    pad_token_id = 0
    is_fast = True
    truncation_side = "right"

    def convert_tokens_to_ids(self, text):
        return {"yes": 1, "no": 2}[text]

    def encode(self, text, **kwargs):
        if text in ("yes", "no"):
            return [self.convert_tokens_to_ids(text)]
        return [ord(c) + 10 for c in text]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(s) for s in texts],
                "offset_mapping": [[(i, i+1) for i in range(len(s))] for s in texts]}


def test_template_has_exact_instruction_empty_think_and_document_tail_truncation():
    raw = CharacterTokenizer()
    prompt = r.PairTokenizer(raw)
    q = "one\ntwo\nthree\nfour"
    d = "x" * 700
    encoded, cut = next(prompt.encode(q, [d]))
    body = r.prompt_text(q, "")
    assert body == f"<Instruct>: {r.INSTRUCTION}\n<Query>: {q}\n<Document>: "
    assert len(encoded) == 512 and cut
    assert encoded[:len(prompt.prefix) + len(body)] == prompt.prefix + raw.encode(body)
    assert encoded[-len(prompt.suffix):] == raw.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not next(prompt.encode(q, ["short"]))[1]


def test_last_logit_yes_minus_no_and_sigmoid_sign():
    score = r.logit_difference(np.array([[1., 4., 2.], [1., -1., 3.]]), 1, 2)
    assert score.tolist() == [2., -4.]
    assert r.probability(score[0]) > .5 > r.probability(score[1])


def test_first_fit_matches_bm25_ties_and_threshold_filter_in_both_modes():
    scores = [.4, .3, .7, .7]
    common = {"kept_ids": [0,1], "new_ids": [2,3], "pid_by_lid": {i:i for i in range(4)},
              "ntok": {0:2,1:2,2:6,3:5}, "budget": 10}
    bm25 = bm25_policy(None, scores=scores)
    for mode in ("reselect", "append"):
        state = {**common, "mode": mode}
        assert r.reranker_action(state, scores) == bm25(state)
        assert r.reranker_action(state, scores, .70)["add"] == [3]
    assert set(r.reranker_action({**common, "mode": "reselect"}, scores, .8)["drop"]) == {0, 1}
    assert r.reranker_action({**common, "mode": "append"}, scores, .8) == {"add": [], "drop": []}


def test_preregistered_tau_and_checkpoint_ties():
    arms = {x: .3 for x in r.GRID}
    arms[.5] = .5
    arms[.1] = .5 - 5e-10
    assert r.choose_tau(arms) == .1
    assert r.choose_checkpoint([{"step": 2000, "mAP": .5}, {"step": 1000, "mAP": .5}])["step"] == 1000
    assert r.average_precision([(1,1),(1,0),(0,0)]) == pytest.approx(.5)


def test_hop_denominators_and_first_dropped_piece():
    tasks = {1: {"steps": [{"question_index": 0, "hop_index": 0, "gold": 4},
                           {"question_index": 0, "hop_index": 1, "gold": 7}]}}
    finals = {(("append","tight"),1): [4], (("reselect","tight"),1): [4,7]}
    audit = rollout.hop_diagnostics(finals,tasks)
    assert audit["pooled"]["retained"]["1"] == {"count":2,"n":2,"rate":1.}
    assert audit["pooled"]["retained"]["2+"] == {"count":1,"n":2,"rate":.5}
    assert audit["append"]["first_loss"]["2"]["count"] == 1
    assert audit["reselect"]["first_loss"]["none"]["count"] == 1


def test_existing_and_symlinked_outputs_are_refused(tmp_path):
    old = tmp_path / "keep"
    old.write_bytes(b"document")
    alias = tmp_path / "alias"
    alias.symlink_to(old)
    for path in (old, alias):
        with pytest.raises(FileExistsError):
            r.require_new(path)
    assert old.read_bytes() == b"document"

