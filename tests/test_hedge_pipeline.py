"""CPU-only guards for Task 09 admission and frozen threshold selection."""
import copy

import pytest

from scripts import run_hedge_pipeline as pipeline


def score_receipt():
    return {"status": "ok", "sample_n": 300,
            "argmax": {"n": 300, "matched": 297},
            "logprob": {"n": 300, "within_tolerance": 300, "max_abs_error": 0.02, "tolerance": 0.02},
            "reencode": {"n": 300, "returned_raw_full_id_matches": 300,
                         "text_token_matches": 300, "full_with_terminal_matches": 300}}


def test_score_threshold_boundaries():
    receipt = score_receipt()
    pipeline.check_scores(receipt)
    for field, values in (("argmax", {"matched": 296}),
                          ("logprob", {"max_abs_error": 0.0200001}),
                          ("logprob", {"within_tolerance": 299}),
                          ("reencode", {"returned_raw_full_id_matches": 299})):
        changed = copy.deepcopy(receipt)
        changed[field].update(values)
        with pytest.raises(ValueError):
            pipeline.check_scores(changed)


def test_train_only_selection_and_tie_break(monkeypatch):
    scores = {"path": "/external/train/state_scores.jsonl", "sha256": "test", "bytes": 1}
    means = {str(float(t)): {"n": 22158, "mean_reward": {"n": 22158, "mean": 1.0}} for t in pipeline.GRID}
    means["0.0"].update(current_gold_miss_rate={"count": 1529, "n": 22146},
                        hedge_rate={"count": 0}, mean_extra_tokens={"sum": 0})
    cells = {f"{mode}|{tier}": {k: {"n": 3693} for k in means}
             for mode in ("append", "reselect") for tier in ("loose", "mid", "tight")}
    receipt = {"status": "complete", "n_states": 22158, "n_tasks": 400, "n_trajectories": 2400,
               "heldout_overlap": 0, "split": "train", "lambda": 0.3, "grid": pipeline.GRID, "tau_star": 0,
               "state_file": {"sha256": pipeline.TRAIN_SHA}, "score_file": scores, "pooled": means, "cells": cells}
    monkeypatch.setattr(pipeline, "read", lambda _: receipt)
    monkeypatch.setattr(pipeline, "file_record", lambda _: scores)
    assert pipeline.check_selection()["tau_star"] == 0
    for k, value in (("tau_star", 0.8), ("split", "heldout"), ("heldout_overlap", 1)):
        original = receipt[k]
        receipt[k] = value
        with pytest.raises(ValueError):
            pipeline.check_selection()
        receipt[k] = original


def test_three_runs_have_distinct_output_names():
    assert pipeline.RUNS == (("H0", "nohedge"), ("H1", "tau_star"), ("H2", "always"))

