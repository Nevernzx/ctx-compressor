"""Synthetic model histories test the independent audit without model imports."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts.audit_model_rollouts import (AuditError, CHECKPOINT_STEPS, STRATEGIES,
    TrackedJsonl, audit_strategy, completion_status, independent_selection, independently_execute)
from src.heldout_eval import HeldoutReplay
from src.model_inference import inference_prompt
from src.policies import no_compress_policy


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
        return [151644, *messages[0]["content"].encode(), 151645]


@pytest.fixture
def fixture():
    corpus = [{"pid": i, "text": f"title {i}\nparagraph {i}", "ntok": 600 if i < 6 else 2000}
              for i in range(8)]
    tasks = [{"task": task, "split": "heldout", "questions": ["a", "b", "c", "d"],
              "example_ids": [f"{task}_{q}" for q in range(4)],
              "steps": [{"new": [2*q, 2*q+1], "gold": 2*q,
                         "question_index": q, "hop_index": 0} for q in range(4)]}
             for task in range(2)]
    baseline = HeldoutReplay(tasks, corpus, "no_compress")
    while not baseline.completed():
        baseline.advance_batch([{"raw_output": json.dumps(no_compress_policy(item["state"]))}
                                for item in baseline.next_batch()])
    baseline = baseline.aggregate()
    replay, records = HeldoutReplay(tasks, corpus, "base"), []
    tokenizer = FakeTokenizer()
    while not replay.completed():
        outputs = []
        for item in replay.next_batch():
            state, step = item["state"], item["step"]
            if step == 0:
                action = {"add": [state["new_ids"][0]], "drop": []}
            elif step == 1:
                action = ({"add": [], "drop": state["kept_ids"]} if item["mode"] == "append"
                          else {"add": [999], "drop": []})
            else:
                action = {"add": state["new_ids"], "drop": []}
            raw = "```json\n{}\n```" if step == 2 else json.dumps(action)
            limit = step == 2
            outputs.append({**inference_prompt(tokenizer, state), "raw_output": raw,
                "generated_tokens": 256 if limit else 1, "output_token_ids": [42] * (256 if limit else 1),
                "finish_reason": "length" if limit else "stop", "stop_reason": None,
                "output_limit_hit": limit, "input_truncated": False, "runtime_prompt_ids_match": True})
        records.extend(replay.advance_batch(outputs))
    metrics = replay.aggregate(baseline=baseline)
    metrics["checkpoint_step"] = None
    # Exercise the exact on-disk integer-key restoration path.
    records, metrics, baseline = json.loads(json.dumps((records, metrics, baseline)))
    return records, tasks, corpus, metrics, baseline, tokenizer


def verify(fixture, **kwargs):
    records, tasks, corpus, metrics, baseline, tokenizer = fixture
    return audit_strategy(records, tasks, corpus, metrics, baseline, tokenizer=tokenizer, **kwargs)


def test_independent_full_reconstruction_matches_all_metrics(fixture):
    receipt = verify(fixture, prompt_check="all")
    assert receipt["n_trajectories"] == 12
    assert receipt["n_step_records"] == 48
    assert receipt["prompt_reencoding"]["n"] == 48
    assert all(receipt["combined"]["invalid_outputs"][reason]["count"] > 0
               for reason in ("invalid_json", "invalid_candidate_id", "append_drop", "over_budget"))
    assert receipt["combined"]["output_limit"]["count"] == 12


def test_first_prompt_only_still_hash_checks_every_record(fixture):
    receipt = verify(fixture)
    assert receipt["prompt_reencoding"]["n"] == 1
    assert receipt["prompt_reencoding"]["all_recorded_id_hashes_checked"] == 48
    assert receipt["prompt_reencoding"]["all_visible_prompt_text_hashes_checked"] == 48


@pytest.mark.parametrize("target", ["raw_output", "kept_before", "keep_after", "stable_id", "mode",
                                   "budget", "step", "gold", "invalid_reason", "action_valid",
                                   "prompt_ids", "text_hash", "output_limit", "generated_tokens"])
def test_rejects_record_corruption(fixture, target):
    rows = fixture[0]
    if target == "raw_output":
        rows[0][target] = '{"add":[],"drop":[]}'
    elif target in ("kept_before", "keep_after"):
        rows[0][target] = [999]
    elif target == "stable_id":
        rows[0]["state"]["pid_by_lid"]["0"] = 7
    elif target == "mode":
        rows[0][target] = "append"
    elif target == "budget":
        rows[0][target] += 1
    elif target in ("step", "gold"):
        rows[0][target] = 1
    elif target == "invalid_reason":
        rows[12][target] = "over_budget"
    elif target == "action_valid":
        rows[0][target] = 1  # Do not silently accept int == bool.
    elif target == "prompt_ids":
        rows[0]["generation_metadata"]["prompt_token_ids"][0] += 1
    elif target == "text_hash":
        rows[0]["generation_metadata"]["user_text_sha256"] = "wrong"
    elif target == "output_limit":
        rows[0]["output_limit_hit"] = True
    else:
        rows[0]["generation_metadata"][target] = 2
    with pytest.raises(AuditError):
        verify(fixture)


def test_rejects_wrong_prompt_even_if_recorded_id_hash_was_rewritten(fixture):
    metadata = fixture[0][0]["generation_metadata"]
    metadata["prompt_token_ids"][0] += 1
    data = json.dumps(metadata["prompt_token_ids"], ensure_ascii=False, separators=(",", ":")).encode()
    metadata["prompt_sha256"] = hashlib.sha256(data).hexdigest()
    with pytest.raises(AuditError, match="reencoded_prompt_ids"):
        verify(fixture)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "out_of_step_order"])
def test_rejects_missing_duplicated_or_reordered_trajectories(fixture, mutation):
    rows = fixture[0]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[-1]))
    else:
        rows[0], rows[12] = rows[12], rows[0]
    with pytest.raises(AuditError):
        verify(fixture)


@pytest.mark.parametrize("metric", ["final_recall", "cost", "invalid_outputs", "unknown_field"])
def test_recomputes_each_metric_instead_of_trusting_summary(fixture, metric):
    combined = fixture[3]["combined"]
    if metric == "final_recall":
        combined[metric]["mean"] += .01
    elif metric == "cost":
        combined[metric]["cached"]["mean"] += 1
    elif metric == "invalid_outputs":
        combined[metric]["invalid_json"]["count"] = 0
    else:
        combined[metric] = "unverified"
    with pytest.raises(AuditError):
        verify(fixture)


@pytest.mark.parametrize("raw", ['{"add":[true]}', '{"add":[],"add":[0]}', '{"add":[],"other":NaN}',
                                 '{"add":[]} trailing', '[1]', '{"add":1}'])
def test_independent_parser_rejects_malformed_json_without_salvage(raw):
    state = {"kept_ids": [], "new_ids": [0], "mode": "append", "budget": 10, "ntok": {0: 1}, "gold": [0]}
    actual = independently_execute(state, raw)
    assert actual["invalid_reason"] == "invalid_json" and actual["keep_after"] == []


def test_independent_parser_keeps_original_optional_drop_duplicate_and_extra_key_semantics():
    state = {"kept_ids": [], "new_ids": [0], "mode": "append", "budget": 10, "ntok": {0: 1}, "gold": [0]}
    actual = independently_execute(state, '{"add":[0,0],"other":"ignored"}')
    assert actual["action_valid"] and actual["keep_after"] == [0]


def test_missing_model_is_pending_without_a_provisional_winner(tmp_path):
    results, rollouts = tmp_path / "results", tmp_path / "rollouts"
    results.mkdir()
    rollouts.mkdir()
    run = {"status": "complete", "strategies": {name: {"status": "complete"} for name in STRATEGIES}}
    (results / "model_rollout_run.json").write_text(json.dumps(run))
    for name in STRATEGIES:
        if name == "checkpoint-2500":
            continue
        (rollouts / name).mkdir()
        for path in (rollouts / name / "steps.jsonl", rollouts / name / "metrics.json", results / f"{name}_results.json"):
            path.write_text("{}")
    receipt = completion_status(results, rollouts)
    assert receipt["status"] == "pending" and receipt["selected_checkpoint"] is None
    assert len(receipt["missing_inputs"]) == 3


def test_independent_selection_requires_all_five_and_follows_all_ties():
    audits = [{"strategy": f"checkpoint-{step}", "checkpoint_step": step, "n_trajectories": 1494,
               "combined": {"evidence_loss": {"count": 9 if step == 2500 else 8, "n": 1494},
                            "final_recall": {"mean": .9 if step == 500 else .95, "n": 1494}}}
              for step in CHECKPOINT_STEPS]
    selected = independent_selection(audits)
    assert selected["strategy"] == "checkpoint-1000"
    with pytest.raises(AuditError, match="all five"):
        independent_selection(audits[:-1])


def test_tracked_reader_hashes_exact_bytes_and_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "steps.jsonl"
    data = b'{"step":0}\n{"step":1}\n'
    path.write_bytes(data)
    reader = TrackedJsonl(path)
    assert list(reader) == [{"step": 0}, {"step": 1}]
    assert reader.record == {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    path.write_text('{"step":0,"step":1}\n')
    with pytest.raises(ValueError, match="duplicate JSON"):
        list(TrackedJsonl(path))
