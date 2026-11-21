"""Task 09 held-out policy, provenance and closed-loop integration tests on CPU."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import eval_hedge_v2 as run
from src.model_inference import inference_prompt


@pytest.mark.parametrize(("policy","expected"), [("nohedge",0),("tau_star",.6),("always",1.01)])
def test_no_heldout_threshold_selection(policy, expected):
    assert run.run_tau(policy, {"status":"complete","tau_star":.6}) == expected
    with pytest.raises(ValueError):
        run.run_tau(policy, {"status":"running","tau_star":.9})
    with pytest.raises(ValueError):
        run.run_tau(policy, {"status":"complete","tau_star":.63})


def assets():
    corpus = [{"pid": p, "ntok": n, "text": f"text {p}", "title": "a" if p < 2 else "b"}
              for p, n in enumerate((2, 3, 4, 1))]
    task = {"task": 12, "questions": list("abcd"), "steps":[
        {"new": [0,1], "gold":1, "question_index":0, "hop_index":0},
        {"new": [1,2], "gold":2, "question_index":1, "hop_index":0},
        {"new": [3], "gold":3, "question_index":2, "hop_index":0},
        {"new": [0], "gold":0, "question_index":3, "hop_index":0}]}
    return [task], corpus


class SyntheticEngine:
    def get_tokenizer(self):
        return self

    def apply_chat_template(self, messages, **kwargs):
        return [len(messages[0]["content"]), 2]

    def generate(self, prompts, sampling_params, lora_request, use_tqdm):
        assert sampling_params == "greedy" and lora_request == "sft-2000"
        reply = SimpleNamespace(text='{"add":[0],"drop":[]}', token_ids=[8], finish_reason="stop", stop_reason=None)
        return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[reply]) for p in prompts]


@pytest.mark.parametrize(("policy","threshold","first_keep","second_new"),
                         [("nohedge", 0., [0], [1,2]), ("tau_star", .9, [0,1], [2]), ("always", 1.01, [0,1], [2])])
def test_replay_feeds_executed_keep_into_next_prompt(tmp_path, policy, threshold, first_keep, second_new):
    tasks, corpus = assets()
    engine = SyntheticEngine()
    grid = run.HeldoutReplay(tasks, corpus, policy).next_batch()
    digest = hashlib.sha256()
    for item in grid:
        digest.update(run.sft.prompt_digest(run.sft.grid_id(item), inference_prompt(engine, item["state"])))
    proof = {"engine_parameters":{"max_model_len": 40000}, "step0":{"n":6, "identity_and_hashes_sha256":digest.hexdigest()}}
    def score(llm, adapter, prompt_ids, candidate_ids):
        assert llm is engine and adapter == "sft-2000" and len(prompt_ids) == len(candidate_ids)
        return [{lid: -.1 * i for i, lid in enumerate(candidates)} for candidates in candidate_ids]
    metrics, by_cell = run.replay_run(engine, "sft-2000", "greedy", score, policy, threshold, tasks, corpus,
                                      proof, None, tmp_path, tmp_path / "run.json", {}, SimpleNamespace(check=lambda: None))
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == metrics["combined"]["n_steps"] == 24
    assert rows[0]["greedy_raw_output"] == '{"add":[0],"drop":[]}'
    assert rows[0]["keep_after"] == rows[6]["kept_before"] == first_keep
    assert rows[6]["new"] == second_new and rows[6]["action_valid"] is False
    assert rows[6]["raw_output"] == rows[6]["greedy_raw_output"] == rows[6]["executed_raw_output"]
    assert rows[0]["state"]["parsed_action"]["add"] == first_keep
    assert rows[0]["raw_output"] == rows[0]["executed_raw_output"]
    assert rows[0]["generation_metadata"]["output_token_ids"] == [8]
    assert rows[0]["generation_metadata"]["output_token_ids_source"] == "greedy_raw_output"
    extra = run.summarize(by_cell["reselect|tight"])
    yes = int(threshold > 0)
    assert extra["hedge_rate"] == {"count":yes, "n":4, "rate":yes / 4}
    assert extra["hedge_hit_rate"] == {"count":yes, "n":yes, "rate":1 if yes else None}
    assert extra["mean_kept_tokens_per_step"]["mean"] == sum(row["used_after"] for row in rows[: :6]) / 4
    assert sum(bucket["count"] for bucket in extra["executed_add_counts_all_steps"].values()) == 4
    assert extra["executed_add_counts_all_steps"]["0"]["count"] == 3


def test_missing_score_batch_rejected_before_advancing():
    tasks, corpus = assets()
    replay = run.HeldoutReplay(tasks, corpus, "always")
    items = replay.next_batch()
    with pytest.raises(ValueError, match="Incomplete"):
        run.wrap_batch(replay, items, [{"raw_output": ""} for _ in items], [], 1.01)
    assert replay.next_batch() == items and items[0]["step"] == 0

