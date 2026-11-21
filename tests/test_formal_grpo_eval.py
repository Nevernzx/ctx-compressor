"""Pure CPU tests for Task 08 held-out extrapolation and replay audits."""
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import eval_formal_grpo as ev


def step(valid=True):
    return {"mode":"reselect", "budget_tier":"mid", "task":11, "step":1,
            "kept_before":[3], "keep_after":[3, 1, 2] if valid else [3],
            "returned":[0, 1, 2, 3], "new":[0, 1, 2], "gold":1, "action_valid":valid,
            "state":{"questions":["q"], "mode":"reselect", "budget":100, "kept_ids":[8],
                     "new_ids":[9,10,11], "ntok":{"9":1,"10":1,"11":1,"8":1},
                     "pid_by_lid":{"9":0,"10":1,"11":2,"8":3},"gold":[10,9]},
            "generation_metadata":{"prompt_token_ids":[17,19], "user_text_sha256":"abc"}, "raw_output":"a"}


def test_output_step_matches_model_and_checkpoint():
    assert ev.output_step("merged_sft_v2", None) is None
    assert ev.output_step("checkpoint-125", Path("train/R/checkpoint-125")) == 125
    for name, adapter in [("../checkpoint-125", Path("checkpoint-125")), ("checkpoint-125", None),
                          ("checkpoint-125", Path("checkpoint-250")), ("merged_sft_v2", Path("x"))]:
        with pytest.raises(ValueError):
            ev.output_step(name, adapter)


def test_add_counts_and_title_strata_include_denominators():
    corpus = [{"title":x} for x in ("b","a","a","s")]
    a = ev.observations(step(), corpus)
    a.update(ev.observations(step(False), corpus))
    s = ev.summarize_extra(a)
    assert s["n_steps"] == 2 and s["n_valid"] == 1
    assert s["mean_added_per_valid"] == {"sum":2,"n":1,"mean":2}
    assert s["add_counts"]["2"]["rate"] == 1
    assert s["new_gold_by_title"]["same"] == {"count":1,"n":2,"rate":0.5}
    assert s["new_gold_by_title"]["different"] == {"count":0,"n":2,"rate":0}
    assert s["current_gold_valid_new_by_title"]["same"]["n"] == 1
    assert s["current_gold_valid_new_by_title"]["different"]["n"] == 0


def test_raw_output_difference_only_for_matching_inputs():
    row = step()
    old = deepcopy(row)
    old["raw_output"] += " "
    different = ev.raw_comparison(row, old)
    assert different["raw_changed_same_state"] == 1
    # Recorded action outcomes are never part of the input.
    old["state"]["action"] = {}
    assert ev.raw_comparison(row,old) == different
    old["state"]["gold"] = []
    diff = ev.raw_comparison(row, old)
    assert diff["raw_changed_same_prompt"] == 1 and not diff["raw_changed_same_state"]
    old["generation_metadata"]["prompt_token_ids"] = [18]
    assert not ev.raw_comparison(row, old)["same_prompt"]
    old["step"] = 0
    with pytest.raises(ValueError):
        ev.raw_comparison(row, old)


def test_merged_only_runner_with_native_sft_writer(tmp_path, monkeypatch):
    """Exercise a no-LoRA engine request and all six cells through the unchanged Task 06 writer."""
    from types import SimpleNamespace

    class SyntheticMerged:
        def get_tokenizer(self):
            return self

        def apply_chat_template(self, messages, **kwargs):
            return [7, 8]

        def generate(self, prompts, sampling_params, lora_request, use_tqdm):
            assert lora_request is None and sampling_params == "greedy" and not use_tqdm
            reply = SimpleNamespace(text='{"add":[]}', token_ids=[99], finish_reason="stop", stop_reason=None)
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[reply]) for p in prompts]

    raw, result = tmp_path / "raw", tmp_path / "result"
    raw.mkdir()
    monkeypatch.setattr(ev.sft, "OUTPUT", raw)
    monkeypatch.setattr(ev.sft, "RESULTS", result)
    c = [{"pid": j, "text": str(j), "title": str(j), "ntok": 1} for j in range(4)]
    steps = [{"new": [j], "gold": j, "question_index": j, "hop_index": 0} for j in range(4)]
    tasks = [{"task": i, "questions": list("abcd"), "steps": steps} for i in range(249)]
    engine = SyntheticMerged()
    h = ev.hashlib.sha256()
    for item in ev.HeldoutReplay(tasks, c, ev.F0).next_batch():
        prompt = ev.inference_prompt(engine, item["state"])
        h.update(ev.sft.prompt_digest(ev.sft.grid_id(item), prompt))
    proof = {"step0": {"identity_and_hashes_sha256": h.hexdigest()},
             "engine_parameters": {"max_model_len": 40960}, "expected_outputs_per_strategy": 5976}
    ev.sft.evaluate_adapter(engine, "greedy", None, {"strategy":ev.F0, "step":None}, tasks, c, None,
                            proof, SimpleNamespace(check=lambda: None), {"strategies":{}}, result / "run.json")
    metrics = ev.validate_strategy(ev.sft.load_json(result / (ev.F0 + "_results.json")), ev.F0)
    assert metrics["combined"]["n_steps"] == 5976
    check = ev.diagnostics.summarize_strategy(ev.F0, raw / ev.F0 / "steps.jsonl", metrics,
            {t["task"]:t["steps"] for t in tasks}, metrics["steps_artifact"]["sha256"])
    assert check["combined"]["n_trajectories"] == 1494
    assert check["combined"]["first_error"]["new_gold_not_added"]["count"] == 1494

