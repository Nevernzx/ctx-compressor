"""Check diagnostic timing and exercise the v2 adapter writer on synthetic, CPU-only responses."""
import hashlib
import json
from types import SimpleNamespace

from scripts import heldout_model_v2 as runner
from scripts.heldout_diagnostics_v2 import step_observations
from src.heldout_eval import HeldoutReplay
from src.model_inference import inference_prompt


def test_gold_known_before_and_current_return():
    row = {"mode": "reselect", "step": 2, "kept_before": [1, 9], "keep_after": [],
           "returned": [1, 2, 9], "gold": 9, "action_valid": True, "invalid_reason": None}
    counts, cause = step_observations(row, {1, 2})
    assert cause == "both"
    # 9 is first revealed this step. Before retrieval, the kept area contained noise.
    assert counts["all_gold_among_valid_reselect"] == 0
    assert counts["all_gold_among_valid_reselect_n"] == 1
    assert counts["all_gold_valid_drop_n"] == 0
    assert (counts["kept_gold_survival"], counts["kept_gold_survival_n"]) == (0, 1)
    # Previously missed 2 is returned again, while previously kept 9 is excluded.
    assert (counts["new_gold_add"], counts["new_gold_add_n"]) == (0, 1)
    row.update(kept_before=[1], keep_after=[1], action_valid=False, invalid_reason="invalid_json")
    counts, cause = step_observations(row, {1, 2})
    assert cause == "invalid_output" and counts["all_gold_valid_drop_n"] == 0
    assert (counts["new_gold_add"], counts["new_gold_add_n"]) == (0, 2)


class SyntheticEngine:
    def apply_chat_template(self, messages, **options):
        assert options == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
        return [3, 4]

    def get_tokenizer(self):
        return self

    def generate(self, prompts, sampling_params, lora_request, use_tqdm):
        assert sampling_params == "greedy" and lora_request == "synthetic-v2-500" and not use_tqdm
        completion = SimpleNamespace(text='{"add":[],"drop":[]}', token_ids=[4],
                                     finish_reason="stop", stop_reason=None)
        return [SimpleNamespace(outputs=[completion], prompt_token_ids=item["prompt_token_ids"])
                for item in prompts]


def test_completed_adapter_writes_six_cells_and_verified_prompt_records(tmp_path, monkeypatch):
    output, results = tmp_path/"external", tmp_path/"results"
    output.mkdir()
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "RESULTS", results)
    corpus = [{"pid": i, "text": str(i), "ntok": 1} for i in range(4)]
    tasks = [{"task": i, "split": "heldout", "questions": list("abcd"),
              "example_ids": [str(i)+"_"+q for q in "abcd"],
              "steps": [{"new": [j], "gold": j, "question_index": j, "hop_index": 0}
                        for j in range(4)]} for i in range(249)]
    engine = SyntheticEngine()
    name = "checkpoint-500"
    digest = hashlib.sha256()
    for item in HeldoutReplay(tasks, corpus, name).next_batch():
        prompt = inference_prompt(engine, item["state"])
        digest.update(runner.prompt_digest(runner.grid_id(item), prompt))
    proof = {"step0": {"identity_and_hashes_sha256": digest.hexdigest()},
             "engine_parameters": {"max_model_len": 40960}, "expected_outputs_per_strategy": 5976}
    manifest = {"strategies": {}}
    runner.evaluate_adapter(engine, "greedy", "synthetic-v2-500", {"strategy": name, "step": 500},
                            tasks, corpus, None, proof, SimpleNamespace(check=lambda: None),
                            manifest, results/"model_rollout_run.json")
    metrics = json.loads((results/(name+"_results.json")).read_bytes())
    assert metrics["combined"]["n_tasks"] == 1494 and metrics["combined"]["n_steps"] == 5976
    assert all(c["metrics"]["n_tasks"] == 249 for c in metrics["configs"].values())
    assert manifest["strategies"][name]["matched_task05_step0_inputs"] == 1494
    assert manifest["strategies"][name]["status"] == "complete"
    with (output/name/"steps.jsonl").open() as stream:
        assert sum(json.loads(line)["generation_metadata"]["runtime_prompt_ids_match"] for line in stream) == 5976
