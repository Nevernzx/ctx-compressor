import json
from pathlib import Path

import pytest

from scripts import build_grpo_states as states
from scripts.probe_grpo_sampling import stratified, summarize
from src.heldout_eval import BUDGETS, MODES, HeldoutReplay
from src.model_inference import inference_prompt
from src.build_traj import render_input


class ByteTokenizer:
    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return list(messages[0]["content"].encode())


def test_step0_comparison_preserves_every_byte_except_budget_header():
    original = "任务问题：\n1. Explain 1876 token.\n模式：append\n预算：1876 token\n已用 token：0\n\n已保留：\n（空）"
    target = original.replace("\n预算：1876", "\n预算：2814")
    assert states.check_sft(original, "tight", original, "tight") == {"direct": True, "matches": True}
    assert states.check_sft(target, "tight", original, "mid") == {"direct": False, "matches": True}
    assert not states.check_sft(target.replace("Explain 1876", "Explain 2814"), "tight", original, "mid")["matches"]
    with pytest.raises(ValueError):
        states.check_sft(target, "tight", original + original, "mid")


def test_training_task_gate_and_wrong_gpu(monkeypatch):
    with pytest.raises(ValueError):
        states.require_train_file(Path("/tmp/heldout_tasks.jsonl"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    with pytest.raises(RuntimeError):
        states.check_gpu(0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    states.check_gpu(0)
    with pytest.raises(RuntimeError):
        states.check_gpu(1)


def test_closed_loop_state_audit_uses_frozen_corpus_and_lost_gold(tmp_path, monkeypatch):
    tasks = [{"task":9, "questions":["Q"], "split":"train", "steps":[
        {"gold":0, "new":[0,1], "hop_index":0, "question_index":0},
        {"gold":2, "new":[2], "hop_index":1, "question_index":0}]}]
    corpus = [{"pid":i,"text":f"p{i}","ntok":3} for i in range(3)]
    replay = HeldoutReplay(tasks,corpus,"test")
    tokenizer = ByteTokenizer()
    rows = []
    refs = {}
    while not replay.completed():
        items = replay.next_batch()
        outputs = []
        for item in items:
            prompt = inference_prompt(tokenizer, item["state"])
            # At step 0 keep only the non-gold lid, at step 1 force an invalid action.
            raw = '{"add":[1]}' if item["step"] == 0 else '{"add":[77]}'
            outputs.append({**prompt, "raw_output":raw, "output_token_ids":[77]})
            if item["step"] == 0 and item["budget_tier"] == "mid":
                refs[item["task"],item["mode"]] = ("mid",render_input(item["state"]))
        records = replay.advance_batch(outputs)
        for i,o,r in zip(items, outputs, records):
            rows.append(states.make_row(i,o,r,{"model_output_label":"unit test"}))
    monkeypatch.setattr(states,"read_sft_references",lambda tasks:refs)
    p = tmp_path / "states.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = states.verify_rows(p,tasks,corpus,tokenizer)
    assert summary["n_trajectories"] == 6 and summary["n_states"] == 12
    assert summary["gold_mismatches"] == summary["reencode_mismatches"] == summary["rerender_mismatches"] == 0
    assert summary["step0_sft"]["direct_n"] == 2 and summary["step0_sft"]["budget_substituted_n"] == 4
    assert all(summary["pooled"][k] == {"count":6,"n":12,"rate":0.5}
               for k in ("invalid","kept_non_gold","lost_gold","prior_gold_missing_from_kept"))
    for r in rows:
        assert json.loads(r["reward_state_json"])["prompt_token_ids_tail"] == r["prompt_token_ids"][-24:]


def test_stratification_reproducible_distinct_and_train_only():
    pool = [{"split":"train", "mode":mode,"budget_tier":tier,"index":i} for mode in MODES for tier in BUDGETS for i in range(10)]
    sample = stratified(pool,3,7)
    assert sample == stratified(pool,3,7)
    assert sample != stratified(pool,3,8)
    assert len({(s["mode"],s["budget_tier"],s["index"]) for s in sample}) == 18
    with pytest.raises(ValueError):
        stratified([dict(pool[0],split="dev"),*pool],3,7)


def test_probe_denominators_std_and_signed_gain():
    g1 = {"greedy_reward":1.2, "samples":[{"reward":1.0,"invalid_reason":None}]*2}
    g2 = {"greedy_reward":.2, "samples":[{"reward":-1.0,"invalid_reason":"invalid_json"},{"reward":1.0,"invalid_reason":None}]}
    s = summarize([g1,g2])
    assert s["n_groups"] == 2 and s["n_completions"] == 4
    assert s["invalid_by_category"]["invalid_json"] == {"count":1,"n":4,"rate":.25}
    assert s["zero_std_groups"] == s["some_sample_beats_greedy"] == {"count":1,"n":2,"rate":.5}
    assert s["mean_best_minus_greedy"] == pytest.approx(.3)
    assert s["mean_within_group_std"] == pytest.approx(.5)
