"""Task 10 C. Closed-loop Task 09 hedge on Task 08's frozen training grid."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import build_formal_grpo_states as formal, hedge_scorer as scorer
from scripts.build_grpo_states import lines, sha256, file_record, TRAIN_CORPUS
from scripts.eval_hedge_v2 import wrap_batch, run_tau
from scripts.hedge_action import candidate_ids
from scripts.sample_sft_subset import require, write_json_exclusive
from src.build_traj import render_input
from src.heldout_eval import HeldoutReplay, apply_model_output
from src.model_inference import inference_prompt, guard_prompt, generation_record, compact, SEED, MAX_NEW_TOKENS

EXTERNAL = scorer.EXTERNAL
ROOT = EXTERNAL / "dagger_v3"
SMALL = PROJECT / "eval/sft_run_v3"
PRECHECK = SMALL / "dagger_precheck.json"
SMOKE = SMALL / "dagger_smoke.json"
SUMMARY = SMALL / "dagger_rollouts.json"
CAPTURE = ROOT / "score_logits"
FROZEN = formal.ROOT / "states/state_checks.json"
FROZEN_SHA = "216e77f5edbdf6f7bb1af2aff2a25ee24771063089e86e9215f84af58ffd3b42"
SELECTION = PROJECT / "eval/hedge_v2/train_selection.json"
SELECTION_SHA = "a0f38184b25d9b29064ed5d61976ad3de1a5e3f184e675b4afc64e0861ca33ae"
POLICY = "sft_v2_checkpoint_2000_tau_0.99"
TAU = 0.99


def load(path):
    return json.loads(path.read_bytes())


def frozen_inputs():
    require(sha256(FROZEN) == FROZEN_SHA and sha256(SELECTION) == SELECTION_SHA, "Frozen C receipts changed")
    source, selection = load(FROZEN), load(SELECTION)
    require(source["state_file"] == selection["state_file"] and source["available_checks_pass"]
            and selection["split"] == "train" and run_tau("tau_star", selection) == TAU
            and source["source_adapter_sha256"] == selection["adapter_sha256"] == scorer.WEIGHT_SHA,
            "Task 08/09 provenance or tau mismatch")
    return source, selection


def precheck():
    require(not PRECHECK.exists(), "C precheck receipt exists")
    source, selection = frozen_inputs()
    # Independently regenerate the Task 08 seeded list from train source order.
    population = []
    for row in lines(formal.TRAIN):
        require(row["split"] == "train", "Unexpected train split")
        population.append(row["task"])
    require(len(population) == len(set(population)) == 4735, "Unexpected train population")
    selected = random.Random(formal.old.SEED).sample(population, 400)
    require(selected == source["task_ids"] and len(set(selected)) == 400, "Task 08 selection mismatch")
    overlap = formal.train_overlap(formal.TRAIN, formal.HELDOUT, selected)
    require(overlap["heldout_overlap"] == 0 and overlap["train_example_ids"] == 1600
            and overlap["heldout_example_ids"] == 996, "Unexpected question identity counts")
    actual = file_record(formal.TRAIN)
    require(actual["sha256"] == selection["train_task_file"]["sha256"], "Train source hash changed")
    SMALL.mkdir(parents=True, exist_ok=True)
    result = {"status":"complete","task_ids":selected,"n_tasks":400,"seed":formal.old.SEED,
              "task08_id_list_equal":True,**overlap,"train_file":actual,"corpus_file":file_record(TRAIN_CORPUS),
              "source_receipt":file_record(FROZEN),"hedge_selection":file_record(SELECTION)}
    write_json_exclusive(PRECHECK, result)
    print(json.dumps({k:v for k,v in result.items() if k != "task_ids"}), flush=True)


def train_assets(ids):
    keep = set(ids)
    selected = {}
    for row in lines(formal.TRAIN):
        task = row["task"]
        if task in keep:
            require(row["split"] == "train" and task not in selected, "Ambiguous training task")
            selected[task] = {k:row[k] for k in ("task","questions","steps","split")}
    require(set(selected) == keep, "Missing train task")
    corpus = list(lines(TRAIN_CORPUS))
    require(all(row["pid"] == i for i,row in enumerate(corpus)), "Corpus ordering changed")
    return [selected[t] for t in ids], corpus


def create_engine():
    # Invoke the frozen initializer; the worker class only redirects its capture file.
    import vllm
    original = vllm.LLM

    def with_task10_capture(**kwargs):
        require(kwargs.get("logits_processors") == ["scripts.hedge_raw_logits:CaptureCandidateLogits"],
                "Task 09 logit processor changed")
        kwargs["logits_processors"] = ["scripts.dagger_v3_logits:CaptureCandidateLogits"]
        return original(**kwargs)

    with patch.object(vllm, "LLM", with_task10_capture):
        return scorer.create_engine()


def complete_row(row, item):
    state = item["state"]
    row.update(split="train", state_id=f"{item['trajectory_id']}:{item['step']}",
               user_text=render_input(state), greedy_keep_after=apply_model_output(state,row["greedy_raw_output"])["keep_after"])
    row["state"]["text"] = state["text"]
    expected = hashlib.sha256(row["user_text"].encode()).hexdigest()
    require(row["generation_metadata"]["user_text_sha256"] == expected, "Stored user input differs from generation")
    return row


def rollout(smoke):
    from vllm import SamplingParams
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
            "GPU 1 only")
    require(Path(sys.prefix).resolve() == EXTERNAL/"envs/grpo", "Use the pinned grpo environment")
    source, selection = frozen_inputs()
    check = load(PRECHECK)
    require(check["task08_id_list_equal"] and check["heldout_overlap"] == 0 and check["task_ids"] == source["task_ids"]
            and check["status"] == "complete" and check["train_file"]["sha256"] == sha256(formal.TRAIN)
            and check["corpus_file"]["sha256"] == sha256(TRAIN_CORPUS), "C precheck inputs changed")
    ids = check["task_ids"][:2] if smoke else check["task_ids"]
    receipt = SMOKE if smoke else SUMMARY
    if not smoke:
        prior = load(SMOKE)
        require(prior["status"] == "complete" and prior["n_tasks"] == 2 and prior["n_trajectories"] == 12
                and prior["steps_file"]["sha256"] == sha256(ROOT/"smoke/steps.jsonl"), "C smoke validation missing")
    raw = ROOT / ("smoke" if smoke else "rollouts")
    require(not receipt.exists() and not raw.exists() and raw.resolve() == raw, "Existing rollout destination")
    raw.mkdir()
    tasks, corpus = train_assets(ids)
    replay = HeldoutReplay(tasks, corpus, POLICY)
    expected = 6 * sum(len(t["steps"]) for t in tasks)
    require(expected == source["n_states"] if not smoke else expected > 0, "Unexpected state count")
    gpu = scorer.gate_gpu()
    llm, adapter, params = create_engine()
    require(params == selection["engine_parameters"] and Path(adapter.lora_path) == scorer.ADAPTER, "Wrong engine or adapter")
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, seed=SEED)
    count, cells, seen, hedges = 0, Counter(), set(), 0
    began = time.monotonic()
    with patch.object(scorer, "capture_file", lambda llm: CAPTURE/(llm._hedge_logit_session+".jsonl")), \
            (raw/"steps.jsonl").open("x", encoding="utf-8") as out:
        while not replay.completed():
            items = replay.next_batch()
            prompts = [inference_prompt(tokenizer, item["state"]) for item in items]
            for prompt,item in zip(prompts,items):
                guard_prompt(prompt,params["max_model_len"],identity=(item["trajectory_id"],item["step"]))
            outputs = llm.generate([{"prompt_token_ids":p["prompt_token_ids"]} for p in prompts],
                                   sampling_params=sampling,lora_request=adapter,use_tqdm=False)
            require(len(outputs)==len(items), "Incomplete greedy batch")
            generated = [generation_record(p,o) for p,o in zip(prompts,outputs)]
            scores = []
            for start in range(0,len(items),32):
                batch = items[start:start+32]
                scores.extend(scorer.score_candidates(llm, adapter, [p["prompt_token_ids"] for p in prompts[start:start+32]],
                                                      [candidate_ids(i["state"]) for i in batch]))
                if start and start % 320 == 0:
                    print(f"step={items[0]['step']} scored={start+len(batch)}/{len(items)}", flush=True)
            for row,item in zip(wrap_batch(replay,items,generated,scores,TAU),items):
                complete_row(row,item)
                require(row["state_id"] not in seen, "Duplicate state identity")
                seen.add(row["state_id"])
                count += 1
                cells[row["mode"]+"|"+row["budget_tier"]] += 1
                hedges += row["hedge"]["hedged"]
                out.write(compact(row)+"\n")
            out.flush()
            os.fsync(out.fileno())
            print(f"completed step={items[0]['step']} rows={count}/{expected}", flush=True)
    require(count == len(seen) == expected and len(cells) == 6 and len(set(cells.values())) == 1, "Incomplete rollout")
    data = {"status":"complete","n_tasks":len(ids),"n_trajectories":len(replay.trajectories),"n_steps":count,"cells":dict(cells),
            "n_hedged":hedges,"task_ids":ids,"tau":TAU,"adapter_sha256":scorer.WEIGHT_SHA,"engine_parameters":params,
            "gpu_before":gpu,"precheck":file_record(PRECHECK),"elapsed_seconds":time.monotonic()-began,
            "steps_file":file_record(raw/"steps.jsonl"),"capture_path":str(CAPTURE/(llm._hedge_logit_session+".jsonl"))}
    write_json_exclusive(receipt,data)
    print(f"{receipt} steps={count} hedged={hedges} sha256={data['steps_file']['sha256']}",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["precheck","smoke","run"])
    stage = p.parse_args().stage
    require(PROJECT == Path("/root/workplace/ctx-compressor"), "Run on 5880")
    if stage == "precheck":
        precheck()
    else:
        rollout(stage == "smoke")

