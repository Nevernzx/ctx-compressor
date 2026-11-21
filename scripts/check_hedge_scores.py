"""Three pre-registered Task 09 scorer checks on a fixed random draw of 300 valid Task 06 states."""
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hedge_scorer import (ADAPTER, EXTERNAL, PROJECT, WEIGHT_SHA, continuation,
                                  create_engine, gate_gpu, raw_logprob, score_with_reference)
from scripts.merge_sft_lora import sha256
from src.model_inference import compact, sha256_json

SEED = 20261121
N = 300
STEPS = EXTERNAL / "eval_rollouts_v2/checkpoint-2000/steps.jsonl"
STEP_SHA = "d1b0de03f70b8ffc3dbc5535c9b5166c50f31c01c766a0981854439483aa0751"
OUT = PROJECT / "eval/hedge_v2/score_checks.json"
DETAILS = EXTERNAL / "hedge_v2/implementation"


def sample():
    # Draw from all action-valid steps, without conditioning on candidate rank or confidence.
    valid = []
    with STEPS.open("rb") as stream:
        digest = hashlib.sha256()
        for line in stream:
            digest.update(line)
            if json.loads(line)["action_valid"]:
                valid.append(stream.tell() - len(line))
        if digest.hexdigest() != STEP_SHA or len(valid) != 13926:
            raise ValueError("Task 06 evaluation data differ from the registered input")
        selected = random.Random(SEED).sample(valid, N)
        rows = []
        for offset in selected:
            stream.seek(offset)
            row = json.loads(stream.readline())
            prompt = row["generation_metadata"]
            if sha256_json(prompt["prompt_token_ids"]) != prompt["prompt_sha256"]:
                raise ValueError("Archived inference prompt failed checksum")
            rows.append(row)
    return rows


def check_row(row, scores, output, tokenizer):
    gold_id, = row["parsed_action"]["add"]
    suffix_ids = continuation(tokenizer, gold_id)
    archived = row["generation_metadata"]
    generated = list(output.token_ids)
    eos = tokenizer.eos_token_id
    archived_ids = archived["output_token_ids"]
    terminal = bool(archived_ids and archived_ids[-1] == eos and archived["finish_reason"] == "stop")
    old_text_ids = tokenizer.encode(row["raw_output"], add_special_tokens=False)
    body = list(archived_ids[:-1] if terminal else archived_ids)
    visible = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # skip_special_tokens=False asks vLLM itself to return the actual serialized EOS token.
    reference_ids = tokenizer.encode(output.text, add_special_tokens=False)
    current_id, = json.loads(visible)["add"]
    generation_score = raw_logprob(output, continuation(tokenizer, current_id))
    if gold_id != current_id:
        # Pairing is fixed using archived id only; a replay difference cannot be dropped.
        error = abs(scores[current_id] - generation_score)
    else:
        error = abs(scores[gold_id] - generation_score)
    return {"trajectory_id": row["trajectory_id"], "step":row["step"], "prompt_sha256":archived["prompt_sha256"],
            "archived_id":gold_id, "new_greedy_id":current_id, "argmax":max(scores,key=scores.get),
            "scores":scores, "checked_tokens":len(continuation(tokenizer,current_id)),
            "forced_score":scores[current_id], "generation_score":generation_score, "abs_error":error,
            "archived_text_reencodes_all_ids":old_text_ids==archived_ids,
            "archived_text_reencodes_body_ids":old_text_ids==body,
            "one_terminal_eos":terminal,
            "body_plus_observed_terminal_reencodes_all_ids":
                    tokenizer.encode(row["raw_output"] + (tokenizer.eos_token if terminal else ""),
                                     add_special_tokens=False) == archived_ids,
            "returned_diagnostic_raw_full_id_match":reference_ids==generated,
            "new_text_matches_archived":visible==row["raw_output"],
            "new_ids_match_archived":generated==archived_ids,
            "diagnostic_raw":output.text, "task06_raw":row["raw_output"],
            "diagnostic_ids":generated, "task06_ids":archived_ids}


def main():
    if OUT.exists() or DETAILS.exists():
        raise FileExistsError("Scorer check outputs already exist")
    gpu = gate_gpu()
    rows = sample()
    ids = [r["parsed_action"]["add"] for r in rows]
    if any(len(a) != 1 for a in ids):
        raise ValueError("Unexpected non-singleton archived greedy add list in fixed sample")
    # Fix and persist the sample and the algorithm before loading the model.
    DETAILS.mkdir()
    with (DETAILS/"sample.json").open("x") as f:
        json.dump({"seed":SEED, "population":13926, "step_file_sha256":STEP_SHA,
                   "sample":[[r["trajectory_id"],r["step"],r["generation_metadata"]["prompt_sha256"]]
                             for r in rows]}, f)
    started = time.monotonic()
    llm, adapter, params = create_engine()
    tokenizer = llm.get_tokenizer()
    checked = []
    try:
        path = DETAILS / "scores.jsonl"
        with path.open("x") as stream:
            for start in range(0, len(rows), 16):
                part = rows[start:start+16]
                prompts = [r["generation_metadata"]["prompt_token_ids"] for r in part]
                candidates = [[c for c in r["state"]["new_ids"] if c not in r["state"]["kept_ids"]] for r in part]
                scores, outputs = score_with_reference(llm, adapter, prompts, candidates,
                                                       [r["parsed_action"]["add"][0] for r in part])
                for r, s, o in zip(part, scores, outputs):
                    evidence = check_row(r, s, o, tokenizer)
                    stream.write(compact(evidence)+"\n")
                    checked.append(evidence)
                stream.flush()
                print(f"Score checks {len(checked)}/{N}", flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()

    def count(key):
        return sum(row[key] for row in checked)
    n = len(checked)
    matches = sum(r["argmax"]==r["archived_id"] for r in checked)
    near = sum(r["abs_error"]<=0.02 for r in checked)
    recon = {"n":n,"text_token_matches":count("archived_text_reencodes_body_ids"),
             "strict_full_id_matches":count("archived_text_reencodes_all_ids"),
             "terminal_eos_count":count("one_terminal_eos"),
             "full_with_terminal_matches":count("body_plus_observed_terminal_reencodes_all_ids"),
             "returned_raw_full_id_matches":count("returned_diagnostic_raw_full_id_match"),
             "diagnostic_raw_skip_special_tokens":False,
             "task06_raw_skip_special_tokens":True,"terminal_eos_id":tokenizer.eos_token_id,
             "terminal_eos_text":tokenizer.eos_token}
    ok = (n==N and matches/n>=0.99 and near==n and recon["text_token_matches"]==n
          and recon["full_with_terminal_matches"]==n and recon["returned_raw_full_id_matches"]==n)
    result = {"status":"ok" if ok else "failed", "sample_n":n,"sample_seed":SEED,
              "valid_population":13926,"sample_sha256":sha256(DETAILS/"sample.json"),"source_steps_sha256":STEP_SHA,
              "adapter":str(ADAPTER),"adapter_model_sha256":WEIGHT_SHA,"gpu_before":gpu,"engine_parameters":params,
              "method":"vLLM raw log-softmax before request-local teacher forcing, exact frozen prompt IDs",
              "argmax":{"n":n,"matched":matches,"rate":matches/n,"required_rate":0.99},
              "logprob":{"n":n,"within_tolerance":near,"max_abs_error":max(r["abs_error"] for r in checked),
                         "tolerance":0.02,"units":"nats"},
              "reencode":recon,"regenerated_raw_matches_task06":count("new_text_matches_archived"),
              "regenerated_full_ids_match_task06":count("new_ids_match_archived"),
              "details_path":str(path),"details_sha256":sha256(path),"wall_seconds":time.monotonic()-started}
    with OUT.open("x") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write("\n")
    print(compact(result),flush=True)
    if not ok:
        raise SystemExit(2)


if __name__=="__main__":
    main()

