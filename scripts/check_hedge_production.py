"""Verify the shipping scorer against unconstrained vLLM generation on the frozen 300 states."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_hedge_scores import DETAILS, N, OUT, SEED, STEP_SHA, check_row, sample
from scripts.hedge_scorer import (WEIGHT_SHA, _evaluate, continuation, create_engine, gate_gpu,
                                  natural_prefixes, raw_logprob, score_candidates)
from scripts.merge_sft_lora import sha256
from src.model_inference import MAX_NEW_TOKENS, SEED as EVAL_SEED, compact


def main():
    first = OUT.with_name("score_checks_initial.json")
    prod = OUT.with_name("score_production_checks.json")
    rows_path = DETAILS / "production_scores.jsonl"
    if not first.is_file() or any(p.exists() for p in (OUT, prod, rows_path)):
        raise FileExistsError("Archive the initial measurement and keep the final output paths fresh")
    gpu = gate_gpu()
    started = time.monotonic()
    rows = sample()
    old = json.loads(first.read_bytes())
    frozen = json.loads((DETAILS / "sample.json").read_bytes())
    if (old["sample_sha256"] != sha256(DETAILS / "sample.json")
            or frozen["sample"] != [[r["trajectory_id"],r["step"],r["generation_metadata"]["prompt_sha256"]]
                                    for r in rows]):
        raise ValueError("Frozen sample differs")
    llm, adapter, params = create_engine()
    from vllm import SamplingParams
    tokenizer = llm.get_tokenizer()
    full_greedy = SamplingParams(temperature=0, seed=EVAL_SEED, max_tokens=MAX_NEW_TOKENS, logprobs=0,
                                 skip_special_tokens=False, include_stop_str_in_output=True)
    checked = []
    try:
        with rows_path.open("x") as stream:
            for start in range(0,N,16):
                part = rows[start:start+16]
                prompts = [r["generation_metadata"]["prompt_token_ids"] for r in part]
                candidates = [[c for c in r["state"]["new_ids"] if c not in r["state"]["kept_ids"]] for r in part]
                full = _evaluate(llm,adapter,prompts,full_greedy)
                natural_ids = []
                for output in full:
                    text = tokenizer.decode(output.token_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
                    ident, = json.loads(text)["add"]
                    natural_ids.append(ident)
                actual = score_candidates(llm,adapter,prompts,candidates)
                refs = natural_prefixes(llm,adapter,prompts,candidates,natural_ids)
                for row, cid, scores, whole, prefix in zip(part,natural_ids,actual,full,refs):
                    record = check_row(row,scores,whole,tokenizer)
                    ids = continuation(tokenizer,cid)
                    natural = raw_logprob(prefix,ids)
                    if list(prefix.token_ids)!=ids:
                        raise ValueError(f"Unconstrained prefix disagrees with full generation for {row['trajectory_id']}")
                    record.update(exact_prefix_ids=ids,unforced_vllm_prefix_score=natural,
                                  production_abs_error=abs(scores[cid]-natural),
                                  production_score=scores[cid])
                    stream.write(compact(record)+"\n")
                    checked.append(record)
                stream.flush()
                print(f"Production score checks {len(checked)}/{N}",flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()
    n = len(checked)
    def count(key):
        return sum(row[key] for row in checked)
    matched = sum(r["argmax"]==r["archived_id"] for r in checked)
    prod_errors = [r["production_abs_error"] for r in checked]
    full_errors = [r["abs_error"] for r in checked]
    def errors(values):
        return {"n":n,"within_tolerance":sum(v<=0.02 for v in values),
                "max_abs_error":max(values),"tolerance":0.02,"units":"nats"}
    rec = {"n":n,"text_token_matches":count("archived_text_reencodes_body_ids"),
           "strict_full_id_matches":count("archived_text_reencodes_all_ids"),
           "terminal_eos_count":count("one_terminal_eos"),
           "full_with_terminal_matches":count("body_plus_observed_terminal_reencodes_all_ids"),
           "returned_raw_full_id_matches":count("returned_diagnostic_raw_full_id_match"),
           "task06_raw_skip_special_tokens":True,"diagnostic_skip_special_tokens":False,
           "diagnostic_include_stop_str_in_output":True,"terminal_eos_id":tokenizer.eos_token_id,
           "terminal_eos_text":tokenizer.eos_token}
    ok = (n==N and matched/n>=0.99 and max(prod_errors)<=0.02 and rec["text_token_matches"]==n
          and rec["full_with_terminal_matches"]==n and rec["returned_raw_full_id_matches"]==n)
    summary = {"status":"ok" if ok else "failed", "sample_n":n,"sample_seed":SEED,
               "valid_population":13926,"source_steps_sha256":STEP_SHA,"sample_sha256":old["sample_sha256"],
               "gpu_before":gpu,"engine_parameters":params,"adapter_model_sha256":WEIGHT_SHA,
               "argmax":{"n":n,"matched":matched,"rate":matched/n,"required_rate":0.99},
               "logprob":errors(prod_errors),"reencode":rec,
               "reference":"Independent unconstrained vLLM greedy requests limited to the target prefix length, at the original production queue position and batch shape. Compared with scores from unmodified score_candidates.",
               "full_generation_other_batch_shape_logprob":errors(full_errors),
               "regenerated_raw_matches_task06":count("new_text_matches_archived"),
               "regenerated_full_ids_match_task06":count("new_ids_match_archived"),
               "details_path":str(rows_path), "details_sha256":sha256(rows_path),
               "wall_seconds":time.monotonic()-started,
               "initial_check":{"path":str(first),"sha256":sha256(first),"status":old["status"],
                                "logprob":old["logprob"],"reencode":old["reencode"],
                                "note":"Initial request placed the forced and natural sequences next to one another with different maximum lengths; default stop-token exclusion hides terminal EOS even when skip_special_tokens=False."}}
    with prod.open("x") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
        f.write("\n")
    with OUT.open("x") as f:
        json.dump(dict(summary,production_receipt=str(prod),production_receipt_sha256=sha256(prod)),
                  f,ensure_ascii=False,indent=2)
        f.write("\n")
    print(compact(summary),flush=True)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

