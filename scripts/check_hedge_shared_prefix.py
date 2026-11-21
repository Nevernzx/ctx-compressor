"""Frozen-sample validation of the shared-prefix production scorer against independent greedy decoding."""
from pathlib import Path
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_hedge_scores import DETAILS, N, OUT, SEED, STEP_SHA, check_row, sample
from scripts.hedge_scorer import WEIGHT_SHA, create_engine, gate_gpu, score_candidates
from scripts.merge_sft_lora import sha256
from src.model_inference import MAX_NEW_TOKENS, SEED as EVAL_SEED, compact


def independent_greedy(llm, adapter, ids):
    from vllm import SamplingParams
    prompt = [{"prompt_token_ids":ids}]
    prime = SamplingParams(temperature=0,max_tokens=1,seed=EVAL_SEED)
    llm.generate(prompt,prime,lora_request=adapter,use_tqdm=False)
    greedy = SamplingParams(temperature=0,max_tokens=MAX_NEW_TOKENS,seed=EVAL_SEED,logprobs=0,
                            skip_special_tokens=False,include_stop_str_in_output=True)
    result, = llm.generate(prompt,greedy,lora_request=adapter,use_tqdm=False)
    if list(result.prompt_token_ids)!=ids or len(result.outputs)!=1:
        raise ValueError("Incorrect independent reference input")
    return result.outputs[0]


def main():
    started=time.monotonic()
    initial=OUT.with_name("score_checks_initial.json")
    prior=OUT.with_name("score_production_checks_separate_paths.json")
    prod=OUT.with_name("score_production_checks.json")
    rows_path=DETAILS/"shared_prefix_scores.jsonl"
    if not initial.is_file() or not prior.is_file() or any(p.exists() for p in (prod,OUT,rows_path)):
        raise FileExistsError("Preserve previous receipts and use fresh output paths")
    gpu=gate_gpu()
    rows=sample()
    frozen=json.loads((DETAILS/"sample.json").read_bytes())
    sample_hash=sha256(DETAILS/"sample.json")
    if (json.loads(initial.read_bytes())["sample_sha256"]!=sample_hash or
        frozen["sample"]!=[[r["trajectory_id"],r["step"],r["generation_metadata"]["prompt_sha256"]] for r in rows]):
        raise ValueError("Frozen sample mismatch")
    llm,adapter,params=create_engine()
    tokenizer=llm.get_tokenizer()
    checked=[]
    try:
        with rows_path.open("x") as stream:
            for start in range(0,len(rows),16):
                part=rows[start:start+16]
                prompts=[r["generation_metadata"]["prompt_token_ids"] for r in part]
                candidates=[[c for c in r["state"]["new_ids"] if c not in r["state"]["kept_ids"]] for r in part]
                scores=score_candidates(llm,adapter,prompts,candidates)
                for r,ids,values in zip(part,prompts,scores):
                    independent=independent_greedy(llm,adapter,ids)
                    record=check_row(r,values,independent,tokenizer)
                    stream.write(compact(record)+"\n")
                    checked.append(record)
                stream.flush()
                print(f"Shared prefix scorer checks {len(checked)}/{N}",flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()
    n=len(checked)
    def count(key):
        return sum(r[key] for r in checked)
    matched=sum(r["argmax"]==r["archived_id"] for r in checked)
    errors=[r["abs_error"] for r in checked]
    rec={"n":n,"text_token_matches":count("archived_text_reencodes_body_ids"),
         "strict_full_id_matches":count("archived_text_reencodes_all_ids"),"terminal_eos_count":count("one_terminal_eos"),
         "full_with_terminal_matches":count("body_plus_observed_terminal_reencodes_all_ids"),
         "returned_raw_full_id_matches":count("returned_diagnostic_raw_full_id_match"),
         "diagnostic_skip_special_tokens":False,"diagnostic_include_stop_str_in_output":True,
         "task06_skip_special_tokens":True,"terminal_eos_id":tokenizer.eos_token_id,"terminal_eos_text":tokenizer.eos_token}
    ok=(n==N and matched/n>=0.99 and max(errors)<=0.02 and rec["text_token_matches"]==n
        and rec["full_with_terminal_matches"]==n and rec["returned_raw_full_id_matches"]==n)
    summary={"status":"ok" if ok else "failed","sample_n":n,"sample_seed":SEED,"valid_population":13926,
             "source_steps_sha256":STEP_SHA,"sample_sha256":sample_hash,"gpu_before":gpu,"engine_parameters":params,
             "adapter_model_sha256":WEIGHT_SHA,"argmax":{"n":n,"matched":matched,"rate":matched/n,"required_rate":0.99},
             "logprob":{"n":n,"within_tolerance":sum(v<=0.02 for v in errors),"max_abs_error":max(errors),
                        "tolerance":0.02,"units":"nats"},"reencode":rec,
             "regenerated_raw_matches_task06":count("new_text_matches_archived"),
             "regenerated_full_ids_match_task06":count("new_ids_match_archived"),
             "method":"All candidates reuse the raw log probabilities on their longest token prefix shared with a natural singleton decode, then use raw teacher-forced log probabilities through the end of their divergent branch. Independent full greedy generation provides the check and terminal-inclusive raw text.",
             "details_path":str(rows_path),"details_sha256":sha256(rows_path),"wall_seconds":time.monotonic()-started,
             "prior_diagnostics":[{"path":str(p),"sha256":sha256(p),"status":json.loads(p.read_bytes())["status"],
                 "argmax":json.loads(p.read_bytes())["argmax"],"logprob":json.loads(p.read_bytes())["logprob"],
                 "reencode":json.loads(p.read_bytes())["reencode"]} for p in (initial,prior)],
             "source_sha256":{name:sha256(PROJECT_SCRIPT) for name,PROJECT_SCRIPT in (
                ("scorer",Path(__file__).with_name("hedge_scorer.py")),("check",Path(__file__)),
                ("processor",Path(__file__).with_name("hedge_forced_tokens.py")))}}
    with prod.open("x") as stream:
        json.dump(summary,stream,ensure_ascii=False,indent=2)
        stream.write("\n")
    with OUT.open("x") as stream:
        json.dump(dict(summary,production_receipt=str(prod),production_receipt_sha256=sha256(prod)),
                  stream,ensure_ascii=False,indent=2)
        stream.write("\n")
    print(compact(summary),flush=True)
    if not ok:
        raise SystemExit(2)


if __name__=="__main__":
    main()

