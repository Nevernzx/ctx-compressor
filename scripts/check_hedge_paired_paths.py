"""Frozen-sample validation of the shared-prefix production scorer against independent greedy decoding."""
from pathlib import Path
import json
import math
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_hedge_scores import DETAILS, N, OUT, SEED, STEP_SHA, check_row, sample
from scripts.hedge_scorer import WEIGHT_SHA, create_engine, gate_gpu, score_candidates, token_values
from scripts.merge_sft_lora import sha256
from src.model_inference import MAX_NEW_TOKENS, SEED as EVAL_SEED, compact


def main():
    started=time.monotonic()
    initial=OUT.with_name("score_checks_initial.json")
    prior=OUT.with_name("score_production_checks_separate_paths.json")
    cross=OUT.with_name("score_production_checks_shared_prefix_crossforward.json")
    prod=OUT.with_name("score_production_checks.json")
    rows_path=DETAILS/"paired_shared_prefix_scores.jsonl"
    if not initial.is_file() or not prior.is_file() or not cross.is_file() or any(p.exists() for p in (prod,OUT,rows_path)):
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
                audit=[]
                scores=score_candidates(llm,adapter,prompts,candidates,audit=audit)
                if len(audit)!=len(part):
                    raise ValueError("Missing independent references")
                for r,values,(first,second,branches) in zip(part,scores,audit):
                    independent=second.outputs[0]
                    record=check_row(r,values,independent,tokenizer)
                    primary=first.outputs[0]
                    record.update(primary_request_id=first.request_id, reference_request_id=second.request_id,
                                  primary_raw=primary.text,primary_ids=list(primary.token_ids),
                                  primary_raw_token_logprobs=token_values(primary),
                                  reference_raw_token_logprobs=token_values(independent))
                    if set(values)!=set(branches):
                        raise ValueError("Missing candidate probability audit")
                    for cid,(tokens,forced_lp) in branches.items():
                        total=[]
                        matches=True
                        for j,target in enumerate(tokens):
                            matches = matches and j < len(primary.token_ids) and primary.token_ids[j]==target
                            total.append(record["primary_raw_token_logprobs"][j] if matches else forced_lp[j])
                        if math.fsum(total)!=values[cid]:
                            raise ValueError("Independent full candidate score reconstruction mismatch")
                    record["forced_token_details"]=branches
                    stream.write(compact(record)+"\n")
                    checked.append(record)
                stream.flush()
                print(f"Paired production scorer checks {len(checked)}/{N}",flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()
    n=len(checked)
    def count(key):
        return sum(r[key] for r in checked)
    distinct=len({r[k] for r in checked for k in ("primary_request_id","reference_request_id")})
    if distinct!=2*n:
        raise ValueError("Repeated request IDs in independent diagnostic")
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
              "n_candidate_scores_reconstructed":sum(len(r["scores"]) for r in checked),"independent_request_ids":distinct,"verifier_path":str(Path(__file__).resolve()),
             "method":"Production submits two independent, unconstrained, identical greedy requests together. Only the first contributes shared-prefix raw log probabilities to candidate scores; each divergent suffix is teacher forced. The second supplies separate raw log probabilities, generated token IDs and terminal-inclusive text for the check. The ordinary Task 06 full greedy run and separately scheduled checks are preserved in the prior diagnostics.",
             "details_path":str(rows_path),"details_sha256":sha256(rows_path),"wall_seconds":time.monotonic()-started,
             "prior_diagnostics":[{"path":str(p),"sha256":sha256(p),"status":json.loads(p.read_bytes())["status"],
                 "argmax":json.loads(p.read_bytes())["argmax"],"logprob":json.loads(p.read_bytes())["logprob"],
                 "reencode":json.loads(p.read_bytes())["reencode"]} for p in (initial,prior,cross)],
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

