"""Time the exact Task 17 states with the two compressor engines and trained reranker."""
import argparse
from collections import Counter
from importlib.metadata import version
from itertools import chain
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import reranker_v1_data as data, train_reranker_v1 as fit
from scripts import run_reader_heldout as reader
from src.reader_heldout_inputs import rows, write_new
from src.reranker_v1 import DATA, MODEL, SMALL, WORK, PairTokenizer, distribution, need, read, record, require_new, verify

N_WARM=20
N_STATES=300


class GPU0Background:
    """The only GPU 0 operation in Task 17 is read-only latency context."""
    def __init__(self,path):
        self.path=path
        self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.error=None

    def run(self):
        try:
            with self.path.open("x",encoding="utf-8",buffering=1) as out:
                while True:
                    raw=subprocess.check_output(["nvidia-smi","-i","0",
                          "--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],
                          text=True,timeout=10).strip()
                    index,memory,utilization=map(int,raw.split(","))
                    need(index==0,"invalid background GPU telemetry")
                    out.write(data.compact({"utc":reader.now(),"index":index,"memory_used_mib":memory,
                                            "utilization_percent":utilization})+"\n")
                    if self.stop_event.wait(30):
                        break
        except Exception as exc:
            self.error=exc

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self,*exc_info):
        self.stop_event.set()
        self.thread.join()
        if self.error:
            raise self.error


def get_states():
    proof=read(SMALL/"latency_input.json")
    src=Path(proof["states"]["path"])
    need(proof["status"]=="passed" and proof["states"]["n"]==N_STATES,"latency state spool unverified")
    verify(src,proof["states"]["sha256"])
    states=list(rows(src))
    counts=Counter((r["mode"],r["budget_tier"]) for r in states)
    need(len(states)==N_STATES and len(counts)==6 and set(counts.values())=={50},"latency grid differs")
    path=Path(proof["r2_run"]["path"])
    verify(path,proof["r2_run"]["sha256"])
    return proof,states,read(path)


def sample_latency(items,generate,synchronize=lambda:None):
    for item in items[:N_WARM]:
        generate([item])
    synchronize()
    wall,outputs=[],[]
    for item in items:
        start=time.monotonic()
        result=generate([item])
        synchronize()
        wall.append(time.monotonic()-start)
        outputs.append(result[0])
    start=time.monotonic()
    batch=generate(items)
    synchronize()
    seconds=time.monotonic()-start
    need(len(outputs)==len(batch)==N_STATES and seconds>0,"incomplete latency outputs")
    return wall,outputs,{"n_states":len(batch),"wall_seconds":seconds,"steps_per_second":len(batch)/seconds}


def compressor(proof,states,previous,mode,directory):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from src.model_inference import generation_record, guard_prompt
    eager=mode=="eager"
    parameters={**previous["engine_parameters"],"enforce_eager":eager}
    need(previous["greedy"] and previous["max_tokens"]==256 and previous["temperature"]==0
         and parameters["max_num_seqs"]==16 and parameters["enable_lora"],"signed inference config differs")
    adapter=Path(previous["adapter"])
    verify(adapter/"adapter_model.safetensors",previous["adapter_weight"]["sha256"])
    for item in states:
        guard_prompt(item,parameters["max_model_len"],256,(item["task"],item["step"]))
    payload={"kind":"compressor","engine":parameters,"adapter":previous["adapter_weight"],
             "sampling":{"temperature":0,"max_tokens":256,"seed":previous["seed"]},
             "n_states":N_STATES,"warmup_n":N_WARM,"payloads_pretokenized":True,
             "versions":{k:version(k) for k in ("torch","vllm")}}
    gpu=reader.gpu()
    need(gpu["index"]==1 and gpu["memory_used_mib"]<2048,"GPU reservation was lost before latency engine start")
    started=time.monotonic()
    try:
        engine=LLM(**parameters)
    except Exception as exc:
        if eager:
            raise
        return {**payload,"status":"load_failed","error_type":type(exc).__name__,"reason":repr(exc),
                "load_wall_seconds":time.monotonic()-started}
    load_seconds=time.monotonic()-started
    sampling=SamplingParams(**payload["sampling"])
    request=LoRARequest("checkpoint-125",1,str(adapter))

    def generate(chunk):
        prompts=[TokensPrompt(prompt_token_ids=row["prompt_token_ids"]) for row in chunk]
        return engine.generate(prompts,sampling,lora_request=request,use_tqdm=False)

    try:
        times,raw,throughput=sample_latency(states,generate)
        lengths=[]
        timings=[]
        for state,elapsed,observed in zip(states,times,raw,strict=True):
            result=generation_record(state,observed,256)
            lengths.append(result["generated_tokens"])
            timings.append({k:state[k] for k in ("mode","budget_tier","task","step","prompt_tokens")}
                           | {"output_tokens":result["generated_tokens"],"wall_seconds":elapsed,
                              "output_sha256":data.sha256_json(result["output_token_ids"])})
        payload.update(status="complete", load_wall_seconds=load_seconds, sequential_seconds=distribution(times),
                       prompt_tokens=distribution([r["prompt_tokens"] for r in states]),
                       output_tokens=distribution(lengths), submitted_together=throughput,
                       per_state=data.dump_rows(directory/"per_state.jsonl",timings))
    finally:
        engine.llm_engine.engine_core.shutdown()
    payload["engine_shutdown"]=True
    return payload


def reranker(proof,states,previous,directory):
    import torch
    from transformers import AutoTokenizer
    train=read(SMALL/"training.json")
    need(train["status"]=="complete","selected reranker unavailable")
    chosen=Path(train["chosen"]["checkpoint"])
    need(chosen.is_relative_to(WORK/"train"),"selected model path redirected")
    verify(chosen/"model.safetensors",train["chosen"]["weights"]["sha256"])
    split=data.verify_split()
    src=DATA/"corpus_train.jsonl"
    verify(src,split["input_files"]["corpus"]["sha256"])
    rendered=data.rendered_documents(list(rows(src)))
    tokenizer=PairTokenizer(AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False))
    need(tokenizer.config()==train["template"],"reranker template differs from training")
    encoded=[]
    for state in states:
        query="\n".join(state["questions"])
        IDs=[p for p in state["candidate_pids"]]
        pairs=list(tokenizer.encode(query,[rendered[pid] for pid in IDs])) if IDs else []
        encoded.append(pairs)
    gpu=reader.gpu()
    need(gpu["index"]==1 and gpu["memory_used_mib"]<2048,"GPU reservation was lost before reranker load")
    start=time.monotonic()
    model=fit.load_model(chosen,torch.bfloat16).eval()
    torch.cuda.synchronize()
    load_seconds=time.monotonic()-start

    def generate(groups):
        values=[]
        for group in groups:
            logical=[ids for ids,_ in group]
            values.append(fit.score_pairs(model,logical,train["template"]) if logical else None)
        return values

    # Each step submits all candidates together, including the throughput pass.
    with torch.inference_mode():
        times,scores,throughput=sample_latency(encoded,generate,torch.cuda.synchronize)
    timing=[]
    for row,pairs,scores_one,elapsed in zip(states,encoded,scores,times,strict=True):
        ntokens=sum(len(ids) for ids,_ in pairs)
        output=scores_one.tolist() if scores_one is not None else []
        need(len(output)==len(pairs),"incomplete latency score set")
        timing.append({k:row[k] for k in ("mode","budget_tier","task","step")} |
                      {"n_candidates":len(pairs),"prompt_tokens":ntokens,"n_truncated":sum(t for _,t in pairs),
                       "scores":output,"generated_tokens":0,"wall_seconds":elapsed})
    payload={"kind":"reranker","status":"complete","n_states":N_STATES,"warmup_n":N_WARM,
             "versions":{k:version(k) for k in ("torch","transformers")},"dtype":"BF16","checkpoint":train["chosen"]["weights"],
             "candidate_batch":"all candidates for a single step together","payloads_pretokenized":True,
             "load_wall_seconds":load_seconds,"sequential_seconds":distribution(times),
             "prompt_tokens":distribution([r["prompt_tokens"] for r in timing]),
             "pair_prompt_tokens":distribution([len(ids) for ids,truncated in chain.from_iterable(encoded)]),
             "n_truncated_pairs":sum(r["n_truncated"] for r in timing),
             "output_tokens":distribution([0]*N_STATES),"submitted_together":throughput,
             "candidates_per_second":sum(r["n_candidates"] for r in timing)/throughput["wall_seconds"],
             "per_state":data.dump_rows(directory/"per_state.jsonl",timing)}
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("eager","graph","reranker"))
    opts=p.parse_args()
    mode=opts.mode
    need(Path.cwd()==PROJECT and os.environ.get("CUDA_VISIBLE_DEVICES")=="1" and os.environ.get("CUDA_DEVICE_ORDER")=="PCI_BUS_ID",
         "Task17 inference requires physical GPU 1 admission")
    directory=WORK/"latency"/mode
    require_new(directory,SMALL/f"latency_{mode}.json")
    directory.mkdir(parents=True)
    stage=WORK/"stages"/f"latency-{mode}"/"receipt_gate.json"
    need(read(stage)["gpu"]["index"]==1,"latency GPU gate missing")
    proof,states,config=get_states()
    gpu0=directory/"gpu0_background.jsonl"
    with GPU0Background(gpu0):
        payload=reranker(proof,states,config,directory) if mode=="reranker" else compressor(proof,states,config,mode,directory)
    payload.update(state_selection=proof["states"],gpu0_background=record(gpu0),
                   gpu0_role="read-only context for latency; separate job on same machine")
    write_new(SMALL/f"latency_{mode}.json",payload)
    print(mode,payload["status"],flush=True)


if __name__=="__main__":
    main()

