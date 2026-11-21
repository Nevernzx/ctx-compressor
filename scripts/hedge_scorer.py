"""Teacher-forced candidate log probabilities on the exact Task 06 inference prompts."""
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

from scripts.merge_sft_lora import sha256
from src.model_inference import MAX_NEW_TOKENS, SEED

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
ADAPTER = EXTERNAL / "runs/sft_20k_v2_20261118/checkpoints/checkpoint-2000"
WEIGHT_SHA = "7253eae08072782971fa8b7874e97fc3e76dee4ebd11ced5c014147090cf1f9c"
GPU_UUID = "GPU-4318dc90-4126-038a-9402-1323533db94f"


def gate_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("Task 09 requires only physical GPU 1 in PCI bus order")
    sample = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=index,uuid,memory.used",
                                      "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    if sample[:2] != ["1", GPU_UUID] or int(sample[2]) >= 2048:
        raise RuntimeError(f"Physical GPU 1 preflight failed: {sample}")
    return {"index": 1, "uuid": sample[1], "memory_used_mib": int(sample[2])}


def create_engine():
    gate_gpu()
    if sha256(ADAPTER / "adapter_model.safetensors") != WEIGHT_SHA:
        raise ValueError("SFT v2 adapter checksum mismatch")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    params = json.loads((PROJECT / "eval/sft_run_v2/heldout_v2_preflight.json").read_bytes())["engine_parameters"]
    if params["model"] != str(EXTERNAL / "models/Qwen3-1.7B") or not params["enable_prefix_caching"]:
        raise ValueError("Unexpected frozen model or cache configuration")
    llm = LLM(**params, logprobs_mode="raw_logprobs",
              logits_processors=["scripts.hedge_raw_logits:CaptureCandidateLogits"])
    llm._hedge_logit_session = uuid.uuid4().hex
    llm._hedge_next_serial = 0
    llm._hedge_read_offset = 0
    # Task 06 loaded the fourth adapter (step 2000) with LoRA id 4.
    return llm, LoRARequest(ADAPTER.name, 4, str(ADAPTER)), params


def continuation(tokenizer, candidate):
    if type(candidate) is not int or candidate < 0:
        raise ValueError(f"Invalid candidate id: {candidate}")
    string = '{"add":[' + str(candidate) + '],"'
    ids = tokenizer.encode(string, add_special_tokens=False)
    if not ids or tokenizer.decode(ids, skip_special_tokens=False) != string:
        raise ValueError(f"Candidate tokenization is not lossless: {candidate}")
    return list(ids)


def raw_logprob(completion, token_ids):
    if list(completion.token_ids[:len(token_ids)]) != token_ids or len(completion.logprobs) < len(token_ids):
        raise ValueError("Requested teacher-forced token sequence is absent from runtime output")
    values = [completion.logprobs[i][token].logprob for i, token in enumerate(token_ids)]
    if not all(math.isfinite(value) and value <= 0.000001 for value in values):
        raise ValueError(f"Non-finite or positive raw logprob: {values}")
    return math.fsum(values)


def _evaluate(llm, adapter, prompts, params):
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], sampling_params=params,
                           lora_request=adapter, use_tqdm=False)
    if len(outputs) != len(prompts):
        raise ValueError("Incomplete vLLM batch")
    for prompt, output in zip(prompts, outputs):
        if list(output.prompt_token_ids) != prompt or len(output.outputs) != 1:
            raise ValueError("Runtime changed prompt ids, order or completion count")
    return [output.outputs[0] for output in outputs]


def _score(llm, adapter, prompt_ids, candidate_ids, paired_ids=None, natural_ids=None, shared_paths=None, branch_evidence=None):
    """During validation, pair the archived greedy-id request with an independent unforced decode."""
    from vllm import SamplingParams

    if len(prompt_ids) != len(candidate_ids) or paired_ids is not None and len(paired_ids) != len(prompt_ids):
        raise ValueError("Misaligned prompts/candidates")
    tokenizer = llm.get_tokenizer()
    forced = {}
    results = [{} for _ in prompt_ids]
    references = [None for _ in prompt_ids]
    warmup = SamplingParams(temperature=0, seed=SEED, max_tokens=1)
    greedy = SamplingParams(temperature=0, seed=SEED, max_tokens=MAX_NEW_TOKENS,
                            logprobs=0, skip_special_tokens=False, include_stop_str_in_output=True)
    for first in range(0, len(prompt_ids), 16):
        end = min(len(prompt_ids), first + 16)
        _evaluate(llm, adapter, prompt_ids[first:end], warmup)
        common, paired = [], []
        for i in range(first, end):
            if len(set(candidate_ids[i])) != len(candidate_ids[i]):
                raise ValueError("Duplicate candidate IDs")
            for c in candidate_ids[i]:
                if c not in forced:
                    ids = continuation(tokenizer, c)
                    forced[c] = ids, SamplingParams(temperature=0, seed=SEED, max_tokens=len(ids),
                                                    logprobs=0, extra_args={"hedge_teacher_tokens": ids})
                if natural_ids is not None and c == natural_ids[i]:
                    # Same queue, maximum length, KV cache and batch shape as production.
                    # This request samples the model's natural argmax, with no forced tokens.
                    params = SamplingParams(temperature=0, seed=SEED, max_tokens=len(forced[c][0]), logprobs=0)
                else:
                    params = forced[c][1]
                job = (i, c, params)
                if paired_ids is not None and c == paired_ids[i]:
                    paired.extend((job, (i, None, greedy)))
                else:
                    common.append(job)
        # Keep paired requests in the same batch to compare identical BF16 kernels.
        for jobs in (paired, common):
            if not jobs:
                continue
            for start in range(0, len(jobs), 16 if jobs is paired else len(jobs)):
                batch = jobs[start:start + (16 if jobs is paired else len(jobs))]
                completions = _evaluate(llm, adapter, [prompt_ids[i] for i, _, _ in batch],
                                        [params for _, _, params in batch])
                for (i, c, _), out in zip(batch, completions):
                    if c is None:
                        references[i] = out
                    else:
                        if natural_ids is not None and c == natural_ids[i]:
                            references[i] = out
                            continue
                        ids = forced[c][0]
                        if list(out.token_ids) != ids:
                            raise ValueError(f"Teacher-forced output mismatch on candidate {c}")
                        if shared_paths is None:
                            results[i][c] = raw_logprob(out, ids)
                        else:
                            results[i][c] = shared_score(ids, shared_paths[i], out)
                            if branch_evidence is not None:
                                branch_evidence[i][c] = (ids, token_values(out))
    return results, references


def token_values(completion):
    return [completion.logprobs[i][token].logprob for i, token in enumerate(completion.token_ids)]


def shared_score(candidate_tokens, natural_path, forced_completion):
    # Every candidate uses the same prefix-tree rule, including candidates that
    # diverge at the first digit or share all but the closing punctuation.
    natural_ids, natural_values = natural_path
    length = 0
    for natural, candidate in zip(natural_ids,candidate_tokens):
        if natural != candidate:
            break
        length += 1
    forced_values = token_values(forced_completion)
    values = list(natural_values[:length]) + forced_values[length:]
    if len(values) != len(candidate_tokens) or not all(math.isfinite(v) and v <= 1e-6 for v in values):
        raise ValueError("Incomplete or invalid autoregressive prefix path")
    return math.fsum(values)


def greedy_path(llm, adapter, ids, show_terminal=False):
    """Evaluate one shared trie path in a fixed singleton request shape."""
    from vllm import SamplingParams
    warm = SamplingParams(temperature=0,max_tokens=1,seed=SEED)
    _evaluate(llm,adapter,[ids],warm)
    cfg = SamplingParams(temperature=0,max_tokens=MAX_NEW_TOKENS,seed=SEED,logprobs=0,
                         skip_special_tokens=not show_terminal,include_stop_str_in_output=show_terminal)
    return _evaluate(llm,adapter,[ids],cfg)[0]


def capture_file(llm):
    return EXTERNAL / "hedge_v2/score_logits" / (llm._hedge_logit_session + ".jsonl")


def read_logit_observations(path, offset, serials, outputs):
    """Reconcile worker observations with the IDs actually committed by vLLM."""
    observed = {serial:{} for serial in serials}
    with path.open("rb") as stream:
        stream.seek(offset)
        for line in stream:
            serial,position,token,logit,normalizer=json.loads(line)
            if serial not in observed or type(position) is not int or position<0:
                raise ValueError("Unexpected worker observation")
            if not (math.isfinite(logit) and math.isfinite(normalizer)):
                raise ValueError("Non-finite raw logit or log-sum-exp")
            # A discarded chunked prefill can emit a provisional value at
            # offset zero. The final observation precedes the committed token.
            observed[serial][position]=(token,logit,normalizer)
        offset=stream.tell()
    ordered=[]
    for serial,out in zip(serials,outputs):
        expected=list(out.token_ids)
        rows=observed[serial]
        if (set(rows)!=set(range(len(expected))) or
                any(rows[i][0]!=t for i,t in enumerate(expected))):
            raise ValueError(f"Worker argmax disagrees with generated token IDs at request {serial}")
        ordered.append([rows[i] for i in range(len(expected))])
    return ordered,offset


def captured_natural_paths(llm,adapter,prompts):
    from vllm import SamplingParams
    serials=list(range(llm._hedge_next_serial,llm._hedge_next_serial+len(prompts)))
    llm._hedge_next_serial+=len(prompts)
    warm=SamplingParams(temperature=0,max_tokens=1,seed=SEED)
    _evaluate(llm,adapter,prompts,warm)
    configurations=[SamplingParams(temperature=0,max_tokens=MAX_NEW_TOKENS,seed=SEED,logprobs=0,
                         skip_special_tokens=False,include_stop_str_in_output=True,
                         extra_args={"hedge_logit_session":llm._hedge_logit_session,"hedge_logit_request":serial})
                         for serial in serials]
    raw=llm.generate([{"prompt_token_ids":ids} for ids in prompts], sampling_params=configurations,
                     lora_request=adapter,use_tqdm=False)
    if len(raw)!=len(prompts) or any(list(r.prompt_token_ids)!=p or len(r.outputs)!=1 for r,p in zip(raw,prompts)):
        raise ValueError("Incomplete natural generation")
    observations,llm._hedge_read_offset=read_logit_observations(
        capture_file(llm),llm._hedge_read_offset,serials,[r.outputs[0] for r in raw])
    return list(zip(raw,observations))


def score_candidates(llm, adapter, prompt_ids, candidate_ids, *, audit=None):
    """Sum independently normalized greedy-prefix probabilities and forced branch suffixes."""
    if len(prompt_ids) != len(candidate_ids):
        raise ValueError("Misaligned prompts and candidates")
    results=[]
    for start in range(0,len(prompt_ids),16):
        prompts=prompt_ids[start:start+16]
        ids=candidate_ids[start:start+16]
        paths=captured_natural_paths(llm,adapter,prompts)
        shared=[([t for t,_,_ in tokens],[logit-z for _,logit,z in tokens]) for _,tokens in paths]
        evidence=[{} for _ in prompts] if audit is not None else None
        results.extend(_score(llm,adapter,prompts,ids,shared_paths=shared,branch_evidence=evidence)[0])
        if audit is not None:
            audit.extend((ref,tokens,e) for (ref,tokens),e in zip(paths,evidence))
    return results


def score_with_reference(llm, adapter, prompt_ids, candidate_ids, greedy_ids):
    return _score(llm, adapter, prompt_ids, candidate_ids, paired_ids=greedy_ids)



def natural_prefixes(llm, adapter, prompt_ids, candidate_ids, selected_ids):
    if len(selected_ids) != len(prompt_ids):
        raise ValueError("Unaligned natural reference IDs")
    return _score(llm, adapter, prompt_ids, candidate_ids, natural_ids=selected_ids)[1]
