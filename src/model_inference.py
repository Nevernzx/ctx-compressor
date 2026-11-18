"""Pure prompt and output guards for Task 05; no model/runtime imports."""

import hashlib
import json

from src.build_traj import render_input
from src.sft_protocol import INPUT_FIELDS


MAX_NEW_TOKENS = 256
SEED = 20261118


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def inference_prompt(tokenizer, state):
    """Use exactly the Task 04 HF generation prefix, including empty think."""
    visible = {name: state[name] for name in INPUT_FIELDS}
    user_text = render_input(visible)
    messages = [{"role": "user", "content": user_text}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False)
    if not ids or any(type(value) is not int for value in ids):
        raise ValueError("tokenizer must produce a nonempty integer token list")
    return {"prompt_token_ids": list(ids), "prompt_tokens": len(ids),
            "prompt_sha256": sha256_json(ids),
            "user_text_sha256": hashlib.sha256(user_text.encode("utf-8")).hexdigest()}


def guard_prompt(prompt, max_model_len, max_tokens=MAX_NEW_TOKENS, identity=None):
    """Never crop prompts or drop a trajectory to make inference fit."""
    if type(max_model_len) is not int or max_model_len <= 0:
        raise ValueError("max_model_len must be a positive integer")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if prompt["prompt_tokens"] != len(prompt["prompt_token_ids"]):
        raise ValueError("prompt length metadata mismatch")
    if prompt["prompt_tokens"] + max_tokens > max_model_len:
        raise RuntimeError(f"Prompt overflow; refusing truncation or trajectory omission: "
                           f"{identity}: {prompt['prompt_tokens']} + {max_tokens} > {max_model_len}")


def generation_record(prompt, output, max_tokens=MAX_NEW_TOKENS):
    """Assert that the runtime used the exact supplied IDs; preserve raw text."""
    if list(output.prompt_token_ids) != prompt["prompt_token_ids"]:
        raise RuntimeError("Inference prompt IDs changed or were truncated")
    if len(output.outputs) != 1:
        raise RuntimeError("Greedy inference must return exactly one completion")
    completion = output.outputs[0]
    output_ids = list(completion.token_ids)
    if len(output_ids) > max_tokens:
        raise RuntimeError("Runtime exceeded the fixed output token budget")
    return {**prompt, "raw_output": completion.text,
            "generated_tokens": len(output_ids), "output_token_ids": output_ids,
            "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
            "output_limit_hit": len(output_ids) == max_tokens or completion.finish_reason == "length",
            "input_truncated": False, "runtime_prompt_ids_match": True}


def engine_parameters(model, native_config):
    """Single GPU with native context length and a bounded LoRA working set."""
    if native_config.get("rope_scaling") is not None:
        raise ValueError("Task 05 requires native context without rope scaling")
    maximum = native_config["max_position_embeddings"]
    if type(maximum) is not int or maximum < MAX_NEW_TOKENS:
        raise ValueError("invalid native model context")
    return {"model": str(model), "tokenizer": str(model), "trust_remote_code": False,
            "dtype": "bfloat16", "max_model_len": maximum,
            "gpu_memory_utilization": 0.70, "tensor_parallel_size": 1,
            "enable_prefix_caching": True, "enable_chunked_prefill": True,
            "max_num_seqs": 16, "max_num_batched_tokens": 8192, "enforce_eager": True,
            "enable_lora": True, "max_lora_rank": 16, "max_loras": 1,
            "max_cpu_loras": 5, "seed": SEED}
