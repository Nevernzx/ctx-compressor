"""Task 05 guards run in the original CPU env, without importing vLLM."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.model_inference import (engine_parameters, generation_record, guard_prompt,
                                 inference_prompt)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True,
                          "enable_thinking": False}
        assert [m["role"] for m in messages] == ["user"]
        self.messages = messages
        return [151644, 7, 151645, 8]


def state():
    return {"questions": ["question"], "mode": "reselect", "budget": 20,
            "kept_ids": [0], "new_ids": [1], "ntok": {0: 3, 1: 4},
            "text": {0: "old", 1: "new"}, "gold": [1], "answer": "SECRET"}


def test_prompt_is_label_blind_and_includes_no_system_message():
    tok = Tokenizer()
    first = inference_prompt(tok, state())
    assert "SECRET" not in tok.messages[0]["content"]
    changed = deepcopy(state())
    changed.update(gold=[0], answer="OTHER", sft_target={"add": [0]})
    assert inference_prompt(tok, changed) == first


def test_context_guard_reserves_all_256_output_tokens():
    prompt = inference_prompt(Tokenizer(), state())
    guard_prompt(prompt, 260)
    with pytest.raises(RuntimeError, match="refusing truncation"):
        guard_prompt(prompt, 259, identity="task-7")
    prompt["prompt_tokens"] = 3
    with pytest.raises(ValueError, match="metadata"):
        guard_prompt(prompt, 260)


def test_actual_engine_prompt_ids_must_equal_sent_ids():
    prompt = inference_prompt(Tokenizer(), state())
    completion = SimpleNamespace(text='{"add":[],"drop":[]}', token_ids=[12, 13],
                                 finish_reason="stop", stop_reason=None)
    output = SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[completion])
    result = generation_record(prompt, output)
    assert result["raw_output"] == completion.text
    assert not result["input_truncated"] and not result["output_limit_hit"]
    output.prompt_token_ids = prompt["prompt_token_ids"][:-1]
    with pytest.raises(RuntimeError, match="changed or were truncated"):
        generation_record(prompt, output)


def test_raw_length_limited_output_preserved_for_invalid_json_counting():
    prompt = inference_prompt(Tokenizer(), state())
    completion = SimpleNamespace(text='{"add":[', token_ids=list(range(256)),
                                 finish_reason="length", stop_reason=None)
    output = SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[completion])
    result = generation_record(prompt, output)
    assert result["output_limit_hit"] and result["generated_tokens"] == 256
    assert result["raw_output"] == '{"add":['


def test_native_context_and_single_device_lora_configuration():
    result = engine_parameters("local-model", {"max_position_embeddings": 40960})
    assert result["max_model_len"] == 40960
    assert result["max_model_len"] > 8392
    assert result["tensor_parallel_size"] == 1 and result["max_lora_rank"] == 16
    assert result["gpu_memory_utilization"] == 0.70
    with pytest.raises(ValueError, match="rope scaling"):
        engine_parameters("model", {"max_position_embeddings": 40960, "rope_scaling": {}})
