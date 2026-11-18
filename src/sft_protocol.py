"""One explicit ms-swift Qwen3 protocol for Task 04 preparation and training.

Swift imports are lazy, so schema/statistics tests also run in the old ctxc env.
"""
from collections import Counter
from copy import deepcopy
import math


TEMPLATE_OPTIONS = {
    "template_type": "qwen3", "truncation_strategy": "raise",
    "loss_scale": "last_round", "enable_thinking": False,
    "add_non_thinking_prefix": True, "template_backend": "swift",
    "padding_side": "right", "padding_free": False,
}
EMPTY_THINK = "<think>\n\n</think>\n\n"
ASSISTANT_END = "<|im_end|>\n"
INPUT_FIELDS = ("questions", "mode", "budget", "kept_ids", "new_ids", "ntok", "text")


def make_sft_template(model_dir=None, max_length=None, *, processor=None):
    """Load tokenizer only (or reuse processor); return a training-mode template.

    None deliberately disables the model's default length cap for lossless
    measurement. Benchmark callers pass the measured global max_length, which
    raises on excess length rather than cropping or splitting a sample.
    """
    from swift.llm import get_model_tokenizer, get_template
    if processor is None:
        if model_dir is None:
            raise ValueError("model_dir or processor is required")
        _, processor = get_model_tokenizer(str(model_dir), load_model=False,
                                           model_type="qwen3", use_hf=False)
    template = get_template(processor=processor, max_length=max_length, **TEMPLATE_OPTIONS)
    template.max_length = max_length
    template.set_mode("train")
    return template


def restore_state_keys(state):
    return {**state, **{field: {int(k): v for k, v in state[field].items()}
                       for field in ("ntok", "text", "pid_by_lid")}}


def make_messages(state):
    """Render only the observable state; oracle fields enter assistant content."""
    from src.build_traj import render_input, render_target
    visible = {name: state[name] for name in INPUT_FIELDS}
    return [{"role": "user", "content": render_input(visible)},
            {"role": "assistant", "content": render_target(state)}]


def encoding_counts(input_ids, labels):
    """Check a masked prefix followed by a single supervised assistant suffix."""
    if not input_ids or len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have the same nonzero length")
    first = next((i for i, label in enumerate(labels) if label != -100), None)
    if first is None or first == 0:
        raise ValueError("both masked input and supervised assistant are required")
    if labels[first:] != input_ids[first:]:
        raise ValueError("labels must equal token IDs throughout the assistant suffix")
    return {"n_input_tokens": first, "n_target_tokens": len(labels) - first,
            "total_tokens": len(input_ids), "loss_tokens": len(labels) - first}


def encode_sft_record(template, messages):
    """Actual Swift training encoding plus token counts; never truncate a row.

    n_input counts -100 labels; n_target counts supervised tokens, including the
    empty think block and assistant terminator. Their sum is the full sequence.
    The returned input_ids/labels can be passed to template.data_collator.
    """
    if [message.get("role") for message in messages] != ["user", "assistant"]:
        raise ValueError("exactly one user and one assistant message are required")
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError("message content must be text")
    if template.mode != "train" or template.truncation_strategy != "raise":
        raise ValueError("training mode and truncation_strategy=raise are required")
    encoded = template.encode({"messages": deepcopy(messages)})
    counts = encoding_counts(encoded["input_ids"], encoded["labels"])
    target = template.tokenizer.decode(encoded["labels"][counts["n_input_tokens"]:],
                                       skip_special_tokens=False)
    if target != EMPTY_THINK + messages[-1]["content"] + ASSISTANT_END:
        raise ValueError("supervised tokens differ from the exact assistant-only content")
    return {**encoded, **counts}


def template_evidence(template, messages, encoded):
    """Check full rendered bytes and inference prefix, keeping all loss labels."""
    tokenizer = template.tokenizer
    prefix = tokenizer.apply_chat_template(messages[:1], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    prefix_ids = tokenizer.apply_chat_template(messages[:1], tokenize=True,
                                                add_generation_prompt=True, enable_thinking=False)
    text = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    if text != prefix + messages[-1]["content"] + ASSISTANT_END:
        raise ValueError("actual Swift training string differs from HF inference prefix + answer")
    if encoded["input_ids"][:len(prefix_ids)] != prefix_ids:
        raise ValueError("HF inference token prefix differs from actual Swift training tokens")
    first = encoded["n_input_tokens"]
    return {"full_text": text, "full_input_ids": encoded["input_ids"],
            "masked_label_range": [0, first],
            "supervised_label_range": [first, encoded["total_tokens"]],
            "supervised_tokens": [
                {"position": i, "input_id": token, "label": encoded["labels"][i],
                 "token": tokenizer.convert_ids_to_tokens(token)}
                for i, token in enumerate(encoded["input_ids"][first:], first)],
            "hf_inference_prefix_tokens": len(prefix_ids),
            "empty_think_tokens": len(prefix_ids) - first,
            "training_text_matches_hf_prefix_plus_answer": True,
            "training_ids_start_with_hf_inference_prefix": True,
            **{name: encoded[name] for name in ("n_input_tokens", "n_target_tokens", "total_tokens", "loss_tokens")}}


def sample_flags(state):
    from src.build_traj import apply_action, oracle_action
    target = state["sft_target"]
    kept = apply_action(state, target)
    if kept is None or target != oracle_action(state):
        raise ValueError("every SFT target must be the legal pure oracle action")
    return {"target_add_empty": not target["add"],
            "target_drop_nonempty": bool(target["drop"]),
            "budget_binding": not set(state["gold"]).issubset(kept)}


def length_distribution(histogram):
    """Exact linearly interpolated percentiles from a bounded integer histogram."""
    counts = Counter(histogram)
    n = sum(counts.values())
    if not n:
        return {"n": 0, **dict.fromkeys(("mean", "p50", "p90", "p99", "max"))}
    ordered = sorted(counts.items())

    def order_stat(rank):
        cumulative = 0
        for value, count in ordered:
            cumulative += count
            if rank < cumulative:
                return value
        raise AssertionError("rank outside histogram")

    def percentile(q):
        rank = (n - 1) * q
        lo, hi = math.floor(rank), math.ceil(rank)
        return order_stat(lo) + (order_stat(hi) - order_stat(lo)) * (rank - lo)

    return {"n": n, "mean": sum(value * count for value, count in ordered) / n,
            "p50": percentile(0.5), "p90": percentile(0.9), "p99": percentile(0.99),
            "max": ordered[-1][0]}


def choose_max_length(maximum):
    """Round upward for alignment; callers must stop benchmark if over 16384."""
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("maximum must be a positive integer")
    return ((maximum + 7) // 8) * 8
