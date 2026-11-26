"""CPU-safe registered constants and helpers for Task 17."""
from collections import Counter
from math import exp, fsum, isfinite
import os
from pathlib import Path
import random
import statistics

from src.policies import _pack, _resolve_mode
from src.reader_heldout_inputs import EXTERNAL, PROJECT, checksum, need, read, sums, verify

WORK = EXTERNAL / "reranker_v1"
SMALL = PROJECT / "eval/reranker_v1"
MODEL = EXTERNAL / "models/Qwen3-Reranker-0.6B"
LOCK = EXTERNAL / "rt/gpu1_t17_t18.lock"
DATA = EXTERNAL / "data/traj_train"
SEED = 20261128
LIMIT = 512
BATCH = 64
GRID = (0.0, .02, .05, .1, .2, .3, .5, .7, .9)
R2 = "grpo_r2:checkpoint-125"
V3 = "sft_v3:checkpoint-3000"
STRATEGIES = ("RR-fill", "RR-τ", "RR-zs-fill", R2, V3)
INSTRUCTION = "Given the four questions of a multi-hop QA task, judge whether the passage is a supporting fact needed to answer any of them."
# Qwen's published Transformers format. The task-specific instruction is in the user body.
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
BODY = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
TEMPLATE_SOURCE = "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#using-transformers"


def require_new(*paths):
    for path in paths:
        if path.resolve() != path or os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite or follow a redirected path: {path}")


def signed_project(path):
    index = sums(PROJECT / "SHA256SUMS")
    digest = index[str(path.relative_to(PROJECT))]
    verify(path, digest)
    return digest


def manifest_entries(path):
    signed_project(path)
    return sums(path)


def record(path):
    return {"path": str(path), "size": path.stat().st_size, "sha256": checksum(path)}


def signed_model():
    card = read(SMALL / "model_card.json")
    need(card["model"] == "Qwen/Qwen3-Reranker-0.6B" and card["license"] == "Apache-2.0"
         and card["directory"] == str(MODEL)
         and card["revision"] == "0311d7b88fd535395d154b17fe08f85327306508", "wrong reranker snapshot receipt")
    names = set()
    for item in card["files"]:
        path = MODEL / item["file"]
        need(path.is_relative_to(MODEL) and path.resolve() == path and path.stat().st_size == item["bytes"],
             "snapshot path or size changed")
        verify(path, item["sha256"])
        names.add(item["file"])
    need({"config.json", "model.safetensors", "tokenizer.json"} <= names, "incomplete reranker weights")
    return card


def split_task_ids(ids, n_val=200, n_val100=100):
    ordered = sorted(ids)
    need(len(ordered) == len(set(ordered)) and 0 < n_val100 <= n_val < len(ordered), "invalid split population")
    val = random.Random(SEED).sample(ordered, n_val)
    held = set(val)
    return [task for task in ordered if task not in held], val, val[:n_val100]


def candidates(steps):
    labels = {s["gold"] for s in steps}
    ordered = list(dict.fromkeys(pid for step in steps for pid in step["new"]))
    need(bool(ordered) and bool(labels) and labels <= set(ordered), "invalid saved candidates or gold missing")
    return [(pid, int(pid in labels)) for pid in ordered]


def sample_pair_rows(index_by_task, multiplier):
    need(multiplier in (3, 7), "unregistered negative ratio")
    rng = random.Random(SEED)
    order = []
    for task in sorted(index_by_task):
        group = index_by_task[task]
        positive = [row for row, label in group if label]
        negative = [row for row, label in group if not label]
        order.extend(positive)
        order.extend(rng.sample(negative, min(multiplier * len(positive), len(negative))))
    rng.shuffle(order)
    need(len(order) == len(set(order)), "within-epoch repetition")
    return order


def prompt_text(query, document):
    return BODY.format(instruction=INSTRUCTION, query=query, document=document)


class PairTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.room = LIMIT - len(self.prefix) - len(self.suffix)
        self.yes = tokenizer.convert_tokens_to_ids("yes")
        self.no = tokenizer.convert_tokens_to_ids("no")
        need(self.yes != self.no and tokenizer.encode("yes", add_special_tokens=False) == [self.yes]
             and tokenizer.encode("no", add_special_tokens=False) == [self.no], "invalid single-token labels")
        self.pad = tokenizer.pad_token_id
        need(self.pad is not None and tokenizer.truncation_side == "right" and tokenizer.is_fast,
             "expected local fast tokenizer with right truncation and a pad token")

    def encode(self, query, documents):
        separator = prompt_text(query, "")
        batch = self.tokenizer([separator + doc for doc in documents], add_special_tokens=False,
                               return_attention_mask=False, return_offsets_mapping=True, truncation=False)
        for tokens, offsets in zip(batch["input_ids"], batch["offset_mapping"], strict=True):
            truncated = len(tokens) > self.room
            if truncated:
                # The boundary is checked in characters, even when BPE merges across the leading document space.
                need(offsets[self.room][0] >= len(separator.rstrip()),
                     "the 512-token limit would truncate the instruction or query")
            yield self.prefix + tokens[:self.room] + self.suffix, truncated

    def config(self):
        return {"source": TEMPLATE_SOURCE, "prefix": PREFIX, "body": BODY, "suffix": SUFFIX,
                "instruction": INSTRUCTION, "yes_token_id": self.yes, "no_token_id": self.no,
                "pad_token_id": self.pad, "padding_side": "left", "max_pair_tokens": LIMIT,
                "truncation": "right end of document only", "score": "last_position_logit_yes_minus_no",
                "probability": "sigmoid"}


def logit_difference(last_logits, yes, no):
    return last_logits[..., yes] - last_logits[..., no]


def probability(score):
    need(isfinite(score), "nonfinite reranker logit difference")
    if score >= 0:
        return 1 / (1 + exp(-score))
    value = exp(score)
    return value / (1 + value)


def choose_checkpoint(checkpoints):
    need(checkpoints and all(isfinite(row["mAP"]) for row in checkpoints), "missing or nonfinite AP")
    return min(checkpoints, key=lambda row: (-row["mAP"], row["step"]))


def choose_tau(arms):
    need(set(arms) == set(GRID) and all(isfinite(v) for v in arms.values()), "invalid or incomplete tau grid")
    best = max(arms.values())
    return min(tau for tau in GRID if best - arms[tau] < 1e-9)


def average_precision(scored):
    """Task-level PR area with ties scored together at one threshold."""
    order = sorted(scored, key=lambda row: row[0], reverse=True)
    need(order and all(isfinite(s) and y in (0, 1) for s, y in order), "invalid AP inputs")
    positives = sum(y for _, y in order)
    need(positives > 0, "AP requires a task with at least one gold candidate")
    from itertools import groupby
    seen = hit = 0
    terms = []
    for _, group in groupby(order, key=lambda row: row[0]):
        labels = [label for _, label in group]
        increment = sum(labels)
        seen += len(labels)
        hit += increment
        terms.append(increment / positives * hit / seen)
    return fsum(terms)


def reranker_action(state, scores, tau=None):
    mode = _resolve_mode(state, None)
    pids = state["pid_by_lid"]
    options = state["new_ids"] if mode == "append" else state["kept_ids"] + state["new_ids"]
    ranked = [(i, lid, float(scores[pids[lid]])) for i, lid in enumerate(options)]
    need(all(isfinite(score) for _, _, score in ranked), "nonfinite candidate scores")
    priority = [lid for i, lid, score in sorted(ranked, key=lambda row: (-row[2], -row[0]))
                if tau is None or score >= tau]
    return _pack(state, priority, mode)


def distribution(values):
    values = sorted(values)
    need(bool(values), "empty distribution")
    p90 = (len(values) - 1) * .9
    i = int(p90)
    return {"n": len(values), "min": values[0], "max": values[-1], "mean": fsum(values) / len(values),
            "p50": statistics.median(values), "p90": values[i] + (values[min(i+1,len(values)-1)]-values[i]) * (p90-i)}


def histogram(values):
    return {str(value): count for value, count in sorted(Counter(values).items())}

