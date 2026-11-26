"""Signed held-out inputs for the Task 14 DeepSeek-reader comparison."""
from dataclasses import dataclass
import hashlib
import random

from src import reader_heldout_inputs as t12
from src.qa_metrics import score_answer
from src.reader import parse_answer

R2 = "grpo_r2:checkpoint-125"
V3 = "sft_v3:checkpoint-3000"
B_POLICY = "deepseek-flash"
MODES = ("reselect", "append")
A_POLICIES = ("oracle", R2, V3)
SEED = 20261127
MANIFEST = "eval/reader_heldout/TASK12_SHA256SUMS"
MANIFEST_SHA256 = "a0b093d2cd8d777995fb51dffd1ec569491a24bd3007a24e00a0138b28d2515e"


def subsets(task_ids):
    ids = sorted(task_ids)
    t12.need(len(ids) == len(set(ids)) and len(ids) >= 60, "invalid held-out task population")
    s60 = random.Random(SEED).sample(ids, 60)
    return s60, s60[:30]


def configurations(s60, s30, phase):
    if phase == "smoke":
        for task in s60[:2]:
            yield "oracle", "reselect", "mid", task
        yield "no_compress", None, None, s30[0]
    elif phase == "full":
        for task in s60:
            for name in A_POLICIES:
                for mode in MODES:
                    yield name, mode, "mid", task
            if task in s30:
                yield "no_compress", None, None, task
            yield "no_context", None, None, task
    else:
        raise ValueError(f"unknown phase: {phase}")


def question_rows(strategy, mode, task_id, pids, tasks, corpus):
    task = tasks[task_id]
    kept = set(pids)
    t12.need(len(kept) == len(pids) and all(type(pid) is int and 0 <= pid < len(corpus) for pid in pids),
             "invalid ordered final context")
    t12.need(mode in (None, *MODES) and (mode is None) == (strategy in ("no_context", "no_compress")),
             "mismatched reader policy/mode")
    t12.need(strategy != "no_context" or not pids, "closed-book context must be empty")
    passages = t12.render_passages(corpus, pids)
    tier = "mid" if mode is not None else None
    result = []
    for qi, question in enumerate(task["metadata"]):
        # Reuse the exact passage and user-message builders that supplied the Task 12 reader.
        user = t12.render_user_message(passages, question["question"], no_context=strategy == "no_context")
        result.append({
            "strategy": strategy, "mode": mode, "tier": tier, "task": task_id, "question_index": qi,
            "config": f"{strategy}|{mode}|{tier}" if mode else strategy,
            "example_id": question["example_id"], "question": question["question"],
            "answer": question["answer"], "answer_aliases": question["answer_aliases"],
            "question_hops": question["question_hops"], "evidence_complete": question["gold"] <= kept,
            "trajectory_lost": not task["all_gold"] <= kept,
            "non_gold_count": len(kept - task["all_gold"]),
            "messages": [{"role": "user", "content": user}],
            "prompt_sha1": hashlib.sha1(user.encode("utf-8")).hexdigest(),
        })
    t12.need(len(result) == 4, "expected four reader questions")
    return result


def key(row):
    return tuple(row[k] for k in ("strategy", "mode", "tier", "task", "question_index"))


def compare_qwen(request, prediction):
    t12.need(key(request) == key(prediction), "reader identity mismatch")
    identity = ("prompt_sha1", "config", "example_id", "answer", "answer_aliases",
                "question_hops", "evidence_complete", "trajectory_lost")
    for field in identity:
        t12.need(request[field] == prediction[field], f"Task 12 {field} mismatch at {key(request)}")
    t12.need(not prediction["input_truncated"], "Task 12 truncated an input")
    pred = parse_answer(prediction["raw_output"])
    scores = score_answer(pred, request["answer"], request["answer_aliases"])
    t12.need(pred == prediction["pred"] and all(scores[k] == prediction[k] for k in ("em", "f1")),
             f"Task 12 scoring mismatch at {key(request)}")
    return {**request, "qwen_em": scores["em"], "qwen_f1": scores["f1"],
            "qwen_prompt_tokens": prediction["prompt_tokens"]}


@dataclass
class PreparedInputs:
    s60: list
    s30: list
    tasks: dict
    corpus: list
    requests: dict
    audit: dict

    def a_rows(self, phase):
        return [self.requests[name, mode, tier, task, qi]
                for name, mode, tier, task in configurations(self.s60, self.s30, phase)
                for qi in range(4)]

    def reader_rows(self, task, mode, final_pids):
        t12.need(task in self.s30 and mode in MODES, "B reader outside registered subset")
        return question_rows(B_POLICY, mode, task, final_pids, self.tasks, self.corpus)


def prepare():
    """Check all registered A prompts before the caller can send the first request."""
    project = t12.PROJECT
    index = t12.sums(project / "SHA256SUMS")
    t12.need(index[MANIFEST] == MANIFEST_SHA256, "Task 12 manifest index changed")
    t12.verify(project / MANIFEST, MANIFEST_SHA256)
    manifest = t12.sums(project / MANIFEST)
    t12.verify(project / "src/reader_heldout_inputs.py", manifest["src/reader_heldout_inputs.py"])
    folder = t12.EXTERNAL / "reader_heldout"
    preflight_path = folder / "input_preflight/preflight.json"
    pre = t12.checked_json(preflight_path, manifest[str(preflight_path)])
    t12.need(pre["status"] == "passed" and pre["input_truncations"] == 0 and
             pre["n_tasks"] == 249 and pre["n_questions"] == 996, "invalid signed Task 12 preflight")
    corpus, tasks, raw_hashes = t12.load_task_inputs(index)
    t12.need(raw_hashes == pre["inputs_sha256"], "held-out source differs from Task 12")
    for task, row in tasks.items():
        row.update(task=task, split="heldout")
    s60, s30 = subsets(tasks)
    ntok = [p["ntok"] for p in corpus]
    finals = {}
    source_hashes = {}
    for source in t12.historical_sources(index):
        if source.name not in (*A_POLICIES, "no_compress"):
            continue
        sealed = pre["sources"][source.name]
        t12.need(sealed["steps"] == str(source.steps) and sealed["metrics"] == str(source.metrics)
                 and sealed["steps_sha256"] == source.steps_sha256
                 and sealed["metrics_sha256"] == source.metrics_sha256, "Task 12 rollout lineage mismatch")
        finals[source.name], _ = t12.replay(source, tasks, ntok)
        source_hashes[source.name] = sealed
    t12.need(set(finals) == {*A_POLICIES, "no_compress"}, "incomplete reader policies")
    nc = t12.dedup_contexts(finals["no_compress"], tasks)
    t12.need(nc == pre["no_compress"] and nc["all_identical"], "shared no_compress context changed")

    pred_path = folder / "predictions.jsonl"
    t12.verify(pred_path, manifest[str(pred_path)])
    configs = list(configurations(s60, s30, "full"))
    wanted = {(*c, qi) for c in configs for qi in range(4)}
    t12.need(len(wanted) == 1800, "registered subset size differs")
    predictions = {}
    for pred in t12.rows(pred_path):
        k = key(pred)
        if k in wanted:
            t12.need(k not in predictions, f"duplicate Task 12 prediction: {k}")
            predictions[k] = pred
    t12.need(predictions.keys() == wanted, "Task 12 predictions incomplete")
    requests = {}
    for name, mode, tier, task in configs:
        if name == "no_context":
            pids = []
        else:
            cell = (mode, tier) if mode else t12.CELLS[0]
            pids = finals[name][cell, task]
        for row in question_rows(name, mode, task, pids, tasks, corpus):
            k = key(row)
            requests[k] = compare_qwen(row, predictions[k])
    t12.need(requests.keys() == wanted, "incomplete prepared input matrix")
    audit = {"status": "passed", "subset_seed": SEED, "s60": s60, "s30": s30,
             "n_prompts_verified": len(requests), "prompt_sha1_mismatches": 0,
             "n_prompts_by_strategy": {name: sum(k[0] == name for k in requests) for name in
                                       (*A_POLICIES, "no_compress", "no_context")},
             "task12_manifest_sha256": MANIFEST_SHA256,
             "task12_preflight_sha256": manifest[str(preflight_path)],
             "task12_predictions_sha256": manifest[str(pred_path)], "input_sha256": raw_hashes,
             "signed_rollouts": source_hashes, "no_compress": nc}
    return PreparedInputs(s60, s30, tasks, corpus, requests, audit)

