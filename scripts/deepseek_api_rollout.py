"""Task 14 compressor policy. All action semantics and cost metrics use held-out replay."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import random
import re

from src.build_traj import render_input
from src.heldout_eval import BUDGETS, HeldoutReplay, MODES, apply_model_output
from src.reader_heldout_inputs import (
    PROJECT, checked_json, historical_sources, need, rows, sums, verify,
)
from src.sft_protocol import INPUT_FIELDS


SYSTEM = """你是一个多步检索 Agent 的上下文压缩器。每一步你会看到：任务问题、模式、预算、已用 token、"已保留"的片段和本步"新返回"的片段，每个片段标有编号和 token 数。
你要决定：新返回的片段里保留哪些（add），已保留的片段里丢掉哪些（drop），使保留下来的内容包含回答全部任务问题所需的证据，同时保留片段的 token 总数不超过预算。
只输出一个 JSON 对象，不要任何其他文字，也不要代码块标记，格式：{"add": [编号, ...], "drop": [编号, ...]}
规则：add 只能是"新返回"里的编号；drop 只能是"已保留"里的编号；模式为 append 时 drop 必须为空；执行后保留片段的 token 总和不能超过预算。不合法的输出视为本步什么都不改。"""
STRATEGY = "deepseek-flash"
BASELINES = ("grpo_r2:checkpoint-125", "sft_v3:checkpoint-3000", "oracle", "bm25")
BUDGET = {"mid": BUDGETS["mid"]}
FENCE = re.compile(r"\A```(?:json\b)?\s*(.*?)\s*```\Z", re.I | re.S)


def user_text(state):
    return render_input({field: state[field] for field in INPUT_FIELDS})


def text_sha(text):
    return sha256(text.encode("utf-8")).hexdigest()


def validate_step0(text, expected):
    value = text_sha(text)
    need(value == expected, "B step 0 user text differs from the signed r2 rollout")
    return value


def diagnostic_rescue(state, raw, reason):
    """Count only invalid outputs that become legal after outer whitespace/fence removal."""
    if reason is None:
        return False
    candidate = raw.strip()
    match = FENCE.fullmatch(candidate)
    if match:
        candidate = match.group(1).strip()
    return candidate != raw and apply_model_output(state, candidate)["action_valid"]


@dataclass
class BInputs:
    corpus: list
    tasks: dict
    s30: list
    step0: dict
    receipt: dict


def check_grid(corpus, tasks, selected, references):
    wanted = {(tid, mode) for tid in selected for mode in MODES}
    need(references.keys() == wanted, "incomplete B step 0 reference grid")
    replay = HeldoutReplay([tasks[tid] for tid in selected], corpus, STRATEGY, budgets=BUDGET)
    grid = []
    for item in replay.next_batch():
        tid, mode = item["task"], item["mode"]
        h = validate_step0(user_text(item["state"]), references[tid,mode])
        grid.append({"task":tid, "mode":mode, "tier":"mid",
                     "n_steps":len(tasks[tid]["steps"]), "user_text_sha256":h})
    need(len(grid) == len(wanted), "incomplete B step 0 generated grid")
    return {"n_trajectories":len(grid), "n_steps":sum(r["n_steps"] for r in grid), "rows":grid,
            "step0_mismatches":0, "system_sha256":text_sha(SYSTEM)}


def prepare_b(corpus, tasks, s30):
    selected = list(s30)
    need(len(tasks) == 249 and selected == random.Random(20261127).sample(sorted(tasks),60)[:30],
         "B must use the registered first 30 sampled held-out tasks")
    tasks = {tid:{**tasks[tid], "task":tid} for tid in selected}
    index = sums(PROJECT/"SHA256SUMS")
    source, = (s for s in historical_sources(index) if s.name == BASELINES[0])
    archive = checked_json(source.metrics,source.metrics_sha256)["steps_artifact"]
    need(archive["path"] == str(source.steps) and archive["sha256"] == source.steps_sha256,
         "B step 0 metrics reference changed")
    verify(source.steps,source.steps_sha256)
    wanted = set(selected)
    reference = {}
    for row in rows(source.steps):
        if row["task"] in wanted and row["step"] == 0 and row["budget_tier"] == "mid":
            key = row["task"], row["mode"]
            need(key not in reference and row["strategy"] == "checkpoint-125", "B step 0 reference identity")
            reference[key] = row["generation_metadata"]["user_text_sha256"]
    receipt = check_grid(corpus,tasks,selected,reference)
    receipt.update(reference_path=str(source.steps),reference_sha256=source.steps_sha256,
                   s30=selected,budget=BUDGET["mid"])
    return BInputs(corpus,tasks,selected,reference,receipt)


def run_b_task(prepared, task_id, mode, generate):
    """One trajectory, sequential network callback; the caller records full transport receipts."""
    need(task_id in prepared.s30 and mode in MODES, "unregistered B task or mode")
    replay = HeldoutReplay([prepared.tasks[task_id]],prepared.corpus,STRATEGY,budgets=BUDGET,modes=(mode,))
    records = []
    while not replay.completed():
        item, = replay.next_batch()
        step = item["step"]
        state = item["state"]
        text = user_text(state)
        fingerprint = (validate_step0(text,prepared.step0[task_id,mode]) if step == 0 else text_sha(text))
        messages = [{"role":"system","content":SYSTEM},{"role":"user","content":text}]
        result = generate(messages, task_id, mode, step)
        raw = result["content"]
        need(isinstance(raw,str), "missing compressor content")
        tokens = result["usage"]
        need(type(tokens["completion_tokens"]) is int and 0 <= tokens["completion_tokens"] <= 256,
             "compressor completion token count out of bounds")
        generation = {"raw_output":raw,"generated_tokens":tokens["completion_tokens"],
                      "prompt_tokens":tokens["prompt_tokens"],"finish_reason":result["finish_reason"],
                      "api_id":result["id"],"user_text_sha256":fingerprint,
                      "system_sha256":text_sha(SYSTEM)}
        record, = replay.advance_batch([generation])
        record["fence_or_whitespace_rescued"] = diagnostic_rescue(state,raw,record["invalid_reason"])
        record["actual_add_count"] = len(set(record["keep_after"]).difference(record["kept_before"]))
        records.append(record)
    return {"task":task_id,"mode":mode,"tier":"mid","steps":records,
            "final_pids":list(records[-1]["keep_after"])}


def checked_aggregate(prepared, groups, strategy):
    tids = sorted({tid for tid,mode in groups})
    modes = [mode for mode in MODES if any(mode == m for t,m in groups)]
    need(tids and groups.keys() == {(tid,mode) for tid in tids for mode in modes},
         "incomplete selected trajectory grid")
    need(set(tids) <= set(prepared.s30), "B metrics are restricted to S30")
    replay = HeldoutReplay([prepared.tasks[tid] for tid in tids],prepared.corpus,strategy,
                           budgets=BUDGET,modes=modes)
    for (tid,mode), records in groups.items():
        need(len(records) == len(prepared.tasks[tid]["steps"]), "trajectory step count differs")
    counts = {}
    for mode in (*modes,"combined"):
        counts[mode] = {"n":0, "add_counts":Counter(), "used_after_sum":0, "fence_or_whitespace_rescued":0}
    while not replay.completed():
        batch = replay.next_batch()
        records = [groups[i["task"],i["mode"]][i["step"]] for i in batch]
        generations = []
        for row in records:
            meta = row["generation_metadata"]
            generations.append({"raw_output":row["raw_output"],
                                **{k:meta[k] for k in ("generated_tokens","prompt_tokens","finish_reason") if k in meta}})
        computed = replay.advance_batch(generations)
        for item,row,check in zip(batch,records,computed,strict=True):
            fields = ("task","mode","step","budget_tier","keep_after","kept_before","used_after",
                      "recall","invalid_reason","action_valid","returned","gold")
            need(all(row[k] == check[k] for k in fields), "B archived action or context disagrees with replay")
            added = len(set(check["keep_after"]).difference(check["kept_before"]))
            rescued = diagnostic_rescue(item["state"],row["raw_output"],check["invalid_reason"])
            if strategy == STRATEGY:
                need(row["actual_add_count"] == added and row["fence_or_whitespace_rescued"] == rescued
                     and row["generation_metadata"]["user_text_sha256"] == text_sha(user_text(item["state"])),
                     "B generation prompt, action or diagnostic changed")
            for scope in (row["mode"],"combined"):
                counts[scope]["n"] += 1
                counts[scope]["used_after_sum"] += row["used_after"]
                counts[scope]["add_counts"][added] += 1
                counts[scope]["fence_or_whitespace_rescued"] += rescued
    result = replay.aggregate()
    for scope,summary in counts.items():
        group = result["combined"] if scope == "combined" else result["configs"][f"{strategy}|{scope}|mid"]["metrics"]
        n = group["n_steps"]
        need(n == summary["n"], "B diagnostic denominator differs")
        group["actual_add_distribution"] = {str(k):v for k,v in sorted(summary["add_counts"].items())}
        group["used_after"] = {"sum":summary["used_after_sum"], "n":n, "mean":summary["used_after_sum"]/n}
        group["fence_or_whitespace_rescued"] = {"count":summary["fence_or_whitespace_rescued"], "n":n,
                                               "n_invalid":group["invalid_outputs"]["any_invalid"]["count"]}
    return result


def aggregate_b(prepared, trajectories):
    groups = {}
    for traj in trajectories:
        key = traj["task"],traj["mode"]
        need(key not in groups and traj["tier"] == "mid" and traj["steps"]
             and traj["final_pids"] == traj["steps"][-1]["keep_after"], "duplicate or invalid B trajectory")
        groups[key] = traj["steps"]
    return checked_aggregate(prepared,groups,STRATEGY)


def baseline_metrics(prepared, task_ids):
    """Recompute r2, v3, oracle and bm25 on exactly the supplied subset of S30."""
    selected = set(task_ids)
    need(selected and selected <= set(prepared.s30), "B baseline task IDs outside the registered subset")
    index = sums(PROJECT/"SHA256SUMS")
    result = {}
    for src in historical_sources(index):
        if src.name not in BASELINES:
            continue
        verify(src.steps,src.steps_sha256)
        groups = {(tid,mode):[] for tid in selected for mode in MODES}
        for row in rows(src.steps):
            key = row["task"],row["mode"]
            if key in groups and row["budget_tier"] == "mid":
                need(len(groups[key]) == row["step"], "baseline step order differs")
                groups[key].append(row)
        result[src.name] = checked_aggregate(prepared,groups,src.name)
    need(result.keys() == set(BASELINES), "B baseline coverage")
    return result

