"""Task 05 saved-step replay and metric aggregation, without model imports.

All inference backends use this state machine. JSON is parsed as a whole (no
markdown extraction or repair); action semantics then come from apply_action.
The four mutually exclusive invalid reasons use the order listed below.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

from src.build_traj import apply_action, make_sample
from src.feasibility import aggregate_tasks, summarize_task

BUDGETS = {"tight": 1876, "mid": 2814, "loose": 5628}
MODES = ("reselect", "append")
INVALID_REASONS = ("invalid_json", "invalid_candidate_id", "append_drop", "over_budget")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def parse_action(raw_output):
    """Whole-string strict JSON; preserve existing add/drop action grammar.

add is required; omitted drop has the existing empty-list meaning. Additional
JSON keys are ignored by apply_action, as in Task 01. Duplicate IDs retain set
semantics. Booleans/floats are not integer IDs. Duplicate object keys and the
non-JSON NaN/Infinity extensions are rejected instead of silently normalized.
"""
    if not isinstance(raw_output, str):
        return None
    try:
        action = json.loads(raw_output, object_pairs_hook=_json_object,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(action, dict) or "add" not in action:
        return None
    add, drop = action["add"], action.get("drop", [])
    if not isinstance(add, list) or not isinstance(drop, list):
        return None
    if any(type(item) is not int for item in add + drop):
        return None
    return action


def apply_model_output(state, raw_output):
    """Parse, classify, and execute; invalid outputs leave retained IDs intact."""
    action = parse_action(raw_output)
    invalid_reason = None
    keep = None
    if action is None:
        invalid_reason = "invalid_json"
    else:
        add, drop = set(action["add"]), set(action.get("drop", []))
        if not add <= set(state["new_ids"]) or not drop <= set(state["kept_ids"]):
            invalid_reason = "invalid_candidate_id"
        elif state["mode"] == "append" and drop:
            invalid_reason = "append_drop"
        else:
            keep = apply_action(state, action)
            if keep is None:
                invalid_reason = "over_budget"
    if invalid_reason is not None:
        keep = list(state["kept_ids"])
    gold = set(state["gold"])
    return {"action": action, "parsed_action": action,
            "action_valid": invalid_reason is None, "invalid_reason": invalid_reason,
            "keep_after": keep, "used_after": sum(state["ntok"][i] for i in keep),
            "recall": len(gold.intersection(keep)) / len(gold) if gold else 1.0}


def load_heldout_assets(data_dir, expected_tasks=249):
    """Load only saved train corpus and heldout steps, discarding teacher states.

No retrieval, raw train/dev dataset, teacher trajectory or SFT dev file is
needed to evaluate. Corpus IDs are verified rather than silently reindexed.
"""
    data_dir = Path(data_dir)
    corpus = []
    with (data_dir / "corpus_train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["pid"] != len(corpus):
                raise ValueError("train corpus IDs must be contiguous and source ordered")
            corpus.append(row)
    tasks = []
    with (data_dir / "heldout_tasks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "heldout":
                raise ValueError("only heldout train tasks may be evaluated")
            tasks.append({key: row[key] for key in ("task", "split", "example_ids", "questions", "steps")})
    if len(tasks) != expected_tasks or len({t["task"] for t in tasks}) != expected_tasks:
        raise ValueError(f"expected exactly {expected_tasks} distinct heldout tasks")
    ids = [example for task in tasks for example in task["example_ids"]]
    if len(ids) != len(set(ids)):
        raise ValueError("heldout questions may only occur once")
    for task in tasks:
        if len(task["questions"]) != 4 or not task["steps"]:
            raise ValueError("heldout task requires four questions and saved steps")
        for step in task["steps"]:
            if step["gold"] not in step["new"] or any(not 0 <= pid < len(corpus) for pid in step["new"]):
                raise ValueError("saved retrieval does not match train corpus")
    return tasks, corpus


def _rate(count, n):
    return {"count": count, "n": n, "rate": count / n if n else None}


class HeldoutReplay:
    """Replay all active trajectories one step per batch using their own outputs.

next_batch is idempotent until advance_batch; a caller must supply exactly one
output per pending item in the returned order. No model/torch/vLLM is imported.
Only compact metric samples are retained between steps, never prompt text.
"""
    def __init__(self, tasks, corpus, strategy, budgets=None, modes=MODES, max_tokens=256):
        self.tasks, self.corpus, self.strategy = list(tasks), corpus, strategy
        self.budgets = dict(BUDGETS if budgets is None else budgets)
        self.modes, self.max_tokens = tuple(modes), max_tokens
        if not self.tasks or len({t["task"] for t in self.tasks}) != len(self.tasks):
            raise ValueError("tasks must be nonempty with unique IDs")
        if not self.budgets or any(type(v) is not int or v <= 0 for v in self.budgets.values()):
            raise ValueError("positive integer budgets required")
        if not self.modes or len(set(self.modes)) != len(self.modes) or any(m not in MODES for m in self.modes):
            raise ValueError("distinct append/reselect modes required")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.pid_ntok = {pid: paragraph["ntok"] for pid, paragraph in enumerate(corpus)}
        nonbinding = max(sum(self.pid_ntok[p] for p in {p for step in task["steps"] for p in step["new"]})
                         for task in self.tasks) + 1
        self.trajectories = []
        for mode in self.modes:
            for tier, budget in self.budgets.items():
                for task in self.tasks:
                    effective = nonbinding if strategy == "no_compress" else budget
                    self.trajectories.append({"task_data": task, "mode": mode, "budget_tier": tier,
                        "budget": budget, "effective_budget": effective,
                        "trajectory_id": f"{strategy}|{mode}|{tier}|{task['task']}",
                        "config": f"{strategy}|{mode}|{tier}", "step": 0,
                        "kept": [], "gold_upto": set(), "lid": {}, "samples": [],
                        "invalid": Counter(), "output_limit_hits": 0,
                        "generated_tokens": 0, "prompt_tokens": 0})
        self._pending = None

    def completed(self):
        return all(t["step"] == len(t["task_data"]["steps"]) for t in self.trajectories)

    def next_batch(self):
        if self._pending is None:
            self._pending = []
            for trajectory in self.trajectories:
                task, step = trajectory["task_data"], trajectory["step"]
                if step >= len(task["steps"]):
                    continue
                returned = task["steps"][step]
                trajectory["gold_upto"].add(returned["gold"])
                state = make_sample(task["questions"], trajectory["kept"], returned["new"],
                                    trajectory["gold_upto"], self.corpus,
                                    trajectory["effective_budget"], trajectory["mode"], trajectory["lid"])
                item = {key: trajectory[key] for key in
                        ("trajectory_id", "config", "mode", "budget_tier", "budget", "effective_budget", "step")}
                item.update(task=task["task"], state=state)
                self._pending.append((trajectory, item))
        return [item for _, item in self._pending]

    def advance_batch(self, outputs):
        outputs = list(outputs)
        if self._pending is None:
            raise RuntimeError("call next_batch before advancing")
        if len(outputs) != len(self._pending):
            raise ValueError("exactly one output is required per pending trajectory")
        # Validate the whole batch before changing any trajectory.
        for output in outputs:
            if not isinstance(output, dict) or "raw_output" not in output:
                raise ValueError("each output requires raw_output")
            for key in ("generated_tokens", "prompt_tokens"):
                if key in output and (type(output[key]) is not int or output[key] < 0):
                    raise ValueError(f"{key} must be a nonnegative integer")
        records = []
        for (trajectory, item), output in zip(self._pending, outputs):
            state = item["state"]
            executed = apply_model_output(state, output["raw_output"])
            sample = {**{key: value for key, value in state.items() if key != "text"}, **executed}
            mapping = state["pid_by_lid"]
            kept_pids = [mapping[i] for i in executed["keep_after"]]
            if state["mode"] == "append" and kept_pids[:len(trajectory["kept"])] != trajectory["kept"]:
                raise AssertionError("append prefix changed")
            limit_hit = (output.get("generated_tokens", 0) >= self.max_tokens
                         or output.get("finish_reason") == "length")
            returned = trajectory["task_data"]["steps"][item["step"]]
            record = {**{key: value for key, value in item.items() if key != "state"},
                      "schema_version": 1, "strategy": self.strategy, "state": sample,
                      "raw_output": output["raw_output"], "parsed_action": executed["parsed_action"],
                      "invalid_reason": executed["invalid_reason"], "action_valid": executed["action_valid"],
                      "output_limit_hit": limit_hit,
                      "generation_metadata": {key: value for key, value in output.items() if key != "raw_output"},
                      "kept_before": list(trajectory["kept"]), "new": [mapping[i] for i in state["new_ids"]],
                      "returned": returned["new"], "gold": returned["gold"],
                      "question_index": returned["question_index"], "hop_index": returned["hop_index"],
                      "keep_after": kept_pids, "used_after": executed["used_after"],
                      "recall": executed["recall"]}
            records.append(record)
            trajectory["kept"] = kept_pids
            trajectory["samples"].append({key: sample[key] for key in
                ("pid_by_lid", "keep_after", "recall", "action_valid")})
            trajectory["step"] += 1
            if executed["invalid_reason"] is not None:
                trajectory["invalid"][executed["invalid_reason"]] += 1
            trajectory["output_limit_hits"] += limit_hit
            trajectory["generated_tokens"] += output.get("generated_tokens", 0)
            trajectory["prompt_tokens"] += output.get("prompt_tokens", 0)
        self._pending = None
        return records

    def aggregate(self, baseline=None):
        if not self.completed():
            raise RuntimeError("all trajectories must finish before aggregate")
        grouped = defaultdict(list)
        for trajectory in self.trajectories:
            task = trajectory["task_data"]
            question_gold = [set() for _ in task["questions"]]
            for step in task["steps"]:
                question_gold[step["question_index"]].add(step["gold"])
            summary = summarize_task(trajectory["samples"], task["steps"], question_gold,
                                     self.pid_ntok, trajectory["effective_budget"])
            summary.update(task=task["task"], mode=trajectory["mode"], budget_tier=trajectory["budget_tier"],
                           requested_occupancy=summary["final_used"] / trajectory["budget"],
                           invalid_counts=dict(trajectory["invalid"]), output_limit_hits=trajectory["output_limit_hits"],
                           generated_tokens=trajectory["generated_tokens"], prompt_tokens=trajectory["prompt_tokens"])
            grouped[trajectory["config"]].append(summary)

        def aggregate_group(summaries, base=None):
            aggregate = aggregate_tasks(summaries, baseline=base)
            n = aggregate["n_steps"]
            aggregate["invalid_outputs"] = {
                reason: _rate(sum(row["invalid_counts"].get(reason, 0) for row in summaries), n)
                for reason in INVALID_REASONS}
            aggregate["invalid_outputs"]["any_invalid"] = _rate(
                sum(row["invalid_action_count"] for row in summaries), n)
            aggregate["output_limit"] = _rate(sum(row["output_limit_hits"] for row in summaries), n)
            aggregate["requested_budget_occupancy"] = {"n": len(summaries),
                "mean": sum(row["requested_occupancy"] for row in summaries) / len(summaries),
                "max": max(row["requested_occupancy"] for row in summaries)}
            aggregate["generation"] = {"n": n,
                "output_tokens": sum(row["generated_tokens"] for row in summaries),
                "prompt_tokens": sum(row["prompt_tokens"] for row in summaries)}
            return aggregate

        configs = {}
        for name, rows in grouped.items():
            mode, tier = rows[0]["mode"], rows[0]["budget_tier"]
            base = baseline["configs"][f"no_compress|{mode}|{tier}"]["metrics"] if baseline else None
            configs[name] = {"policy": self.strategy, "strategy": self.strategy, "mode": mode,
                             "budget_tier": tier, "budget": self.budgets[tier],
                             "budget_binding": self.strategy != "no_compress",
                             "metrics": aggregate_group(rows, base)}
        combined = aggregate_group([row for rows in grouped.values() for row in rows],
                                   baseline["combined"] if baseline else None)
        return {"schema_version": 1, "strategy": self.strategy,
                "settings": {"n_tasks": len(self.tasks), "n_trajectories": len(self.trajectories),
                             "modes": list(self.modes), "budgets": self.budgets,
                             "split": "train-heldout", "saved_retrieval_steps": True,
                             "invalid_reason_priority": list(INVALID_REASONS),
                             "max_tokens": self.max_tokens, "r_read": 0.05, "r_write": 1.25,
                             "no_compress": "nonbinding reference; requested and effective budgets recorded"},
                "configs": configs, "combined": combined}


def select_checkpoint(results):
    """Select by all 1494 final-task loss outcomes, final recall, then step.

Each input is an aggregate result plus an integer checkpoint_step. Baselines
and raw model results are excluded by the caller, never ranked as checkpoints.
"""
    results = list(results)
    if not results:
        raise ValueError("checkpoint results required")
    for result in results:
        if type(result.get("checkpoint_step")) is not int or result["checkpoint_step"] <= 0:
            raise ValueError("positive integer checkpoint_step required")
        if (result["settings"]["n_tasks"] != 249 or len(result["configs"]) != 6
                or result["combined"]["n_tasks"] != 1494):
            raise ValueError("checkpoint selection requires 249 tasks in every one of six cells")
        if any(config["metrics"]["n_tasks"] != 249 for config in result["configs"].values()):
            raise ValueError("each checkpoint configuration must contain 249 tasks")
    return min(results, key=lambda result: (
        result["combined"]["evidence_loss"]["count"],
        -result["combined"]["final_recall"]["mean"], result["checkpoint_step"]))
