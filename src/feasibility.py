"""Offline metrics for TASK_02; no model, network, or training dependencies.

Step recall follows PLAN's currently available gold. Final recall and loss use
the task's complete, deduplicated gold pid set. Costs are evaluated separately
for each task after converting every trajectory-local ID to its corpus pid.

Loss buckets by hop or question position count each step's gold occurrence.
A pid shared by several hops contributes to each relevant occurrence/bucket;
the task-loss count in any one bucket is still binary. Mechanism buckets count
unique gold pids within each task, using all tasks as their task denominator.
Only evidence missing at the end is classified as never_kept or dropped.
"""

import re
import string

import numpy as np

from eval.cost import trajectory_cost


def distribution(values):
    """Return n/mean/linear percentiles/max; empty distributions use None."""
    values = np.asarray(list(values), dtype=float)
    keys = ("mean", "p10", "p50", "p90", "p99", "max")
    if values.size == 0:
        return {"n": 0, **dict.fromkeys(keys)}
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("distribution requires a finite, one-dimensional sequence")
    p10, p50, p90, p99 = np.percentile(values, [10, 50, 90, 99])
    return {"n": int(values.size), "mean": float(np.mean(values)),
            "p10": float(p10), "p50": float(p50), "p90": float(p90),
            "p99": float(p99), "max": float(np.max(values))}


def normalize_answer(text):
    """SQuAD normalization: lowercase, ASCII punctuation, articles, spaces."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_answer(text, answer):
    """Find a normalized answer at word boundaries; an empty answer never hits."""
    normalized_answer = normalize_answer(answer)
    if not normalized_answer:
        return False
    return re.search(r"(?<!\w)" + re.escape(normalized_answer) + r"(?!\w)",
                     normalize_answer(text)) is not None


def _ratio(count, n):
    return count / n if n else None


def _bucket(occurrences, missing):
    """One task's contribution to a hop/position bucket."""
    occurrences = list(occurrences)
    lost = sum(pid in missing for pid in occurrences)
    return {"n": int(bool(occurrences)), "count": int(bool(lost)),
            "evidence_n": len(occurrences), "lost_evidence_count": lost}


def summarize_task(samples, steps, question_gold_sets, pid_ntok, budget):
    """Summarize one completed rollout against all of its question evidence.

    ``samples`` are the existing build_traj.rollout states, with integer local
    IDs and ``pid_by_lid``. ``steps`` use zero-based question_index/hop_index and
    corpus ``gold`` pids. ``question_gold_sets`` is in task-question order.
    ``pid_ntok`` maps corpus pids to rendered-fragment token counts.

    Invalid actions and over-budget states are counted, not silently excluded;
    the experiment runner must assert both counts equal zero.
    """
    samples, steps = list(samples), list(steps)
    question_gold_sets = [set(gold) for gold in question_gold_sets]
    if not samples or len(samples) != len(steps):
        raise ValueError("one nonempty rollout sample is required per step")
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if not question_gold_sets or any(not g for g in question_gold_sets):
        raise ValueError("each question must have at least one gold pid")
    all_gold = set().union(*question_gold_sets)
    if all_gold != {step["gold"] for step in steps}:
        raise ValueError("question gold and step gold sets differ")
    for step in steps:
        q, hop = step["question_index"], step["hop_index"]
        if not 0 <= q < len(question_gold_sets) or not 0 <= hop < 4:
            raise ValueError("invalid zero-based question/hop index")
        if step["gold"] not in question_gold_sets[q]:
            raise ValueError("step gold is absent from its question gold set")
    if len(question_gold_sets) > 4:
        raise ValueError("TASK_02 supports at most four questions per task")

    contexts = [[s["pid_by_lid"][lid] for lid in s["keep_after"]]
                for s in samples]
    if any(len(cur) != len(set(cur)) for cur in contexts):
        raise ValueError("duplicate retained corpus pid")
    final = set(contexts[-1])
    ever_kept = set().union(*(set(cur) for cur in contexts))
    missing = all_gold - final
    never_kept = missing - ever_kept
    dropped = missing & ever_kept
    used = [sum(pid_ntok[pid] for pid in cur) for cur in contexts]
    step_recalls = [float(s["recall"]) for s in samples]
    if any(not 0 <= value <= 1 for value in step_recalls):
        raise ValueError("step recall must be within [0, 1]")
    plain, cached = trajectory_cost(contexts, pid_ntok)

    by_hop = {str(i + 1): _bucket((step["gold"] for step in steps
                                  if step["hop_index"] == i), missing)
              for i in range(4)}
    by_position = {str(i + 1): _bucket((step["gold"] for step in steps
                                       if step["question_index"] == i), missing)
                   for i in range(4)}
    by_mechanism = {
        name: {"n": 1, "count": int(bool(pids)), "evidence_n": len(all_gold),
               "lost_evidence_count": len(pids)}
        for name, pids in (("never_kept", never_kept), ("dropped", dropped))
    }
    return {
        "n_steps": len(samples), "n_questions": len(question_gold_sets),
        "step_recall_sum": sum(step_recalls),
        "step_recall_mean": sum(step_recalls) / len(samples),
        "all_gold_count": len(all_gold),
        "retained_gold_count": len(all_gold & final),
        "final_recall": len(all_gold & final) / len(all_gold),
        "evidence_loss": bool(missing), "missing_pids": sorted(missing),
        "never_kept_pids": sorted(never_kept), "dropped_pids": sorted(dropped),
        "complete_question_count": sum(gold <= final for gold in question_gold_sets),
        "final_used": used[-1], "final_occupancy": used[-1] / budget,
        "invalid_action_count": sum(not s["action_valid"] for s in samples),
        "over_budget_count": sum(count > budget for count in used),
        "cost_plain": float(plain), "cost_cached": float(cached),
        "loss_by_hop": by_hop, "loss_by_question_position": by_position,
        "loss_by_mechanism": by_mechanism,
    }


def _aggregate_bucket(buckets):
    buckets = list(buckets)
    out = {key: sum(bucket[key] for bucket in buckets)
           for key in ("n", "count", "evidence_n", "lost_evidence_count")}
    out["rate"] = _ratio(out["count"], out["n"])
    return out


def aggregate_tasks(metrics, baseline=None):
    """Aggregate task summaries, optionally using a no_compress aggregate.

    Every average/rate includes its denominator. ``step_recall.micro`` is an
    average over steps (not a ratio of pooled gold hits); ``task_macro`` gives
    each task equal weight. Cost ratios divide configuration and no_compress
    mean costs from matching task sets; no cache state crosses task boundaries.
    """
    metrics = list(metrics)
    n_tasks = len(metrics)
    n_steps = sum(m["n_steps"] for m in metrics)
    n_questions = sum(m["n_questions"] for m in metrics)

    def mean(key):
        return _ratio(sum(m[key] for m in metrics), n_tasks)

    loss_count = sum(m["evidence_loss"] for m in metrics)
    complete_count = sum(m["complete_question_count"] for m in metrics)
    loss = {"n": n_tasks, "count": loss_count, "rate": _ratio(loss_count, n_tasks)}
    for name, keys in (("hop", ("1", "2", "3", "4")),
                       ("question_position", ("1", "2", "3", "4")),
                       ("mechanism", ("never_kept", "dropped"))):
        loss["by_" + name] = {
            key: _aggregate_bucket(m["loss_by_" + name][key] for m in metrics)
            for key in keys
        }
    cost = {}
    if baseline is not None and baseline["n_tasks"] != n_tasks:
        raise ValueError("cost baseline must contain the same number of tasks")
    for name in ("plain", "cached"):
        value = mean("cost_" + name)
        base = baseline["cost"][name]["mean"] if baseline is not None else None
        cost[name] = {"n": n_tasks, "mean": value,
                      "relative_to_no_compress": value / base if value is not None and base else None}
    return {
        "n_tasks": n_tasks, "n_steps": n_steps,
        "step_recall": {"micro": {"n": n_steps,
                                    "mean": _ratio(sum(m["step_recall_sum"] for m in metrics), n_steps)},
                        "task_macro": {"n": n_tasks, "mean": mean("step_recall_mean")}},
        "final_recall": {"n": n_tasks, "mean": mean("final_recall")},
        "evidence_loss": loss,
        "evidence_complete_questions": {"n": n_questions, "count": complete_count,
                                        "rate": _ratio(complete_count, n_questions)},
        "occupancy": {"n": n_tasks, "mean": mean("final_occupancy"),
                      "max": max((m["final_occupancy"] for m in metrics), default=None)},
        "action_validity": {"n": n_steps,
                            "invalid_count": sum(m["invalid_action_count"] for m in metrics),
                            "over_budget_count": sum(m["over_budget_count"] for m in metrics)},
        "cost": cost,
    }
