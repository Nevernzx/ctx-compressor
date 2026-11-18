#!/usr/bin/env python3
"""Independently audit saved TASK_02 results, without production imports.

Only standard library and installed tokenizers are used. Inputs are read-only;
only independent_check.json/md are written. Run on 5880 via run_python.sh.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import string
import time
import traceback

if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])

from tokenizers import Tokenizer


def jsonlines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def percentile(values, q):
    ordered = sorted(values)
    x = (len(ordered) - 1) * q / 100
    low, high = math.floor(x), math.ceil(x)
    return ordered[low] + (ordered[high] - ordered[low]) * (x - low)


def distribution(values):
    values = list(values)
    if not values:
        return {"n": 0, **dict.fromkeys(("mean", "p10", "p50", "p90", "p99", "max"))}
    return {"n": len(values), "mean": sum(values) / len(values),
            **{f"p{q}": percentile(values, q) for q in (10, 50, 90, 99)}, "max": max(values)}


def ratio(hits, n):
    return {"hits": hits, "n": n, "rate": hits / n if n else None}


def normalize(text):
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(?:a|an|the)\b", " ", text).split())


class Checks:
    def __init__(self):
        self.count, self.mismatches = 0, []

    def compare(self, expected, actual, path):
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key, value in expected.items():
                self.compare(value, actual.get(key, "<MISSING>"), f"{path}.{key}")
            return
        if isinstance(expected, list) and isinstance(actual, list):
            self.compare(len(expected), len(actual), path + ".length")
            for i, (a, b) in enumerate(zip(expected, actual)):
                self.compare(a, b, f"{path}.{i}")
            return
        self.count += 1
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            matches = (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                       and math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10))
        else:
            matches = expected == actual
        if not matches:
            self.mismatches.append({"path": path, "expected": expected, "actual": actual})


def decompose(example, key2pid):
    answers, steps = [], []
    paragraphs = {p["idx"]: p for p in example["paragraphs"]}
    for hop in example["question_decomposition"]:
        def replace(match):
            k = int(match.group(1))
            if not 1 <= k <= len(answers):
                raise ValueError("invalid previous-hop reference")
            return answers[k - 1]
        query = re.sub(r"#(\d+)", replace, hop["question"])
        if re.search(r"#\d", query) or not isinstance(hop["answer"], str):
            raise ValueError("invalid resolved query or answer")
        support = hop["paragraph_support_idx"]
        if type(support) is not int:
            raise ValueError("invalid support index")
        p = paragraphs[support]
        steps.append({"query": query, "gold": key2pid[(p["title"], p["paragraph_text"])]})
        answers.append(hop["answer"])
    if not steps:
        raise ValueError("empty decomposition")
    return steps


def first_fit(order, lengths, budget, kept=()):
    selected = set(kept)
    used = sum(lengths[p] for p in selected)
    for pid in order:
        if used + lengths[pid] <= budget:
            selected.add(pid)
            used += lengths[pid]
    return selected


def context_cost(contexts, lengths):
    plain = cached = 0.0
    previous = []
    for current in contexts:
        common = 0
        for before, after in zip(previous, current):
            if before != after:
                break
            common += 1
        hit = sum(lengths[p] for p in current[:common])
        fresh = sum(lengths[p] for p in current[common:])
        plain += hit + fresh
        cached += 0.05 * hit + 1.25 * fresh
        previous = current
    return plain, cached


def render_for_token_check(task, row, lid, texts, lengths):
    lines = ["任务问题："]
    lines += [f"{i + 1}. {q['question']}" for i, q in enumerate(task["questions"])]
    used = sum(lengths[p] for p in row["kept_before"])
    lines += [f"模式：{row['mode']}", f"预算：{row['budget']} token", f"已用 token：{used}", "", "已保留："]
    for pids, heading in ((row["kept_before"], None), (row["new"], "新返回：")):
        if heading:
            lines += ["", heading]
        if not pids:
            lines.append("（空）")
        for pid in pids:
            lines += [f"[编号 {lid[pid]} | {lengths[pid]} token]", texts[pid]]
    return "\n".join(lines)


def aggregate(metrics, baseline=None):
    n = len(metrics)
    nsteps = sum(len(m["recalls"]) for m in metrics)
    mean = lambda key: sum(m[key] for m in metrics) / n
    loss = {"n": n, "count": sum(bool(m["missing"]) for m in metrics)}
    loss["rate"] = loss["count"] / n
    for axis, keys in (("hop", ("1", "2", "3", "4")),
                       ("question_position", ("1", "2", "3", "4")),
                       ("mechanism", ("never_kept", "dropped"))):
        buckets = {}
        for key in keys:
            parts = [m["buckets"][axis][key] for m in metrics]
            total = {f: sum(b[f] for b in parts) for f in ("n", "count", "evidence_n", "lost_evidence_count")}
            total["rate"] = total["count"] / total["n"] if total["n"] else None
            buckets[key] = total
        loss["by_" + axis] = buckets
    cost = {}
    for field in ("plain", "cached"):
        value = mean(field)
        base = baseline["cost"][field]["mean"] if baseline else None
        cost[field] = {"n": n, "mean": value, "relative_to_no_compress": value / base if base else None}
    complete = sum(m["complete"] for m in metrics)
    return {"n_tasks": n, "n_steps": nsteps,
            "step_recall": {"micro": {"n": nsteps, "mean": sum(sum(m["recalls"]) for m in metrics) / nsteps},
                            "task_macro": {"n": n, "mean": mean("recall_mean")}},
            "final_recall": {"n": n, "mean": mean("final_recall")}, "evidence_loss": loss,
            "evidence_complete_questions": {"n": 4 * n, "count": complete, "rate": complete / (4 * n)},
            "occupancy": {"n": n, "mean": mean("occupancy"), "max": max(m["occupancy"] for m in metrics)},
            "action_validity": {"n": nsteps, "invalid_count": 0, "over_budget_count": 0}, "cost": cost}


def audit(base, checks):
    fd, out = base / "data/feasibility", base / "eval/feasibility"
    results = json.loads((out / "feasibility_results.json").read_text())
    config = json.loads((out / "run_config.json").read_text())
    checks.compare(150, results["settings"]["n_tasks"], "settings.n_tasks")
    checks.compare(20261117, results["settings"]["seed"], "settings.seed")
    tokenizer = Tokenizer.from_file(str(base / "data/raw/qwen3_tokenizer/tokenizer.json"))
    dev = list(jsonlines(base / "data/raw/musique/musique_ans_v1.0_dev.jsonl"))
    by_id = {ex["id"]: ex for ex in dev}
    key2pid, titles, bodies, texts = {}, [], [], []
    for example in dev:
        for p in example["paragraphs"]:
            key = p["title"], p["paragraph_text"]
            if key not in key2pid:
                key2pid[key] = len(texts)
                titles.append(key[0]); bodies.append(key[1]); texts.append(key[0] + "\n" + key[1])
    lengths = {}
    delivered = list(jsonlines(fd / "corpus_dev_ntok.jsonl"))
    checks.compare(len(texts), len(delivered), "corpus.n")
    for pid, row in enumerate(delivered):
        checks.compare(pid, row["pid"], f"corpus.{pid}.pid")
        checks.compare(titles[pid], row["title"], f"corpus.{pid}.title")
        checks.compare(hashlib.sha1(texts[pid].encode()).hexdigest(), row["sha1"], f"corpus.{pid}.sha1")
        lengths[pid] = len(tokenizer.encode(texts[pid], add_special_tokens=False).ids)
        checks.compare(lengths[pid], row["ntok"], f"corpus.{pid}.ntok")
    valid, rejected = [], []
    for ex in dev:
        try:
            decompose(ex, key2pid)
        except (KeyError, TypeError, ValueError):
            rejected.append(ex["id"])
        else:
            valid.append(ex)
    checks.compare(len(rejected), results["dev_filter"]["rejected"], "dev_filter.rejected")
    checks.compare(sorted(rejected), sorted(x["id"] for x in results["dev_filter"]["rejected_examples"]), "dev_filter.ids")
    tasks = list(jsonlines(fd / "tasks_dev150.jsonl"))
    checks.compare(150, len(tasks), "tasks.n")
    sampled = random.Random(20261117).sample(valid, 600)
    checks.compare([ex["id"] for ex in sampled], [i for task in tasks for i in task["example_ids"]], "tasks.exact_seeded_order")
    task_gold, question_gold = {}, {}
    for tid, task in enumerate(tasks):
        checks.compare(tid, task["task"], f"tasks.{tid}.id")
        expected_steps, qgold = [], []
        examples = [by_id[qid] for qid in task["example_ids"]]
        checks.compare([{k: ex[k] for k in ("id", "question", "answer", "answer_aliases")} for ex in examples], task["questions"], f"tasks.{tid}.questions")
        for qi, ex in enumerate(examples):
            qsteps = decompose(ex, key2pid)
            qgold.append({st["gold"] for st in qsteps})
            for hi, st in enumerate(qsteps):
                expected_steps.append({**st, "question_index": qi, "hop_index": hi, "example_id": ex["id"],
                                       "question_hops": len(qsteps), "answer": ex["question_decomposition"][hi]["answer"],
                                       "original_query": ex["question_decomposition"][hi]["question"]})
        checks.compare(len(expected_steps), len(task["steps"]), f"tasks.{tid}.steps")
        for si, (expected, row) in enumerate(zip(expected_steps, task["steps"])):
            checks.compare(expected, row, f"tasks.{tid}.steps.{si}")
            checks.compare(20, len(row["new"]), f"tasks.{tid}.{si}.returned_n")
            checks.compare(20, len(set(row["new"])), f"tasks.{tid}.{si}.returned_unique")
            checks.compare(True, row["gold"] in row["new"], f"tasks.{tid}.{si}.gold_returned")
        question_gold[tid] = qgold
        task_gold[tid] = set().union(*qgold)
        checks.compare([sorted(g) for g in qgold], task["question_gold"], f"tasks.{tid}.question_gold")
        checks.compare(sorted(task_gold[tid]), task["all_gold"], f"tasks.{tid}.all_gold")
        checks.compare([len(ex["question_decomposition"]) for ex in examples], task["question_hops"], f"tasks.{tid}.question_hops")
    print(f"Independent corpus/task checks: {len(texts)} paragraphs, {len(tasks)} tasks.", flush=True)

    series = {k: [] for k in ("task_raw_tokens", "task_unique_tokens", "task_steps", "question_hops", "task_evidence_tokens", "question_evidence_tokens")}
    hit, total = Counter(), Counter()
    normalized = [normalize(body) for body in bodies]
    rendered = [normalize(text) for text in texts]
    leaks = defaultdict(Counter)
    duplicate = different_title = empty_answer = 0
    for tid, task in enumerate(tasks):
        steps = task["steps"]
        series["task_raw_tokens"].append(sum(lengths[p] for st in steps for p in st["new"]))
        series["task_unique_tokens"].append(sum(lengths[p] for p in {p for st in steps for p in st["new"]}))
        series["task_steps"].append(len(steps))
        series["task_evidence_tokens"].append(sum(lengths[p] for p in task_gold[tid]))
        series["question_hops"].extend(len(by_id[i]["question_decomposition"]) for i in task["example_ids"])
        series["question_evidence_tokens"].extend(sum(lengths[p] for p in g) for g in question_gold[tid])
        for st in steps:
            for key in ("overall", "hop:" + str(st["hop_index"] + 1), "total:" + str(st["question_hops"])):
                hit[key] += st["natural_hit"]
                total[key] += 1
            answer = normalize(st["answer"])
            empty_answer += not answer
            pattern = re.compile(r"(?<!\w)" + re.escape(answer) + r"(?!\w)") if answer else None
            others = [p for p in st["new"] if p not in task_gold[tid]]
            multi = len(answer.split()) >= 2
            for label, corpus_text in (("all", normalized), ("rendered_all", rendered)):
                found = sum(bool(pattern and pattern.search(corpus_text[p])) for p in others)
                buckets = [label]
                if multi:
                    buckets.append("multiword" if label == "all" else "rendered_multiword")
                for bucket in buckets:
                    leaks[bucket].update(paragraph_hits=found, paragraph_n=len(others), hop_hits=bool(found), hop_n=1)
            for pid in others:
                if normalized[pid] == normalized[st["gold"]]:
                    duplicate += 1
                    different_title += titles[pid] != titles[st["gold"]]
    checks.compare({k: distribution(v) for k, v in series.items()}, results["distributions"], "distributions")
    tight = math.ceil(percentile(series["task_evidence_tokens"], 90))
    budgets = {"tight": tight, "mid": math.ceil(1.5 * tight), "loose": 3 * tight}
    checks.compare(budgets, results["budgets"], "budgets")
    retrieval = {"overall": ratio(hit["overall"], total["overall"]),
                 "by_hop": {str(i): ratio(hit[f"hop:{i}"], total[f"hop:{i}"]) for i in range(1, 5)},
                 "by_question_hops": {str(i): ratio(hit[f"total:{i}"], total[f"total:{i}"]) for i in (2, 3, 4)}}
    checks.compare(retrieval, results["retrieval"], "retrieval")
    answer_leak = {name: {"paragraph": ratio(c["paragraph_hits"], c["paragraph_n"]),
                          "hop": ratio(c["hop_hits"], c["hop_n"])} for name, c in leaks.items()}
    answer_leak["body_duplicates"] = {"count": duplicate, "n": leaks["all"]["paragraph_n"],
                                      "rate": duplicate / leaks["all"]["paragraph_n"], "different_title_count": different_title}
    answer_leak["empty_normalized_answer_hops"] = empty_answer
    checks.compare(answer_leak, results["answer_leak"], "answer_leak")

    finals_list = list(jsonlines(fd / "final_contexts.jsonl"))
    finals = {(r["config"], r["task"]): r for r in finals_list}
    checks.compare(2850, len(finals_list), "final_contexts.n")
    checks.compare(len(finals_list), len(finals), "final_contexts.unique_keys")
    expected_configs = {"no_compress"} | {f"{p}|{m}|{b}" for p in ("latest", "bm25", "oracle") for m in ("reselect", "append") for b in budgets}
    checks.compare(sorted(expected_configs), sorted(results["configs"]), "configs.names")
    metrics, prompt_lengths, fragment_lengths = defaultdict(list), defaultdict(list), defaultdict(list)
    score_cache, visited = defaultdict(dict), set()
    row_number = prompt_checks = 0
    for (cfg, tid), group in itertools.groupby(jsonlines(fd / "steps.jsonl"), key=lambda r: (r["config"], r["task"])):
        key = cfg, tid
        checks.compare(False, key in visited, f"groups.{key}.duplicate")
        visited.add(key)
        task, spec = tasks[tid], results["configs"][cfg]
        policy, mode = spec["policy"], spec["mode"]
        budget = max(series["task_unique_tokens"]) + 1 if cfg == "no_compress" else budgets[cfg.split("|")[2]]
        checks.compare(budget, spec["budget"], f"configs.{cfg}.budget")
        previous, seen_gold, ever, contexts, recalls, lid = [], set(), set(), [], [], {}
        for si, row in enumerate(group):
            row_number += 1
            prefix = f"steps.{cfg}.{tid}.{si}"
            checks.compare(si, row["step"], prefix + ".step")
            raw = task["steps"][si]
            checks.compare({"policy": policy, "mode": mode, "budget": budget, "gold": raw["gold"],
                            "question_index": raw["question_index"], "hop_index": raw["hop_index"],
                            "kept_before": previous, "returned": raw["new"]}, row, prefix)
            new = [pid for pid in raw["new"] if pid not in set(previous)]
            candidates = previous + new
            checks.compare(new, row["new"], prefix + ".new")
            for pid in candidates:
                lid.setdefault(pid, len(lid))
            seen_gold.add(raw["gold"])
            available = [pid for pid in candidates if pid in seen_gold]
            checks.compare(available, row["available_gold"], prefix + ".available_gold")
            initial = previous if mode == "append" else []
            if policy == "no_compress":
                selected = set(candidates)
            elif policy == "latest":
                selected = first_fit(reversed(new if mode == "append" else candidates), lengths, budget, initial)
            elif policy == "bm25":
                scores = {int(k): v for k, v in row["bm25_scores"].items()}
                checks.compare(sorted(candidates), sorted(scores), prefix + ".score_ids")
                for pid, value in scores.items():
                    checks.compare(True, math.isfinite(value), prefix + f".score_finite.{pid}")
                    checks.compare(score_cache[tid].setdefault(pid, value), value, prefix + f".consistent_score.{pid}")
                position = {pid: i for i, pid in enumerate(candidates)}
                pool = new if mode == "append" else candidates
                priority = sorted(pool, key=lambda p: (-scores[p], -position[p]))
                selected = first_fit(priority, lengths, budget, initial)
            elif policy == "oracle":
                priority = (sorted(set(available) - set(previous), key=lambda p: (lengths[p], lid[p]))
                            if mode == "append" else sorted(available, key=lambda p: (lengths[p], p not in set(previous), lid[p])))
                selected = first_fit(priority, lengths, budget, initial)
            else:
                raise ValueError(f"unknown policy: {policy}")
            expected = [pid for pid in candidates if pid in selected]
            checks.compare(expected, row["keep_after"], prefix + ".keep_after_replay")
            checks.compare(sorted(set(new) & selected), sorted(row["action"]["add"]), prefix + ".action.add")
            checks.compare(sorted(set(previous) - selected), sorted(row["action"]["drop"]), prefix + ".action.drop")
            used = sum(lengths[p] for p in expected)
            checks.compare(used, row["used_after"], prefix + ".used_after")
            checks.compare(True, used <= budget, prefix + ".budget_respected")
            checks.compare(True, row["action_valid"], prefix + ".action_valid")
            if mode == "append":
                checks.compare(previous, expected[:len(previous)], prefix + ".append_prefix")
            recall = len(set(available) & selected) / len(available) if available else 1.0
            checks.compare(recall, row["recall"], prefix + ".recall")
            fragment = used if policy == "no_compress" else sum(lengths[p] for p in candidates)
            checks.compare(fragment, row["fragment_input_tokens"], prefix + ".fragment_input_tokens")
            if policy == "no_compress":
                checks.compare(used, row["input_tokens"], prefix + ".input_tokens")
            elif row_number % 97 == 0:
                prompt = render_for_token_check(task, row, lid, texts, lengths)
                count = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
                checks.compare(count, row["input_tokens"], prefix + ".input_tokens_sample")
                prompt_checks += 1
            prompt_lengths[cfg].append(row["input_tokens"])
            fragment_lengths[cfg].append(fragment)
            previous = expected
            contexts.append(expected)
            ever.update(expected)
            recalls.append(recall)
        checks.compare(len(task["steps"]), len(contexts), f"groups.{key}.n_steps")
        checks.compare({"keep_after": previous, "questions": task["questions"], "all_gold": sorted(task_gold[tid]),
                        "policy": policy, "mode": mode, "budget": budget}, finals[key], f"final_contexts.{key}")
        missing = task_gold[tid] - set(previous)
        buckets = {"hop": {}, "question_position": {}, "mechanism": {}}
        for axis, field in (("hop", "hop_index"), ("question_position", "question_index")):
            for i in range(4):
                occurrences = [s["gold"] for s in task["steps"] if s[field] == i]
                lost = sum(p in missing for p in occurrences)
                buckets[axis][str(i + 1)] = {"n": int(bool(occurrences)), "count": int(bool(lost)),
                                             "evidence_n": len(occurrences), "lost_evidence_count": lost}
        for name, pids in (("never_kept", missing - ever), ("dropped", missing & ever)):
            buckets["mechanism"][name] = {"n": 1, "count": int(bool(pids)), "evidence_n": len(task_gold[tid]), "lost_evidence_count": len(pids)}
        plain, cached = context_cost(contexts, lengths)
        metrics[cfg].append({"task": tid, "recalls": recalls, "recall_mean": sum(recalls) / len(recalls),
                             "final_recall": (len(task_gold[tid]) - len(missing)) / len(task_gold[tid]), "missing": missing,
                             "complete": sum(g <= set(previous) for g in question_gold[tid]),
                             "occupancy": sum(lengths[p] for p in previous) / budget,
                             "plain": plain, "cached": cached, "buckets": buckets})
        if len(visited) % 570 == 0:
            print(f"Independent replay: {len(visited)}/2850 task/config groups.", flush=True)
    checks.compare(True, {(cfg, tid) for cfg in expected_configs for tid in range(150)} == visited, "groups.complete_matrix")
    baseline = aggregate(metrics["no_compress"])
    recomputed = {}
    for cfg in sorted(expected_configs):
        summary = aggregate(metrics[cfg], baseline)
        recomputed[cfg] = summary
        checks.compare(summary, results["configs"][cfg]["metrics"], f"configs.{cfg}.metrics")
        checks.compare(distribution(prompt_lengths[cfg]), results["configs"][cfg]["input_tokens"], f"configs.{cfg}.input_tokens")
        checks.compare(distribution(fragment_lengths[cfg]), results["configs"][cfg]["fragment_input_tokens"], f"configs.{cfg}.fragment_input_tokens")
    checks.compare({"configurations": 19, "tasks_per_configuration": 150, "distinct_dev_questions": 600,
                    "step_records": row_number, "final_context_records": len(finals), "invalid_actions": 0,
                    "over_budget_states": 0, "no_compress_final_recall": 1.0,
                    "no_compress_evidence_loss_rate": 0.0}, results["validation"], "validation")
    checks.compare(19 * total["overall"], row_number, "step_records.complete_matrix")
    for comparison in results["validation"]["oracle_comparisons"]:
        suffix = f"|{comparison['mode']}|{comparison['tier']}"
        oracle = sorted(metrics["oracle" + suffix], key=lambda x: x["task"])
        other = sorted(metrics[comparison["other"] + suffix], key=lambda x: x["task"])
        difference = sum(m["final_recall"] for m in oracle) / 150 - sum(m["final_recall"] for m in other) / 150
        counterexamples = [{"task": a["task"], "oracle_recall": a["final_recall"], "other_recall": b["final_recall"],
                            "oracle_missing": sorted(a["missing"]), "other_missing": sorted(b["missing"])}
                           for a, b in zip(oracle, other) if a["final_recall"] + 1e-12 < b["final_recall"]]
        checks.compare(difference, comparison["macro_difference"], f"comparison.{suffix}.{comparison['other']}.difference")
        checks.compare(difference >= -1e-12, comparison["oracle_macro_at_least_other"], f"comparison.{suffix}.{comparison['other']}.flag")
        checks.compare(counterexamples, comparison["individual_task_counterexamples"], f"comparison.{suffix}.{comparison['other']}.counterexamples")
    return {"corpus_paragraphs_checked_and_retokenized": len(texts), "tasks": len(tasks), "questions": 600,
            "raw_hops": total["overall"], "configurations": 19, "task_config_groups": len(visited),
            "states_replayed": row_number, "final_contexts_checked": len(finals),
            "compressor_prompts_retokenized": prompt_checks, "budgets": budgets,
            "retrieval": retrieval, "answer_leak": answer_leak, "recomputed_metrics": recomputed,
            "run_finished_at_utc": config["finished_at_utc"],
            "scope": {"production_helpers_imported": False, "experiment_rerun": False,
                      "natural_hit": "Reaggregated saved raw task steps; full-corpus ranking equivalence separately covered by 20 real-dev query integration tests.",
                      "bm25": "Priorities replayed from saved corpus scores; checked cross-mode/budget score consistency.",
                      "input_tokens": "All distributions reaggregated; every 97th non-baseline state rendered and retokenized.",
                      "train_reference": "Not rerun; this independent script covers dev statistics and the three dev-derived budgets.",
                      "word_boundary": "SQuAD removes ASCII string.punctuation only; matching uses Unicode-aware (?<!\\w)/(?!\\w), not padded ASCII spaces."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/root/workplace/ctx-compressor"))
    args = parser.parse_args()
    base = args.project.resolve()
    out = base / "eval/feasibility"
    if not (out / "run_config.json").exists() or not (out / "feasibility_report.md").exists():
        parser.error("formal run is not complete; no check output has been written")
    started = time.monotonic()
    checks = Checks()
    try:
        findings = audit(base, checks)
    except Exception:
        findings = {"fatal_error": traceback.format_exc()}
        checks.mismatches.append({"path": "fatal_error", "actual": findings["fatal_error"]})
    report = {"passed": not checks.mismatches, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.monotonic() - started, "comparisons": checks.count,
              "mismatch_count": len(checks.mismatches), "mismatches": checks.mismatches[:100], **findings}
    (out / "independent_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fence = chr(96) * 3
    lines = ["# 任务 02：独立落盘结果复核", "", f"结果：**{'通过' if report['passed'] else '失败'}**。",
             f"独立比较 {checks.count:,} 项，差异 {len(checks.mismatches):,} 项；用时 {report['elapsed_seconds']:.2f} 秒。", "",
             "复现命令（5880）：", "", fence + "bash", "cd /root/workplace/ctx-compressor",
             "bash scripts/run_python.sh scripts/check_feasibility_independent.py", fence, "",
             "脚本只读取原始数据及正式实验产物，不导入生产轨迹、策略或指标函数，不重新运行实验。只写本独立报告及对应 JSON。", ""]
    if report["passed"]:
        lines += [f"重建并重新分词 {report['corpus_paragraphs_checked_and_retokenized']:,} 个 dev 段落，逐项核对 pid、标题和渲染文本 SHA1。",
                  f"核对 {report['tasks']} 个任务、{report['questions']} 道互不重复问题、{report['raw_hops']} 个原始跳的题号、跳号、证据 pid、查询替换和返回候选。",
                  f"重放全部 {report['configurations']} 种配置 × 150 任务，共 {report['states_replayed']:,} 步，核对 {report['final_contexts_checked']:,} 个最终上下文。",
                  "三档预算独立复算：" + json.dumps(report["budgets"], ensure_ascii=False) + "。", "",
                  "核对了原始/去重轨迹长度、题目及任务证据长度分布、自然命中各分组、正文与标题+正文两种答案匹配口径的段级/跳级分母，以及正文重复计数。",
                  "词边界采用 Unicode-aware 的 (?<!\\w)/(?!\\w)；SQuAD 仅去除 string.punctuation。保留的 Unicode 引号可构成词边界，因此未采用 Claude 示例脚本仅检查 ASCII 空格前后的简化匹配。",
                  "每步独立执行 first-fit（放不下则跳过）、最新优先、BM25 同分按呈现位置、oracle token/已有状态/稳定编号排序和 no_compress；核对预算、动作、append 前缀与当前可用金标准。",
                  "任务末完整 gold 的召回、丢失率、证据完整题比例、丢失跳位/题位/方式分桶、预算占用、缓存/非缓存成本及相对基线比值，均与正式 JSON 一致。",
                  f"输入长度分布全量重新聚合，另独立渲染并重新分词 {report['compressor_prompts_retokenized']} 个压缩器输入；所有 no_compress 累计上下文长度逐步核对。", "",
                  "范围说明：自然命中率复算来自已保存的原始检索步骤；BM25 全语料打分与 retrieve 的等价性由另一个 20 个真实 dev 查询的集成测试验证。BM25 策略重放使用落盘候选分数，并验证其跨模式、预算一致。train 参照 P90 不在本独立 dev 验收脚本中重跑。"]
    else:
        lines += [fence + "json", json.dumps(checks.mismatches[:20], ensure_ascii=False, indent=2), fence]
    (out / "independent_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "comparisons", "mismatch_count", "elapsed_seconds")}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
