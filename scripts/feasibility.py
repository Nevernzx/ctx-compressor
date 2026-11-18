#!/usr/bin/env python3
"""TASK_02 single-pass CPU feasibility experiment using existing local assets."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import shlex
import sys
import time

# Bound CPU placement before importing numerical libraries; the project wrapper
# also limits BLAS/OpenMP to 4 and disables tokenizer parallelism and CUDA.
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])

import numpy as np
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.build_traj import (build_corpus, hop_queries, oracle_action,
                            render_input, render_paragraph, rollout,
                            sample_tasks, simulate_task)
from src.feasibility import aggregate_tasks, distribution, normalize_answer, summarize_task
from src.feasibility_report import render_report
from src.policies import bm25_policy, latest_policy, no_compress_policy, score_query


def utc():
    return datetime.now(timezone.utc).isoformat()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def filter_examples(examples, keys=None):
    valid, rejected = [], []
    for example in examples:
        # Train uses a temporary mapping for this question only, not a corpus.
        local_keys = keys if keys is not None else {
            (p["title"], p["paragraph_text"]): i for i, p in enumerate(example["paragraphs"])
        }
        try:
            hop_queries(example, local_keys)
        except (KeyError, ValueError, TypeError) as error:
            rejected.append({"id": example["id"], "reason": str(error)})
        else:
            valid.append(example)
    return valid, {"total": len(examples), "valid": len(valid), "rejected": len(rejected),
                   "reasons": dict(Counter(item["reason"] for item in rejected)),
                   "rejected_examples": rejected}


def ratio(hits, n):
    return {"hits": int(hits), "n": int(n), "rate": hits / n if n else None}


def input_threads():
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("Threads:"):
                return int(line.split()[1])
    return None


def output_size(*directories):
    return sum(p.stat().st_size for d in directories for p in d.rglob("*") if p.is_file())


def guard_size(*directories):
    size = output_size(*directories)
    if size >= 1_000_000_000:
        raise RuntimeError("TASK_02 output reached 1 GB; stopped without continuing")
    return size


def task02_stats(tasks, corpus, ntok_by_pid, seed):
    dist = {name: [] for name in ("task_raw_tokens", "task_unique_tokens", "task_steps",
                                 "question_hops", "task_evidence_tokens", "question_evidence_tokens")}
    normalized_body = [normalize_answer(p["paragraph_text"]) for p in corpus]
    normalized_rendered = [normalize_answer(p["text"]) for p in corpus]
    hits, totals = Counter(), Counter()
    leak = {name: Counter() for name in ("all", "multiword", "rendered_all", "rendered_multiword")}
    body_duplicates = different_title_duplicates = empty_answers = 0
    examples = []
    for task in tasks:
        steps = task["steps"]
        all_gold = set(task["all_gold"])
        dist["task_raw_tokens"].append(sum(ntok_by_pid[p] for st in steps for p in st["new"]))
        dist["task_unique_tokens"].append(sum(ntok_by_pid[p] for p in {p for st in steps for p in st["new"]}))
        dist["task_steps"].append(len(steps))
        dist["task_evidence_tokens"].append(sum(ntok_by_pid[p] for p in all_gold))
        for question_gold, question_hops in zip(task["question_gold"], task["question_hops"]):
            dist["question_evidence_tokens"].append(sum(ntok_by_pid[p] for p in question_gold))
            dist["question_hops"].append(question_hops)
        for step_index, st in enumerate(steps):
            for group in ("overall", f"hop:{st['hop_index'] + 1}", f"total:{st['question_hops']}"):
                totals[group] += 1
                hits[group] += st["natural_hit"]
            answer = normalize_answer(st["answer"])
            empty_answers += not bool(answer)
            pattern = re.compile(r"(?<!\w)" + re.escape(answer) + r"(?!\w)") if answer else None
            multiword = len(answer.split()) >= 2
            nongold = [pid for pid in st["new"] if pid not in all_gold]
            body_hits, rendered_hits = [], []
            for pid in nongold:
                if pattern and pattern.search(normalized_body[pid]):
                    body_hits.append(pid)
                    examples.append({"task": task["task"], "step": step_index, "query": st["original_query"],
                                     "resolved_query": st["query"], "answer": st["answer"],
                                     "gold": st["gold"], "nongold": pid})
                if pattern and pattern.search(normalized_rendered[pid]):
                    rendered_hits.append(pid)
                if normalized_body[pid] == normalized_body[st["gold"]]:
                    body_duplicates += 1
                    different_title_duplicates += corpus[pid]["title"] != corpus[st["gold"]]["title"]
            for name, matched in (("all", body_hits), ("rendered_all", rendered_hits)):
                buckets = [name]
                if multiword:
                    buckets.append("multiword" if name == "all" else "rendered_multiword")
                for bucket in buckets:
                    leak[bucket]["paragraph_n"] += len(nongold)
                    leak[bucket]["paragraph_hits"] += len(matched)
                    leak[bucket]["hop_n"] += 1
                    leak[bucket]["hop_hits"] += bool(matched)
    retrieval = {"overall": ratio(hits["overall"], totals["overall"]),
                 "by_hop": {str(i): ratio(hits[f"hop:{i}"], totals[f"hop:{i}"]) for i in range(1, 5)},
                 "by_question_hops": {str(i): ratio(hits[f"total:{i}"], totals[f"total:{i}"]) for i in (2, 3, 4)}}
    answer_leak = {name: {"paragraph": ratio(c["paragraph_hits"], c["paragraph_n"]),
                          "hop": ratio(c["hop_hits"], c["hop_n"])} for name, c in leak.items()}
    denominator = leak["all"]["paragraph_n"]
    answer_leak["body_duplicates"] = {"count": body_duplicates, "n": denominator,
                                       "rate": body_duplicates / denominator if denominator else None,
                                       "different_title_count": different_title_duplicates}
    answer_leak["empty_normalized_answer_hops"] = empty_answers
    answer_leak["primary_text"] = "paragraph_text only; rendered title+body sensitivity also reported"
    picked = random.Random(seed).sample(examples, min(20, len(examples)))
    return {k: distribution(v) for k, v in dist.items()}, retrieval, answer_leak, picked


def leak_markdown(samples, corpus, seed, total_hits):
    lines = ["# 非证据段含本跳答案：人工核对样例", "",
             f"正文主口径共 {total_hits} 次匹配，使用 random.Random({seed}) 无放回抽取 {len(samples)} 条。",
             "非证据段排除了整个四题任务的全部金标准；匹配按 SQuAD 归一化及词边界。命中不等同于真正标注遗漏。", ""]
    for i, sample in enumerate(samples, 1):
        gold, other = corpus[sample["gold"]], corpus[sample["nongold"]]
        lines += [f"## 样例 {i}：任务 {sample['task']}，步骤 {sample['step']}（均从0开始）", "",
                  f"- 子问题：{sample['query']}", f"- 替换后查询：{sample['resolved_query']}",
                  f"- 答案：{sample['answer']}", f"- 金标准 pid={sample['gold']}，标题：{gold['title']}",
                  f"- 金标准正文前300字符：{compact(gold['paragraph_text'][:300])}",
                  f"- 非证据 pid={sample['nongold']}，标题：{other['title']}",
                  f"- 非证据正文前300字符：{compact(other['paragraph_text'][:300])}", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--n-tasks", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "eval/feasibility")
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/feasibility")
    args = parser.parse_args()
    if PROJECT != Path("/root/workplace/ctx-compressor"):
        parser.error("execute only on 5880 in /root/workplace/ctx-compressor")
    if args.n_tasks <= 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("positive n_tasks and CUDA_VISIBLE_DEVICES='' are required")
    args.output_dir, args.data_dir = args.output_dir.resolve(), args.data_dir.resolve()
    if not args.output_dir.is_relative_to(PROJECT / "eval/feasibility") or not args.data_dir.is_relative_to(PROJECT / "data/feasibility"):
        parser.error("outputs must stay in the dedicated TASK_02 directories")
    paths = [args.data_dir / f"tasks_dev{args.n_tasks}.jsonl", args.data_dir / "steps.jsonl",
             args.data_dir / "final_contexts.jsonl", args.data_dir / "corpus_dev_ntok.jsonl",
             args.output_dir / "feasibility_results.json", args.output_dir / "run_config.json",
             args.output_dir / "feasibility_report.md", args.output_dir / "answer_leak_samples.md"]
    if any(path.exists() for path in paths):
        parser.error("refusing to overwrite an existing experiment; select unused subdirectories")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    started, started_utc = time.monotonic(), utc()
    timings = {}
    train_path = PROJECT / "data/raw/musique/musique_ans_v1.0_train.jsonl"
    dev_path = PROJECT / "data/raw/musique/musique_ans_v1.0_dev.jsonl"
    tok_path = PROJECT / "data/raw/qwen3_tokenizer/tokenizer.json"
    print(f"{started_utc} TASK_02 started: n_tasks={args.n_tasks}, seed={args.seed}, one process; threads={input_threads()}", flush=True)
    tokenizer = Tokenizer.from_file(str(tok_path))
    ntok = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    stage = time.monotonic()
    dev = load_jsonl(dev_path)
    corpus, index, key2pid = build_corpus(dev, ntok)
    valid, dev_filter = filter_examples(dev, key2pid)
    selected = sample_tasks(valid, args.n_tasks, args.seed)
    pid_ntok = {pid: p["ntok"] for pid, p in enumerate(corpus)}
    timings["dev_corpus_and_filter_seconds"] = time.monotonic() - stage
    with (args.data_dir / "corpus_dev_ntok.jsonl").open("w") as stream:
        for pid, paragraph in enumerate(corpus):
            stream.write(compact({"pid": pid, "title": paragraph["title"], "ntok": paragraph["ntok"],
                                  "sha1": hashlib.sha1(paragraph["text"].encode()).hexdigest()}) + "\n")
    stage = time.monotonic()
    tasks = []
    with (args.data_dir / f"tasks_dev{args.n_tasks}.jsonl").open("w") as stream:
        for task_index, examples in enumerate(selected):
            steps = simulate_task(examples, index, key2pid, top_n=20, seed=task_index)
            question_gold = [set() for _ in examples]
            for step in steps:
                ex = examples[step["question_index"]]
                step["answer"] = ex["question_decomposition"][step["hop_index"]]["answer"]
                step["question_hops"] = len(ex["question_decomposition"])
                question_gold[step["question_index"]].add(step["gold"])
                assert len(step["new"]) == 20 and len(set(step["new"])) == 20
                assert step["gold"] in step["new"] and re.search(r"#\d", step["query"]) is None
            task = {"task": task_index, "example_ids": [ex["id"] for ex in examples], "steps": steps,
                    "questions": [{k: ex[k] for k in ("id", "question", "answer", "answer_aliases")} for ex in examples],
                    "question_gold": [sorted(g) for g in question_gold],
                    "all_gold": sorted(set().union(*question_gold)),
                    "question_hops": [len(ex["question_decomposition"]) for ex in examples]}
            tasks.append(task)
            stream.write(compact(task) + "\n")
    timings["one_time_dev_retrieval_seconds"] = time.monotonic() - stage
    stage = time.monotonic()
    distributions, retrieval, answer_leak, leak_samples = task02_stats(tasks, corpus, pid_ntok, args.seed)
    timings["dev_statistics_seconds"] = time.monotonic() - stage
    tight = math.ceil(distributions["task_evidence_tokens"]["p90"])
    budgets = {"tight": tight, "mid": math.ceil(1.5 * tight), "loose": 3 * tight}
    (args.output_dir / "answer_leak_samples.md").write_text(leak_markdown(leak_samples, corpus, args.seed, answer_leak["all"]["paragraph"]["hits"]))
    print(f"{utc()} dev built: {len(corpus)} paragraphs; valid/rejected={len(valid)}/{dev_filter['rejected']}; budgets={budgets}", flush=True)
    stage = time.monotonic()
    train = load_jsonl(train_path)
    train_valid, train_filter = filter_examples(train)
    train_tasks = sample_tasks(train_valid, 150, args.seed)
    gold_token_cache, train_evidence_tokens = {}, []
    for examples in train_tasks:
        keys = set()
        for ex in examples:
            by_idx = {p["idx"]: p for p in ex["paragraphs"]}
            for hop in ex["question_decomposition"]:
                p = by_idx[hop["paragraph_support_idx"]]
                keys.add((p["title"], p["paragraph_text"]))
        for key in keys:
            if key not in gold_token_cache:
                gold_token_cache[key] = ntok(render_paragraph(*key))
        train_evidence_tokens.append(sum(gold_token_cache[key] for key in keys))
    budgets["train_reference_p90"] = float(np.percentile(train_evidence_tokens, 90))
    train_reference = {"n_tasks": 150, "n_questions": 600, "distribution": distribution(train_evidence_tokens),
                       "unique_gold_paragraphs_tokenized": len(gold_token_cache), "corpus_or_index_built": False}
    del train, train_valid, train_tasks, gold_token_cache
    timings["train_reference_seconds"] = time.monotonic() - stage
    nonbinding = int(distributions["task_unique_tokens"]["max"]) + 1
    configs = {"no_compress": {"policy": "no_compress", "mode": "append", "budget_tier": "unbounded", "budget": nonbinding}}
    for policy in ("latest", "bm25", "oracle"):
        for mode in ("reselect", "append"):
            for tier in ("tight", "mid", "loose"):
                configs[f"{policy}|{mode}|{tier}"] = {"policy": policy, "mode": mode, "budget_tier": tier, "budget": budgets[tier]}
    metrics = defaultdict(list)
    input_lengths, fragment_lengths = defaultdict(list), defaultdict(list)
    step_records, final_records = 0, 0
    stage = time.monotonic()
    with (args.data_dir / "steps.jsonl").open("w") as step_file, (args.data_dir / "final_contexts.jsonl").open("w") as final_file:
        for task in tasks:
            questions = [q["question"] for q in task["questions"]]
            bm25_scores = score_query(index, " ".join(questions))
            for name, configuration in configs.items():
                policy_name, mode, budget = configuration["policy"], configuration["mode"], configuration["budget"]
                policy = (no_compress_policy if policy_name == "no_compress" else
                          latest_policy(mode) if policy_name == "latest" else
                          bm25_policy(index, mode=mode, scores=bm25_scores) if policy_name == "bm25" else oracle_action)
                samples = rollout(questions, task["steps"], corpus, budget, mode, policy=policy)
                summary = summarize_task(samples, task["steps"], task["question_gold"], pid_ntok, budget)
                assert summary["invalid_action_count"] == summary["over_budget_count"] == 0, (task["task"], name)
                metrics[name].append(summary)
                previous = []
                for step_index, (sample, step) in enumerate(zip(samples, task["steps"])):
                    mapping = sample["pid_by_lid"]
                    kept = [mapping[i] for i in sample["keep_after"]]
                    if mode == "append":
                        assert kept[:len(previous)] == previous
                    previous = kept
                    fragment = sum(sample["ntok"].values())
                    prompt_tokens = ntok(render_input(sample)) if policy_name != "no_compress" else sample["used_after"]
                    input_lengths[name].append(prompt_tokens)
                    fragment_lengths[name].append(fragment if policy_name != "no_compress" else sample["used_after"])
                    record = {"task": task["task"], "config": name, "policy": policy_name,
                              "mode": mode, "budget": budget, "step": step_index,
                              "question_index": step["question_index"], "hop_index": step["hop_index"],
                              "kept_before": [mapping[i] for i in sample["kept_ids"]],
                              "new": [mapping[i] for i in sample["new_ids"]], "returned": step["new"],
                              "gold": step["gold"], "available_gold": [mapping[i] for i in sample["gold"]],
                              "keep_after": kept, "used_after": sample["used_after"],
                              "recall": sample["recall"], "action_valid": sample["action_valid"],
                              "action": {k: [mapping[i] for i in sample["action"][k]] for k in ("add", "drop")},
                              "input_tokens": prompt_tokens, "fragment_input_tokens": fragment_lengths[name][-1],
                              "input_kind": "cumulative_context" if policy_name == "no_compress" else "compressor_render_input"}
                    if policy_name == "bm25":
                        record["bm25_scores"] = {str(mapping[i]): float(bm25_scores[mapping[i]]) for i in sample["kept_ids"] + sample["new_ids"]}
                    step_file.write(compact(record) + "\n")
                    step_records += 1
                final_file.write(compact({"task": task["task"], "config": name, "policy": policy_name,
                                          "mode": mode, "budget": budget, "keep_after": previous,
                                          "questions": task["questions"], "all_gold": task["all_gold"]}) + "\n")
                final_records += 1
            step_file.flush()
            final_file.flush()
            size = guard_size(args.data_dir, args.output_dir)
            assert input_threads() is None or input_threads() <= 16, "CPU thread limit exceeded"
            if (task["task"] + 1) % 5 == 0:
                print(f"{utc()} {task['task'] + 1}/{args.n_tasks} tasks × 19 configs; {step_records} states; outputs={size / 1e6:.1f} MB", flush=True)
    timings["all_closed_loop_configurations_seconds"] = time.monotonic() - stage
    baseline = aggregate_tasks(metrics["no_compress"])
    for name, configuration in configs.items():
        configuration["metrics"] = aggregate_tasks(metrics[name], baseline=baseline)
        configuration["input_tokens"] = distribution(input_lengths[name])
        configuration["fragment_input_tokens"] = distribution(fragment_lengths[name])
    invalid = sum(c["metrics"]["action_validity"]["invalid_count"] for c in configs.values())
    over = sum(c["metrics"]["action_validity"]["over_budget_count"] for c in configs.values())
    assert invalid == over == 0
    assert baseline["final_recall"]["mean"] == baseline["step_recall"]["micro"]["mean"] == 1.0
    assert baseline["evidence_loss"]["rate"] == 0.0
    comparisons = []
    for mode in ("reselect", "append"):
        for tier in ("tight", "mid", "loose"):
            oracle_name = f"oracle|{mode}|{tier}"
            for other in ("latest", "bm25"):
                other_name = f"{other}|{mode}|{tier}"
                difference = configs[oracle_name]["metrics"]["final_recall"]["mean"] - configs[other_name]["metrics"]["final_recall"]["mean"]
                per_task = [{"task": i, "oracle_recall": a["final_recall"], "other_recall": b["final_recall"],
                             "oracle_missing": a["missing_pids"], "other_missing": b["missing_pids"]}
                            for i, (a, b) in enumerate(zip(metrics[oracle_name], metrics[other_name]))
                            if a["final_recall"] + 1e-12 < b["final_recall"]]
                comparisons.append({"mode": mode, "tier": tier, "other": other, "macro_difference": difference,
                                    "oracle_macro_at_least_other": difference >= -1e-12,
                                    "individual_task_counterexamples": per_task})
    results = {"settings": {"seed": args.seed, "n_tasks": args.n_tasks, "top_n": 20,
                            "questions_per_task": 4, "task_index_base": 0, "retrieval_seed": "task_index",
                            "r_read": 0.05, "r_write": 1.25, "train_reference_n_tasks": 150,
                            "reading_model_status": "未测（需阅读模型）"},
               "budgets": budgets, "dev_filter": dev_filter, "train_filter": train_filter,
               "distributions": distributions, "retrieval": retrieval, "answer_leak": answer_leak,
               "train_reference": train_reference, "configs": configs,
               "validation": {"configurations": len(configs), "tasks_per_configuration": args.n_tasks,
                              "distinct_dev_questions": len({ex for task in tasks for ex in task["example_ids"]}),
                              "step_records": step_records, "final_context_records": final_records,
                              "invalid_actions": invalid, "over_budget_states": over,
                              "no_compress_final_recall": baseline["final_recall"]["mean"],
                              "no_compress_evidence_loss_rate": baseline["evidence_loss"]["rate"],
                              "oracle_comparisons": comparisons,
                              "oracle_semantics": "Unchanged PLAN local greedy oracle; no future-gold access or global optimization. Counterexamples, if any, remain reported rather than changing policies."}}
    stage = time.monotonic()
    data_hashes = {str(p.relative_to(PROJECT)): sha256(p) for p in (train_path, dev_path)}
    tokenizer_hashes = {str(p.relative_to(PROJECT)): sha256(p) for p in sorted(tok_path.parent.iterdir()) if p.is_file()}
    timings["input_hashes_seconds"] = time.monotonic() - stage
    elapsed = time.monotonic() - started
    run_config = {"started_at_utc": started_utc, "finished_at_utc": utc(), "elapsed_seconds": elapsed,
                  "timings_seconds": timings, "command": "bash scripts/run_python.sh " + shlex.join(["scripts/feasibility.py", *sys.argv[1:]]),
                  "seed": args.seed, "n_tasks": args.n_tasks, "top_n": 20, "questions_per_task": 4,
                  "retrieval_seed": "zero-based task_index", "dataset_sha256": data_hashes,
                  "tokenizer_sha256": tokenizer_hashes, "python_version": sys.version,
                  "package_versions": {p: version(p) for p in ("bm25s", "transformers", "tokenizers", "numpy", "scipy", "pytest")},
                  "cpu": {"processes": 1, "threads_observed": input_threads(), "maximum_allowed_threads": 16,
                          "affinity": sorted(os.sched_getaffinity(0)),
                          "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}},
                  "maximum_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "data_bytes": output_size(args.data_dir), "no_downloads_or_installs_or_api": True,
                  "output_directory": str(args.output_dir), "data_directory": str(args.data_dir)}
    results["run"] = {"elapsed_seconds": elapsed, "timings_seconds": timings}
    write_json(args.output_dir / "feasibility_results.json", results)
    write_json(args.output_dir / "run_config.json", run_config)
    (args.output_dir / "feasibility_report.md").write_text(render_report(results))
    guard_size(args.data_dir, args.output_dir)
    print(compact({"completed": True, "budgets": budgets, "natural_hit": retrieval["overall"],
                   "answer_leak": answer_leak["all"], "invalid_actions": invalid,
                   "over_budget_states": over, "elapsed_seconds": elapsed,
                   "output_bytes": output_size(args.data_dir, args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
