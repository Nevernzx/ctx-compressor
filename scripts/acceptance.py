#!/usr/bin/env python3
"""Generate exactly three MuSiQue dev tasks in both modes; no §6 experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.audit_data import load_jsonl, valid_examples
from src.build_traj import (
    apply_action, build_corpus, oracle_action, render_input, render_target,
    rollout, sample_tasks, simulate_task,
)


def j(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=PROJECT / "data/raw/musique/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT / "data/raw/qwen3_tokenizer/tokenizer.json")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "eval/acceptance")
    parser.add_argument("--trajectory-file", type=Path, default=PROJECT / "data/traj/acceptance.jsonl")
    args = parser.parse_args()
    started = time.monotonic()
    examples = load_jsonl(args.dev)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ntok = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    print(f"Loaded {len(examples)} dev examples; constructing dev-only BM25 corpus.", flush=True)
    corpus, index, key2pid = build_corpus(examples, ntok)
    valid, skipped = valid_examples(examples, key2pid)
    tasks = sample_tasks(valid, 3, args.seed)
    checks = {"resolved_queries": 0, "gold_in_candidates": 0, "budget_respected": 0,
              "legal_sft_targets": 0, "append_prefix_preserved": 0,
              "stable_local_ids": 0, "step_count_matches_hops": 0,
              "distinct_questions": len({ex["id"] for task in tasks for ex in task})}
    failures, records, details, task_summaries = [], [], [], []
    for task_number, task in enumerate(tasks, 1):
        questions = [ex["question"] for ex in task]
        steps = simulate_task(task, index, key2pid, top_n=20, seed=args.seed + task_number)
        expected_hops = sum(len(ex["question_decomposition"]) for ex in task)
        assert len(steps) == expected_hops
        checks["step_count_matches_hops"] += 1
        task_summaries.append({"task": task_number, "example_ids": [ex["id"] for ex in task],
                               "hops": len(steps), "natural_hits": sum(st["natural_hit"] for st in steps)})
        details.extend([f"## 任务 {task_number}", "", f"题号：`{j([ex['id'] for ex in task])}`", ""])
        details.extend(f"{i}. {question}" for i, question in enumerate(questions, 1))
        details.extend(["", f"步数 {len(steps)}，等于所选 4 题跳数之和；自然命中 {sum(st['natural_hit'] for st in steps)} / {len(steps)}。", ""])
        for mode in ("reselect", "append"):
            samples = rollout(questions, steps, corpus, args.budget, mode)
            details.extend([f"### 模式：{mode}", ""])
            previous, seen_ids = [], {}
            for step_number, (st, s) in enumerate(zip(steps, samples), 1):
                prefix = f"task={task_number},mode={mode},step={step_number}"
                pid_by_lid = s["pid_by_lid"]
                lid_by_pid = {pid: lid for lid, pid in pid_by_lid.items()}
                gold_lid = lid_by_pid.get(st["gold"])
                tests = {
                    "resolved_queries": re.search(r"#\d", st["query"]) is None,
                    "gold_in_candidates": st["gold"] in st["new"] and gold_lid in s["kept_ids"] + s["new_ids"],
                    "budget_respected": sum(s["ntok"][i] for i in s["keep_after"]) <= args.budget,
                    "legal_sft_targets": apply_action(s, s["sft_target"]) is not None and s["sft_target"] == oracle_action(s),
                    "stable_local_ids": all(pid not in seen_ids or seen_ids[pid] == lid for lid, pid in pid_by_lid.items()),
                }
                if mode == "append":
                    tests["append_prefix_preserved"] = s["keep_after"][:len(previous)] == previous
                for name, passed in tests.items():
                    if passed:
                        checks[name] += 1
                    else:
                        failures.append(f"{prefix}: {name}")
                seen_ids.update(lid_by_pid)
                previous = s["keep_after"]
                evidence = corpus[st["gold"]]
                details.extend([
                    f"#### 第 {step_number} 步（第 {st['question_index'] + 1} 题，第 {st['hop_index'] + 1} 跳）", "",
                    f"- 原始子问题：{st['original_query']}",
                    f"- 替换后的查询：{st['query']}",
                    f"- 证据段：编号 **{gold_lid}**，语料 pid {st['gold']}；标题：{evidence['title']}",
                    f"- 证据正文前 200 字符：{j(evidence['paragraph_text'][:200])}",
                    f"- 自然命中：{'是' if st['natural_hit'] else '否（已替换检索末位候选）'}",
                    f"- 本步工具返回编号（含已保留的重复返回）：`{j([lid_by_pid[p] for p in st['new']])}`",
                    f"- 新返回编号（去除已保留）：`{j(s['new_ids'])}`",
                    f"- 已保留编号：`{j(s['kept_ids'])}`",
                    f"- 当前可用金标准编号：`{j(s['gold'])}`",
                    f"- oracle / SFT 动作：`{render_target(s)}`",
                    f"- 执行后保留：`{j(s['keep_after'])}`",
                    f"- 执行后 token：{s['used_after']} / {args.budget}；recall：{s['recall']:.6f}",
                    f"- 自动检查：{'通过' if all(tests.values()) else '失败'}", "",
                ])
                records.append({"task": task_number, "step": step_number, "retrieval": st,
                                "state": s, "input": render_input(s), "target": render_target(s)})
            if mode == "reselect":
                details.extend(["<details><summary>首步模型输入模板示例（已保留 / 新返回）</summary>", "", "```text",
                                render_input(samples[0]), "```", "", "</details>", ""])
        print(f"Task {task_number}: {expected_hops} steps × 2 modes completed.", flush=True)
    if checks["distinct_questions"] != 12:
        failures.append("questions are not distinct across three tasks")
    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "budget": args.budget, "budget_status": "占位，正式值由 §6 试验定",
        "dataset": str(args.dev), "corpus_split": "dev only", "corpus_size": len(corpus),
        "valid_examples": len(valid), "skipped_examples": len(skipped), "skipped": skipped,
        "tasks": task_summaries, "checks": checks, "failures": failures,
        "passed": not failures, "trajectory_records": len(records),
        "elapsed_seconds": time.monotonic() - started,
    }
    header = ["# 任务 01：真实 MuSiQue-Ans dev 轨迹验收", "",
              f"固定种子：{args.seed}；3 个任务，每任务 4 道题，12 题互不重复；每个任务分别运行 reselect 与 append。",
              f"预算 **{args.budget} token，占位，正式值由 §6 试验定**。本报告不执行 §6 可行性试验。",
              f"语料与 BM25 索引只来自 dev，共 {len(corpus):,} 个去重段落；每跳 top-20，未命中的证据替换末位，再带种子打乱。",
              "片段 token 只计算 `标题 + 换行 + 正文`，与保留文本一致；不包含模板头及编号元数据。",
              f"因非法引用或证据映射跳过 {len(skipped)} 题。当前回合 recall 的分母遵循 PLAN：只计已执行各跳中当前仍可获得的证据。", "",
              "## 自动检查汇总", "",
              f"结果：**{'全部通过' if not failures else '存在失败'}**。", "",
              "| 检查 | 通过次数 |", "|---|---:|"]
    header.extend(f"| {name} | {count} |" for name, count in checks.items())
    header.extend(["", f"失败列表：`{j(failures)}`。", ""])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trajectory_file.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "traj_samples.md").write_text("\n".join(header + details) + "\n", encoding="utf-8")
    (args.output_dir / "acceptance_checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.trajectory_file.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(j(record) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
