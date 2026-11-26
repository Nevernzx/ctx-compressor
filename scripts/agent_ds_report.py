"""Task 21 pre-registered metrics (TASK_21 §3). CPU only, reads episodes, writes eval/agent_ds/<stage>/.

usage: python -B -m scripts.agent_ds_report smoke|main
"""
from collections import defaultdict
from decimal import Decimal
import json
from math import fsum
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.agent_v1_metrics import answer_metrics, paired
from src.reader_heldout_inputs import EXTERNAL, need, rows, write_new
from src.reranker_v1 import record

WORK = EXTERNAL / "agent_ds"
SMALL = PROJECT / "eval/agent_ds"
QWEN = {"smoke": EXTERNAL / "agent_v1/try3/smoke/episodes.jsonl", "smoke2": EXTERNAL / "agent_v1/try3/smoke/episodes.jsonl",
        "main": EXTERNAL / "agent_v1/main_try2/episodes.jsonl"}
DATASETS = ("musique", "hotpotqa", "2wiki")
ROWS = ("no_compress", "oracle", "LLMLingua-2", "RR-τ reselect", "r4 reselect", "r4 append")
PAIRS = (("r4 reselect", "RR-τ reselect"), ("r4 reselect", "no_compress"), ("r4 append", "no_compress"),
         ("r4 reselect", "LLMLingua-2"), ("oracle", "r4 reselect"))


def mean(values):
    values = list(values)
    return fsum(values) / len(values) if values else None


def load(path, keep_tasks=None):
    out = {}
    for ep in rows(path):
        key = (ep["dataset"], ep["task"], ep["row"])
        if ep["row"] not in ROWS or (keep_tasks is not None and key[:2] not in keep_tasks):
            continue
        need(key not in out, f"Duplicate episode {key}")
        out[key] = ep
    return out


def questions(episodes):
    """(dataset, task, qi, row) -> Task 20 answer metrics."""
    result = {}
    for (ds, task, row), ep in episodes.items():
        for qi, (q, a) in enumerate(zip(ep["questions"], ep["answers"], strict=True)):
            result[ds, task, qi, row] = answer_metrics(ep, q, a)
    return result


def row_stats(eps, qs, *, deepseek):
    n_q = len(qs)
    stats = {
        "n_tasks": len(eps), "n_questions": n_q,
        "f1": mean(q["f1"] for q in qs), "em": mean(q["em"] for q in qs),
        "searches_per_task": mean(sum(t["action"] == "search" for t in e["turns"]) for e in eps),
        "turns_per_task": mean(len(e["turns"]) for e in eps),
        "self_finish_rate": mean(e["force_reason"] is None for e in eps),
        "format_invalid_turns": sum(t.get("format_invalid") is True for e in eps for t in e["turns"]),
        "retrieval_complete_rate": mean(q["retrieval_complete"] for q in qs),
        "evidence_complete_rate": mean(q["evidence_complete"] for q in qs),
        "gold_lost": sum(q["gold_lost"] for q in qs), "gold_retrieved": sum(q["gold_retrieved"] for q in qs),
        "kept_passages": mean(len(e["final_keep"]) for e in eps),
        "uncached_qwen_tokens_per_task": mean(sum(t["prompt_tokens"] for t in e["turns"]) for e in eps),
        "cached_cost_per_task": mean(sum(t["cached_input_token_cost"] for t in e["turns"]) for e in eps),
    }
    prompt = sum(t["prompt_tokens"] for e in eps for t in e["turns"])
    stats["predicted_hit_fraction"] = sum(t["prefix_hit_tokens"] for e in eps for t in e["turns"]) / prompt if prompt else None
    if deepseek:
        calls = [t["deepseek"] for e in eps for t in e["turns"]]
        hit, miss, out = (sum(c[k] for c in calls) for k in ("hit", "miss", "output"))
        stats.update(
            deepseek_calls=len(calls),
            deepseek_yuan_per_task=float(sum(Decimal(c["cost_yuan"]) for c in calls) / len(eps)),
            deepseek_hit_per_task=hit / len(eps), deepseek_miss_per_task=miss / len(eps),
            deepseek_output_per_task=out / len(eps),
            actual_hit_fraction=hit / (hit + miss) if hit + miss else None,
            finish_reason_length=sum(c["finish_reason"] == "length" for c in calls))
    return stats


def table(eps_all, qs_all, *, deepseek):
    out = {}
    for ds in (*DATASETS, "ALL"):
        for row in ROWS:
            eps = [e for (d, _, r), e in eps_all.items() if r == row and ds in (d, "ALL")]
            qs = [q for (d, _, _, r), q in qs_all.items() if r == row and ds in (d, "ALL")]
            if eps:
                out[f"{row}|{ds}"] = row_stats(eps, qs, deepseek=deepseek)
    return out


def pair(qs_a, qs_b, a, b, ds):
    left = {k[:3]: v for k, v in qs_a.items() if k[3] == a and ds in (k[0], "ALL")}
    right = {k[:3]: v for k, v in qs_b.items() if k[3] == b and ds in (k[0], "ALL")}
    shared = left.keys() & right.keys()
    return paired({k: left[k] for k in sorted(shared)}, {k: right[k] for k in sorted(shared)}) if shared else None


def main(stage):
    path = WORK / stage / "episodes.jsonl"
    episodes = load(path)
    by_task = defaultdict(set)
    for ds, task, row in episodes:
        by_task[ds, task].add(row)
    complete = {k for k, v in by_task.items() if v == set(ROWS)}
    partial = sorted(k for k in by_task if k not in complete)
    refused = {k[:2] for k, e in episodes.items() if e["force_reason"] == "content_refused" and k[:2] in complete}
    refusals = {row: sum(e["force_reason"] == "content_refused" for k, e in episodes.items() if k[2] == row and k[:2] in complete)
                for row in ROWS}
    with_refused = {k: v for k, v in episodes.items() if k[:2] in complete}
    complete -= refused  # TASK_21 §1.5: a task refused in any row leaves every paired comparison
    episodes = {k: v for k, v in episodes.items() if k[:2] in complete}
    qs = questions(episodes)
    qwen_eps = load(QWEN[stage], complete)
    need({k[:2] for k in qwen_eps} == complete and len(qwen_eps) == len(episodes), "Qwen episodes do not cover the tasks")
    qwen_qs = questions(qwen_eps)
    for key in qs:  # the same question text and gold in both runs
        need(qwen_eps[key[0], key[1], key[3]]["questions"][key[2]] == episodes[key[0], key[1], key[3]]["questions"][key[2]],
             f"Question identity differs {key}")
    result = {
        "stage": stage, "episodes": record(path), "qwen_episodes": record(QWEN[stage]),
        "complete_tasks": {ds: sum(k[0] == ds for k in complete) for ds in DATASETS}, "partial_tasks": partial,
        "content_refused_tasks": sorted(refused), "content_refusals_by_row": refusals,
        "sensitivity_refused_as_zero": {f"{a} − {b}|ALL": pair(questions(with_refused), questions(with_refused), a, b, "ALL")
                                        for a, b in PAIRS} if refused else None,
        "deepseek": table(episodes, qs, deepseek=True), "qwen_same_tasks": table(qwen_eps, qwen_qs, deepseek=False),
        "pairs": {f"{a} − {b}|{ds}": pair(qs, qs, a, b, ds) for a, b in PAIRS for ds in (*DATASETS, "ALL")},
        "qwen_pairs": {f"{a} − {b}|{ds}": pair(qwen_qs, qwen_qs, a, b, ds) for a, b in PAIRS for ds in (*DATASETS, "ALL")},
        "deepseek_minus_qwen": {f"{row}|{ds}": pair(qs, qwen_qs, row, row, ds)
                                for row in ROWS for ds in (*DATASETS, "ALL")},
    }
    out = SMALL / stage
    write_new(out / "agent_ds_results.json", result)
    with (out / "agent_ds_report.md").open("x", encoding="utf-8") as f:
        f.write(markdown(result))
    print((out / "agent_ds_report.md").read_text())


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}"


def get(d, key, field):
    item = d.get(key)
    return None if item is None else item.get(field)


def delta(p):
    return "—" if p is None else f"{100 * p['delta_f1']:+.2f}"


def markdown(r):
    lines = [f"# 任务21 · DeepSeek 主模型真实 Agent（{r['stage']}）", "",
             f"完整配对任务：{r['complete_tasks']}；不完整任务 {len(r['partial_tasks'])} 个；"
             f"因 DeepSeek 内容审核被拒而整体排除的任务 {len(r['content_refused_tasks'])} 个（每行被拒次数 {r['content_refusals_by_row']}）。单次运行，点估计。", "",
             "## 每行（合并）", "",
             "| 行 | n 题 | DeepSeek F1 | EM | Qwen F1 | 搜索/任务 DS / Qwen | 检索覆盖齐全 DS / Qwen | 最终证据齐全 | 保留段 | 开缓存成本/任务 | DS 实际 ¥/任务 | 命中率 预测 / 实际 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in ROWS:
        d, q = r["deepseek"][f"{row}|ALL"], r["qwen_same_tasks"][f"{row}|ALL"]
        lines.append(f"| {row} | {d['n_questions']} | {pct(d['f1'])} | {pct(d['em'])} | {pct(q['f1'])} | "
                     f"{d['searches_per_task']:.2f} / {q['searches_per_task']:.2f} | "
                     f"{pct(d['retrieval_complete_rate'])} / {pct(q['retrieval_complete_rate'])} | "
                     f"{pct(d['evidence_complete_rate'])} | {d['kept_passages']:.1f} | {d['cached_cost_per_task']:.0f} | "
                     f"{d['deepseek_yuan_per_task']:.5f} | {pct(d['predicted_hit_fraction'])} / {pct(d['actual_hit_fraction'])} |")
    lines += ["", "## 每数据集 F1（DeepSeek / Qwen）", "", "| 行 | " + " | ".join(DATASETS) + " |",
              "|---|" + "---|" * len(DATASETS)]
    for row in ROWS:
        cells = [f"{pct(get(r['deepseek'], f'{row}|{ds}', 'f1'))} / {pct(get(r['qwen_same_tasks'], f'{row}|{ds}', 'f1'))}" for ds in DATASETS]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines += ["", "## 配对比较 ΔF1（DeepSeek 主模型；括号里是同任务上的 Qwen）", "",
              "| 比较 | 合并 | " + " | ".join(DATASETS) + " |", "|---|---|" + "---|" * len(DATASETS)]
    for a, b in PAIRS:
        cells = []
        for ds in ("ALL", *DATASETS):
            p, q = r["pairs"][f"{a} − {b}|{ds}"], r["qwen_pairs"][f"{a} − {b}|{ds}"]
            cells.append("—" if p is None else f"{delta(p)}（{delta(q)}）n={p['n']} 高/平/低 {p['f1_higher']}/{p['f1_equal']}/{p['f1_lower']}")
        lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines += ["", "## DeepSeek − Qwen（同一行、同一道题）", "", "| 行 | 合并 ΔF1 | " + " | ".join(DATASETS) + " |",
              "|---|---|" + "---|" * len(DATASETS)]
    for row in ROWS:
        cells = [delta(r['deepseek_minus_qwen'][f'{row}|{ds}']) for ds in ("ALL", *DATASETS)]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    if r["sensitivity_refused_as_zero"]:
        lines += ["", "## 敏感性：把被拒任务也算进来（被拒的 episode 按 F1 = 0 计）", "", "| 比较 | 合并 ΔF1 | n |", "|---|---|---|"]
        for key, p in r["sensitivity_refused_as_zero"].items():
            lines.append(f"| {key.split('|')[0]} | {delta(p)} | {p['n'] if p else '—'} |")
    total = sum(r["deepseek"][f"{row}|ALL"]["deepseek_yuan_per_task"] * r["deepseek"][f"{row}|ALL"]["n_tasks"] for row in ROWS)
    lines += ["", f"本阶段按账本单价计的 DeepSeek 花费（只算完整任务）：¥{total:.4f}。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    need(len(sys.argv) == 2 and sys.argv[1] in ("smoke", "smoke2", "main"), "usage: agent_ds_report smoke|smoke2|main")
    main(sys.argv[1])
