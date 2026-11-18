#!/usr/bin/env python3
"""Offline MuSiQue data audit; execute on 5880 only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.build_traj import hop_queries, render_paragraph


OTHER_REFERENCE_PATTERNS = {
    "hash_without_adjacent_digit": re.compile(r"#(?!\d)"),
    "bracketed_number": re.compile(r"\[\d+\]"),
    "dollar_number": re.compile(r"\$\d+"),
    "angle_or_xml_reference": re.compile(r"<\d+>|<\s*(?:ref|answer|hop)[^>]*>", re.I),
    "brace_reference": re.compile(r"\{+\s*(?:\d+|(?:ref|answer|hop)[_ ]*\d+)\s*\}+", re.I),
}


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def reference_warnings(ex):
    warnings = []
    for i, hop in enumerate(ex["question_decomposition"]):
        query = hop.get("question", "")
        if not isinstance(query, str):
            continue
        for name, pattern in OTHER_REFERENCE_PATTERNS.items():
            if pattern.search(query):
                warnings.append({"hop": i + 1, "pattern": name, "query": query})
    return warnings


def paragraph_keys(examples):
    return {(p["title"], p["paragraph_text"]) for ex in examples for p in ex["paragraphs"]}


def valid_examples(examples, keys):
    good, skipped = [], []
    for ex in examples:
        warnings = reference_warnings(ex)
        if warnings:
            skipped.append({"id": ex["id"], "reason": "unrecognized_reference_pattern", "details": warnings})
            continue
        try:
            hop_queries(ex, keys)
        except (ValueError, KeyError, TypeError) as exc:
            skipped.append({"id": ex.get("id"), "reason": str(exc)})
            continue
        good.append(ex)
    return good, skipped


def decomposition_audit(examples, keys):
    counts = Counter()
    hop_distribution = Counter()
    warning_examples = []
    samples = {2: [], 3: [], 4: []}
    for ex in examples:
        hops = ex["question_decomposition"]
        hop_distribution[len(hops)] += 1
        if len(hops) in samples and len(samples[len(hops)]) < 5:
            samples[len(hops)].append({"id": ex["id"], "question": ex["question"], "question_decomposition": hops})
        has_reference = has_relation = bad_support = False
        warnings = reference_warnings(ex)
        if warnings:
            counts["questions_with_other_reference_patterns"] += 1
            counts["hops_with_other_reference_patterns"] += len({w["hop"] for w in warnings})
            warning_examples.append({"id": ex["id"], "warnings": warnings})
        for hop in hops:
            counts["hops"] += 1
            q = hop.get("question", "")
            if isinstance(q, str):
                refs = re.findall(r"#(\d+)", q)
                counts["hash_k_occurrences"] += len(refs)
                counts["hops_with_hash_k"] += bool(refs)
                counts["hops_with_relation_separator"] += ">>" in q
                has_reference |= bool(refs)
                has_relation |= ">>" in q
            if "paragraph_support_idx" not in hop:
                counts["hops_missing_support_idx"] += 1
                bad_support = True
            elif hop["paragraph_support_idx"] in (None, ""):
                counts["hops_empty_support_idx"] += 1
                bad_support = True
        counts["questions_with_hash_k"] += has_reference
        counts["questions_with_relation_separator"] += has_relation
        counts["questions_missing_or_empty_support_idx"] += bad_support
    good, skipped = valid_examples(examples, keys)
    for name in ("questions_with_other_reference_patterns", "hops_with_other_reference_patterns",
                 "hops_missing_support_idx", "hops_empty_support_idx", "questions_missing_or_empty_support_idx"):
        counts.setdefault(name, 0)
    return {
        "examples": len(examples), "counts": dict(counts),
        "hop_distribution": dict(sorted(hop_distribution.items())),
        "valid_examples": len(good), "skipped_examples": len(skipped),
        "skip_reason_counts": dict(Counter(item["reason"] for item in skipped)),
        "skipped": skipped, "other_reference_examples": warning_examples,
        "samples_by_hop_count": samples,
        "other_reference_detection": "Conservative lexical scan: residual #, [n], $n, <n>/XML ref/answer/hop tags, brace references. Flags are suspected syntax, not semantic proof; flagged examples are skipped rather than guessed.",
    }


def overlap_stats(train_keys, dev_keys):
    train_titles = {title for title, _ in train_keys}
    dev_titles = {title for title, _ in dev_keys}
    shared_keys = train_keys & dev_keys
    shared_titles = train_titles & dev_titles
    train_shared_paragraphs = sum(title in shared_titles for title, _ in train_keys)
    dev_shared_paragraphs = sum(title in shared_titles for title, _ in dev_keys)
    return {
        "train_unique_paragraphs": len(train_keys), "dev_unique_paragraphs": len(dev_keys),
        "train_unique_titles": len(train_titles), "dev_unique_titles": len(dev_titles),
        "exact_title_body": {"intersection": len(shared_keys),
                             "fraction_of_train": len(shared_keys) / len(train_keys),
                             "fraction_of_dev": len(shared_keys) / len(dev_keys)},
        "title_only": {"intersection": len(shared_titles),
                       "fraction_of_train_titles": len(shared_titles) / len(train_titles),
                       "fraction_of_dev_titles": len(shared_titles) / len(dev_titles),
                       "train_paragraphs_with_shared_title": train_shared_paragraphs,
                       "dev_paragraphs_with_shared_title": dev_shared_paragraphs,
                       "train_paragraph_fraction": train_shared_paragraphs / len(train_keys),
                       "dev_paragraph_fraction": dev_shared_paragraphs / len(dev_keys)},
    }


def report_markdown(audit):
    overlap = audit["overlap"]
    dev = audit["decomposition_dev"]
    counts = dev["counts"]
    exact, titles = overlap["exact_title_body"], overlap["title_only"]
    p = audit["dev_unique_paragraph_tokens"]
    lines = ["## MuSiQue 原始数据核对", "",
             f"核对时间（UTC）：{audit['generated_at_utc']}。仅做任务 01 规定的字段、重叠及段落长度核对。", "",
             f"MuSiQue-Ans train / dev：{audit['train_examples']:,} / {audit['dev_examples']:,} 题。",
             f"按 `(title, paragraph_text)` 精确去重后：train **{overlap['train_unique_paragraphs']:,}** 段，dev **{overlap['dev_unique_paragraphs']:,}** 段。", "",
             "| 重叠口径 | 交集数 | train 分母及比例 | dev 分母及比例 |",
             "|---|---:|---:|---:|",
             f"| 标题 + 正文完全一致 | {exact['intersection']:,} | {overlap['train_unique_paragraphs']:,}；{exact['fraction_of_train']:.6%} | {overlap['dev_unique_paragraphs']:,}；{exact['fraction_of_dev']:.6%} |",
             f"| 标题完全一致（唯一标题） | {titles['intersection']:,} | {overlap['train_unique_titles']:,}；{titles['fraction_of_train_titles']:.6%} | {overlap['dev_unique_titles']:,}；{titles['fraction_of_dev_titles']:.6%} |", "",
             f"共有标题对应 train {titles['train_paragraphs_with_shared_title']:,} 段（占 train 去重段落 {titles['train_paragraph_fraction']:.6%}）、dev {titles['dev_paragraphs_with_shared_title']:,} 段（占 dev 去重段落 {titles['dev_paragraph_fraction']:.6%}）；标题可对应不同正文。去重和重叠均不做大小写、空白或 Unicode 归一化。", "",
             f"dev 去重段落按 `标题 + 换行 + 正文` 渲染；Qwen3-1.7B tokenizer.json 离线分词，`add_special_tokens=False`。P50 / P90 / P99 = **{p['p50']:g} / {p['p90']:g} / {p['p99']:g} token**；共 {p['count']:,} 段。分位数采用 NumPy 线性插值；预算是这些片段 token 数的和，不含输入模板元数据。", "",
             "### 子问题占位符核对", "",
             f"dev 跳数分布：`{json.dumps(dev['hop_distribution'], ensure_ascii=False, sort_keys=True)}`。",
             f"`#k` 出现在 {counts['questions_with_hash_k']:,} 题、{counts['hops_with_hash_k']:,} 跳，共 {counts['hash_k_occurrences']:,} 次；`>>` 出现在 {counts['questions_with_relation_separator']:,} 题、{counts['hops_with_relation_separator']:,} 跳。",
             "`#k` 按题内第 k 个已完成子问题的答案替换；`实体 >> 关系` 保留为普通检索字符串。",
             f"疑似其他引用写法：{counts['questions_with_other_reference_patterns']} 题 / {counts['hops_with_other_reference_patterns']} 跳。检测范围：非标准 #、[n]、$n、数字或 ref/answer/hop XML 标签、花括号引用；无法由有限正则证明不存在任意自然语言引用。",
             f"`paragraph_support_idx` 缺字段 {counts['hops_missing_support_idx']} 跳、空值 {counts['hops_empty_support_idx']} 跳，涉及 {counts['questions_missing_or_empty_support_idx']} 题。",
             f"经过引用、证据索引和语料映射校验，可用于模拟 {dev['valid_examples']:,} 题，保守跳过 {dev['skipped_examples']} 题；原因计数：`{json.dumps(dev['skip_reason_counts'], ensure_ascii=False, sort_keys=True)}`。", "",
             "### 2 / 3 / 4 跳各 5 条原始 question_decomposition", "",
             "按 dev 文件顺序取各类前 5 题；以下保留原始字段及占位符，供人工核对。", ""]
    for hops, examples in dev["samples_by_hop_count"].items():
        lines.extend([f"#### {hops} 跳", ""])
        for ex in examples:
            lines.extend([f"题号：`{ex['id']}`", "", f"问题：{ex['question']}", "", "```json",
                          json.dumps(ex["question_decomposition"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PROJECT / "data/raw/musique/musique_ans_v1.0_train.jsonl")
    parser.add_argument("--dev", type=Path, default=PROJECT / "data/raw/musique/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT / "data/raw/qwen3_tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "eval/acceptance")
    args = parser.parse_args()
    started = time.monotonic()
    train, dev = load_jsonl(args.train), load_jsonl(args.dev)
    train_keys, dev_keys = paragraph_keys(train), paragraph_keys(dev)
    key2pid = {key: i for i, key in enumerate(sorted(dev_keys))}
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    lengths = [len(tokenizer.encode(render_paragraph(*key), add_special_tokens=False).ids) for key in sorted(dev_keys)]
    percentiles = np.percentile(lengths, [50, 90, 99], method="linear")
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_examples": len(train), "dev_examples": len(dev),
        "overlap": overlap_stats(train_keys, dev_keys),
        "decomposition_dev": decomposition_audit(dev, key2pid),
        "dev_unique_paragraph_tokens": {"count": len(lengths), "p50": float(percentiles[0]),
                                       "p90": float(percentiles[1]), "p99": float(percentiles[2]),
                                       "tokenizer": str(args.tokenizer), "special_tokens": False,
                                       "rendering": "title + newline + paragraph_text"},
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "data_audit.md").write_text(report_markdown(audit), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "decomposition_dev"}, ensure_ascii=False, indent=2))
    print(f"decomposition: {audit['decomposition_dev']['valid_examples']} valid, {audit['decomposition_dev']['skipped_examples']} skipped", flush=True)


if __name__ == "__main__":
    main()
