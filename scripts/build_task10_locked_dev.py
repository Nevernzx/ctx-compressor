#!/usr/bin/env python3
"""Reproduce Task 02 retrieval, then build and lock the untouched Task 10 dev set on CPU."""
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EXT = Path("/root/userfolder/ctx-compressor")
DATA = EXT / "data/test_dev_v1"
EVAL = ROOT / "eval/test_dev_v1"
SEED = 20261122
BUDGETS = {"tight": 1876, "mid": 2814, "loose": 5628}
ID_PATTERN = re.compile(rb"[234]hop[123]?__[0-9]+(?:_[0-9]+)+")
LOCK = "no strategy has been run on this set; first use requires a Claude-written task"
sys.path.insert(0, str(ROOT))

from scripts.feasibility import compact, filter_examples, load_jsonl, ratio, sha256
from src.build_traj import build_corpus, sample_tasks, simulate_task
from tokenizers import Tokenizer
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open("rb") as stream:
        for line in stream:
            yield json.loads(line)


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def build_task(i, examples, index, keys):
    steps = simulate_task(examples, index, keys, top_n=20, seed=i)
    gold = [set() for _ in examples]
    for s in steps:
        ex = examples[s["question_index"]]
        s["answer"] = ex["question_decomposition"][s["hop_index"]]["answer"]
        s["question_hops"] = len(ex["question_decomposition"])
        gold[s["question_index"]].add(s["gold"])
        require(len(set(s["new"])) == len(s["new"]) == 20 and s["gold"] in s["new"], "invalid retrieval")
    return {"task": i, "example_ids": [e["id"] for e in examples], "steps": steps,
            "questions": [{k: e[k] for k in ("id", "question", "answer", "answer_aliases")} for e in examples],
            "question_gold": [sorted(g) for g in gold], "all_gold": sorted(set().union(*gold)),
            "question_hops": [len(e["question_decomposition"]) for e in examples]}


def reproduce(valid, index, keys):
    cfg = json.loads((ROOT / "eval/feasibility/run_config.json").read_bytes())
    group = sample_tasks(valid, cfg["n_tasks"], cfg["seed"])
    old = ROOT / "data/feasibility"
    out = hashlib.sha256()
    all_steps = {}
    config_seen = defaultdict(set)
    task_path = old / "tasks_dev150.jsonl"
    with task_path.open("rb") as stream:
        for i, (examples, existing) in enumerate(zip(group, stream, strict=True)):
            task = build_task(i, examples, index, keys)
            encoded = (compact(task) + "\n").encode()
            require(encoded == existing, f"Task 02 retrieval reproduction mismatch at task {i}")
            out.update(encoded)
            all_steps.update({(i, j): s for j, s in enumerate(task["steps"])})
    count = 0
    step_path = old / "steps.jsonl"
    for r in rows(step_path):
        key = r["task"], r["step"]
        s = all_steps[key]
        require(r["returned"] == s["new"] and all(r[k] == s[k] for k in ("gold", "question_index", "hop_index")),
                f"Task 02 historical steps mismatch at {key}, config={r['config']}")
        require(r["config"] not in config_seen[key], f"duplicate historical config/step {key}")
        config_seen[key].add(r["config"])
        count += 1
    expected = {"no_compress"} | {f"{p}|{m}|{t}" for p in ("latest", "bm25", "oracle")
                                 for m in ("append", "reselect") for t in BUDGETS}
    require(set(config_seen) == set(all_steps) and all(v == expected for v in config_seen.values()),
            "missing Task 02 steps/configs")
    return {"task_count": len(group), "distinct_steps": len(all_steps), "historical_rows": count,
            "configs_per_step": len(expected), "rebuilt_tasks_byte_identical": True,
            "tasks_sha256": out.hexdigest(), "historical_steps_sha256": sha256(step_path),
            "step_comparison": "All configs: (task,step,question_index,hop_index,gold,returned); returned is ordered full retrieval; historical new is the fresh subset after removing retained candidates. Other retrieval fields are covered by byte-identical task JSONL; policy outputs were not recomputed."}


def historical_files():
    # Restrict to the earlier experiments. Do not traverse raw data, model and dependency
    # caches, or outputs from the current task (which could feed the new lock back into itself).
    subroots = [ROOT / p for p in ("data/feasibility", "data/reader", "data/splits", "data/traj",
                                  "eval/acceptance", "eval/feasibility", "eval/reader", "eval/sft_prep",
                                  "eval/sft_run", "eval/sft_run_v2", "eval/grpo_prep", "eval/grpo_v2",
                                  "eval/hedge_v2", "specs", "logs")]
    subroots += [EXT / p for p in ("data/traj_train", "data/traj_train_v2", "data/sft", "data/sft_v2",
                                  "data/splits", "runs/sft_20k_20261117", "runs/sft_20k_v2_20261118",
                                  "runs/batch_bench_20261117", "runs/bench_task04_20261117",
                                  "eval_rollouts", "eval_rollouts_v2", "eval_rollouts_grpo_v2",
                                  "eval_rollouts_hedge_v2", "hedge_v2", "grpo_prep", "grpo_v2", "logs")]
    yield from sorted(ROOT.glob("*.md"))
    for base in subroots:
        require(base.is_dir(), f"missing previous experiment tree {base}")
        for folder, children, files in os.walk(base):
            children[:] = sorted(c for c in children if c not in ("tmp", "cache", "__pycache__", ".git"))
            for f in sorted(files):
                p = Path(folder) / f
                if p.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".csv", ".tsv", ".xml", ".log", ".py"}:
                    yield p


def history(dev_ids):
    found = set()
    hits = {}
    nfiles, nbytes = 0, 0
    require(all(ID_PATTERN.fullmatch(x.encode()) for x in dev_ids), "dev ID syntax changed")
    accepted = {i for t in json.loads((ROOT / "eval/acceptance/acceptance_checks.json").read_bytes())["tasks"]
                for i in t["example_ids"]}
    feasibility = {i for t in rows(ROOT / "data/feasibility/tasks_dev150.jsonl") for i in t["example_ids"]}
    known = accepted | feasibility
    for p in historical_files():
        nfiles += 1
        nbytes += p.stat().st_size
        matches = set()
        with p.open("rb") as stream:
            while block := stream.read(4 << 20):
                # Complete the last line so an ID never straddles two blocks.
                if not block.endswith(b"\n"):
                    block += stream.readline()
                matches.update(x.decode() for x in set(ID_PATTERN.findall(block)) if x.decode() in dev_ids)
        if matches:
            hits[str(p)] = {"n_distinct_dev_ids": len(matches), "extra_ids_vs_task01_acceptance_and_02": sorted(matches-known)}
            found.update(matches)
    require(len(accepted) == 12 and len(feasibility) == 600 and known <= found <= dev_ids, "incomplete source audit")
    return found, {"scanned_files": nfiles, "scanned_bytes": nbytes, "files_with_dev_ids": hits,
                   "rule": "Exact raw dev question IDs found in previous experiment records, logs, reports, and saved examples; scan includes external training and held-out data and outputs. Raw source corpus membership alone is not prior question use. Training and dev raw question ID sets were checked disjoint.",
                   "task01_acceptance": len(accepted), "task02_feasibility": len(feasibility),
                   "acceptance_and_feasibility_intersection": len(accepted & feasibility),
                   "additional_ids": sorted(found-known), "union": len(found)}


def grouped_ids(valid, excluded, seed=SEED):
    ids = [q["id"] for q in valid]
    require(len(ids) == len(set(ids)), "duplicate question ID")
    remaining = [i for i in ids if i not in excluded]
    random.Random(seed).shuffle(remaining)
    used = len(remaining) // 4 * 4
    return [remaining[k:k+4] for k in range(0, used, 4)], remaining[used:]


def row_keys(path):
    out = set()
    for r in rows(path):
        m = r["meta"]
        key = m["task"], m["mode"], m["step"]
        require(m["split"] == "train" and key not in out, f"invalid train row {key}")
        out.add(key)
    return out


def overlap_stats(corpus, test_pids):
    test_keys = {(corpus[p]["title"], corpus[p]["paragraph_text"]) for p in test_pids}
    train_corpus = EXT / "data/traj_train/corpus_train.jsonl"
    exact = {r["pid"] for r in rows(train_corpus) if (r["title"], r["paragraph_text"]) in test_keys}
    manifest = json.loads((ROOT / "eval/sft_run_v2/sample_manifest.json").read_bytes())
    pool = row_keys(Path(manifest["sources"]["train"]["path"]))
    actual = row_keys(Path(manifest["files"]["train"]["path"]))
    require(len(pool) == manifest["sources"]["train"]["lines"] and
            len(actual) == manifest["files"]["train"]["lines"] and actual <= pool, "v2 train row identity")
    all_gold, optimized_gold = set(), set()
    seen = set()
    for t in rows(EXT / "data/traj_train_v2/train_tasks.jsonl"):
        for mode, states in t["rollouts"].items():
            for i, s in enumerate(states):
                key = t["task"], mode, i
                if key not in pool:
                    continue
                require(key not in seen, f"duplicate trajectory state {key}")
                seen.add(key)
                gold = {s["pid_by_lid"][str(lid)] for lid in s["gold"]}
                all_gold.update(gold)
                if key in actual:
                    optimized_gold.update(gold)
    require(pool == seen, "v2 train state metadata incomplete")
    require(len(test_pids) == len(test_keys), "duplicate dev PID for one title/body")
    n = len(test_keys)
    return {"denominator": "distinct test gold (title, original paragraph_text) across all locked tasks, counted once per pair; compare strings verbatim, including case and whitespace",
            "train_corpus": ratio(len(exact), n), "sft_v2_gold_any_training_pool_row": ratio(len(exact & all_gold), n),
            "sft_v2_gold_any_optimized_row": ratio(len(exact & optimized_gold), n),
            "full_sft_v2_training_pool_rows": len(pool), "actually_optimized_unique_rows": len(actual),
            "gold_lookup": "row meta (task,mode,step) -> matching Task 06 train rollout state -> state.gold mapped via pid_by_lid -> Task 04 train corpus; held-out rows excluded"}


def main():
    require(ROOT == Path("/root/workplace/ctx-compressor") and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "run on 5880 with CUDA_VISIBLE_DEVICES=''")
    require(not DATA.exists() and not EVAL.exists(), "refusing to replace existing lock directory")
    raw = ROOT / "data/raw/musique"
    cfg = json.loads((ROOT / "eval/feasibility/run_config.json").read_bytes())
    for name, expected in {**cfg["dataset_sha256"], **cfg["tokenizer_sha256"]}.items():
        require(sha256(ROOT / name) == expected, f"Task 02 input changed: {name}")
    dev = load_jsonl(raw / "musique_ans_v1.0_dev.jsonl")
    dev_ids = {q["id"] for q in dev}
    train_ids = {q["id"] for q in rows(raw / "musique_ans_v1.0_train.jsonl")}
    require(len(dev_ids) == len(dev) == 2417 and dev_ids.isdisjoint(train_ids), "dev/train ID overlap")
    tok = Tokenizer.from_file(str(ROOT / "data/raw/qwen3_tokenizer/tokenizer.json"))
    corpus, index, keys = build_corpus(dev, lambda s: len(tok.encode(s, add_special_tokens=False).ids))
    valid, reject = filter_examples(dev, keys)
    require(reject["rejected"] == 0, "unexpected invalid dev questions")
    repro = reproduce(valid, index, keys)
    print("Task 02 reproduction passed", compact(repro), flush=True)
    excluded, audit = history(dev_ids)
    groups, remainder = grouped_ids(valid, excluded)
    by_id = {q["id"]: q for q in valid}
    tasks = [build_task(i, [by_id[k] for k in ids], index, keys) for i, ids in enumerate(groups)]
    gold = {p for t in tasks for p in t["all_gold"]}
    steps = [s for t in tasks for s in t["steps"]]
    union = {i for t in tasks for i in t["example_ids"]}
    require(len(tasks)*4 == len(union) and union.isdisjoint(excluded)
            and dev_ids == excluded | union | set(remainder), "test-set partition")
    p90 = float(np.percentile([sum(corpus[p]["ntok"] for p in t["all_gold"]) for t in tasks], 90))
    stats = {"split": "MuSiQue-Ans dev", "seed": SEED, "budgets": BUDGETS, "source_questions": len(dev_ids),
             "dev_corpus_paragraphs": len(corpus), "excluded_questions": len(excluded), "remaining_before_grouping": len(dev_ids-excluded),
             "dropped_remainder_ids": remainder, "tasks": len(tasks), "questions": len(union), "steps": len(steps),
             "hop_distribution_questions": dict(sorted(Counter(h for t in tasks for h in t["question_hops"]).items())),
             "natural_retrieval_hit": ratio(sum(s["natural_hit"] for s in steps), len(steps)),
             "per_task_unique_gold_token_p90": {"value": p90, "n_tasks": len(tasks), "task02_reference": 1875.6,
                                               "task02_n_tasks": 150, "delta": round(p90-1875.6, 9)},
             "gold_overlap": overlap_stats(corpus, gold), "reproduction_task02": repro,
             "package_versions": {p: version(p) for p in ("numpy", "bm25s", "tokenizers", "scipy")}}
    DATA.mkdir(parents=True)
    EVAL.mkdir(parents=True)
    save(DATA / "excluded_ids.json", json_text(sorted(excluded)))
    save(DATA / "test_tasks.jsonl", "".join(compact(t)+"\n" for t in tasks))
    save(DATA / "steps.jsonl", "".join(compact({"task": t["task"], "step": i, **s})+"\n"
                                       for t in tasks for i, s in enumerate(t["steps"])))
    save(EVAL / "exclusion_audit.json", json_text(audit))
    save(EVAL / "structural_stats.json", json_text(stats))
    paths = [DATA / n for n in ("test_tasks.jsonl", "steps.jsonl", "excluded_ids.json")]
    paths += [EVAL / "exclusion_audit.json", EVAL / "structural_stats.json"]
    hashes = {str(p): sha256(p) for p in paths}
    md = ["# MuSiQue dev 最终测试集锁定记录", "", LOCK, "",
          f"原始 dev 共 {len(dev_ids)} 题。历史排除 {len(excluded)} 题，包含任务01验收 {audit['task01_acceptance']} 题、任务02/03可行性 {audit['task02_feasibility']} 题，两者交集 {audit['acceptance_and_feasibility_intersection']} 题。其他历史产物额外排除 {len(audit['additional_ids'])} 题，逐文件来源见 exclusion_audit.json。", "",
          f"按原始 dev 顺序取剩余 ID，用 random.Random({SEED}).shuffle 后连续每四题分组。锁定 {len(tasks)} 个任务（n={len(union)} 题），丢弃尾部 {len(remainder)} 题；预算 1876 / 2814 / 5628。", "",
          f"先复现任务02的 {repro['task_count']} 个任务，完整任务文件逐字节相同。历史 steps.jsonl 中 {repro['historical_rows']} 行（n={repro['distinct_steps']} 个检索步骤，每步 {repro['configs_per_step']} 个配置）均与复现的有序返回列表及金标、题序和跳序相同。", "",
          "结构量、证据 token P90、训练集逐字重合率及分母见 structural_stats.json。", "", "## SHA256", ""]
    md.extend(f"- {p}  {d}" for p, d in hashes.items())
    save(EVAL / "LOCK.md", "\n".join(md)+"\n")
    print(json_text({"stats": stats, "hashes": hashes, "lock_sha256": sha256(EVAL / "LOCK.md")}), flush=True)


if __name__ == "__main__":
    main()

