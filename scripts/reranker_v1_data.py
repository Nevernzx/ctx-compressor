"""Prepare disjoint Task 17 candidates, tokenizer banks and read-only audit receipts."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.build_traj import make_sample, render_input, render_paragraph
from src.model_inference import sha256_json
from src.policies import bm25_policy
from src.reader_heldout_inputs import CELLS, historical_sources, rows, write_new
from src.reranker_v1 import (DATA, MODEL, SEED, SMALL, WORK, PairTokenizer, candidates,
                            checksum, distribution, histogram, manifest_entries, need,
                            read, record, require_new, reranker_action, sample_pair_rows,
                            signed_project, split_task_ids, sums, verify)

PARTITIONS = ("train", "train-val", "heldout")
DATA_OUT = WORK / "data"
RAWS = PROJECT / "data/raw/musique"
SPLIT = DATA_OUT / "split.json"
TOKENS = DATA_OUT / "token_banks.json"
R2 = "grpo_r2:checkpoint-125"


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def dump_rows(path, iterable):
    count = 0
    with path.open("x", encoding="utf-8") as f:
        for row in iterable:
            f.write(compact(row) + "\n")
            count += 1
    return {"n": count, **record(path)}


def signed_inputs():
    manifest = PROJECT / "eval/sft_prep/trajectory_stats.json"
    sig = signed_project(manifest)
    stats = read(manifest)
    files = {"trajectory_stats": {"path": str(manifest), "sha256": sig}}
    for kind, basename in (("train", "train_tasks.jsonl"), ("heldout", "heldout_tasks.jsonl"),
                           ("corpus", "corpus_train.jsonl")):
        path = DATA / basename
        info = stats["files"][kind]
        need(path == Path(info["path"]) and path.stat().st_size == info["bytes"], f"{kind} path/size changed")
        verify(path, info["sha256"])
        files[kind] = dict(info)
    return files


def render_audit(task, corpus):
    """Take the document from make_sample; verify against actual full prompt bytes."""
    checked = set()
    span_count = 0
    state_ids = {}
    for step in task["steps"]:
        state = make_sample(task["questions"], [], step["new"], {step["gold"]}, corpus, task["budget"],
                            "append", state_ids)
        full = render_input(state)
        for lid, pid in state["pid_by_lid"].items():
            body = state["text"][lid]
            expected = f"[编号 {lid} | {state['ntok'][lid]} token]\n{body}"
            need(expected in full and body.encode("utf-8") == corpus[pid]["text"].encode("utf-8"),
                 "reranker document differs from the compressor prompt")
            span_count += 1
            checked.add(pid)
    return {"task": task["task"], "rendered_prompt_spans": span_count, "distinct_documents": len(checked),
            "different": 0}


def rendered_documents(corpus):
    docs = [render_paragraph(p["title"], p["paragraph_text"]) for p in corpus]
    need(all(rendered.encode("utf-8") == p["text"].encode("utf-8") for rendered,p in zip(docs,corpus,strict=True)),
         "corpus texts differ from the compressor fragment renderer")
    return docs


def compact_task(row):
    return {k: row[k] for k in ("task", "split", "questions", "example_ids", "steps")}


def split():
    require_new(DATA_OUT)
    files = signed_inputs()
    raw_train = RAWS / "musique_ans_v1.0_train.jsonl"
    raw_dev = RAWS / "musique_ans_v1.0_dev.jsonl"
    files["raw_train"] = {"path": str(raw_train), "sha256": signed_project(raw_train)}
    files["raw_dev_identity_source"] = {"path": str(raw_dev), "sha256": signed_project(raw_dev)}
    corpus = list(rows(DATA / "corpus_train.jsonl"))
    need(all(p["pid"] == i for i,p in enumerate(corpus)), "corpus order changed")
    train, held = {}, {}
    audits = {}
    selected_audit = set(random.Random(SEED).sample(range(4735), 3))
    for kind, output in (("train", train), ("heldout", held)):
        for row in rows(Path(files[kind]["path"])):
            task = row["task"]
            need(task not in output and row["split"] == kind and len(row["questions"]) == len(row["example_ids"]) == 4,
                 "saved task identity or split differs")
            if kind == "train" and task in selected_audit:
                audits[task] = render_audit(row, corpus)
            output[task] = compact_task(row)
    need(set(train) == set(range(4735)) and set(held) == set(range(249)) and len(audits) == 3, "task counts differ")
    train_ids, val_ids, val100 = split_task_ids(train)
    all_splits = {"train": [train[i] for i in train_ids], "train-val": [train[i] for i in val_ids],
                  "heldout": list(held.values())}
    ids = {kind: {i for row in group for i in row["example_ids"]} for kind, group in all_splits.items()}
    locked_source_ids = {row["id"] for row in rows(raw_dev)}
    train_source_ids = {row["id"] for row in rows(raw_train)}
    counts = {}
    for kind, group in all_splits.items():
        need(len(ids[kind]) == 4 * len(group), "within-split question reuse")
        need(ids[kind] <= train_source_ids and not ids[kind] & locked_source_ids, "source-level split overlap")
        counts[kind] = {"examples": len(ids[kind]), "raw_dev_overlap": 0, "n_tasks": len(group)}
    need(all(ids[kind].isdisjoint(ids[other]) for i,kind in enumerate(PARTITIONS) for other in PARTITIONS[i+1:]),
         "train, train-val and heldout overlap")
    DATA_OUT.mkdir()
    partitions = {}
    for kind, group in all_splits.items():
        ps, ns = [], []
        for row in group:
            pairs = candidates(row["steps"])
            need(all(0 <= pid < len(corpus) for pid,_ in pairs), "candidate pid missing from corpus")
            ps.append(sum(y for _, y in pairs))
            ns.append(len(pairs))
        partitions[kind] = {"tasks": dump_rows(DATA_OUT / f"{kind}.jsonl", group), "counts": counts[kind],
                            "candidates_per_task": distribution(ns), "candidates_histogram": histogram(ns),
                            "positives_per_task": distribution(ps), "positives_histogram": histogram(ps)}
    proof = {"status": "passed", "seed": SEED, "train_tasks": train_ids, "train_val_tasks": val_ids,
             "val100_tasks": val100, "partitions": partitions, "input_files": files,
             "train_val_heldout_question_overlaps": 0, "locked_test_overlap": 0,
             "locked_test_identity_check": "train-source example IDs are disjoint from the entire raw dev source; no locked artifact opened",
             "raw_dev_source_examples": len(locked_source_ids),
             "document_render_audits": [audits[i] for i in sorted(audits)]}
    write_new(SPLIT, proof)
    write_new(SMALL / "data_preflight.json", proof)
    print("Data splits:", len(train_ids), len(val_ids), len(held), flush=True)


def verify_split():
    verify(SPLIT, checksum(SMALL / "data_preflight.json"))
    split = read(SPLIT)
    need(split["status"] == "passed" and len(split["val100_tasks"]) == 100 and split["locked_test_overlap"] == 0,
         "data preflight did not pass")
    for key in PARTITIONS:
        r = split["partitions"][key]["tasks"]
        verify(Path(r["path"]), r["sha256"])
    return split


def tokenizer_card():
    pinned = read(WORK / "download/pinned_files.json")
    wanted = {"tokenizer.json", "vocab.json", "merges.txt", "config.json", "tokenizer_config.json"}
    hashes = {r["Path"]: r["Sha256"] for r in pinned["files"] if r["Path"] in wanted}
    need(hashes.keys() == wanted, "tokenizer file set missing")
    for name, sha in hashes.items():
        verify(MODEL / name, sha)
    return hashes


def pair_tokens():
    require_new(TOKENS)
    split = verify_split()
    hashes = tokenizer_card()
    from transformers import AutoTokenizer
    pair = PairTokenizer(AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False))
    corpus_path = DATA / "corpus_train.jsonl"
    verify(corpus_path, split["input_files"]["corpus"]["sha256"])
    rendered = rendered_documents(list(rows(corpus_path)))
    tokens = {}
    for kind in PARTITIONS:
        print(f"Encoding {kind}", flush=True)
        index_path = DATA_OUT / (kind + ".pairs.jsonl")
        token_path = DATA_OUT / (kind + ".tokens.i32")
        require_new(index_path, token_path)
        offset = count = 0
        labels = defaultdict(list)
        lengths, truncations = [], []
        encodings = rows(DATA_OUT / f"{kind}.jsonl")
        with token_path.open("xb") as binfile, index_path.open("x", encoding="utf-8") as index:
            for row in encodings:
                pids = candidates(row["steps"])
                query = "\n".join(row["questions"])
                documents = [rendered[pid] for pid,_ in pids]
                for (pid, label), (ids, truncated) in zip(pids, pair.encode(query, documents), strict=True):
                    payload = np.asarray(ids, dtype="<i4")
                    binfile.write(payload.tobytes())
                    index.write(compact({"task": row["task"], "pid": pid, "label": label, "offset": offset,
                                         "length": len(ids), "truncated": truncated}) + "\n")
                    labels[row["task"]].append((count,label))
                    lengths.append(len(ids))
                    truncations.append(truncated)
                    offset += len(ids)
                    count += 1
        bank = {"index": record(index_path), "tokens": record(token_path), "pairs": count,
                "tokens_count": offset, "truncated": sum(truncations), "lengths": distribution(lengths)}
        if kind == "train":
            bank["negative_ratios"] = {}
            for multiplier in (7,3):
                sampled = sample_pair_rows(labels, multiplier)
                target = DATA_OUT / f"sample_order_{multiplier}.npy"
                with target.open("xb") as f:
                    np.save(f, np.asarray(sampled, dtype=np.int64), allow_pickle=False)
                bank["negative_ratios"][str(multiplier)] = {"file": record(target), "n": len(sampled),
                    "positives": sum(label for pairs in labels.values() for _, label in pairs),
                    "truncated": sum(truncations[i] for i in sampled)}
        tokens[kind] = bank
        print(f"{kind}: n={count}, truncated={sum(truncations)}", flush=True)
    summary = {"status": "passed", "split": record(SPLIT), "template": pair.config(),
               "tokenizer_sha256": hashes, "partitions": tokens}
    write_new(TOKENS, summary)
    write_new(SMALL / "tokenization.json", summary)


class Bank:
    def __init__(self, kind, description):
        need(kind in PARTITIONS, "wrong pair bank")
        entry = description["partitions"][kind]
        path = Path(entry["index"]["path"])
        token_path = Path(entry["tokens"]["path"])
        for p, key in ((path,"index"),(token_path,"tokens")):
            need(p.is_relative_to(DATA_OUT), "unexpected encoded file path")
            verify(p, entry[key]["sha256"])
        self.index = list(rows(path))
        self.ids = np.memmap(token_path, mode="r", dtype="<i4")
        need(len(self.index) == entry["pairs"] and len(self.ids) == entry["tokens_count"], "bank incomplete")
        self.by_task = defaultdict(list)
        for index, row in enumerate(self.index):
            self.by_task[row["task"]].append(index)

    def get(self, index):
        item = self.index[index]
        start = item["offset"]
        count = item["length"]
        return self.ids[start:start+count].tolist()


def eval_sources():
    global_sums = sums(PROJECT / "SHA256SUMS")
    return {source.name:source for source in historical_sources(global_sums)}


def audit_bm25():
    source = eval_sources()["bm25"]
    verify(source.steps, source.steps_sha256)
    picked = sorted(random.Random(SEED).sample(range(249), 10))
    cases = [r for r in rows(source.steps) if r["task"] in picked]
    corpus_count = read(SPLIT)["input_files"]["corpus"]["lines"]
    per_task = {task: np.zeros(corpus_count) for task in picked}
    scores = {}
    for row in cases:
        truck = scores.setdefault(row["task"], {})
        for pid, score in row["bm25_scores"].items():
            if pid in truck:
                need(truck[pid] == score, "inconsistent archived BM25 score for one pair")
            truck[pid] = score
            per_task[row["task"]][int(pid)] = score
    refs = {task: bm25_policy(None, scores=values) for task, values in per_task.items()}
    grid, mismatch = defaultdict(int), 0
    for row in cases:
        state = dict(row["state"])
        for field in ("ntok","pid_by_lid"):
            state[field] = {int(k):v for k,v in state[field].items()}
        score = {int(pid):v for pid,v in row["bm25_scores"].items()}
        expected = row["parsed_action"]
        action = reranker_action(state, score)
        mismatch += int(action != expected or action != refs[row["task"]](state))
        grid[f"{row['mode']}|{row['budget_tier']}"] += 1
    need(len(grid) == 6 and len(cases)>0 and mismatch == 0, "archived BM25 and reranker first-fit actions differ")
    audit = {"status":"passed", "task_ids":picked, "n_tasks":10, "n_steps":len(cases),
             "n_different":mismatch, "by_cell":dict(grid), "source": {"path":str(source.steps),"sha256":source.steps_sha256}}
    write_new(SMALL/"bm25_action_check.json", audit)
    print(f"BM25 policy replay matched {len(cases)} steps across 10 tasks",flush=True)


def latency_states():
    output = DATA_OUT / "latency_states.jsonl"
    require_new(output)
    source = eval_sources()[R2]
    verify(source.steps,source.steps_sha256)
    r2_dir = PROJECT/"eval/grpo_r2"
    signatures = manifest_entries(r2_dir/"TASK11_SHA256SUMS")
    cfg_path = r2_dir/"checkpoint-125/run.json"
    verify(cfg_path,signatures[str(cfg_path.relative_to(PROJECT))])
    cfg = read(cfg_path)
    need(cfg["status"] == "complete" and cfg["max_tokens"] == 256 and cfg["engine_parameters"]["enforce_eager"],
         "unexpected signed r2 configuration")
    groups = defaultdict(list)
    for row in rows(source.steps):
        meta = row["generation_metadata"]
        ids = meta["prompt_token_ids"]
        need(sha256_json(ids) == meta["prompt_sha256"] and len(ids) == meta["prompt_tokens"]
             and meta["runtime_prompt_ids_match"] and not meta["input_truncated"], "invalid r2 prompt identity")
        state = row["state"]
        ids_to_pid = {int(k):v for k,v in state["pid_by_lid"].items()}
        candidates = state["new_ids"] if row["mode"]=="append" else state["kept_ids"]+state["new_ids"]
        item = {k:row[k] for k in ("task","mode","budget_tier","step")}
        item.update(prompt_token_ids=ids, prompt_tokens=len(ids), prompt_sha256=meta["prompt_sha256"],
                    candidate_pids=[ids_to_pid[lid] for lid in candidates], questions=state["questions"])
        groups[(row["mode"],row["budget_tier"])].append(item)
    need(set(groups)==set(CELLS),"latency source does not cover six cells")
    rng = random.Random(20261129)
    chosen = [row for cell in CELLS for row in rng.sample(groups[cell],50)]
    write_new(SMALL/"latency_input.json", {"status":"passed", "source": {"path":str(source.steps),"sha256":source.steps_sha256},
              "r2_run": {"path":str(cfg_path),"sha256":signatures[str(cfg_path.relative_to(PROJECT))]},
              "sample_seed":20261129,"selection_order":[list(c) for c in CELLS],
              "states":dump_rows(output,chosen),"n_per_cell":50,
              "warmup_rule":"first 20 selected states repeated as warmups before timing all 300"})
    print("Latency state spool: 300 states, 50 per cell",flush=True)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("action", choices=("split","tokenize","bm25","latency-input"))
    opts = args.parse_args()
    need(Path.cwd()==PROJECT and os.environ.get("CUDA_VISIBLE_DEVICES")=="", "Task17 CPU only")
    {"split":split,"tokenize":pair_tokens,"bm25":audit_bm25,"latency-input":latency_states}[opts.action]()


if __name__ == "__main__":
    main()

