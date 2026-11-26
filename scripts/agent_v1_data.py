"""Freeze Task 20's sampled questions, evidence, corpora, and per-dataset budgets."""
import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.build_traj import build_corpus, sample_tasks
from src.reader_heldout_inputs import EXTERNAL, need, read, rows, sums, verify, write_new
from src.reranker_v1 import distribution, record, require_new
from src.model_inference import compact

WORK = EXTERNAL / "agent_v1"
SMALL = PROJECT / "eval/agent_v1"
DATA = WORK / "data"
RECEIPT = SMALL / "data_preflight.json"
TOKENIZER = PROJECT / "data/raw/qwen3_tokenizer/tokenizer.json"
TEST19 = PROJECT / "eval/test_v1"
LOCKED = EXTERNAL / "data/test_dev_v1"
MUSIQUE_RAW = PROJECT / "data/raw/musique/musique_ans_v1.0_dev.jsonl"
HOT_POT = PROJECT / "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"
WIKI = PROJECT / "data/raw/2wikimultihopqa/dev.json"
SEEDS = {"musique": 20261125, "hotpotqa": 20261126, "2wiki": 20261127}
N_MAIN = 100
N_SMOKE = 10
K = 3


def budget_for(groups, corpus):
    counts = [sum(corpus[p]["ntok"] for p in {pid for q in task for pid in q["gold"]})
              for task in groups]
    stats = distribution(counts)
    return round(1.5 * stats["p90"]), stats


def select(main_pool, smoke_pool, dataset):
    seed = SEEDS[dataset]
    main = sample_tasks(sorted(main_pool, key=lambda q: q["id"]), N_MAIN, seed, K)
    used = {q["id"] for group in main for q in group}
    if smoke_pool is None:
        smoke_pool = [q for q in main_pool if q["id"] not in used]
    smoke = sample_tasks(sorted(smoke_pool, key=lambda q: q["id"]), N_SMOKE, seed + 100, K)
    need(not used.intersection(q["id"] for group in smoke for q in group), "Smoke and main questions overlap")
    return main, smoke


def question(qid, text, answer, aliases, gold_hop, kind=None):
    need(all(isinstance(x, str) for x in (qid, text, answer)), "A question field has an unexpected type")
    if isinstance(aliases, str):
        aliases = [aliases]
    aliases = [] if aliases is None else aliases
    need(isinstance(aliases, list) and all(isinstance(x, str) for x in aliases), "Unexpected answer aliases")
    need(gold_hop and all(type(x) is int and x >= 0 for x in gold_hop), "Missing or invalid gold evidence")
    return {"id": qid, "question": text, "answer": answer, "answer_aliases": aliases,
            "gold": list(dict.fromkeys(gold_hop)), "gold_hop": gold_hop if kind is None else [], "type": kind}


def signed(path, index, key=None):
    verify(path, index[key or str(path.relative_to(PROJECT))])
    return record(path)


def checked_record(entry):
    path = Path(entry["path"])
    need(path.resolve() == path and path.stat().st_size == entry["size"], f"Changed data size or path: {path}")
    verify(path, entry["sha256"])
    return path


def locked_question_index(path):
    locked = {}
    for task in rows(path):
        for index, (meta, gold) in enumerate(zip(task["questions"], task["question_gold"], strict=True)):
            ordered_hops = sorted((r["hop_index"], r["gold"]) for r in task["steps"] if r["question_index"] == index)
            need([h for h, _ in ordered_hops] == list(range(len(ordered_hops))) and
                 {p for _, p in ordered_hops} == set(gold) and meta["id"] == task["example_ids"][index],
                 f"Locked hop evidence differs from question_gold: {meta['id']}")
            item = question(meta["id"], meta["question"], meta["answer"], meta["answer_aliases"],
                            [p for _, p in ordered_hops])
            need(item["id"] not in locked, "Duplicate locked question")
            locked[item["id"]] = item
    need(len(locked) == 1804, "Expected exactly 1,804 locked questions")
    return locked


def make_musique(index):
    proof = read(TEST19 / "data_preflight.json")
    lock = read(TEST19 / "lock_preflight.json")
    need(proof["status"] == "passed" and lock["status"] == "passed", "Task 19 corpus and lock verification required")
    proof_inputs = {"dev": signed(MUSIQUE_RAW, index), "lock_preflight": record(TEST19 / "lock_preflight.json"),
                    "corpus_preflight": record(TEST19 / "data_preflight.json")}
    for name in ("test_tasks.jsonl", "excluded_ids.json"):
        path = LOCKED / name
        proof_inputs[name] = record(checked_record({"path": str(path), **lock["files"][str(path)]}))
    corpus = list(rows(checked_record(proof["corp"])))
    need(len(corpus) == 21100 and all(i == x["pid"] for i, x in enumerate(corpus)),
         "Task 19 corpus inventory differs")
    for p in proof["bm25_index"]:
        checked_record(p)
    locked = locked_question_index(LOCKED / "test_tasks.jsonl")
    excluded = read(LOCKED / "excluded_ids.json")
    need(isinstance(excluded, list) and len(set(excluded)) == len(excluded) and
         not set(excluded).intersection(locked), "Invalid or overlapping MuSiQue smoke source")
    excluded_set = set(excluded)
    keys = {(p["title"], p["paragraph_text"]): p["pid"] for p in corpus}
    raw_smoke = []
    for row in rows(MUSIQUE_RAW):
        if row["id"] in excluded_set:
            s = {p["idx"]: keys[p["title"], p["paragraph_text"]] for p in row["paragraphs"]}
            hops = [s[h["paragraph_support_idx"]] for h in row["question_decomposition"]]
            raw_smoke.append(question(row["id"], row["question"], row["answer"], row["answer_aliases"], hops))
    need({q["id"] for q in raw_smoke} == excluded_set, "MuSiQue excluded_ids not covered by raw dev")
    groups = select(list(locked.values()), raw_smoke, "musique")
    need(locked.keys().isdisjoint(q["id"] for group in groups[1] for q in group),
         "Smoke questions intersect locked MuSiQue set")
    return corpus, (proof["corp"], proof["bm25_index"]), groups, proof_inputs


def context_examples(raw, joiner):
    for ex in raw:
        yield {"paragraphs": [{"title": title, "paragraph_text": joiner.join(sents)}
                              for title, sents in ex["context"]]}


def make_other(dataset, path, joiner, tokenizer, index):
    source = signed(path, index)
    population = read(path)
    need(isinstance(population, list) and population, f"Empty dataset: {path}")
    if dataset == "hotpotqa":
        need(len(population) == 7405, "Unexpected HotpotQA dev population")
    corpus, bm25, keys = build_corpus(context_examples(population, joiner),
                                      lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids))
    questions = []
    for ex in population:
        title_to_pid = {}
        for title, sents in ex["context"]:
            title_to_pid.setdefault(title, set()).add(keys[title, joiner.join(sents)])
        titles = list(dict.fromkeys(title for title, _ in ex["supporting_facts"]))
        need(all(len(title_to_pid.get(title, ())) == 1 for title in titles),
             f"An evidence title does not identify exactly one local paragraph: {ex['_id']}")
        gold = [next(iter(title_to_pid[title])) for title in titles]
        aliases = None if dataset == "hotpotqa" else ex.get("answer_aliases", ex.get("aliases"))
        questions.append(question(ex["_id"], ex["question"], ex["answer"], aliases, gold, ex["type"]))

    directory = DATA / dataset
    corpus_path, index_dir = directory / "corpus.jsonl", directory / "bm25"
    require_new(directory)
    directory.mkdir()
    with corpus_path.open("x", encoding="utf-8") as f:
        for pid, passage in enumerate(corpus):
            f.write(compact({**passage, "pid": pid}) + "\n")
    index_dir.mkdir()
    bm25.save(str(index_dir))
    return (corpus, (record(corpus_path), [record(path) for path in sorted(index_dir.iterdir())]),
            select(questions, None, dataset), {"raw_dev": source, "population": len(questions)})


def save_dataset(dataset, corpus, references, groups, source):
    directory = DATA / dataset
    directory.mkdir(exist_ok=True)
    main, smoke = groups
    budget, ntok = budget_for(main, corpus)
    need(budget > 0, f"Invalid Task 20 budget for {dataset}")
    files = {}
    for split, selected in (("main", main), ("smoke", smoke)):
        path = directory / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as f:
            for i, questions in enumerate(selected):
                f.write(compact({"task": i, "dataset": dataset, "budget": budget, "questions": questions}) + "\n")
        flat = [q for task in selected for q in task]
        yn = sum(q["answer"].strip().casefold() in ("yes", "no") for q in flat)
        files[split] = {"file": record(path), "n_tasks": len(selected), "n_questions": len(flat),
                        "question_ids": [q["id"] for q in flat],
                        "question_type": dict(Counter(q["type"] for q in flat if q["type"])),
                        "yes_no": {"count": yn, "n": len(flat), "rate": yn / len(flat)}}
    return {"n_paragraphs": len(corpus), "corpus": references[0], "bm25_index": references[1],
            "source": source, "main_seed": SEEDS[dataset], "smoke_seed": SEEDS[dataset] + 100,
            "budget": budget, "gold_tokens": ntok, **files}


def prepare():
    import os
    from tokenizers import Tokenizer

    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Data preparation requires CUDA hidden")
    require_new(RECEIPT)
    need(DATA.resolve() == DATA and (not DATA.exists() or not any(DATA.iterdir())), "Task 20 data output exists")
    index = sums(PROJECT / "SHA256SUMS")
    tok = signed(TOKENIZER, index)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    DATA.mkdir(exist_ok=True)
    mus = save_dataset("musique", *make_musique(index))
    hot = save_dataset("hotpotqa", *make_other("hotpotqa", HOT_POT, "", tokenizer, index))
    wiki = save_dataset("2wiki", *make_other("2wiki", WIKI, " ", tokenizer, index))
    write_new(RECEIPT, {
        "status": "passed", "tokenizer": tok, "task_questions": K,
        "sample_rule": "random.Random(seed).sample(sorted(question IDs), size), grouped in draw order",
        "musique_smoke_seed_reason": "Use the same seed-plus-100 convention as the other two datasets",
        "hotpotqa_join": "", "2wiki_join": " ", "dedupe": ["title", "paragraph_text"],
        "budget_rule": "round(1.5 * P90) of distinct gold paragraph token totals per main task",
        "datasets": {"musique": mus, "hotpotqa": hot, "2wiki": wiki},
    })
    print({k: {"n_corpus": v["n_paragraphs"], "budget": v["budget"], "gold_tokens": v["gold_tokens"]}
           for k, v in read(RECEIPT)["datasets"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
