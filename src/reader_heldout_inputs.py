"""CPU-only, signed input and prompt preflight for the held-out Qwen3 reader."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from eval.cost import trajectory_cost
from src.reader import render_passages, render_user_message

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
CELLS = [(mode, tier) for mode in ("reselect", "append") for tier in ("tight", "mid", "loose")]
OLD05 = "eval/sft_run_v2/TASK05_SHA256SUMS"
MAX_LEN = 40960
MAX_NEW = 48


def need(test, reason):
    if not test:
        raise ValueError(reason)


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, digest):
    need(isinstance(digest, str) and re.fullmatch("[a-f0-9]{64}", digest),
         f"missing or invalid SHA256 for {path}")
    need(checksum(path) == digest, f"signed SHA256 mismatch: {path}")


def sums(path):
    return {name: digest for digest, name in (line.split("  ", 1)
            for line in path.read_text().splitlines())}


def read(path):
    return json.loads(path.read_text())


def checked_json(path, digest):
    verify(path, digest)
    return read(path)


def write_new(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class Source:
    name: str
    metrics: Path
    metrics_sha256: str
    steps: Path
    steps_sha256: str
    signature: str


def historical_sources(index):
    base = PROJECT / "eval/sft_run"
    v2 = PROJECT / "eval/sft_run_v2"
    verify(PROJECT / OLD05, index[OLD05])
    task05 = sums(PROJECT / OLD05)
    manifest = base / "external_artifact_manifest.json"
    old = checked_json(manifest, task05[str(manifest.relative_to(PROJECT))])
    signed05 = {r["path"]: r["sha256"] for r in old["files"]}
    sig06 = "eval/sft_run_v2/external_manifest_v2.json"
    signed06 = {r["path"]: r["sha256"] for r in checked_json(PROJECT / sig06, index[sig06])["files"]}
    entries = []
    for name in ("no_compress", "latest", "bm25", "oracle"):
        metrics = base / (name + "_results.json")
        steps = EXTERNAL / "eval_rollouts" / name / "steps.jsonl"
        entries.append(Source(name, metrics, task05[str(metrics.relative_to(PROJECT))],
                              steps, signed05[str(steps)], OLD05 + " -> " + str(manifest)))
    metrics = v2 / "checkpoint-2000_results.json"
    steps = EXTERNAL / "eval_rollouts_v2/checkpoint-2000/steps.jsonl"
    entries.append(Source("sft_v2:checkpoint-2000", metrics, index[str(metrics.relative_to(PROJECT))],
                          steps, signed06[str(steps)], sig06))
    for name, local, raw, task in (
        ("grpo_r1:checkpoint-500", "grpo_v2", "eval_rollouts_grpo_v2", "08"),
        ("h1:tau_star", "hedge_v2", "eval_rollouts_hedge_v2", "09"),
        ("sft_v3:checkpoint-3000", "sft_run_v3", "eval_rollouts_v3", "10"),
        ("grpo_r2:checkpoint-125", "grpo_r2", "eval_rollouts_grpo_r2", "11"),
    ):
        model = name.split(":")[1]
        sig = f"eval/{local}/TASK{task}_SHA256SUMS"
        verify(PROJECT / sig, index[sig])
        m = sums(PROJECT / sig)
        filename = "metrics.json" if name == "h1:tau_star" else model + "_results.json"
        metrics = f"eval/{local}/{model}/{filename}"
        steps = EXTERNAL / raw / model / "steps.jsonl"
        entries.append(Source(name, PROJECT / metrics, m[metrics], steps, m[str(steps)], sig))
    return entries


def load_task_inputs(index):
    pre = "eval/sft_run_v2/heldout_v2_preflight.json"
    signed = checked_json(PROJECT / pre, index[pre])["input_files"]
    files = {key: EXTERNAL / "data/traj_train" / (key + ".jsonl")
             for key in ("corpus_train", "heldout_tasks")}
    for path in files.values():
        verify(path, signed[str(path)]["sha256"])
    corpus = list(rows(files["corpus_train"]))
    need(all(p["pid"] == i for i, p in enumerate(corpus)), "corpus pid order changed")
    tasks = {}
    for row in rows(files["heldout_tasks"]):
        need(row["split"] == "heldout" and row["task"] not in tasks, "heldout identity")
        tasks[row["task"]] = {k: row[k] for k in ("example_ids", "questions", "steps")}
    wanted = {eid for t in tasks.values() for eid in t["example_ids"]}
    need(set(tasks) == set(range(249)) and len(wanted) == 996, "heldout task/question count")
    raw = PROJECT / "data/raw/musique/musique_ans_v1.0_train.jsonl"
    verify(raw, index[str(raw.relative_to(PROJECT))])
    answers = {}
    for row in rows(raw):
        if row["id"] not in wanted:
            continue
        need(row["id"] not in answers, "duplicate example in raw train")
        paragraphs = {p["idx"]: p for p in row["paragraphs"]}
        supports = [paragraphs[hop["paragraph_support_idx"]] for hop in row["question_decomposition"]]
        answers[row["id"]] = (row["question"], row["answer"], row["answer_aliases"], supports)
    need(answers.keys() == wanted, "raw train answer coverage")
    for task in tasks.values():
        need(len(task["questions"]) == len(task["example_ids"]) == 4, "question count")
        metadata = []
        for qi, (eid, question) in enumerate(zip(task["example_ids"], task["questions"], strict=True)):
            original, answer, aliases, supports = answers[eid]
            need(original == question, f"heldout question differs from raw train: {eid}")
            hops = [step for step in task["steps"] if step["question_index"] == qi]
            need(len(hops) == len(supports) and len(hops) in (2, 3, 4), f"hop count: {eid}")
            for hop_index, (step, support) in enumerate(zip(hops, supports, strict=True)):
                p = corpus[step["gold"]]
                need(step["hop_index"] == hop_index and step["example_id"] == eid
                     and (p["title"], p["paragraph_text"]) == (support["title"], support["paragraph_text"]),
                     f"support mismatch at {eid}/{hop_index}")
            metadata.append({"example_id": eid, "question": question, "answer": answer,
                             "answer_aliases": aliases, "question_hops": len(supports),
                             "gold": {s["gold"] for s in hops}})
        task["metadata"] = metadata
        task["all_gold"] = set.union(*(r["gold"] for r in metadata))
    return corpus, tasks, {str(path): signed[str(path)]["sha256"] for path in files.values()} | {str(raw): index[str(raw.relative_to(PROJECT))]}


def replay(source, tasks, ntok):
    # Scores and ordered contexts must be attached to the very same signed rollout.
    metrics = checked_json(source.metrics, source.metrics_sha256)
    artifact = metrics.get("steps_artifact") or metrics["run"]["steps_file"]
    need(Path(artifact["path"]) == source.steps and artifact["sha256"] == source.steps_sha256,
         f"rollout and metric signatures disagree: {source.name}")
    verify(source.steps, source.steps_sha256)
    paths = defaultdict(list)
    for r in rows(source.steps):
        cell = (r["mode"], r["budget_tier"])
        task = r["task"]
        contexts = paths[cell, task]
        step = len(contexts)
        expected = tasks[task]["steps"][step]
        need(cell in CELLS and r["strategy"] == metrics["strategy"] and r["step"] == step
             and r["question_index"] == expected["question_index"] and r["gold"] == expected["gold"]
             and r["hop_index"] == expected["hop_index"] and r["returned"] == expected["new"]
             and r["kept_before"] == (contexts[-1] if contexts else []),
             f"trajectory alignment mismatch: {source.name}/{cell}/{task}/{step}")
        kept = r["keep_after"]
        need(len(kept) == len(set(kept)) and all(0 <= pid < len(ntok) for pid in kept)
             and sum(ntok[p] for p in kept) == r["used_after"], "retained context/token mismatch")
        contexts.append(kept)
    need(len(paths) == len(CELLS) * len(tasks), "incomplete rollout")
    finals, costs = {}, {}
    for mode, tier in CELLS:
        restored, lost, complete = [], 0, 0
        for task, data in tasks.items():
            contexts = paths[(mode, tier), task]
            need(len(contexts) == len(data["steps"]), "incomplete trajectory")
            kept = contexts[-1]
            finals[(mode, tier), task] = kept
            kept_set = set(kept)
            lost += not data["all_gold"] <= kept_set
            complete += sum(r["gold"] <= kept_set for r in data["metadata"])
            restored.append(trajectory_cost(contexts, ntok)[1])
        saved = metrics["configs"]["|".join((metrics["strategy"], mode, tier))]["metrics"]
        cached = saved["cost"]["cached"]
        mean = math.fsum(restored) / len(restored)
        need(cached["n"] == len(restored) and math.isclose(cached["mean"], mean, rel_tol=1e-12)
             and saved["evidence_loss"]["count"] == lost
             and saved["evidence_complete_questions"]["count"] == complete,
             f"metric/rollout mismatch: {source.name}/{mode}/{tier}")
        costs[f"{mode}|{tier}"] = {"n":len(restored), "cached_mean":cached["mean"],
                                   "recomputed_cached_mean":mean,
                                   "lost_trajectories":lost, "complete_questions":complete}
    return finals, costs


def dedup_contexts(finals, task_ids, cells=CELLS):
    reference = cells[0]
    differing = {cell: {task for task in task_ids if finals[cell, task] != finals[reference, task]}
                 for cell in cells}
    return {"reference_cell":"|".join(reference), "tasks":len(task_ids),
            "differing_tasks":len(set.union(*differing.values())),
            "differing_cells":sum(bool(v) for v in differing.values()),
            "by_cell":{"|".join(k):len(v) for k,v in differing.items()},
            "all_identical":not any(differing.values())}


def request_tokens(tokenizer, passages, question, closed=False):
    user = render_user_message(passages, question, no_context=closed)
    ids = tokenizer.apply_chat_template([{"role":"user", "content":user}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False)
    return ids, hashlib.sha1(user.encode("utf-8")).hexdigest()


def check_length(n, identity):
    if n + MAX_NEW > MAX_LEN:
        raise ValueError(f"prompt overflow, truncation forbidden: {identity}: {n} + {MAX_NEW} > {MAX_LEN}")

def task03_smoke(index, tokenizer):
    """Reconstruct the 40 documented prompts and attach raw outputs from Task 03's full run."""
    names = ("eval/reader/smoke_samples.md", "data/reader/predictions.jsonl",
             "data/feasibility/final_contexts.jsonl", "data/feasibility/corpus_dev_ntok.jsonl",
             "data/raw/musique/musique_ans_v1.0_dev.jsonl")
    for name in names:
        verify(PROJECT / name, index[name])
    headers = re.findall(r"^## (.+?) / task (\d+) / question (\d+)$",
                         (PROJECT / names[0]).read_text(), re.MULTILINE)
    keys = [(cfg, int(task), int(qi)) for cfg, task, qi in headers]
    expected = {(cfg, task, i) for cfg in ("oracle|reselect|mid", "no_compress")
                for task in range(5) for i in range(4)}
    need(len(keys) == 40 and set(keys) == expected, "Task 03 smoke identity list")
    records = {}
    for r in rows(PROJECT / names[1]):
        key = r["config"], r["task"], r["question_index"]
        if key in expected:
            need(key not in records, "duplicate smoke reference")
            records[key] = r
    by_config = {(cfg,t) for cfg,t,_ in keys}
    contexts = {(r["config"], r["task"]):r for r in rows(PROJECT / names[2])
                if (r["config"],r["task"]) in by_config}
    corpus, seen = [], set()
    for example in rows(PROJECT / names[4]):
        for p in example["paragraphs"]:
            key = p["title"], p["paragraph_text"]
            if key not in seen:
                seen.add(key)
                corpus.append({"title":key[0], "paragraph_text":key[1]})
    subset = {pid for r in contexts.values() for pid in r["keep_after"]}
    for i, (signed, original) in enumerate(zip(rows(PROJECT / names[3]), corpus, strict=True)):
        if i in subset:
            one = render_passages(corpus, [i])
            need(signed["pid"] == i and signed["title"] == original["title"]
                 and hashlib.sha1(one.encode()).hexdigest() == signed["sha1"], "smoke paragraph mismatch")
    for cfg, task, qi in keys:
        rec = records[cfg,task,qi]
        ctx = contexts[cfg,task]
        q = ctx["questions"][qi]
        ids, user_hash = request_tokens(tokenizer, render_passages(corpus,ctx["keep_after"]), q["question"])
        check_length(len(ids), (cfg, task, qi))
        need(user_hash == rec["prompt_sha1"] and len(ids) == rec["prompt_tokens"]
             and q["id"] == rec["example_id"], "Task 03 reference prompt changed")
        yield {"config":cfg, "task":task, "question_index":qi, "question":q["question"],
               "example_id":q["id"], "answer":q["answer"], "answer_aliases":q["answer_aliases"],
               "question_hops":rec["question_hops"], "evidence_complete":rec["evidence_complete"],
               "token_ids":ids, "prompt_tokens":len(ids), "prompt_sha1":user_hash,
               "expected_raw_output":rec["raw_output"]}


def make_requests(sources, finals, tasks, corpus, tokenizer, singleton):
    for task in sorted(tasks):
        # Most methods keep many identical final PID sequences for the same questions.
        # Reuse only byte-identical contexts in this task; every output keeps its own signature.
        cache = {}
        configs = []
        for src in sources:
            if src.name == "no_compress" and singleton:
                configs.append((src.name, None, None, finals[src.name][CELLS[0],task]))
            else:
                configs.extend((src.name, m, t, finals[src.name][(m,t),task]) for m,t in CELLS)
        if any(s.name == "no_compress" for s in sources):
            configs.append(("no_context",None,None,[]))
        data = tasks[task]
        for strategy, mode, tier, pids in configs:
            cfg = "|".join((strategy, mode, tier)) if mode else strategy
            key = (strategy == "no_context", tuple(pids))
            if key not in cache:
                passages = render_passages(corpus, pids)
                cache[key] = [request_tokens(tokenizer, passages, q["question"], key[0])
                              for q in data["metadata"]]
            kept = set(pids)
            for qi,(metadata,(ids,prompt_sha1)) in enumerate(zip(data["metadata"],cache[key],strict=True)):
                check_length(len(ids), (cfg,task,qi))
                yield {"strategy":strategy, "config":cfg, "mode":mode, "tier":tier,
                       "task":task, "question_index":qi,
                       **{k:v for k,v in metadata.items() if k != "gold"},
                       "evidence_complete":metadata["gold"] <= kept,
                       "trajectory_lost":not data["all_gold"] <= kept, "token_ids":ids,
                       "prompt_tokens":len(ids), "prompt_sha1":prompt_sha1}


def export_requests(path, stream):
    digest = hashlib.sha256()
    maxima, counts = {}, Counter()
    with path.open("xb") as fh:
        for r in stream:
            data = (json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(",",":"))+"\n").encode("utf-8")
            fh.write(data)
            digest.update(data)
            cfg = r["config"]
            counts[cfg] += 1
            maxima[cfg] = max(maxima.get(cfg,0), r["prompt_tokens"])
    return {"path":str(path), "sha256":digest.hexdigest(), "n":sum(counts.values()),
            "bytes":path.stat().st_size, "max_prompt_tokens":max(maxima.values()),
            "max_prompt_tokens_by_config":maxima, "n_by_config":dict(counts)}


def prepare(out_dir, extension=None):
    from transformers import AutoTokenizer
    out_dir = out_dir.resolve()
    need(out_dir.is_relative_to(EXTERNAL / "reader_heldout") and out_dir != EXTERNAL / "reader_heldout",
         "preflight output must be a fresh subdirectory under external reader_heldout")
    out_dir.mkdir(parents=True, exist_ok=False)
    report = {"status":"preparing","max_model_len":MAX_LEN,"max_new_tokens":MAX_NEW,"input_truncations":0}
    try:
        index = sums(PROJECT / "SHA256SUMS")
        corpus, tasks, inputs = load_task_inputs(index)
        report["inputs_sha256"] = inputs
        ntok = [p["ntok"] for p in corpus]
        sources = [extension] if extension else historical_sources(index)
        finals, costs, proofs = {}, {}, {}
        for source in sources:
            finals[source.name], costs[source.name] = replay(source,tasks,ntok)
            proofs[source.name] = {"steps":str(source.steps),"steps_sha256":source.steps_sha256,
                                   "metrics":str(source.metrics),"metrics_sha256":source.metrics_sha256,
                                   "signature":source.signature,"n_trajectories":len(finals[source.name])}
            print(f"Verified signed rollout, task gold, and six cached costs: {source.name}",flush=True)
        report.update(sources=proofs, costs=costs, n_tasks=len(tasks), n_questions=4*len(tasks))
        tokenizer = AutoTokenizer.from_pretrained(str(EXTERNAL / "models/Qwen3-8B"), local_files_only=True,
                                                   trust_remote_code=False)
        singleton = False
        if not extension:
            report["no_compress"] = dedup_contexts(finals["no_compress"], tasks.keys())
            singleton = report["no_compress"]["all_identical"]
            report["smoke"] = export_requests(out_dir / "smoke.jsonl", task03_smoke(index,tokenizer))
            need(report["smoke"]["n"] == 40,"smoke request count")
        report["requests"] = export_requests(out_dir / "requests.jsonl",
                make_requests(sources,finals,tasks,corpus,tokenizer,singleton))
        n_configs = 6 if extension else 8 * 6 + (1 if singleton else 6) + 1
        need(len(report["requests"]["n_by_config"]) == n_configs
             and set(report["requests"]["n_by_config"].values()) == {996}, "reader request count")
        report["status"] = "passed"
        write_new(out_dir / "preflight.json", report)
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        write_new(out_dir / "failed.json", report)
        raise
    print(f"Prepared {report['requests']['n']} prompts, max={report['requests']['max_prompt_tokens']}",flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=EXTERNAL / "reader_heldout/input_preflight")
    parser.add_argument("--strategy")
    parser.add_argument("--rollout",type=Path)
    parser.add_argument("--metrics",type=Path)
    parser.add_argument("--steps-sha256")
    parser.add_argument("--metrics-sha256")
    args = parser.parse_args()
    extension = None
    values = (args.strategy, args.rollout, args.metrics, args.steps_sha256, args.metrics_sha256)
    if any(values):
        need(all(values),"single-policy mode requires strategy, rollout, metrics and both receipt hashes")
        checkpoint = re.fullmatch(r"grpo_r3:(checkpoint-\d+)",args.strategy)
        need(checkpoint is not None,"only a named R3 checkpoint may extend the run")
        name = checkpoint.group(1)
        raw = EXTERNAL / "eval_rollouts_grpo_r3" / name
        steps = args.rollout.resolve()
        if steps == raw:
            steps = steps / "steps.jsonl"
        need(steps == raw / "steps.jsonl" and args.metrics.resolve().parent in (raw, PROJECT / "eval/grpo_r3" / name),
             "R3 input path escaped selected checkpoint")
        need(args.out_dir.resolve().is_relative_to(EXTERNAL / "reader_heldout" / ("grpo_r3_" + name)),
             "new checkpoint needs a dedicated reader output")
        extension = Source(args.strategy, args.metrics.resolve(), args.metrics_sha256, steps,
                           args.steps_sha256, "Task 13 F stage receipt")
    prepare(args.out_dir,extension)


if __name__ == "__main__":
    main()
