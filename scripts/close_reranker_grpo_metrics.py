"""CPU closeout shared by the reranker and GRPO round-four reports."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from math import fsum, isclose
from pathlib import Path

from scripts import grpo_r4_common as r4
from scripts import reader_heldout_report as qa
from src import reader_heldout_inputs as inputs
from src.reranker_v1 import R2, V3, STRATEGIES, need, read, record, verify

PROJECT, EXT = inputs.PROJECT, inputs.EXTERNAL
T17, T18 = PROJECT / "eval/reranker_v1", r4.SMALL
W17, W18 = EXT / "reranker_v1", r4.RUN
HELD17, HELD18 = W17 / "heldout_reader", r4.READER
T16 = PROJECT / "eval/reader_heldout_v2"
C16 = T16 / "reader_heldout_v2_results.json"
R3 = "grpo_r3:checkpoint-250"
R4_500 = "grpo_r4:checkpoint-500"
CST = timezone(timedelta(hours=8))
CELLS = qa.CELLS
LABEL = {
    R2: "r2 ckpt-125", V3: "v3 ckpt-3000", R3: "r3 ckpt-250",
    "h1:tau_star": "H1",
    **{f"grpo_r4:checkpoint-{i}": f"r4-{i}" for i in r4.CHECKPOINT_ORDER},
}
GOLD = ("seen", "unseen")


def beijing(value):
    return datetime.fromisoformat(value).astimezone(CST).strftime("%Y-%m-%d %H:%M:%S")


def checked(ref):
    path = Path(ref["path"])
    need(path.is_absolute() and path.resolve() == path and not {".git", "test_dev_v1"}.intersection(path.parts)
         and (path.is_relative_to(EXT) or path.is_relative_to(PROJECT)), f"protected or redirected receipt: {path}")
    need(ref.get("size", ref.get("bytes", path.stat().st_size)) == path.stat().st_size,
         f"receipt size differs: {path}")
    verify(path, ref["sha256"])
    return path


def finish_receipts():
    st17 = read(W17 / "status.json")
    st18 = read(W18 / "pipeline_status.json")
    need(st17["status"] == "complete" and st17["stage"] == "all" and not st17["automatic_retry"]
         and st18["status"] == "evaluation_complete" and not st18["automatic_retry"],
         "Task 17 or 18 did not complete")
    need(not Path(f"/proc/{st17['pid']}").exists() and not Path(f"/proc/{st18['pipeline_pid']}").exists(),
         "GPU pipeline supervisor still present")
    run17_path = checked(st17["config"])
    run17 = read(run17_path)
    r17_keys = ("S1", "prepare_val", "reader_val", "tau", "prepare_heldout", "reader_heldout",
                "latency_eager", "latency_graph", "latency_reranker", "compare")
    need(tuple(run17["stages"]) == r17_keys and run17["status"] == "complete" and run17["GPU"] == 1,
         "Task 17 ordered stage receipt set differs")
    for key, receipt in run17["stages"].items():
        name = {"tau": "select-tau", "compare": "compare"}.get(key, key.replace("_", "-"))
        need(receipt == read(W17 / "stages" / name / "receipt_complete.json")
             and receipt["stage"] == "complete" and receipt["exit_code"] == 0, f"incomplete stage {name}")
        for ref in receipt["outputs"].values():
            checked(ref)
        if receipt["gpu"]:
            need(read(W17 / "stages" / name / "receipt_gate.json")["gpu"]["index"] == 1,
                 "Task 17 wrong GPU in gate")
    resume = read(checked(run17["resume"]))
    then = read(checked(resume["previous_resume"]))
    need(resume["first_failure"] == then["failure"] and resume["reused"]["tau"] == .5,
         "Task 17 resume provenance differs")
    for failure in (then["failure"], resume["failure"]):
        p = checked(failure["log"])
        need(failure["exception"] in p.read_text(), "failure message differs from archived log")
    for mode, ref in run17["compressor_environments"].items():
        env = read(checked(ref))
        need(env["status"] == "passed" and env["stage"] == f"latency-{mode}"
             and env["cuda_visible_devices"] == "" and Path(env["TRITON_CACHE_DIR"]).is_relative_to(W17)
             and env["ipc"]["uri_bytes"] < 100, "missing latency compiler check")
        checked(env["environment_script"])

    expected = ["smoke", "smoke_check", "E", "E_check",
                *(f"F_checkpoint-{n}" for n in r4.CHECKPOINT_ORDER),
                "F_select", "G_inputs", "G", "G_check"]
    need([s["stage"] for s in st18["stages"]] == expected and not st18["lock"]["held"], "Task 18 stage set differs")
    for s in st18["stages"]:
        need(s["status"] == "complete" and s["exit_code"] == 0, f"incomplete Task 18 stage {s['stage']}")
        checked(s["receipt"])
        if s["uses_gpu"]:
            need(s["gpu_before"]["index"] == 1 and s["gpu_before"]["memory_used_mib"] < 2048,
                 "Task 18 wrong GPU admission")
    return {
        "task17": {"beijing_end": beijing(st17["utc"]), "n_stages": len(r17_keys),
                   "status": record(W17 / "status.json"), "run_config": st17["config"],
                   "failure_records": [then["failure"], resume["failure"]]},
        "task18": {"beijing_end": beijing(st18["ended_utc"]), "n_stages": len(expected),
                   "status": record(W18 / "pipeline_status.json")},
    }


def read_scores(root, cfg_path, names, n):
    cfg = read(cfg_path)
    prep = read(root / "input_preflight.json")
    inference = read(root / "receipt_inference.json")
    smoke_receipt = read(root / "receipt_smoke.json")
    need(cfg["engine_shutdown"] and cfg["effective_disable_cascade_attn"] is True
         and cfg["status"] == "complete" and cfg["tasks"] == prep["n_tasks"] == 249
         and n == prep["requests"]["n"] == cfg["predictions"] == inference["n"]
         and cfg["predictions_sha256"] == inference["predictions_sha256"]
         and inputs.checksum(cfg_path) == inference["run_config_sha256"]
         and tuple(prep["strategies"]) == tuple(names), f"incomplete reader receipts: {root}")
    expected_n = smoke_receipt["n"]
    smoke_p = root / "smoke_predictions.jsonl"
    verify(smoke_p, smoke_receipt["predictions_sha256"])
    count = sum(1 for _ in inputs.rows(smoke_p))
    need(expected_n == count == cfg["smoke"]["n"] == 40
         and cfg["smoke"]["raw_output_mismatches"] == smoke_receipt["raw_output_mismatches"],
         f"reader smoke differs: {root}")

    hashes = {}
    request_p, pred_p = root / "requests.jsonl", root / "predictions.jsonl"
    requests = qa.index_requests(qa.input_rows(request_p, hashes))
    scores = qa.collect(qa.input_rows(pred_p, hashes), requests, names)
    qa.audit_matrix(scores, None)
    need(hashes[str(request_p)] == prep["requests"]["sha256"]
         and hashes[str(pred_p)] == cfg["predictions_sha256"], f"reader bytes differ: {root}")
    return scores, {str(p): hashes[str(p)] for p in (request_p, pred_p)}


def paired(scores, left, right, cell_gate=None):
    left_cell, right_cell = scores[left], scores[right]
    all_diffs, all_em = [], []
    counts = Counter(left_higher=0, equal=0, right_higher=0)
    matrix = [[0, 0], [0, 0]]
    cells = {}
    for cell in CELLS:
        a, b = left_cell[cell], right_cell[cell]
        need(a.keys() == b.keys(), f"unpaired tasks: {left}, {right}, {cell}")
        d_f1, d_em = [], []
        for key, x in a.items():
            y = b[key]
            need((x.example, x.hops) == (y.example, y.hops), f"unpaired question: {cell}, {key}")
            if cell_gate is not None and key not in cell_gate:
                continue
            d_f1.append(x.f1 - y.f1)
            d_em.append(x.em - y.em)
            counts["left_higher" if x.f1 > y.f1 else "right_higher" if x.f1 < y.f1 else "equal"] += 1
            matrix[1 - int(x.em)][1 - int(y.em)] += 1
        n = len(d_f1)
        need(n > 0, "empty paired subgroup")
        cells[cell] = {"n": n, "f1_difference_pp": 100 * fsum(d_f1) / n, "em_difference_pp": 100 * fsum(d_em) / n}
        all_diffs.extend(d_f1)
        all_em.extend(d_em)
    n = len(all_diffs)
    return {"left": left, "right": right, "n": n, "f1_difference_pp": 100 * fsum(all_diffs) / n,
            "em_difference_pp": 100 * fsum(all_em) / n, "f1_counts": dict(counts),
            "em_matrix_correct_then_incorrect": matrix, "by_cell": cells}


def compare_receipt(pair, expected):
    for name in ("left", "right", "n", "f1_counts", "em_matrix_correct_then_incorrect"):
        need(pair[name] == expected[name], f"paired receipt field differs: {name}")
    for name in ("f1_difference_pp", "em_difference_pp"):
        need(isclose(pair[name], expected[name], abs_tol=1e-11, rel_tol=0), f"paired delta differs: {name}")
    for cell, stats in pair["by_cell"].items():
        e = expected.get("by_cell", expected.get("cells"))[cell]
        for name, value in stats.items():
            if name in e:
                need(isclose(value, e[name], abs_tol=1e-11, rel_tol=0), f"paired cell differs: {cell} / {name}")


def trace_finals(ref, tasks):
    """Read executed actions and align every row with the frozen held-out trajectory."""
    path = Path(ref["path"])
    provenance = {}
    progress, kept_at_end = {}, {}
    for row in qa.input_rows(path, provenance):
        cell = f"{row['mode']}|{row['budget_tier']}"
        task = row["task"]
        identity = (cell, task)
        steps = tasks[task]["steps"]
        i, kept_before = progress.get(identity, (0, []))
        expected = steps[i]
        need(cell in CELLS and row["step"] == i and row["kept_before"] == kept_before
             and all(row[k] == expected[k] for k in ("gold", "hop_index", "question_index")),
             f"rollout alignment error: {path} / {identity} / step {i}")
        kept = row["keep_after"]
        progress[identity] = (i + 1, kept)
        if i + 1 == len(steps):
            kept_at_end[identity] = set(kept)
    need(provenance[str(path)] == ref["sha256"] and set(kept_at_end) == {(c, t) for c in CELLS for t in tasks},
         f"trace hash or task coverage differs: {path}")
    return kept_at_end


def posthoc(scores17, scores18):
    """Stratify by labels in only the reranker's optimizer training split."""
    path = W17 / "data/split.json"
    split = read(path)
    need(path.read_bytes() == (T17 / "data_preflight.json").read_bytes(), "reranker split receipt differs")
    ids = set(split["train_tasks"])
    train_file = Path(split["input_files"]["train"]["path"])
    test_file = Path(split["input_files"]["heldout"]["path"])
    hashes = {}
    observed_ids, train_gold = set(), set()
    for row in qa.input_rows(train_file, hashes):
        if row["task"] in ids:
            need(row["task"] not in observed_ids and row["split"] == "train", "invalid train task in split")
            observed_ids.add(row["task"])
            train_gold.update(step["gold"] for step in row["steps"])
    tasks = {row["task"]: row for row in qa.input_rows(test_file, hashes)}
    need(len(ids) == 4535 and observed_ids == ids and not ids.intersection(split["train_val_tasks"])
         and len(tasks) == 249 and all(row["split"] == "heldout" for row in tasks.values()),
         "training and held-out partitions differ")
    for split_name, file in (("train", train_file), ("heldout", test_file)):
        need(hashes[str(file)] == split["input_files"][split_name]["sha256"], "raw task bytes differ from split receipt")

    q_gold = defaultdict(set)
    evidence = {}
    for task, row in tasks.items():
        pids = {step["gold"] for step in row["steps"]}
        need(len(pids) == len(row["steps"]), "gold pid occurrences are not unique within a task")
        evidence[task] = {group: pids & train_gold if group == "seen" else pids - train_gold for group in GOLD}
        for step in row["steps"]:
            q_gold[task, step["question_index"]].add(step["gold"])
    need(len(q_gold) == 996 and set(q_gold) == {(t, i) for t in tasks for i in range(4)}, "held-out question coverage")
    q_sets = {"all_seen": {key for key, gold in q_gold.items() if gold <= train_gold}}
    q_sets["any_unseen"] = q_gold.keys() - q_sets["all_seen"]

    pre = read(HELD17 / "input_preflight.json")
    refs = {name: pre["rollouts"][name]["steps"] for name in ("RR-τ", "RR-fill", R2, V3)}
    refs[R4_500] = read(T18 / "checkpoint-500/checkpoint-500_results.json")["steps_artifact"]
    retention = {}
    for name, ref in refs.items():
        kept = trace_finals(ref, tasks)
        bucket = {group: Counter(count=0, n=0) for group in GOLD}
        for (cell, task), pids in kept.items():
            for group, target in evidence[task].items():
                bucket[group].update(count=len(target & pids), n=len(target))
            source = scores18 if name == R4_500 else scores17
            for q in range(4):
                score = source[name][cell][task, q]
                need(score.example == tasks[task]["example_ids"][q]
                     and score.complete == (q_gold[task, q] <= pids), "reader evidence does not match the rollout")
        retention[name] = {group: {**b, "rate": b["count"] / b["n"]} for group, b in bucket.items()}

    merged = {**scores17, R4_500: scores18[R4_500]}
    strata, comparisons = {}, {}
    for group, keys in q_sets.items():
        strata[group] = {"unique_questions": len(keys), "question_cells": len(keys) * len(CELLS)}
        comparisons[group] = {name: paired(merged, name, "RR-τ", keys) for name in (R2, V3, R4_500)}
    for name in (R2, V3, R4_500):
        complete = paired(merged, name, "RR-τ")
        for field in ("n", "f1_difference_pp", "em_difference_pp"):
            subtotal = sum(comparisons[g][name][field] *
                           (1 if field == "n" else comparisons[g][name]["n"] / complete["n"]) for g in strata)
            need(isclose(subtotal, complete[field], abs_tol=1e-11, rel_tol=0), f"post hoc subgroup partition differs: {name}")
        for key, value in complete["f1_counts"].items():
            need(sum(comparisons[g][name]["f1_counts"][key] for g in strata) == value, "subgroup counts differ")

    return {
        "preregistered": False,
        "gold_training_pid_count": len(train_gold), "n_train_tasks": len(ids),
        "train_split": record(path), "task_source_sha256": hashes, "rollout_sources": refs,
        "n_heldout_tasks": len(tasks), "n_heldout_questions": len(q_gold),
        "gold_stratum_unit": "A gold pid occurrence in one held-out task and mode-budget trajectory; no within-task duplicates.",
        "reader_batch": {"RR-τ": "task17", R2: "task17", V3: "task17", R4_500: "task18"},
        "question_strata": strata, "kept_gold": retention, "paired": comparisons,
    }


def pct(value, digits=2):
    return f"{value * 100:.{digits}f}%"


def rate(m, digits=2):
    return f"{pct(m['rate'], digits)} ({m['count']:,}/{m['n']:,})"


def mean(m, digits=3):
    return f"{m['mean']:,.{digits}f} (n={m['n']:,})"


def table(header, rows):
    def line(items):
        return "| " + " | ".join(str(x).replace("|", r"\|") for x in items) + " |"
    return "\n".join([line(header), line(["---"] * len(header)), *(line(items) for items in rows)])


def paired_md(pairs):
    rows = []
    cell_rows = []
    for p in pairs:
        label = f"{LABEL.get(p['left'], p['left'])} − {LABEL.get(p['right'], p['right'])}"
        k = p["f1_counts"]
        matrix = p["em_matrix_correct_then_incorrect"]
        rows.append((label, f"{p['n']:,}", f"{p['f1_difference_pp']:+.3f}", f"{p['em_difference_pp']:+.3f}",
                     f"{k['left_higher']:,} / {k['equal']:,} / {k['right_higher']:,}",
                     f"{matrix[0][0]:,} / {matrix[0][1]:,} / {matrix[1][0]:,} / {matrix[1][1]:,}"))
        cell_rows.extend((label, cell, f"{s['n']:,}", f"{s['f1_difference_pp']:+.3f}",
                          f"{s['em_difference_pp']:+.3f}") for cell, s in p["by_cell"].items())
    return [
        table(("比较", "配对题次 n", "ΔF1（百分点）", "ΔEM（百分点）", "F1 高 / 平 / 低",
               "EM 同对 / 仅左对 / 仅右对 / 同错"), rows),
        table(("比较", "格", "配对题次 n", "ΔF1（百分点）", "ΔEM（百分点）"), cell_rows),
    ]



def audit_readers(path):
    """Reproduce the paired reader metrics from the archived requests and predictions."""
    import os

    from scripts import grpo_r4_reader as reader
    from src.reader_heldout_inputs import write_new

    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and Path.cwd() == PROJECT, "run from the project on CPU")
    support = finish_receipts()
    s17, h17 = read_scores(HELD17, T17 / "heldout_reader_config.json", STRATEGIES, 29880)
    s18, h18 = read_scores(HELD18, T18 / "reader_run_config.json", reader.NAMES, 35856)
    compare17 = read(T17 / "heldout_comparison.json")
    compare18 = read(T18 / "reader_f1_results.json")
    verified = []
    for scores, packets in ((s17, compare17["primary"]), (s18, compare18["comparisons"])):
        pairs = []
        for row in packets:
            item = paired(scores, row["left"], row["right"])
            compare_receipt(item, row)
            pairs.append(item)
        verified.append(pairs)
    decomp = reader.decomposition(compare18["selected"], s18, read(HELD18 / "final_context_counts.json"))
    need(decomp == compare18["decomposition_r4_minus_r2"], "G decomposition differs")
    write_new(path, {"stage_completion": support, "reader17_input_hashes": h17, "reader18_input_hashes": h18,
                     "pairs17": verified[0], "pairs18": verified[1], "decomposition18": decomp,
                     "posthoc": posthoc(s17, s18)})


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    audit_readers(cli.parse_args().output)
