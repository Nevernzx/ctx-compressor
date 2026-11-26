"""Compute Task 20 preregistered episode, evidence, cost, and paired metrics."""
from collections import Counter
from itertools import chain
from math import fsum

from src.agent_v1 import K, PAIRS, ROWS
from src.qa_metrics import score_answer
from src.reader_heldout_inputs import need
from src.reranker_v1 import distribution, histogram


def ratio(count, n):
    return {"count": count, "n": n, "rate": count / n if n else None}


def paired(rows_a, rows_b):
    need(rows_a.keys() == rows_b.keys() and rows_a, "Paired question identities differ")
    joint = [[0, 0], [0, 0]]
    f1_deltas, em_deltas = [], []
    for key, a in rows_a.items():
        b = rows_b[key]
        f1_deltas.append(a["f1"] - b["f1"])
        em_deltas.append(a["em"] - b["em"])
        joint[int(a["em"])][int(b["em"])] += 1
    n = len(f1_deltas)
    return {"n": n, "delta_f1": fsum(f1_deltas) / n, "delta_em": fsum(em_deltas) / n,
            "f1_higher": sum(d > 0 for d in f1_deltas), "f1_equal": sum(d == 0 for d in f1_deltas),
            "f1_lower": sum(d < 0 for d in f1_deltas), "em_2x2_left_rows_right_columns": joint}


def answer_metrics(episode, question, answer):
    searched = set(episode["searched_pids"])
    kept = set(episode["final_keep"])
    gold = set(question["gold"])
    found = gold.intersection(searched)
    have = gold.intersection(kept)
    lost = found - have
    score = score_answer(answer, question["answer"], question["answer_aliases"]) if answer else {"em": 0.0, "f1": 0.0}
    item = {**score, "gold_total": len(gold), "gold_retrieved": len(found), "gold_retained": len(have),
            "gold_lost": len(lost), "evidence_complete": gold <= kept,
            "retrieval_complete": gold <= searched, "lost_any_retrieved": bool(lost),
            "hop_retained": [pid in kept for pid in question["gold_hop"]]}
    if episode["row"] == "LLMLingua-2":
        item["canonical_answer_substring"] = bool(question["answer"]) and any(
            question["answer"] in passage["text"] for passage in episode["compressed_final"].values())
    return item


def analyze_row(episodes):
    candidate_scores = {}
    stats, round_stats, compressor_diagnostics = [], [], []
    for ep in episodes:
        turns = ep["turns"]
        need(turns and len(ep["questions"]) == len(ep["answers"]) == K, "Incomplete episode archive")
        for q, answer in zip(ep["questions"], ep["answers"], strict=True):
            identity = ep["dataset"], ep["task"], q["id"]
            need(identity not in candidate_scores, f"Duplicate question in row: {identity}")
            candidate_scores[identity] = answer_metrics(ep, q, answer)
        gold = {pid for q in ep["questions"] for pid in q["gold"]}
        queries = ep["queries"]
        stats.append({"searches": len(queries), "duplicates": len(queries) - len(set(queries)),
                      "force_reason": ep["force_reason"], "retained": len(ep["final_keep"]),
                      "nongold": len(set(ep["final_keep"]) - gold),
                      "uncached": sum(r["prompt_tokens"] for r in turns),
                      "cached": fsum(r["cached_input_token_cost"] for r in turns),
                      "output": sum(r["output_tokens"] for r in turns)})
        round_stats.extend(turns)
        compressor_diagnostics.extend(t["compression"] for t in turns if t.get("compression"))
    answers = list(candidate_scores.values())
    n, n_q = len(stats), len(answers)
    need(n > 0 and n_q == n * K, "Incomplete Task 20 row")
    forced = Counter(r["force_reason"] for r in stats if r["force_reason"] is not None)
    process = {"searches": {**distribution([r["searches"] for r in stats]),
                            "histogram": histogram(r["searches"] for r in stats)},
               "self_finished": ratio(n - sum(forced.values()), n),
               "forced_finish": ratio(sum(forced.values()), n), "force_reasons": dict(forced),
               "overlength": ratio(forced["overlength"], n),
               "invalid_format": ratio(sum(t["format_invalid"] for t in round_stats), len(round_stats)),
               "duplicate_queries": ratio(sum(r["duplicates"] for r in stats), sum(r["searches"] for r in stats))}
    evidence = {
        "retrieved": ratio(sum(q["gold_retrieved"] for q in answers), sum(q["gold_total"] for q in answers)),
        "retained": ratio(sum(q["gold_retained"] for q in answers), sum(q["gold_total"] for q in answers)),
        "retrieval_complete": ratio(sum(q["retrieval_complete"] for q in answers), n_q),
        "evidence_complete": ratio(sum(q["evidence_complete"] for q in answers), n_q),
        "lost_given_retrieved": ratio(sum(q["gold_lost"] for q in answers), sum(q["gold_retrieved"] for q in answers)),
        "questions_lost_any_retrieved": ratio(sum(q["lost_any_retrieved"] for q in answers), n_q),
    }
    hops = [q["hop_retained"] for q in answers if q["hop_retained"]]
    if hops:
        evidence["musique_by_hop_index"] = {
            str(h): ratio(sum(h < len(q) and q[h] for q in hops), sum(h < len(q) for q in hops))
            for h in range(max(map(len, hops)))}
    if "canonical_answer_substring" in answers[0]:
        evidence["compressed_context_contains_verbatim_canonical_answer"] = ratio(
            sum(q["canonical_answer_substring"] for q in answers), n_q)

    row = episodes[0]["row"]
    codec = {"retained_passages": distribution([r["retained"] for r in stats]),
             "nongold_passages": distribution([r["nongold"] for r in stats]),
             "calls_with_new": len(compressor_diagnostics)}
    if row.startswith("r4"):
        codec["invalid"] = ratio(sum(x["invalid_reason"] is not None for x in compressor_diagnostics),
                                 len(compressor_diagnostics))
        codec["over_budget"] = ratio(sum(x["invalid_reason"] == "over_budget" for x in compressor_diagnostics),
                                     len(compressor_diagnostics))
    if row.startswith(("r4", "RR-τ", "LLMLingua-2")):
        elapsed = [x["wall_seconds"] for x in compressor_diagnostics]
        codec["seconds_per_call"] = distribution(elapsed) if elapsed else None
    return {"n_tasks": n, "n_questions": n_q,
            "f1": fsum(a["f1"] for a in answers) / n_q,
            "em": fsum(a["em"] for a in answers) / n_q,
            "process": process, "evidence": evidence, "compression": codec,
            "cost": {k: distribution([r[k] for r in stats]) for k in ("uncached", "cached", "output")}}, candidate_scores


def analyze(episodes, n_tasks):
    data = {d: {k: [] for k in ROWS} for d in ("musique", "hotpotqa", "2wiki")}
    seen = set()
    for ep in episodes:
        key = ep["dataset"], ep["row"], ep["task"]
        need(key not in seen, f"Duplicate episode: {key}")
        data[key[0]][key[1]].append(ep)
        seen.add(key)
    need(all(len(rows) == n_tasks for groups in data.values() for rows in groups.values()), "Episode coverage differs")
    data["pooled"] = {row: list(chain.from_iterable(group[row] for group in data.values())) for row in ROWS}
    output = {}
    for dataset, groups in data.items():
        summarized = {row: analyze_row(eps) for row, eps in groups.items()}
        output[dataset] = {
            "rows": {row: x[0] for row, x in summarized.items()},
            "paired": {f"{a} − {b}": paired(summarized[a][1], summarized[b][1]) for a, b in PAIRS},
        }
    return output
