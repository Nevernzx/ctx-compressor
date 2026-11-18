#!/usr/bin/env python3
"""Validate and report TASK_03 predictions, using only Python's standard library.

This script never imports a model runtime and never modifies source predictions
or TASK_01/02 artifacts. Public render_criteria_markdown(results) returns the
complete PLAN §6 table for direct inclusion in README.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.qa_metrics import score_answer
from src.reader import parse_answer, render_passages, render_user_message


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values, p):
    rank = (len(values) - 1) * p / 100
    lower, upper = math.floor(rank), math.ceil(rank)
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def token_distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {"n": len(values), "mean": sum(values) / len(values),
            "p50": _percentile(values, 50), "p90": _percentile(values, 90), "max": max(values)}


def _score_group(rows):
    n = len(rows)
    return {"n": n, "em": sum(row["em"] for row in rows) / n if n else None,
            "f1": sum(row["f1"] for row in rows) / n if n else None}


def _corpus(raw_dev, corpus_manifest):
    keys, corpus = {}, []
    for example in raw_dev:
        for paragraph in example["paragraphs"]:
            key = (paragraph["title"], paragraph["paragraph_text"])
            if key not in keys:
                keys[key] = len(corpus)
                corpus.append({"title": key[0], "paragraph_text": key[1]})
    require(len(corpus) == len(corpus_manifest), "raw dev corpus length differs from TASK_02 manifest")
    for pid, (paragraph, stored) in enumerate(zip(corpus, corpus_manifest)):
        rendered = paragraph["title"] + "\n" + paragraph["paragraph_text"]
        require(stored["pid"] == pid and stored["title"] == paragraph["title"], f"corpus metadata mismatch pid={pid}")
        require(hashlib.sha1(rendered.encode("utf-8")).hexdigest() == stored["sha1"], f"corpus SHA1 mismatch pid={pid}")
    return corpus, keys


def summarize_predictions(predictions, final_contexts, tasks, raw_dev, feasibility_results, corpus_manifest):
    """Strictly validate all 12,000 records, then return machine-readable metrics.

    Input EM/F1 and output aggregate EM/F1 are fractions in [0,1]. Deltas whose
    keys end in _pp are percentage points. No predictions are repaired in place.
    """
    n_tasks = feasibility_results["settings"]["n_tasks"]
    require(n_tasks == 150, "TASK_03 requires the original 150 TASK_02 tasks")
    configurations = list(feasibility_results["configs"])
    require(len(configurations) == 19 and "no_context" not in configurations, "expected original 19 TASK_02 configurations")
    expected_configs = configurations + ["no_context"]
    require(len(tasks) == n_tasks and {t["task"] for t in tasks} == set(range(n_tasks)), "task IDs must be exactly 0..149")
    task_by_id = {task["task"]: task for task in tasks}
    examples = {example["id"]: example for example in raw_dev}
    require(len(examples) == len(raw_dev), "duplicate raw dev example IDs")
    corpus, key2pid = _corpus(raw_dev, corpus_manifest)
    question_meta, question_gold = {}, {}
    for task_id, task in task_by_id.items():
        require(len(task["example_ids"]) == 4, f"task {task_id} does not have four questions")
        for question_index, example_id in enumerate(task["example_ids"]):
            example = examples[example_id]
            by_idx = {paragraph["idx"]: paragraph for paragraph in example["paragraphs"]}
            gold = set()
            for hop in example["question_decomposition"]:
                paragraph = by_idx[hop["paragraph_support_idx"]]
                gold.add(key2pid[(paragraph["title"], paragraph["paragraph_text"])])
            require(gold == set(task["question_gold"][question_index]), f"question gold mismatch task={task_id} question={question_index}")
            question_meta[task_id, question_index] = example
            question_gold[task_id, question_index] = gold
        require(set(task["all_gold"]) == set().union(*(question_gold[task_id, i] for i in range(4))), f"task gold mismatch task={task_id}")
    require(len({ex["id"] for ex in question_meta.values()}) == 600, "TASK_02 questions are not 600 distinct examples")
    expected_question_keys = set(question_meta)
    contexts = {}
    require(len(final_contexts) == 19 * n_tasks, "expected 2,850 final contexts")
    for final in final_contexts:
        config, task_id = final["config"], final["task"]
        require(config in configurations and task_id in task_by_id, "unexpected final-context configuration or task")
        key = config, task_id
        require(key not in contexts, f"duplicate final context {key}")
        pids = final["keep_after"]
        require(len(pids) == len(set(pids)), f"duplicate pid in final context {key}")
        require(all(type(pid) is int and 0 <= pid < len(corpus) for pid in pids), f"invalid pid in final context {key}")
        require(len(final["questions"]) == 4, f"incorrect question metadata count in {key}")
        for qi, metadata in enumerate(final["questions"]):
            example = question_meta[task_id, qi]
            for field in ("id", "question", "answer", "answer_aliases"):
                require(metadata[field] == example[field], f"raw dev/final context {field} mismatch {key}/{qi}")
        contexts[key] = final
    require(set(contexts) == {(config, task_id) for config in configurations for task_id in range(n_tasks)}, "final context matrix incomplete")
    require(len(predictions) == 12000, f"expected 12,000 predictions, found {len(predictions)}")
    by_config = {config: {} for config in expected_configs}
    # Hash the original user message; no chat-template generation or tokenization.
    passage_cache = {}
    for record in predictions:
        config = record["config"]
        require(config in by_config, f"unexpected prediction configuration {config}")
        task_id, qi = record["task"], record["question_index"]
        require(type(task_id) is int and type(qi) is int and (task_id, qi) in question_meta, "invalid prediction task/question index")
        question_key = task_id, qi
        require(question_key not in by_config[config], f"duplicate prediction {config}/{question_key}")
        example = question_meta[question_key]
        require(record["example_id"] == example["id"], f"prediction example ID mismatch {config}/{question_key}")
        require(type(record["question_hops"]) is int and record["question_hops"] == len(example["question_decomposition"]), f"question hops mismatch {config}/{question_key}")
        if config == "no_context":
            expected_complete, passages = False, ""
        else:
            key = config, task_id
            final = contexts[key]
            expected_complete = question_gold[question_key] <= set(final["keep_after"])
            if key not in passage_cache:
                passage_cache[key] = render_passages(corpus, final["keep_after"])
            passages = passage_cache[key]
        require(type(record["evidence_complete"]) is bool and record["evidence_complete"] == expected_complete,
                f"evidence_complete mismatch {config}/{question_key}")
        user = render_user_message(passages, example["question"], no_context=config == "no_context")
        require(hashlib.sha1(user.encode("utf-8")).hexdigest() == record["prompt_sha1"], f"user-message SHA1 mismatch {config}/{question_key}")
        parsed = parse_answer(record["raw_output"])
        require(parsed == record["pred"], f"answer parsing mismatch {config}/{question_key}")
        scores = score_answer(parsed, example["answer"], example["answer_aliases"])
        for metric in ("em", "f1"):
            value = record[metric]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                    f"invalid {metric} {config}/{question_key}")
            require(math.isclose(value, scores[metric], rel_tol=0, abs_tol=1e-12), f"recomputed {metric} mismatch {config}/{question_key}")
        require(record["em"] in (0, 1), f"nonbinary EM {config}/{question_key}")
        require(type(record["prompt_tokens"]) is int and record["prompt_tokens"] > 0, f"invalid prompt token count {config}/{question_key}")
        if "input_truncated" in record:
            require(type(record["input_truncated"]) is bool, f"invalid input_truncated flag {config}/{question_key}")
        if "finish_reason" in record:
            require(record["finish_reason"] in ("stop", "length"), f"incomplete generation {config}/{question_key}")
        if "long_answer" in record:
            require(record["long_answer"] == (len(parsed.split()) > 20), f"long_answer mismatch {config}/{question_key}")
        if "output_tokens" in record:
            require(type(record["output_tokens"]) is int and 0 <= record["output_tokens"] <= 48, f"invalid output token count {config}/{question_key}")
        by_config[config][question_key] = record
    aggregates = {}
    for config, indexed in by_config.items():
        require(set(indexed) == expected_question_keys, f"incomplete question matrix in {config}")
        rows = list(indexed.values())
        aggregate = _score_group(rows)
        aggregate["by_question_position"] = {str(i + 1): _score_group([r for r in rows if r["question_index"] == i]) for i in range(4)}
        aggregate["by_question_hops"] = {str(hops): _score_group([r for r in rows if r["question_hops"] == hops]) for hops in (2, 3, 4)}
        aggregate["by_evidence_complete"] = {str(complete).lower(): _score_group([r for r in rows if r["evidence_complete"] == complete]) for complete in (True, False)}
        aggregate["prompt_tokens"] = token_distribution([r["prompt_tokens"] for r in rows])
        long_answers = sum(len(r["pred"].split()) > 20 for r in rows)
        aggregate["format_checks"] = {
            "n": len(rows), "long_answers_over_20_words": long_answers,
            "long_answer_rate": long_answers / len(rows),
            "empty_predictions": sum(not r["pred"] for r in rows),
            "finish_reasons": dict(Counter(r.get("finish_reason", "not_recorded") for r in rows)),
            "prompt_plus_48_over_limit": sum(r["prompt_tokens"] + 48 > 40960 for r in rows),
        }
        if config != "no_context":
            expected_count = feasibility_results["configs"][config]["metrics"]["evidence_complete_questions"]["count"]
            require(aggregate["by_evidence_complete"]["true"]["n"] == expected_count, f"TASK_02 evidence completion aggregate mismatch {config}")
        aggregates[config] = aggregate
    all_gaps, medium_gaps = {}, {}
    for mode in ("reselect", "append"):
        all_gaps[mode] = {}
        for tier in ("tight", "mid", "loose"):
            oracle_name, bm25_name = f"oracle|{mode}|{tier}", f"bm25|{mode}|{tier}"
            om, bm = feasibility_results["configs"][oracle_name]["metrics"], feasibility_results["configs"][bm25_name]["metrics"]
            gap = {
                "n_questions": 600, "n_tasks": 150, "n_steps": om["n_steps"],
                "oracle_f1": aggregates[oracle_name]["f1"], "bm25_f1": aggregates[bm25_name]["f1"],
                "f1_gap_pp": 100 * (aggregates[oracle_name]["f1"] - aggregates[bm25_name]["f1"]),
                "final_recall_gap_pp": 100 * (om["final_recall"]["mean"] - bm["final_recall"]["mean"]),
                "step_recall_gap_pp": 100 * (om["step_recall"]["micro"]["mean"] - bm["step_recall"]["micro"]["mean"]),
                "task_step_recall_gap_pp": 100 * (om["step_recall"]["task_macro"]["mean"] - bm["step_recall"]["task_macro"]["mean"]),
            }
            gap["f1_gap_at_least_5pp"] = gap["f1_gap_pp"] >= 5
            gap["final_recall_gap_at_least_15pp"] = gap["final_recall_gap_pp"] >= 15
            gap["step_recall_gap_at_least_15pp"] = gap["step_recall_gap_pp"] >= 15
            all_gaps[mode][tier] = gap
        medium_gaps[mode] = all_gaps[mode]["mid"]
    paired = {}
    for mode in ("reselect", "append"):
        oracle_name = f"oracle|{mode}|mid"
        comparison_counts = Counter(oracle_higher=0, equal=0, no_compress_higher=0)
        em_table = {"oracle_correct": {"no_compress_correct": 0, "no_compress_incorrect": 0},
                    "oracle_incorrect": {"no_compress_correct": 0, "no_compress_incorrect": 0}}
        for key in sorted(expected_question_keys):
            oracle, no_compress = by_config[oracle_name][key], by_config["no_compress"][key]
            if math.isclose(oracle["f1"], no_compress["f1"], rel_tol=0, abs_tol=1e-12):
                comparison_counts["equal"] += 1
            elif oracle["f1"] > no_compress["f1"]:
                comparison_counts["oracle_higher"] += 1
            else:
                comparison_counts["no_compress_higher"] += 1
            em_table["oracle_correct" if oracle["em"] == 1 else "oracle_incorrect"][
                "no_compress_correct" if no_compress["em"] == 1 else "no_compress_incorrect"] += 1
        paired[mode] = {
            "n": 600, "oracle_config": oracle_name,
            "oracle": _score_group(list(by_config[oracle_name].values())),
            "no_compress": _score_group(list(by_config["no_compress"].values())),
            "oracle_minus_no_compress_f1_pp": 100 * (aggregates[oracle_name]["f1"] - aggregates["no_compress"]["f1"]),
            "no_compress_minus_oracle_f1_pp": 100 * (aggregates["no_compress"]["f1"] - aggregates[oracle_name]["f1"]),
            "paired_f1_counts": dict(comparison_counts), "paired_em_2x2": em_table,
        }
    length_finishes = sum(r.get("finish_reason") == "length" for r in predictions)
    oversized = sum(r["prompt_tokens"] + 48 > 40960 for r in predictions)
    input_truncated = sum(r.get("input_truncated", False) for r in predictions)
    return {
        "settings": {"model": "Qwen3-8B", "n_tasks": 150, "questions_per_task": 4,
                     "n_configs": 20, "n_per_config": 600, "temperature": 0,
                     "max_tokens": 48, "max_model_len": 40960, "thinking": False,
                     "score_units": "em/f1 fractions; *_pp fields percentage points",
                     "f1_pair_equality_absolute_tolerance": 1e-12},
        "n_predictions": len(predictions), "configs": aggregates,
        "criteria": {"mid_oracle_minus_bm25": medium_gaps, "all_budget_gaps": all_gaps,
                     "no_compress_vs_oracle_mid": paired, "no_context": aggregates["no_context"]},
        "validation": {"passed": not oversized and not input_truncated,
                       "prediction_matrix": "20 configurations × 600 distinct questions",
                       "reparsed_and_rescored_records": len(predictions),
                       "metadata_and_user_sha1_checked_records": len(predictions),
                       "corpus_sha1_checked_paragraphs": len(corpus),
                       "task02_final_contexts_checked": len(contexts),
                       "prompt_plus_48_over_limit": oversized,
                       "input_truncated_count": input_truncated,
                       "input_truncated_flag_missing_count": sum("input_truncated" not in r for r in predictions),
                       "generation_length_finish_count": length_finishes,
                       "generation_length_finish_rate": length_finishes / len(predictions),
                       "output_limit_interpretation": "The fixed 48-token generation limit is reported without filtering or reruns; it is not input-context truncation",
                       "finish_reason_missing_count": sum("finish_reason" not in r for r in predictions),
                       "prompt_token_verification": "Recorded prompt counts checked against limit; no model/tokenizer loaded by reporting script",
                       "no_compress_max_prompt_tokens": aggregates["no_compress"]["prompt_tokens"]["max"]},
    }


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(map(_cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(_cell, row)) + " |" for row in rows])


def _percentage(value):
    return "未适用" if value is None else f"{100 * value:.3f}%"


def _number(value):
    return "未适用" if value is None else f"{value:,.3f}".rstrip("0").rstrip(".")


def _group(group):
    return f"{_percentage(group['f1'])}（n={group['n']}）"


def _ordered_names(results):
    return (["no_compress", "no_context"] +
            [f"{policy}|{mode}|{tier}" for policy in ("latest", "bm25", "oracle")
             for mode in ("reselect", "append") for tier in ("tight", "mid", "loose")])


def _paired_text(mode, entry):
    counts, matrix = entry["paired_f1_counts"], entry["paired_em_2x2"]
    return (f"{mode}/mid，n={entry['n']}：oracle − no_compress F1={entry['oracle_minus_no_compress_f1_pp']:.3f} pp；"
            f"逐题 oracle 更高/相等/no_compress 更高={counts['oracle_higher']}/{counts['equal']}/{counts['no_compress_higher']}；"
            f"EM 配对 2×2（行=oracle 对/错，列=no_compress 对/错）="
            f"[[{matrix['oracle_correct']['no_compress_correct']},{matrix['oracle_correct']['no_compress_incorrect']}],"
            f"[{matrix['oracle_incorrect']['no_compress_correct']},{matrix['oracle_incorrect']['no_compress_incorrect']}]]")


def render_criteria_markdown(results):
    """Return the full, self-contained criterion section for report or README."""
    criteria = results["criteria"]
    medium_evidence, medium_f1, all_evidence, all_f1 = [], [], [], []
    for mode in ("reselect", "append"):
        gap = criteria["mid_oracle_minus_bm25"][mode]
        medium_evidence.append(
            f"{mode}/mid：任务末召回差 {gap['final_recall_gap_pp']:.3f} pp（≥15：{'是' if gap['final_recall_gap_at_least_15pp'] else '否'}，n={gap['n_tasks']}任务）；"
            f"单步按步差 {gap['step_recall_gap_pp']:.3f} pp（≥15：{'是' if gap['step_recall_gap_at_least_15pp'] else '否'}，n={gap['n_steps']}步）；"
            f"单步按任务差 {gap['task_step_recall_gap_pp']:.3f} pp（n={gap['n_tasks']}任务）")
        medium_f1.append(f"{mode}/mid：oracle {_percentage(gap['oracle_f1'])} − BM25 {_percentage(gap['bm25_f1'])} = "
                         f"{gap['f1_gap_pp']:.3f} pp（≥5：{'是' if gap['f1_gap_at_least_5pp'] else '否'}；各 n={gap['n_questions']}题）")
        for tier in ("tight", "mid", "loose"):
            other = criteria["all_budget_gaps"][mode][tier]
            all_evidence.append(f"{mode}/{tier}：任务末差 {other['final_recall_gap_pp']:.3f} pp（n={other['n_tasks']}任务）；"
                                f"单步按步差 {other['step_recall_gap_pp']:.3f} pp（n={other['n_steps']}步）；"
                                f"单步按任务差 {other['task_step_recall_gap_pp']:.3f} pp（n={other['n_tasks']}任务）")
            all_f1.append(f"{mode}/{tier}：{other['f1_gap_pp']:.3f} pp（各 n={other['n_questions']}题）")
    paired = "<br>".join(_paired_text(mode, criteria["no_compress_vs_oracle_mid"][mode]) for mode in ("reselect", "append"))
    closed = criteria["no_context"]
    rows = [
        ["中档预算下，BM25 的证据召回比 oracle 低 15 个点以上，F1 低 5 个点以上", "<br>".join(medium_evidence), "<br>".join(medium_f1)],
        ["BM25 已接近 oracle", "<br>".join(all_evidence) + "<br>差值仅作描述，不新增‘接近’阈值。", "<br>".join(all_f1)],
        ["不压缩的 F1 与 oracle 接近", "两种 oracle 中档，使用相同 600 题逐题配对。", paired],
        ["不压缩的 F1 明显低于 oracle", "两种 oracle 中档，使用相同 600 题逐题配对。", paired],
        ["附加闭卷对照 no_context", f"无段落上下文，同一批 n={closed['n']} 题。", f"EM {_percentage(closed['em'])}；F1 {_percentage(closed['f1'])}；n={closed['n']}题"],
    ]
    return ("### PLAN §6 判据表（填数，不作开工判断）\n\n"
            "前四行观察条件保留 PLAN 原文。oracle − BM25 和 oracle − no_compress 的差值均以百分点（pp）表示；"
            "不对‘接近’或‘明显更低’另定阈值，不据本表调整任务、检索或预算。\n\n" +
            _table(["PLAN 原观察条件", "证据指标 / 配对范围", "阅读模型观测"], rows) + "\n")


def render_report(results):
    """Render Chinese tables; all EM/F1 are displayed as percentages with n."""
    configs = results["configs"]
    names = _ordered_names(results)
    lines = ["# §6 阅读模型 EM / F1：Qwen3-8B", "",
             "20 种配置 × 同一批 600 道题，共 12,000 个独立 prompt。每任务 4 道题分别回答，复用任务 02 已保存的最终上下文；"
             "本任务没有重跑压缩、检索或预算。单次运行，报告均值和 n，不计算置信区间。",
             "模型为本地 Qwen3-8B；单条 user 消息，关闭 thinking，greedy，最多生成 48 token。"
             "答案按任务书固定流程解析，对标准答案及全部 aliases 计算 SQuAD 归一化后的最大 EM / F1。"
             "报告脚本重新解析原始输出、重新评分，并核对逐题元数据、证据齐全标记和 user 原文 SHA1。", "",
             "## 各配置整体表现", "",
             _table(["配置", "n 题", "EM", "F1"],
                    [[name, configs[name]["n"], _percentage(configs[name]["em"]), _percentage(configs[name]["f1"])] for name in names]), "",
             "## 按任务内题目位置的 F1", "",
             _table(["配置", "第 1 题", "第 2 题", "第 3 题", "第 4 题"],
                    [[name] + [_group(configs[name]["by_question_position"][str(i)]) for i in range(1, 5)] for name in names]), "",
             "## 按题目跳数的 F1", "",
             _table(["配置", "2 跳", "3 跳", "4 跳"],
                    [[name] + [_group(configs[name]["by_question_hops"][str(i)]) for i in (2, 3, 4)] for name in names]), "",
             "## 按证据是否齐全的 F1", "",
             "证据齐全指该题全部金标准段 pid 都出现在该任务该配置的最终保留集合中；沿用任务 02 的定义。"
             "no_context 没有上下文，全部归入证据不齐全。空分组显示未适用，不记作零分。", "",
             _table(["配置", "证据齐全", "证据不齐全"],
                    [[name, _group(configs[name]["by_evidence_complete"]["true"]),
                      _group(configs[name]["by_evidence_complete"]["false"])] for name in names]), "",
             "## 每题 prompt token 分布", "",
             "这里使用实际提交给 vLLM 的 chat-template 后 prompt token 数；不是任务 02 的片段预算。"
             "输入不得截断，每条 prompt 加上 48 个生成 token 的预留必须不超过 40,960。", "",
             _table(["配置", "n", "mean", "P50", "P90", "max"],
                    [[name, configs[name]["prompt_tokens"]["n"]] + [_number(configs[name]["prompt_tokens"][key])
                     for key in ("mean", "p50", "p90", "max")] for name in names]), "",
             f"**no_compress 最大 prompt：{configs['no_compress']['prompt_tokens']['max']:,} token**；"
             f"加 48 token 预留后为 {configs['no_compress']['prompt_tokens']['max'] + 48:,}，上限 40,960。"
             "分位数采用线性插值，n 均为实际题数。", "",
             render_criteria_markdown(results), "",
             "## no_compress 与 oracle 中档的逐题配对", ""]
    for mode in ("reselect", "append"):
        entry = results["criteria"]["no_compress_vs_oracle_mid"][mode]
        count, matrix = entry["paired_f1_counts"], entry["paired_em_2x2"]
        lines.extend([f"### {entry['oracle_config']} 对 no_compress", "",
                      f"同一批 n={entry['n']} 题；oracle F1 {_percentage(entry['oracle']['f1'])}，"
                      f"no_compress F1 {_percentage(entry['no_compress']['f1'])}；"
                      f"oracle − no_compress = {entry['oracle_minus_no_compress_f1_pp']:.3f} pp。",
                      f"逐题 F1：oracle 更高 {count['oracle_higher']}，相等 {count['equal']}，no_compress 更高 {count['no_compress_higher']}。", "",
                      _table(["EM 配对（n=600）", "no_compress 正确", "no_compress 错误"], [
                          ["oracle 正确", matrix["oracle_correct"]["no_compress_correct"], matrix["oracle_correct"]["no_compress_incorrect"]],
                          ["oracle 错误", matrix["oracle_incorrect"]["no_compress_correct"], matrix["oracle_incorrect"]["no_compress_incorrect"]],
                      ]), ""])
    lines.extend(["## 输出格式与完整性检查", "",
                  "`finish_reason=length` 表示用满任务书固定的 48-token 生成额度，单独统计；它不等于输入上下文被截断。"
                  "这些原始输出保留、照常解析评分，不剔除、不延长生成、不选择性补跑。输入完整性由每条 prompt+48 长度检查和推理时 token 序列一致性断言核验。", "",
                  _table(["配置", "n", ">20 词答案数 / 比例", "空解析答案", "结束原因", "prompt+48 超限"], [
                      [name, configs[name]["format_checks"]["n"],
                       f"{configs[name]['format_checks']['long_answers_over_20_words']} / {_percentage(configs[name]['format_checks']['long_answer_rate'])}",
                       configs[name]["format_checks"]["empty_predictions"],
                       json.dumps(configs[name]["format_checks"]["finish_reasons"], ensure_ascii=False, sort_keys=True),
                       configs[name]["format_checks"]["prompt_plus_48_over_limit"]] for name in names]), "",
                  _table(["检查项", "结果"], [[key, json.dumps(value, ensure_ascii=False)] for key, value in results["validation"].items()]), "",
                  "报告脚本不加载模型或分词器：prompt token 数来自预测记录，并检查长度上限；user 消息原文 SHA1 则由任务 02 上下文和原始 dev 独立重建核对。"
                  "[机器可读结果](reader_results.json)保存同一份数字；[推理运行配置](run_config.json)记录 GPU、环境、生成参数与资源用量。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PROJECT / "data/reader/predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "eval/reader")
    parser.add_argument("--final-contexts", type=Path, default=PROJECT / "data/feasibility/final_contexts.jsonl")
    parser.add_argument("--tasks", type=Path, default=PROJECT / "data/feasibility/tasks_dev150.jsonl")
    parser.add_argument("--raw-dev", type=Path, default=PROJECT / "data/raw/musique/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--feasibility-results", type=Path, default=PROJECT / "eval/feasibility/feasibility_results.json")
    parser.add_argument("--corpus-manifest", type=Path, default=PROJECT / "data/feasibility/corpus_dev_ntok.jsonl")
    args = parser.parse_args()
    if PROJECT != Path("/root/workplace/ctx-compressor"):
        parser.error("execute report generation only on 5880 in the formal project directory")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(PROJECT / "eval/reader"):
        parser.error("report output must remain in eval/reader")
    result_path, report_path = args.output_dir / "reader_results.json", args.output_dir / "reader_report.md"
    if result_path.exists() or report_path.exists():
        parser.error("refusing to overwrite an existing reader report")
    results = summarize_predictions(
        _read_jsonl(args.predictions), _read_jsonl(args.final_contexts), _read_jsonl(args.tasks),
        _read_jsonl(args.raw_dev), json.loads(args.feasibility_results.read_text(encoding="utf-8")),
        _read_jsonl(args.corpus_manifest),
    )
    results["provenance"] = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                             "input_sha256": {str(path): _sha256(path) for path in
                                               (args.predictions, args.final_contexts, args.tasks, args.raw_dev,
                                                args.feasibility_results, args.corpus_manifest)}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(render_report(results), encoding="utf-8")
    print(json.dumps({"n_predictions": results["n_predictions"], "validation": results["validation"],
                      "mid_f1_gaps_pp": {mode: entry["f1_gap_pp"] for mode, entry in
                                         results["criteria"]["mid_oracle_minus_bm25"].items()}}, ensure_ascii=False))
    if not results["validation"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
