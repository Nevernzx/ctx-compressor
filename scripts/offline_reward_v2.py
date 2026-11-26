#!/usr/bin/env python3
"""Re-execute the frozen Task 09 action grid, then price its actions with reward v2."""
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import score_hedge_train as previous
from scripts.hedge_action import TAU_GRID, execute_hedge
from scripts.heldout_eval import file_record
from src.grpo_reward_v2 import score_completion_v2

MU_GRID = (0, 0.1, 0.25, 0.5, 1.0)
SCOPES = ("pooled", "append", "reselect")
OUT = PROJECT / "eval/reward_v2"


def v2_cell(counts, mu):
    n = counts["n"]
    # Integer reward units preserve exact ties under each registered token price.
    numerator = -100000 * (counts["n_lost"] + counts["invalid"]) - round(100 * mu) * counts["used"]
    def rate(field, denominator):
        return {"count": counts[field], "n": denominator,
                "rate": counts[field] / denominator if denominator else None}
    return {"n": n, "mean_reward": {"mean": numerator / (100000 * n), "sum": numerator / 100000,
                                    "n": n, "numerator_in_1e5_reward_units": numerator},
            "hedge_rate": rate("hedged", n),
            "current_gold_miss_rate": rate("miss", counts["current_n"]),
            "mean_extra_tokens": {"sum": counts["extra_tokens"], "mean": counts["extra_tokens"] / n, "n": n},
            "n_lost_total": counts["n_lost"], "invalid_total": counts["invalid"], "used_tokens_total": counts["used"]}


def select(table):
    return max(sorted(TAU_GRID), key=lambda tau: table[str(tau)]["mean_reward"]["numerator_in_1e5_reward_units"])


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [OUT / f"offline_tau_table.{ext}" for ext in ("json","md")]
    if any(p.exists() for p in targets):
        raise FileExistsError(targets)
    state_file, gold_pids = previous.train_inputs()
    prior = json.loads(previous.RESULT.read_bytes())
    score_file = file_record(previous.RAW)
    if state_file != prior["state_file"] or score_file != prior["score_file"] or prior["status"] != "complete":
        raise ValueError("Frozen training inputs differ from Task 09")
    old_tables = {s: {t: previous.Aggregate() for t in TAU_GRID} for s in ("pooled", *previous.CELLS)}
    counts = {s: {t: Counter() for t in TAU_GRID} for s in SCOPES}
    n = 0
    with previous.STATES.open("rb") as states, previous.RAW.open("rb") as scores:
        for line, scored_line in zip_longest(states, scores):
            if line is None or scored_line is None:
                raise ValueError("Unmatched training input lengths")
            row, saved = json.loads(line), json.loads(scored_line)
            cell = f"{row['mode']}|{row['budget_tier']}"
            state = json.loads(row["reward_state_json"])
            current = previous.local_current_gold(row, state, gold_pids)
            if (row["split"] != "train" or cell != saved["cell"] or current != saved["current_gold_new_lid"]
                or row["greedy_output"] != saved["greedy_raw"]
                or any(row[k] != saved[k] for k in ("split","task","trajectory_id","step","prompt_sha256"))):
                raise ValueError("Task 09 training identity mismatch")
            ranked = {int(k): v for k, v in saved["scores"].items()}
            for tau in TAU_GRID:
                x = execute_hedge(state, row["greedy_output"], ranked, tau)
                reference = saved["grid_results"][str(tau)]
                if x["p1"] != saved["p1"] or any(x[k] != v for k, v in reference.items()):
                    raise ValueError(f"Task 09 state/action mismatch at {n}, {tau}")
                for s in ("pooled", cell):
                    old_tables[s][tau].add(x, state["gold"], current)
                parts = score_completion_v2(x["raw_output"], state, 0)
                if any(parts[k] != x[k] for k in ("keep_after", "used_after", "invalid_reason")):
                    raise ValueError("Reward parser execution mismatch")
                for scope in ("pooled", row["mode"]):
                    c = counts[scope][tau]
                    c.update(n=1, n_lost=parts["n_lost"], used=parts["used_after"],
                             invalid=parts["invalid_penalty"], hedged=x["hedged"], extra_tokens=x["extra_tokens"],
                             current_n=current is not None, miss=current is not None and current not in x["keep_after"])
            n += 1
    rebuilt = {s: {str(t): agg.result() for t, agg in by_tau.items()} for s, by_tau in old_tables.items()}
    if (n != 22158 or n != prior["n_states"] or rebuilt != {"pooled": prior["pooled"], **prior["cells"]}
        or previous.select_tau(rebuilt["pooled"]) != prior["tau_star"]):
        raise ValueError("Full v1 table reproduction failed")
    tables = {s: {str(mu): {str(t): v2_cell(counts[s][t], mu) for t in TAU_GRID}
                  for mu in MU_GRID} for s in SCOPES}
    stars = {s: {str(mu): select(tables[s][str(mu)]) for mu in MU_GRID} for s in SCOPES}
    result = {"status":"complete","scope":"train","n_states":n,"mu_grid":MU_GRID,"tau_grid":TAU_GRID,
              "inputs":{"states":state_file,"scores":score_file,"task09_table":file_record(previous.RESULT)},
              "v1_parity":{"exact":True,"n_state_tau":n*len(TAU_GRID),"n_table_cells":len(rebuilt)*len(TAU_GRID),
                           "tau_star":prior["tau_star"],"all_tables":rebuilt},
              "tau_star":stars,"tables":tables,"selection":"最高平均奖励，并列时取较小阈值。"}
    markdown = ["# 训练状态上的奖励 v2 阈值表", "",
                f"在 n={n:,} 个任务09训练状态上按原代码执行每个动作。v1 的合并表及六个模式×预算分层完全复现，共 n={len(rebuilt)*len(TAU_GRID)} 个表格单元，逐状态逐阈值核对 n={n*len(TAU_GRID):,} 次。",
                "", "μ 为每千个已保留 token 的价格。当前金段遗漏率以当前金段新出现的状态为分母。各价格独立选择平均奖励最高的阈值，并列取较小值。", ""]
    for scope, title in zip(SCOPES, ("合并", "追加模式", "重选模式")):
        table = tables[scope]
        n_scope = table["0"]["0.0"]["n"]
        markdown.extend([f"## {title}（n={n_scope:,} 个状态）", "",
                         "| μ | 最优阈值 | 状态数 n |","|---:|---:|---:|"])
        for mu in MU_GRID:
            markdown.append(f"| {mu} | {stars[scope][str(mu)]} | {n_scope} |")
        markdown.extend(["", "| 阈值 | 加一段率（计数/n）| 当前金段遗漏率（计数/n）| 平均额外 token（n）| "
                         + " | ".join(f"平均奖励 μ={m}（n）" for m in MU_GRID)+" |",
                         "|---:|---:|---:|---:|" + "---:|" * len(MU_GRID)])
        for tau in TAU_GRID:
            base = table["0"][str(tau)]
            rates = [f"{x['rate']:.3%}（{x['count']}/{x['n']}）" for x in
                     (base["hedge_rate"], base["current_gold_miss_rate"])]
            tokens = base["mean_extra_tokens"]["mean"]
            means = [f"{table[str(m)][str(tau)]['mean_reward']['mean']:.6f}（{n_scope}）" for m in MU_GRID]
            markdown.append("| " + " | ".join([str(tau), *rates, f"{tokens:.3f}（{n_scope}）", *means]) + " |")
        markdown.append("")
    with targets[1].open("x", encoding="utf-8") as f:
        f.write("\n".join(markdown)+"\n")
    with targets[0].open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"n":n,"tau_star":stars,"v1_exact":True}),flush=True)


if __name__ == "__main__":
    run()

