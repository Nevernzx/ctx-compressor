#!/usr/bin/env python3
"""Task 07 (Claude): turn closed-loop train states into a swift GRPO dataset (balanced, seeded).

Each row: messages=[user] (the exact eval user text) + reward_state_json (state + prompt tail
for the reward plugin) + identifiers. Train-split states only.
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--per-cell", type=int, required=True, help="prompts per mode x tier cell")
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--exclude", type=Path, default=None, help="probe_outputs.jsonl whose states are left out")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    excluded = set()
    if args.exclude:
        for line in open(args.exclude):
            r = json.loads(line)
            excluded.add((r["trajectory_id"], r["step"]))
    cells = defaultdict(list)
    for line in open(args.states):
        r = json.loads(line)
        if r["split"] != "train":
            raise ValueError("train states only")
        if (r["trajectory_id"], r["step"]) in excluded:
            continue
        cells[(r["mode"], r["budget_tier"])].append(r)
    rng = random.Random(args.seed)
    picked = []
    for key in sorted(cells):
        picked += rng.sample(cells[key], args.per_cell)
    rng.shuffle(picked)
    with args.out.open("x", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({"messages": r["messages"], "reward_state_json": r["reward_state_json"],
                                "trajectory_id": r["trajectory_id"], "state_step": r["step"],
                                "cell": f"{r['mode']}|{r['budget_tier']}"}, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(picked), "cells": {f"{k[0]}|{k[1]}": len(v) for k, v in cells.items()},
                      "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
