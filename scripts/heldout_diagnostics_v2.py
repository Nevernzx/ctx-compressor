#!/usr/bin/env python3
"""Aggregate Task 06 diagnostics from closed step files, without model or policy replay."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
RESULTS = PROJECT / "eval/sft_run_v2"
V1_RESULTS = PROJECT / "eval/sft_run"
MANIFEST_SHA = "152d13bc15f5583d93ce3ba7b53410abcefab46f43e3ab1aa681251ffb6e4d14"
CHECKPOINTS = tuple(f"checkpoint-{n}" for n in (500, 1000, 1500, 2000, 2500))
CPU_METRIC_SHA = {
    "bm25": "e407ac8005fd0841a64a942f166e6684ed377a41391d87978d6b92bb2f6d117f",
    "latest": "d13935b6b46af12d3bbf8d8a4d1cea777bd874b8928d7e2a47185397501809a4",
    "no_compress": "73bbf3061c1396d02a5871f66f4548e14cf9db219e18e9338779618a3860fe96",
    "oracle": "0d690a2c882865752545d668012ac84b02488097319bc932457e10af8c6d5a86",
}
CAUSES = ("invalid_output", "new_gold_not_added", "old_gold_dropped", "both", "none")
RATES = ("new_gold_add", "kept_gold_survival", "all_gold_valid_drop",
         "all_gold_among_valid_reselect")
DEFINITIONS = {
    "new_gold_add": "Fragment-weighted: returned, known gold through the current step, not in kept_before; count how many are in keep_after. Includes invalid steps.",
    "kept_gold_survival": "Reselect only. Among kept fragments already revealed as gold BEFORE this return, count how many remain in keep_after. Includes invalid steps.",
    "all_gold_among_valid_reselect": "Eligible are valid reselect steps >=1; count those whose kept_before is a subset of previously revealed gold (including an empty set).",
    "all_gold_valid_drop": "Among those eligible all-gold steps, count actual deletions from kept_before.",
    "first_error": "Earliest step with an invalid output, an eligible new gold missing after the action, or a previously revealed kept gold dropped. Invalid output takes precedence. Both means a miss and a drop on the same valid step. The denominator for each cause (including none) is all trajectories.",
    "by_step": "Zero-based retrieval position. First-error counts at a position are divided by trajectories that reached that position. None is counted at termination only.",
    "rate": "count/n, null if n=0",
}


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def rate(count, n):
    return {"count": count, "n": n, "rate": count/n if n else None}


def increment(stats, name, count, n):
    stats[name] += count
    stats[name + "_n"] += n


def step_observations(row, known_before):
    kept = set(row["kept_before"])
    after = set(row["keep_after"])
    new_gold = (set(row["returned"]) & (known_before | {row["gold"]})) - kept
    old_gold = kept & known_before
    new_missed = bool(new_gold - after)
    old_dropped = bool(old_gold - after)
    valid = row["action_valid"]
    if valid != (row["invalid_reason"] is None) or (not valid and kept != after):
        raise ValueError("Inconsistent action validity or invalid action changed kept set")
    if row["mode"] == "append" and not kept <= after:
        raise ValueError("Append action removed an earlier fragment")
    counts = Counter(n_steps=1)
    increment(counts, "new_gold_add", len(new_gold & after), len(new_gold))
    if row["mode"] == "reselect":
        increment(counts, "kept_gold_survival", len(old_gold & after), len(old_gold))
        if row["step"] >= 1 and valid:
            all_gold = kept <= known_before
            increment(counts, "all_gold_among_valid_reselect", int(all_gold), 1)
            if all_gold:
                increment(counts, "all_gold_valid_drop", int(bool(kept - after)), 1)
    if not valid:
        cause = "invalid_output"
    elif new_missed and old_dropped:
        cause = "both"
    elif new_missed:
        cause = "new_gold_not_added"
    elif old_dropped:
        cause = "old_gold_dropped"
    else:
        cause = None
    return counts, cause


def summary(counts):
    return {"n_steps": counts["n_steps"], "n_trajectories": counts["n_trajectories"],
            **{key: rate(counts[key], counts[key + "_n"]) for key in RATES},
            "first_error": {key: rate(counts["first_" + key], counts["n_trajectories"])
                            for key in CAUSES}}


def summarize_strategy(name, path, metrics, tasks, expected_sha):
    cells, positions = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    # A trajectory carries only known evidence, last kept IDs, step, and first error.
    state = {}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            cell = f'{row["mode"]}|{row["budget_tier"]}'
            task_id, step = row["task"], row["step"]
            key = (cell, task_id)
            trajectory = state.setdefault(key, {"known": set(), "kept": [], "next": 0, "first": None})
            saved = tasks[task_id][step]
            if (row["strategy"] != name or row["config"] != f"{name}|{cell}"
                    or row["trajectory_id"] != f"{name}|{cell}|{task_id}"
                    or trajectory["next"] != step or trajectory["kept"] != row["kept_before"]
                    or row["gold"] != saved["gold"] or row["returned"] != saved["new"]):
                raise ValueError(f"Saved steps, closed-loop state or trajectory identity differ at {name}/{key}/{step}")
            counts, cause = step_observations(row, trajectory["known"])
            if trajectory["first"] is None and cause:
                counts["first_" + cause] = 1
                trajectory["first"] = cause
            if step + 1 == len(tasks[task_id]):
                counts["n_trajectories"] = 1
                if trajectory["first"] is None:
                    counts["first_none"] = 1
            cells[cell].update(counts)
            positions[cell][step].update(counts)
            positions[cell][step]["n_trajectories"] += 1 - counts["n_trajectories"]
            trajectory.update(kept=row["keep_after"], next=step + 1)
            trajectory["known"].add(row["gold"])
    if digest.hexdigest() != expected_sha:
        raise ValueError(f"Recorded rollout digest differs for {name}")
    expected_cells = {f"{mode}|{tier}" for mode in ("append", "reselect")
                      for tier in ("tight", "mid", "loose")}
    if set(cells) != expected_cells or len(state) != 1494:
        raise ValueError(f"Incomplete evaluation grid for {name}")
    total, total_positions = Counter(), defaultdict(Counter)
    configs = {}
    for cell, counts in sorted(cells.items()):
        official = metrics["configs"][f"{name}|{cell}"]["metrics"]
        if (counts["n_trajectories"] != len(tasks) or counts["n_trajectories"] != official["n_tasks"]
                or counts["n_steps"] != official["n_steps"]
                or sum(counts["first_" + key] for key in CAUSES) != len(tasks)):
            raise ValueError(f"Invalid denominator for {name}/{cell}")
        total.update(counts)
        for step, counter in positions[cell].items():
            total_positions[step].update(counter)
        configs[cell] = {**summary(counts),
                         "by_step": {k: summary(v) for k, v in sorted(positions[cell].items())}}
    return {"source": {"path": str(path), "sha256": digest.hexdigest(),
                       "bytes": path.stat().st_size},
            "combined": {**summary(total),
                         "by_step": {k: summary(v) for k, v in sorted(total_positions.items())}},
            "configs": configs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=("v1", "v2"))
    args = parser.parse_args()
    is_v1 = args.version == "v1"
    heldout = EXTERNAL / "data/traj_train/heldout_tasks.jsonl"
    if sha256(heldout) != "5a6c8ce2c2c738ff2a0e3a15e95d43e8b6ecde7806d7104e5472cf74dddaafac":
        raise RuntimeError("Task 05 heldout input changed")
    with heldout.open("rb") as stream:
        tasks = {row["task"]: row["steps"] for row in map(json.loads, stream) if row["split"] == "heldout"}
    if len(tasks) != 249 or sum(map(len, tasks.values())) != 2326:
        raise RuntimeError("Incomplete saved held-out tasks")
    v1_manifest = V1_RESULTS / "external_artifact_manifest.json"
    if sha256(v1_manifest) != MANIFEST_SHA:
        raise RuntimeError("Task 05 external manifest changed")
    files = {row["path"]: row for row in read_json(v1_manifest)["files"]}
    root = EXTERNAL / ("eval_rollouts" if is_v1 else "eval_rollouts_v2")
    result = {"schema_version": 1, "version": args.version, "definitions": DEFINITIONS,
              "heldout_sha256": sha256(heldout), "task05_external_manifest_sha256": MANIFEST_SHA,
              "strategies": {}}
    names = (*CHECKPOINTS, "base", "no_compress", "latest", "bm25", "oracle") if is_v1 else CHECKPOINTS
    for name in names:
        path, metric = root / name / "steps.jsonl", root / name / "metrics.json"
        if is_v1 and name in CPU_METRIC_SHA:
            metric = V1_RESULTS / f"{name}_results.json"
        checksum = sha256(metric)
        if is_v1 and (CPU_METRIC_SHA[name] if name in CPU_METRIC_SHA else files[str(metric)]["sha256"]) != checksum:
            raise ValueError(f"Task 05 metric checksum differs for {name}")
        official = read_json(metric)
        expected_sha = files[str(path)]["sha256"] if is_v1 else official["steps_artifact"]["sha256"]
        value = summarize_strategy(name, path, official, tasks, expected_sha)
        value["metrics_source"] = {"path": str(metric), "sha256": checksum}
        result["strategies"][name] = value
        print(f'{args.version}/{name}: {value["combined"]["n_steps"]} steps', flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / f"diagnostics_{args.version}.json"
    tmp = destination.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(destination)


if __name__ == "__main__":
    main()
