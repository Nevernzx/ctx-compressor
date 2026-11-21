#!/usr/bin/env python3
"""CPU-only Task 06 closeout audit, streaming the five raw rollouts from disk."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
OUT = PROJECT / "eval/sft_run_v2"
RUN = EXTERNAL / "runs/sft_20k_v2_20261118"
sys.path.insert(0, str(PROJECT))
from src.heldout_eval import parse_action

CHECKPOINTS = [500, 1000, 1500, 2000, 2500]
RATES = ("new_gold_add", "kept_gold_survival", "all_gold_among_valid_reselect", "all_gold_valid_drop")
CAUSES = ("invalid_output", "new_gold_not_added", "old_gold_dropped", "both", "none")
REASONS = ("invalid_json", "invalid_candidate_id", "append_drop", "over_budget")


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as f:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: f.read(2**20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def require(condition, msg):
    if not condition:
        raise ValueError(msg)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11)


def input_proof():
    previous = load(PROJECT / "eval/sft_run/model_rollout_run.json")
    current = load(OUT / "model_rollout_run.json")
    pre = load(OUT / "heldout_v2_preflight.json")
    for key in ("input_files", "engine_parameters", "generation", "lora_loading"):
        require(current[key] == previous[key], key)
    for name, record in current["input_files"].items():
        require(digest(Path(name)) == record["sha256"], name)
    require(digest(Path(pre["v2_heldout_file"]["path"])) == pre["v2_heldout_file"]["sha256"], "v2 sources")
    selected_fields = pre["projection_fields"]
    tasks = {}
    projected = hashlib.sha256()
    source = EXTERNAL / "data/traj_train/heldout_tasks.jsonl"
    with source.open("rb") as a, Path(pre["v2_heldout_file"]["path"]).open("rb") as b:
        for left, right in zip(a, b, strict=True):
            left, right = json.loads(left), json.loads(right)
            fields = {k: left[k] for k in selected_fields}
            require(fields == {k: right[k] for k in selected_fields} and left["split"] == "heldout", "retrieval")
            require(left["task"] not in tasks, "task ID repeated")
            tasks[left["task"]] = left["steps"]
            projected.update((json.dumps(fields, ensure_ascii=False, separators=(",", ":"))+"\n").encode())
    require(projected.hexdigest() == pre["shared_task_questions_retrieval_sha256"], "recorded projection hash")
    require(len(tasks) == 249 and sum(map(len,tasks.values())) == 2326, "held-out dimensions")
    reference = {}
    path = Path(pre["task05_rollout_reference"]["path"])
    require(digest(path) == pre["task05_rollout_reference"]["sha256"], "reference checksum")
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["step"] != 0:
                break
            m = row["generation_metadata"]
            reference[(row["task"], row["mode"], row["budget_tier"])] = (
                m["prompt_token_ids"], m["user_text_sha256"])
    require(len(reference) == 1494, "reference step zero")
    return tasks, reference, {"task05_heldout_sha256": digest(source), "task06_heldout_sha256":
        pre["v2_heldout_file"]["sha256"], "shared_projection_sha256": projected.hexdigest(),
        "retrieval_tasks": len(tasks), "retrieval_steps": sum(map(len,tasks.values())),
        "step0_prompts_per_checkpoint": len(reference), "shared_engine_generation_and_lora": True}


def apply(raw, state):
    """Recompute priority, execution and order after parsing the raw output."""
    action = parse_action(raw)
    kept, new = state["kept_ids"], state["new_ids"]
    if action is None:
        return "invalid_json", kept
    add, drop = set(action["add"]), set(action.get("drop", []))
    if not add.issubset(new) or not drop.issubset(kept):
        return "invalid_candidate_id", kept
    if state["mode"] == "append" and drop:
        return "append_drop", kept
    after = [i for i in kept if i not in drop] + [i for i in new if i in add]
    if sum(state["ntok"][str(i)] for i in after) > state["budget"]:
        return "over_budget", kept
    return None, after


def check_rates(count, official):
    for key in RATES:
        require((count[key], count[key+"_den"]) == (official[key]["count"], official[key]["n"]), key)
    for key in CAUSES:
        require((count["first_"+key],count["n_trajectories"]) ==
                (official["first_error"][key]["count"],official["first_error"][key]["n"]), "first_"+key)


def aggregate(count, official):
    n, s = count["n_trajectories"], count["steps"]
    require(n == official["n_tasks"] and s == official["n_steps"], "pooled denominators")
    loss = count["loss"]/n
    final = count["final"]/n
    cached = count["cached"]/n
    step = count["step_sum"]/s
    require(count["loss"] == official["evidence_loss"]["count"] and close(loss, official["evidence_loss"]["rate"])
            and close(final, official["final_recall"]["mean"]) and close(step, official["step_recall"]["micro"]["mean"])
            and close(cached, official["cost"]["cached"]["mean"]), "recomputed metrics")
    invalid = {k: count[k] for k in REASONS}
    require(all(v == official["invalid_outputs"][k]["count"] for k,v in invalid.items()), "invalid categories")
    return {"trajectory_n": n, "step_n": s, "final_recall": final, "evidence_loss_count": count["loss"],
            "evidence_loss_rate": loss, "single_step_recall": step, "mean_cached_cost": cached,
            "invalid_counts": invalid, "diagnostics": {k: [count[k],count[k+"_den"]] for k in RATES},
            "first_error_counts": {k:count["first_"+k] for k in CAUSES}}


def audit(step, tasks, reference, diagnostics):
    name = f"checkpoint-{step}"
    directory = EXTERNAL / "eval_rollouts_v2" / name
    official = load(OUT / f"{name}_results.json")
    require(official == load(directory/"metrics.json"), "metrics copy")
    cells, positions = defaultdict(Counter), defaultdict(Counter)
    previous, next_step, failed, seen = {}, Counter(), set(), defaultdict(set)
    sha, n_step0 = hashlib.sha256(), 0
    with (directory/"steps.jsonl").open("rb") as stream:
        for line in stream:
            sha.update(line)
            r = json.loads(line)
            task, mode, tier, pos = r["task"], r["mode"], r["budget_tier"], r["step"]
            key, cell = (task,mode,tier), f"{mode}|{tier}"
            require(r["strategy"] == name and pos == next_step[key] and r["kept_before"] == previous.get(key, []), "continuity")
            saved = tasks[task][pos]
            gold, returned = saved["gold"], saved["new"]
            require(r["gold"] == gold and r["returned"] == returned, "saved retrieval step")
            state, gen = r["state"], r["generation_metadata"]
            require(gen["runtime_prompt_ids_match"] and not gen["input_truncated"] and
                    len(gen["prompt_token_ids"]) == gen["prompt_tokens"] and
                    gen["generated_tokens"] == len(gen["output_token_ids"]) <= 256, "generation bounds")
            require(not r["output_limit_hit"] and not gen["output_limit_hit"], "output length cap")
            if pos == 0:
                require((gen["prompt_token_ids"],gen["user_text_sha256"]) == reference[key], "step0 mismatch")
                n_step0 += 1
            reason, after_ids = apply(r["raw_output"],state)
            ids = state["pid_by_lid"]
            after = [ids[str(i)] for i in after_ids]
            require(reason == r["invalid_reason"] and (reason is None) == r["action_valid"] and
                    after == r["keep_after"] and after_ids == state["keep_after"], "action replay")
            prev = r["kept_before"]
            old_gold = set(prev) & seen[key]
            all_gold_before = len(old_gold) == len(prev)
            seen[key].add(gold)
            available = (set(returned) | set(prev)) & seen[key]
            arrived = (available-set(prev))
            now = set(after)
            recall = len(now&available)/len(available) if available else 1.
            require(recall == r["recall"] and {ids[str(i)] for i in state["gold"]} == available, "step recall")
            c = Counter(steps=1, step_sum=recall)
            if reason is not None:
                c[reason] += 1
            c["new_gold_add"] = len(now&arrived)
            c["new_gold_add_den"] = len(arrived)
            if mode == "reselect":
                c["kept_gold_survival"] = len(now&old_gold)
                c["kept_gold_survival_den"] = len(old_gold)
                if pos and reason is None:
                    c["all_gold_among_valid_reselect_den"] = 1
                    c["all_gold_among_valid_reselect"] = int(all_gold_before)
                    if all_gold_before:
                        c["all_gold_valid_drop_den"] = 1
                        c["all_gold_valid_drop"] = int(bool(set(prev)-now))
            miss, drop = bool(arrived-now), bool(old_gold-now)
            if key not in failed and (reason or miss or drop):
                cause = "invalid_output" if reason else ("both" if miss and drop else
                        "new_gold_not_added" if miss else "old_gold_dropped")
                c["first_"+cause] = 1
                failed.add(key)
            tokens = {ids[k]:v for k,v in state["ntok"].items()}
            retained_prefix_tokens = 0
            for p,q in zip(prev,after):
                if p != q:
                    break
                retained_prefix_tokens += tokens[p]
            used = sum(tokens[p] for p in after)
            require(used == r["used_after"], "size of kept area")
            c["cached"] = 0.05*retained_prefix_tokens + 1.25*(used-retained_prefix_tokens)
            next_step[key] += 1
            previous[key] = after
            if pos + 1 == len(tasks[task]):
                c["n_trajectories"] = 1
                c["loss"] = int(not seen[key].issubset(now))
                c["final"] = len(seen[key]&now)/len(seen[key])
                if key not in failed:
                    c["first_none"] = 1
            cells[cell].update(c)
            c["n_trajectories"] = 1  # per-position denominator counts all trajectories reaching that step
            positions[cell,pos].update(c)
    require(sha.hexdigest() == official["steps_artifact"]["sha256"] and n_step0 == 1494 and
            set(next_step) == set(reference) and all(v==len(tasks[k[0]]) for k,v in next_step.items()), "complete rollout")
    combined = Counter()
    results = {}
    for cell,c in cells.items():
        combined.update(c)
        results[cell] = aggregate(c,official["configs"][name+"|"+cell]["metrics"])
        check_rates(c,diagnostics["configs"][cell])
    for (cell,pos),c in positions.items():
        check_rates(c,diagnostics["configs"][cell]["by_step"][str(pos)])
    check_rates(combined,diagnostics["combined"])
    return {"checkpoint_step": step, "raw_sha256": sha.hexdigest(), "matched_step0_prompts": n_step0,
            "input_truncations": 0, "output_cap_hits": 0, "combined": aggregate(combined,official["combined"]),
            "cells":results, "checked_cell_position_pairs":len(positions)}


def main():
    tasks, reference, proof = input_proof()
    diag = load(OUT/"diagnostics_v2.json")["strategies"]
    rows = [audit(s,tasks,reference,diag[f"checkpoint-{s}"]) for s in CHECKPOINTS]
    best = min(rows,key=lambda r: (r["combined"]["evidence_loss_rate"],-r["combined"]["final_recall"],r["checkpoint_step"]))
    w, s = load(RUN/"worker_result.json"),load(RUN/"continuation_status.json")
    require(s["status"] == "evaluation_complete" and w["status"] == "ok" and not w["oom"] and not w["nonfinite"]
            and [r["step"] for r in w["loss_curve"]] == list(range(1,2501)) and
            [r["step"] for r in w["eval_curve"]] == CHECKPOINTS and w["epoch"] == 2 and w["consumed_samples"]==40000,
            "training completeness")
    require(all(math.isfinite(v) for r in w["loss_curve"]+w["eval_curve"] for v in r.values() if isinstance(v,float)),
            "finite metrics")
    m = load(OUT/"model_rollout_run.json")
    require(m["status"]=="complete" and m["engine_shutdown"]["completed"] and m["monitor_thread_stopped"]
            and w["initial_gpu"]["index"] == 1 and m["cuda_visible_devices"]=="1", "resource completion")
    for checkpoint in m["checkpoint_inputs"]:
        require(all(digest(Path(f["path"])) == f["sha256"] for f in checkpoint["files"]), "adapter identity")
    result = {"status":"ok","source":"CPU replay of original completion strings, saved returns and state IDs",
              "inputs":proof,"checkpoints":rows,"selected_checkpoint":best["checkpoint_step"],
              "n_steps_all":sum(r["combined"]["step_n"] for r in rows),"training":{
                  "optimizer_steps":w["optimizer_steps"],"consumed_samples":w["consumed_samples"],"epoch":w["epoch"],
                  "finite_training_and_eval_logs":True,"oom":w["oom"],"nonfinite":w["nonfinite"],
                  "wait_seconds":s["wait_seconds"],"train_seconds":w["train_seconds"],
                  "train_process_wall_seconds":s["train_wall_seconds"],"evaluation_wall_seconds":s["eval_wall_seconds"]}}
    (OUT/"final_rollout_audit_v2.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(result["n_steps_all"],result["selected_checkpoint"])


if __name__ == "__main__":
    main()
