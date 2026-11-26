#!/usr/bin/env python3
"""Evaluate one Task 10 adapter using the Task 06 greedy replay, or select across the completed grid."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_formal_grpo as formal
from scripts import sft_v3_config as cfg
from scripts.sft_run_report import validate_strategy
from src.heldout_eval import load_heldout_assets, select_checkpoint
from src.model_inference import MAX_NEW_TOKENS, SEED

sft, v1, diag = formal.sft, formal.v1, formal.diagnostics
RAW = cfg.ROOT / "eval_rollouts_v3"
HELDOUT_SHA = "5a6c8ce2c2c738ff2a0e3a15e95d43e8b6ecde7806d7104e5472cf74dddaafac"
CUSTOM = ("multiadd_contains_current_gold", "multiadd_fragment_gold", "drop_precision",
          "claude_nongold_kept_drop_old_gold", "claude_allgold_kept_drop_old_gold")
DEFS = {
    "multiadd_contains_current_gold": "Valid steps with >=2 distinct added PIDs: fraction whose additions contain this step's supporting PID.",
    "multiadd_fragment_gold": "Among all distinct fragments added on eligible >=2-add steps, fraction revealed as gold by the current step.",
    "drop_precision": "Fragment-weighted across valid actions: drops not known as gold through the current step / all drops.",
    "claude_nongold_kept_drop_old_gold": "Valid reselect at step>=1; kept_before includes >=1 PID outside previously revealed gold. Count steps dropping any previously revealed and kept gold.",
    "claude_allgold_kept_drop_old_gold": "Same valid reselect population, with kept_before entirely contained in previously revealed gold (including empty).",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def extra_counts(row, known_before):
    counts = Counter()
    if not row["action_valid"]:
        return counts
    before, after = set(row["kept_before"]), set(row["keep_after"])
    added, dropped = after - before, before - after
    known = known_before | {row["gold"]}
    if len(added) >= 2:
        diag.increment(counts, "multiadd_contains_current_gold", int(row["gold"] in added), 1)
        diag.increment(counts, "multiadd_fragment_gold", len(added & known), len(added))
    diag.increment(counts, "drop_precision", len(dropped-known), len(dropped))
    if row["step"] >= 1 and row["mode"] == "reselect":
        key = "claude_nongold_kept_drop_old_gold" if before-known_before else "claude_allgold_kept_drop_old_gold"
        diag.increment(counts, key, int(bool(dropped & known_before)), 1)
    return counts


def attach_extra(path, metrics, diagnostic, corpus):
    extra = formal.collect_extra(path, corpus)
    cells = defaultdict(Counter)
    by_trajectory = defaultdict(set)
    with path.open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            cell = f"{row['mode']}|{row['budget_tier']}"
            known = by_trajectory[row["trajectory_id"]]
            cells[cell].update(extra_counts(row, known))
            known.add(row["gold"])
    for cell in (*formal.CELL_KEYS, "combined"):
        if cell == "combined":
            value, checks, m = extra["combined"], diagnostic["combined"], metrics["combined"]
            counts = sum(cells.values(), Counter())
        else:
            value, checks = extra["configs"][cell], diagnostic["configs"][cell]
            m = metrics["configs"][f"{metrics['strategy']}|{cell}"]["metrics"]
            counts = cells[cell]
        value.update({k: diag.rate(counts[k], counts[k+"_n"]) for k in CUSTOM})
        value.update(kept_gold_survival=checks["kept_gold_survival"], first_error=checks["first_error"],
                     occupancy=m["occupancy"], cached_cost=m["cost"]["cached"])
        title = value["new_gold_by_title"].values()
        require((sum(x["count"] for x in title), sum(x["n"] for x in title)) ==
                (checks["new_gold_add"]["count"], checks["new_gold_add"]["n"]), f"title strata mismatch: {cell}")
        require(value["multiadd_contains_current_gold"]["n"] ==
                value["add_counts"]["2"]["count"]+value["add_counts"]["3+"]["count"], f"multi-add count mismatch: {cell}")
        require(value["claude_allgold_kept_drop_old_gold"] == checks["all_gold_valid_drop"],
                f"all-gold Claude stratum mismatch: {cell}")
        total = checks["all_gold_among_valid_reselect"]
        require(value["claude_nongold_kept_drop_old_gold"]["n"] == total["n"] - total["count"],
                f"non-gold Claude stratum mismatch: {cell}")
    extra["definitions"].update(DEFS, kept_gold_survival=diag.DEFINITIONS["kept_gold_survival"],
                                first_error=diag.DEFINITIONS["first_error"])
    return extra


def check_adapter(step):
    adapter = cfg.checkpoint(step)
    require(adapter.resolve() == adapter and adapter.is_relative_to(cfg.RUN), f"unexpected checkpoint path {adapter}")
    config = sft.load_json(adapter / "adapter_config.json")
    expected = {"base_model_name_or_path": str(v1.MODEL), "peft_type": "LORA",
                "r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
    require(all(config.get(k) == v for k, v in expected.items()), f"adapter/base mismatch {adapter}")
    require(sft.load_json(adapter / "trainer_state.json")["global_step"] == step, f"wrong trainer step {adapter}")
    weight = v1.file_record(adapter / "adapter_model.safetensors")
    require(weight["bytes"] > 0, f"empty adapter {adapter}")
    return adapter, weight


def frozen_inputs():
    require(v1.sha256(v1.DATA / "heldout_tasks.jsonl") == HELDOUT_SHA, "changed Task 05 held-out input")
    tasks, corpus = load_heldout_assets(v1.DATA)
    proof = formal.load_signed(sft.PREFLIGHT, formal.SFT_PREFLIGHT_SHA)
    for k, value in sft.validate_inputs(tasks).items():
        require(json.loads(json.dumps(value)) == proof[k], f"Task 06 preflight drift: {k}")
    previous = formal.load_signed(formal.SFT_RESULTS / "checkpoint-2000_results.json", formal.SFT_METRICS_SHA)
    source = previous["steps_artifact"]
    ref = Path(source["path"])
    require(v1.sha256(ref) == source["sha256"], "Task 06 step-0 reference changed")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(v1.MODEL), local_files_only=True, trust_remote_code=False)
    step0 = formal.compare_step0(tasks, corpus, tokenizer, ref, proof["step0"]["identity_and_hashes_sha256"])
    return tasks, corpus, proof, step0


def run(step):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
            "GPU 1 visibility and PCI_BUS_ID required")
    require(Path(sys.prefix).resolve() == cfg.ROOT / "envs/reader", "use the Task 06 reader environment")
    require(all(os.environ.get(k) == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")),
            "offline operation required")
    name = f"checkpoint-{step}"
    adapter, weight = check_adapter(step)
    directory = cfg.OUT / name
    raw = RAW / name
    resources = RAW / f"{name}_resources.jsonl"
    require(all(p.resolve() == p and not p.exists() for p in (directory, raw, resources)), "existing evaluation output")
    directory.mkdir(parents=True)
    RAW.mkdir(parents=True, exist_ok=True)
    tasks, corpus, proof, step0 = frozen_inputs()
    monitor = sft.Monitor(resources, {})
    started = time.monotonic()
    receipt = directory / "run.json"
    manifest = {"status": "initializing", "name": name, "checkpoint_step": step, "started_at_utc": v1.utc(),
                "pid": os.getpid(), "gpu": 1, "model": str(v1.MODEL), "adapter": str(adapter),
                "adapter_weight": weight, "heldout_input": proof["input_files"][str(v1.DATA / "heldout_tasks.jsonl")],
                "engine_parameters": proof["engine_parameters"], "step0": step0, "seed": SEED, "temperature": 0,
                "max_tokens": MAX_NEW_TOKENS, "greedy": True, "hedge": False, "strategies": {}}
    sft.OUTPUT, sft.RESULTS = RAW, directory
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = None
    try:
        v1.write_json(receipt, manifest)
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        gpu = v1.gpu_info()
        require(gpu["index"] == 1 and gpu["memory_used_mib"] < 2048, f"GPU 1 startup gate: {gpu}")
        manifest["gpu_before"] = gpu
        monitor.thread.start()
        llm = LLM(**proof["engine_parameters"])
        manifest["engine_tokenizer_check"] = v1.check_reference(sft.V1_ARGS, llm.get_tokenizer(), tasks, corpus)
        sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, seed=SEED)
        request = LoRARequest(name, 1, str(adapter))
        sft.evaluate_adapter(llm, sampling, request, {"strategy": name, "step": step}, tasks, corpus,
                             sft.load_json(v1.RESULTS / "no_compress_results.json"), proof, monitor, manifest, receipt)
        metrics = validate_strategy(sft.load_json(raw / "metrics.json"), name)
        steps = raw / "steps.jsonl"
        diagnostic = diag.summarize_strategy(name, steps, metrics, {t["task"]: t["steps"] for t in tasks},
                                             metrics["steps_artifact"]["sha256"])
        extra = attach_extra(steps, metrics, diagnostic, corpus)
        require(extra["combined"]["n_steps"] == 13956 and diagnostic["combined"]["n_trajectories"] == 1494,
                "incomplete held-out coverage")
        v1.write_json(directory / "diagnostics.json", {"definitions": diag.DEFINITIONS, **diagnostic})
        v1.write_json(directory / "extra.json", extra)
        manifest["checks"] = {"n_steps": 13956, "n_trajectories": 1494, "n_cells": 6, "step0_matched": 1494}
        manifest["artifacts"] = {p.name: v1.file_record(p) for p in (
            directory / f"{name}_results.json", directory / "diagnostics.json", directory / "extra.json")}
        monitor.check()
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                        oom="out of memory" in str(error).lower())
        raise
    finally:
        v1.finish_run(manifest, monitor, llm, receipt, started)


def select():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "checkpoint selection must be CPU-only")
    path = cfg.OUT / "selected_checkpoint.json"
    require(not path.exists() and not path.is_symlink(), "existing checkpoint selection")
    metrics = []
    candidates = []
    for step in cfg.EVAL:
        name = f"checkpoint-{step}"
        directory = cfg.OUT / name
        receipt = sft.load_json(directory / "run.json")
        adapter, weight = check_adapter(step)
        require(receipt["status"] == "complete" and receipt["checkpoint_step"] == step
                and receipt["adapter"] == str(adapter) and receipt["adapter_weight"] == weight
                and receipt["engine_shutdown"]["completed"] and receipt["step0"]["n"] == 1494,
                f"incomplete evaluation receipt {name}")
        for filename in (f"{name}_results.json", "diagnostics.json", "extra.json"):
            require(v1.file_record(directory / filename) == receipt["artifacts"][filename], f"artifact changed {filename}")
        m = validate_strategy(sft.load_json(directory / f"{name}_results.json"), name)
        require(m["combined"]["n_steps"] == 13956 and v1.file_record(RAW / name / "steps.jsonl") == m["steps_artifact"],
                f"rollout coverage or checksum changed {name}")
        metrics.append(m)
        candidates.append({"strategy": name, "checkpoint_step": step, "path": str(adapter), "adapter_sha256": weight["sha256"],
                           "evidence_loss": m["combined"]["evidence_loss"], "final_recall": m["combined"]["final_recall"]})
    winner = select_checkpoint(metrics)
    selected = next(c for c in candidates if c["checkpoint_step"] == winner["checkpoint_step"])
    result = {**selected, "rule": "lowest pooled evidence-loss rate, highest pooled final recall, earliest step",
              "n_checkpoints": len(candidates), "n_trajectories_per_checkpoint": 1494, "eval_loss_used": False,
              "candidates": candidates, "heldout_sha256": HELDOUT_SHA}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(selected), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--step", type=int, choices=cfg.EVAL)
    commands.add_argument("--select", action="store_true")
    args = parser.parse_args()
    if args.select:
        select()
    else:
        run(args.step)

