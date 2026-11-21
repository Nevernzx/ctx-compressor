#!/usr/bin/env python3
"""Task 08 held-out replay on physical GPU 1, with the Task 05/06 evaluator."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import heldout_model_v2 as sft
from scripts import heldout_diagnostics_v2 as diagnostics
from scripts.sft_run_report import validate_strategy
from src.build_traj import render_input
from src.heldout_eval import BUDGETS, INVALID_REASONS, MODES, HeldoutReplay, load_heldout_assets
from src.model_inference import MAX_NEW_TOKENS, SEED, engine_parameters, inference_prompt

v1 = sft.v1
SMALL = PROJECT / "eval/grpo_v2"
FORMAL = v1.EXTERNAL / "grpo_v2"
RAW = v1.EXTERNAL / "eval_rollouts_grpo_v2"
SFT_RESULTS = PROJECT / "eval/sft_run_v2"
SFT_NAME = "checkpoint-2000"
SFT_METRICS_SHA = "131d73ef6bf0e1ca6d9a4f651f3b6ddd850d0c90e6d458d0d1f617d7e54c674b"
SFT_PREFLIGHT_SHA = "9ccfa939d4bc9efa963b59b6d06794f18547cf32257d925c366b56ad842fe9ff"
SFT_ADAPTER_SHA = "7253eae08072782971fa8b7874e97fc3e76dee4ebd11ced5c014147090cf1f9c"
F0 = "merged_sft_v2"
CELL_KEYS = tuple(f"{mode}|{tier}" for mode in MODES for tier in BUDGETS)
STATE_FIELDS = ("questions", "mode", "budget", "kept_ids", "new_ids", "ntok", "pid_by_lid", "gold")
BUCKETS = ("0", "1", "2", "3+")


def load_signed(path, expected):
    if v1.sha256(path) != expected:
        raise ValueError(f"Unexpected reference SHA256: {path}")
    return sft.load_json(path)


def output_step(name, adapter):
    if name == F0 and adapter is None:
        return None
    match = re.fullmatch(r"checkpoint-([1-9]\d*)", name)
    if match and adapter is not None and adapter.name == name:
        return int(match.group(1))
    raise ValueError("F0 must be merged_sft_v2 with no adapter; checkpoints need a matching adapter directory")


def require_environment():
    if PROJECT != Path("/root/workplace/ctx-compressor"):
        raise ValueError("Run on the formal 5880 checkout")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("Task 08 evaluation requires CUDA_VISIBLE_DEVICES=1 and PCI_BUS_ID ordering")
    if Path(sys.prefix).resolve() not in [v1.EXTERNAL / "envs" / name for name in ("reader", "grpo")]:
        raise ValueError("Use the validated local reader or grpo environment")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise ValueError(f"{key}=1 is required")
    for key in ("TMPDIR", "HF_HOME", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR"):
        path = Path(os.environ.get(key, ".")).resolve()
        if not any(path.is_relative_to(root) for root in (FORMAL, v1.EXTERNAL / "grpo_prep/cache")):
            raise ValueError(f"Unexpected cache location for {key}: {path}")


def compare_step0(tasks, corpus, tokenizer, path, expected_hash):
    """Compare saved Task 06 input bytes and actual token IDs before serving any prompt."""
    grid = HeldoutReplay(tasks, corpus, F0).next_batch()
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for item in grid:
            old = json.loads(next(src))
            if old["step"] != 0 or sft.grid_id(item) != sft.grid_id(old):
                raise ValueError("Frozen step-0 evaluation grid changed")
            restored = dict(old["state"])
            restored["ntok"] = {int(i): v for i, v in restored["ntok"].items()}
            restored["text"] = {int(i): corpus[pid]["text"] for i, pid in restored["pid_by_lid"].items()}
            text = render_input(item["state"]).encode()
            prompt = inference_prompt(tokenizer, item["state"])
            saved = old["generation_metadata"]
            if (render_input(restored).encode() != text or hashlib.sha256(text).hexdigest() != saved["user_text_sha256"]
                    or any(prompt[k] != saved[k] for k in ("prompt_token_ids", "prompt_tokens", "prompt_sha256", "user_text_sha256"))):
                raise ValueError(f"Task 06 step-0 prompt differs at {sft.grid_id(item)}")
            digest.update(sft.prompt_digest(sft.grid_id(item), prompt))
    if len(grid) != 1494 or digest.hexdigest() != expected_hash:
        raise ValueError("Incomplete or changed Task 06 step-0 prompts")
    return {"n": len(grid), "user_byte_mismatches": 0, "token_id_mismatches": 0,
            "identity_and_hashes_sha256": digest.hexdigest()}


def observations(row, corpus):
    """Report both all available gold and the current supporting passage used in the failure analysis."""
    c = Counter(n_steps=1)
    before, after = set(row["kept_before"]), set(row["keep_after"])
    new, returned = set(row["new"]), set(row["returned"])
    if row["action_valid"]:
        n_add = len(after - before)
        c["valid"] += 1
        c["n_add"] += n_add
        c["bucket_" + BUCKETS[min(n_add, 3)]] += 1
    elif before != after:
        raise ValueError("Invalid action changed the kept set")
    state = row["state"]
    available = {state["pid_by_lid"][str(i)] for i in state["gold"]}
    for pid in new & available:
        competing = any(other != pid and corpus[other]["title"] == corpus[pid]["title"] for other in returned)
        key = "same" if competing else "different"
        diagnostics.increment(c, key, int(pid in after), 1)
    current = row["gold"]
    if row["action_valid"] and current in new:
        competing = any(other != current and corpus[other]["title"] == corpus[current]["title"] for other in new)
        diagnostics.increment(c, "current_same" if competing else "current_different", int(current in after), 1)
    return c


def raw_comparison(row, old):
    keys = ("mode", "budget_tier", "task", "step", "gold", "returned")
    if any(row[k] != old[k] for k in keys):
        raise ValueError("Unaligned SFT and merged-model retrieval rows")
    gen, reference = row["generation_metadata"], old["generation_metadata"]
    same_prompt = (gen["user_text_sha256"] == reference["user_text_sha256"]
                   and gen["prompt_token_ids"] == reference["prompt_token_ids"])
    same_state = same_prompt and all(row["state"][k] == old["state"][k] for k in STATE_FIELDS)
    if row["step"] == 0 and not same_state:
        raise ValueError("Step-0 SFT and merged states differ")
    changed = row["raw_output"] != old["raw_output"]
    return Counter(same_prompt=int(same_prompt), same_state=int(same_state),
                   raw_changed_same_prompt=int(changed and same_prompt),
                   raw_changed_same_state=int(changed and same_state))


def summarize_extra(c, compare=False):
    if sum(c["bucket_" + k] for k in BUCKETS) != c["valid"]:
        raise ValueError("Incomplete add-count distribution")
    result = {"n_steps": c["n_steps"], "n_valid": c["valid"],
              "mean_added_per_valid": {"sum": c["n_add"], "n": c["valid"],
                                       "mean": c["n_add"]/c["valid"] if c["valid"] else None},
              "add_counts": {k: diagnostics.rate(c["bucket_"+k], c["valid"]) for k in BUCKETS},
              "new_gold_by_title": {k: diagnostics.rate(c[k], c[k+"_n"]) for k in ("same", "different")},
              "current_gold_valid_new_by_title": {k: diagnostics.rate(c["current_"+k], c["current_"+k+"_n"])
                                                  for k in ("same", "different")}}
    if compare:
        result["raw_diff_identical_state"] = diagnostics.rate(c["raw_changed_same_state"], c["same_state"])
        result["raw_diff_identical_model_prompt"] = diagnostics.rate(c["raw_changed_same_prompt"], c["same_prompt"])
    return result


def groups(cells, compare=False):
    if set(cells) != set(CELL_KEYS):
        raise ValueError("Incomplete extra-diagnostics grid")
    return {"configs": {k: summarize_extra(cells[k], compare) for k in CELL_KEYS},
            "modes": {mode: summarize_extra(sum((cells[f"{mode}|{tier}"] for tier in BUDGETS), Counter()), compare)
                      for mode in MODES}, "combined": summarize_extra(sum(cells.values(), Counter()), compare)}


def collect_extra(path, corpus, old_path=None):
    cells = defaultdict(Counter)
    reference_cells = defaultdict(Counter)
    with path.open("rb") as src:
        if old_path is None:
            for line in src:
                r = json.loads(line)
                cells[f'{r["mode"]}|{r["budget_tier"]}'].update(observations(r, corpus))
        else:
            with old_path.open("rb") as previous:
                for line, old_line in zip_longest(src, previous):
                    if line is None or old_line is None:
                        raise ValueError("Unaligned rollout lengths")
                    r, old = json.loads(line), json.loads(old_line)
                    cell = f'{r["mode"]}|{r["budget_tier"]}'
                    cells[cell].update(raw_comparison(r, old))
                    cells[cell].update(observations(r, corpus))
                    reference_cells[cell].update(observations(old, corpus))
    result = groups(cells, compare=old_path is not None)
    result["definitions"] = {"add_counts": "Distinct newly retained PIDs per valid step, including zero additions.",
        "new_gold_by_title": "Task06 eligible gold (gold through this step, newly returned and not previously kept); same = another distinct PID in the original retrieval has exactly its title. Invalid steps included.",
        "current_gold_valid_new_by_title": "Task06 failure analysis comparator: current gold not previously kept, valid action; compare against other visible new PIDs with exactly its title.",
        "raw_diff_identical_state": "Aligned task/mode/tier/step, model-visible user text hash and actual prompt IDs equal, and all input fields (including available gold and candidate PID map) equal."}
    if old_path is not None:
        result["sft_v2_baseline"] = groups(reference_cells)
    return result


def metric_deltas(current, source):
    def compare(lhs, rhs):
        paths = [("final_recall", "mean"), ("evidence_loss", "rate"), ("step_recall", "micro", "mean"),
                 ("step_recall", "task_macro", "mean"), ("occupancy", "mean"), ("cost", "cached", "mean")]
        paths.extend(("invalid_outputs", key, "rate") for key in INVALID_REASONS)
        result = {}
        for parts in paths:
            a,b = lhs, rhs
            for k in parts[:-1]:
                a,b = a[k], b[k]
            if a["n"] != b["n"]:
                raise ValueError(f"Different denominator at {parts}")
            field = parts[-1]
            result[".".join(parts)] = {"n": a["n"], "value": a[field], "sft_v2": b[field],
                                       "delta": a[field] - b[field]}
        return result
    return {"combined": compare(current["combined"], source["combined"]),
            "configs": {k: compare(current["configs"][f'{current["strategy"]}|{k}']["metrics"],
                                   source["configs"][f"{SFT_NAME}|{k}"]["metrics"]) for k in CELL_KEYS}}


def run(args):
    require_environment()
    model = args.model.resolve(strict=True)
    adapter = args.adapter.resolve(strict=True) if args.adapter else None
    step = output_step(args.name, adapter)
    if model != FORMAL / "merged_v2_ckpt2000" or (adapter and not adapter.is_relative_to(FORMAL / "train")):
        raise ValueError("Unexpected Task 08 model/adapter path")
    merge = sft.load_json(model / "merge_record.json")
    if merge["adapter_sha256"] != SFT_ADAPTER_SHA:
        raise ValueError("Merged checkpoint is not the selected Task 06 adapter")
    directory = SMALL / args.name
    raw = RAW / args.name
    monitor_path = RAW / f"{args.name}_resources.jsonl"
    if any(p.exists() or p.is_symlink() for p in (directory, raw, monitor_path)):
        raise FileExistsError("This evaluation output already exists")
    if any(p.resolve() != p for p in (SMALL, RAW)):
        raise ValueError("Evaluation destinations must be dedicated Task 08 directories")
    directory.mkdir(parents=True, exist_ok=False)
    RAW.mkdir(parents=True, exist_ok=True)
    run_file = directory / "run.json"
    started = time.monotonic()
    manifest = {"status": "initializing", "name": args.name, "started_at_utc": v1.utc(), "pid": os.getpid(),
                "gpu": 1, "model": str(model), "adapter": str(adapter) if adapter else None, "strategies": {}}
    # The historical evaluator only uses these module parameters for its output sinks.
    # This entry point has an independent process for each stage, with fresh destinations.
    sft.OUTPUT, sft.RESULTS = RAW, directory
    monitor = sft.Monitor(monitor_path, {})
    llm = None
    try:
        v1.write_json(run_file, manifest)
        tasks, corpus = load_heldout_assets(v1.DATA)
        sft_metrics = load_signed(SFT_RESULTS / "checkpoint-2000_results.json", SFT_METRICS_SHA)
        prev_steps = Path(sft_metrics["steps_artifact"]["path"])
        if v1.sha256(prev_steps) != sft_metrics["steps_artifact"]["sha256"]:
            raise ValueError("Task 06 checkpoint-2000 raw rollout changed")
        proof = load_signed(sft.PREFLIGHT, SFT_PREFLIGHT_SHA)
        for k, v in sft.validate_inputs(tasks).items():
            if json.loads(json.dumps(v)) != proof[k]:
                raise ValueError(f"Frozen evaluation input/setting changed: {k}")
        parameters = engine_parameters(model, sft.load_json(model / "config.json"))
        for k, v in parameters.items():
            if k not in ("model", "tokenizer") and proof["engine_parameters"].get(k) != v:
                raise ValueError(f"Changed engine parameter: {k}")
        proof["engine_parameters"] = parameters
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
        proof["step0"] = compare_step0(tasks, corpus, tokenizer, prev_steps, proof["step0"]["identity_and_hashes_sha256"])
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        gpu = v1.gpu_info()
        if gpu["index"] != 1 or gpu["memory_used_mib"] >= 2048:
            raise RuntimeError(f"GPU 1 gate failed: {gpu}")
        manifest.update(gpu_before=gpu, engine_parameters=parameters, step0=proof["step0"],
                        heldout_input=proof["input_files"][str(v1.DATA / "heldout_tasks.jsonl")],
                        seed=SEED, temperature=0, max_tokens=MAX_NEW_TOKENS, merge_record=v1.file_record(model / "merge_record.json"))
        if adapter:
            manifest["adapter_weight"] = v1.file_record(adapter / "adapter_model.safetensors")
        monitor.thread.start()
        llm = LLM(**parameters)
        manifest["engine_tokenizer_check"] = v1.check_reference(sft.V1_ARGS, llm.get_tokenizer(), tasks, corpus)
        sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, seed=SEED)
        request = LoRARequest(args.name, 1, str(adapter)) if adapter else None
        sft.evaluate_adapter(llm, sampling, request, {"strategy":args.name, "step":step}, tasks, corpus,
                             sft.load_json(v1.RESULTS / "no_compress_results.json"), proof, monitor, manifest, run_file)
        metrics = validate_strategy(sft.load_json(raw / "metrics.json"), args.name)
        steps = raw / "steps.jsonl"
        checks = diagnostics.summarize_strategy(args.name, steps, metrics, {t["task"]:t["steps"] for t in tasks},
                                                metrics["steps_artifact"]["sha256"])
        v1.write_json(directory / "diagnostics.json", {"definitions":diagnostics.DEFINITIONS, **checks})
        extra = collect_extra(steps, corpus, prev_steps if step is None else None)
        for cell in CELL_KEYS:
            nums = extra["configs"][cell]["new_gold_by_title"].values()
            observed = checks["configs"][cell]["new_gold_add"]
            if (sum(v["count"] for v in nums), sum(v["n"] for v in nums)) != (observed["count"], observed["n"]):
                raise ValueError(f"New gold title strata disagree with Task 06 in {cell}")
        if step is None:
            extra["metric_deltas_vs_sft_v2"] = metric_deltas(metrics, sft_metrics)
            source = sft.load_json(SFT_RESULTS / "diagnostics_v1.json")["strategies"]
            extra["cited_rule_baselines"] = {}
            for name in ("bm25", "oracle"):
                frozen = source[name]["source"]
                path = Path(frozen["path"])
                if v1.sha256(path) != frozen["sha256"]:
                    raise ValueError(f"Frozen rule-baseline rollout changed: {name}")
                extra["cited_rule_baselines"][name] = {"source": frozen, **collect_extra(path, corpus)}
        v1.write_json(directory / "extra.json", extra)
        if extra["combined"]["n_steps"] != 13956 or checks["combined"]["n_trajectories"] != 1494:
            raise ValueError("Incomplete held-out episode coverage")
        manifest["checks"] = {"n_steps":13956, "n_trajectories":1494, "n_cells":6, "step0_matched":1494}
        monitor.check()
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc(),
                        oom="out of memory" in str(exc).lower())
        raise
    finally:
        v1.finish_run(manifest, monitor, llm, run_file, started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    run(parser.parse_args())

