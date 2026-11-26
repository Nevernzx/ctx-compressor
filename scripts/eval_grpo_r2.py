"""Evaluate the merged round-2 start or its trained LoRA with the Task 10 E replay."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import eval_sft_v3 as v3
from scripts.build_grpo_r2_states import RUN, SMALL, start_model
from scripts.sft_run_report import validate_strategy
from src.grpo_reward_v2 import score_completion_v2
from src.model_inference import engine_parameters, MAX_NEW_TOKENS, SEED

formal, sft, v1 = v3.formal, v3.sft, v3.v1
RAW = v3.cfg.ROOT / "eval_rollouts_grpo_r2"
MERGED = RUN / "merged_start"
require = v3.require


def reward_observation(row):
    score = score_completion_v2(row["raw_output"], row["state"], .5)
    require(score["keep_after"] == row["state"]["keep_after"]
            and score["used_after"] == row["used_after"] and score["invalid_reason"] == row["invalid_reason"],
            "held-out v2 scorer and closed-loop evaluator disagree")
    # Sum in integral 1/2000 reward units for a stable, exactly reproducible pooled mean.
    return Counter(n=1, n_lost=score["n_lost"], used=score["used_after"], invalid=score["invalid_penalty"],
                   cost_units=round(-score["reward"] * 2000))


def mean(total, n):
    return {"sum":total, "n":n, "mean":total/n if n else None}


def objective(c):
    n = c["n"]
    return {"mu":.5, "n_steps":n, "reward":mean(-c["cost_units"]/2000, n),
            "n_lost":mean(c["n_lost"], n), "used_per_1000":mean(c["used"]/1000, n),
            "invalid":v3.diag.rate(c["invalid"], n)}


def augment_objective(extra, path, previous=None):
    cells, prior_cells, paired = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    with path.open("rb") as stream, (previous.open("rb") if previous else nullcontext()) as ref:
        rows = zip_longest(stream,ref) if ref else ((line,None) for line in stream)
        for line, old_line in rows:
            require(line is not None and (ref is None or old_line is not None), "unaligned start and F0 lengths")
            row = json.loads(line)
            cell = f"{row['mode']}|{row['budget_tier']}"
            cells[cell].update(reward_observation(row))
            if old_line is not None:
                old = json.loads(old_line)
                paired[cell].update(formal.raw_comparison(row, old))
                prior_cells[cell].update(reward_observation(old))
    require(set(cells) == set(extra["configs"]), "held-out reward grid differs from extra diagnostics")
    for cell in (*cells, "combined"):
        entry = extra["combined"] if cell == "combined" else extra["configs"][cell]
        counts = sum(cells.values(),Counter()) if cell == "combined" else cells[cell]
        require(counts["n"] == entry["n_steps"], "held-out reward denominator differs")
        entry["heldout_reward_v2"] = objective(counts)
        if previous:
            source = sum(prior_cells.values(),Counter()) if cell == "combined" else prior_cells[cell]
            pairs = sum(paired.values(),Counter()) if cell == "combined" else paired[cell]
            baseline = objective(source)
            delta = {k:entry["heldout_reward_v2"][k]["mean"]-baseline[k]["mean"]
                     for k in ("reward", "n_lost", "used_per_1000")}
            delta["invalid_rate"] = entry["heldout_reward_v2"]["invalid"]["rate"] - baseline["invalid"]["rate"]
            entry["start_reward_v2"] = baseline
            entry["objective_deltas_vs_start"] = {"n_steps":counts["n"], **delta}
            entry["raw_diff_identical_state"] = v3.diag.rate(pairs["raw_changed_same_state"],pairs["same_state"])
            entry["raw_diff_identical_model_prompt"] = v3.diag.rate(pairs["raw_changed_same_prompt"],pairs["same_prompt"])
    return extra


def start_baseline(start):
    name = start["strategy"]
    expected = ((v3.cfg.OUT/name if start["version"] == "v3" else formal.SFT_RESULTS) / f"{name}_results.json")
    require(str(expected) == start["metrics_path"], "unexpected SFT start metrics path")
    source = validate_strategy(sft.load_json(expected), name)
    raw = Path(start["steps_path"])
    raw_root = v3.RAW if start["version"] == "v3" else v1.EXTERNAL/"eval_rollouts_v2"
    require(raw == raw_root/name/"steps.jsonl" and source["steps_artifact"] == v1.file_record(raw),
            "SFT start rollout bytes changed")
    return source, raw


def model_files(name, explicit=None):
    if name == "merged_start":
        require(explicit is None, "merged-only F0 cannot load an adapter")
        return None, None
    match = re.fullmatch(r"checkpoint-([1-9]\d*)", name)
    require(match is not None, "invalid evaluation name")
    step = int(match.group(1))
    adapter = RUN/"train"/name
    require((explicit is None or explicit == adapter) and adapter == adapter.resolve()
            and adapter.is_dir(), "unexpected GRPO adapter path")
    trainer = sft.load_json(adapter/"trainer_state.json")
    config = sft.load_json(adapter/"adapter_config.json")
    require(trainer["global_step"] == step and config["base_model_name_or_path"] == str(MERGED)
            and config["peft_type"] == "LORA" and config["r"] == 16, "checkpoint/base provenance mismatch")
    return step, adapter


def metric_deltas(current, source):
    # The existing Task 08 implementation indexes its baseline by a module-level strategy name.
    with patch.object(formal,"SFT_NAME",source["strategy"]):
        result = formal.metric_deltas(current, source)
    for table in [result["combined"], *result["configs"].values()]:
        for item in table.values():
            item["start"] = item.pop("sft_v2")
    return result


def step0_for_task10(tasks, corpus, tokenizer, proof):
    selected = sft.load_json(v3.cfg.OUT / "selected_checkpoint.json")
    require(selected["checkpoint_step"] in v3.cfg.EVAL and selected["strategy"] == f"checkpoint-{selected['checkpoint_step']}",
            "Task 10 selected reference not valid")
    name = selected["strategy"]
    metrics = sft.load_json(v3.cfg.OUT / name / f"{name}_results.json")
    ref = v3.RAW / name / "steps.jsonl"
    require(metrics["steps_artifact"] == v1.file_record(ref), "Task 10 step-0 reference changed")
    return formal.compare_step0(tasks, corpus, tokenizer, ref, proof["step0"]["identity_and_hashes_sha256"])


def run(name, explicit=None):
    require(Path(sys.prefix).resolve() == v3.cfg.ROOT / "envs/reader" and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
            and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "use the Task 10 reader environment on physical GPU 1")
    require(all(os.environ.get(k) == "1" for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_DATASETS_OFFLINE")),
            "offline evaluation required")
    start = start_model()
    merge = sft.load_json(MERGED/"merge_record.json")
    require(merge["base"] == str(v1.MODEL) and merge["adapter"] == start["path"] and
            merge["adapter_sha256"] == start["adapter_sha256"], "incorrect merged starting policy")
    step, adapter = model_files(name, explicit)
    source, source_raw = start_baseline(start)
    directory, raw = SMALL/name, RAW/name
    monitor_file = RAW/f"{name}_resources.jsonl"
    require(all(p == p.resolve() and not os.path.lexists(p) for p in (directory,raw,monitor_file)),
            "existing evaluation outputs or redirected destination")
    directory.mkdir(parents=True)
    RAW.mkdir(parents=True,exist_ok=True)
    tasks, corpus, proof, _ = v3.frozen_inputs()
    parameters = engine_parameters(MERGED, sft.load_json(MERGED/"config.json"))
    require(parameters == {**proof["engine_parameters"], "model":str(MERGED),"tokenizer":str(MERGED)},
            "engine settings changed from Task 10")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MERGED), local_files_only=True, trust_remote_code=False)
    step0 = step0_for_task10(tasks,corpus,tokenizer,proof)
    proof = {**proof,"engine_parameters":parameters,"step0":step0}
    monitor, started = sft.Monitor(monitor_file, {}), time.monotonic()
    receipt = directory/"run.json"
    manifest = {"status":"initializing","name":name,"checkpoint_step":step,"started_at_utc":v1.utc(),"pid":os.getpid(),
                "model":str(MERGED),"adapter":str(adapter) if adapter else None,
                "adapter_weight":v1.file_record(adapter/"adapter_model.safetensors") if adapter else None,
                "start_version":start["version"],"start_strategy":start["strategy"],"start_adapter_sha256":start["adapter_sha256"],
                "merge_record":v1.file_record(MERGED/"merge_record.json"),"heldout_input":proof["input_files"][str(v1.DATA/"heldout_tasks.jsonl")],
                "gpu":1,"seed":SEED,"temperature":0,"max_tokens":MAX_NEW_TOKENS,"step0":step0,"greedy":True,
                "hedge":False,"engine_parameters":parameters,"strategies":{}}
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = None
    try:
        v1.write_json(receipt, manifest)
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        gpu = v1.gpu_info()
        require(gpu["index"] == 1 and gpu["memory_used_mib"] < 2048, "GPU 1 startup gate")
        manifest["gpu_before"] = gpu
        monitor.thread.start()
        llm = LLM(**parameters)
        manifest["engine_tokenizer_check"] = v1.check_reference(sft.V1_ARGS,llm.get_tokenizer(),tasks,corpus)
        request = LoRARequest(name,1,str(adapter)) if adapter else None
        sampling = SamplingParams(temperature=0,max_tokens=MAX_NEW_TOKENS,seed=SEED)
        with patch.multiple(sft, OUTPUT=RAW, RESULTS=directory):
            sft.evaluate_adapter(llm,sampling,request,{"strategy":name,"step":step},tasks,corpus,
                                 sft.load_json(v1.RESULTS/"no_compress_results.json"),proof,monitor,manifest,receipt)
        metrics = validate_strategy(sft.load_json(raw/"metrics.json"),name)
        steps = raw/"steps.jsonl"
        diagnostic = v3.diag.summarize_strategy(name,steps,metrics,{t["task"]:t["steps"] for t in tasks},
                                                metrics["steps_artifact"]["sha256"])
        extra = v3.attach_extra(steps,metrics,diagnostic,corpus)
        augment_objective(extra,steps, source_raw if step is None else None)
        if step is None:
            extra["metric_deltas_vs_start"] = metric_deltas(metrics,source)
            extra["start_reference"] = {"version":start["version"],"strategy":start["strategy"],
                                        "metrics":v1.file_record(Path(start["metrics_path"])),"steps":source["steps_artifact"]}
        require(extra["combined"]["n_steps"] == 13956 and diagnostic["combined"]["n_trajectories"] == 1494,
                "incomplete evaluation grid")
        v1.write_json(directory/"extra.json",extra)
        v1.write_json(directory/"diagnostics.json",{"definitions":v3.diag.DEFINITIONS,**diagnostic})
        manifest["artifacts"] = {p.name:v1.file_record(p) for p in (
            directory/f"{name}_results.json",directory/"extra.json",directory/"diagnostics.json")}
        manifest["checks"] = {"n_steps":13956,"n_trajectories":1494,"n_cells":6,"step0_matched":step0["n"]}
        monitor.check()
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed",error=repr(exc),traceback=traceback.format_exc(),oom="out of memory" in str(exc).lower())
        raise
    finally:
        v1.finish_run(manifest,monitor,llm,receipt,started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    run(args.name,args.adapter)

