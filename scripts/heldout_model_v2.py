#!/usr/bin/env python3
"""Task 06: certify the fixed heldout inputs and replay five v2 LoRA checkpoints."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import rollout_model as v1
from src.build_traj import render_input
from src.heldout_eval import HeldoutReplay, load_heldout_assets
from src.model_inference import (MAX_NEW_TOKENS, SEED, compact, engine_parameters, generation_record,
                                 guard_prompt, inference_prompt)

RESULTS = v1.PROJECT / "eval/sft_run_v2"
OUTPUT = v1.EXTERNAL / "eval_rollouts_v2"
TRAIN = v1.EXTERNAL / "runs/sft_20k_v2_20261118/checkpoints"
V2_TASKS = v1.EXTERNAL / "data/traj_train_v2/heldout_tasks.jsonl"
PREFLIGHT = RESULTS / "heldout_v2_preflight.json"
V1_ARGS = SimpleNamespace(results=v1.RESULTS, data_dir=v1.DATA, model=v1.MODEL)
V1_RUN_SHA = "a9e82afc0107bc88a78ecda14b543e51e2fe1942fd6338188cdaaa4452e55efa"
V1_STEP_FILE_SHA = "f794336249c1b02c8285d9d3f93e39c76006e38e59b8d88719911e32c699b4d5"
TASK_KEYS = ("task", "split", "example_ids", "questions", "steps")


def load_json(path):
    return json.loads(path.read_bytes())


def certified_v1_run():
    path = v1.RESULTS / "model_rollout_run.json"
    if v1.sha256(path) != V1_RUN_SHA:
        raise ValueError("Task 05 inference receipt has changed")
    return load_json(path)


def validate_inputs(tasks):
    original = certified_v1_run()
    inputs = v1.input_records(v1.DATA, v1.MODEL)
    if inputs != original["input_files"]:
        raise ValueError("Task 05 heldout inputs, tokenizer or model configuration changed")
    if (v1.file_record(v1.RESULTS / "no_compress_results.json") != original["no_compress_baseline"]
            or v1.file_record(v1.RESULTS / "model_template_reference.json") != original["template_check"]["reference"]):
        raise ValueError("Task 05 baseline or training-template proof changed")
    with V2_TASKS.open("rb") as stream:
        digest = hashlib.sha256()
        n = 0
        for n, line in enumerate(stream, start=1):
            row = json.loads(line)
            projection = {k: row[k] for k in TASK_KEYS}
            if n > len(tasks) or projection != tasks[n-1]:
                raise ValueError(f"Task 06 saved heldout questions or retrieval steps differ at row {n}")
            digest.update((compact(projection) + "\n").encode())
    if n != 249 or n != len(tasks) or sum(len(t["steps"]) for t in tasks) != 2326:
        raise ValueError("Incomplete heldout retrieval")
    parameters = engine_parameters(v1.MODEL, load_json(v1.MODEL / "config.json"))
    if parameters != original["engine_parameters"]:
        raise ValueError("vLLM settings changed since Task 05")
    return {"input_files": inputs, "engine_parameters": parameters,
            "v2_heldout_file": v1.file_record(V2_TASKS),
            "shared_task_questions_retrieval_sha256": digest.hexdigest(),
            "projection_fields": TASK_KEYS, "n_tasks": n, "n_saved_steps": 2326}


def prompt_digest(key, prompt):
    return (compact([key, prompt["user_text_sha256"], prompt["prompt_sha256"]]) + "\n").encode()


def grid_id(item):
    return [item["mode"], item["budget_tier"], item["task"]]


def preflight():
    from transformers import AutoTokenizer
    import torch
    tasks, corpus = load_heldout_assets(v1.DATA)
    proof = validate_inputs(tasks)
    tokenizer = AutoTokenizer.from_pretrained(str(v1.MODEL), local_files_only=True, trust_remote_code=False)
    proof["training_template_reference"] = v1.check_reference(V1_ARGS, tokenizer, tasks, corpus)
    path = v1.EXTERNAL / "eval_rollouts/checkpoint-2000/steps.jsonl"
    if v1.sha256(path) != V1_STEP_FILE_SHA:
        raise ValueError("Task 05 saved prompt/rollout checksum changed")
    # Recorded outputs store the token IDs and UTF-8 prompt SHA. Restore the actual
    # user text from stored state and the pinned corpus to compare bytes as well.
    grid = HeldoutReplay(tasks, corpus, "checkpoint-500").next_batch()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for item in grid:
            row = json.loads(next(stream))
            state = dict(row["state"])
            state["ntok"] = {int(k): v for k, v in state["ntok"].items()}
            state["text"] = {int(k): corpus[pid]["text"] for k, pid in state["pid_by_lid"].items()}
            old_text = render_input(state).encode("utf-8")
            new_text = render_input(item["state"]).encode("utf-8")
            prompt, saved = inference_prompt(tokenizer, item["state"]), row["generation_metadata"]
            if (row["step"] != 0 or item["step"] != 0 or grid_id(item) != grid_id(row)
                    or old_text != new_text or hashlib.sha256(old_text).hexdigest() != saved["user_text_sha256"]
                    or any(prompt[k] != saved[k] for k in ("prompt_token_ids", "prompt_tokens",
                                                         "prompt_sha256", "user_text_sha256"))):
                raise ValueError(f"Task 05 and v2 step-0 input differs for {grid_id(item)}")
            h.update(prompt_digest(grid_id(item), prompt))
    if len(grid) != 1494 or torch.cuda.is_initialized():
        raise ValueError("Incomplete or non-CPU preflight")
    proof.update(status="ok", created_utc=v1.utc(), task05_run_sha256=V1_RUN_SHA,
                 task05_rollout_reference={"path": str(path), "sha256": V1_STEP_FILE_SHA},
                 step0={"n": len(grid), "user_utf8_byte_identical": True,
                        "all_token_ids_identical": True, "identity_and_hashes_sha256": h.hexdigest()},
                 cuda_initialized=False, expected_outputs_per_strategy=13956,
                 expected_strategies=[f"checkpoint-{n}" for n in v1.STEPS])
    v1.write_json(PREFLIGHT, proof)
    print(compact({"preflight": str(PREFLIGHT), "step0": proof["step0"]}), flush=True)


class Monitor(v1.Monitor):
    """Keep the Task 05 sampling and cleanup methods; raw df is observational in Task 06."""
    def record_disk(self, disk, *, phase):
        self.root_samples += 1
        return {"utc": v1.utc(), "phase": phase, "root_df": disk}

    def diagnostics(self):
        return {"policy": "raw df only; Task 06 accounts for task-written overlay files separately",
                "n_disk_samples": self.root_samples}


def evaluate_adapter(llm, sampling, adapter, info, tasks, corpus, baseline, proof, monitor, manifest, run_file):
    name, step = info["strategy"], info["step"]
    directory = OUTPUT / name
    directory.mkdir()
    replay = HeldoutReplay(tasks, corpus, name)
    tokenizer = llm.get_tokenizer()
    progress = {"status": "running", "checkpoint_step": step, "n_steps": 0, "n_trajectories": 1494,
                "max_prompt_tokens": 0, "output_limit_hits": 0, "invalid_counts": Counter(), "batches": []}
    started = time.monotonic()
    manifest.update(status="running", active_strategy=name)
    manifest["strategies"][name] = progress
    v1.write_json(run_file, manifest)
    with (directory / "steps.jsonl").open("x", encoding="utf-8") as stream:
        while not replay.completed():
            monitor.check()
            items = replay.next_batch()
            prompts = [inference_prompt(tokenizer, item["state"]) for item in items]
            for item, prompt in zip(items, prompts):
                guard_prompt(prompt, proof["engine_parameters"]["max_model_len"],
                             identity=(item["trajectory_id"], item["step"]))
            position = items[0]["step"]
            if position == 0:
                h = hashlib.sha256()
                for item, prompt in zip(items, prompts):
                    h.update(prompt_digest(grid_id(item), prompt))
                if len(items) != 1494 or h.hexdigest() != proof["step0"]["identity_and_hashes_sha256"]:
                    raise ValueError(f"Actual v2 engine step-0 input differs from Task 05 for {name}")
                progress["matched_task05_step0_inputs"] = len(items)
            start_batch = time.monotonic()
            outputs = llm.generate([{"prompt_token_ids": p["prompt_token_ids"]} for p in prompts],
                                   sampling_params=sampling, lora_request=adapter, use_tqdm=False)
            if len(outputs) != len(items):
                raise ValueError(f"Incomplete vLLM batch for {name}, step {position}")
            generation = [generation_record(p, out) for p, out in zip(prompts, outputs)]
            records = replay.advance_batch(generation)
            for row in records:
                stream.write(compact(row) + "\n")
                progress["output_limit_hits"] += int(row["output_limit_hit"])
                if row["invalid_reason"]:
                    progress["invalid_counts"][row["invalid_reason"]] += 1
            stream.flush()
            os.fsync(stream.fileno())
            progress["n_steps"] += len(items)
            progress["max_prompt_tokens"] = max(progress["max_prompt_tokens"], max(p["prompt_tokens"] for p in prompts))
            progress["batches"].append({"step": position, "n": len(items),
                                        "seconds": time.monotonic()-start_batch,
                                        "prompt_tokens": sum(p["prompt_tokens"] for p in prompts),
                                        "generated_tokens": sum(x["generated_tokens"] for x in generation)})
            v1.write_json(run_file, manifest)
            print(f'{v1.utc()} {name} step={position} outputs={progress["n_steps"]}', flush=True)
    if progress["n_steps"] != proof["expected_outputs_per_strategy"]:
        raise ValueError(f"Incomplete checkpoint {name}")
    metrics = replay.aggregate(baseline=baseline)
    metrics.update(checkpoint_step=step, runtime={"elapsed_seconds": time.monotonic()-started,
                   "max_prompt_tokens": progress["max_prompt_tokens"], "input_truncations": 0},
                   steps_artifact=v1.file_record(directory / "steps.jsonl"))
    v1.write_json(directory / "metrics.json", metrics)
    v1.write_json(RESULTS / f"{name}_results.json", metrics)
    progress.update(status="complete", metrics=v1.file_record(directory / "metrics.json"))
    v1.write_json(run_file, manifest)


def run():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    proof = load_json(PREFLIGHT)
    tasks, corpus = load_heldout_assets(v1.DATA)
    fresh_inputs = validate_inputs(tasks)
    for key, value in fresh_inputs.items():
        if proof[key] != json.loads(json.dumps(value)):
            raise ValueError(f"Changed after CPU preflight: {key}")
    if proof.get("status") != "ok" or proof["step0"]["n"] != 1494:
        raise ValueError("Validated Task 06 preflight required")
    adapters = v1.checkpoints(TRAIN)
    source = certified_v1_run()
    baseline = load_json(v1.RESULTS / "no_compress_results.json")
    run_file = RESULTS / "model_rollout_run.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    occupied = [run_file, OUTPUT / "model_resources.jsonl"]
    for info in adapters:
        occupied.extend([OUTPUT / info["strategy"], RESULTS / (info["strategy"] + "_results.json")])
    if any(path.exists() for path in occupied):
        raise FileExistsError("A v2 evaluation already started; refusing to retry or overwrite it")
    monitor = Monitor(OUTPUT / "model_resources.jsonl", {})
    first_gpu = v1.gpu_info()
    if first_gpu["index"] != 1:
        raise RuntimeError("Task 06 inference requires physical GPU 1")
    started = time.monotonic()
    manifest = {"schema_version": 1, "status": "initializing", "started_at_utc": v1.utc(),
                "pid": os.getpid(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "gpu_before": first_gpu, "root_before": monitor.started_disk,
                "vllm_worker_multiproc_method": os.environ["VLLM_WORKER_MULTIPROC_METHOD"],
                "engine_parameters": proof["engine_parameters"], "generation": source["generation"],
                "lora_loading": source["lora_loading"], "strategies": {},
                "checkpoint_inputs": [{**x, "path": str(x["path"])} for x in adapters],
                "input_files": proof["input_files"], "preflight": v1.file_record(PREFLIGHT),
                "no_compress_baseline": source["no_compress_baseline"]}
    llm = None
    try:
        v1.write_json(run_file, manifest)
        monitor.thread.start()
        llm = LLM(**proof["engine_parameters"])
        sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, seed=SEED)
        manifest["engine_initialization_seconds"] = time.monotonic() - started
        manifest["engine_tokenizer_check"] = v1.check_reference(V1_ARGS, llm.get_tokenizer(), tasks, corpus)
        for number, info in enumerate(adapters, start=1):
            adapter = LoRARequest(info["strategy"], number, str(info["path"]))
            evaluate_adapter(llm, sampling, adapter, info, tasks, corpus, baseline, proof, monitor, manifest, run_file)
        monitor.check()
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                        oom="out of memory" in str(error).lower())
        raise
    finally:
        v1.finish_run(manifest, monitor, llm, run_file, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    args = parser.parse_args()
    v1.validate_environment(args.command)
    if args.command == "preflight":
        preflight()
    else:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        run()


if __name__ == "__main__":
    main()
