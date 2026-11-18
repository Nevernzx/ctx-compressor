#!/usr/bin/env python3
"""Task 05 local LoRA replay; CPU proof/preflight and explicitly separate GPU run.

This is the only new source importing vLLM. Existing reader code/environments
remain untouched. Run template-proof in train, preflight/run in reader.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
MODEL = EXTERNAL / "models/Qwen3-1.7B"
DATA = EXTERNAL / "data/traj_train"
RESULTS = PROJECT / "eval/sft_run"
CHECKPOINTS = EXTERNAL / "runs/sft_20k_20261117/checkpoints"
STEPS = (500, 1000, 1500, 2000, 2500)
sys.path.insert(0, str(PROJECT))

from src.build_traj import make_sample, oracle_action
from src.heldout_eval import BUDGETS, MODES, HeldoutReplay, load_heldout_assets
from src.model_inference import (MAX_NEW_TOKENS, SEED, compact, engine_parameters,
                                 generation_record, guard_prompt, inference_prompt, sha256_json)
from src.sft_protocol import (TEMPLATE_OPTIONS, encode_sft_record, make_messages,
                              make_sft_template, template_evidence)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_record(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def input_records(data_dir, model):
    return {str(path): file_record(path) for path in
            [data_dir / "heldout_tasks.jsonl", data_dir / "corpus_train.jsonl",
             model / "tokenizer.json", model / "tokenizer_config.json", model / "config.json"]}


def validate_environment(command):
    if PROJECT != Path("/root/workplace/ctx-compressor"):
        raise RuntimeError("Run only on 5880 in the formal project")
    gpu = command == "run"
    expected_env = "train" if command == "template-proof" else "reader"
    if Path(sys.prefix).resolve() != EXTERNAL / "envs" / expected_env:
        raise RuntimeError(f"Use the existing {expected_env} environment")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ("1" if gpu else ""):
        raise RuntimeError("Explicit CUDA_VISIBLE_DEVICES=1 for run, empty for CPU commands")
    for name in ("TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "MODELSCOPE_CACHE", "VLLM_CACHE_ROOT",
                 "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "PYTHONPYCACHEPREFIX"):
        value = os.environ.get(name)
        if not value or not Path(value).resolve().is_relative_to(EXTERNAL / "cache"):
            raise RuntimeError(f"External runtime cache required: {name}")
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        if os.environ.get(name):
            raise RuntimeError(f"Proxy must be unset: {name}")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"Offline mode required: {name}=1")


def first_state(tasks, corpus):
    task = tasks[0]
    first = task["steps"][0]
    state = make_sample(task["questions"], [], first["new"], {first["gold"]}, corpus,
                        BUDGETS["tight"], "reselect", {})
    state["sft_target"] = oracle_action(state)
    return state, {"task": task["task"], "step": 0, "mode": "reselect", "budget_tier": "tight",
                   "budget": BUDGETS["tight"], "split": "heldout"}


def template_proof(args):
    path = args.results / "model_template_reference.json"
    if path.exists():
        raise FileExistsError(f"Refusing to replace proof: {path}")
    tasks, corpus = load_heldout_assets(args.data_dir)
    state, identity = first_state(tasks, corpus)
    template = make_sft_template(args.model, max_length=None)
    messages = make_messages(state)
    encoded = encode_sft_record(template, messages)
    evidence = template_evidence(template, messages, encoded)
    prompt = inference_prompt(template.tokenizer, state)
    if encoded["input_ids"][:prompt["prompt_tokens"]] != prompt["prompt_token_ids"]:
        raise RuntimeError("Inference token IDs disagree with actual Swift training")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU template proof initialized CUDA")
    proof = {"status": "ok", "created_at_utc": utc(), "identity": identity,
             "template_options": TEMPLATE_OPTIONS, "max_length": None,
             "versions": {name: version(name) for name in ("ms-swift", "torch", "transformers", "tokenizers")},
             "input_files": input_records(args.data_dir, args.model),
             "prompt": prompt, "training_input_ids": encoded["input_ids"],
             "masked_label_range": evidence["masked_label_range"],
             "supervised_label_range": evidence["supervised_label_range"],
             "empty_think_tokens": evidence["empty_think_tokens"],
             "actual_training_prefix_equals_inference_ids": True,
             "cuda_initialized": torch.cuda.is_initialized()}
    write_json(path, proof)
    print(compact({"proof": str(path), "prompt_tokens": prompt["prompt_tokens"],
                   "masked_input_tokens": encoded["n_input_tokens"],
                   "empty_think_tokens": evidence["empty_think_tokens"]}), flush=True)


def check_reference(args, tokenizer, tasks, corpus):
    path = args.results / "model_template_reference.json"
    proof = json.loads(path.read_text())
    if proof.get("status") != "ok" or not proof["actual_training_prefix_equals_inference_ids"]:
        raise RuntimeError("Actual Task 04-compatible training template proof is required")
    if proof["input_files"] != input_records(args.data_dir, args.model):
        raise RuntimeError("Heldout data/model tokenizer changed after template proof")
    state, identity = first_state(tasks, corpus)
    prompt = inference_prompt(tokenizer, state)
    if identity != proof["identity"] or prompt != proof["prompt"]:
        raise RuntimeError("Reader tokenizer prefix differs from actual Swift training prefix")
    if proof["training_input_ids"][:prompt["prompt_tokens"]] != prompt["prompt_token_ids"]:
        raise RuntimeError("Saved training and inference token IDs differ")
    return {"reference": file_record(path), "identity": identity, "prompt_tokens": prompt["prompt_tokens"],
            "masked_input_tokens": proof["masked_label_range"][1],
            "empty_think_tokens": proof["empty_think_tokens"],
            "exact_token_ids_match": True, "prompt_sha256": prompt["prompt_sha256"]}


def union_context_check(tokenizer, tasks, corpus, maximum):
    """Measure an oversized all-history context, plus current returns, per step.

The envelope retains every earlier fragment even when this exceeds budget.
It is diagnostic; every real closed-loop prompt is separately length-checked
before being submitted, so unusual token boundary behavior cannot crop input.
"""
    rows = []
    maximum_prompt = 0
    worst = None
    for task in tasks:
        kept, lid, gold = [], {}, set()
        for step, returned in enumerate(task["steps"]):
            gold.add(returned["gold"])
            row_max = 0
            for mode in MODES:
                state = make_sample(task["questions"], kept, returned["new"], gold, corpus,
                                    max(BUDGETS.values()), mode, lid)
                prompt = inference_prompt(tokenizer, state)
                row_max = max(row_max, prompt["prompt_tokens"])
            kept = list(dict.fromkeys(kept + returned["new"]))
            rows.append(row_max)
            if row_max > maximum_prompt:
                maximum_prompt = row_max
                worst = {"task": task["task"], "step": step, "union_fragments": len(kept)}
    if maximum_prompt + MAX_NEW_TOKENS > maximum:
        raise RuntimeError(f"Full-history prompt envelope exceeds native context: "
                           f"{maximum_prompt} + {MAX_NEW_TOKENS} > {maximum}; stop without truncation")
    return {"method": "all historical paragraphs retained + current returns, both modes, budget=5628",
            "n_tasks": len(tasks), "n_saved_steps": len(rows), "n_prompts": 2 * len(rows),
            "max_union_prompt_tokens": maximum_prompt, "worst": worst,
            "max_tokens": MAX_NEW_TOKENS, "max_model_len": maximum,
            "remaining_tokens_at_union_max": maximum - maximum_prompt - MAX_NEW_TOKENS,
            "every_actual_prompt_checked_before_generation": True, "input_truncations": 0}


def vllm_cpu_check():
    # Only this new source imports vLLM. No engine/model weights are initialized.
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.engine.arg_utils import EngineArgs
    import torch
    package = Path(vllm.__file__).parent
    model_source = package / "model_executor/models/qwen3.py"
    source = model_source.read_text()
    if "class Qwen3ForCausalLM(nn.Module, SupportsLoRA" not in source:
        raise RuntimeError("Installed Qwen3 backend does not declare LoRA support")
    required = ("enable_lora", "max_lora_rank", "max_loras", "max_cpu_loras")
    if any(name not in EngineArgs.__dataclass_fields__ for name in required):
        raise RuntimeError("Installed engine lacks required LoRA parameters")
    if "lora_request" not in inspect.signature(LLM.generate).parameters:
        raise RuntimeError("Installed LLM.generate lacks lora_request")
    request = LoRARequest("cpu-probe-only", 1, "/unused/cpu-probe")
    sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, seed=SEED)
    if torch.cuda.is_initialized():
        raise RuntimeError("vLLM CPU import check initialized CUDA")
    return {"versions": {name: version(name) for name in ("vllm", "torch", "transformers", "tokenizers")},
            "qwen3_supports_lora": True, "qwen3_source": file_record(model_source),
            "engine_args": list(required), "generate_accepts_lora_request": True,
            "request_id": request.lora_int_id, "sampling_params": repr(sampling),
            "cuda_initialized": torch.cuda.is_initialized(), "engine_created": False}


def preflight(args):
    path = args.results / "model_inference_preflight.json"
    if path.exists():
        raise FileExistsError(f"Refusing to replace preflight: {path}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    tasks, corpus = load_heldout_assets(args.data_dir)
    native = json.loads((args.model / "config.json").read_text())
    parameters = engine_parameters(args.model, native)
    proof = check_reference(args, tokenizer, tasks, corpus)
    union = union_context_check(tokenizer, tasks, corpus, parameters["max_model_len"])
    checks = vllm_cpu_check()
    result = {"status": "ok", "created_at_utc": utc(), "template_check": proof,
              "context_check": union, "vllm_cpu_check": checks, "engine_parameters": parameters,
              "input_files": input_records(args.data_dir, args.model),
              "expected_strategies": ["base", *[f"checkpoint-{step}" for step in STEPS]],
              "expected_trajectories_per_strategy": len(tasks) * 6,
              "expected_outputs_per_strategy": sum(len(t["steps"]) for t in tasks) * 6}
    write_json(path, result)
    print(compact(result), flush=True)


def gpu_info():
    values = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True).strip().split(",")
    return dict(zip(("index", "memory_used_mib", "memory_total_mib", "utilization_gpu"),
                    (int(value.strip()) for value in values)))


def disk_info():
    raw = subprocess.check_output(["df", "-B1", "--output=used,avail", "/"], text=True)
    values = raw.splitlines()[-1].split()
    return {"used_bytes": int(values[0]), "available_bytes": int(values[1]), "df_raw": raw}


class Monitor:
    """Record shared-root usage; only finalization enforces the final 300 MB cap."""

    def __init__(self, path, baseline):
        self.path, self.started_disk = path, disk_info()
        self.baseline = baseline
        self.stop = threading.Event()
        self.error = None
        self.max_memory_mib = 0
        self.root_samples = 0
        self.root_over_limit_samples = 0
        self.root_peak_delta_bytes = None
        self.root_last_delta_bytes = None
        self.root_over_limit = False
        self.root_limit_events = []
        self.startup_root = self.record_disk(self.started_disk, phase="startup")
        self.thread = threading.Thread(target=self.run, daemon=True)

    def record_disk(self, disk, *, phase):
        """Use the unchanged Task 05 baseline; retain threshold transitions."""
        delta = disk["used_bytes"] - self.baseline["root_df_used_bytes"]
        limit = self.baseline["root_final_growth_limit_bytes"]
        over_limit = delta > limit
        event = ("enter" if over_limit else "recover") if over_limit != self.root_over_limit else None
        self.root_samples += 1
        self.root_over_limit_samples += int(over_limit)
        self.root_peak_delta_bytes = delta if self.root_peak_delta_bytes is None else max(self.root_peak_delta_bytes, delta)
        self.root_last_delta_bytes, self.root_over_limit = delta, over_limit
        sample = {"utc": utc(), "phase": phase, "root_df": disk, "root_delta_bytes": delta,
                  "root_peak_delta_bytes": self.root_peak_delta_bytes, "root_limit_bytes": limit,
                  "root_over_limit": over_limit, "root_limit_event": event}
        if event is not None:
            self.root_limit_events.append({key: sample[key] for key in
                                          ("utc", "phase", "root_delta_bytes", "root_limit_bytes", "root_limit_event")})
        return sample

    def diagnostics(self):
        return {"policy": "observe during inference; finalizer enforces final raw df delta <= 300000000 bytes",
                "baseline_used_bytes": self.baseline["root_df_used_bytes"],
                "limit_bytes": self.baseline["root_final_growth_limit_bytes"],
                "sample_count_including_startup_and_shutdown": self.root_samples,
                "over_limit_sample_count": self.root_over_limit_samples,
                "peak_delta_bytes": self.root_peak_delta_bytes,
                "last_delta_bytes": self.root_last_delta_bytes, "last_over_limit": self.root_over_limit,
                "threshold_events": list(self.root_limit_events),
                "monitor_error": repr(self.error) if self.error is not None else None}

    def run(self):
        try:
            with self.path.open("x") as stream:
                while not self.stop.is_set():
                    gpu, disk = gpu_info(), disk_info()
                    self.max_memory_mib = max(self.max_memory_mib, gpu["memory_used_mib"])
                    stream.write(compact({**self.record_disk(disk, phase="monitor"), "gpu": gpu}) + "\n")
                    stream.flush()
                    self.stop.wait(20)
        except BaseException as error:
            self.error = error

    def check(self):
        if self.error is not None:
            raise RuntimeError("Model replay resource monitor failed") from self.error


def finish_run(manifest, monitor, llm, run_path, started):
    """Stop monitoring and release the engine before publishing terminal status.

    Cleanup errors cannot publish success or suppress an existing inference error.
    If the manifest cannot be written, one failed-status write is attempted and the
    I/O error still propagates; no model work or inference is retried.
    """
    primary_error = sys.exc_info()[1]
    first_error = None

    def failed(stage, error):
        nonlocal first_error
        if first_error is None:
            first_error = error
        manifest["status"] = "failed"
        manifest.setdefault("error", repr(error))
        manifest.setdefault("cleanup_errors", []).append(
            {"stage": stage, "error": repr(error), "traceback": traceback.format_exc()})

    manifest["monitor_thread_stopped"] = False
    manifest["root_disk_diagnostics_complete"] = False
    manifest["engine_shutdown"] = {"required": llm is not None, "completed": llm is None}
    stage = "monitor_stop"
    try:
        monitor.stop.set()
        if monitor.thread.is_alive():
            monitor.thread.join(timeout=10)
        manifest["monitor_thread_stopped"] = not monitor.thread.is_alive()
        if not manifest["monitor_thread_stopped"]:
            raise RuntimeError("Resource monitor thread did not stop within 10 seconds")
        stage = "monitor_error_check"
        monitor.check()
        stage = "final_disk_read"
        final_disk = disk_info()
        manifest["root_after"] = final_disk
        stage = "final_disk_record"
        manifest["root_shutdown_diagnostics"] = monitor.record_disk(final_disk, phase="shutdown")
        stage = "final_disk_diagnostics"
        manifest["root_disk_diagnostics"] = monitor.diagnostics()
        manifest["root_disk_diagnostics_complete"] = True
    except BaseException as error:
        failed(stage, error)
    finally:
        # This cleanup runs even when join, disk sampling, or diagnostics fail.
        try:
            if llm is not None:
                core = getattr(llm.llm_engine, "engine_core", None)
                if core is None or not hasattr(core, "shutdown"):
                    raise RuntimeError("Initialized inference engine has no shutdown method")
                core.shutdown()
                manifest["engine_shutdown"]["completed"] = True
        except BaseException as error:
            failed("engine_shutdown", error)
        finally:
            try:
                gc.collect()
            except BaseException as error:
                failed("garbage_collection", error)

    manifest.update(completed_at_utc=utc(), elapsed_seconds=time.monotonic() - started,
                    peak_gpu_memory_used_mib=monitor.max_memory_mib)
    try:
        write_json(run_path, manifest)
    except BaseException as error:
        failed("terminal_manifest_write", error)
        try:
            write_json(run_path, manifest)
        except BaseException as retry_error:
            failed("failed_manifest_write", retry_error)
    if first_error is not None and primary_error is None:
        raise first_error


def checkpoints(root):
    expected = {f"checkpoint-{step}" for step in STEPS}
    actual = {path.name for path in root.glob("checkpoint-*") if path.is_dir()}
    if actual != expected:
        raise RuntimeError(f"Exactly five checkpoints required: expected {sorted(expected)}, got {sorted(actual)}")
    result = []
    for step in STEPS:
        path = root / f"checkpoint-{step}"
        config_path, weights = path / "adapter_config.json", path / "adapter_model.safetensors"
        config = json.loads(config_path.read_text())
        if config.get("peft_type") != "LORA" or config.get("r") != 16 or config.get("lora_alpha") != 32:
            raise RuntimeError(f"Unexpected LoRA configuration: {path}")
        if not weights.is_file() or weights.stat().st_size == 0:
            raise RuntimeError(f"Missing checkpoint weights: {path}")
        state = json.loads((path / "trainer_state.json").read_text())
        if state.get("global_step") != step:
            raise RuntimeError(f"Checkpoint step metadata mismatch: {path}")
        result.append({"strategy": f"checkpoint-{step}", "step": step, "path": path,
                       "files": [file_record(config_path), file_record(weights)]})
    return result


def run(args):
    preflight_path = args.results / "model_inference_preflight.json"
    checked = json.loads(preflight_path.read_text())
    if checked.get("status") != "ok" or checked["input_files"] != input_records(args.data_dir, args.model):
        raise RuntimeError("CPU preflight missing or inputs changed")
    adapters = checkpoints(args.checkpoint_root)
    native = json.loads((args.model / "config.json").read_text())
    parameters = engine_parameters(args.model, native)
    if checked["engine_parameters"] != parameters:
        raise RuntimeError("Engine configuration differs from checked preflight")
    strategies = [{"strategy": "base", "step": None, "path": None, "files": []}, *adapters]
    for strategy in strategies:
        if (args.output_root / strategy["strategy"]).exists():
            raise FileExistsError(f"Refusing to overwrite prior model output: {strategy['strategy']}")
        if (args.results / f"{strategy['strategy']}_results.json").exists():
            raise FileExistsError(f"Prior model metrics already exist: {strategy['strategy']}")
    run_path = args.results / "model_rollout_run.json"
    if run_path.exists():
        raise FileExistsError(run_path)
    first_gpu = gpu_info()
    if first_gpu["memory_used_mib"] >= 2048:
        raise RuntimeError("NEED: physical GPU 1 already uses >=2 GB; stopped before engine initialization")
    tasks, corpus = load_heldout_assets(args.data_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    reference = check_reference(args, tokenizer, tasks, corpus)
    args.output_root.mkdir(parents=True, exist_ok=True)
    resource_baseline = json.loads((args.results / "preflight.json").read_text())
    monitor = Monitor(args.output_root / "model_resources.jsonl", resource_baseline)
    if monitor.path.exists():
        raise FileExistsError(monitor.path)
    baseline_path = args.results / "no_compress_results.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline["settings"]["n_tasks"] != 249 or baseline["combined"]["n_tasks"] != 1494:
        raise RuntimeError("Matching heldout CPU no_compress baseline required")
    started = time.monotonic()
    manifest = {"schema_version": 1, "status": "initializing", "started_at_utc": utc(), "pid": os.getpid(),
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_before": first_gpu,
                "vllm_worker_multiproc_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
                "root_before": monitor.started_disk, "task05_resource_baseline": resource_baseline,
                "root_startup_diagnostics": monitor.startup_root,
                "root_disk_diagnostics": monitor.diagnostics(),
                "engine_parameters": parameters,
                "preflight": file_record(preflight_path), "template_check": reference,
                "generation": {"temperature": 0, "max_tokens": MAX_NEW_TOKENS, "n": 1, "seed": SEED,
                               "enable_thinking": False, "input_truncation": False},
                "lora_loading": "native vLLM LoRA; base loaded once; sequential adapters, rank16",
                "checkpoint_inputs": [{**row, "path": str(row["path"])} for row in adapters],
                "strategies": {}, "input_files": checked["input_files"]}
    llm = None
    try:
        write_json(run_path, manifest)
        monitor.thread.start()
        # No other newly added file imports vLLM; old CPU tests stay torch-free.
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        monitor.check()
        immediate = gpu_info()
        if immediate["memory_used_mib"] >= 2048:
            raise RuntimeError("NEED: GPU 1 became occupied before initialization")
        llm = LLM(**parameters)
        sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, seed=SEED)
        actual_tokenizer = llm.get_tokenizer()
        manifest["engine_tokenizer_check"] = check_reference(args, actual_tokenizer, tasks, corpus)
        manifest["engine_initialization_seconds"] = time.monotonic() - started
        manifest["no_compress_baseline"] = file_record(baseline_path)
        for number, strategy in enumerate(strategies):
            name = strategy["strategy"]
            directory = args.output_root / name
            directory.mkdir()
            replay = HeldoutReplay(tasks, corpus, name)
            adapter = None if strategy["path"] is None else LoRARequest(name, number, str(strategy["path"]))
            strategy_started = time.monotonic()
            progress = {"checkpoint_step": strategy["step"], "status": "running", "n_steps": 0,
                        "n_trajectories": len(tasks) * 6, "batches": [], "output_limit_hits": 0,
                        "max_prompt_tokens": 0, "invalid_counts": Counter()}
            manifest["status"], manifest["active_strategy"] = "running", name
            manifest["strategies"][name] = progress
            write_json(run_path, manifest)
            with (directory / "steps.jsonl").open("x", encoding="utf-8") as stream:
                while not replay.completed():
                    monitor.check()
                    items = replay.next_batch()
                    prompts = [inference_prompt(actual_tokenizer, item["state"]) for item in items]
                    for item, prompt in zip(items, prompts):
                        guard_prompt(prompt, parameters["max_model_len"], identity=(item["trajectory_id"], item["step"]))
                    batch_started = time.monotonic()
                    outputs = llm.generate([{"prompt_token_ids": p["prompt_token_ids"]} for p in prompts],
                                           sampling_params=sampling, lora_request=adapter, use_tqdm=False)
                    elapsed = time.monotonic() - batch_started
                    if len(outputs) != len(items):
                        raise RuntimeError("Incomplete vLLM batch; refusing to drop trajectories")
                    generation = [generation_record(prompt, output) for prompt, output in zip(prompts, outputs)]
                    if not progress["n_steps"]:
                        if generation[0]["prompt_sha256"] != reference["prompt_sha256"]:
                            raise RuntimeError("First actual generation differs from the Swift training prefix")
                        progress["actual_training_prefix_check"] = {
                            "identity": reference["identity"], "exact_token_ids_match": True,
                            "runtime_echoed_exact_prompt_ids": generation[0]["runtime_prompt_ids_match"],
                            "prompt_sha256": generation[0]["prompt_sha256"],
                            "training_reference_sha256": reference["reference"]["sha256"]}
                    records = replay.advance_batch(generation)
                    for record in records:
                        stream.write(compact(record) + "\n")
                        progress["output_limit_hits"] += int(record["output_limit_hit"])
                        if record["invalid_reason"]:
                            progress["invalid_counts"][record["invalid_reason"]] += 1
                    stream.flush()
                    os.fsync(stream.fileno())
                    progress["n_steps"] += len(records)
                    progress["max_prompt_tokens"] = max(progress["max_prompt_tokens"], max(p["prompt_tokens"] for p in prompts))
                    progress["batches"].append({"step": items[0]["step"], "n": len(items), "seconds": elapsed,
                                                "prompt_tokens": sum(p["prompt_tokens"] for p in prompts),
                                                "generated_tokens": sum(r["generated_tokens"] for r in generation)})
                    progress["elapsed_seconds"] = time.monotonic() - strategy_started
                    write_json(run_path, manifest)
                    print(f"{utc()} {name} step={items[0]['step']} n={len(items)} total={progress['n_steps']} "
                          f"seconds={elapsed:.3f} invalid={dict(progress['invalid_counts'])}", flush=True)
                monitor.check()
            if progress["n_steps"] != checked["expected_outputs_per_strategy"]:
                raise RuntimeError("Completed output count disagrees with fixed saved heldout steps")
            metrics = replay.aggregate(baseline=baseline)
            metrics["checkpoint_step"] = strategy["step"]
            metrics["runtime"] = {"elapsed_seconds": time.monotonic() - strategy_started,
                                  "max_prompt_tokens": progress["max_prompt_tokens"], "input_truncations": 0}
            metrics["steps_artifact"] = file_record(directory / "steps.jsonl")
            write_json(directory / "metrics.json", metrics)
            write_json(args.results / f"{name}_results.json", metrics)
            progress["status"], progress["metrics"] = "complete", file_record(directory / "metrics.json")
            write_json(run_path, manifest)
        monitor.check()
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                        oom="out of memory" in str(error).lower())
        raise
    finally:
        finish_run(manifest, monitor, llm, run_path, started)
    print(compact({"status": manifest["status"], "strategies": len(manifest["strategies"]),
                   "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("template-proof", "preflight", "run"))
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINTS)
    parser.add_argument("--output-root", type=Path, default=EXTERNAL / "eval_rollouts")
    args = parser.parse_args()
    validate_environment(args.command)
    if args.command == "run":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if args.model.resolve() != MODEL or args.data_dir.resolve() != DATA:
        parser.error("Only the fixed local Qwen3-1.7B and train-heldout assets are allowed")
    for name in ("checkpoint_root", "output_root"):
        if not getattr(args, name).resolve().is_relative_to(EXTERNAL):
            parser.error(f"{name} must be on userfolder")
    {"template-proof": template_proof, "preflight": preflight, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
