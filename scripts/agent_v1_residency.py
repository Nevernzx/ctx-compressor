"""Host weight parking while the resident Qwen3-8B engine keeps its CUDA KV blocks."""
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import time

from scripts import agent_v1_engines as engines
from scripts import run_formal_grpo_pipeline as controller
from src.reader_heldout_inputs import need


def gpu_apps():
    raw = subprocess.check_output([
        "nvidia-smi", "-i", controller.GPU_UUID, "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    apps = {}
    for line in raw.splitlines():
        pid, used = (value.strip() for value in line.split(",", 1))
        apps[int(pid)] = int(used)
    return apps


def owned_apps():
    session = os.getsid(0)
    apps = gpu_apps()
    foreign = []
    for pid in apps:
        try:
            if os.getsid(pid) != session:
                foreign.append(pid)
        except ProcessLookupError:
            # NVML can still return the PID of a just-reaped engine.
            if pid in gpu_apps():
                foreign.append(pid)
    need(not foreign, f"Unexpected compute process on physical GPU 1: {foreign}")
    return apps


def gpu_memory():
    raw = subprocess.check_output([
        "nvidia-smi", "-i", controller.GPU_UUID, "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    return dict(zip(("total_mib", "used_mib", "free_mib"), map(int, raw.split(",")), strict=True))


def wait_for_engine_release(retained_pid):
    deadline = time.monotonic() + 60
    while True:
        if set(owned_apps()) <= {retained_pid, os.getpid()}:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("r4 GPU allocations did not release before weight restoration")
        time.sleep(1)


def worker_state(worker, fingerprint=False):
    model = worker.model_runner.get_model()
    params = {}
    sizes = {}
    digest = sha256()
    for name, tensor in model.named_parameters():
        size = tensor.numel() * tensor.element_size()
        device = str(tensor.device)
        sizes[device] = sizes.get(device, 0) + size
        params[name] = [list(tensor.shape), str(tensor.dtype)]
        if fingerprint:
            # Small fixed samples verify bfloat16 values across both transfers.
            for index in ((0,) * tensor.ndim, (-1,) * tensor.ndim):
                digest.update(tensor.detach()[index].to(dtype=tensor.dtype, device="cpu").float().numpy().tobytes())
    kv = {}
    for layer in worker.model_runner.kv_caches:
        need(layer.device.type == "cuda", "Agent KV data left GPU 1")
        store = layer.untyped_storage()
        kv[str(store.data_ptr())] = store.nbytes()
    need(kv and params, "vLLM worker is missing cache or model parameters")
    result = {"pid": os.getpid(), "param_bytes": sizes, "kv_buffers": kv,
              "param_layout_sha256": sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}
    if fingerprint:
        result["sampled_weights_sha256"] = digest.hexdigest()
    return result


def move_worker_weights(worker, target, fingerprint):
    import torch

    torch.cuda.synchronize(worker.device)
    with torch.inference_mode():
        worker.model_runner.get_model().to(device=torch.device(target))
    torch.cuda.synchronize(worker.device)
    torch.cuda.empty_cache()
    return worker_state(worker, fingerprint)


class AgentWeightExtension:
    def task20_cuda_state(self, fingerprint=False):
        return worker_state(self, fingerprint)

    def task20_move_weights(self, target, fingerprint=False):
        return move_worker_weights(self, target, fingerprint)


EXTENSION = "scripts.agent_v1_residency.AgentWeightExtension"


def clear_cache(llm):
    need(llm.llm_engine.get_num_unfinished_requests() == 0, "Cannot clear the cache with requests in flight")
    need(llm.reset_prefix_cache() is not False, "Engine rejected the prefix-cache reset")


def state(llm, fingerprint=False):
    result, = llm.collective_rpc("task20_cuda_state", timeout=60, args=(fingerprint,))
    return result


def move(llm, target, fingerprint=False):
    need(llm.llm_engine.get_num_unfinished_requests() == 0, "Cannot transfer weights with requests in flight")
    result, = llm.collective_rpc("task20_move_weights", timeout=120, args=(target, fingerprint))
    need(set(result["param_bytes"]) == {target}, f"Incomplete Agent weight transfer: {result['param_bytes']}")
    return result


def check_transfer(before, parked, restored=None):
    candidates = (parked,) if restored is None else (parked, restored)
    for state_record in candidates:
        need(before["pid"] == state_record["pid"] and before["kv_buffers"] == state_record["kv_buffers"],
             "Agent engine process or CUDA KV allocations changed during transfer")
        need(before["param_layout_sha256"] == state_record["param_layout_sha256"]
             and sum(before["param_bytes"].values()) == sum(state_record["param_bytes"].values()),
             "Agent weight layout changed")
        if "sampled_weights_sha256" in before:
            need(before["sampled_weights_sha256"] == state_record["sampled_weights_sha256"],
                 "Agent weight sample changed during host transfer")


@contextmanager
def compressor_session(deployment, runtime, retained_pid):
    from vllm import LLM

    runtime.check_storage()
    engines.require_cuda()
    apps = owned_apps()
    need(retained_pid in apps and set(apps) <= {retained_pid, os.getpid()},
         f"Unexpected compute allocations before r4: {apps}")
    parameters = deployment["r4"]["params"]
    admission = gpu_memory()
    need(admission["free_mib"] >= math.ceil(parameters["gpu_memory_utilization"] * admission["total_mib"]),
         f"Insufficient GPU 1 capacity for the registered r4 engine: {admission}")
    trace = runtime.root_telemetry
    trace.active_engine = "r4"
    start = time.monotonic()
    try:
        with trace.engine_start("r4"):
            llm = LLM(**parameters)
        measurement = {"engine": "r4", "startup_seconds": time.monotonic() - start, "admission": admission}
        runtime.loads.append(measurement)
        try:
            runtime.gpu("r4_loaded")
            measurement["compute_apps_mib"] = owned_apps()
            yield llm
        finally:
            end = time.monotonic()
            llm.llm_engine.engine_core.shutdown()
            wait_for_engine_release(retained_pid)
            measurement["shutdown_seconds"] = time.monotonic() - end
    finally:
        trace.active_engine = "agent"


def with_compressor(llm, before, deployment, runtime, run, log, *, fingerprint=False):
    runtime.check_storage()
    start = time.monotonic()
    parked = move(llm, "cpu", fingerprint)
    parked_seconds = time.monotonic() - start
    check_transfer(before, parked)
    runtime.gpu("agent_weights_parked")
    parked_usage = gpu_memory()
    with compressor_session(deployment, runtime, before["pid"]) as compressor:
        result = run(compressor)
    runtime.check_storage()
    start = time.monotonic()
    restored = move(llm, "cuda:0", fingerprint)
    restore_seconds = time.monotonic() - start
    check_transfer(before, parked, restored)
    runtime.gpu("agent_weights_restored")
    evidence = {"utc": controller.now(), "agent_pid": before["pid"], "park_seconds": parked_seconds,
                "restore_seconds": restore_seconds, "parked_gpu": parked_usage,
                "restored_gpu": gpu_memory(), "resident_kv_bytes": sum(parked["kv_buffers"].values()),
                "weights_bytes": sum(parked["param_bytes"].values()), "fingerprint_checked": fingerprint}
    with log.open("a", encoding="utf-8") as out:
        print(json.dumps(evidence), file=out)
    return result, evidence


def prove_cache(llm, deployment, runtime, log):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    seed = deployment["agent"]["generation"]["seed"]
    tokenizer = llm.get_tokenizer()
    tokens = tokenizer.encode("This is a synthetic prefix-cache check with no evaluation content. " * 48,
                              add_special_tokens=False)
    first, = llm.generate([TokensPrompt(prompt_token_ids=tokens)], SamplingParams(
        temperature=0, seed=seed, max_tokens=1), use_tqdm=False)
    need(first.num_cached_tokens == 0 and list(first.prompt_token_ids) == tokens, "Invalid first cache probe")
    before = state(llm, fingerprint=True)

    def run(compressor):
        options = deployment["r4"]
        prompt = compressor.get_tokenizer().encode("Return a short JSON object.", add_special_tokens=False)
        row, = compressor.generate([TokensPrompt(prompt_token_ids=prompt)],
            SamplingParams(temperature=0, seed=options["sampling"]["seed"], max_tokens=1),
            lora_request=LoRARequest("checkpoint-500", 1, options["adapter"]), use_tqdm=False)
        need(len(row.outputs) == 1, "r4 LoRA cache probe did not generate")
        return len(row.outputs[0].token_ids)

    compressed, transfer = with_compressor(llm, before, deployment, runtime, run, log, fingerprint=True)
    extension = tokenizer.encode(" The same prefix precedes this extension.", add_special_tokens=False)
    second, = llm.generate([TokensPrompt(prompt_token_ids=tokens + extension)], SamplingParams(
        temperature=0, seed=seed, max_tokens=1), use_tqdm=False)
    hit = second.num_cached_tokens
    need(isinstance(hit, int) and 0 < hit <= len(tokens), f"Native prefix cache did not survive r4 handoff: {hit}")
    clear_cache(llm)
    return {"status": "passed", "scope": "synthetic warmup, no evaluation tasks", "first_prompt_tokens": len(tokens),
            "first_cached_tokens": first.num_cached_tokens, "next_prompt_tokens": len(tokens + extension),
            "next_cached_tokens": hit, "r4_probe_generated_tokens": compressed, "transfer": transfer}

