"""Local Task 20 model backends and registered engine admission evidence."""
from contextlib import contextmanager, nullcontext
from importlib.metadata import version
import gc
import json
import os
from pathlib import Path
import time

from scripts import agent_v1_disk as disk, grpo_r4_common as grpo
from scripts import reader_diag_cascade
from scripts import run_formal_grpo_pipeline as pipeline
from scripts.train_reranker_v1 import load_model, score_pairs
from scripts.agent_v1_data import PROJECT, SMALL
from src import agent_v1 as agent
from src.reader_heldout_inputs import EXTERNAL, need, read, verify
from src.reranker_v1 import PairTokenizer, manifest_entries, probability, record, signed_project
from src.model_inference import generation_record, guard_prompt

AGENT_UTIL = 0.65
AGENT_SEQUENCES = 32
AGENT_SEED = 20261117
READER_CONFIG = PROJECT / "eval/reader/run_config.json"
READER_MANIFEST = PROJECT / "eval/reader/model_manifest.json"
R4_DIR = PROJECT / "eval/grpo_r4"
RR_DIR = PROJECT / "eval/reranker_v1"
R4_RUN = R4_DIR / "checkpoint-500/run.json"
RR_SNAPSHOT = EXTERNAL / "models/Qwen3-Reranker-0.6B"
MODEL_CARDS = SMALL / "model_cards.json"
TIKTOKEN_CACHE = EXTERNAL / "vendor/tiktoken_cache"
RR_BLOCK = 64
GENERATION_STOP = "\nObservation"


def checked_artifact(item):
    path = Path(item["path"])
    need(path.is_absolute() and path.resolve() == path and
         path.stat().st_size == item.get("size", item.get("bytes")), f"Asset size or path changed: {path}")
    verify(path, item["sha256"])


def registered_assets(*, verify_models):
    spec_hash = signed_project(READER_CONFIG)
    source = read(READER_CONFIG)
    params = dict(source["parameters"])
    need(params["max_model_len"] == agent.MAX_MODEL_LEN and params["enable_prefix_caching"] and
         params["seed"] == AGENT_SEED and "disable_cascade_attn" not in params, "Reader baseline engine config changed")
    params.update(gpu_memory_utilization=AGENT_UTIL, max_num_seqs=AGENT_SEQUENCES, disable_cascade_attn=True)
    verify(READER_MANIFEST, source["model_manifest_sha256"])
    reader = read(READER_MANIFEST)
    need(reader["model_root"] == params["model"] == params["tokenizer"], "Reader model manifest root differs")

    r4_manifest = R4_DIR / "TASK18_SHA256SUMS"
    r4_file = grpo.signature(R4_RUN, r4_manifest)
    r4 = read(R4_RUN)
    need(r4["status"] == "complete" and r4["checkpoint_step"] == 500 and
         (r4["temperature"], r4["max_tokens"], r4["greedy"]) == (0, agent.MAX_NEW, True),
         "Saved Task 18 F deployment config differs")
    adapter = grpo.signature(Path(r4["adapter_weight"]["path"]), r4_manifest)
    need(adapter == r4["adapter_weight"] and Path(adapter["path"]).parent == Path(r4["adapter"]),
         "Saved Task 18 F adapter differs")
    grpo.signature(Path(r4["adapter"]) / "adapter_config.json", r4_manifest)

    rr_manifest = manifest_entries(RR_DIR / "TASK17_SHA256SUMS")
    for p in (RR_DIR/"training.json", RR_DIR/"tau_selection.json"):
        verify(p, rr_manifest[str(p.relative_to(PROJECT))])
    rr_train = read(RR_DIR / "training.json")
    need(rr_train["status"] == "complete" and read(RR_DIR/"tau_selection.json")["tau"] == agent.TAU,
         "Unexpected Task 17 fitted reranker or threshold")
    rr = rr_train["chosen"]
    config = Path(rr["checkpoint"]) / "config.json"
    verify(config, rr_manifest[str(config)])
    need(rr["weights"]["sha256"] == rr_manifest[rr["weights"]["path"]], "Task 17 weight signature differs")

    cards = read(MODEL_CARDS)
    cache = cards["runtime_dependencies"]["cl100k_base"]
    need(cards["offline_cpu_preflight"]["status"] == "passed" and
         cards["open_provence"]["status"] == "not_run" and
         cache["TIKTOKEN_CACHE_DIR"] == str(TIKTOKEN_CACHE), "Unexpected Task 20 model cards")
    checked_artifact(cache["cache_entry"])
    wheel = cards["llmlingua_wheel"]
    llmlingua = cards["llmlingua2"]
    if verify_models:
        for item in reader["files"]:
            checked_artifact({"path": str(Path(reader["model_root"]) / item["path"]),
                              "size": item["size_bytes"], "sha256": item["sha256"]})
        grpo.verify_model()
        checked_artifact(rr["weights"])
        for file in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            p = RR_SNAPSHOT / file
            verify(p, rr_manifest[str(p)])
        for item in [wheel["wheel"], *wheel["files"], *llmlingua["files"]]:
            checked_artifact(item)
    return {
        "agent": {"params": params, "package_versions": source["package_versions"], "reader_config_sha256": spec_hash,
                  "model_manifest_sha256": source["model_manifest_sha256"], "generation": {
                     "temperature": 0, "seed": AGENT_SEED, "max_tokens": agent.MAX_NEW,
                     "stop": [GENERATION_STOP], "enable_thinking": False}},
        "r4": {"params": r4["engine_parameters"], "adapter": r4["adapter"], "adapter_weight": adapter,
               "sampling": {"temperature": 0, "seed": r4["seed"], "max_tokens": agent.MAX_NEW},
               "source_run": r4_file},
        "reranker": {"checkpoint": rr["checkpoint"], "weights": rr["weights"], "template": rr_train["template"],
                     "tau": agent.TAU, "tokenizer": str(RR_SNAPSHOT)},
        "llmlingua2": {"model": llmlingua["directory"], "wheel": wheel["vendor"], "config": {
            "use_llmlingua2": True, "rate": agent.RATE, "force_tokens": list(agent.FORCE_TOKENS)}},
        "model_cards": record(MODEL_CARDS),
    }


def enforce_cascade(engine, logfile):
    actual, source, evidence = reader_diag_cascade.effective_cascade(engine, logfile)
    need(actual is True, f"Qwen3-8B disable_cascade_attn must be effective True, got {actual!r}")
    return {"effective_disable_cascade_attn": actual, "source": source, "log_evidence": evidence}


def require_cuda():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
         "Task 20 smoke worker only runs on physical GPU 1")


def offline_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=False, local_files_only=True)


def engine_ready():
    until = time.monotonic() + 60
    while True:
        observed = pipeline.gpu()
        if observed["memory_used_mib"] < 2048:
            return observed
        if time.monotonic() >= until:
            raise RuntimeError("GPU 1 did not return below 2048 MiB after the preceding local engine")
        time.sleep(1)


class RuntimeMeasurements:
    def __init__(self, path, root_telemetry=None):
        self.path = path
        self.root_telemetry = root_telemetry
        self.peaks = []
        self.root_usage = []
        self.loads = []

    def check_storage(self):
        if self.root_telemetry:
            self.root_telemetry.policy.check()

    def gpu(self, stage):
        if self.root_telemetry:
            sample = self.root_telemetry.sample(stage, self.root_telemetry.active_engine)
            used, free = sample["root_used_bytes"], sample["root_free_bytes"]
        else:
            reading = disk.root_usage()
            used, free = reading.used, reading.free
        item = {"stage": stage, "utc": pipeline.now(), **pipeline.gpu(),
                "root_used_bytes": used, "root_free_bytes": free}
        self.peaks.append(item["memory_used_mib"])
        self.root_usage.append(used)
        with self.path.open("a", encoding="utf-8") as f:
            print(json.dumps(item), file=f)
        self.check_storage()


@contextmanager
def engine_session(kind, deployment, runtime, log):
    from vllm import LLM

    require_cuda()
    runtime.check_storage()
    engine_ready()
    start = time.monotonic()
    telemetry = runtime.root_telemetry
    if telemetry:
        telemetry.active_engine = kind
    try:
        startup = telemetry.engine_start(kind) if telemetry else nullcontext()
        with startup:
            llm = LLM(**deployment[kind]["params"])
        result = {"engine": kind, "startup_seconds": time.monotonic() - start}
        try:
            if kind == "agent":
                result.update(enforce_cascade(llm.llm_engine, log))
            runtime.loads.append(result)
            runtime.gpu(f"{kind}_loaded")
            yield llm
        finally:
            before = time.monotonic()
            llm.llm_engine.engine_core.shutdown()
            result["shutdown_seconds"] = time.monotonic() - before
    finally:
        if telemetry:
            telemetry.active_engine = None


def generate_agent(llm, deployment, batch):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    opts = deployment["agent"]["generation"]
    sampling = SamplingParams(temperature=opts["temperature"], max_tokens=opts["max_tokens"],
                              seed=opts["seed"], stop=opts["stop"], n=1)
    prompts = [ep.next_prompt() for ep in batch]
    outputs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sampling, use_tqdm=False)
    for out, tokens in zip(outputs, prompts, strict=True):
        need(list(out.prompt_token_ids) == tokens and len(out.outputs) == 1, "Agent input changed or truncated")
        msg = out.outputs[0]
        yield tokens, msg.text, len(msg.token_ids), getattr(out, "num_cached_tokens", None)


def execute_r4(llm, deployment, pending, runtime):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    params = deployment["r4"]
    sampling = SamplingParams(**params["sampling"], n=1)
    request = LoRARequest("checkpoint-500", 1, params["adapter"])
    tokenizer = llm.get_tokenizer()
    width = params["params"]["max_num_seqs"]
    for offset in range(0, len(pending), width):
        runtime.check_storage()
        group = pending[offset:offset + width]
        begun = time.monotonic()
        prompts = [agent.compressor_prompt(tokenizer, item.state) for item in group]
        for pack, item in zip(prompts, group, strict=True):
            guard_prompt(pack, params["params"]["max_model_len"], agent.MAX_NEW, item.episode.task["task"])
        batch = llm.generate([TokensPrompt(prompt_token_ids=p["prompt_token_ids"]) for p in prompts],
                             sampling, lora_request=request, use_tqdm=False)
        runtime.check_storage()
        wall = time.monotonic() - begun
        for pack, out, item in zip(prompts, batch, group, strict=True):
            evidence = generation_record(pack, out, agent.MAX_NEW)
            decision = agent.apply_compression(item, evidence["raw_output"])
            decision.details.update(wall_seconds=wall, vllm_batch_size=len(group),
                                    compressor_prompt_tokens=evidence["prompt_tokens"],
                                    compressor_prompt_sha256=evidence["prompt_sha256"],
                                    compressor_generated_tokens=evidence["generated_tokens"])
            item.episode.apply_decision(item.turn, decision)
    runtime.gpu("r4_requests_done")


def new_reranker(config):
    import torch

    pair = PairTokenizer(offline_tokenizer(config["tokenizer"]))
    need(pair.config() == config["template"], "Task 17 reranker template changed")
    model = load_model(Path(config["checkpoint"]), torch.bfloat16).eval()

    def compress(item):
        mode = item.episode.mode
        state = item.state
        candidate_ids = state["new_ids"] if mode == "append" else state["kept_ids"] + state["new_ids"]
        pids = [state["pid_by_lid"][lid] for lid in candidate_ids]
        documents = [item.episode.corpus[pid]["text"] for pid in pids]
        pairs = list(pair.encode("\n".join(item.episode.question_text), documents))
        enc = [ids for ids, _ in pairs]
        scores = []
        with torch.inference_mode():
            for i in range(0, len(enc), RR_BLOCK):
                scores.extend(score_pairs(model, enc[i:i + RR_BLOCK], pair.config()).tolist())
        decision = agent.reranker_compression(item, {pid: probability(s) for pid, s in zip(pids, scores, strict=True)})
        decision.details["score_pairs"] = len(pairs)
        decision.details["truncated_pairs"] = sum(cut for _, cut in pairs)
        return decision

    return compress


def new_llmlingua(config, ntok):
    import torch
    from llmlingua import PromptCompressor

    engine = PromptCompressor(config["model"], use_llmlingua2=True, device_map="cuda",
                              model_config={"trust_remote_code": False, "local_files_only": True})

    def compressor(raw):
        with torch.inference_mode():
            out = engine.compress_prompt([raw], rate=agent.RATE, force_tokens=list(agent.FORCE_TOKENS))
        return out["compressed_prompt"]

    return lambda item: agent.llmlingua_compression(item, compressor, ntok)


def release_torch():
    import torch
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def check_versions(deployment):
    expected = deployment["agent"]["package_versions"]
    need(all(version(name) == val for name, val in expected.items()), "Reader runtime package versions not reproduced")
    return {name: version(name) for name in [*expected, "llmlingua", "tiktoken", "nltk", "accelerate"]}
