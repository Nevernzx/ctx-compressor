"""Detached, one-shot Task 20 smoke run on the Task 19/20 GPU 1 gate."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts import agent_v1_data as data
from scripts import agent_v1_disk as disk, agent_v1_engines as engines
from scripts import run_formal_grpo_pipeline as old
from scripts import run_grpo_r4_pipeline as grpo
from scripts.run_sft_v3_pipeline import ResumedPipeline
from src import agent_v1 as agent, agent_v1_metrics
from src.model_inference import compact
from src.reader_heldout_inputs import EXTERNAL, need, read, rows, write_new
from src.reranker_v1 import record, require_new

WORK, SMALL = data.WORK, data.SMALL
LOCK = EXTERNAL / "rt/gpu1_t19_t20.lock"
TASK19_STATUS = EXTERNAL / "test_v1/status.json"
GRPO = EXTERNAL / "envs/grpo/bin/python"
CPU = PROJECT / ".conda/envs/ctxc/bin/python"
PID = WORK / "pipeline.pid"
STATUS = WORK / "status.json"
LOGS = WORK / "logs"
RUNTIME = EXTERNAL / "rt/t20/initial"
ATTEMPT = None
ATTEMPTS = ("try2", "try3")
SMOKE = WORK / "smoke"
PROMPTS = SMALL / "prompts.json"
P_RECEIPT = SMALL / "phase_p.json"
CPU_TESTS = SMALL / "cpu_tests.json"
MONITOR_TESTS = SMALL / "monitor_scope_cpu_tests.json"
SMOKE_REPORT = SMALL / "smoke_results.json"
SMOKE_CONFIG = SMALL / "smoke_run_config.json"
LOG = LOGS / "smoke.log"
R4_ROWS = tuple(row for row in agent.ROWS if row.startswith("r4"))
ORDINARY = tuple(row for row in agent.ROWS if row not in R4_ROWS)
CACHE_KEYS = ("TMPDIR", "TMP", "TEMP", "RAY_TMPDIR", "VLLM_RPC_BASE_PATH", "XDG_CACHE_HOME", "HF_HOME",
              "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH")


def select_attempt(name):
    global ATTEMPT, WORK, PID, STATUS, LOGS, RUNTIME, SMOKE, PROMPTS, P_RECEIPT, MONITOR_TESTS, SMOKE_REPORT, SMOKE_CONFIG, LOG
    need(name in ATTEMPTS, f"Unknown Task 20 attempt {name}")
    ATTEMPT = name
    WORK = data.WORK / name
    small = data.SMALL / name
    PROMPTS = (small if name == "try3" else data.SMALL) / "prompts.json"
    PID, STATUS, LOGS, SMOKE = (WORK / name for name in ("pipeline.pid", "status.json", "logs", "smoke"))
    RUNTIME = EXTERNAL / "rt/t20" / name
    P_RECEIPT, MONITOR_TESTS, SMOKE_REPORT, SMOKE_CONFIG = (small / name for name in (
        "phase_p.json", "monitor_scope_cpu_tests.json", "smoke_results.json", "smoke_run_config.json"))
    LOG = LOGS / "smoke.log"


def invocation(python, action):
    command = [str(python), "-B", "-u", "-m", "scripts.agent_v1_smoke", action]
    return command + (["--attempt", ATTEMPT] if ATTEMPT else [])


def prompt_spec():
    return {
        "system": agent.SYSTEM,
        "user_format": "{question_number}. {question}",
        "history_round_format": "{raw_output}\nObservation:\n{observation}\n\n",
        "observation_line": "[P{first_appearance_number}] {title}: {text}",
        "empty_observation": agent.EMPTY_OBSERVATION, "invalid_observation": agent.INVALID_OBSERVATION,
        "forced_user": agent.FORCE_MESSAGE,
        "assembly": "One Qwen assistant transcript with stable tokenized round chunks. Forced finish adds a user message.",
        "agent_generation_prefill": agent.THOUGHT_PREFILL,
        "prefill_applies_to": "Every strategy and generation, including the first round, each continuation, and forced finish.",
        "raw_output": "The exact prefill joined to the generated continuation before parsing and archiving.",
        "forced_history_guard": "If role framing would exceed the hard window, keep the triggering turn in the archive but omit it from the final prompt",
        "enable_thinking": False, "prompt_limit": agent.PROMPT_LIMIT, "max_searches": agent.MAX_SEARCHES,
        "max_consecutive_invalid": agent.MAX_INVALID, "top_k": agent.TOP_K,
    }


def freeze_prompt():
    write_new(PROMPTS, prompt_spec())


def runtime_environment(label):
    with patch.object(grpo, "RUNTIME", RUNTIME), patch.object(grpo.qa, "SHORT", RUNTIME / "tmp"):
        env = grpo.worker_environment(label, GRPO)
    need(RUNTIME.resolve() == RUNTIME and RUNTIME.stat().st_dev != Path("/").stat().st_dev,
         "Task 20 runtime must be on the external filesystem")
    need(all(RUNTIME in Path(env[key]).resolve().parents for key in CACHE_KEYS),
         "Task 20 cache or IPC path left the external runtime directory")
    return env


def new_env():
    env = runtime_environment("smoke")
    cards = read(engines.MODEL_CARDS)
    env.update(
        PYTHONPATH=os.pathsep.join((cards["llmlingua_wheel"]["vendor"], str(PROJECT))),
        TIKTOKEN_CACHE_DIR=str(engines.TIKTOKEN_CACHE),
    )
    return env


def frozen_code():
    paths = (
        PROJECT / "src/agent_v1.py",
        PROJECT / "src/agent_v1_metrics.py",
        PROJECT / "scripts/agent_v1_data.py",
        PROJECT / "scripts/agent_v1_engines.py",
        PROJECT / "scripts/agent_v1_smoke.py",
        PROJECT / "scripts/agent_v1_queue.py",
        PROJECT / "scripts/agent_v1_disk.py",
        PROJECT / "tests/test_agent_v1.py",
        PROJECT / "tests/test_agent_v1_monitor_paths.py",
        PROJECT / "tests/test_agent_v1_disk.py",
        PROJECT / "tests/fixtures/agent_v1_rr5.json",
    )
    return {str(p.relative_to(PROJECT)): record(p) for p in paths}


def phase_p(*, verify_models):
    content = read(data.RECEIPT)
    tests = read(CPU_TESTS)
    need(tests["status"] == "passed" and tests["failures"] == tests["errors"] == tests["skipped"] == 0
         and tests["tests"] > 0, "Task 20 full offline CPU suite is required")
    scope = read(MONITOR_TESTS)
    need(scope["status"] == "passed" and scope["signed_history_unchanged"] is True
         and scope["tests"] >= tests["tests"] and not any(scope[k] for k in ("errors", "failures", "skipped", "cpu_guard_events")),
         "Task 20 controller isolation and full CPU tests are required")
    code = frozen_code()
    need(all(code[name] == item for name, item in scope["controller_sources"].items()),
         "Task 20 controller or regression changed after CPU tests")
    need(content["status"] == "passed" and read(PROMPTS) == prompt_spec(), "Task 20 data or registered prompt differs")
    for dataset, info in content["datasets"].items():
        need(info["main"]["n_tasks"] == data.N_MAIN and info["smoke"]["n_tasks"] == data.N_SMOKE
             and set(info["main"]["question_ids"]).isdisjoint(info["smoke"]["question_ids"])
             and info["main_seed"] == data.SEEDS[dataset] and info["smoke_seed"] == data.SEEDS[dataset] + 100,
             "Task 20 sampled questions differ")
        for item in (info["main"]["file"], info["smoke"]["file"], info["corpus"], *info["bm25_index"]):
            engines.checked_artifact(item)
    return {
        "status": "passed", "data": record(data.RECEIPT), "cpu_tests": record(CPU_TESTS), "prompt": record(PROMPTS),
        "code": code, "controller_scope_tests": record(MONITOR_TESTS),
        "assets": engines.registered_assets(verify_models=verify_models),
    }


def registries():
    import bm25s

    datasets = read(data.RECEIPT)["datasets"]
    for label, receipt in datasets.items():
        tasks = list(rows(Path(receipt["smoke"]["file"]["path"])))
        corpus = list(rows(Path(receipt["corpus"]["path"])))
        index = bm25s.BM25.load(str(Path(receipt["bm25_index"][0]["path"]).parent))
        need(len(tasks) == data.N_SMOKE and len(corpus) == receipt["n_paragraphs"] and index.scores["num_docs"] == len(corpus),
             f"Smoke source length mismatch: {label}")
        yield (label, tasks, corpus, index)


class SmokeArchive:
    def __init__(self, path):
        self.file = path.open("x", encoding="utf-8")
        self.seen = set()

    def finish(self, ep):
        detail = ep.record()
        key = detail["dataset"], detail["row"], detail["task"]
        need(key not in self.seen, f"Smoke episode already written: {key}")
        self.seen.add(key)
        print(compact(detail), file=self.file)
        self.file.flush()


def model_pass(llm, deployment, active, indexes, archive, runtime=None):
    """Advance one generation round for each active episode, saving completions."""
    pending = []
    width = deployment["agent"]["params"]["max_num_seqs"]
    for offset in range(0, len(active), width):
        if runtime:
            runtime.check_storage()
        batch = active[offset:offset + width]
        results = engines.generate_agent(llm, deployment, batch)
        for ep, (tokens, continuation, n, physical_prefix) in zip(batch, results, strict=True):
            if runtime:
                runtime.check_storage()
            index = indexes[ep.task["dataset"]]
            result = ep.accept_reply(tokens, continuation, n, lambda q: agent.search_bm25(index, q))
            ep.turns[-1]["vllm_cached_tokens"] = physical_prefix
            if ep.completed():
                archive.finish(ep)
            if result:
                pending.append(result)
    return pending


def ordinary(episode_rows, llm, deployment, indexes, runtime, archive, ntok):
    for name in ORDINARY:
        if name.startswith("RR-τ"):
            codec = engines.new_reranker(deployment["reranker"])
        elif name == "LLMLingua-2":
            codec = engines.new_llmlingua(deployment["llmlingua2"], ntok)
        else:
            codec = agent.baseline_compression
        runtime.gpu(f"ready:{name}")
        active = episode_rows[name]
        while active:
            pending = model_pass(llm, deployment, active, indexes, archive, runtime)
            for item in pending:
                runtime.check_storage()
                agent.execute_compression(item, codec)
            active = [ep for ep in active if not ep.completed()]
        del codec
        engines.release_torch()
        print(f"Completed smoke row {name} at {old.now()}", flush=True)


def r4_search_boundary(llm, deployment, active, indexes, archive, runtime=None):
    """Advance r4 episodes until local compressor generation is required."""
    while any(not ep.completed() for ep in active):
        pending = model_pass(llm, deployment, [ep for ep in active if not ep.completed()], indexes, archive, runtime)
        if pending:
            return pending
    return []


def run_smoke():
    from tokenizers import Tokenizer

    engines.require_cuda()
    need(read(STATUS)["status"] == "running" and read(STATUS)["lock"]["held"], "Task 20 admission records missing")
    need(Path(sys.executable).resolve() == GRPO.resolve(), "Smoke must use the existing GRPO environment")
    require_new(SMOKE, SMOKE_CONFIG, SMOKE_REPORT)
    frozen = read(P_RECEIPT)
    need(frozen == phase_p(verify_models=False), "Frozen Task 20 setup changed")
    need(os.environ.get("TIKTOKEN_CACHE_DIR") == str(engines.TIKTOKEN_CACHE), "TIKTOKEN_CACHE_DIR changed")
    if any(key.lower() in ("http_proxy", "https_proxy", "all_proxy") for key in os.environ):
        raise RuntimeError("Unexpected proxy in local Task 20 worker")

    SMOKE.mkdir()
    archive = SmokeArchive(SMOKE / "episodes.jsonl")
    telemetry = SMOKE / "gpu_snapshots.jsonl"
    with telemetry.open("x"):
        pass
    root_trace = disk.RootTelemetry(SMOKE)
    runtime = engines.RuntimeMeasurements(telemetry, root_trace)
    root_check = SMOKE / "root_usage_check.json"
    batches = 0
    started = None
    try:
        root_trace.start()
        deployment = frozen["assets"]
        packages = engines.check_versions(deployment)
        tokenizer = engines.offline_tokenizer(deployment["agent"]["params"]["tokenizer"])
        count_tokenizer = Tokenizer.from_file(str(data.TOKENIZER))
        ntok = lambda text: len(count_tokenizer.encode(text, add_special_tokens=False).ids)
        episodes = {row: [] for row in agent.ROWS}
        indexes = {}
        for label, tasks, corpus, index in registries():
            indexes[label] = index
            for row, batch in episodes.items():
                batch.extend(agent.Episode(t, row, corpus, tokenizer) for t in tasks)
        r4_episodes = [ep for row in R4_ROWS for ep in episodes[row]]

        started = time.monotonic()
        with engines.engine_session("agent", deployment, runtime, LOG) as llm:
            ordinary(episodes, llm, deployment, indexes, runtime, archive, ntok)
            pending = r4_search_boundary(llm, deployment, r4_episodes, indexes, archive, runtime)
        del llm
        engines.release_torch()
        while pending:
            batches += 1
            with engines.engine_session("r4", deployment, runtime, LOG) as llm:
                engines.execute_r4(llm, deployment, pending, runtime)
            del llm
            engines.release_torch()
            with engines.engine_session("agent", deployment, runtime, LOG) as llm:
                pending = r4_search_boundary(llm, deployment, r4_episodes, indexes, archive, runtime)
            del llm
            engines.release_torch()
            print(f"Completed smoke r4 compressor handoff {batches} at {old.now()}", flush=True)
    finally:
        elapsed = time.monotonic() - started if started is not None else 0.0
        archive.file.close()
        root_trace.close()
        if root_trace.policy.closing is not None:
            root_summary = root_trace.policy.report()
            write_new(root_check, {
                **root_summary, "mount": "/", "measurement": "shutil.disk_usage(/).used and .free",
                "fast_sample_errors": root_trace.errors, "samples": root_trace.samples,
                "root_telemetry": record(root_trace.trace), "workplace_telemetry": record(root_trace.workplace_trace),
                "gpu_snapshots": record(telemetry), "episodes": record(SMOKE / "episodes.jsonl"),
            })

    root_trace.policy.check()
    need(len(archive.seen) == 3 * data.N_SMOKE * len(agent.ROWS), "Smoke episode count differs")
    need(frozen_code() == frozen["code"] and record(PROMPTS) == frozen["prompt"], "Smoke code or prompts changed")
    measurements = agent_v1_metrics.analyze(list(rows(SMOKE / "episodes.jsonl")), data.N_SMOKE)
    startup = sum(load["startup_seconds"] + load["shutdown_seconds"] for load in runtime.loads)
    eta = startup + (elapsed - startup) * data.N_MAIN / data.N_SMOKE
    write_new(SMOKE_REPORT, {
        "status": "smoke_complete", "episodes": record(SMOKE/"episodes.jsonl"), "results": measurements,
        "wall_seconds": elapsed, "model_start_and_shutdown_seconds": startup,
        "main_eta_seconds_extrapolated": eta, "n_smoke_episodes": len(archive.seen),
        "eta_caveat": "Estimate scales smoke work by 10, holds measured model startup/shutdown overhead fixed, and excludes time waiting for GPU admission. The main runtime may differ.",
        "awaiting_main_approval": True,
    })
    write_new(SMOKE_CONFIG, {
        "status": "frozen_after_smoke", "datasets": read(data.RECEIPT),
        "task19_order": "Task 19 evaluation_complete before Task 20 smoke", "lock": str(LOCK),
        "gpu_index": 1, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "assets": deployment, "package_versions": packages, "prompt": prompt_spec(), "prompt_file": frozen["prompt"],
        "code_sha256": frozen["code"], "rows": agent.ROWS,
        "smoke_modifications": [{
            "component": "controller",
            "change": "Bind inherited monitoring to the Task 20 directory and place IPC scratch there.",
            "monitor_log": str(WORK / "resource_monitor.jsonl"), "ipc_dir": os.environ["TMPDIR"],
            "cpu_verification": frozen["controller_scope_tests"],
        }, {
            "component": "runtime storage",
            "change": "Move all process caches and temporary files under rt/t20 and sample root use at 1 s.",
            "directories": {key: os.environ[key] for key in CACHE_KEYS},
        }, {
            "component": "agent response format",
            "change": "Prefill Thought: for every Agent generation. Archive it with the continuation and keep the existing parser.",
            "prefill": agent.THOUGHT_PREFILL, "all_strategies": True, "includes_forced_finish": True,
            "reason": "The prior assistant transcript could continue the Observation by fabricating a passage before Thought/Action.",
        }],
        "root_usage_check": record(root_check), "ipc_check": record(WORK / "ipc_check.json") if ATTEMPT else None,
        "root_before_bytes": root_summary["root_before_bytes"], "root_after_bytes": root_summary["root_after_bytes"],
        "peak_root_bytes": root_summary["peak_root_bytes"],
        "root_storage_policy": {
            "a_sustained_growth_limit_bytes": disk.ROOT_LIMIT_BYTES,
            "b_attributed_growth_limit_bytes": disk.ROOT_LIMIT_BYTES,
            "c_minimum_available_bytes": disk.MIN_ROOT_FREE_BYTES,
            "d_unattributed_transient_global_highs": "record time, bytes, and engine",
        },
        "gpu_snapshots": record(telemetry),
        "peak_observed_gpu1_mib": max(runtime.peaks), "engine_loads": runtime.loads, "r4_handoffs": batches,
        "smoke_memory_plan": "Keep one 8B engine running for ordinary rows; hand off GPU 1 between frozen 8B and r4 engines by round.",
        "physical_prefix_cache": "Ordinary rows reuse the resident engine. r4 smoke restarts the 8B engine, so physical hits cannot be carried across handoffs. Reported adjacent-token prefix costs use the preregistered formula for every episode.",
        "r4_latency_scope": "Batch submission to return, attributed to each request in the batch",
        "open_provence": read(engines.MODEL_CARDS)["open_provence"],
        "results": record(SMOKE_REPORT), "main_approval": False,
    })
    print(f"Task 20 smoke archived {len(archive.seen)} episodes in {elapsed:.3f} seconds", flush=True)


def require_task19(path):
    proof = read(path)
    need(proof["status"] == "evaluation_complete", "Task 19 G3 must complete before Task 20 smoke")
    return record(path)


class SmokePipeline(ResumedPipeline):
    def save(self):
        super().save()
        if self.record["status"] in {"running", "waiting_gpu"}:
            sample = self.record.get("last_monitor")
            if sample and sample["overlay_free_bytes"] < disk.MIN_ROOT_FREE_BYTES:
                self.record["root_free_violation"] = dict(sample)
                raise disk.RootLimitError(f"Task 20 storage rule c: {sample['overlay_free_bytes']} root bytes available")


def detached():
    need(os.getpid() == os.getsid(0) and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Expected detached CPU controller")
    require_new(PID, STATUS, LOGS, RUNTIME / "smoke", SMOKE, SMOKE_CONFIG, SMOKE_REPORT,
                WORK / "resource_monitor.jsonl", WORK / "pipeline_monitor.jsonl")
    def stop(signum, _frame):
        raise InterruptedError(f"Task 20 controller received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    with PID.open("x") as file:
        print(os.getpid(), file=file)
    LOGS.mkdir()
    workflow = SmokePipeline(monitor_dir=WORK)
    lock = {"path": str(LOCK), "held": False, "order": ["19", "20"]}
    workflow.record.update(lock=lock, stage="gpu_lock", status="waiting_lock")
    with patch.multiple(old, RUN=WORK, LOGS=LOGS, STATUS=STATUS, TRAIN=WORK/"no_training"), \
            patch.object(grpo.c, "LOCK", LOCK):
        try:
            workflow.save()
            need(read(P_RECEIPT) == phase_p(verify_models=False), "Task 20 setup changes after launch")
            with grpo.exclusive_gpu():
                lock["held"] = True
                workflow.record["task19"] = require_task19(TASK19_STATUS)
                env = new_env()
                if ATTEMPT:
                    write_new(WORK / "ipc_check.json", {
                        "path": env["VLLM_RPC_BASE_PATH"], "estimated_path_bytes": len(os.fsencode(env["VLLM_RPC_BASE_PATH"])) + 43,
                        "limit_bytes": 100, "cache_paths": {key: env[key] for key in CACHE_KEYS},
                    })
                workflow.stage("smoke", invocation(GRPO, "run-smoke"), True, env)
                report = read(SMOKE_REPORT)
                config = read(SMOKE_CONFIG)
                need(report["status"] == "smoke_complete" and config["status"] == "frozen_after_smoke"
                     and report["awaiting_main_approval"] and report["n_smoke_episodes"] == 3 * data.N_SMOKE * len(agent.ROWS),
                     "Incomplete Task 20 smoke")
                workflow.active["receipt"] = record(SMOKE_REPORT)
                workflow.active["run_config"] = record(SMOKE_CONFIG)
                workflow.record.update(status="smoke_complete", stage="waiting_main_approval", main_approval=False)
        except BaseException as exc:
            workflow.record.update(status="failed", first_error=str(exc), traceback=traceback.format_exc())
            if workflow.active is not None:
                workflow.active["status"] = "failed"
            raise
        finally:
            lock["held"] = False
            workflow.record.update(ended_utc=old.now(), stage_pid=None)
            workflow.save()


def launch():
    need(Path.cwd() == PROJECT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Launch on 5880 with CUDA hidden")
    require_task19(TASK19_STATUS)
    need(disk.root_usage().free >= disk.MIN_ROOT_FREE_BYTES, "Task 20 storage rule c before launch")
    require_new(WORK/"launcher.log", P_RECEIPT, PID, STATUS, LOGS, RUNTIME / "controller", SMOKE, SMOKE_CONFIG, SMOKE_REPORT)
    write_new(P_RECEIPT, phase_p(verify_models=True))
    grpo.clear_proxy()
    with (WORK/"launcher.log").open("xb") as log:
        proc = subprocess.Popen(invocation(CPU, "detached"),
                                cwd=PROJECT, env={**os.environ, **runtime_environment("controller"),
                                                  "CUDA_VISIBLE_DEVICES":"", "PYTHONDONTWRITEBYTECODE":"1"},
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                close_fds=True, start_new_session=True)
    print(f"Task 20 detached smoke PID {proc.pid}. Status {STATUS}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("freeze-prompt", "launch", "detached", "run-smoke"))
    cli.add_argument("--attempt", choices=ATTEMPTS)
    args = cli.parse_args()
    if args.attempt:
        select_attempt(args.attempt)
    {"freeze-prompt": freeze_prompt, "launch": launch, "detached": detached, "run-smoke": run_smoke}[args.action]()


if __name__ == "__main__":
    main()
