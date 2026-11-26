"""Run the approved Task 20 main experiment once on the shared GPU 1 slot."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from scripts import agent_v1_data as data, agent_v1_disk as disk, agent_v1_engines as engines
from scripts import agent_v1_queue as queue, agent_v1_residency as residency, agent_v1_smoke as smoke
from scripts import run_formal_grpo_pipeline as old, run_grpo_r4_pipeline as grpo
from src import agent_v1 as agent
from src.reader_heldout_inputs import EXTERNAL, need, read, rows, write_new
from src.reranker_v1 import record, require_new

ROOT, SMALL = data.WORK / "main_try2", data.SMALL / "main_try2"
STATUS, PID, LOGS = ROOT / "status.json", ROOT / "pipeline.pid", ROOT / "logs"
CACHE = EXTERNAL / "rt/t20/main_try2"
PRE, CPU_TEST = SMALL / "preflight.json", SMALL / "cpu_tests.json"
FROZEN = data.SMALL / "try3/smoke_run_config.json"
AUDIT = data.SMALL / "try3/smoke_archive_audit.json"
CONFIG = data.SMALL / "run_config.json"
GPU_RECEIPT = SMALL / "gpu_worker.json"
GPU_LOG = LOGS / "main.log"
EPISODES = ROOT / "episodes.jsonl"
AGENT_UTILIZATION = 0.50
R4_COHORT = 64
PILOT = False
PILOT_ROOT = data.WORK / "main_pilot"
PILOT_SMALL = data.SMALL / "main_pilot"
PILOT_DATASETS = ("musique", "hotpotqa")


def enable_pilot():
    global PILOT, ROOT, SMALL, STATUS, PID, LOGS, CACHE, PRE, CONFIG, GPU_RECEIPT, GPU_LOG, EPISODES
    PILOT = True
    ROOT, SMALL = PILOT_ROOT, PILOT_SMALL
    STATUS, PID, LOGS = ROOT / "status.json", ROOT / "pipeline.pid", ROOT / "logs"
    CACHE = EXTERNAL / "rt/t20/main_pilot"
    PRE, CONFIG, GPU_RECEIPT = SMALL / "preflight.json", SMALL / "run_config.json", SMALL / "gpu_worker.json"
    GPU_LOG, EPISODES = LOGS / "main.log", ROOT / "episodes.jsonl"


def selected_task_count():
    return len(PILOT_DATASETS) if PILOT else data.N_MAIN * 3


def argv(python, action):
    command = [str(python), "-B", "-u", "-m", "scripts.agent_v1_main", action]
    return command + (["--pilot"] if PILOT else [])


def effective_assets(frozen):
    result = deepcopy(frozen)
    result["agent"]["params"]["gpu_memory_utilization"] = AGENT_UTILIZATION
    return result


def agent_runtime_assets(assets):
    deployment = deepcopy(assets)
    deployment["agent"]["params"]["worker_extension_cls"] = residency.EXTENSION
    return deployment


def current_source():
    paths = ("scripts/agent_v1_main.py", "scripts/agent_v1_residency.py", "tests/test_agent_v1_main.py")
    return {name: record(smoke.PROJECT / name) for name in paths}


def pilot_evidence(source, assets, datasets):
    status = read(PILOT_ROOT / "status.json")
    config_path = PILOT_SMALL / "run_config.json"
    config = read(config_path)
    need(status["status"] == "evaluation_complete" and not status["lock"]["held"] and status["stage_pid"] is None
         and status["stages"][0]["exit_code"] == 0 and status["stages"][0]["run_config"] == record(config_path),
         "The two-task main-path pilot has not passed")
    need(config["scope"] == "pilot" and config["main_code"] == source and config["assets"] == assets,
         "Pilot source or deployment differs from the formal run")
    runtime = config["runtime"]
    need(runtime["n_episodes"] == 2 * len(agent.ROWS), "Unexpected pilot episode count")
    for key in ("episodes", "residency_check", "root_usage_check"):
        engines.checked_artifact(runtime[key])
    canary = read(Path(runtime["residency_check"]["path"]))
    need(canary["status"] == "passed" and canary["next_cached_tokens"] > 0
         and read(Path(runtime["root_usage_check"]["path"]))["status"] == "passed",
         "Pilot cache or storage check failed")
    expected = {}
    for label in PILOT_DATASETS:
        part = datasets[label]["smoke"]
        engines.checked_artifact(part["file"])
        first = next(rows(Path(part["file"]["path"])))
        need(set(q["id"] for q in first["questions"]).isdisjoint(datasets[label]["main"]["question_ids"]),
             f"Pilot leaked into the formal tasks for {label}")
        expected[label] = first
    seen, calls = Counter(), Counter()
    for ep in rows(Path(runtime["episodes"]["path"])):
        need(ep["dataset"] in expected and ep["row"] in agent.ROWS, "Unregistered pilot row")
        task = expected[ep["dataset"]]
        need(ep["task"] == task["task"] and ep["questions"] == task["questions"] and ep["turns"],
             "Pilot did not use the frozen smoke task")
        seen[ep["dataset"], ep["row"]] += 1
        calls[ep["row"]] += sum(turn.get("compression") is not None for turn in ep["turns"])
    need(seen == Counter({(label, row): 1 for label in PILOT_DATASETS for row in agent.ROWS}),
         "Pilot did not cover both tasks and all strategies")
    need(all(calls[row] > 0 for row in (*smoke.R4_ROWS, "RR-τ append", "RR-τ reselect", "LLMLingua-2")),
         "Pilot did not exercise every model-backed compressor")
    engines.checked_artifact(runtime["weight_transfers"])
    need(len(list(rows(Path(runtime["weight_transfers"]["path"])))) > 1,
         "Pilot only exercised the synthetic r4 transfer")
    return {"status": record(PILOT_ROOT / "status.json"), "config": record(config_path),
            "cache": runtime["residency_check"], "episodes": runtime["episodes"], "rows": len(agent.ROWS)}


def preflight(*, verify_models):
    frozen = read(FROZEN)
    audit = read(AUDIT)
    need(audit["status"] == "waiting_main_approval" and audit["frozen_run_config"] == record(FROZEN),
         "The approved try3 archive no longer matches its config")
    need(frozen["status"] == "frozen_after_smoke" and frozen["rows"] == list(agent.ROWS),
         "Unexpected try3 config")
    need(frozen["code_sha256"] == smoke.frozen_code(), "Frozen try3 code changed")
    prompt_path = data.SMALL / "try3/prompts.json"
    need(frozen["prompt_file"] == record(prompt_path) and read(prompt_path) == smoke.prompt_spec(),
         "Frozen try3 prompt changed")
    proof = read(CPU_TEST)
    source = current_source()
    need(proof["status"] == "passed" and proof["tests"] >= 897 and proof["source"] == source,
         "Main controller's full CPU tests are missing")
    assets = engines.registered_assets(verify_models=verify_models)
    need(assets == frozen["assets"], "Registered model, compressor settings, or generation changed")
    need(frozen["datasets"] == read(data.RECEIPT), "Frozen Task 20 dataset receipt changed")
    inputs = read(data.RECEIPT)["datasets"]
    for label, dataset in inputs.items():
        need(dataset["main"]["n_tasks"] == data.N_MAIN and dataset["smoke"]["n_tasks"] == data.N_SMOKE,
             f"Unexpected task counts for {label}")
        need(len(set(dataset["main"]["question_ids"])) == data.N_MAIN * agent.K
             and set(dataset["main"]["question_ids"]).isdisjoint(dataset["smoke"]["question_ids"]),
             f"Formal-smoke partition differs for {label}")
        for item in (dataset["main"]["file"], dataset["corpus"], *dataset["bm25_index"]):
            engines.checked_artifact(item)
    approved = queue.task19_evidence(read(smoke.TASK19_STATUS))
    effective = effective_assets(assets)
    pilot = None if PILOT else pilot_evidence(source, effective, inputs)
    if PILOT:
        for label in PILOT_DATASETS:
            engines.checked_artifact(inputs[label]["smoke"]["file"])
    return {"status": "passed", "scope": "pilot" if PILOT else "formal",
            "n_tasks": selected_task_count(), "pilot": pilot, "try3": record(FROZEN), "try3_archive": record(AUDIT),
            "prompt": record(prompt_path), "frozen_code": frozen["code_sha256"], "main_code": source,
            "cpu_tests": record(CPU_TEST), "task19": approved, "data": record(data.RECEIPT),
            "assets": effective, "lock": str(smoke.LOCK), "gpu_index": 1,
            "r4_cohort": R4_COHORT, "approval": "Claude approved main after independent try3 review",
            "automatic_retry": False}


def environment(label):
    with patch.object(smoke, "RUNTIME", CACHE):
        values = smoke.runtime_environment(label)
    vendor = read(engines.MODEL_CARDS)["llmlingua_wheel"]["vendor"]
    values.update(PYTHONPATH=os.pathsep.join((vendor, str(smoke.PROJECT))),
                  TIKTOKEN_CACHE_DIR=str(engines.TIKTOKEN_CACHE))
    return values


def preparation():
    import bm25s

    indexes, registries = {}, {}
    for label, item in read(data.RECEIPT)["datasets"].items():
        if PILOT and label not in PILOT_DATASETS:
            continue
        part = item["smoke" if PILOT else "main"]
        tasks = list(rows(Path(part["file"]["path"])))
        corpus = list(rows(Path(item["corpus"]["path"])))
        index = bm25s.BM25.load(str(Path(item["bm25_index"][0]["path"]).parent))
        need(len(tasks) == part["n_tasks"] and {t["dataset"] for t in tasks} == {label}
             and len(corpus) == item["n_paragraphs"] and index.scores["num_docs"] == len(corpus),
             f"Runtime registry mismatch for {label}")
        flat = [q["id"] for task in tasks for q in task["questions"]]
        need(flat == part["question_ids"], f"Ordered questions differ for {label}")
        indexes[label] = index
        registries[label] = (tasks[:1] if PILOT else tasks), corpus
    return indexes, registries


def event(kind, **facts):
    entry = {"utc": old.now(), "kind": kind, **facts}
    with (ROOT / "events.jsonl").open("a", encoding="utf-8") as out:
        print(json.dumps(entry, ensure_ascii=False), file=out)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def run_rows(llm, deployment, runtime, archive, transfer_log):
    from tokenizers import Tokenizer

    tokenizer = engines.offline_tokenizer(deployment["agent"]["params"]["tokenizer"])
    count = Tokenizer.from_file(str(data.TOKENIZER))

    def ntok(text):
        return len(count.encode(text, add_special_tokens=False).ids)

    indexes, registries = preparation()
    expected = sum(len(tasks) for tasks, _ in registries.values())
    need(expected == selected_task_count(), "Unexpected selected task count")
    for name in smoke.ORDINARY:
        residency.clear_cache(llm)
        episode_rows = {name: [agent.Episode(task, name, corpus, tokenizer)
                               for tasks, corpus in registries.values() for task in tasks]}
        started = time.monotonic()
        with patch.object(smoke, "ORDINARY", (name,)):
            smoke.ordinary(episode_rows, llm, deployment, indexes, runtime, archive, ntok)
        need(sum(key[1] == name for key in archive.seen) == expected, f"Incomplete ordinary row {name}")
        event("row_complete", row=name, episodes=len(archive.seen), wall_seconds=time.monotonic() - started)

    reference = residency.state(llm)
    for name in smoke.R4_ROWS:
        residency.clear_cache(llm)
        episodes = [agent.Episode(task, name, corpus, tokenizer)
                    for tasks, corpus in registries.values() for task in tasks]
        row_start = time.monotonic()
        for offset in range(0, len(episodes), R4_COHORT):
            active = episodes[offset:offset + R4_COHORT]
            started = time.monotonic()
            waves = 0
            pending = smoke.r4_search_boundary(llm, deployment, active, indexes, archive, runtime)
            while pending:
                waves += 1
                residency.with_compressor(
                    llm, reference, deployment, runtime,
                    lambda compressor: engines.execute_r4(compressor, deployment, pending, runtime),
                    transfer_log,
                )
                active.reverse()
                pending = smoke.r4_search_boundary(llm, deployment, active, indexes, archive, runtime)
            event("cohort_complete", row=name, offset=offset, n=len(active), waves=waves,
                  episodes=len(archive.seen), wall_seconds=time.monotonic() - started)
        need(sum(key[1] == name for key in archive.seen) == expected, f"Incomplete r4 row {name}")
        event("row_complete", row=name, episodes=len(archive.seen), wall_seconds=time.monotonic() - row_start)


def worker():
    engines.require_cuda()
    need(Path(sys.executable).resolve() == smoke.GRPO.resolve(), "Unexpected Task 20 worker environment")
    state = read(STATUS)
    need(state["status"] == "running" and state["lock"]["held"], "Main worker was not admitted")
    proof = read(PRE)
    need(proof == preflight(verify_models=False), "Main preflight changed after admission")
    need(all(CACHE in Path(os.environ[key]).parents for key in smoke.CACHE_KEYS), "Unexpected runtime storage paths")
    require_new(EPISODES, GPU_RECEIPT, SMALL / "residency_check.json", ROOT / "gpu_snapshots.jsonl",
                ROOT / "root_usage_check.json", ROOT / "weight_transfers.jsonl")
    telemetry = disk.RootTelemetry(ROOT)
    runtime = engines.RuntimeMeasurements(ROOT / "gpu_snapshots.jsonl", telemetry)
    archive = smoke.SmokeArchive(EPISODES)
    transfer_log = ROOT / "weight_transfers.jsonl"
    started = time.monotonic()
    try:
        telemetry.start()
        versions = engines.check_versions(proof["assets"])
        residency.owned_apps()
        with engines.engine_session("agent", agent_runtime_assets(proof["assets"]), runtime, GPU_LOG) as llm:
            canary = residency.prove_cache(llm, proof["assets"], runtime, transfer_log)
            write_new(SMALL / "residency_check.json", canary)
            event("cache_proven", next_cached_tokens=canary["next_cached_tokens"],
                  first_prompt_tokens=canary["first_prompt_tokens"])
            run_rows(llm, proof["assets"], runtime, archive, transfer_log)
    finally:
        elapsed = time.monotonic() - started
        archive.file.close()
        telemetry.close()
        if telemetry.policy.closing is not None:
            write_new(ROOT / "root_usage_check.json", {**telemetry.policy.report(), "samples": telemetry.samples,
                      "fast_sample_errors": telemetry.errors, "root_telemetry": record(telemetry.trace),
                      "workplace_telemetry": record(telemetry.workplace_trace)})
    telemetry.policy.check()
    need(len(archive.seen) == selected_task_count() * len(agent.ROWS), "Unexpected final episode count")
    need(proof == preflight(verify_models=False), "Source or data changed during main")
    residency.wait_for_engine_release(None)
    write_new(GPU_RECEIPT, {
        "status": "passed", "episodes": record(EPISODES), "n_episodes": len(archive.seen),
        "wall_seconds": elapsed, "engine_loads": runtime.loads, "versions": versions,
        "peak_observed_gpu1_mib": max(runtime.peaks), "gpu_snapshots": record(runtime.path),
        "root_usage_check": record(ROOT / "root_usage_check.json"),
        "weight_transfers": record(transfer_log), "residency_check": record(SMALL / "residency_check.json"),
    })


class MainPipeline(smoke.SmokePipeline):
    def save(self):
        super().save()
        sample = self.record.get("last_monitor")
        if sample is None:
            return
        utc = sample["utc"]
        if utc == getattr(self, "ownership_utc", None):
            return
        self.ownership_utc = utc
        if self.record["status"] in ("running", "waiting_gpu"):
            apps = residency.owned_apps() if sample["pid"] is not None else residency.gpu_apps()
            with (ROOT / "gpu_ownership.jsonl").open("a", encoding="utf-8") as out:
                print(json.dumps({"utc": utc, "compute_apps_mib": apps, "memory_used_mib": sample["memory_used_mib"]}), file=out)


def detached():
    need(os.getpid() == os.getsid(0) and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Expected detached CPU supervisor")
    require_new(PID, STATUS, LOGS, EPISODES, GPU_RECEIPT, CONFIG)
    with PID.open("x", encoding="utf-8") as out:
        print(os.getpid(), file=out)
    LOGS.mkdir()

    def stop(signum, _frame):
        raise InterruptedError(f"Main controller received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    workflow = MainPipeline(monitor_dir=ROOT)
    lock = {"path": str(smoke.LOCK), "held": False, "order": [19, 20]}
    workflow.record.update(lock=lock, stage="gpu_lock", status="waiting_lock", task=20, run="pilot" if PILOT else "main")
    with patch.multiple(old, RUN=ROOT, LOGS=LOGS, STATUS=STATUS, TRAIN=ROOT / "no_training"), \
            patch.object(grpo.c, "LOCK", smoke.LOCK):
        try:
            workflow.save()
            proof = read(PRE)
            need(proof == preflight(verify_models=False), "Main setup changed after launch")
            with grpo.exclusive_gpu():
                lock["held"] = True
                workflow.record["task19"] = queue.task19_evidence(read(smoke.TASK19_STATUS))
                env = environment("gpu")
                ipc = {"path": env["VLLM_RPC_BASE_PATH"],
                       "estimated_path_bytes": len(os.fsencode(env["VLLM_RPC_BASE_PATH"])) + 43,
                       "limit_bytes": 100, "cache_paths": {k: env[k] for k in smoke.CACHE_KEYS}}
                write_new(ROOT / "ipc_check.json", ipc)
                workflow.stage("main", argv(smoke.GRPO, "worker"), True, env)
                need(not residency.owned_apps(), "Compute allocation remained after the worker exited")
                receipt = read(GPU_RECEIPT)
                need(receipt["status"] == "passed" and receipt["n_episodes"] == selected_task_count() * len(agent.ROWS),
                     "Task 20 GPU receipt is incomplete")
                root = read(ROOT / "root_usage_check.json")
                need(root["status"] == "passed", "Main did not pass the approved storage rules")
                need(read(PRE) == proof, "Main preflight changed")
                old_config = data.SMALL / "main/preflight.json"
                old_sources = read(old_config)["main_code"]
                config = {
                    "status": "evaluation_complete", "scope": proof["scope"], "n_tasks": selected_task_count(),
                    "pilot": proof["pilot"], "task": 20, "gpu_index": 1,
                    "rpc_repair": {
                        "method": "Named methods on the installed vLLM worker extension",
                        "previous_launch": record(data.SMALL / "main_try2/previous_launch.json"),
                        "original_preflight": record(old_config),
                        "source_sha256": {
                            key: {"old": old_sources[key]["sha256"], "new": value["sha256"]}
                            for key, value in proof["main_code"].items()
                        },
                    },
                    "offload": "Agent weights temporarily in host RAM. Agent engine and KV blocks stay resident on GPU 1.",
                    "agent_worker_extension": residency.EXTENSION, "r4_cohort": R4_COHORT, "schedule": "R4 reverses the episode order between compressor handoffs within a cohort.",
                    "task19": proof["task19"], "shared_lock": str(smoke.LOCK), "ipc_check": record(ROOT / "ipc_check.json"),
                    "runtime": receipt, "gpu_ownership": record(ROOT / "gpu_ownership.jsonl"), "preflight": record(PRE), "data_preflight": record(data.RECEIPT),
                    "assets": proof["assets"], "prompt": proof["prompt"], "try3": proof["try3"],
                    "frozen_code": proof["frozen_code"], "main_code": proof["main_code"],
                    "peak_observed_gpu1_mib": max(receipt["peak_observed_gpu1_mib"],
                        *(sample["memory_used_mib"] for sample in rows(ROOT / "resource_monitor.jsonl"))),
                    "storage_policy": read(FROZEN)["root_storage_policy"], "rows": list(agent.ROWS),
                    "open_provence": read(engines.MODEL_CARDS)["open_provence"],
                }
                write_new(CONFIG, config)
                workflow.active["receipt"] = record(GPU_RECEIPT)
                workflow.active["run_config"] = record(CONFIG)
                workflow.record.update(status="evaluation_complete", stage="finished")
        except BaseException as error:
            workflow.record.update(status="failed", first_error=str(error), traceback=traceback.format_exc())
            if workflow.active is not None:
                workflow.active["status"] = "failed"
            raise
        finally:
            lock["held"] = False
            workflow.record.update(ended_utc=old.now(), stage_pid=None)
            workflow.save()


def launch():
    need(Path.cwd() == smoke.PROJECT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Launch on 5880 with CUDA hidden")
    need(disk.root_usage().free >= disk.MIN_ROOT_FREE_BYTES, "Task 20 storage rule c before launch")
    require_new(ROOT, PRE, CONFIG, GPU_RECEIPT, CACHE)
    proof = preflight(verify_models=True)
    if PILOT:
        SMALL.mkdir()
    ROOT.mkdir()
    write_new(PRE, proof)
    grpo.clear_proxy()
    with (ROOT / "launcher.log").open("xb") as log:
        process = subprocess.Popen(argv(smoke.CPU, "detached"), cwd=smoke.PROJECT,
                                   env={**os.environ, **environment("controller"), "CUDA_VISIBLE_DEVICES": ""},
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   close_fds=True, start_new_session=True)
    print(f"Task 20 detached main PID {process.pid}. Status {STATUS}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("launch", "detached", "worker"))
    cli.add_argument("--pilot", action="store_true")
    args = cli.parse_args()
    if args.pilot:
        enable_pilot()
    actions = {"launch": launch, "detached": detached, "worker": worker}
    actions[args.action]()

