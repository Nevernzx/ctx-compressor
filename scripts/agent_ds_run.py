"""Task 21: the Task 20 ReAct agent with deepseek-flash as the main model (120-task subset).

Only the agent generation step changes. Episode logic, prompts, BM25 search,
budgets and all compressors are imported from Task 20 unchanged.

  probe  (CPU, foreground)  three API calls that check the prefix-completion response shape
  launch smoke|main         detached worker on GPU 1 after the gate; main needs --approved
  worker smoke|main         (internal) the detached process
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
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

from scripts import agent_ds_client as api
from scripts import agent_v1_data as data, agent_v1_disk as disk, agent_v1_engines as engines
from scripts import agent_v1_smoke as smoke
from src import agent_v1 as agent
from src.reader_heldout_inputs import EXTERNAL, need, read, rows, write_new
from src.reranker_v1 import record, require_new

WORK = EXTERNAL / "agent_ds"
SMALL = PROJECT / "eval/agent_ds"
RT = EXTERNAL / "rt/ads"
LOCK = EXTERNAL / "rt/gpu1_t19_t20.lock"
GRPO = EXTERNAL / "envs/grpo/bin/python"
TASK20_FROZEN = data.SMALL / "try3/smoke_run_config.json"
TASK20_PROMPTS = data.SMALL / "try3/prompts.json"
DATASETS = ("musique", "hotpotqa", "2wiki")
ROWS = ("no_compress", "oracle", "LLMLingua-2", "RR-τ reselect", "r4 reselect", "r4 append")
SAFE = {"no_compress": "nc", "oracle": "oracle", "LLMLingua-2": "llml2", "RR-τ reselect": "rrtau_rs",
        "r4 reselect": "r4_rs", "r4 append": "r4_ap"}
PER_DATASET = {"probe": 1, "dryrun": 1, "smoke": 3, "smoke2": 3, "main": 40}
CHUNK_PER_DATASET = 4
GATE_READINGS, GATE_SECONDS, GATE_MIB, GATE_TIMEOUT = 3, 300, 2048, 36 * 3600
CODE = ("scripts/agent_ds_client.py", "scripts/agent_ds_run.py", "scripts/agent_ds_report.py", "tests/test_agent_ds.py")


def utc():
    return datetime.now(timezone.utc).isoformat()


# ---------- prompt text: the same characters Task 20 tokenised for Qwen3-8B ----------

def transcript(ep, history_size=None):
    parts = []
    for i, (raw, special) in enumerate(ep.blocks[:history_size]):
        observation = ep.observation(i, ep.kept, ep.origin, ep.views, special)
        completion = raw[len(agent.THOUGHT_PREFILL):].rstrip()
        parts.append(f"{agent.THOUGHT_PREFILL}{completion}\nObservation:\n{observation}\n\n")
    return "".join(parts)


def messages(ep, history_size=None):
    head = [{"role": "system", "content": agent.SYSTEM},
            {"role": "user", "content": agent.user_text(ep.question_text)}]
    text = transcript(ep, history_size)
    if ep.force_reason is not None:
        need(text, "A forced finish needs an earlier transcript")
        return head + [{"role": "assistant", "content": text},
                       {"role": "user", "content": agent.FORCE_MESSAGE},
                       {"role": "assistant", "content": agent.THOUGHT_PREFILL, "prefix": True}]
    return head + [{"role": "assistant", "content": text + agent.THOUGHT_PREFILL, "prefix": True}]


def prepare(ep, tag):
    """Main thread only: Qwen tokens for limits and our cost model, plus the API messages."""
    tokens = ep.next_prompt()
    omitted = ep.force_reason is not None and bool(ep.turns) and ep.turns[-1].get("force_prompt_omits_triggering_block")
    msgs = messages(ep, len(ep.blocks) - 1 if omitted else None)
    cid = f"{tag}|{ep.task['dataset']}|t{ep.task['task']}|{SAFE[ep.row]}|r{len(ep.turns)}"
    return tokens, msgs, cid


def refuse(ep, tokens, cid):
    """Provider moderation refusal: end the episode with empty answers; analysis excludes the task (TASK_21 §1.5)."""
    ep.turns.append({"round": len(ep.turns), "raw_output": "", "action": "content_refused", "query": None,
                     "prompt_tokens": len(tokens), "prefix_hit_tokens": 0, "cached_input_token_cost": 0.0,
                     "output_tokens": 0, "forced": ep.force_reason, "keep_before": ep.kept.copy(),
                     "keep_after": ep.kept.copy(), "malformed_finish_json": False, "format_invalid": False,
                     "vllm_cached_tokens": None,
                     "deepseek": {"call_id": cid, "refused": True, "finish_reason": "content_refused", "hit": 0,
                                  "miss": 0, "output": 0, "cost_yuan": "0"}})
    ep.force_reason = "content_refused"
    ep.answers = ("",) * agent.K


def accept(ep, tokens, call, search):
    result = ep.accept_reply(tokens, call["content"], call["output"], search)
    ep.turns[-1]["vllm_cached_tokens"] = None
    ep.turns[-1]["deepseek"] = {k: call[k] for k in (
        "call_id", "id", "model", "system_fingerprint", "finish_reason", "hit", "miss", "output", "cost_yuan",
        "latency_seconds", "attempts", "messages_sha256")}
    return result


# ---------- one stage ----------

def selection(stage):
    receipt = read(data.RECEIPT)["datasets"]
    tasks, corpora, index_dirs = {}, {}, {}
    for label in DATASETS:
        item = receipt[label]
        part = item["main" if stage == "main" else "smoke"]
        engines.checked_artifact(part["file"])
        engines.checked_artifact(item["corpus"])
        chosen = list(rows(Path(part["file"]["path"])))[:PER_DATASET[stage]]
        need(len(chosen) == PER_DATASET[stage] and {t["dataset"] for t in chosen} == {label}, f"Selection failed: {label}")
        tasks[label] = chosen
        corpora[label] = list(rows(Path(item["corpus"]["path"])))
        need(len(corpora[label]) == item["n_paragraphs"], f"Corpus size changed: {label}")
        index_dirs[label] = Path(item["bm25_index"][0]["path"]).parent
    return tasks, corpora, index_dirs


def chunks(tasks, stage):
    width = CHUNK_PER_DATASET if stage == "main" else PER_DATASET[stage]
    n = PER_DATASET[stage]
    return [[t for label in DATASETS for t in tasks[label][i:i + width]] for i in range(0, n, width)]


def compress(pending, codecs, r4):
    later = []
    for item in pending:
        if item.episode.row.startswith("r4"):
            later.append(item)
        else:
            agent.execute_compression(item, codecs[item.episode.row])
    if later:
        r4(later)


def run_episodes(episodes, client, codecs, r4, searchers, archive, tag, progress=lambda **k: None, workers=api.SLOTS):
    """Advance every episode in lock-step waves: API turn for all, then batched compression."""
    active = [ep for ep in episodes if not ep.completed()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while active:
            prepared = [(ep, *prepare(ep, tag)) for ep in active]
            futures = [(ep, tokens, cid, pool.submit(client.complete, cid, msgs, input_upper=int(1.5 * len(tokens)) + 512))
                       for ep, tokens, msgs, cid in prepared]
            calls, error = [], None
            for ep, tokens, cid, future in futures:
                try:
                    calls.append((ep, tokens, future.result()))
                except api.ContentRefused:
                    refuse(ep, tokens, cid)
                    archive.finish(ep)
                except BaseException as exc:  # let in-flight calls settle in the ledger, then stop
                    error = error or exc
            if error is not None:
                raise error
            pending = []
            for ep, tokens, call in calls:
                request = accept(ep, tokens, call, searchers[ep.task["dataset"]])
                if ep.completed():
                    archive.finish(ep)
                if request:
                    pending.append(request)
            compress(pending, codecs, r4)
            active = [ep for ep in active if not ep.completed()]
            progress(active=len(active), done=len(archive.seen))


class Status:
    def __init__(self, path):
        self.path, self.data = path, {"pid": os.getpid(), "started_utc": utc(), "events": []}

    def update(self, **facts):
        self.data.update(facts, updated_utc=utc())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1))
        os.replace(tmp, self.path)

    def event(self, item):
        self.data["events"] = (self.data["events"] + [{"utc": utc(), **item}])[-50:]
        self.update(**({"api_status": item["status"]} if "status" in item else {}))


def gpu_gate(status):
    readings, began = [], time.monotonic()
    while True:
        readings.append({"utc": utc(), **engines.pipeline.gpu()})
        last = readings[-GATE_READINGS:]
        status.update(status="waiting_gpu", gate_readings=last)
        if len(last) == GATE_READINGS and all(r["memory_used_mib"] < GATE_MIB for r in last):
            return last
        need(time.monotonic() - began < GATE_TIMEOUT, "GPU 1 gate timed out")
        time.sleep(GATE_SECONDS)


def deployment():
    """Task 20's registered compressor deployment, re-verified, and proven identical to the frozen try3 config."""
    assets = engines.registered_assets(verify_models=False)
    frozen = read(TASK20_FROZEN)["assets"]
    for key in ("r4", "reranker", "llmlingua2"):
        need(assets[key] == frozen[key], f"Task 20 compressor deployment differs: {key}")
    need(read(TASK20_PROMPTS) == smoke.prompt_spec(), "Task 20 prompt spec differs")
    engines.grpo.verify_model()
    engines.checked_artifact(assets["reranker"]["weights"])
    cards = read(engines.MODEL_CARDS)
    for item in [cards["llmlingua_wheel"]["wheel"], *cards["llmlingua_wheel"]["files"], *cards["llmlingua2"]["files"]]:
        engines.checked_artifact(item)
    return assets


def agent_settings():
    return {"model": api.MODEL, "endpoint": f"https://{api.HOST}{api.CHAT_PATH}", "thinking": "disabled",
            "temperature": 0, "max_tokens": api.MAX_TOKENS, "stop": api.STOP, "prefix_completion": True,
            "concurrency": api.SLOTS, "total_cap_yuan": api.yuan(api.TOTAL_CAP),
            "stage_caps_yuan": {k: api.yuan(v) for k, v in api.STAGE_CAP.items()},
            "price_units_1e-8_cny": {"hit": api.HIT_UNITS, "miss": api.MISS_UNITS, "output": api.OUT_UNITS},
            "peak_allowed": True, "peak_multiplier": api.PEAK_MULTIPLIER,
            "peak_approval": "user 2026-11-26: 直接开始，不用错峰",
            "limits_in_qwen3_tokens": {"prompt_limit": agent.PROMPT_LIMIT, "max_model_len": agent.MAX_MODEL_LEN}}


def dry_transport():
    """GPU dry run only: scripted replies (one search per question, then finish), no network, no spending."""
    def send(path, body, key):
        if path == api.BALANCE_PATH:
            return 200, {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "99"}]}
        history = body["messages"][2]["content"]
        n = history.count("\nObservation:\n")
        qs = [line.split(". ", 1)[1] for line in body["messages"][1]["content"].splitlines()]
        text = (f" dry run.\nAction: search[{qs[n]}]" if n < len(qs) and len(body["messages"]) == 3
                else ' dry run.\nAction: finish[{"1":"a","2":"b","3":"c"}]')
        return 200, {"id": "dry", "model": api.MODEL, "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 100,
                               "completion_tokens": 10, "total_tokens": 110}}
    return send


def new_client(stage, status):
    if stage != "dryrun":
        return api.AgentClient(WORK, stage, publish=status.event, allow_peak=True)
    work = WORK / "dry"  # separate ledger: dry-run fake costs never touch the real one
    (work / "smoke").mkdir(parents=True)
    saturday = datetime(2026, 11, 28, 10, 0, tzinfo=timezone.utc)
    return api.AgentClient(work, "smoke", publish=status.event, key="dryrun" + "0" * 12, transport=dry_transport(),
                           clock=lambda: saturday)


def worker(stage):
    engines.require_cuda()
    need(Path(sys.executable).resolve() == GRPO.resolve(), "Worker must use the GRPO environment")
    need(not any(k.lower() in ("http_proxy", "https_proxy", "all_proxy") for k in os.environ), "Proxy set in worker")
    run, small = WORK / stage, SMALL / stage
    status = Status(run / "status.json")
    def stop(signum, _frame):
        raise InterruptedError(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    began = time.monotonic()
    archive = telemetry = client = None
    try:
        status.update(status="waiting_gpu", stage=stage)
        gate = gpu_gate(status)
        with LOCK.open("rb") as lock:
            status.update(status="waiting_lock")
            fcntl.flock(lock, fcntl.LOCK_EX)
            status.update(status="loading", lock_held=True)
            telemetry = disk.RootTelemetry(run)
            runtime = engines.RuntimeMeasurements(run / "gpu_snapshots.jsonl", telemetry)
            telemetry.start()
            assets = deployment()
            versions = engines.check_versions(assets)
            from tokenizers import Tokenizer
            import bm25s
            count = Tokenizer.from_file(str(data.TOKENIZER))
            ntok = lambda text: len(count.encode(text, add_special_tokens=False).ids)
            tokenizer = engines.offline_tokenizer(assets["agent"]["params"]["tokenizer"])
            tasks, corpora, index_dirs = selection(stage)
            indexes = {label: bm25s.BM25.load(str(path)) for label, path in index_dirs.items()}
            searchers = {label: (lambda idx: (lambda q: agent.search_bm25(idx, q)))(indexes[label]) for label in DATASETS}
            client = new_client(stage, status)
            client.balance("start")
            archive = smoke.SmokeArchive(run / "episodes.jsonl")
            with engines.engine_session("r4", assets, runtime, run / "engine.log") as llm:
                codecs = {"no_compress": agent.baseline_compression, "oracle": agent.baseline_compression,
                          "RR-τ reselect": engines.new_reranker(assets["reranker"]),
                          "LLMLingua-2": engines.new_llmlingua(assets["llmlingua2"], ntok)}
                runtime.gpu("compressors_loaded")
                r4 = lambda pending: engines.execute_r4(llm, assets, pending, runtime)
                groups = chunks(tasks, stage)
                for i, group in enumerate(groups):
                    status.update(status="running", chunk=i, n_chunks=len(groups),
                                  spent_yuan=client.ledger.summary(stage)["spent_yuan"])
                    episodes = [agent.Episode(t, row, corpora[t["dataset"]], tokenizer) for t in group for row in ROWS]
                    run_episodes(episodes, client, codecs, r4, searchers, archive, stage,
                                 progress=lambda **k: status.update(**k))
                    runtime.check_storage()
                del codecs
                engines.release_torch()
        status.update(status="closing", lock_held=False)
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if archive is not None:
            archive.file.close()
        if client is not None:
            try:
                client.balance("end")
            except Exception as exc:  # recorded, not fatal: spending already stopped
                status.update(balance_end_error=str(exc))
        if telemetry is not None and telemetry.threads:
            telemetry.close()
            write_new(run / "root_usage_check.json", telemetry.policy.report())
    summary = client.summary()
    scan = api.base.scan_artifacts(sorted({run, client.directory}), key=client.key)
    write_new(small / "run_config.json", {
        "status": "complete", "task": 21, "stage": stage, "gpu_index": 1, "gate_readings": gate,
        "rows": list(ROWS), "datasets": {k: [t["task"] for t in v] for k, v in tasks.items()},
        "n_episodes": len(archive.seen), "wall_seconds": time.monotonic() - began,
        "agent": agent_settings(), "compressors": {k: assets[k] for k in ("r4", "reranker", "llmlingua2")},
        "task20_frozen_config": record(TASK20_FROZEN), "task20_prompts": record(TASK20_PROMPTS),
        "data_receipt": record(data.RECEIPT), "versions": versions, "engine_loads": runtime.loads,
        "ledger": summary, "balances": {k: read(client.directory / f"balance_{k}.json")["total_balance"]
                                        for k in ("start", "end") if (client.directory / f"balance_{k}.json").exists()},
        "root_usage_check": record(run / "root_usage_check.json"), "episodes": record(run / "episodes.jsonl"),
        "attempts": record(client.directory / "attempts.jsonl"), "ledger_file": record(client.work / "ledger.jsonl"),
        "secret_scan": scan, "code": {p: record(PROJECT / p) for p in CODE},
    })
    status.update(status="complete", spent_yuan=summary["stage_totals"]["spent_yuan"])


# ---------- foreground commands ----------

def environment(stage):
    with patch.object(smoke, "RUNTIME", RT / stage):
        env = smoke.runtime_environment(stage)
    cards = read(engines.MODEL_CARDS)
    env.update(PYTHONPATH=os.pathsep.join((cards["llmlingua_wheel"]["vendor"], str(PROJECT))),
               TIKTOKEN_CACHE_DIR=str(engines.TIKTOKEN_CACHE), CUDA_VISIBLE_DEVICES="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    merged = {**os.environ, **env}
    return {k: v for k, v in merged.items() if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}


def launch(stage, approved):
    need(Path.cwd() == PROJECT, "Run from the project root")
    need(stage in ("dryrun", "smoke", "smoke2") or approved, "Main needs the user's approval after the smoke (--approved)")
    if stage == "main":
        need(read(SMALL / "smoke2/closeout.json")["status"] == "stopped_content_risk_main_approved", "Smoke2 closeout missing")
    need(disk.root_usage().free >= disk.MIN_ROOT_FREE_BYTES, "Root free space below 5 GB")
    run, small = WORK / stage, SMALL / stage
    require_new(run, small, RT / stage)
    run.mkdir(parents=True)
    small.mkdir(parents=True)
    env = environment(stage)
    with (run / "worker.log").open("xb") as log:
        process = subprocess.Popen([str(GRPO), "-B", "-u", "-m", "scripts.agent_ds_run", "worker", stage],
                                   cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
    (run / "pipeline.pid").write_text(f"{process.pid}\n")
    print(f"Task 21 {stage} worker PID {process.pid}; status {run / 'status.json'}", flush=True)


def probe():
    """Three cheap calls: first turn, continuation (cache hit), forced finish. Checks the response shape."""
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Probe is CPU-only")
    import bm25s
    run = WORK / "probe"
    require_new(run, SMALL / "probe")
    run.mkdir(parents=True)
    (SMALL / "probe").mkdir(parents=True)
    tasks, corpora, index_dirs = selection("probe")
    assets = engines.registered_assets(verify_models=False)
    tokenizer = engines.offline_tokenizer(assets["agent"]["params"]["tokenizer"])
    index = bm25s.BM25.load(str(index_dirs["musique"]))
    client = api.AgentClient(WORK, "probe", allow_peak=True)
    client.balance("start")
    ep = agent.Episode(tasks["musique"][0], "no_compress", corpora["musique"], tokenizer)
    report = []
    for step in range(3):
        if ep.completed():
            break
        if step == 2:
            ep.force_reason = "probe_forced_shape"
        tokens, msgs, cid = prepare(ep, "probe")
        call = client.complete(cid, msgs)
        request = accept(ep, tokens, call, lambda q: agent.search_bm25(index, q))
        if request:
            agent.execute_compression(request, agent.baseline_compression)
        turn = ep.turns[-1]
        report.append({"roles": [m["role"] for m in msgs], "qwen_prompt_tokens": len(tokens),
                       "content_head": call["content"][:400], "content_starts_with_thought": call["content"].lstrip().startswith("Thought"),
                       "parsed_action": turn["action"], "format_invalid": turn["format_invalid"],
                       "finish_reason": call["finish_reason"], "hit": call["hit"], "miss": call["miss"],
                       "output": call["output"], "cost_yuan": call["cost_yuan"]})
    client.balance("end")
    write_new(SMALL / "probe/probe.json", {"calls": report, "ledger": client.summary(),
                                          "secret_scan": api.base.scan_artifacts([run], key=client.key)})
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("action", choices=("probe", "launch", "worker"))
    cli.add_argument("stage", nargs="?", choices=("dryrun", "smoke", "smoke2", "main"))
    cli.add_argument("--approved", action="store_true")
    args = cli.parse_args()
    if args.action == "probe":
        probe()
    elif args.action == "launch":
        launch(args.stage, args.approved)
    else:
        worker(args.stage)
