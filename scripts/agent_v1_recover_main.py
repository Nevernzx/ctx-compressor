"""Recover Task 20 from its terminal main_try2 archive using CPU only."""
from ast import literal_eval
from collections import Counter
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import sys

from scripts import agent_v1_main as main, agent_v1_report as report, agent_v1_smoke as smoke
from src import agent_v1 as agent
from src.reader_heldout_inputs import need, read, rows, write_new
from src.reranker_v1 import record, require_new, verify

ROOT = main.ROOT
CLOSE = ROOT / "closeout"
RECOVERY = main.SMALL / "cpu_archive_recovery.json"
TESTS = main.SMALL / "cpu_recovery_tests.json"
STORAGE = ROOT / "root_usage_check.json"
ATTRIBUTION = {
    "source": "Claude's read-only inspection, relayed with authorization for CPU closeout",
    "new_files_since": "2026-11-25T22:00:00+08:00",
    "new_root_files": 7,
    "new_root_files_approx_kb": 58,
    "processes_holding_deleted_files": 0,
    "attributed_growth_kib": 4,
    "root_available_gb": 13.31,
    "hypothesis": "Root df counts the host backing filesystem shared with other workloads. "
                  "A global before/after measurement in rule (a) can include external writes.",
    "specific_writer_verified": False,
}


def terminal(state, storage):
    stage, = state["stages"]
    limit = main.disk.ROOT_LIMIT_BYTES
    violation = storage["violation"]
    need(state["status"] == stage["status"] == "failed" and state["stage"] == stage["stage"] == "main"
         and state["gpu_index"] == 1 and not state["automatic_retry"] and state["stage_pid"] is None
         and not state["lock"]["held"] and stage["exit_code"] == 1 and stage["uses_gpu"]
         and state["last_monitor"]["pid"] == stage["pid"] and not state["last_monitor"]["alive"]
         and not (state["last_monitor"]["oom"] or state["last_monitor"]["nonfinite"]),
         "Expected a terminal, released GPU 1 run with a failed main stage")
    need(storage["status"] == "failed" and violation["rule"] == "a"
         and "Task 20 storage rule a triggered" in state["first_error"]
         and str(violation["bytes"]) in state["first_error"]
         and storage["limit_bytes"] == limit and storage["min_free_limit_bytes"] == main.disk.MIN_ROOT_FREE_BYTES
         and violation["bytes"] == storage["sustained_growth_bytes"]
         == storage["root_after_bytes"] - storage["root_before_bytes"] > limit,
         "Recovery is restricted to the original rule (a) failure")


def audit_trace(storage):
    root = report.checked(storage["root_telemetry"])
    report.checked(storage["workplace_telemetry"])
    trace = list(rows(root))
    opening, closing = trace[0], trace[-1]
    fd_measurements = [sample["root_writable_fds"] for sample in trace if "root_writable_fds" in sample]
    errors = [sample["error"] for sample in trace if "error" in sample]
    need(len(trace) == storage["samples"] and errors == storage["fast_sample_errors"]
         and opening["boundary"] == "opening" and closing["boundary"] == "closing"
         and closing["utc"] == storage["violation"]["utc"]
         and opening["root_used_bytes"] == storage["root_before_bytes"]
         and closing["root_used_bytes"] == storage["root_after_bytes"]
         and max(s["global_growth_bytes"] for s in trace) == storage["peak_growth_bytes"]
         and all(s["global_growth_bytes"] == s["root_used_bytes"] - opening["root_used_bytes"] for s in trace),
         "Storage receipt and immutable root samples disagree")
    peak = max(s["attributed_growth_bytes"] for s in trace)
    free = min(s["root_free_bytes"] for s in trace)
    need(peak == storage["peak_attributed_growth_bytes"] <= storage["limit_bytes"]
         and free == storage["min_root_free_bytes"] >= storage["min_free_limit_bytes"],
         "Archive has a rule (b) or (c) violation")
    return {
        "opening_utc": opening["utc"], "closing_utc": closing["utc"], "samples": len(trace),
        "fd_samples": len(fd_measurements), "samples_with_root_writable_fds": sum(map(bool, fd_measurements)),
        "samples_with_global_growth_over_limit": sum(s["global_growth_bytes"] > storage["limit_bytes"] for s in trace),
        "closing_attributed_growth_bytes": closing["attributed_growth_bytes"],
    }


def engine_launches(logfile, proof):
    configs = main.agent_runtime_assets(proof["assets"])
    models = {configs[name]["params"]["model"]: name for name in ("agent", "r4")}
    engine_rows = []
    flags = set(("model", "tokenizer", "dtype", "seed", "max_model_len", "enable_prefix_caching",
                 "enable_chunked_prefill", "gpu_memory_utilization", "max_num_batched_tokens",
                 "max_num_seqs", "enforce_eager"))
    prefix = "non-default args: "
    for number, line in enumerate(logfile.read_text(errors="replace").splitlines(), start=1):
        if prefix not in line:
            continue
        args = literal_eval(line.split(prefix, 1)[1])
        kind = models[args["model"]]
        params = configs[kind]["params"]
        extras = {"disable_cascade_attn", "worker_extension_cls"} if kind == "agent" else {"enable_lora", "max_cpu_loras"}
        need(flags | extras <= args.keys() and all(args[key] == params[key] for key in args.keys() & params.keys()),
             f"Saved vLLM startup settings differ from preflight at line {number}")
        details = {"engine": kind, "line": number, "logged_non_default_arguments": args}
        if kind == "agent":
            need(args["disable_cascade_attn"] is True, "Archive lacks disable_cascade_attn=True")
            details.update(effective_disable_cascade_attn=True,
                           source="vLLM non-default startup args and archived post-enforcement agent_loaded sample")
        engine_rows.append(details)
    return engine_rows


def collect_runtime(state, proof, storage, trace):
    events_path = ROOT / "events.jsonl"
    events = list(rows(events_path))
    check_path = main.SMALL / "residency_check.json"
    check = read(check_path)
    first = events[0]
    need(check["status"] == "passed" and check["transfer"]["fingerprint_checked"]
         and first["kind"] == "cache_proven" and first["next_cached_tokens"] == check["next_cached_tokens"] > 0,
         "Main's archived physical prefix cache proof differs")
    row_events = [item for item in events if item["kind"] == "row_complete"]
    expected_events = [(name, (i + 1) * proof["n_tasks"]) for i, name in enumerate(agent.ROWS)]
    need([(x["row"], x["episodes"]) for x in row_events] == expected_events
         and events[-1] == row_events[-1] and row_events[-1]["utc"] < storage["violation"]["utc"],
         "Row completion receipts do not cover the whole formal run before rule (a)")

    transfers_path = ROOT / "weight_transfers.jsonl"
    transfers = list(rows(transfers_path))
    expected_transfers = 1 + sum(x["waves"] for x in events if x["kind"] == "cohort_complete")
    need(len(transfers) == expected_transfers and transfers[0] == check["transfer"]
         and {x["agent_pid"] for x in transfers} == {check["transfer"]["agent_pid"]}
         and {x["resident_kv_bytes"] for x in transfers} == {check["transfer"]["resident_kv_bytes"]},
         "Archived engine handoffs differ from the main schedule")

    samples_path = ROOT / "gpu_snapshots.jsonl"
    monitor_path = ROOT / "resource_monitor.jsonl"
    ownership_path = ROOT / "gpu_ownership.jsonl"
    snapshots = list(rows(samples_path))
    monitor = list(rows(monitor_path))
    ownership = list(rows(ownership_path))
    launches = engine_launches(main.GPU_LOG, proof)
    need(Counter(x["engine"] for x in launches) == {"agent": 1, "r4": expected_transfers}
         and Counter(x["stage"] for x in snapshots if x["stage"].endswith("_loaded"))
         == {"agent_loaded": 1, "r4_loaded": expected_transfers}
         and all(x["index"] == 1 for x in snapshots + monitor)
         and ownership[-1]["compute_apps_mib"] == {} and ownership[-1]["utc"] == state["last_monitor"]["utc"],
         "Archived engine start or GPU ownership evidence differs")
    versions = main.engines.check_versions(proof["assets"])
    pilot_cfg = read(report.checked(proof["pilot"]["config"]))
    need(versions == pilot_cfg["runtime"]["versions"], "Package versions differ from the signed pilot")
    elapsed = (datetime.fromisoformat(trace["closing_utc"]) - datetime.fromisoformat(trace["opening_utc"])).total_seconds()
    need(elapsed > 0, "Invalid root monitoring interval")
    return check, {
        "status": "archive_verified_cpu", "episodes": record(main.EPISODES), "n_episodes": row_events[-1]["episodes"],
        "row_events": record(events_path), "row_wall_seconds": {x["row"]: x["wall_seconds"] for x in row_events},
        "wall_seconds": elapsed, "wall_seconds_source": "Original root monitor opening to closing",
        "engine_loads": launches, "engine_log": record(main.GPU_LOG), "versions": versions,
        "versions_source": "Same local GRPO environment package metadata, checked against the pilot receipt",
        "peak_observed_gpu1_mib": max(x["memory_used_mib"] for x in snapshots + monitor),
        "gpu_snapshots": record(samples_path), "gpu_resource_monitor": record(monitor_path),
        "gpu_ownership": record(ownership_path), "root_usage_check": record(STORAGE),
        "weight_transfers": record(transfers_path), "residency_check": record(check_path),
    }


def reference_check(scored):
    pooled = scored["pooled"]
    reference = {
        "r4 reselect F1": (34.52, pooled["rows"]["r4 reselect"]["f1"]),
        "RR-τ reselect F1": (33.52, pooled["rows"]["RR-τ reselect"]["f1"]),
    }
    for name, expected in (
        ("r4 reselect − RR-τ reselect", 0.99), ("r4 append − RR-τ append", 1.66),
        ("r4 reselect − no_compress", -1.03), ("r4 reselect − LLMLingua-2", 10.07),
        ("oracle − r4 reselect", 6.87),
    ):
        reference[name] = (expected, pooled["paired"][name]["delta_f1"])
    verified = {}
    for name, (expected, actual) in reference.items():
        need(round(100 * actual, 2) == expected, f"Independent reference differs for {name}")
        verified[name] = {"claude_rounded": expected, "cpu_recount": 100 * actual, "n": 900}
    return verified


def verify_repair(proof):
    previous = main.SMALL / "previous_launch.json"
    prior = read(previous)
    original = read(report.checked(prior["old_preflight"]))
    for key in ("old_status", "old_log"):
        report.checked(prior[key])
    backups = smoke.data.WORK / "monitor_path_recovery/before_main_rpc_fix"
    for name, old_file in original["main_code"].items():
        verify(backups / Path(name).name, old_file["sha256"])
    repair = {
        "method": "Named methods on the installed vLLM worker extension",
        "previous_launch": record(previous), "original_preflight": prior["old_preflight"],
        "source_sha256": {
            name: {"old": original["main_code"][name]["sha256"], "new": file["sha256"]}
            for name, file in proof["main_code"].items()
        },
    }
    pilot_cfg = read(report.checked(proof["pilot"]["config"]))
    need(repair == pilot_cfg["rpc_repair"], "Formal repair sources differ from the successful pilot")
    return repair


def recover(verify_only):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and Path.cwd() == smoke.PROJECT
         and Path(sys.executable).resolve() == smoke.GRPO.resolve(),
         "Use the 5880 GRPO Python with CUDA hidden for CPU reconstruction")
    proof = read(main.PRE)
    sys.path.insert(0, proof["assets"]["llmlingua2"]["wheel"])
    need(proof == main.preflight(verify_models=False), "Formal preflight, data, or frozen sources changed")
    need(not main.GPU_RECEIPT.exists(), "The original attempt unexpectedly has a success receipt")
    failed = {"status": record(main.STATUS), "root_usage_check": record(STORAGE),
              "main_log": record(main.GPU_LOG), "launcher_log": record(ROOT / "launcher.log"),
              "initial_closeout_status": record(CLOSE / "cpu_status.json")}
    state, storage = read(main.STATUS), read(STORAGE)
    terminal(state, storage)
    need(all(not Path(f"/proc/{pid}").exists() for pid in (state["pipeline_pid"], state["stages"][0]["pid"])),
         "Recorded original controller or worker PID still exists")
    need(state["lock"]["path"] == proof["lock"] and state["task19"] == proof["task19"]
         and read(report.checked(failed["initial_closeout_status"]))["status"] == "failed",
         "Original task ordering or automated closeout evidence differs")
    trace = audit_trace(storage)
    check, runtime = collect_runtime(state, proof, storage, trace)
    repair = verify_repair(proof)
    scored = report.rescore({"runtime": runtime})
    evidence = {
        "status": "archive_verified_cpu", "source": "Authorized CPU-only reconstruction of main_try2",
        "original_gpu_status": "failed", "original_failure": failed, "storage_rule": "a",
        "trace_audit": trace, "attribution_evidence": ATTRIBUTION, "archive": runtime["episodes"],
        "row_events": runtime["row_events"], "episodes": runtime["n_episodes"], "rounds": scored[2],
        "rounded_reference_check": reference_check(scored[0]),
    }
    cfg = {
        "status": "archive_verified_cpu", "original_gpu_status": "failed", "original_failure": failed,
        "scope": proof["scope"], "task": 20, "n_tasks": proof["n_tasks"], "gpu_index": 1,
        "pilot": proof["pilot"], "rpc_repair": repair, "runtime": runtime,
        "offload": "Agent weights temporarily in host RAM. Agent engine and KV blocks stay resident on GPU 1.",
        "agent_worker_extension": main.residency.EXTENSION, "r4_cohort": proof["r4_cohort"],
        "schedule": "R4 reverses the episode order between compressor handoffs within a cohort.",
        "shared_lock": proof["lock"], "task19": proof["task19"], "ipc_check": record(ROOT / "ipc_check.json"),
        "gpu_ownership": runtime["gpu_ownership"], "preflight": record(main.PRE), "data_preflight": proof["data"],
        "assets": proof["assets"], "prompt": proof["prompt"], "try3": proof["try3"], "frozen_code": proof["frozen_code"],
        "main_code": proof["main_code"], "peak_observed_gpu1_mib": runtime["peak_observed_gpu1_mib"],
        "rows": list(agent.ROWS), "storage_policy": read(main.FROZEN)["root_storage_policy"],
        "open_provence": read(main.engines.MODEL_CARDS)["open_provence"],
    }
    bundle = report.assemble(state, cfg, check, storage, scored)
    bundle["cpu_archive_recovery"] = evidence
    for reference in (*failed.values(), runtime["episodes"], cfg["ipc_check"]):
        report.checked(reference)
    if verify_only:
        report.render(bundle)
        print(json.dumps({"episode_count": evidence["episodes"], "rounds": evidence["rounds"],
                          "reference": evidence["rounded_reference_check"], "peak_gpu1_mib": runtime["peak_observed_gpu1_mib"],
                          "hotpotqa_gold": scored[3]["hotpotqa"]}, ensure_ascii=False))
        return
    tests = read(TESTS)
    need(tests["status"] == "passed" and tests["tests"] >= read(main.CPU_TEST)["tests"]
         and all(tests[k] == 0 for k in ("failures", "errors", "skipped")), "CPU recovery tests did not pass")
    for reference in (tests["xml"], tests["log"], *tests["failed_artifacts"].values()):
        report.checked(reference)
    need(all(item == record(Path(item["path"])) for item in tests["sources"].values()), "Tested delivery sources changed")
    require_new(main.CONFIG, RECOVERY, report.RESULT, *report.SVG.values())
    evidence["cpu_tests"] = record(TESTS)
    write_new(RECOVERY, evidence)
    cfg["cpu_archive_recovery"] = record(RECOVERY)
    write_new(main.CONFIG, cfg)
    report.deliver(bundle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    recover(parser.parse_args().verify_only)

