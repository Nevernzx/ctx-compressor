"""Complete Task 19 G3 on CPU from intact, ordered reader outputs."""
import ast
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import reader_heldout_report as qa
from scripts import reader_heldout_v2 as v16
from scripts import test_v1_assets as assets, test_v1_data as data, test_v1_pipeline as pipeline
from scripts import test_v1_reader as reader
from scripts.run_reader_heldout import publish
from src import reader_heldout_inputs as reuse
from src.reranker_v1 import need, read, record, require_new, verify

W, OUT = data.WORK, data.SMALL
PREDICT = reader.READER / "predictions.jsonl"
REQUESTS = reader.READER / "requests.jsonl"
ARCHIVE = W / "g3_df_opening_recovery"
AUDIT = OUT / "G3_recovery.json"
TESTS = OUT / "g3_recovery_cpu_tests.json"
CONFIG = OUT / "run_config.json"
RECEIPT = W / "stages/G3/receipt_complete.json"
INFERENCE = reader.READER / "receipt_inference.json"
STARTUP_ARGS = "[utils.py:233] non-default args: "
G3 = (
    ("G0", "G0", OUT / "G0_smoke.json", True),
    ("G1", "G1", OUT / "G1_scores.json", True),
    ("RR", "RR", OUT / "rr_rollouts.json", False),
    ("G2-v3", "G2-v3", OUT / "G2_v3.json", True),
    ("G2-r2", "G2-r2", OUT / "G2_r2.json", True),
    ("G2-r4", "G2-r4", OUT / "G2_r4.json", True),
    ("G3_prepare", "prepare-reader", reader.PREFLIGHT, False),
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def checked(item, expected):
    need(item["path"] == str(expected) and expected.resolve() == expected
         and expected.stat().st_size == item.get("size", item.get("bytes")),
         f"archived file identity differs: {expected}")
    verify(expected, item["sha256"])


def args_evidence(lines, expected):
    rows = [(n, line) for n, line in enumerate(lines, 1) if STARTUP_ARGS in line]
    need(len(rows) == 1, "expected one vLLM startup configuration in launcher.log")
    number, line = rows[0]
    logged = ast.literal_eval(line.split(STARTUP_ARGS, 1)[1])
    need(isinstance(logged, dict) and logged.get("disable_cascade_attn") is True,
         "vLLM startup must show disable_cascade_attn=True")
    fields = ("model", "tokenizer", "dtype", "seed", "max_model_len", "enable_prefix_caching",
              "gpu_memory_utilization", "max_num_batched_tokens", "max_num_seqs",
              "enforce_eager", "enable_chunked_prefill", "disable_cascade_attn")
    need(all(k in logged and k in expected and logged[k] == expected[k] for k in fields)
         and all(v == {**expected, "disable_log_stats":True}.get(k) for k,v in logged.items()),
         "vLLM startup arguments differ from the frozen reader")
    return {"line_number":number, "excerpt":line, "logged_configuration":logged,
            "source":"vLLM startup non-default arguments in launcher.log"}


def log_evidence(frozen):
    path = reader.READER / "launcher.log"
    lines = path.read_bytes().decode().split("\n")
    evidence = args_evidence(lines, frozen["engine"])
    init = [(n, re.search(r"\(EngineCore_DP0 pid=(\d+)\).*Initializing a V1 LLM engine \(v([^)]+)\)", line))
            for n, line in enumerate(lines, 1)]
    init = [(n, m[1], m[2]) for n, m in init if m]
    need(len(init) == 1 and init[0][2] == frozen["packages"]["vllm"], "vLLM startup version differs")
    trace = [line for line in lines if 'scripts/reader_heldout_v2.py", line 230, in run_engine' in line]
    key_lines = [n for n, line in enumerate(lines, 1) if line == "KeyError: 'df_opening'"]
    need(len(trace) == len(key_lines) == 1 and 'cfg["df_opening"] = reader.read(PRE)["df_opening"]' in "\n".join(lines),
         "expected post-inference df_opening traceback")
    need(lines[-2] == "KeyError: 'df_opening'" or lines[-1] == "KeyError: 'df_opening'",
         "launcher log contains later output")
    signed_source = Path(v16.__file__).resolve()
    verify(signed_source, reuse.sums(PROJECT/"SHA256SUMS")[str(signed_source.relative_to(PROJECT))])
    return {**evidence, "log":record(path), "vllm_initialization_line":init[0][0],
            "engine_pid":int(init[0][1]), "key_error_line":key_lines[0], "source_at_failure":record(signed_source)}


def cpu_tests():
    test = read(TESTS)
    initial = read(PROJECT/"eval/agent_v1/smoke_queue_cpu_tests.json")
    need(test["status"]=="passed" and test["matched_queue_tests"]==initial["tests"]
         and test["tests"]>initial["tests"] and all(test[k]==0 for k in ("failures","errors","skipped"))
         and not test["cpu_guard_events"],"G3 CPU regression receipt is incomplete")
    verify(Path(test["xml"]), test["xml_sha256"])
    return test


def previous_receipts(events, child):
    observed = [e for e in events if e["status"]=="stage_complete"]
    need([e["stage"] for e in observed] == [row[1] for row in G3],"completed pre-G3 stages differ")
    result = {}
    for (key, stage, out, use_gpu), event in zip(G3, observed, strict=True):
        path = W/"stages"/stage/"receipt_complete.json"
        stamp = read(path)
        need(event["receipt"]==str(path) and event["child_pid"]==stamp["child_pid"]
             and stamp["exit_code"]==0 and stamp["stage"]=="complete" and stamp["gpu"] is use_gpu
             and set(stamp["outputs"])=={out.name},"earlier completed stage receipt differs")
        checked(stamp["outputs"][out.name], out)
        result[key]=stamp
    launch = [e for e in events if e["stage"]=="G3" and e["status"]=="running" and e.get("log")]
    need(len(launch)==1 and launch[0]["child_pid"]==child and launch[0]["log"]==str(reader.READER/"launcher.log"),
         "G3 launcher receipt does not identify the reader")
    gate = read(W/"stages/G3/receipt_gate.json")
    reader_gate = read(reader.READER/"receipt_gate.json")
    polls = [e for e in events if e["stage"]=="G3" and e["status"]=="waiting_gpu"][-3:]
    need(gate["stage"]==reader_gate["stage"]=="gate" and [e["consecutive_free"] for e in polls]==[1,2,3]
         and all(e["gpu"]["index"]==1 and e["gpu"]["memory_used_mib"]<2048 and e["poll_seconds"]==300 for e in polls)
         and gate["gpu"]==reader_gate["gpu"]==launch[0]["gpu"]==polls[-1]["gpu"],
         "G3 GPU admission differs")
    return result, launch[0], gate


def match_archives(pre):
    _, tasks = data.load_test()
    need(pre["status"]=="passed" and pre["n_tasks"]==len(tasks)==reader.N_TASKS and pre["tasks"]==list(tasks)
         and pre["batch_size"]==reader.BATCH and pre["strategies"]==list(data.NAMES)
         and pre["requests"]["n"]==reader.N_PREDICTIONS and pre["requests"]["path"]==str(REQUESTS)
         and pre["requests"]["bytes"]==REQUESTS.stat().st_size and "df_opening" not in pre,"G3 preflight differs")
    signatures={}
    predicted=iter(qa.input_rows(PREDICT,signatures))
    frozen=iter(qa.input_rows(REQUESTS,signatures))
    counts=Counter()
    batches=n=extra_ids=0
    for task, batch in data.batches(frozen, pre["tasks"]):
        observed=[]
        for req in batch:
            out=next(predicted, None)
            key=tuple(req[i] for i in qa.IDENTITY)
            need(out is not None and key==tuple(out[i] for i in qa.IDENTITY),
                 f"prediction missing or out of order after {n} rows: {key}")
            item=tasks[task]["metadata"][req["question_index"]]
            need((req["example_id"],req["question"])==(item["example_id"],item["question"])
                 and all(req[k]==item[k] for k in ("answer","answer_aliases","question_hops"))
                 and req["example_id"]==out["example_id"],f"request identity differs: {key}")
            if "request_id" in req or "request_id" in out:
                need("request_id" in req and "request_id" in out and out["request_id"]==req["request_id"],
                     f"serialized request_id differs: {key}")
                extra_ids+=1
            need(req["prompt_tokens"]==len(req["token_ids"])>0,"request token metadata differs")
            reuse.check_length(req["prompt_tokens"],key)
            counts[req["config"]]+=1
            observed.append(out)
            tail={"strategy":out["strategy"],"task":task,"question_index":out["question_index"],
                  "example_id":out["example_id"],"configuration":out["config"]}
            n+=1
        qa.collect(observed, qa.index_requests(batch), data.NAMES)
        batches+=1
    need(next(predicted,None) is None,"surplus predictions after final expected task")
    need(batches==reader.N_TASKS and n==reader.N_PREDICTIONS
         and signatures[str(REQUESTS)]==pre["requests"]["sha256"] and dict(counts)==pre["requests"]["n_by_config"]
         and (tail["strategy"],tail["task"],tail["question_index"])==("r4",450,3),"read-out shape or request hash differs")
    need(extra_ids in (0,n), "inconsistent request_id serialization")
    return {"n":n,"n_tasks":batches,"batch_size":reader.BATCH,"last":tail,"strategy_order":list(data.NAMES),
            "request_id":{"serialized_request_id_n":extra_ids,"paired_fields":[*qa.IDENTITY,"example_id","prompt_sha1"]},
            "file_sha256":signatures,"prediction_rows_rescored":n}


def smoke_receipt():
    fixtures = v16.SMOKE
    verify(fixtures, reuse.sums(v16.M12)[str(fixtures)])
    path = reader.READER/"smoke_predictions.jsonl"
    proof = read(reader.READER/"receipt_smoke.json")
    verify(path,proof["predictions_sha256"])
    n=diff=0
    for orig, out in zip(reuse.rows(fixtures),reuse.rows(path),strict=True):
        need(all(orig[k]==out[k] for k in (*qa.IDENTITY,"example_id","prompt_sha1","expected_raw_output")),
             "prior smoke request identity differs")
        n+=1
        diff+=out["raw_output"]!=orig["expected_raw_output"]
    need(n==proof["n"]==40 and diff==proof["raw_output_mismatches"],"prior smoke output differs")
    return {"n":n,"raw_output_mismatches":diff}


def recover():
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and Path.cwd()==PROJECT,"recover G3 on 5880 CPU")
    require_new(ARCHIVE,AUDIT,CONFIG,RECEIPT,INFERENCE,reader.CONFIG,W/"status.next",reader.READER/"status.next")
    suite=cpu_tests()
    source = read(pipeline.STATUS)
    reader_status = read(reader.READER/"status.json")
    events = list(reuse.rows(W/"events.jsonl"))
    child = reader_status["pid"]
    need(source==events[-1] and source["pid"]==3481257==int(pipeline.PID.read_text())
         and source["status"]=="failed" and source["stage"]=="terminal" and "G3 exited 1" in source["error"]
         and all(row["pid"]==source["pid"] and not row["automatic_retry"] for row in events),
         "refusing to recover a different Task 19 pipeline")
    earlier, launched, gate = previous_receipts(events,child)
    phase=read(pipeline.PHASE_P)
    checked(phase["inputs"]["model_preflight.json"],assets.PRE)
    frozen=assets.load()["reader"]
    hist=Path(frozen["frozen_source"]["path"])
    checked(frozen["frozen_source"],hist)
    evidence=log_evidence(frozen)
    for pid in (source["pid"],child,evidence["engine_pid"]):
        need(not Path(f"/proc/{pid}").exists(),f"original Task 19 PID {pid} is still present")
    pre=read(reader.PREFLIGHT)
    old_pred,old_req=[fingerprint(p) for p in (PREDICT,REQUESTS)]
    proof=match_archives(pre)
    smoke=smoke_receipt()
    need(old_pred==fingerprint(PREDICT) and old_req==fingerprint(REQUESTS),"archive changed while reading")
    print(f"Verified {proof['n']} ordered request/prediction pairs across {proof['n_tasks']} batches",flush=True)

    res_path=reader.READER/"resources.jsonl"
    runtime=list(reuse.rows(res_path))
    need(runtime and all(x["gpu"]["index"]==1 and x["gpu"]["uuid"]==gate["gpu"]["uuid"] for x in runtime),
         "G3 archived GPU telemetry differs")
    before, after = runtime[0]["root"], runtime[-1]["root"]
    opening=earlier["G3_prepare"]["root_before"]
    peak=max(x["root"]["used_bytes"] for x in runtime)
    pred_time=datetime.fromtimestamp(old_pred[3]/1e9, timezone.utc).isoformat()
    wall=(datetime.fromisoformat(source["utc"])-datetime.fromisoformat(launched["utc"])).total_seconds()
    need(datetime.fromisoformat(launched["utc"]) < datetime.fromisoformat(pred_time) <= datetime.fromisoformat(source["utc"]),
         "archive timestamps differ from original GPU stage")
    original_status = pipeline.STATUS.read_bytes()
    original_reader_status = (reader.READER/"status.json").read_bytes()
    need(read(pipeline.STATUS)==source and read(reader.READER/"status.json")==reader_status,"status changed during recovery")
    ARCHIVE.mkdir()
    snapshots={}
    for name, raw in (("pipeline_status_before.json",original_status),("reader_status_before.json",original_reader_status)):
        path=ARCHIVE/name
        with path.open("xb") as f:
            f.write(raw)
        snapshots[name]=record(path)

    end=utc()
    error={"type":"KeyError","missing_field":"df_opening","original_g3_exit_code":1,"failed_utc":source["utc"],
           "cause":"Task 19's signed preflight did not supply the disk snapshot that the shared reader loads after inference.",
           "after_engine_shutdown":True,
           "repair":"Provide the snapshot from Task 19 prep and check it before engine startup. Recover the original outputs on CPU."}
    pred_ref={"path":str(PREDICT),"size":old_pred[2],"sha256":proof["file_sha256"][str(PREDICT)]}
    reuse.write_new(AUDIT,{"status":"passed","recovery_cpu_only":True,"inference_reused":True,
                           "original_pipeline_pid":source["pid"], "original_reader_pid":child,
                           "completed_prediction_archive_utc":pred_time,"recovered_utc":end,
                           "failure":error,"effective_flag_from_log":evidence,"engine_shutdown":True,
                           "original_status_snapshots":snapshots,"original_preflight":record(reader.PREFLIGHT),
                           "predictions":pred_ref,"coverage":proof,"smoke":smoke,"cpu_tests":record(TESTS),
                           "cpu_test_count":suite["tests"],"raw_files_open_mode":"read-only"})
    config={"status":"complete","pid":child,"started_utc":launched["utc"],"ended_utc":source["utc"],
            "parameters":frozen["engine"],"generation":frozen["generation"],"source_sha256":frozen["frozen_source"]["sha256"],
            "package_versions":frozen["packages"],"cuda_visible_devices":"1","cuda_device_order":"PCI_BUS_ID",
            "effective_disable_cascade_attn":True,"effective_disable_cascade_attn_source":evidence["source"],
            "effective_disable_cascade_attn_log_excerpt":evidence["excerpt"],
            "effective_disable_cascade_attn_log_line":evidence["line_number"],
            "engine_shutdown":True,"engine_shutdown_evidence":"The traceback occurs after the engine shutdown and monitor stop in run_engine.",
            "automatic_retry":False,"elapsed_seconds":wall,"predictions":proof["n"],"tasks":proof["n_tasks"],
            "smoke":smoke,"predictions_sha256":pred_ref["sha256"],
            "before":before,"df_after_reader":after,"df_opening":opening,
            "df_opening_source":"Archived prepare-reader receipt root_before, replacing the missing field in the original preflight.",
            "peak_observed_root_used_bytes":peak,"peak_observed_root_delta_bytes":max(0,peak-opening["used_bytes"]),
            "peak_gpu1_mib":max(x["gpu"]["memory_used_mib"] for x in runtime),"n_resource_samples":len(runtime),
            "input_provenance":{"preflight":record(reader.PREFLIGHT),"requests":pre["requests"],
                                "task":19,"batch_size":reader.BATCH,"order":list(data.NAMES)},
            "recovery":{"cpu_only":True,"inference_reused":True,"receipt":record(AUDIT),
                        "completed_prediction_archive_utc":pred_time,"recovered_utc":end,"failure":error}}
    reuse.write_new(reader.CONFIG,config)
    config_ref=record(reader.CONFIG)
    reuse.write_new(INFERENCE,{"stage":"inference","utc":end,"n":proof["n"],"run_config_sha256":config_ref["sha256"],
                                "predictions_sha256":pred_ref["sha256"],"cpu_recovery":record(AUDIT)})
    stage={"stage":"complete","utc":end,"gpu":True,"exit_code":0,"original_g3_exit_code":1,
           "exit_code_scope":"CPU verification acceptance", "child_pid":child, "wall_seconds":wall,
           "original_gpu_gate":gate,"outputs":{reader.CONFIG.name:config_ref,AUDIT.name:record(AUDIT)},
           "root_before":before,"root_after":after,"recovery_cpu_only":True,"inference_reused":True}
    reuse.write_new(RECEIPT,stage)
    earlier["G3"]=stage
    size=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    need(size<300_000_000,"Task 19 overlay exceeds 300 MB")
    run={"status":"evaluation_complete","task":19,"locked_first_use":True,
         "automatic_retry":False,"gpu":1,"shared_lock":str(pipeline.GPU_LOCK),"priority":phase["priority"],
         "settings_frozen":record(assets.PRE),"budgets":data.BUDGETS,"strategy_order":list(data.NAMES),
         "reader_batch_size":reader.BATCH,"phase_p":record(pipeline.PHASE_P),"stages":earlier,
         "reader_run_config":config_ref,"effective_disable_cascade_attn":True,
         "effective_disable_cascade_attn_source":{"log":evidence["log"],"line_number":evidence["line_number"],
                                                  "kind":evidence["source"]},
         "G3_recovery":record(AUDIT),"root_before":earlier["G0"]["root_before"],"root_after":after,
         "small_artifact_bytes":size}
    reuse.write_new(CONFIG,run)
    need(old_pred==fingerprint(PREDICT) and old_req==fingerprint(REQUESTS),"archive changed before publishing")
    publish(reader.READER,{"pid":child,"stage":"inference","status":"complete","tasks":proof["n_tasks"],
                            "predictions":proof["n"],"cpu_recovery":record(AUDIT)})
    pipeline.publish("all","evaluation_complete",pid=source["pid"],config=record(CONFIG),lock_released=True,
                     cpu_recovery=record(AUDIT),recovered_by_pid=os.getpid())
    print("G3 CPU recovery complete, SHA256",pred_ref["sha256"],flush=True)


if __name__=="__main__":
    recover()
