#!/usr/bin/env python3
"""Local Qwen3-8B reader: exact prompts, GPU 1 only, smoke then full evaluation."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.reader import parse_answer, render_passages, render_user_message
from src.qa_metrics import score_answer

MAX_MODEL_LEN = 40960
MAX_NEW_TOKENS = 48


def utc():
    return datetime.now(timezone.utc).isoformat()


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_df():
    fields = subprocess.check_output(["df", "-B1", "--output=size,used,avail", "/"], text=True).splitlines()[-1].split()
    return {"at_utc": utc(), "size_bytes": int(fields[0]), "used_bytes": int(fields[1]), "available_bytes": int(fields[2])}


def gpu_info():
    value = subprocess.check_output([
        "nvidia-smi", "-i", "1", "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    return {"physical_index": int(value[0]), "uuid": value[1], "name": value[2],
            "memory_total_mib": int(value[3]), "memory_used_mib": int(value[4]),
            "driver_version": value[5], "utilization_percent": int(value[6])}


def descendants():
    rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines()
    children = defaultdict(list)
    for row in rows:
        pid, parent = map(int, row.split())
        children[parent].append(pid)
    found, pending = {os.getpid()}, [os.getpid()]
    while pending:
        for child in children[pending.pop()]:
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


class ResourceMonitor:
    def __init__(self, baseline):
        self.baseline = baseline
        self.stopped = threading.Event()
        self.thread = None
        self.samples = []
        self.problems = []
        self.observed_gpu_processes = {}
        self.disk_alerts = []
        self.disk_incidents = []
        self.handled_disk_alerts = 0

    def sample(self):
        disk, gpu = root_df(), gpu_info()
        delta = disk["used_bytes"] - self.baseline["root_df_used_bytes"]
        self.samples.append({"at_utc": utc(), "root_used_bytes": disk["used_bytes"],
                             "root_delta_bytes": delta, "gpu_memory_used_mib": gpu["memory_used_mib"],
                             "gpu_utilization_percent": gpu["utilization_percent"]})
        if delta > self.baseline["root_growth_limit_bytes"]:
            from diagnose_reader_disk import writable_root_fds
            own_pids = descendants()
            descriptors = writable_root_fds()
            self.disk_alerts.append({"at_utc": utc(), "root_delta_bytes": delta,
                                     "own_writable_root_files": [fd for fd in descriptors if fd['pid'] in own_pids],
                                     "other_visible_writable_root_files": [
                                         {k: fd[k] for k in ('pid', 'process', 'bytes', 'allocated_bytes')}
                                         for fd in descriptors if fd['pid'] not in own_pids]})
        if gpu["uuid"] != self.baseline["gpu"]["uuid"]:
            self.problems.append("Physical GPU 1 identity changed")
        ours = descendants()
        output = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
        for line in output.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) in ours:
                self.observed_gpu_processes[parts[0]] = parts[1]
                if parts[1] != self.baseline["gpu"]["uuid"]:
                    self.problems.append(f"Our process {parts[0]} appeared on an unauthorized GPU")

    def loop(self):
        while not self.stopped.is_set():
            try:
                self.sample()
            except Exception as error:
                self.problems.append(f"Resource monitor failed: {error!r}")
            self.stopped.wait(2)

    def start(self):
        self.sample()
        self.check()
        self.thread = threading.Thread(target=self.loop, name="reader-resources", daemon=True)
        self.thread.start()

    def check(self):
        if self.problems:
            raise RuntimeError(self.problems[-1])
        if self.handled_disk_alerts == len(self.disk_alerts):
            return
        print(f"{utc()} Root-disk threshold alert: pausing new model submissions and inspecting root writable-file metadata", flush=True)
        pending = self.disk_alerts[self.handled_disk_alerts:]
        if any(fd['bytes'] > 100_000_000 for alert in pending for fd in alert['own_writable_root_files']):
            raise RuntimeError("Root-disk alert includes a large writable file owned by this task; stop and investigate")
        checks, stable = [], 0
        started = time.monotonic()
        while stable < 2:
            current = root_df()
            delta = current['used_bytes'] - self.baseline['root_df_used_bytes']
            checks.append({**current, 'delta_bytes': delta})
            stable = stable + 1 if delta <= self.baseline['root_growth_limit_bytes'] else 0
            if stable == 2:
                break
            if time.monotonic() - started >= 60:
                raise RuntimeError(f"Root disk still exceeds the 300 MB limit after pausing for investigation: {delta} bytes")
            time.sleep(2)
        pending = self.disk_alerts[self.handled_disk_alerts:]
        # Include alerts that arrived while the two below-limit rechecks ran.
        if any(fd['bytes'] > 100_000_000 for alert in pending for fd in alert['own_writable_root_files']):
            raise RuntimeError("A large writable root file owned by this task appeared during the pause; stop and investigate")
        self.disk_incidents.append({
            'action': 'paused new submissions; inspected visible writable root-file metadata; resumed after two below-limit measurements',
            'observation': 'root filesystem used-byte count rose temporarily; exact writer attribution is not established',
            'alerts': pending, 'recheck_measurements': checks,
            'actual_tempfile_directory': tempfile.gettempdir(), 'pause_seconds': time.monotonic() - started})
        self.handled_disk_alerts = len(self.disk_alerts)
        print(f"{utc()} Root disk is below 300 MB again; diagnostic incident retained; resuming submissions", flush=True)

    def stop(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def validate_runtime_environment():
    if PROJECT != Path("/root/workplace/ctx-compressor"):
        raise RuntimeError("This runner must execute on 5880 in the formal project directory")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly equal 1")
    expected = Path("/root/userfolder/ctx-compressor")
    if Path(sys.prefix).resolve() != expected / "envs/reader":
        raise RuntimeError("Reader must use its independent external environment")
    for name in ("TMPDIR", "PIP_CACHE_DIR", "XDG_CACHE_HOME", "HF_HOME", "MODELSCOPE_CACHE",
                 "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "PYTHONPYCACHEPREFIX"):
        value = os.environ.get(name)
        if not value or not Path(value).resolve().is_relative_to(expected / "cache"):
            raise RuntimeError(f"{name} is not redirected to external reader cache")
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        if os.environ.get(name):
            raise RuntimeError(f"Proxy variable must be unset: {name}")


def prepare_requests(model_path):
    # build_corpus reconstructs exactly the existing pid order; its index is not
    # queried. No trajectories, policies, budgets, or retrievals are rerun.
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from src.build_traj import build_corpus

    dev = read_jsonl(PROJECT / "data/raw/musique/musique_ans_v1.0_dev.jsonl")
    dev_by_id = {example["id"]: example for example in dev}
    old_tokenizer = Tokenizer.from_file(str(PROJECT / "data/raw/qwen3_tokenizer/tokenizer.json"))
    corpus, unused_index, unused_keys = build_corpus(
        dev, lambda text: len(old_tokenizer.encode(text, add_special_tokens=False).ids))
    del unused_index, unused_keys
    original = read_jsonl(PROJECT / "data/feasibility/corpus_dev_ntok.jsonl")
    assert len(original) == len(corpus) == 21100
    for pid, (row, paragraph) in enumerate(zip(original, corpus)):
        assert row["pid"] == pid and row["title"] == paragraph["title"]
        assert row["sha1"] == hashlib.sha1(paragraph["text"].encode()).hexdigest()
        assert row["ntok"] == paragraph["ntok"]
    tasks = {row["task"]: row for row in read_jsonl(PROJECT / "data/feasibility/tasks_dev150.jsonl")}
    contexts = read_jsonl(PROJECT / "data/feasibility/final_contexts.jsonl")
    by_key = {(row["task"], row["config"]): row for row in contexts}
    feasible = json.loads((PROJECT / "eval/feasibility/feasibility_results.json").read_text())
    config_order = list(feasible["configs"])
    assert len(tasks) == 150 and len(contexts) == len(by_key) == 2850 and len(config_order) == 19
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    requests, unique_examples = [], set()
    for task_id in sorted(tasks):
        task = tasks[task_id]
        base_questions = by_key[(task_id, "no_compress")]["questions"]
        unique_examples.update(q["id"] for q in base_questions)
        for config in [*config_order, "no_context"]:
            row = by_key[(task_id, config)] if config != "no_context" else None
            pids = row["keep_after"] if row is not None else []
            assert len(pids) == len(set(pids))
            questions = row["questions"] if row is not None else base_questions
            assert questions == base_questions and [q["id"] for q in questions] == task["example_ids"]
            passages = render_passages(corpus, pids)
            for question_index, question in enumerate(questions):
                raw = dev_by_id[question["id"]]
                assert all(question[k] == raw[k] for k in ("question", "answer", "answer_aliases"))
                user_text = render_user_message(passages, question["question"], no_context=config == "no_context")
                token_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
                if len(token_ids) + MAX_NEW_TOKENS > MAX_MODEL_LEN:
                    raise RuntimeError(f"Prompt overflow, refusing truncation: {config}/{task_id}/{question_index}: {len(token_ids)} + 48 > 40960")
                requests.append({"config": config, "task": task_id, "question_index": question_index,
                                 "example_id": question["id"], "question": question["question"],
                                 "answer": question["answer"], "answer_aliases": question["answer_aliases"],
                                 "question_hops": len(raw["question_decomposition"]),
                                 "evidence_complete": set(task["question_gold"][question_index]) <= set(pids),
                                 "prompt_sha1": hashlib.sha1(user_text.encode()).hexdigest(),
                                 "prompt_tokens": len(token_ids), "token_ids": token_ids})
    assert len(unique_examples) == 600 and len(requests) == 12000
    assert len({(r["config"], r["task"], r["question_index"]) for r in requests}) == 12000
    return requests, {"corpus_paragraphs_verified": len(corpus), "distinct_questions": len(unique_examples),
                      "n_prompts": len(requests), "input_truncations": 0,
                      "max_prompt_tokens_by_config": {c: max(r["prompt_tokens"] for r in requests if r["config"] == c)
                                                      for c in [*config_order, "no_context"]},
                      "max_prompt_tokens": max(r["prompt_tokens"] for r in requests),
                      "max_new_tokens": MAX_NEW_TOKENS, "max_model_len": MAX_MODEL_LEN}


def generate_batch(llm, sampling, requests, monitor):
    monitor.check()
    outputs = llm.generate([{"prompt_token_ids": row["token_ids"]} for row in requests],
                           sampling_params=sampling, use_tqdm=False)
    if len(outputs) != len(requests):
        raise RuntimeError("vLLM returned an incomplete batch")
    records = []
    for source, output in zip(requests, outputs):
        assert output.prompt_token_ids == source["token_ids"], "Input tokens changed or were truncated"
        assert len(output.outputs) == 1
        completion = output.outputs[0]
        raw = completion.text
        prediction = parse_answer(raw)
        scores = score_answer(prediction, source["answer"], source["answer_aliases"])
        record = {key: source[key] for key in ("config", "task", "question_index", "example_id",
                                                "prompt_sha1", "prompt_tokens", "evidence_complete", "question_hops")}
        record.update(raw_output=raw, pred=prediction, **scores,
                      output_tokens=len(completion.token_ids), output_token_ids=list(completion.token_ids),
                      finish_reason=completion.finish_reason, stop_reason=completion.stop_reason,
                      input_truncated=False, long_answer=len(prediction.split()) > 20)
        assert record["output_tokens"] <= MAX_NEW_TOKENS
        records.append(record)
    return records


def smoke_markdown(records, requests, checks):
    source = {(r["config"], r["task"], r["question_index"]): r for r in requests}
    lines = ["# Qwen3-8B 阅读小样", "", "前5个任务；oracle|reselect|mid 与 no_compress 各20题。",
             "模板、greedy、max_tokens=48、bf16、关闭thinking与全量保持一致。", "",
             "自动检查：`" + compact(checks) + "`。", ""]
    for record in records:
        original = source[(record["config"], record["task"], record["question_index"])]
        lines += [f"## {record['config']} / task {record['task']} / question {record['question_index']}", "",
                  f"- 问题：{original['question']}",
                  f"- 标准答案：{compact(original['answer'])}；aliases：{compact(original['answer_aliases'])}",
                  f"- prompt token：{record['prompt_tokens']}；output token：{record['output_tokens']}；finish：{record['finish_reason']}",
                  f"- 原始输出：{compact(record['raw_output'])}", f"- 解析结果：{compact(record['pred'])}",
                  f"- EM：{record['em']}；F1：{record['f1']}", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/root/userfolder/ctx-compressor/models/Qwen3-8B"))
    parser.add_argument("--await-smoke-review", action="store_true", help="Keep the engine alive until the assistant records its local smoke review")
    parser.add_argument("--reuse-reviewed-smoke", action="store_true", help="Reuse the original 40 checked outputs after a pre-full-run implementation correction; never resample them")
    args = parser.parse_args()
    validate_runtime_environment()
    args.model = args.model.resolve()
    if not args.model.is_relative_to(Path("/root/userfolder/ctx-compressor/models")):
        parser.error("Model weights must be on the external userfolder filesystem")
    output_dir, data_dir = PROJECT / "eval/reader", PROJECT / "data/reader"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = data_dir / "predictions.jsonl"
    if predictions_path.exists() or ((output_dir / "smoke_samples.md").exists() and not args.reuse_reviewed_smoke):
        parser.error("Reader artifacts already exist; refusing an accidental second run")
    prior_smoke_run = None
    if args.reuse_reviewed_smoke:
        review = json.loads((output_dir / "smoke_review.json").read_text())
        prior_smoke_run = json.loads((output_dir / "run_config_smoke_initial.json").read_text())
        if review.get("passed") is not True or prior_smoke_run["predictions"] != 0:
            parser.error("Reusing smoke requires a passed local review and zero prior full predictions")
        if review["smoke_predictions_sha256"] != sha256(data_dir / "smoke_predictions.jsonl"):
            parser.error("Reviewed smoke outputs changed")
    manifest_path = output_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not (manifest.get("status") == "verified" and manifest.get("verified_file_count") == 16
            and len(manifest["files"]) == 16
            and all(row["status"] == "verified" for row in manifest["files"])
            and Path(manifest["model_root"]).resolve() == args.model):
        raise RuntimeError("All 16 local model files must be verified before inference")
    native_config = json.loads((args.model / "config.json").read_text())
    assert native_config["max_position_embeddings"] == MAX_MODEL_LEN
    assert native_config.get("rope_scaling") is None, "YaRN / RoPE scaling must remain disabled"
    baseline = json.loads((output_dir / "preflight.json").read_text())
    first_gpu, first_disk = gpu_info(), root_df()
    if first_gpu["memory_used_mib"] >= 2048:
        raise RuntimeError("NEED: physical GPU 1 has at least 2 GiB allocated; stop before GPU initialization")
    if first_disk["used_bytes"] - baseline["root_df_used_bytes"] > baseline["root_growth_limit_bytes"]:
        raise RuntimeError("Root filesystem grew beyond 300 MB; stop before inference")
    monitor = ResourceMonitor(baseline)
    started, started_utc = time.monotonic(), utc()
    parameters = {"model": str(args.model), "tokenizer": str(args.model), "trust_remote_code": False,
                  "dtype": "bfloat16", "max_model_len": MAX_MODEL_LEN, "gpu_memory_utilization": 0.85,
                  "tensor_parallel_size": 1, "enable_prefix_caching": True, "enable_chunked_prefill": True,
                  "max_num_seqs": 16, "max_num_batched_tokens": 8192, "enforce_eager": True, "seed": 20261117}
    run = {"started_at_utc": started_utc, "command": "bash scripts/run_reader_job.sh " + shlex.join(sys.argv[1:]),
           "gpu": first_gpu, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
           "df_preflight": baseline, "df_before_reader": first_disk, "parameters": parameters,
           "model_manifest_sha256": sha256(manifest_path), "native_model_config": native_config,
           "vllm_worker_multiproc_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
           "generation": {"temperature": 0, "max_tokens": MAX_NEW_TOKENS, "seed": 20261117,
                          "enable_thinking": False, "n": 1, "system_messages": 0},
           "package_versions": {name: version(name) for name in ("vllm", "torch", "transformers", "tokenizers", "numpy", "bm25s", "scipy")},
           "cache_environment": {key: os.environ.get(key) for key in ("TMPDIR", "PIP_CACHE_DIR", "XDG_CACHE_HOME", "HF_HOME", "MODELSCOPE_CACHE", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "PYTHONPYCACHEPREFIX")},
           "input_files_sha256": {name: sha256(PROJECT / name) for name in (
               "data/feasibility/final_contexts.jsonl", "data/feasibility/tasks_dev150.jsonl",
               "data/feasibility/corpus_dev_ntok.jsonl", "data/raw/musique/musique_ans_v1.0_dev.jsonl")},
           "stage": "preparing", "predictions": 0, "completed": False}
    if prior_smoke_run is not None:
        assert prior_smoke_run["parameters"] == parameters
        assert prior_smoke_run["generation"] == run["generation"]
        assert prior_smoke_run["model_manifest_sha256"] == run["model_manifest_sha256"]
        assert prior_smoke_run["input_files_sha256"] == run["input_files_sha256"]
        run["smoke_reused_without_regeneration"] = True
        run["prior_smoke_run_config"] = "eval/reader/run_config_smoke_initial.json"
        run["prior_smoke_elapsed_seconds"] = prior_smoke_run["elapsed_seconds"]
    llm = None
    failure = None
    try:
        monitor.start()
        phase = time.monotonic()
        print(f"{utc()} Preparing and length-checking all 12000 prompts before model initialization", flush=True)
        requests, preflight = prepare_requests(args.model)
        write_json(output_dir / "prompt_preflight.json", preflight)
        run["prompt_preflight"] = preflight
        run["prepare_seconds"] = time.monotonic() - phase
        monitor.check()
        # Only this source file imports vLLM. The old ctxc remains torch-free.
        from vllm import LLM, SamplingParams
        phase = time.monotonic()
        run["gpu_immediately_before_initialization"] = gpu_info()
        if run["gpu_immediately_before_initialization"]["memory_used_mib"] >= 2048:
            raise RuntimeError("NEED: physical GPU 1 became occupied before model initialization")
        print(f"{utc()} Initializing local Qwen3-8B on physical GPU 1", flush=True)
        llm = LLM(**parameters)
        sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, seed=20261117)
        run["sampling_params_repr"] = repr(sampling)
        run["engine_init_seconds"] = time.monotonic() - phase
        phase = time.monotonic()
        run["stage"] = "smoke"
        smoke_requests = [r for r in requests if r["task"] < 5 and r["config"] in ("oracle|reselect|mid", "no_compress")]
        assert len(smoke_requests) == 40
        if args.reuse_reviewed_smoke:
            smoke = read_jsonl(data_dir / "smoke_predictions.jsonl")
            sources = {(r["config"], r["task"], r["question_index"]): r for r in smoke_requests}
            assert len(smoke) == 40
            assert {(r["config"], r["task"], r["question_index"]) for r in smoke} == set(sources)
            for row in smoke:
                source = sources[row["config"], row["task"], row["question_index"]]
                for field in ("example_id", "prompt_sha1", "prompt_tokens", "question_hops", "evidence_complete"):
                    assert row[field] == source[field], f"Reused smoke changed: {field}"
                assert not row["input_truncated"] and row["pred"] == parse_answer(row["raw_output"])
                assert {k: row[k] for k in ("em", "f1")} == score_answer(row["pred"], source["answer"], source["answer_aliases"])
        else:
            smoke = generate_batch(llm, sampling, smoke_requests, monitor)
            with (data_dir / "smoke_predictions.jsonl").open("w") as stream:
                for row in smoke:
                    stream.write(compact(row) + "\n")
        checks = {"n": 40, "length_finished": sum(r["finish_reason"] == "length" for r in smoke),
                  "empty_predictions": sum(not r["pred"] for r in smoke),
                  "long_answers_over_20_words": sum(r["long_answer"] for r in smoke),
                  "unclosed_think": sum("<think>" in r["pred"] for r in smoke),
                  "max_no_compress_prompt_tokens": max(r["prompt_tokens"] for r in smoke if r["config"] == "no_compress")}
        monitor.check()
        # The fixed 48-token output budget is part of the protocol. Hitting it
        # is a recorded model/format outcome, not truncation of the input.
        checks["automatic_pass"] = not (checks["empty_predictions"] or checks["unclosed_think"])
        if args.reuse_reviewed_smoke:
            write_json(output_dir / "smoke_checks_after_review.json", checks)
        else:
            write_json(output_dir / "smoke_checks.json", checks)
            (output_dir / "smoke_samples.md").write_text(smoke_markdown(smoke, smoke_requests, checks))
        run["smoke_seconds"] = time.monotonic() - phase
        run["smoke_checks"] = checks
        write_json(output_dir / "run_config.json", run)
        print(f"{utc()} Smoke finished: {compact(checks)}", flush=True)
        if not checks["automatic_pass"]:
            raise RuntimeError("Smoke format/length check failed; full evaluation was not started")
        if args.await_smoke_review or args.reuse_reviewed_smoke:
            run["stage"] = "awaiting_local_smoke_review"
            write_json(output_dir / "run_config.json", run)
            print(f"{utc()} Engine kept alive for assistant smoke inspection; waiting for eval/reader/smoke_review.json", flush=True)
            while not (output_dir / "smoke_review.json").exists():
                monitor.check()
                time.sleep(2)
            review = json.loads((output_dir / "smoke_review.json").read_text())
            if review.get("passed") is not True:
                raise RuntimeError("Local smoke review did not pass")
            run["smoke_review"] = review
        run["stage"] = "full"
        phase = time.monotonic()
        finish_reasons, formats = Counter(), Counter()
        with predictions_path.open("w") as stream:
            for task in range(150):
                task_requests = [r for r in requests if r["task"] == task]
                assert len(task_requests) == 80
                records = generate_batch(llm, sampling, task_requests, monitor)
                for record in records:
                    stream.write(compact(record) + "\n")
                    finish_reasons[record["finish_reason"]] += 1
                    formats["empty_predictions"] += not record["pred"]
                    formats["long_answers_over_20_words"] += record["long_answer"]
                stream.flush()
                run["predictions"] += len(records)
                run["full_finish_reasons"] = dict(finish_reasons)
                run["format_checks"] = dict(formats)
                monitor.check()
                if (task + 1) % 5 == 0:
                    write_json(output_dir / "run_config.json", run)
                    print(f"{utc()} {task + 1}/150 tasks; {run['predictions']}/12000 full prompts; finishes={dict(finish_reasons)}", flush=True)
        run["full_seconds"] = time.monotonic() - phase
        assert run["predictions"] == 12000
        run["input_truncations"] = 0
        run["output_limit_hits"] = finish_reasons["length"]
        run["output_limit_hit_rate"] = finish_reasons["length"] / run["predictions"]
        run["completed"] = True
        run["stage"] = "complete"
        print(f"{utc()} All 12000 full predictions completed; input truncations=0; fixed 48-token output-limit hits={finish_reasons['length']}", flush=True)
    except BaseException as error:
        failure = error
        run["stage"] = "failed"
        run["error"] = repr(error)
        run["traceback"] = traceback.format_exc()
        print(run["traceback"], flush=True)
    finally:
        monitor.stop()
        run["finished_at_utc"] = utc()
        run["elapsed_seconds"] = time.monotonic() - started
        run["smoke_and_full_process_seconds"] = run["elapsed_seconds"] + run.get("prior_smoke_elapsed_seconds", 0)
        if prior_smoke_run is not None:
            run["wallclock_from_first_reader_start_seconds"] = (datetime.now(timezone.utc) - datetime.fromisoformat(prior_smoke_run["started_at_utc"])).total_seconds()
        run["peak_gpu_memory_used_mib"] = max((s["gpu_memory_used_mib"] for s in monitor.samples), default=first_gpu["memory_used_mib"])
        run["peak_root_delta_bytes"] = max((s["root_delta_bytes"] for s in monitor.samples), default=0)
        run["resource_monitor_problems"] = monitor.problems
        run["root_disk_incidents"] = monitor.disk_incidents
        run["root_disk_alert_samples"] = len(monitor.disk_alerts)
        run["observed_own_gpu_processes"] = monitor.observed_gpu_processes
        run["resource_samples"] = monitor.samples
        run["maximum_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        run["df_after_reader"] = root_df()
        run["final_root_delta_bytes"] = run["df_after_reader"]["used_bytes"] - baseline["root_df_used_bytes"]
        write_json(output_dir / "run_config.json", run)
        if llm is not None:
            del llm
            gc.collect()
    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
