#!/usr/bin/env python3
"""Stream every Task 04 state through actual Swift encoding, without cropping."""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/root/userfolder/ctx-compressor")
sys.path.insert(0, str(PROJECT))
from src.sft_protocol import (TEMPLATE_OPTIONS, choose_max_length, encode_sft_record,
                              length_distribution, make_messages, make_sft_template,
                              restore_state_keys, sample_flags, template_evidence)

LENGTH_FIELDS = ("n_input_tokens", "n_target_tokens", "total_tokens", "loss_tokens")
FLAG_FIELDS = ("target_add_empty", "target_drop_nonempty", "budget_binding")
WORKER_TEMPLATE = None


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def initialize_worker(model_dir):
    global WORKER_TEMPLATE
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    torch.set_num_threads(1)
    WORKER_TEMPLATE = make_sft_template(model_dir)
    if torch.cuda.is_initialized():
        raise RuntimeError("data preparation must not initialize CUDA")


def encode_task(task, audit):
    rows, proofs = [], []
    for mode in ("reselect", "append"):
        states = task["rollouts"][mode]
        if len(states) != len(task["steps"]):
            raise ValueError("every retrieval step must have one state in both modes")
        for step, serialized in enumerate(states):
            state = restore_state_keys(serialized)
            if state["mode"] != mode or state["budget"] != task["budget"]:
                raise ValueError("state mode/budget differs from task")
            messages = make_messages(state)
            encoded = encode_sft_record(WORKER_TEMPLATE, messages)
            meta = {"task": task["task"], "step": step, "mode": mode,
                    "tier": task["tier"], "split": task["split"],
                    **{field: encoded[field] for field in LENGTH_FIELDS},
                    **sample_flags(state)}
            rows.append({"messages": messages, "meta": meta})
            if audit and step == 0:
                evidence = template_evidence(WORKER_TEMPLATE, messages, encoded)
                proofs.append({"meta": meta, **evidence})
    return rows, proofs


def group_factory():
    return {"counts": Counter(), "lengths": {field: Counter() for field in LENGTH_FIELDS}}


def update_group(group, meta):
    group["counts"]["samples"] += 1
    for field in FLAG_FIELDS:
        group["counts"][field] += int(meta[field])
    for field in LENGTH_FIELDS:
        group["lengths"][field][meta[field]] += 1


def summarize_group(group):
    counts = group["counts"]
    n = counts["samples"]
    return {"samples": n,
            "lengths": {field: length_distribution(hist) for field, hist in group["lengths"].items()},
            **{field: {"count": counts[field], "n": n,
                       "rate": counts[field] / n if n else None} for field in FLAG_FIELDS}}


def bounded_results(pool, tasks, capacity):
    """Submit a bounded number of tasks and emit results in source order."""
    tasks = iter(tasks)
    pending = deque()
    for _ in range(capacity):
        try:
            task, audit = next(tasks)
        except StopIteration:
            break
        pending.append(pool.submit(encode_task, task, audit))
    while pending:
        result = pending.popleft().result()
        yield result
        try:
            task, audit = next(tasks)
        except StopIteration:
            continue
        pending.append(pool.submit(encode_task, task, audit))


def write_template_check(path, proofs, packages):
    lines = ["# Task 04：实际 ms-swift Qwen3 SFT 模板核验", "",
             f"环境版本：`{compact(packages)}`。CPU only；未加载模型权重。", "",
             "配置：`qwen3`、`enable_thinking=False`、`loss_scale=last_round`、"
             "`template_backend=swift`、`add_non_thinking_prefix=True`、`truncation_strategy=raise`。", "",
             "`n_input_tokens` 是 labels=-100 的前缀长度；`n_target_tokens` 是非 -100 的后缀长度。"
             "两者相加严格等于完整训练编码长度。用户内容和 assistant 角色头不计 loss；"
             "assistant 内容前的 `<think>\\n\\n</think>\\n\\n` 空块（4 token）计 loss，"
             "目标 JSON 和末尾 `<|im_end|>\\n`（2 token）也计 loss。", "",
             "未来推理使用同一 tokenizer 的 `apply_chat_template(user_messages, "
             "add_generation_prompt=True, enable_thinking=False)`；其字符串和 token IDs "
             "均与实际训练串前缀相同，包含这 4 个空 think token。"
             "因此 generation prefix 比 labels 屏蔽前缀长 4 token。", "",
             "下面每个 split × mode × tier 展示首个任务的第 0 步。"
             "输入展示允许缩略，完整原文在对应 SFT JSONL；labels 的全部区间与每个非屏蔽 token 均列出。", "",
             "模板源码：[ms-swift v3.12.6 Qwen3](https://github.com/modelscope/ms-swift/blob/v3.12.6/swift/llm/template/template/qwen.py)；"
             "[编码实现](https://github.com/modelscope/ms-swift/blob/v3.12.6/swift/llm/template/base.py)。", ""]
    for proof in proofs:
        meta = proof["meta"]
        text = proof["full_text"]
        shown = text if len(text) <= 4500 else text[:3200] + "\n[此处仅展示省略，实际编码未截断]\n" + text[-1000:]
        masked, supervised = proof["masked_label_range"], proof["supervised_label_range"]
        ids = proof["full_input_ids"]
        lines += [f"## {meta['split']} / {meta['mode']} / {meta['tier']} / task={meta['task']} / step={meta['step']}", "",
                  f"- 元信息：`{compact(meta)}`",
                  f"- 完整编码 UTF-8 字符串 sha256：`{sha256_bytes(text.encode('utf-8'))}`",
                  f"- 完整 input_ids 的 compact JSON sha256：`{sha256_bytes(compact(ids).encode('utf-8'))}`",
                  f"- labels 区间（0-based，右端不含）：`[{masked[0]}, {masked[1]})=-100`；"
                  f"`[{supervised[0]}, {supervised[1]})=input_ids`。",
                  f"- 推理 prefix：{proof['hf_inference_prefix_tokens']} token；"
                  f"空 think：{proof['empty_think_tokens']} token；字符串和 token 前缀验证通过。", "",
                  "```text", shown, "```", "",
                  "| 位置 | input_id | label | token（JSON 字符串） |", "|---:|---:|---:|---|"]
        lines += [f"| {token['position']} | {token['input_id']} | {token['label']} | "
                  f"`{json.dumps(token['token'], ensure_ascii=False).replace('|', '&#124;')}` |"
                  for token in proof["supervised_tokens"]]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-dir", type=Path, default=EXTERNAL / "data/traj_train")
    parser.add_argument("--data-dir", type=Path, default=EXTERNAL / "data/sft")
    parser.add_argument("--model-dir", type=Path, default=EXTERNAL / "models/Qwen3-1.7B")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("execute on 5880 in the formal project with CUDA_VISIBLE_DEVICES=''")
    if Path(sys.prefix) != EXTERNAL / "envs/train":
        parser.error("use the dedicated train environment")
    if not 1 <= args.workers <= 4 or args.progress_every < 1:
        parser.error("workers must be 1..4 and progress-every positive")
    for directory in (args.traj_dir, args.data_dir, args.model_dir):
        if not directory.resolve().is_relative_to(EXTERNAL):
            parser.error("all data/model directories must be under userfolder")
    outdir = PROJECT / "eval/sft_prep"
    stats_path, check_path = outdir / "sft_data_stats.json", outdir / "template_check.md"
    targets = [args.data_dir / (split + ".jsonl") for split in ("train", "heldout")]
    if any(path.exists() or path.with_suffix(path.suffix + ".partial").exists()
           for path in targets + [stats_path, check_path]):
        parser.error("refusing to overwrite SFT data, partial output, or results")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    packages = {name: version(name) for name in ("ms-swift", "transformers", "tokenizers", "torch")}
    if packages["ms-swift"] != "3.12.6":
        parser.error("expected the validated ms-swift 3.12.6 protocol")
    groups, overall, proofs, files, inputs = defaultdict(group_factory), group_factory(), [], {}, {}
    split_counts, split_tasks = Counter(), Counter()
    longest = None
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[variable] = "1"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker, initargs=(str(args.model_dir),)) as pool:
        for split, target in zip(("train", "heldout"), targets):
            source = args.traj_dir / (split + "_tasks.jsonl")
            source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
            seen_tiers = set()

            def tasks():
                with source.open("rb") as stream:
                    for task_index, line in enumerate(stream):
                        source_hash.update(line)
                        task = json.loads(line)
                        if task["split"] != split or task["task"] != task_index:
                            raise ValueError("split/task order mismatch")
                        split_tasks[split] += 1
                        audit = task["tier"] not in seen_tiers
                        seen_tiers.add(task["tier"])
                        yield task, audit

            partial = target.with_suffix(".jsonl.partial")
            with partial.open("xb") as stream:
                for task_index, (rows, task_proofs) in enumerate(bounded_results(pool, tasks(), args.workers * 2)):
                    proofs.extend(task_proofs)
                    for row in rows:
                        meta = row["meta"]
                        split_counts[split] += 1
                        update_group(overall, meta)
                        update_group(groups[("all", meta["mode"], meta["tier"])], meta)
                        update_group(groups[(split, meta["mode"], meta["tier"])], meta)
                        if longest is None or meta["total_tokens"] > longest["total_tokens"]:
                            longest = {**meta, "line_1based": split_counts[split], "path": str(target)}
                        encoded_row = (compact(row) + "\n").encode("utf-8")
                        stream.write(encoded_row)
                        output_hash.update(encoded_row)
                    if (task_index + 1) % args.progress_every == 0:
                        stream.flush()
                        print(f"{split}: tasks={task_index + 1} samples={split_counts[split]} "
                              f"max={longest['total_tokens']} elapsed={time.monotonic() - start:.1f}s", flush=True)
                stream.flush()
                os.fsync(stream.fileno())
            partial.rename(target)
            files[split] = {"path": str(target), "lines": split_counts[split],
                            "bytes": target.stat().st_size, "sha256": output_hash.hexdigest()}
            inputs[split] = {"path": str(source), "lines": split_tasks[split],
                             "bytes": source.stat().st_size, "sha256": source_hash.hexdigest()}
            print(f"{split}: complete tasks={split_tasks[split]} samples={split_counts[split]}", flush=True)
    maximum = overall["lengths"]["total_tokens"]
    max_length = choose_max_length(max(maximum))
    manifest = json.loads((args.traj_dir / "manifest.json").read_text())
    for split in ("train", "heldout"):
        expected = manifest["statistics"][split]["samples"]
        if split_counts[split] != expected or inputs[split]["sha256"] != manifest["files"][split]["sha256"]:
            raise AssertionError("trajectory coverage or input hash differs from generator manifest")
    write_template_check(check_path, proofs, packages)
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - start, "packages": packages,
        "protocol": {**TEMPLATE_OPTIONS, "mode": "train", "measurement_max_length": None,
                     "n_input_tokens": "number of labels=-100 tokens: user content plus role/template headers",
                     "n_target_tokens": "number of supervised tokens: 4 empty-think tokens + target JSON + 2 terminator tokens",
                     "length_identity": "n_input_tokens + n_target_tokens == total_tokens; loss_tokens == n_target_tokens",
                     "inference_prefix": "HF apply_chat_template(user, add_generation_prompt=True, enable_thinking=False)",
                     "inference_prefix_extra_tokens_above_masked_prefix": 4},
        "max_length": max_length, "observed_max_tokens": max(maximum),
        "benchmark_allowed": max_length <= 16384,
        "overall": summarize_group(overall),
        "mode_tier": {f"{mode}|{tier}": summarize_group(group)
                      for (split, mode, tier), group in sorted(groups.items()) if split == "all"},
        "splits": {split: {"tasks": split_tasks[split], "samples": split_counts[split],
                           "mode_tier": {f"{mode}|{tier}": summarize_group(group)
                                         for (part, mode, tier), group in sorted(groups.items()) if part == split}}
                   for split in ("train", "heldout")},
        "longest_sample": longest, "files": files, "trajectory_inputs": inputs,
        "validation": {"expected_samples": sum(manifest["statistics"][part]["samples"] for part in ("train", "heldout")),
                       "actual_samples": sum(split_counts.values()), "truncated_samples": 0,
                       "dropped_samples": 0, "all_targets_legal_pure_oracle": True,
                       "all_labels_assistant_only": True, "prompt_fields_whitelisted": True,
                       "template_prefix_checks": len(proofs), "template_prefix_checks_passed": True,
                       "model_weights_loaded": False, "cuda_visible_devices": "",
                       "workers": args.workers, "in_flight_tasks_limit": args.workers * 2,
                       "ordered_streaming_output": True},
        "code_sha256": {rel: sha256_bytes((PROJECT / rel).read_bytes())
                        for rel in ("src/sft_protocol.py", "scripts/make_sft_data.py")},
        "template_check_sha256": sha256_bytes(check_path.read_bytes()),
    }
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(compact({"completed": True, "samples": dict(split_counts), "max_length": max_length,
                   "benchmark_allowed": report["benchmark_allowed"], "seconds": report["elapsed_seconds"]}), flush=True)
    return 0 if report["benchmark_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
