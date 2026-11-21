"""CPU verification of Task 08 optimizer telemetry, saved adapters and Swift rewards."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.grpo_reward import score_completion
from scripts.grpo_prep_summary import sha256

ROOT = Path("/root/userfolder/ctx-compressor/grpo_v2")
RESULTS = PROJECT / "eval/grpo_v2"
METRICS = ("loss", "reward", "reward_std", "frac_reward_zero_std", "kl", "grad_norm",
           "learning_rate", "completions/mean_length", "completions/clipped_ratio", "memory(GiB)")
USER_START = "<|im_start|>user\n"
RESPONSE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def json_rows(path):
    with path.open("r", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def dataset_states(path):
    prompts = {}
    n = 0
    for row in json_rows(path):
        msg, = row["messages"]
        if msg["role"] != "user":
            raise ValueError("expected one user message per state")
        key = msg["content"]
        state = json.loads(row["reward_state_json"])
        if key in prompts and prompts[key] != state:
            raise ValueError("identical user text corresponds to ambiguous reward state")
        prompts[key] = state
        n += 1
    return n, prompts


def reward_recheck(log_path, prompts, n_steps):
    n = mismatches = 0
    per_step = Counter()
    for row in json_rows(log_path):
        columns = [row[k] for k in ("step", "prompt", "completion", "CtxStepReward")]
        if not all(isinstance(column, list) for column in columns) or len(set(map(len, columns))) != 1:
            raise ValueError("misaligned Swift completion log")
        for step, prompt, completion, logged in zip(*columns):
            if not prompt.startswith(USER_START) or not prompt.endswith(RESPONSE_SUFFIX):
                raise ValueError("unrecognized user prompt in Swift completion log")
            user = prompt[len(USER_START):-len(RESPONSE_SUFFIX)]
            if user not in prompts:
                raise ValueError("Swift logged a prompt absent from the dataset")
            for eos in ("<|im_end|>", "<|endoftext|>"):
                if completion.endswith(eos):
                    completion = completion[:-len(eos)]
                    break
            score = score_completion(completion, prompts[user], lam=0.3)["reward"]
            logged = float(logged)
            n += 1
            per_step[int(step)] += 1
            mismatches += int(not math.isfinite(logged) or abs(score - logged) > 1e-5)
    if mismatches or set(per_step) != set(range(1, n_steps + 1)) or min(per_step.values(), default=0) < 32:
        raise ValueError(f"reward audit failed: {mismatches}/{n} mismatches, {len(per_step)}/{n_steps} steps")
    return {"n": n, "mismatches": mismatches, "unmatched_prompts": 0,
            "step_counts": dict(sorted(per_step.items())), "tolerance": 1e-5,
            "optimizer_steps": n_steps, "n_planned_completions": 32 * n_steps,
            "extra_logged_completions": n - 32 * n_steps,
            "definition": "Re-evaluate every Swift logged row, including its terminal repeated table, with one trailing EOS removed."}


def resolved_config(train, data, steps):
    args = json.loads((train / "args.json").read_text())
    expected = {"rlhf_type": "grpo", "model": str(ROOT / "merged_v2_ckpt2000"),
                "dataset": [str(data)], "template": "qwen3", "model_type": "qwen3",
                "train_type": "lora", "target_modules": ["all-linear"], "lora_rank": 16,
                "lora_alpha": 32, "lora_dropout": .05, "enable_thinking": False,
                "add_non_thinking_prefix": True, "torch_dtype": "bfloat16", "attn_impl": "sdpa",
                "reward_funcs": ["ctx_step_reward"], "remove_unused_columns": False,
                "learning_rate": 5e-6, "beta": .04, "num_generations": 8, "temperature": 1.,
                "top_p": 1., "top_k": -1, "max_completion_length": 256, "max_length": 8392,
                "per_device_train_batch_size": 4, "gradient_accumulation_steps": 8,
                "lr_scheduler_type": "cosine", "warmup_ratio": .03, "max_steps": steps, "num_train_epochs": 1,
                "save_steps": 125, "seed": 20261120, "data_seed": 20261120, "num_iterations": 1,
                "logging_nan_inf_filter": False, "dynamic_sample": False, "bf16": True,
                "gradient_checkpointing": True, "use_vllm": True, "vllm_mode": "colocate",
                "vllm_gpu_memory_utilization": .30, "vllm_enforce_eager": True, "sleep_level": 0,
                "log_completions": True, "logging_steps": 1}
    differences = {k: [v, args.get(k)] for k, v in expected.items() if args.get(k) != v}
    if differences:
        raise ValueError(f"unexpected Swift configuration: {differences}")
    return expected


def curve(train, steps):
    obs = list(json_rows(train / "optimizer_steps.jsonl"))
    swift = [r for r in json_rows(train / "logging.jsonl") if "loss" in r]
    if len(obs) != steps or len(swift) != steps:
        raise ValueError("missing or duplicated optimizer step")
    rows = []
    for step, (row, own) in enumerate(zip(swift, obs), 1):
        if row["global_step/max_steps"] != f"{step}/{steps}" or own["step"] != step:
            raise ValueError("optimizer step ordering changed")
        values = {k: float(row[k]) for k in METRICS}
        values.update({k: own[k] for k in ("step_seconds", "torch_peak_allocated_bytes", "torch_peak_reserved_bytes")})
        if not all(math.isfinite(v) for v in values.values()) or values["step_seconds"] <= 0:
            raise ValueError(f"invalid training metrics at step {step}")
        if any(not math.isclose(values[k], own[k], abs_tol=1e-6, rel_tol=1e-6) for k in ("loss","grad_norm")):
            raise ValueError(f"Swift and callback metrics differ at {step}")
        rows.append({"step": step, **values})
    return rows


def stored_checkpoints(train, steps):
    ordered = sorted(train.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    saved = [int(p.name.split("-")[-1]) for p in ordered]
    scheduled = set(range(125, steps + 1, 125))
    if not saved or not scheduled.issubset(saved) or set(saved) - scheduled - {steps}:
        raise ValueError(f"unexpected saved checkpoint steps: {saved}")
    for p, step in zip(ordered, saved):
        if json.loads((p / "trainer_state.json").read_text())["global_step"] != step:
            raise ValueError(f"step mismatch at {p}")
        cfg=json.loads((p / "adapter_config.json").read_text())
        if cfg["r"] != 16 or cfg["lora_alpha"] != 32 or (p / "adapter_model.safetensors").stat().st_size <= 0:
            raise ValueError(f"invalid LoRA files at {p}")
    return ordered, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    args = parser.parse_args()
    train, data = args.train_dir.resolve(), args.data.resolve()
    if train != ROOT / "train" or ROOT not in data.parents:
        raise ValueError("training validation requires Task 08 files")
    target = RESULTS / "training_checks.json"
    csv_path = RESULTS / "training_curve.csv"
    if target.exists() or csv_path.exists():
        raise FileExistsError("training verification outputs already exist")
    runtime = json.loads((train / "training_runtime.json").read_text())
    launch = json.loads((train / "launch_args.json").read_text())
    if (runtime["status"] != "ok" or runtime["nonfinite"] or runtime["oom"]
            or (train / "training_failure.json").exists() or launch["lambda"] != .3
            or launch["cuda_visible_devices"] != "1" or launch["cuda_device_order"] != "PCI_BUS_ID"):
        raise ValueError("training process failed or used an unexpected configuration")
    n, users = dataset_states(data)
    steps = n // 4
    if not 1 <= steps <= 500 or n > 2000:
        raise ValueError(f"invalid prompt count {n}")
    config = resolved_config(train, data, steps)
    rows = curve(train, steps)
    recheck = reward_recheck(train / "completions.jsonl", users, steps)
    checkpoints, saved = stored_checkpoints(train, steps)
    with (train / "gpu1_monitor.csv").open() as f:
        monitor = list(csv.DictReader(f))
    if not monitor:
        raise ValueError("missing card measurements")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "ok", "runtime": runtime, "n_prompts": n, "optimizer_steps": steps,
              "resolved_config": config, "launch_args": launch, "data_sha256": sha256(data),
              "reward_recheck": recheck, "curve": str(csv_path), "curve_sha256": sha256(csv_path),
              "checkpoints": [str(p) for p in checkpoints], "checkpoint_steps": saved,
              "adapter_sha256": {p.name: sha256(p / "adapter_model.safetensors") for p in checkpoints},
              "peak_gpu1_used_mib": max(int(m["memory_used_mib"]) for m in monitor), "n_gpu_samples": len(monitor),
              "peak_torch_allocated_bytes": max(r["torch_peak_allocated_bytes"] for r in rows),
              "peak_torch_reserved_bytes": max(r["torch_peak_reserved_bytes"] for r in rows)}
    with target.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": "ok", "reward_records": recheck["n"], "checkpoint_steps": saved}))


if __name__ == "__main__":
    main()

