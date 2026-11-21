#!/usr/bin/env python3
"""Task 07 (Claude): collect GRPO-prep measurements into eval/grpo_prep/ (CPU only).

Writes smoke_train.json, smoke_train_log.csv, smoke_reward_recheck.json, projection.json,
resource_check.json and copies the per-stage check files. Every model-output number is
from the stand-in v1 checkpoint-2000, not a formal result.
"""
import csv, hashlib, json, os, shutil, statistics, subprocess, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
P = Path("/root/userfolder/ctx-compressor/grpo_prep")
OUT = PROJECT / "eval/grpo_prep"
LABEL = "替身 v1 checkpoint-2000，非正式结果"


def load(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smoke():
    rows = [json.loads(l) for l in open(P / "smoke_train/logging.jsonl")]
    steps = [r for r in rows if "loss" in r and "global_step/max_steps" in r]
    keys = ["global_step/max_steps", "loss", "reward", "reward_std", "frac_reward_zero_std", "kl",
            "grad_norm", "learning_rate", "completions/mean_length", "completions/clipped_ratio",
            "memory(GiB)", "train_speed(s/it)", "elapsed_time"]
    with open(OUT / "smoke_train_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in steps:
            w.writerow([r.get(k) for k in keys])
    elapsed = []
    for r in steps:
        t = r["elapsed_time"]
        sec = 0
        for part in t.replace("h", "h ").replace("m", "m ").split():
            if part.endswith("h"):
                sec += 3600 * float(part[:-1])
            elif part.endswith("m"):
                sec += 60 * float(part[:-1])
            elif part.endswith("s"):
                sec += float(part[:-1])
        elapsed.append(sec)
    per_step = [b - a for a, b in zip(elapsed, elapsed[1:])]
    mon = []
    if (P / "smoke_train/gpu0_monitor.csv").exists():
        for line in open(P / "smoke_train/gpu0_monitor.csv"):
            ts, used, util = [x.strip() for x in line.split(",")]
            mon.append((int(ts), int(used), int(util)))
    timing = {}
    for name in ("logging.jsonl",):
        pass
    for r in rows:
        for k, v in r.items():
            if k.startswith("profiling/Time taken"):
                timing.setdefault(k, []).append(v)
    return {
        "model_output_label": LABEL,
        "n_optimizer_steps": len(steps),
        "prompts_per_optimizer_step": 4, "completions_per_optimizer_step": 32,
        "seconds_per_step_excluding_first": {"n": len(per_step[1:]), "mean": statistics.mean(per_step[1:]) if len(per_step) > 1 else None,
                                              "median": statistics.median(per_step[1:]) if len(per_step) > 1 else None},
        "first_step_seconds": elapsed[0] if elapsed else None,
        "total_elapsed_seconds": elapsed[-1] if elapsed else None,
        "torch_memory_GiB_max": max(r.get("memory(GiB)", 0) for r in steps),
        "gpu0_whole_card_mib": {"max": max(m[1] for m in mon), "n_samples": len(mon),
                                 "note": "includes about 1,660 MiB of an external process outside this container"} if mon else None,
        "reward_mean_by_step": [r.get("reward") for r in steps],
        "frac_reward_zero_std_by_step": [r.get("frac_reward_zero_std") for r in steps],
        "frac_reward_zero_std_mean": statistics.mean(r.get("frac_reward_zero_std", 0) for r in steps),
        "kl_by_step": [r.get("kl") for r in steps],
        "nonfinite_loss": any(not isinstance(r.get("loss"), (int, float)) or r["loss"] != r["loss"] for r in steps),
        "profiling_seconds_mean": {k: statistics.mean(v) for k, v in timing.items()},
        "config": {"lr": 5e-6, "beta": 0.04, "num_generations": 8, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                   "max_completion_length": 256, "max_length": 8392, "per_device_train_batch_size": 4,
                   "gradient_accumulation_steps": 8, "lora": "new LoRA r16/a32/d0.05 all-linear on merged SFT",
                   "vllm": "colocate, gpu_memory_utilization 0.30, enforce_eager, sleep_level 0",
                   "lambda": 0.3, "seed": 20261120, "data": "120 train states (20 per mode x tier) from the smoke pool"},
        "interpretation": "30 steps prove nothing about learning; reward trend not interpreted",
    }


def reward_recheck():
    """Recompute every logged swift reward with the eval parser on the logged completion text.

    swift logs completion text with the trailing <|im_end|>; vLLM's eval text never contains it
    (skip_special_tokens). The plugin itself decodes from token ids; here we strip exactly one
    trailing EOS string and recompute independently.
    """
    from src.grpo_reward import score_completion
    path = P / "smoke_train/completions.jsonl"
    data = {}
    for line in open(P / "grpo_data/smoke120.jsonl"):
        r = json.loads(line)
        data[r["messages"][0]["content"]] = json.loads(r["reward_state_json"])
    n = mism = unmatched = 0
    examples = []
    for line in open(path):
        r = json.loads(line)
        for prompt, completion, logged in zip(r["prompt"], r["completion"], r["CtxStepReward"]):
            user = prompt.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>\n<|im_start|>assistant", 1)[0]
            state = data.get(user)
            if state is None:
                unmatched += 1
                continue
            text = completion
            for eos in ("<|im_end|>", "<|endoftext|>"):
                if text.endswith(eos):
                    text = text[: -len(eos)]
                    break
            mine = score_completion(text, state, lam=0.3)["reward"]
            n += 1
            if abs(float(logged) - mine) > 1e-5:
                mism += 1
                if len(examples) < 3:
                    examples.append({"completion": completion, "logged": logged, "recomputed": mine})
    return {"source": str(path), "n": n, "mismatches": mism, "unmatched_prompts": unmatched, "examples": examples,
            "definition": "logged swift reward vs score_completion(logged completion minus one trailing EOS string)"}


def projection(sm):
    st = load(P / "states/v1ckpt2000_smoke30/state_checks.json")
    per_traj = st["rollout_seconds"] / st["n_trajectories"]
    startup = st["engine_and_data_startup_seconds"]
    sec_step = sm["seconds_per_step_excluding_first"]["mean"]
    per_prompt = sec_step / 4
    task05 = 1882  # Task 05 checkpoint-2000 held-out eval seconds (1,494 trajectories)
    return {
        "basis": {"state_rollout_seconds_per_trajectory": per_traj, "state_engine_startup_seconds": startup,
                  "grpo_seconds_per_optimizer_step": sec_step, "grpo_seconds_per_prompt_x8": per_prompt,
                  "heldout_eval_seconds_per_checkpoint_task05": task05,
                  "note": "linear extrapolation from the smoke measurements on one RTX 5880 (GPU 0 shared with a ~1.7 GB external process at ~97% util); estimates, not measurements"},
        "state_generation_hours": {str(n): round((6 * n * per_traj + startup) / 3600, 2) for n in (200, 400, 800)},
        "state_generation_states_estimate": {str(n): round(6 * n * st["n_states"] / st["n_trajectories"]) for n in (200, 400, 800)},
        "grpo_epoch_hours": {str(n): round(n * per_prompt / 3600, 2) for n in (1000, 2000, 4000)},
        "heldout_eval_hours_per_checkpoint": round(task05 / 3600, 2),
    }


def main():
    OUT.mkdir(exist_ok=True)
    copies = {
        P / "states/v1ckpt2000_smoke30/state_checks.json": "state_checks.json",
        P / "probe/v1ckpt2000_smoke30/sampling_probe.json": "sampling_probe.json",
        P / "reference/reader_env/greedy_reference.json": "greedy_reference_reader_env.json",
        P / "reference/grpo_env/greedy_reference.json": "greedy_reference_grpo_env.json",
        P / "reference/merged/greedy_reference.json": "greedy_reference_merged.json",
        P / "checks/swift_template_check.json": "swift_template_check.json",
        P / "merged_v1_ckpt2000/merge_record.json": "merge_record.json",
        P / "states/smoke_lora_plumbing/state_checks.json": "plumbing_state_checks.json",
        P / "step0_fixture/step0_preflight.json": "step0_preflight.json",
    }
    for src, name in copies.items():
        if src.exists():
            shutil.copy2(src, OUT / name)
    sm = smoke()
    sm["reward_recheck"] = reward_recheck()
    write("smoke_train.json", sm)
    write("projection.json", projection(sm))
    marker = P / "start.marker"
    out = subprocess.run(["find", "/", "-xdev", "-newer", str(marker), "-type", "f", "-printf", "%s %p\n"],
                         capture_output=True, text=True)
    files = [l.split(" ", 1) for l in out.stdout.splitlines() if l.strip()]
    ours = [(int(s), p) for s, p in files if p.startswith("/root/workplace/ctx-compressor/")]
    others = [(int(s), p) for s, p in files if not p.startswith("/root/workplace/ctx-compressor/")]
    df = subprocess.run(["df", "-B1", "/"], capture_output=True, text=True).stdout.splitlines()[-1].split()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    write("resource_check.json", {
        "task_owned_root_bytes": sum(s for s, _ in ours), "task_owned_root_files": len(ours),
        "limit_bytes": 300_000_000,
        "other_root_files_newer_than_marker_bytes": sum(s for s, _ in others),
        "other_root_files_note": "files under / changed by other processes (other users, Task 06 is on /root/userfolder); listed for transparency, not task-owned",
        "df_root_start": (P / "df_start.txt").read_text().strip().splitlines()[-1],
        "df_root_end": " ".join(df), "gpu_end": gpu, "compute_apps_end": apps,
        "gpu1_policy": "no Task 07 process ever set CUDA_VISIBLE_DEVICES=1; GPU 1 runs only the Task 06 pipeline"})
    print(json.dumps({"ok": True, "smoke_steps": sm["n_optimizer_steps"]}))


if __name__ == "__main__":
    main()
