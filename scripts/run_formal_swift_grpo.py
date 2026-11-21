"""Run the existing Swift GRPO entry point with physical-card and optimizer telemetry."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    with path.open("x", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def check_logged_finite(logs, step):
    for name in ("loss", "grad_norm"):
        value = logs.get(name)
        if value is None or not math.isfinite(float(value)):
            raise FloatingPointError(f"nonfinite or missing {name} at optimizer step {step}: {value}")


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("Task 08 requires physical GPU 1")
    out = Path(os.environ["OUT"])
    if out.resolve() != Path("/root/userfolder/ctx-compressor/grpo_v2/train"):
        raise ValueError("unexpected training output directory")
    started, started_utc = time.monotonic(), utc()
    with (out / "trainer.pid").open("x") as f:
        f.write(f"{os.getpid()}\n")
    write_json(out / "launch_args.json", {"argv": sys.argv, "pid": os.getpid(), "start_utc": started_utc,
               "cuda_visible_devices": "1", "cuda_device_order": "PCI_BUS_ID",
               "lambda": float(os.environ["CTX_GRPO_LAMBDA"])})
    stop = threading.Event()
    errors = []
    def monitor_card():
        try:
            with (out / "gpu1_monitor.csv").open("x", buffering=1) as f:
                f.write("unix_seconds,memory_used_mib,utilization_percent\n")
                while True:
                    sample = subprocess.check_output(
                        ["nvidia-smi", "-i", "1", "--query-gpu=memory.used,utilization.gpu",
                         "--format=csv,noheader,nounits"], text=True, timeout=8).strip()
                    used, util = [int(x) for x in sample.split(",")]
                    f.write(f"{time.time():.3f},{used},{util}\n")
                    if stop.wait(10):
                        return
        except Exception as exc:
            errors.append(exc)

    th = threading.Thread(target=monitor_card, name="gpu1-monitor", daemon=True)
    th.start()
    try:
        from swift.cli.utils import try_use_single_device_mode
        try_use_single_device_mode()
        import torch
        from transformers import TrainerCallback
        from swift.plugin import extra_callbacks
        from swift.llm import rlhf_main

        with (out / "optimizer_steps.jsonl").open("x", buffering=1) as steps:
            class FormalStepMonitor(TrainerCallback):
                def on_step_begin(self, args, state, control, **kwargs):
                    self.step_started = time.monotonic()
                    if errors:
                        raise RuntimeError("GPU 1 monitoring failed") from errors[0]

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs is None or "loss" not in logs:
                        return
                    check_logged_finite(logs, state.global_step)
                    row = {"step": state.global_step, "loss": float(logs["loss"]),
                           "grad_norm": float(logs["grad_norm"]), "utc": utc(),
                           "step_seconds": time.monotonic() - self.step_started,
                           "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                    steps.write(json.dumps(row, allow_nan=False) + "\n")

            extra_callbacks.append(FormalStepMonitor())
            rlhf_main()
        if errors:
            raise RuntimeError("GPU 1 monitoring failed") from errors[0]
    except BaseException as exc:
        # Inspect chained exceptions too; the original exception remains on stderr.
        chain = []
        while exc is not None and id(exc) not in {id(e) for e in chain}:
            chain.append(exc)
            exc = exc.__cause__ or exc.__context__
        detail = " | ".join(f"{type(e).__name__}: {e}" for e in chain)
        if any(isinstance(e, ImportError) for e in chain):
            print(f"NEED: existing GRPO environment is missing a dependency: {detail}", file=sys.stderr, flush=True)
        write_json(out / "training_failure.json", {"utc": utc(), "first_error": detail[:4000],
                   "oom": any(s in detail.lower() for s in ("outofmemory", "out of memory", "cudaerrormemoryallocation")),
                   "nonfinite": any(isinstance(e, FloatingPointError) for e in chain)})
        raise
    finally:
        stop.set()
        th.join()
    write_json(out / "training_runtime.json", {"status": "ok", "start_utc": started_utc, "end_utc": utc(),
               "wall_seconds": time.monotonic() - started, "pid": os.getpid(), "oom": False, "nonfinite": False})


if __name__ == "__main__":
    main()

