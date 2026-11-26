"""One-shot Task 18 detached workflow, holding the common GPU 1 lock through G."""
import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import traceback
from unittest.mock import patch

from scripts import grpo_r4_common as c
from scripts import grpo_r4_eval as evaluation
from scripts import grpo_r4_prepare as prep
from scripts import grpo_r4_reader as qa
from scripts import grpo_r4_training as training
from scripts import run_formal_grpo_pipeline as controller
from scripts.run_sft_v3_pipeline import ResumedPipeline

STATUS = c.RUN / "pipeline_status.json"
LOGS = c.RUN / "logs"
PID = c.RUN / "pipeline.pid"
CPU = c.PROJECT / ".conda/envs/ctxc/bin/python"
CPU_TESTS = c.SMALL / "cpu_tests.json"
P_RECEIPT = c.SMALL / "phase_p.json"
RUNTIME = c.RUN / "runtime"
TOOLCHAIN = c.ROOT / "toolchains/gcc13_20261118/envs/gcc/bin"


def worker_environment(label, environment):
    cache = RUNTIME / label
    cache.mkdir(parents=True)
    qa.SHORT.mkdir(exist_ok=True)
    c.require(qa.SHORT.resolve() == qa.SHORT and len(os.fsencode(qa.SHORT)) + 43 < 100,
              "Task 18 worker IPC directory is too long or redirected")
    locations = {
        "XDG_CACHE_HOME": "xdg", "XDG_CONFIG_HOME": "config", "XDG_STATE_HOME": "state", "XDG_DATA_HOME": "local",
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub", "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
        "HUGGINGFACE_ASSETS_CACHE": "huggingface/assets", "HF_DATASETS_CACHE": "huggingface/datasets",
        "TRANSFORMERS_CACHE": "huggingface/transformers", "MODELSCOPE_CACHE": "modelscope", "SWIFT_HOME": "swift",
        "SWIFT_CACHE_DIR": "swift", "VLLM_CACHE_ROOT": "vllm", "VLLM_CONFIG_ROOT": "vllm_config",
        "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch_extensions", "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "FLASHINFER_WORKSPACE_BASE": "flashinfer",
        "CUDA_CACHE_PATH": "cuda", "NUMBA_CACHE_DIR": "numba", "CUPY_CACHE_DIR": "cupy", "CCACHE_DIR": "ccache",
        "PIP_CACHE_DIR": "pip", "PYTHONPYCACHEPREFIX": "pycache", "MPLCONFIGDIR": "matplotlib",
        "WANDB_DIR": "wandb", "WANDB_CACHE_DIR": "wandb_cache", "WANDB_CONFIG_DIR": "wandb_config",
    }
    env = {key: str(cache / relative) for key, relative in locations.items()}
    for path in set(env.values()):
        Path(path).mkdir(parents=True, exist_ok=True)
    env.update(
        {key: str(qa.SHORT) for key in ("TMPDIR", "TMP", "TEMP", "RAY_TMPDIR", "VLLM_RPC_BASE_PATH")},
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", VLLM_NO_USAGE_STATS="1", WANDB_MODE="disabled",
        DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn", PYTHONPATH=str(c.PROJECT),
        CC=str(TOOLCHAIN / "x86_64-conda-linux-gnu-gcc"), CXX=str(TOOLCHAIN / "x86_64-conda-linux-gnu-g++"),
        PATH=os.pathsep.join((str(environment.parent), str(TOOLCHAIN), os.environ["PATH"])),
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
        NUMEXPR_NUM_THREADS="4", RAYON_NUM_THREADS="4", NUMBA_NUM_THREADS="4", MAX_JOBS="4",
        NO_PROXY="*", no_proxy="*", PIP_CONFIG_FILE="/dev/null", PIP_NO_INPUT="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
    )
    return env


def check_p():
    tests = c.read(CPU_TESTS)
    c.require(tests["status"] == "passed" and tests["tests"] >= 820 and
              tests["errors"] == tests["failures"] == tests["skipped"] == 0
              and tests["matched_task16_tests"] == 820 and not tests["cpu_guard_events"], "Full CPU suite incomplete")
    c.require(c.digest(Path(tests["xml"])) == tests["xml_sha256"], "CPU test log changed")
    probe, data = c.read(prep.RECEIPT_C), c.read(prep.RECEIPT_D)
    c.require(probe["status"] == data["status"] == "complete" and probe["n_samples"] == 179952
              and probe["parity"]["v2_reward_mismatches"] == probe["parity"]["v3_delta_mismatches"] == 0
              and probe["n_informative"] == data["n_informative"] >= c.TARGET
              and data["n"] == c.TARGET and data["seed"] == c.DATA_SEED and data["plugin_greedy_checked"] == c.TARGET
              and data["user_text_mismatches"] == data["plugin_greedy_mismatches"] == data["heldout_rows"] == 0
              and data["source_groups"] == probe["group_file"], "Phase P C/D proof incomplete")
    c.require(c.file_record(c.DATA) == data["output"] and c.file_record(c.SMOKE_DATA) == data["smoke_data"],
              "Formal or smoke dataset changed")
    reference = c.read(evaluation.BASELINES)
    c.require(reference["status"] == "complete" and set(reference["strategies"]) == {c.R2_POLICY, c.V3_POLICY}
              and all(v["heldout"]["pooled"]["heldout_reward_v3"]["n_steps"] == 13956
                      for v in reference["strategies"].values()), "Incomplete offline held-out baselines")
    for env, packages in (("grpo", ("ms-swift", "torch", "vllm")), ("reader", ("torch", "vllm", "transformers"))):
        exe = c.ROOT / "envs" / env / "bin/python"
        c.require(exe.is_file() and os.access(exe, os.X_OK), f"NEED: missing prepared environment {exe}")
        code = "from importlib.metadata import version; " + "; ".join(f"print(version({p!r}))" for p in packages)
        output = subprocess.check_output([str(exe), "-B", "-c", code], text=True, env={
            **os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
        }).splitlines()
        c.require(len(output) == len(packages) and output[packages.index("vllm")] == "0.11.0"
                  and (env != "grpo" or output[0] == "3.12.6"), f"NEED: existing {env} package versions differ")
    assets = c.verify_model()
    c.require(assets == probe["merged_model_files"], "Merged start model differs from Phase P")
    return {
        "status": "passed", "tests": c.file_record(CPU_TESTS), "C": c.file_record(prep.RECEIPT_C),
        "D": c.file_record(prep.RECEIPT_D), "historical_reward_v3": c.file_record(evaluation.BASELINES),
        "model_files": assets, "n_informative": data["n_informative"], "n_prompts": data["n"],
    }


@contextmanager
def exclusive_gpu():
    c.require(c.LOCK.resolve() == c.LOCK, "Unexpected GPU lock path")
    fd = os.open(c.LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        c.require(stat.S_ISREG(os.fstat(fd).st_mode), "GPU lock is not a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


class Workflow(ResumedPipeline):
    def __init__(self):
        super().__init__(c.RUN)

    def step(self, name, module, args, environment, on_gpu, verifier):
        exe = CPU if environment == "cpu" else c.ROOT / "envs" / environment / "bin/python"
        env = worker_environment(name, exe)
        self.stage(name, [str(exe), "-B", "-u", "-m", module, *args], on_gpu, env)
        receipt = verifier()
        self.active.update(receipt=c.file_record(receipt))
        self.save()
        return receipt

    def run(self):
        module = "scripts.grpo_r4_training"
        smoke_runtime = training.SMOKE_TRAIN / "training_runtime.json"
        self.step("smoke", module, ["smoke"], "grpo", True, lambda: runtime_check(smoke_runtime))
        self.step("smoke_check", module, ["check-smoke"], "cpu", False,
                  lambda: training_check(training.SMOKE_RECEIPT, 2))
        self.step("E", module, ["train"], "grpo", True, lambda: runtime_check(c.TRAIN / "training_runtime.json"))
        self.step("E_check", module, ["check"], "cpu", False,
                  lambda: training_check(training.TRAIN_RECEIPT, 500))
        module = "scripts.grpo_r4_eval"
        for n in c.CHECKPOINT_ORDER:
            self.step(f"F_checkpoint-{n}", module, ["eval", "--step", str(n)], "reader", True,
                      lambda: eval_check(n))
        self.step("F_select", module, ["select"], "cpu", False,
                  lambda: simple_receipt(evaluation.SELECTED, lambda v: v["n_candidates"] == 4))
        module = "scripts.grpo_r4_reader"
        self.step("G_inputs", module, ["prepare"], "reader", False,
                  lambda: simple_receipt(qa.PREFLIGHT, lambda v: v["requests"]["n"] == qa.N_PREDICTIONS))
        self.active = None
        # The signed reader runner independently requires this same gate receipt.
        self.step("G", module, ["run"], "reader", True,
                  lambda: simple_receipt(qa.READER_CONFIG, lambda v: v["effective_disable_cascade_attn"] is True
                                         and v["predictions"] == qa.N_PREDICTIONS and v["engine_shutdown"]))
        self.step("G_check", module, ["score"], "cpu", False,
                  lambda: simple_receipt(qa.RESULT, lambda v: v["status"] == "complete" and len(v["comparisons"]) == 2))
        self.record.update(stage="finished", status="evaluation_complete",
                           selected_checkpoint=c.read(evaluation.SELECTED)["strategy"])

    def publish_wait(self, event):
        super().publish_wait(event)
        if event["stage"] == "G" and event["consecutive_free"] == event["required_free"]:
            c.write_json_exclusive(c.READER / "receipt_gate.json", {
                "stage": "gate", "gpu": event["gpu"], "admission": event,
            })


def simple_receipt(path, condition):
    c.require(condition(c.read(path)), f"Incomplete receipt: {path}")
    return path


def runtime_check(path):
    return simple_receipt(path, lambda p: p["status"] == "ok" and not p["nonfinite"] and not p["oom"])


def training_check(path, steps):
    return simple_receipt(path, lambda p: p["optimizer_steps"] == steps and p["status"] == "ok"
                          and p["reward_recheck"]["mismatches"] == p["reward_recheck"]["advantage_mismatches"] == 0)


def eval_check(n):
    evaluation.check_eval(n)
    return c.SMALL / f"checkpoint-{n}" / "run.json"


def clear_proxy():
    for key in list(os.environ):
        if key.lower() in ("http_proxy", "https_proxy", "all_proxy"):
            del os.environ[key]


def detached():
    c.require(os.getpid() == os.getsid(0), "Expected detached session")
    c.expect_new(PID, LOGS, STATUS, RUNTIME, c.TRAIN, c.RAW, c.READER)
    with PID.open("x") as out:
        print(os.getpid(), file=out)
    LOGS.mkdir()
    def stop(signum, _frame):
        raise InterruptedError(f"Workflow PID {os.getpid()} received {signum}")
    signal.signal(signal.SIGTERM, stop)
    workflow = Workflow()
    lock_state = {"path": str(c.LOCK), "held": False, "order": ["18", "17"]}
    workflow.record.update(lock=lock_state)
    with patch.multiple(controller, RUN=c.RUN, LOGS=LOGS, STATUS=STATUS, TRAIN=c.TRAIN):
        try:
            workflow.save()
            workflow.record["phase_p"] = c.file_record(P_RECEIPT)
            proof = check_p()
            c.require(proof == c.read(P_RECEIPT), "Frozen Phase P receipts differ")
            lock_state["waiting_since_utc"] = controller.now()
            workflow.record.update(status="waiting_lock", stage="gpu_lock")
            workflow.save()
            with exclusive_gpu():
                lock_state.update(held=True, acquired_utc=controller.now())
                workflow.record.update(status="running", stage="gpu_admission")
                workflow.save()
                workflow.run()
        except BaseException as error:
            workflow.record.update(status="failed", first_error=str(error), traceback=traceback.format_exc())
            if workflow.active is not None:
                workflow.active.update(status="failed", ended_utc=controller.now())
            raise
        finally:
            if lock_state["held"]:
                lock_state.update(held=False, released_utc=controller.now())
            workflow.record.update(ended_utc=controller.now(), stage_pid=None)
            workflow.save()


def launch():
    clear_proxy()
    c.expect_new(c.RUN / "launcher.log", P_RECEIPT, STATUS, PID, LOGS, RUNTIME, c.TRAIN, c.RAW, c.READER)
    c.write_json_exclusive(P_RECEIPT, check_p())
    with (c.RUN / "launcher.log").open("xb") as log:
        proc = subprocess.Popen(
            [str(CPU), "-B", "-u", "-m", "scripts.run_grpo_r4_pipeline", "--worker"],
            cwd=c.PROJECT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
        )
    print(f"Task 18 detached PID {proc.pid}. Status {STATUS}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    opt = cli.add_mutually_exclusive_group(required=True)
    opt.add_argument("--detach", action="store_true")
    opt.add_argument("--worker", action="store_true")
    args = cli.parse_args()
    c.require(Path.cwd() == c.PROJECT == Path("/root/workplace/ctx-compressor"), "Run from the formal remote project")
    c.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Controller starts without CUDA visibility")
    launch() if args.detach else detached()

