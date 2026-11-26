"""Run one interactive Task 11 GPU stage with the Task 10 admission rule."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import build_grpo_states as protocol
from scripts.build_formal_grpo_states import gpu_gate
from scripts.grpo_r2_start import BASE, FIXTURE, MERGED, RUN, START, compare_fixture, read
from scripts.rollout_model import file_record
from scripts.run_formal_grpo_pipeline import now
from scripts.sample_sft_subset import require
from scripts.sft_v3_wait_gpu import wait_for_gpu


def reference(stage, start):
    merged = stage == "ref_merged"
    output = RUN / stage
    output.mkdir()
    baseline = RUN / "ref_adapter/greedy_reference.json"
    options = argparse.Namespace(
        output_dir=output,
        prompts=FIXTURE,
        compare=baseline if merged else None,
        base_model=MERGED if merged else BASE,
        source_adapter=Path(start["path"]),
        adapter=None if merged else Path(start["path"]),
        gpu=1,
        model_label="SFT v3 checkpoint-3000",
    )
    engines = []
    create = protocol.create_engine

    def open_engine(args):
        result = create(args)
        engines.append(result[0])
        return result

    try:
        with patch.multiple(protocol, check_gpu=gpu_gate, create_engine=open_engine):
            protocol.reference(options)
    finally:
        for engine in engines:
            engine.llm_engine.engine_core.shutdown()
    artifact = output / "greedy_reference.json"
    proof = {"reference": file_record(artifact)}
    if merged:
        proof.update(compare_fixture(read(baseline), read(artifact), start["adapter_sha256"]))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("ref_adapter", "ref_merged", "smoke_train"))
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    require(args.attempt > 0 and Path(sys.prefix).resolve() == RUN.parent/"envs/grpo" and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
            and os.environ.get("HF_HUB_OFFLINE") == "1", "use the offline GRPO environment on physical GPU 1")
    start = read(START)
    require(start["strategy"] == "checkpoint-3000" and start["sft_v3_loss_count"] == 886 and
            start["adapter_sha256"] == read(MERGED/"merge_record.json")["adapter_sha256"], "merge/start mismatch")
    fixture = read(PROJECT/"eval/grpo_v2/input_checks.json")
    require(fixture["fixture_n"] == 30 and fixture["fixture_sha256"] == file_record(FIXTURE)["sha256"],
            "registered reference fixture changed")
    if args.stage == "smoke_train":
        compare_fixture(read(RUN/"ref_adapter/greedy_reference.json"), read(RUN/"ref_merged/greedy_reference.json"),
                        start["adapter_sha256"])
    else:
        require(not os.path.lexists(RUN/args.stage), "reference output exists")
        if args.stage == "ref_merged":
            require((RUN/"ref_adapter/greedy_reference.json").is_file(), "adapter reference missing")
    work = RUN / ("smoke_train" if args.stage == "smoke_train" else "phase_a") / f"gate_{args.stage}_{args.attempt}"
    work.mkdir(parents=True)
    status = work/"status.json"

    def publish(row):
        row = {"pid": os.getpid(), **row}
        with (work/"events.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        draft = work/"status.writing"
        with draft.open("x") as f:
            json.dump(row, f, indent=2)
            f.write("\n")
        os.replace(draft, status)
        print(json.dumps(row), flush=True)

    try:
        admitted = wait_for_gpu(args.stage, publish)
        publish({"status":"running", "stage":args.stage, "utc":now(), "gpu":admitted})
        if args.stage == "smoke_train":
            subprocess.run([sys.executable, "-B", "-u", str(PROJECT/"scripts/smoke_grpo_r2.py"),
                            "train", "--attempt", str(args.attempt)], check=True)
            result = {"training_exit_code":0}
        else:
            result = reference(args.stage, start)
        publish({"status":"complete", "stage":args.stage, "utc":now(), **result})
    except BaseException as exc:
        publish({"status":"failed", "stage":args.stage, "utc":now(), "error":repr(exc)})
        raise


if __name__ == "__main__":
    main()
