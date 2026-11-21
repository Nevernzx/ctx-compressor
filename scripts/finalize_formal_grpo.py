"""Finalize Task 08 hash manifests and account for its overlay files, with no GPU workload."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.merge_sft_lora import sha256

OUT = PROJECT / "eval/grpo_v2"
EXT = Path("/root/userfolder/ctx-compressor")
RUN = EXT / "grpo_v2"
RAW = EXT / "eval_rollouts_grpo_v2"
DOCS = ("README.md", "NOTES.md", "HANDOFF.md")
TASK = OUT / "TASK08_SHA256SUMS"
MAIN = PROJECT / "SHA256SUMS"
RESOURCE = OUT / "final_resource.json"
UUID = "GPU-4318dc90-4126-038a-9402-1323533db94f"


def load(path):
    return json.loads(path.read_bytes())


def lines(path):
    with path.open() as stream:
        return [json.loads(row) for row in stream]


def read_sums(path):
    records = {}
    for line in path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in records:
            raise ValueError(f"Duplicate manifest entry: {name}")
        records[name] = digest
    return records


def manifest(records):
    return "".join(f"{records[k]}  {k}\n" for k in sorted(records)).encode()


def publish(path, data):
    temp = path.with_name(path.name + ".finalizing")
    with temp.open("xb") as stream:
        stream.write(data)
    temp.replace(path)


def external_files():
    # All run products, including optimizer files and the preserved first attempt.
    # Scratch directories contain Python caches, sockets, and transient test fixtures.
    for root in (RUN, RAW):
        for parent, dirs, names in os.walk(root, followlinks=False):
            parent = Path(parent)
            if parent == RUN:
                dirs[:] = [d for d in dirs if d not in {"cache", "tmp"}]
            if any((parent / d).is_symlink() for d in dirs):
                raise ValueError(f"Unexpected symlink below {parent}")
            for name in names:
                path = parent / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Unexpected external artifact {path}")
                yield path


def resource_snapshot(relatives, manifest_sizes):
    marker = RUN / "start.marker"
    found = subprocess.check_output(["find", "/", "-xdev", "-name", ".git", "-prune", "-o",
                                     "-type", "f", "-newer", str(marker), "-print0"])
    changed = {Path(os.fsdecode(s)) for s in found.split(b"\0") if s}
    own = {PROJECT / name for name in relatives} | {MAIN, TASK, RESOURCE}
    sizes = {p: p.stat().st_size for p in own if p not in (*manifest_sizes, RESOURCE)}
    sizes.update(manifest_sizes)
    outside = changed - own
    unaccounted = [p for p in outside if p.is_relative_to(PROJECT)
                   or any("task08" in s.lower() or "grpo_v2" in s.lower() for s in p.parts)]
    if unaccounted:
        raise ValueError(f"Review recent overlay files before assigning them: {sorted(unaccounted)}")

    before = (RUN / "df_start.txt").read_text()
    after = subprocess.check_output(["df", "-B1", "/"], text=True)
    stat = load(RUN / "pipeline_status.json")
    monitors = lines(RUN / "pipeline_monitor.jsonl.failed_C_topk") + lines(RUN / "pipeline_monitor.jsonl")
    if (stat["status"] != "evaluation_complete" or any(m["uuid"] != UUID or m["index"] != 1 for m in monitors)
            or any(s["status"] != "complete" or s["exit_code"] != 0 for s in stat["stages"])):
        raise ValueError("Incomplete stages or incorrect physical GPU")
    stages = [s for s in stat["stages"] if s["uses_gpu"]]
    failed = load(RUN / "pipeline_status.json.failed_C_topk")["stages"][-1]
    if any(s["gpu_before"]["uuid"] != UUID or s["gpu_before"]["memory_used_mib"] >= 2048 for s in [*stages, failed]):
        raise ValueError("Invalid GPU admission sample")
    a_samples = (RUN / "logs/A_gpu1.csv").read_text().splitlines()
    a_fields = [[v.strip() for v in row.split(",")] for row in a_samples]
    if (len(a_fields) != 4 or any(f[1:3] != ["1", UUID] for f in a_fields)
            or any(int(a_fields[i][3].split()[0]) >= 2048 for i in (0, 2))):
        raise ValueError("Invalid GPU samples for A")
    query = ["nvidia-smi", "-i", "1"]
    gpu = subprocess.check_output(query + ["--query-gpu=index,uuid,memory.used,utilization.gpu",
                                          "--format=csv,noheader"], text=True).strip()
    compute = subprocess.check_output(query + ["--query-compute-apps=pid", "--format=csv,noheader"], text=True).split()
    pids = {s["pid"] for s in stat["stages"]} | {stat["pipeline_pid"], stat["resume"]["prior_pid"], failed["pid"]}
    if UUID not in gpu or any(str(pid) in compute for pid in pids):
        raise ValueError("Task GPU process remains active or physical GPU mismatch")
    record = {"utc": datetime.now(timezone.utc).isoformat(), "marker": str(marker),
              "measurement": "find / -xdev -type f -newer start.marker, plus all delivered overlay file sizes",
              "raw_df_start": before.strip(), "raw_df_end": after.strip(),
              "raw_df_start_used_bytes": int(before.splitlines()[-1].split()[2]),
              "raw_df_end_used_bytes": int(after.splitlines()[-1].split()[2]),
              "other_recent_files_outside_scope": len(outside),
              "gpu": {"physical_index": 1, "uuid": UUID, "A_samples": a_samples, "pipeline_monitor_samples": len(monitors),
                      "all_monitor_samples_on_gpu1": True, "successful_gpu_stages": len(stages),
                      "failed_C_attempt_archived": True, "at_closeout": gpu,
                      "observed_compute_pids": compute, "run_pids_still_computing": [], "released": True},
              "limit_bytes": 300_000_000}
    size = 0
    while True:
        sizes[RESOURCE] = size
        record["overlay_files"] = {str(p.relative_to(PROJECT)): sizes[p] for p in sorted(sizes)}
        record["file_count"] = len(sizes)
        record["task_overlay_bytes"] = sum(sizes.values())
        record["within_limit"] = record["task_overlay_bytes"] <= record["limit_bytes"]
        encoded = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode()
        if size == len(encoded):
            break
        size = len(encoded)
    if not record["within_limit"]:
        raise ValueError("Task-owned overlay size exceeds 300 MB")
    return encoded


def main():
    if PROJECT != Path("/root/workplace/ctx-compressor") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run on 5880 with CUDA_VISIBLE_DEVICES empty")
    for required in ("grpo_v2_report.md", "grpo_v2_results.json", "training_curve.svg", "training_curve.csv",
                     "final_history.json", "final_audit.json", "final_tests.json"):
        if not (OUT / required).is_file():
            raise FileNotFoundError(required)
    previous = read_sums(TASK)
    previous_main = read_sums(MAIN)
    rel = {k for k in previous if not Path(k).is_absolute()}
    for directory in ("scripts", "tests"):
        rel.update(p.relative_to(PROJECT).as_posix() for p in (PROJECT / directory).glob("*formal*grpo*") if p.is_file())
    rel.update(p.relative_to(PROJECT).as_posix() for p in OUT.rglob("*") if p.is_file() and p != TASK)
    rel.update(DOCS)
    resource_name = RESOURCE.relative_to(PROJECT).as_posix()
    rel.add(resource_name)
    digests = {name: sha256(PROJECT / name) for name in rel if name != resource_name}
    big = {str(p): sha256(p) for p in external_files()}
    missing = {k for k in previous if Path(k).is_absolute()} - big.keys()
    if missing:
        raise ValueError(f"Missing previous external artifacts: {missing}")
    placeholder = "0" * 64
    digests[resource_name] = placeholder
    task_name = TASK.relative_to(PROJECT).as_posix()
    root_digests = {**previous_main, **digests, task_name: placeholder}
    planned = {MAIN: len(manifest(root_digests)), TASK: len(manifest({**digests, **big}))}
    publish(RESOURCE, resource_snapshot(rel, planned))
    digests[resource_name] = sha256(RESOURCE)
    publish(TASK, manifest({**big, **digests}))
    root_digests.update(digests)
    root_digests[task_name] = sha256(TASK)
    publish(MAIN, manifest(root_digests))
    for name, size in load(RESOURCE)["overlay_files"].items():
        if (PROJECT / name).stat().st_size != size:
            raise ValueError(f"Resource size mismatch: {name}")
    print(json.dumps({"relative": len(digests), "external": len(big), "total_main": len(root_digests),
                      "resource_bytes": load(RESOURCE)["task_overlay_bytes"]}))


if __name__ == "__main__":
    main()
