"""Seal Task 09 and verify historical manifests, including the original versions of shared documents."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.finalize_formal_grpo import DOCS, EXT, UUID, load, manifest, publish, read_sums, sha256

RUN = EXT / "hedge_v2"
RAW = EXT / "eval_rollouts_hedge_v2"
OUT = PROJECT / "eval/hedge_v2"
ORIGINAL = RUN / "history/task08_at_start"
TASK07 = PROJECT / "eval/grpo_prep/TASK07_SHA256SUMS"
TASK08 = PROJECT / "eval/grpo_v2/TASK08_SHA256SUMS"
TASK09 = OUT / "TASK09_SHA256SUMS"
MAIN = PROJECT / "SHA256SUMS"
RESOURCE = OUT / "final_resource.json"


def checked(ok, message):
    if not ok:
        raise ValueError(message)


def command(argv, cwd=PROJECT):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def historical():
    frozen = load(RUN / "protected_at_start.json")
    for path in (TASK07, TASK08):
        checked(sha256(path) == frozen[str(path.relative_to(PROJECT))]["sha256"], f"Historical manifest changed: {path}")
    prefixes = {}
    for name in DOCS:
        current = (PROJECT / name).read_bytes()
        prior = (ORIGINAL / name).read_bytes()
        item = frozen[name]
        checked(len(prior) == item["bytes"] and hashlib.sha256(prior).hexdigest() == item["sha256"]
                and current.startswith(prior) and current[len(prior):].startswith("\n## 任务09".encode()),
                f"Existing document content changed: {name}")
        prefixes[name] = dict(**item, appended_bytes=len(current) - len(prior))

    # sha256sum expects complete files. Only the three appended documents refer to historical copies.
    # All other relative entries resolve to live project files; all absolute external entries remain live.
    view = RUN / "tmp/task08_manifest_view"
    view.mkdir(exist_ok=True)
    records = read_sums(TASK08)
    for name in {Path(k).parts[0] for k in records if not Path(k).is_absolute()}:
        checked(name not in {"..", ".git"}, "Unexpected historical manifest entry")
        target = (ORIGINAL if name in DOCS else PROJECT) / name
        link = view / name
        if not os.path.lexists(link):
            link.symlink_to(target)
        checked(link.is_symlink() and link.resolve(strict=True) == target.resolve(strict=True), "Incorrect verification view")
    command(["sha256sum", "--quiet", "-c", str(TASK07)])
    command(["sha256sum", "--quiet", "-c", str(TASK08)], cwd=view)
    return {"through_task08": prefixes, "task07": {"entries": len(read_sums(TASK07)), "sha256": sha256(TASK07)},
            "task08": {"entries": len(records), "sha256": sha256(TASK08), "verifier_working_directory": str(view),
                       "historical_documents": list(DOCS), "previous_document_directory": str(ORIGINAL),
                       "other_entries_checked_against_live_files": len(records) - len(DOCS)}}


def external_files():
    for base in (RUN, RAW):
        for root, subdirs, files in os.walk(base):
            root = Path(root)
            if root == RUN:
                subdirs[:] = [n for n in subdirs if n not in ("tmp", "cache")]
            for name in subdirs + files:
                checked(not (root / name).is_symlink(), f"Symlink in run deliverables: {root / name}")
            for name in files:
                p = root / name
                checked(p.is_file(), f"Missing regular artifact {p}")
                yield p


def delivered_paths():
    paths = {PROJECT / name for name in DOCS}
    for directory in ("scripts", "tests"):
        paths.update(p for p in (PROJECT / directory).glob("*hedge*") if p.is_file())
    paths.update(p for p in OUT.rglob("*") if p.is_file() and p != TASK09)
    return paths | {RESOURCE}


def resources(paths, planned):
    status = load(RUN / "pipeline_status.json")
    stages = status["stages"]
    checked(status["status"] == "evaluation_complete" and status["stage_pid"] is None
            and [s["stage"] for s in stages] == ["train_scores", "H0", "H1", "H2"]
            and all(s["status"] == "complete" and s["exit_code"] == 0 and s["uses_gpu"] for s in stages),
            "Incomplete stages")
    all_samples = [json.loads(row) for row in (RUN / "pipeline_monitor.jsonl").read_bytes().splitlines()]
    admissions = [s["gpu_before"] for s in stages]
    for path in (OUT / "score_checks_initial.json", OUT / "score_production_checks_separate_paths.json",
                 OUT / "score_production_checks_shared_prefix_crossforward.json", OUT / "score_production_checks_separate_requests.json",
                 OUT / "score_production_checks.json"):
        admissions.append(load(path)["gpu_before"])
    checked(all(s["index"] == 1 and s["uuid"] == UUID for s in admissions + all_samples)
            and all(s["memory_used_mib"] < 2048 for s in admissions)
            and not any(s["oom"] for s in all_samples), "GPU identity, admission or OOM check failed")
    info = command(["nvidia-smi", "-i", "1", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"]).split(", ")
    pids = command(["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader"]).split()
    checked(info[:2] == ["1", UUID] and not ({str(s["pid"]) for s in stages} | {str(status["pipeline_pid"])}) & set(pids)
            and all(load(OUT / n / "run.json")["engine_shutdown"]["completed"] for n in ("nohedge","tau_star","always")),
            "Job GPU release check failed")
    listing = subprocess.check_output(["find", "/", "-xdev", "-name", ".git", "-prune", "-o", "-type", "f",
                                       "-newer", str(RUN / "start.marker"), "-print0"])
    recent = {Path(os.fsdecode(path)) for path in listing.split(b"\0") if path}
    own = paths | set(planned)
    unknown = [p for p in recent - own if p.is_relative_to(PROJECT) or
               any("hedge_v2" in part.lower() or "task09" in part.lower() for part in p.parts)]
    checked(not unknown, f"Unaccounted recent task or project files on overlay: {unknown}")
    sizes = {str(p.relative_to(PROJECT)): p.stat().st_size for p in own if p not in planned and p != RESOURCE}
    sizes.update({str(p.relative_to(PROJECT)): size for p, size in planned.items()})
    resource_key = str(RESOURCE.relative_to(PROJECT))
    start = (RUN / "df_start.txt").read_text().strip()
    end = command(["df", "-B1", "/"])
    data = {"utc": datetime.now(timezone.utc).isoformat(), "test_cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu_index": 1, "gpu_uuid": UUID, "gpu_before_jobs": admissions, "pipeline_monitor_samples": len(all_samples),
            "pipeline_status": status["status"], "stages_exit_codes": {s["stage"]: s["exit_code"] for s in stages},
            "pipeline_finished_utc": status["ended_utc"], "pipeline_final_sample": status["last_monitor"],
            "observed_gpu1_used_mib": int(info[2]), "observed_gpu1_utilization_percent": int(info[3]),
            "observed_compute_pids": pids, "task_gpu_released": True,
            "root_overlay_measurement": "task deliverables and shared documents, cross-checked with find / -xdev -newer start.marker",
            "other_recent_file_count": len(recent - own), "raw_df_start": start, "raw_df_end": end,
            "raw_df_start_used_bytes": int(start.splitlines()[-1].split()[2]),
            "raw_df_end_used_bytes": int(end.splitlines()[-1].split()[2]), "limit_bytes": 300_000_000}
    size = 0
    while True:
        sizes[resource_key] = size
        data.update(task_overlay_bytes=sum(sizes.values()), overlay_files=dict(sorted(sizes.items())), file_count=len(sizes))
        checked(data["task_overlay_bytes"] <= data["limit_bytes"], "Task-owned overlay files exceed 300 MB")
        content = (json.dumps(data, indent=2) + "\n").encode()
        if len(content) == size:
            return content
        size = len(content)


def finalize():
    for name in ("hedge_v2_report.md", "hedge_v2_results.json", "cost_vs_loss.svg", "final_audit.json",
                 "final_tests.json", "final_tests.xml", "final_history.json"):
        checked((OUT / name).is_file(), "Missing " + name)
    checked(not TASK09.exists() and not RESOURCE.exists(), "Task 09 is already sealed")
    original = ORIGINAL / MAIN.name
    checked(sha256(MAIN) == sha256(original) == load(RUN / "protected_at_start.json")[MAIN.name]["sha256"],
            "Shared manifest changed since launch")
    relative = delivered_paths()
    sums = {str(p.relative_to(PROJECT)): sha256(p) for p in relative if p != RESOURCE}
    big = {str(p): sha256(p) for p in external_files()}
    for p in big:
        checked(Path(p).is_relative_to(EXT), f"Unexpected external location: {p}")
    prior = read_sums(original)
    checked(set(prior) & sums.keys() == set(DOCS), "Unexpected overlap with earlier deliverables")
    key = str(RESOURCE.relative_to(PROJECT))
    task_key = str(TASK09.relative_to(PROJECT))
    sums[key] = "0"*64
    main = {**prior, **sums, task_key: "0"*64}
    planned = {MAIN: len(manifest(main)), TASK09: len(manifest({**sums, **big}))}
    publish(RESOURCE, resources(relative, planned))
    sums[key] = sha256(RESOURCE)
    publish(TASK09, manifest({**sums, **big}))
    main.update(sums)
    main[task_key] = sha256(TASK09)
    publish(MAIN, manifest(main))
    checked(all((PROJECT / k).stat().st_size == n for k,n in load(RESOURCE)["overlay_files"].items()),
            "Overlay file size changed while finishing manifests")
    return {"relative":len(sums), "external":len(big), "all_project_entries":len(main),
            "manifest_sha256":sha256(TASK09), "task_overlay_bytes":load(RESOURCE)["task_overlay_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("history","seal","verify"))
    args = parser.parse_args()
    checked(PROJECT == Path("/root/workplace/ctx-compressor") and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Run in the formal checkout with CUDA hidden")
    if args.command == "history":
        report = historical()
        with (OUT / "final_history.json").open("x") as stream:
            json.dump(report,stream,indent=2)
            stream.write("\n")
        print(json.dumps(report))
    elif args.command == "seal":
        print(json.dumps(finalize()))
    else:
        history = historical()
        command(["sha256sum","--quiet","-c",str(TASK09)])
        command(["sha256sum","--quiet","-c",str(MAIN)])
        legacy = json.loads(command([sys.executable,"-B","scripts/checksums_task06.py","check"]))
        print(json.dumps({"checksums_task06":legacy, "task07":history["task07"]["entries"],
                          "task08":history["task08"]["entries"], "task09":len(read_sums(TASK09)),
                          "project":len(read_sums(MAIN))}))


if __name__ == "__main__":
    main()

