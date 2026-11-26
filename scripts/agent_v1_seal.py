"""Sign and verify Task 20 while keeping historical document versions verifiable."""
import argparse
from pathlib import Path
import os
import subprocess

from scripts import agent_v1_main as main, agent_v1_report as report, agent_v1_smoke as smoke
from scripts import test_v1_seal as previous
from src.reader_heldout_inputs import checksum, need, read, write_new
from src.reranker_v1 import record, require_new, verify

PROJECT = smoke.PROJECT
INDEX = PROJECT / "SHA256SUMS"
MANIFEST = smoke.SMALL / "TASK20_SHA256SUMS"
CLOSE = main.ROOT / "closeout"
SNAPSHOT = CLOSE / "before_docs"
DOCS = previous.DOCS


def check_historical_docs(directory):
    for name in DOCS:
        p = directory / name
        need(p.is_file() and (PROJECT / name).read_bytes().startswith(p.read_bytes()), f"{name} history prefix changed")


def check_history():
    old = previous.check_history(previous.BEFORE)
    need({f"{i:02d}" for i in range(6, 19)} <= old.keys(), "Incomplete earlier historical verification")
    check_historical_docs(SNAPSHOT)
    expected = previous.load_sums(previous.MANIFEST)
    expected_sha = previous.load_sums(INDEX)[str(previous.MANIFEST.relative_to(PROJECT))]
    verify(previous.MANIFEST, expected_sha)
    text = "".join(f"{digest}  {SNAPSHOT/name if name in DOCS else name}\n" for name, digest in expected.items())
    subprocess.run(["sha256sum", "--quiet", "-c", "-"], input=text, text=True, cwd=PROJECT, check=True)
    old["19"] = {"entries": len(expected), "manifest": record(previous.MANIFEST)}
    lock = previous.data.verify_locked()
    old["locked_inputs"] = lock["files"]
    outside = smoke.data.WORK / "monitor_path_recovery/historical_phase_checks.json"
    phases = read(outside)["phase_manifests"]
    for name, receipt in phases.items():
        verify(PROJECT / name, receipt["sha256"])
        for filename, digest in previous.load_sums(PROJECT / name).items():
            binding = receipt["history_bindings"].get(filename, {})
            if "signed_prefix_of" in binding:
                with Path(binding["signed_prefix_of"]).open("rb") as source:
                    data = source.read(binding["bytes"])
                from hashlib import sha256
                need(sha256(data).hexdigest() == binding["sha256"] == digest, f"Signed prefix changed: {filename}")
            else:
                verify(Path(binding.get("signed_path", PROJECT / filename)), digest)
    old["phase_manifests"] = {name: receipt["entries"] for name, receipt in phases.items()}
    return old


def snapshot():
    check = read(previous.SYNC / "verified.json")
    need(check["global_index"] == record(INDEX) and check["manifest"] == record(previous.MANIFEST),
         "Task 19 index changed before Task 20 delivery")
    require_new(CLOSE / "history_before.json")
    CLOSE.mkdir(exist_ok=True)
    SNAPSHOT.mkdir(exist_ok=True)
    for name in (*DOCS, INDEX.name):
        previous.exclusive(SNAPSHOT / name, (PROJECT / name).read_bytes())
    d = check_history()
    write_new(CLOSE / "history_before.json", d)
    print("Task 06–19 and phase manifests verified prior to Task 20 append", flush=True)


def attempt_files(attempt):
    need(attempt.is_dir() and attempt.resolve() == attempt, f"Invalid raw attempt directory: {attempt}")
    excluded = previous.EXCLUDE | {"closeout", "cpu_telemetry_check"}
    for path in sorted(attempt.iterdir()):
        if path.name in excluded:
            continue
        if path.is_dir():
            yield from previous.regular(path)
        else:
            need(path.is_file() and path.resolve() == path, f"Invalid raw attempt artifact: {path}")
            yield path


def external_artifacts():
    work = smoke.data.WORK
    inputs = read(smoke.SMALL / "try3/try3_inputs.json")
    refs = list(inputs["old_artifacts"].values())
    for attempt in (work / "try3", work / "main", main.PILOT_ROOT, main.ROOT):
        refs.extend(record(path) for path in attempt_files(attempt))
    for block in read(main.data.RECEIPT)["datasets"].values():
        for ref in (block["main"]["file"], block["smoke"]["file"], block["corpus"], *block["bm25_index"]):
            path = Path(ref["path"])
            # MuSiQue corpus and index are already covered by Task 19.
            if path.is_relative_to(work):
                refs.append(ref)
    cards = read(smoke.SMALL / "model_cards.json")
    for section in ("llmlingua2", "llmlingua_wheel", "open_provence"):
        refs.extend(cards[section]["files"])
    refs.extend((cards["llmlingua_wheel"]["wheel"], cards["runtime_dependencies"]["cl100k_base"]["cache_entry"]))
    for path in (main.CPU_TEST, smoke.SMALL / "main/cpu_tests.json", smoke.SMALL / "try3/monitor_scope_cpu_tests.json"):
        details = read(path)
        refs.extend((details["xml"], details["log"]))
    refs.extend(record(path) for path in SNAPSHOT.iterdir())
    refs.append(record(CLOSE / "history_before.json"))
    refs.append(record(CLOSE / "closeout_cpu.py"))
    recovery_tests = smoke.SMALL / "main_try2/cpu_recovery_tests.json"
    if recovery_tests.is_file():
        refs.append(record(CLOSE / "cpu_status.json"))
        refs.append(record(CLOSE / "audit_cpu_recovery.py"))
        final_tests = recovery_tests.with_name("cpu_seal_tests.json")
        refs.append(record(CLOSE / "agent_v1_seal_before_fixture_pruning.py"))
        for path in (recovery_tests, final_tests):
            receipt = read(path)
            need(receipt["status"] == "passed", f"Incomplete CPU suite receipt: {path}")
            refs.extend((receipt["xml"], receipt["log"]))
    refs.append(record(work / "monitor_path_recovery/run_main_signed_audit.py"))
    refs.append(record(work / "monitor_path_recovery/run_main_try2_signed_audit.py"))
    refs.append(record(work / "monitor_path_recovery/run_main_pilot_signed_audit.py"))
    refs.extend(record(p) for p in (work / "monitor_path_recovery/before_main_rpc_fix").iterdir())
    refs.append(record(work / "monitor_path_recovery/historical_phase_checks.json"))

    result = {}
    for ref in refs:
        p = Path(ref["path"])
        if p.is_relative_to(smoke.SMALL):
            verify(p, ref["sha256"])
            continue
        need(p.is_relative_to(main.EXTERNAL) and p.resolve() == p and p.is_file() and
             not {".git", "test_dev_v1"} & set(p.parts), f"Protected or redirected path in Task 20 manifest: {p}")
        need(result.get(str(p), ref["sha256"]) == ref["sha256"], f"Conflicting SHA256: {p}")
        result[str(p)] = ref["sha256"]
    return result


def merge():
    require_new(MANIFEST, CLOSE / "mirror_files.txt")
    result = read(report.RESULT)
    need(result["status"] == "complete" and result["episodes"] == 2400, "Task 20 report missing")
    for d in (*result["delivery"].values(), result["run_config"], result["terminal_status"], result["model_cards"]):
        report.checked(d)
    need((CLOSE / "history_before.json").is_file(), "Missing Task 20 signing baseline")
    check_historical_docs(SNAPSHOT)
    need(all((PROJECT / n).read_bytes() == (SNAPSHOT / n).read_bytes() for n in DOCS), "Shared documents changed before append")
    need(INDEX.read_bytes() == (SNAPSHOT / INDEX.name).read_bytes(), "Shared index changed before Task 20 append")
    older = previous.load_sums(INDEX)
    external = external_artifacts()
    subprocess.run(["sha256sum", "--quiet", "-c", "-"], input=previous.render(external), cwd=PROJECT, check=True)
    owned = set(previous.regular(smoke.SMALL))
    owned.update((PROJECT / "src").glob("agent_v1*.py"))
    owned.update((PROJECT / "scripts").glob("agent_v1_*.py"))
    owned.update((PROJECT / "tests").glob("test_agent_v1*.py"))
    owned.add(PROJECT / "tests/fixtures/agent_v1_rr5.json")
    project = {str(path.relative_to(PROJECT)): checksum(path) for path in owned}
    need(not (older.keys() & project.keys()), "Task 20 artifacts overlap earlier task signatures")
    d = result["results"]["pooled"]["paired"]["r4 reselect − RR-τ reselect"]
    ending = result["ended_utc"]
    recovery_note = ""
    handoff_opening = f"正式运行于 UTC {ending} 完成。"
    if "cpu_archive_recovery" in result:
        recovery_note = "全部推理落盘后触发根分区规则 (a)，原失败回执保留。按授权仅以现有归档做 CPU 收尾。"
        handoff_opening = f"正式推理在 UTC {ending} 的控制器退出前完成，后续经授权以现有归档做 CPU 收尾。"
    suffix = {
        "README.md": ("\n\n## 任务20 · 真实 Agent 测试\n\n"
                      "本地 Qwen3-8B 在 MuSiQue、HotpotQA 和 2Wiki 上完成 n=300 个多题任务。"
                      f"合并 r4 重选 − RR-τ 重选 ΔF1 为 {100*d['delta_f1']:+.3f} 个百分点（n={d['n']} 题对）。"
                      "[任务20报告](eval/agent_v1/agent_v1_report.md)\n"),
        "NOTES.md": ("\n\n## 任务20 · 正式运行与冒烟记录\n\n"
                     "正式运行沿用 try3 冻结的 Thought: 预填、任务划分和解码。"
                     "任务10日志恢复、监控目录修复、冒烟峰值和规则 (a)–(d) 均已在报告存档。"
                     "open-provence 为第三方复现，缺少 NLTK 数据，本轮未跑。"
                     + recovery_note + "[配置与签名来源](eval/agent_v1/run_config.json)\n"),
        "HANDOFF.md": ("\n\n## 任务20 · 真实 Agent 交接\n\n"
                       + handoff_opening
                       + "5880 的 agent_v1/main_try2 留存全部正式逐轮对话和监控。"
                       "[结构化结果](eval/agent_v1/agent_v1_results.json) · "
                       "[交付清单](eval/agent_v1/TASK20_SHA256SUMS)\n"),
    }
    for name in DOCS:
        with (PROJECT / name).open("ab") as f:
            f.write(suffix[name].encode())
            f.flush()
            os.fsync(f.fileno())
        project[name] = checksum(PROJECT / name)
    check_historical_docs(SNAPSHOT)
    with MANIFEST.open("xb") as f:
        f.write(previous.render(external | project))
    manifest_rel = str(MANIFEST.relative_to(PROJECT))
    project[manifest_rel] = checksum(MANIFEST)
    tmp = INDEX.with_name(".SHA256SUMS.task20.next")
    with tmp.open("xb") as f:
        f.write(previous.render(older | project))
    tmp.chmod(INDEX.stat().st_mode & 0o777)
    os.replace(tmp, INDEX)
    with (CLOSE / "mirror_files.txt").open("x", encoding="utf-8") as f:
        for name in sorted(set(project) | {INDEX.name}):
            print(name, file=f)
    print(f"Task 20 signed {len(external) + len(project) - 1} entries", flush=True)


def confirm():
    old = previous.load_sums(SNAPSHOT / INDEX.name)
    index = previous.load_sums(INDEX)
    need(older_unchanged(old, index), "A historical index entry changed")
    current = previous.load_sums(MANIFEST)
    rel = str(MANIFEST.relative_to(PROJECT))
    verify(MANIFEST, index[rel])
    need(all(index[name] == digest for name, digest in current.items() if not name.startswith("/")),
         "Task 20 global manifest entries differ")
    subprocess.run(["sha256sum", "--quiet", "-c", str(MANIFEST)], cwd=PROJECT, check=True)
    prior = read(CLOSE / "history_before.json")
    after = check_history()
    for h in (prior, after):
        h["06"].pop("other_manifest_entries_preserved", None)
    need(prior == after, "An earlier task changed during Task 20 signing")
    write_new(CLOSE / "verified.json", {"status": "passed", "task20_entries": len(current), "history": after,
              "manifest": record(MANIFEST), "index": record(INDEX), "append_only": list(DOCS)})
    print("Task 06–20, frozen data, and phase manifests verified", flush=True)


def older_unchanged(old, updated):
    return old.keys() <= updated.keys() and all(updated[key] == value for key, value in old.items() if key not in DOCS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("snapshot", "merge", "confirm"))
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and Path.cwd() == PROJECT, "Seal from 5880 with CUDA hidden")
    {"snapshot": snapshot, "merge": merge, "confirm": confirm}[parser.parse_args().phase]()

