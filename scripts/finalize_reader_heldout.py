#!/usr/bin/env python3
"""Finalize the frozen reader evaluation, preserving historical bytes and hashes."""
from collections import defaultdict
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.qa_metrics import score_answer
from src.reader import parse_answer
from src.reader_heldout_inputs import checksum, need, read, rows, sums, verify

WORK = Path("/root/userfolder/ctx-compressor/reader_heldout")
SMALL = PROJECT / "eval/reader_heldout"
OLD = WORK / "before_docs"
DOCS = ("README.md", "NOTES.md", "HANDOFF.md")
R2 = "grpo_r2:checkpoint-125"
V3 = "sft_v3:checkpoint-3000"
H1 = "h1:tau_star"
MANIFEST = SMALL / "TASK12_SHA256SUMS"


def within(value, expected):
    return math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-11)


def write_once(path, data):
    """Resume after a partial closeout only if already-written bytes match exactly."""
    if path.exists():
        need(path.is_file() and not path.is_symlink() and path.read_bytes() == data,
             f"refusing to replace existing file: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


def validate_report(result, cfg, preflight):
    need(result["n_tasks"] == 249 and result["n_predictions"] == cfg["predictions"]
         == preflight["requests"]["n"], "reader population mismatch")
    need(result["smoke"] == cfg["smoke"] and result["smoke"]["n"] == 40,
         "Task 03 smoke agreement missing")
    mask = result["no_compress"]
    need(mask == preflight["no_compress"], "no-compression identity receipt changed")
    wanted = {R2,V3,H1,"sft_v2:checkpoint-2000","grpo_r1:checkpoint-500",
              "oracle","bm25","latest","no_compress","no_context"}
    need(result["strategies"].keys() == wanted, "wrong strategies in reader report")
    for name, report in result["strategies"].items():
        shared = name == "no_context" or (name == "no_compress" and mask["all_identical"])
        need(len(report["cells"]) == (1 if shared else 6) and
             all(s["n"] == 996 for s in report["cells"].values()) and
             report["pooled"]["n"] == (996 if shared else 5976), f"incomplete cells: {name}")
    for filename, h in result["input_sha256"].items():
        p = Path(filename).resolve()
        need((p.is_relative_to(WORK) or p.is_relative_to(SMALL)) and p.is_file(), str(p))
        verify(p,h)
    report_text = (SMALL / "reader_heldout_report.md").read_text()
    for name in ("budget_f1.svg","cost_f1.svg"):
        p = SMALL / name
        root = ET.parse(p).getroot()
        need(root.tag.endswith("}svg") and any(x.tag.endswith("}path") for x in root.iter())
             and p.stat().st_size > 1000 and name in report_text, f"incomplete SVG figure or link: {name}")
    need("F1" in report_text and "EM" in report_text and "Qwen3" in report_text, "incomplete Markdown")


def verify_predictions(result, preflight):
    reference_fields = ("config","strategy","mode","tier","task","question_index","example_id",
                        "question_hops","prompt_tokens","evidence_complete","trajectory_lost","prompt_sha1")
    grouped = defaultdict(list)
    pairs = {name:{} for name in (R2,V3,H1)}
    for src, row in zip(rows(WORK / "input_preflight/requests.jsonl"), rows(WORK / "predictions.jsonl"), strict=True):
        need(all(src[k] == row[k] for k in reference_fields), "prediction/preflight alignment mismatch")
        need(not row["input_truncated"] and len(src["token_ids"]) == row["prompt_tokens"]
             and row["output_tokens"] == len(row["output_token_ids"]) <= 48
             and row["prompt_tokens"]+48 <= 40960, "input or output length violation")
        pred = parse_answer(row["raw_output"])
        score = score_answer(pred,src["answer"],src["answer_aliases"])
        need(pred == row["pred"] and score == {k:row[k] for k in ("em","f1")}, "raw scoring mismatch")
        name = src["strategy"]
        grouped[name].append((row["em"],row["f1"]))
        if name in pairs:
            key = (src["mode"], src["tier"], src["task"], src["question_index"], src["example_id"])
            need(key not in pairs[name], "duplicate paired prediction")
            pairs[name][key] = (row["em"],row["f1"])
    need(sum(len(values) for values in grouped.values()) == result["n_predictions"],
         "scored count mismatch")
    for name, values in grouped.items():
        summary = result["strategies"][name]["pooled"]
        need(summary["n"] == len(values) and within(summary["em"],math.fsum(a for a,b in values)/len(values))
             and within(summary["f1"],math.fsum(b for a,b in values)/len(values)), f"pooled mismatch {name}")
    need(len(result["comparisons"]) == 2 and {c["right"] for c in result["comparisons"]} == {V3,H1},
         "missing registered comparison")
    for comp in result["comparisons"]:
        left, right = comp["left"],comp["right"]
        need(left == R2 and pairs[left].keys() == pairs[right].keys() and len(pairs[left]) == comp["n"] == 5976,
             "pairing mismatch")
        counts = dict(left_higher=0,equal=0,right_higher=0)
        matrix = [[0,0],[0,0]]
        for k,(aem,af1) in pairs[left].items():
            bem, bf1 = pairs[right][k]
            counts["left_higher" if af1>bf1 else "right_higher" if bf1>af1 else "equal"] += 1
            matrix[1-int(aem)][1-int(bem)] += 1
        need(comp["f1_counts"] == counts and comp["em_matrix_correct_then_incorrect"] == matrix,
             f"paired counts mismatch: {right}")
        for metric in ("em","f1"):
            a, b = (result["strategies"][s]["pooled"][metric] for s in (left,right))
            need(within(comp[metric+"_difference_pp"],100*(a-b)), f"wrong paired {metric} delta")
    mismatch, count = 0, 0
    for a,b in zip(rows(WORK / "input_preflight/smoke.jsonl"),rows(WORK / "smoke_predictions.jsonl"),strict=True):
        need(a["prompt_sha1"] == b["prompt_sha1"] and a["config"] == b["config"]
             and a["task"] == b["task"] and a["question_index"] == b["question_index"], "smoke identity mismatch")
        mismatch += a["expected_raw_output"] != b["raw_output"]
        count += 1
    need(count == 40 and mismatch == result["smoke"]["raw_output_mismatches"], "smoke output count mismatch")


def validate_receipts():
    pre = read(WORK / "input_preflight/preflight.json")
    cfg = read(SMALL / "run_config.json")
    tests = read(SMALL / "cpu_tests.json")
    history = read(SMALL / "history_post.json")
    disk = read(SMALL / "disk_final.json")
    result = read(SMALL / "reader_heldout_results.json")
    need(pre["status"] == "passed" and pre["input_truncations"] == 0, "input gate incomplete")
    status = read(WORK / "status.json")
    need(status["status"] == "complete" and status["stage"]=="finished", "reader worker did not finish")
    for stage in ("inputs","gate","smoke","inference"):
        need(read(WORK / ("receipt_"+stage+".json"))["stage"] == stage, f"missing stage: {stage}")
    proof = read(WORK / "receipt_inference.json")
    need(cfg["status"] == "complete" and cfg["engine_shutdown"] and cfg["peak_gpu1_mib"] > 0,
         "inference exit or memory receipt missing")
    verify(SMALL / "run_config.json", proof["run_config_sha256"])
    verify(WORK / "predictions.jsonl",proof["predictions_sha256"])
    need(proof["n"] == cfg["predictions"] and cfg["predictions_sha256"] == proof["predictions_sha256"],
         "inconsistent generation receipt")
    verify(WORK / "input_preflight/preflight.json",read(WORK / "receipt_inputs.json")["sha256"])
    for key in ("requests","smoke"):
        verify(WORK / f"input_preflight/{key}.jsonl",pre[key]["sha256"])
    need(tests["status"] == "passed" and tests["passed"]==tests["tests"]
         and tests["locked_dev_open_attempts"]==tests["gpu_cli_invocations"]==tests["skipped"]==0,
         "CPU suite incomplete")
    verify(Path(tests["xml"]),tests["xml_sha256"])
    old_history = read(WORK/"history_pre.json")
    need(all(k in history for k in ("06","07","08","09","10","11","global"))
         and history == old_history, "historical pre/post proof differs")
    need(disk["status"]=="complete" and 0 < disk["project_new_bytes_at_report"] <= 300000000,
         "task-owned project disk budget exceeded")
    validate_report(result,cfg,pre)
    verify_predictions(result,pre)
    return result, cfg, tests, disk, history

def append_sections(result, cfg, tests, disk):
    def scores(s):
        z = result["strategies"][s]["pooled"]
        return f"F1 {100*z['f1']:.3f}%、EM {100*z['em']:.3f}%（n={z['n']:,}）"
    deltas = {}
    for c in result["comparisons"]:
        deltas[c["right"]] = (f"F1 {c['f1_difference_pp']:+.3f} 个百分点、"
                              f"EM {c['em_difference_pp']:+.3f} 个百分点（n={c['n']:,} 个配对）")
    r = result["smoke"]
    count = result["n_predictions"]
    section = "\n\n## 任务12　Qwen3-8B 留出集阅读评测\n\n"
    link = "[报告](eval/reader_heldout/reader_heldout_report.md) · [结构化结果](eval/reader_heldout/reader_heldout_results.json)"
    readme = (f"在 249 个留出任务的 996 道不同问题上，Qwen3-8B 共给出 {count:,} 条回答。"
              f"GRPO r2 checkpoint-125 的合并结果为 {scores(R2)}，"
              f"SFT v3 checkpoint-3000 为 {scores(V3)}，H1 为 {scores(H1)}。\n\n"
              f"r2 与 SFT v3 的配对差为 {deltas[V3]}，与 H1 的配对差为 {deltas[H1]}。"
              f"报告逐格列出紧、中、松预算下的读者分数及轨迹缓存成本。{link}\n")
    notes = (f"沿用任务03的阅读模板、生成参数和 EM/F1 解析。历史小样的原始回答有 "
             f"{r['raw_output_mismatches']}/40 条不同。判定证据齐全时按问题核对所有支持段，"
             f"判定整条轨迹丢失时核对任务中全部支持段。两种对照的实际评测次数见结构化结果，"
             f"其中闭卷对照为 n=996。完整输入长度预检覆盖 {count:,}/{count:,} 条，截断为 0/{count:,}。"
             f"GPU 1 的峰值显存为 {cfg['peak_gpu1_mib']:,} MiB（n={cfg['n_resource_samples']:,} 个监控样本）。"
             f"{link}\n")
    handoff = (f"读者评测的主比较为 r2 对 SFT v3 {deltas[V3]}、r2 对 H1 {deltas[H1]}。"
               f"CPU 测试通过 {tests['passed']}/{tests['tests']} 项，历史清单回执在 "
               f"[history_post.json](eval/reader_heldout/history_post.json)。"
               f"逐题预测保存在 5880 的 /root/userfolder/ctx-compressor/reader_heldout/predictions.jsonl，"
               f"对应 [TASK12_SHA256SUMS](eval/reader_heldout/TASK12_SHA256SUMS)。"
               f"任务13 G 通过这份清单校验 r2 的 {result['strategies'][R2]['pooled']['n']:,} 条观测。{link}\n")
    return {name:(section+txt).encode("utf-8")
            for name,txt in zip(DOCS,(readme,notes,handoff),strict=True)}


def safe_append(name, suffix):
    base = (OLD / name).read_bytes()
    with (PROJECT / name).open("r+b") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        current = stream.read()
        if current == base:
            stream.write(suffix)
            stream.flush()
            os.fsync(stream.fileno())
        else:
            need(current == base + suffix,f"document changed independently: {name}")
    need((PROJECT / name).read_bytes() == base + suffix, f"document prefix or append changed: {name}")


def inventory(tests):
    files = set()
    for directory in ("scripts","src","tests"):
        files.update(p for p in (PROJECT/directory).glob("*reader_heldout*") if p.is_file())
    files.update(p for p in (PROJECT/"specs").glob("TASK_12_*.md") if p.is_file())
    files.update(p for p in SMALL.rglob("*") if p.is_file() and p != MANIFEST)
    files.update(PROJECT/name for name in DOCS)
    for name in ("before.json","full_cpu.py","history_checks.py","history_pre.json","no_compress_length_probe.json",
                 "disk_samples.jsonl","reader.pid","status.json","events.jsonl","resources.jsonl",
                 "receipt_inputs.json","receipt_gate.json","receipt_smoke.json","receipt_inference.json",
                 "smoke_predictions.jsonl","predictions.jsonl"):
        files.add(WORK/name)
    for name in ("monitor_disk.py","disk_monitor.pid","disk_monitor.log","stop_disk_monitor","reader.log","launcher.log",
                 "after_reader.py","finish_disk.py","check_f1_pairs.py","report_driver.log",
                 "report_driver.pid","cpu_report_finished.json"):
        if (WORK/name).is_file():
            files.add(WORK/name)
    for name in ("requests.jsonl","smoke.jsonl","preflight.json"):
        files.add(WORK/"input_preflight"/name)
    files.update(OLD/name for name in (*DOCS,"SHA256SUMS"))
    # Preserve each completed task12 CPU suite's own XML and log when its receipt is present.
    for f in SMALL.glob("*tests*.json"):
        rec=read(f)
        if rec.get("status")=="passed":
            for key in ("xml","log"):
                if key in rec:
                    path=Path(rec[key]).resolve()
                    need(path.is_relative_to(WORK),f"test diagnostic outside task scope: {path}")
                    files.add(path)
    files.update(Path(tests[k]) for k in ("xml","log"))
    for p in files:
        need(p.resolve() == p and p.is_file() and "test_dev_v1" not in p.parts and ".git" not in p.parts
             and (p.is_relative_to(WORK) or p.is_relative_to(PROJECT)), f"unsafe or missing artifact: {p}")
    return sorted(files)


def rendered(entries):
    return "".join(f"{digest}  {name}\n" for name,digest in sorted(entries.items())).encode("utf-8")


def project_name(p):
    return str(p.relative_to(PROJECT)) if p.is_relative_to(PROJECT) else str(p)


def merged_checksums(project_hashes):
    original = OLD/"SHA256SUMS"
    historical = sums(original)
    for name in DOCS:
        verify(OLD/name,historical[name])
    for name,h in project_hashes.items():
        if name in historical and name not in DOCS:
            need(historical[name]==h,f"attempt to change historical checksum: {name}")
    proposed = dict(historical)
    proposed.update(project_hashes)
    data = rendered(proposed)
    global_path = PROJECT/"SHA256SUMS"
    current = global_path.read_bytes()
    need(current == original.read_bytes() or current == data,"unrelated change to global SHA256SUMS")
    # The history helper substitutes sealed digests for locked paths before verification.
    spec = importlib.util.spec_from_file_location("task12_history_scope",WORK/"history_checks.py")
    history = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(history)
    proof = history.verify(proposed,{})
    need(proof["total"]==len(proposed) and proof["sealed_locked"]==6,"incomplete global checksum verification")
    if current != data:
        stage = PROJECT / "SHA256SUMS.task12.pending"
        write_once(stage,data)
        os.replace(stage,global_path)
    need(global_path.read_bytes()==data,"published global manifest differs")
    return proof


def main():
    need(Path.cwd().resolve()==PROJECT==Path("/root/workplace/ctx-compressor"),"run on 5880 project checkout")
    need(os.environ.get("CUDA_VISIBLE_DEVICES","")=="","finalize on CPU only")
    result,cfg,tests,disk,history = validate_receipts()
    if sys.argv[1:] == ["--check-only"]:
        print(json.dumps({"predictions":result["n_predictions"],"validated":"reports, curves, runs, tests and history"}))
        return
    need(not sys.argv[1:],"usage: finalize_reader_heldout.py [--check-only]")
    artifacts = inventory(tests)
    suffixes = append_sections(result,cfg,tests,disk)
    # Validate every document in advance before publishing any one of the three append sections.
    for name,suffix in suffixes.items():
        current = (PROJECT/name).read_bytes()
        base = (OLD/name).read_bytes()
        need(current in (base,base+suffix),f"existing document has independent changes: {name}")
    for name,suffix in suffixes.items():
        safe_append(name,suffix)
    digests = {project_name(p):checksum(p) for p in artifacts}
    write_once(MANIFEST,rendered(digests))
    subprocess.run(["sha256sum","--quiet","-c",str(MANIFEST)],cwd=PROJECT,check=True)
    relative = {n:h for n,h in digests.items() if not Path(n).is_absolute()}
    relative[project_name(MANIFEST)] = checksum(MANIFEST)
    proof = merged_checksums(relative)
    names = sorted(set(relative)|{"SHA256SUMS"})
    need(all(not name.startswith("/") and ".git" not in Path(name).parts and ".." not in Path(name).parts
             for name in names),"unsafe mirror input")
    write_once(WORK/"mirror_files.txt",("\n".join(names)+"\n").encode())
    receipt = {"status":"complete","predictions":result["n_predictions"],"task12_entries":len(digests),
               "task12_sha256":checksum(MANIFEST),"global_sha256":checksum(PROJECT/"SHA256SUMS"),
               "global":proof,"mirrored_project_files":len(names),"mirror_list":str(WORK/"mirror_files.txt")}
    write_once(WORK/"publication_receipt.json",(json.dumps(receipt,indent=2)+"\n").encode())
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == "__main__":
    main()
