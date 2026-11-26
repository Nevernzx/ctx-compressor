"""Seal Task 19, preserving both the lock and older append-only documents."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

from scripts import test_v1_data as data
from src.reader_heldout_inputs import checksum, read, write_new
from src.reranker_v1 import need, record, verify

E,W,OUT=data.E,data.WORK,data.SMALL
SYNC=W/"closeout"
BEFORE=SYNC/"before_docs"
INDEX=PROJECT/"SHA256SUMS"
MANIFEST=OUT/"TASK19_SHA256SUMS"
DOCS=("README.md","NOTES.md","HANDOFF.md")
EXCLUDE={".git","test_dev_v1","runtime","runtime_cache","tmp",".tmp","__pycache__",".pytest_cache",".cache"}


def load_sums(path):
    items=[line.split("  ",1) for line in path.read_text().splitlines()]
    value={name:digest for digest,name in items}
    need(len(items)==len(value),f"duplicate checksum path: {path}")
    return value


def render(items):
    return "".join(f"{val}  {name}\n" for name,val in sorted(items.items())).encode()


def exclusive(path, data):
    if path.exists():
        need(path.read_bytes()==data,f"existing sealed file differs: {path}")
    else:
        with path.open("xb") as target:
            target.write(data)


def check_history(document_snapshot):
    sys.path.insert(0,str(E/"reader_heldout_v2/close16"))
    from history16 import verify_all
    import history_checks as history
    sign=load_sums(INDEX)
    result=verify_all()
    for n,folder,docs in ((16,"reader_heldout_v2",E/"close17_18_20261125/before_docs"),
                         (17,"reranker_v1",document_snapshot),(18,"grpo_r4",document_snapshot)):
        p=PROJECT/f"eval/{folder}/TASK{n}_SHA256SUMS"
        name=str(p.relative_to(PROJECT))
        verify(p,sign[name])
        result[str(n)]=history.verify(load_sums(p),{f:str(docs/f) for f in DOCS})
    return result


def freeze_docs():
    SYNC.mkdir()
    BEFORE.mkdir()
    for name in (*DOCS,"SHA256SUMS"):
        exclusive(BEFORE/name,(PROJECT/name).read_bytes())
    result=check_history(BEFORE)
    need(all(str(i).zfill(2) in result for i in range(6,19)),"older task check missing")
    write_new(SYNC/"history_before.json",result)
    print("verified task 06–18 before Task 19 delivery",flush=True)


def regular(base):
    need(base.is_dir() and base.resolve()==base, f"wrong inventory root: {base}")
    for directory, dirs, files in os.walk(base):
        dirs[:]=[d for d in dirs if d not in EXCLUDE]
        for name in files:
            p=Path(directory)/name
            need(p.resolve()==p and p.is_file() and not set(p.parts)&EXCLUDE,f"invalid delivery file: {p}")
            yield p


def archive():
    status=read(W/"status.json")
    need(status["status"]=="evaluation_complete" and status["lock_released"],"GPU part incomplete")
    history=read(SYNC/"history_before.json")
    need(all(str(i).zfill(2) in history for i in range(6,19)),"prior verification missing")
    result=read(OUT/"test_v1_results.json")
    need(result["status"]=="complete" and result["task"]==19 and result["lock_final"]["path"]==str(OUT/"lock_final.json"),
         "test report missing")
    need(read(OUT/"lock_final.json")["files"]==read(OUT/"lock_preflight.json")["files"],"locked test bytes changed")
    for item in (*result["delivery"].values(),result["lock_final"]):
        verify(Path(item["path"]),item["sha256"])
    old=load_sums(BEFORE/"SHA256SUMS")
    original={name:(BEFORE/name).read_bytes() for name in DOCS}
    need(all(sha256(v).hexdigest()==old[n] for n,v in original.items()),"shared document signing basis differs")

    p=result["paired"][0]
    r4=result["reader"]["r4"]["pooled"]
    fin=result["completion"]["finished_beijing"]
    overlap=result["gold_exposure"]["unique_test_gold"]
    suffix={
        "README.md":("\n\n## 任务19 · 锁定测试集\n\n锁定测试集已于任务19第一次使用。"
                     f"r4 F1 为 {r4['f1']*100:.3f}%（n={r4['n']:,} 题次），"
                     f"r4 − RR-τ 的 ΔF1 为 {p['f1_difference_pp']:+.3f} 个百分点（n={p['n']:,} 对）。"
                     "[报告](eval/test_v1/test_v1_report.md)\n"),
        "NOTES.md":("\n\n## 任务19 · 锁定检查\n\n锁定测试集已于任务19第一次使用。"
                    "动手前及交付前的五个文件 SHA256 一致。"
                    f"锁定集去重 gold 在全部 train 与 held-out 轨迹中当过 gold 的段数为 {overlap['count']:,}/{overlap['n']:,}。"
                    "[交付前锁定核验](eval/test_v1/lock_final.json)\n"),
        "HANDOFF.md":("\n\n## 任务19 · 测试交接\n\n锁定测试集已于任务19第一次使用。"
                      f"GPU 流水线在北京时间 {fin} 完成。5880 的 /root/userfolder/ctx-compressor/test_v1/ "
                      "留存轨迹、分数、请求、预测和阶段回执。"
                      "[结构化结果](eval/test_v1/test_v1_results.json) · [签名清单](eval/test_v1/TASK19_SHA256SUMS)\n"),
    }
    docs={name:v+suffix[name].encode() for name,v in original.items()}
    own=set(regular(OUT))-{MANIFEST}
    own.update((PROJECT/"scripts").glob("test_v1_*.py"))
    own.add(PROJECT/"tests/test_task19_locked_protocol.py")
    external={p for f in ("data","g0","scores","rollouts","reader","stages") for p in regular(W/f)}
    external|={W/f for f in ("pipeline.pid","status.json","events.jsonl","launcher.log","reader_metrics.json","phase_p_cpu.py")}
    tests=read(OUT/"cpu_tests.json")
    external|={Path(tests[k]) for k in ("xml","log")}
    external|=set(BEFORE.iterdir())|{SYNC/"history_before.json"}
    if "reader_recovery" in result["completion"]:
        external|=set(regular(W/"g3_df_opening_recovery"))
        external|={SYNC/"g3_reader_before.py",SYNC/"g3_recovery_cpu.py"}
        tests=read(OUT/"g3_recovery_cpu_tests.json")
        external|={Path(tests[k]) for k in ("xml","log")}
    all_paths=sorted(own|external)
    need(all(p.resolve()==p and p.is_file() and not set(p.parts)&EXCLUDE for p in all_paths),"protected path in task manifest")
    with ThreadPoolExecutor(max_workers=3) as pool:
        hashes=dict(zip(all_paths,pool.map(checksum,all_paths),strict=True))
    project_hashes={str(p.relative_to(PROJECT)):hashes[p] for p in own}
    project_hashes.update({n:sha256(v).hexdigest() for n,v in docs.items()})
    addition={str(p):hashes[p] for p in external}|project_hashes
    own_bytes=sum(p.stat().st_size for p in own)+sum(map(len,docs.values()))+len(render(addition))
    need(own_bytes<300_000_000,"task-owned overlay exceeds 300 MB")
    manifest_raw=render(addition)
    project_hashes[str(MANIFEST.relative_to(PROJECT))]=sha256(manifest_raw).hexdigest()
    need(all(n in DOCS or n not in old or old[n]==v for n,v in project_hashes.items()),"historical entry would change")
    merged=render(old|project_hashes)
    base=(BEFORE/"SHA256SUMS").read_bytes()
    need(INDEX.read_bytes() in (base,merged),"shared index changed")
    for name,value in docs.items():
        p=PROJECT/name
        with p.open("r+b") as fh:
            fcntl.flock(fh.fileno(),fcntl.LOCK_EX)
            old_bytes=fh.read()
            if old_bytes!=value:
                need(old_bytes==original[name],f"shared prefix changed: {name}")
                fh.write(value[len(old_bytes):])
                fh.flush()
                os.fsync(fh.fileno())
        need(p.read_bytes()==value and p.read_bytes().startswith(original[name]),f"append-only failure: {name}")
    exclusive(MANIFEST,manifest_raw)
    if INDEX.read_bytes()!=merged:
        temp=INDEX.with_name(".SHA256SUMS.task19.next")
        exclusive(temp,merged)
        temp.chmod(INDEX.stat().st_mode & 0o777)
        os.replace(temp,INDEX)
    exclusive(SYNC/"mirror_files.txt","".join(n+"\n" for n in sorted({"SHA256SUMS"}|project_hashes.keys())).encode())
    print("Task19 signed",len(addition),"entries, owned overlay",own_bytes,flush=True)


def final_verify():
    before=read(SYNC/"history_before.json")
    after=check_history(BEFORE)
    for n,prior in before.items():
        now=after[n]
        if n=="06":
            need(all(v==now[k] for k,v in prior.items() if k!="other_manifest_entries_preserved"),"Task 06 receipt differs")
        else:
            need(prior==now,f"Task {n} historical signature verification differs")
    index=load_sums(INDEX)
    older=load_sums(BEFORE/"SHA256SUMS")
    need(older.keys() <= index.keys() and all(index[k]==v for k,v in older.items() if k not in DOCS),
         "historical index was rewritten")
    for n in DOCS:
        old=(BEFORE/n).read_bytes()
        new=(PROJECT/n).read_bytes()
        need(new.startswith(old) and b"19" in new[len(old):] and old!=new,"document prefix differs")
        verify(PROJECT/n,index[n])
    verify(MANIFEST,index[str(MANIFEST.relative_to(PROJECT))])
    current=load_sums(MANIFEST)
    need(all(index[n]==v for n,v in current.items() if not n.startswith("/")),"wrong global task entries")
    subprocess.run(["sha256sum","--quiet","-c",str(MANIFEST)],cwd=PROJECT,check=True)
    write_new(SYNC/"verified.json",{"status":"passed","historical":after,"entries":len(current),
        "global_index":record(INDEX),"manifest":record(MANIFEST),"append_only":list(DOCS)})
    print("Task19 and all Task06–18 signatures verified",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("history","seal","verify"))
    args=p.parse_args()
    need(Path.cwd()==PROJECT and os.environ.get("CUDA_VISIBLE_DEVICES")=="","run checksum stage on CPU")
    {"history":freeze_docs,"seal":archive,"verify":final_verify}[args.action]()
