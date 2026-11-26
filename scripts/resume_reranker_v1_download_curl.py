"""Continue the pinned Task 17 snapshot with curl and preserve progress across retries."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlencode

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.download_reranker_v1 import BASE, CACHE, CARD, DESTINATION, MODEL, REVISION, ROOT
from scripts.resume_reranker_v1_download_ranges import STAGING, digest, safe_partial, write_json

WORK = ROOT / "reranker_v1/download"
ARCHIVE = WORK.with_name("download_attempt3")
OLD_PID = 3338049
CURL = "/usr/bin/curl"
RETRY_SECONDS = 60
NO_PROGRESS_SECONDS = 3 * 3600
CURL_OPTIONS = (
    "-q", "--noproxy", "*", "--proto", "=https", "--proto-redir", "=https",
    "-sS", "--fail", "-L", "-C", "-", "--retry", "1000", "--retry-delay", "60",
    "--retry-all-errors", "--speed-limit", "1000", "--speed-time", "120",
)


def now():
    return datetime.now(timezone.utc)


def publish(status, **fields):
    value = {"utc": now().isoformat(), "pid": os.getpid(), "status": status, "model": MODEL,
             "revision": REVISION, "destination": str(DESTINATION), "transport": "curl", **fields}
    tmp = WORK / "curl_status.next"
    write_json(tmp, value)
    os.replace(tmp, WORK / "status.json")
    with (WORK / "curl_events.jsonl").open("a", encoding="utf-8") as stream:
        print(json.dumps(value, ensure_ascii=False), file=stream)
    print(value["utc"], status, value.get("file", ""), value.get("downloaded_bytes"), flush=True)


def snapshot():
    saved = json.loads((ARCHIVE / "pinned_files.json").read_bytes())
    files = saved["files"]
    if saved["revision"] != REVISION or len(files) != 15:
        raise ValueError("Pinned ModelScope metadata is incomplete")
    names = {}
    for row in files:
        name = row["Path"]
        path = PurePosixPath(name)
        if (path.is_absolute() or str(path) != name or not path.parts or
                set(path.parts) & {".", "..", ".git"} or name in names or
                not re.fullmatch(r"[a-f0-9]{64}", row["Sha256"]) or
                type(row["Size"]) is not int or row["Size"] <= 0):
            raise ValueError("Invalid pinned file identity")
        names[name] = row
    if names["model.safetensors"]["Size"] != 1191588280:
        raise ValueError("Weight size differs from the pinned ModelScope metadata")
    return sorted(files, key=lambda row: (row["Size"], row["Path"]))


def size(path):
    if path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Unsafe partial file: {path}")
    return path.stat().st_size


def extend(dst, src, maximum):
    """Append a stable byte prefix; retain the destination if curl rewinds its file."""
    start = size(dst)
    end = min(size(src), maximum)
    if end > start:
        with src.open("rb", buffering=0) as reader, dst.open("ab", buffering=0) as writer:
            reader.seek(start)
            left = end - start
            while left:
                block = reader.read(min(left, 1 << 20))
                if not block:
                    break
                if writer.write(block) != len(block):
                    raise OSError("Short write while saving the partial")
                left -= len(block)
    return size(dst)


class Stalled(TimeoutError):
    pass


class Progress:
    def __init__(self, files):
        previous = json.loads((ARCHIVE / "status.json").read_bytes())
        age = (now() - datetime.fromisoformat(previous["utc"])).total_seconds()
        self.latest = time.monotonic() - max(0, age + previous.get("seconds_without_progress", 0))
        self.high = previous.get("downloaded_bytes", 0)
        self.verified = 0
        self.total = sum(r["Size"] for r in files)
        self.logged = 0

    def remaining(self):
        left = NO_PROGRESS_SECONDS - (time.monotonic() - self.latest)
        if left <= 0:
            raise Stalled("3 consecutive hours without byte progress")
        return left

    def report(self, row, received, status, *, force=False, **extra):
        t = time.monotonic()
        total = self.verified + received
        if total > self.high:
            self.high = total
            self.latest = t
        self.remaining()
        if force or t - self.logged >= 3:
            publish(status, file=row["Path"], file_bytes=received, file_total_bytes=row["Size"],
                    downloaded_bytes=total, total_bytes=self.total, verified_bytes=self.verified,
                    seconds_without_progress=round(t - self.latest, 2), **extra)
            self.logged = t

    def retry(self):
        time.sleep(min(RETRY_SECONDS, self.remaining()))
        self.remaining()


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def transfer(row, keep, incoming, progress, log):
    expected = row["Size"]
    # curl can rewind its output on an internal retry. Keep bytes in a separate
    # append-only checkpoint and always start a new curl from that checkpoint.
    if not incoming.exists():
        with incoming.open("xb"):
            pass
    extend(incoming, keep, expected)
    offset = size(incoming)
    if offset != size(keep):
        extend(keep, incoming, expected)
        offset = size(keep)
    if offset >= expected:
        return 0, False
    args_offset = offset
    url = BASE + "/repo?" + urlencode({"Revision": REVISION, "FilePath": row["Path"]})
    args = [CURL, *CURL_OPTIONS, "--output", str(incoming), url]
    child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    rolled_back = False
    try:
        progress.report(row, offset, "downloading", force=True, curl_pid=child.pid,
                        resumed_from_bytes=offset, partial=str(keep))
        while child.poll() is None:
            offset = extend(keep, incoming, expected)
            progress.report(row, offset, "downloading", curl_pid=child.pid,
                            resumed_from_bytes=args_offset, partial=str(keep))
            if size(incoming) < offset:
                rolled_back = True
                break
            time.sleep(.2)
    finally:
        stop_child(child)
    extend(keep, incoming, expected)
    return child.returncode, rolled_back


def verify_file(path, row):
    return size(path) == row["Size"] and digest(path) == row["Sha256"]


def fetch(row, progress, log):
    target = DESTINATION / row["Path"]
    if target.resolve() != target:
        raise ValueError("Model destination was redirected")
    while not target.exists():
        keep = safe_partial(row["Path"])
        incoming = keep.with_name(keep.stem + ".curl.partial")
        count = size(keep)
        if count != row["Size"]:
            code, rewind = transfer(row, keep, incoming, progress, log)
            count = size(keep)
            if count < row["Size"]:
                progress.report(row, count, "retry_wait", force=True, curl_exit_code=code,
                                curl_output_rewound=rewind, next_retry_seconds=60, partial=str(keep))
                progress.retry()
                continue
        if not verify_file(keep, row):
            write_json(keep.with_suffix(".rejected.json"), {"file": str(keep), "size": count,
                       "sha256": digest(keep), "reason": "Pinned ModelScope size or SHA-256 mismatch"})
            progress.report(row, min(count, row["Size"]), "retry_wait", force=True,
                            reason="pinned size or SHA-256 mismatch, previous partial preserved", next_retry_seconds=60)
            progress.retry()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(keep, target)
    if not verify_file(target, row):
        raise ValueError(f"Installed file SHA-256/size differs: {row['Path']}")
    progress.verified += row["Size"]
    progress.report(row, 0, "file_verified", force=True)
    return {"file": row["Path"], "bytes": row["Size"], "sha256": row["Sha256"],
            "file_revision": row["Revision"]}


def worker():
    if os.getpid() != os.getsid(0):
        raise ValueError("Expected detached curl supervisor")
    pending = WORK / "curl_pid.next"
    with pending.open("x") as stream:
        print(os.getpid(), file=stream)
    os.replace(pending, WORK / "download.pid")
    files = snapshot()
    progress = Progress(files)
    publish("starting", downloaded_bytes=progress.high, total_bytes=progress.total, replaced_pid=OLD_PID)
    try:
        with (WORK / "curl_transport.log").open("xb", buffering=0) as log:
            inventory = [fetch(row, progress, log) for row in files]
        readme = (DESTINATION / "README.md").read_text(encoding="utf-8")
        if not re.search(r"(?im)^license:[ \t]*apache-2\.0[ \t]*$", readme):
            raise ValueError("Pinned model README lacks the expected license")
        write_json(CARD, {"model": MODEL, "source": "https://modelscope.cn/models/" + MODEL,
                          "revision": REVISION, "license": "Apache-2.0", "license_source": "snapshot README.md",
                          "directory": str(DESTINATION), "cache": str(STAGING), "transport": "curl",
                          "files": sorted(inventory, key=lambda row: row["file"]), "completed_utc": now().isoformat()})
        publish("complete", downloaded_bytes=progress.verified, total_bytes=progress.total,
                receipt=str(CARD), receipt_sha256=digest(CARD))
    except Stalled as exc:
        publish("need", downloaded_bytes=progress.high, total_bytes=progress.total, reason=str(exc))
        print("NEED:", exc, file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        publish("failed", downloaded_bytes=progress.high, total_bytes=progress.total, reason=str(exc))
        raise


def detach():
    if Path.cwd() != PROJECT or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run on 5880 with CUDA hidden")
    cmdline = Path(f"/proc/{OLD_PID}/cmdline")
    if cmdline.exists() and b"resume_reranker_v1_download_ranges.py" in cmdline.read_bytes():
        raise RuntimeError("The exact old downloader is still active")
    if int((WORK / "download.pid").read_text()) != OLD_PID:
        raise ValueError("Unexpected previous PID")
    if ARCHIVE.exists() or CARD.exists() or (WORK / "curl_download.log").exists():
        raise FileExistsError("Existing archive, completion card or curl log")
    shutil.copytree(WORK, ARCHIVE)
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
    env.update(CUDA_VISIBLE_DEVICES="", MODELSCOPE_CACHE=str(CACHE), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", NO_PROXY="*", no_proxy="*")
    with (WORK / "curl_download.log").open("xb") as log:
        proc = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "worker"],
                                cwd=PROJECT, env=env, start_new_session=True, close_fds=True,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({"pid": proc.pid, "status": str(WORK / "status.json"),
                      "archived": str(ARCHIVE), "old_pid": OLD_PID}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("detach", "worker"))
    args = parser.parse_args()
    if args.action == "worker":
        worker()
    else:
        detach()

