"""Persist the pinned ModelScope download across direct network interruptions."""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import build_opener, ProxyHandler, Request

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.download_reranker_v1 import BASE, CACHE, CARD, DESTINATION, MODEL, REVISION, ROOT

WORK = ROOT / "reranker_v1/download"
STAGING = CACHE / ("task17_qwen3_reranker_" + REVISION)
DEADLINE_WITHOUT_PROGRESS = 3 * 3600
IO_TIMEOUT = 50
RETRY_SECONDS = 60
BUFFER_BYTES = 1 << 20


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish(status, **detail):
    value = {"utc": utc(), "pid": os.getpid(), "status": status, "model": MODEL,
             "revision": REVISION, "destination": str(DESTINATION), **detail}
    temporary = WORK / "status.next"
    write_json(temporary, value)
    os.replace(temporary, WORK / "status.json")
    with (WORK / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    progress = detail.get("downloaded_bytes")
    print(f"{value['utc']} {status} file={detail.get('file')} bytes={progress} {detail.get('reason','')}",
          flush=True)


def open_model(endpoint, params, headers=None):
    url = BASE + endpoint + "?" + urlencode(params)
    request = Request(url, headers={"Accept-Encoding": "identity", **(headers or {})})
    return build_opener(ProxyHandler({})).open(request, timeout=IO_TIMEOUT)


def manifest():
    with open_model("/repo/files", {"Revision": REVISION, "Recursive": "true"}) as response:
        envelope = json.load(response)
    data = envelope["Data"]
    if envelope["Code"] != 200 or data["LatestCommitter"]["ShortId"] != REVISION[:8]:
        raise ValueError("immutable ModelScope revision was not returned")
    files = {}
    for row in data["Files"]:
        if row["Type"] != "blob":
            continue
        name = row["Path"]
        path = PurePosixPath(name)
        if (path.is_absolute() or str(path) != name or {".git", "..", "."}.intersection(path.parts)
                or not re.fullmatch(r"[0-9a-f]{64}", row["Sha256"]) or row["Size"] <= 0 or name in files):
            raise ValueError("unexpected name, hash or size in the pinned snapshot")
        files[name] = row
    if len(files) != 15 or not {"README.md", "config.json", "model.safetensors", "tokenizer.json"} <= files.keys():
        raise ValueError("pinned snapshot inventory was incomplete")
    return sorted(files.values(), key=lambda row: (row["Size"], row["Path"]))


def check_response(status, headers, offset, expected):
    if headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("unsupported content encoding for byte ranges")
    content_length = headers.get("Content-Length")
    if offset == 0 and status == 200 and headers.get("Content-Range") is None:
        count = expected
    elif status in (200, 206):
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("Content-Range", ""))
        if match is None:
            raise ValueError("invalid content range")
        start, end, length = map(int, match.groups())
        if not (offset == start <= end < length == expected):
            raise ValueError("server byte range did not match the signed file or prior offset")
        count = end - start + 1
    else:
        raise ValueError("server did not honor resume range")
    if content_length is not None and int(content_length) != count:
        raise ValueError("server length did not match the requested byte range")
    return count


def digest(path):
    with path.open("rb") as stream:
        hasher = hashlib.sha256()
        while chunk := stream.read(4 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_partial(name):
    directory = STAGING / (name + ".parts")
    directory.mkdir(parents=True, exist_ok=True)
    number = 0
    while True:
        path = directory / f"{number:04}.partial"
        if path.resolve() != path or (path.exists() and not path.is_file()):
            raise ValueError("unsafe partial path")
        if not path.exists():
            with path.open("xb"):
                pass
            return path
        if not path.with_suffix(".rejected.json").exists():
            return path
        number += 1


class Progress:
    def __init__(self):
        self.last_progress = self.last_publish = time.monotonic()
        self.highwater = 0
        previous = WORK.with_name("download_attempt2") / "status.json"
        if previous.is_file():
            data = json.loads(previous.read_text(encoding="utf-8"))
            elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(data["utc"])).total_seconds()
            self.last_progress -= max(0, elapsed + data.get("seconds_without_progress",0))
            self.highwater = data.get("downloaded_bytes",0)
        self.complete_bytes = 0
        self.total_bytes = None

    def report(self, row, offset, state, force=False, **extra):
        total = self.complete_bytes + offset
        if total > self.highwater:
            self.highwater = total
            self.last_progress = time.monotonic()
        now = time.monotonic()
        if force or now - self.last_publish >= 5:
            publish(state, file=row["Path"], file_bytes=offset, file_total_bytes=row["Size"],
                    downloaded_bytes=total, verified_bytes=self.complete_bytes,
                    total_bytes=self.total_bytes, seconds_without_progress=round(now - self.last_progress, 2), **extra)
            self.last_publish = now

    def retry(self):
        left = DEADLINE_WITHOUT_PROGRESS - (time.monotonic() - self.last_progress)
        if left <= 0:
            raise TimeoutError("3 hours without download progress")
        time.sleep(min(RETRY_SECONDS, left))
        if time.monotonic() - self.last_progress >= DEADLINE_WITHOUT_PROGRESS:
            raise TimeoutError("3 hours without download progress")


def download(row, progress):
    name = row["Path"]
    target = DESTINATION / name
    if target.resolve() != target:
        raise ValueError("unsafe installed model path")
    while not target.exists():
        part = safe_partial(name)
        offset = part.stat().st_size
        progress.report(row, offset, "downloading", force=True, partial=str(part))
        if offset == row["Size"] and digest(part) == row["Sha256"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(part, target)
            break
        if offset >= row["Size"]:
            write_json(part.with_suffix(".rejected.json"), {"file": str(part), "size": offset,
                       "sha256": digest(part), "reason": "signed size or SHA256 mismatch"})
            progress.report(row, offset, "retry_wait", force=True, reason="signed size or SHA256 mismatch")
            progress.retry()
            continue
        try:
            last_byte = row["Size"] - 1
            headers = {"Range": f"bytes={offset}-{last_byte}"} if offset or row["Size"] > 16 << 20 else None
            with open_model("/repo", {"Revision": REVISION, "FilePath": name}, headers) as remote:
                if offset and remote.status == 200 and remote.headers.get("Content-Range") is None:
                    # ModelScope can ignore ranges on non-LFS files. Keep the former attempt
                    # and accept this body only as a byte-zero response in a new partial.
                    count = check_response(remote.status, remote.headers, 0, row["Size"])
                    write_json(part.with_suffix(".rejected.json"), {"file":str(part),"size":offset,
                               "reason":"HTTP 200 on Range: full-body attempt in a fresh file"})
                    part = safe_partial(name)
                    offset = 0
                    progress.report(row, offset, "downloading", force=True, partial=str(part),
                                    reason="HTTP 200 on range: preserved old partial")
                else:
                    count = check_response(remote.status, remote.headers, offset, row["Size"])
                # Append only to this task's partial, after validating Range and file shape.
                with part.open("ab", buffering=0) as stream:
                    left = count
                    while left:
                        block = remote.read1(min(BUFFER_BYTES, left))
                        if not block:
                            raise OSError("connection ended within the declared range")
                        stream.write(block)
                        left -= len(block)
                        progress.report(row, offset+count-left, "downloading", partial=str(part))
                    os.fsync(stream.fileno())
        except (OSError, ValueError, HTTPError, URLError, http.client.HTTPException) as exc:
            progress.report(row, part.stat().st_size, "retry_wait", force=True, reason=type(exc).__name__,
                            next_retry_seconds=RETRY_SECONDS, partial=str(part))
            progress.retry()
    if not target.is_file() or target.stat().st_size != row["Size"] or digest(target) != row["Sha256"]:
        raise ValueError(f"published model file does not match the pinned inventory: {name}")
    progress.complete_bytes += row["Size"]
    progress.report(row, 0, "file_verified", force=True)
    return {"file": name, "bytes": row["Size"], "sha256": row["Sha256"], "file_revision": row["Revision"]}


def worker():
    if os.getsid(0) != os.getpid():
        raise ValueError("run the download in a detached session")
    with (WORK / "download.pid").open("x") as f:
        f.write(f"{os.getpid()}\n")
    progress = Progress()
    publish("starting", downloaded_bytes=0)
    try:
        archived = WORK.with_name("download_attempt1") / "pinned_files.json"
        if archived.is_file():
            previous = json.loads(archived.read_text(encoding="utf-8"))
            if previous["revision"] != REVISION:
                raise ValueError("archived pinned ModelScope revision differs")
            files = previous["files"]
        else:
            while True:
                try:
                    files = manifest()
                    break
                except (OSError, ValueError, HTTPError, URLError, http.client.HTTPException) as exc:
                    publish("retry_wait", step="pinned_manifest", downloaded_bytes=0, reason=type(exc).__name__,
                            next_retry_seconds=RETRY_SECONDS)
                    progress.retry()
        write_json(WORK / "pinned_files.json", {"revision": REVISION, "files": files})
        DESTINATION.mkdir(exist_ok=True)
        STAGING.mkdir(parents=True, exist_ok=True)
        progress.total_bytes = sum(f["Size"] for f in files)
        inventory = [download(row, progress) for row in files]
        document = (DESTINATION / "README.md").read_text(encoding="utf-8")
        if not re.search(r"(?im)^license:[ \t]*apache-2\.0[ \t]*$", document):
            raise ValueError("pinned ModelScope README does not declare Apache-2.0")
        inventory.sort(key=lambda row: row["file"])
        write_json(CARD, {"model": MODEL, "source": "https://modelscope.cn/models/" + MODEL,
                          "revision": REVISION, "license": "Apache-2.0", "license_source": "snapshot README.md",
                          "directory": str(DESTINATION), "cache": str(STAGING), "files": inventory,
                          "completed_utc": utc()})
        publish("complete", downloaded_bytes=progress.complete_bytes, total_bytes=progress.total_bytes,
                receipt=str(CARD), receipt_sha256=digest(CARD))
    except Exception as exc:
        publish("need", downloaded_bytes=progress.highwater, reason=f"{type(exc).__name__}: {exc}")
        print(f"NEED: Task17 pinned ModelScope download: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None


def detach():
    if Path.cwd() != PROJECT or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("run on 5880 with CUDA hidden")
    for path in (DESTINATION, STAGING, WORK, CARD):
        if path.resolve() != path or os.path.lexists(path) and path in (WORK, CARD):
            raise FileExistsError(f"existing receipt or unsafe download path: {path}")
    WORK.mkdir(parents=True)
    CARD.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy","https_proxy","all_proxy")}
    env.update(CUDA_VISIBLE_DEVICES="", MODELSCOPE_CACHE=str(CACHE), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", NO_PROXY="*", no_proxy="*")
    with (WORK / "download.log").open("xb") as output:
        job = subprocess.Popen([sys.executable,"-B","-u",str(Path(__file__).resolve()),"worker"],
                               cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=output,
                               stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(f"{job.pid} {WORK}", flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("action", choices=("detach", "worker"))
    namespace = args.parse_args()
    (detach if namespace.action == "detach" else worker)()

