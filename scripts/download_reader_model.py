#!/usr/bin/env python3
"""Direct ModelScope snapshot download with pinned revisions and SHA256 checks.

Run on 5880 after sourcing scripts/reader_environment.sh. The first metadata
listing is frozen in the manifest; retries reuse that listing, never a new
master snapshot. No ModelScope/PyTorch/vLLM package import is needed.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

MODEL_ID = "Qwen/Qwen3-8B"
LIST_URL = "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-8B/repo/files?Revision=master&Recursive=true"
MODEL_ROOT = Path("/root/userfolder/ctx-compressor/models/Qwen3-8B")
ROOT_BASELINE = 931190407168


def now():
    return datetime.now(timezone.utc).isoformat()


def root_disk_check():
    st = os.statvfs("/")
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    delta = used - ROOT_BASELINE
    if delta > 300000000:
        raise RuntimeError(f"Root disk growth exceeds 300 MB: {delta} bytes")
    return {"used_bytes": used, "baseline_bytes": ROOT_BASELINE, "delta_bytes": delta}


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def discover():
    with opener().open(LIST_URL, timeout=90) as response:
        listing = json.load(response)
    if listing.get("Code") != 200:
        raise RuntimeError(f"ModelScope listing failed: {listing}")
    files = []
    for item in listing["Data"]["Files"]:
        if item["Type"] != "blob":
            continue
        path = PurePosixPath(item["Path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path: {path}")
        if not item.get("Sha256") or len(item["Sha256"]) != 64:
            raise ValueError(f"Missing SHA256: {path}")
        revision = item["Revision"]
        url = "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-8B/repo?" + urllib.parse.urlencode({"Revision": revision, "FilePath": str(path)})
        files.append({"path": str(path), "size_bytes": item["Size"],
                      "sha256": item["Sha256"].lower(), "revision": revision,
                      "url": url, "status": "pending"})
    return {"model_id": MODEL_ID, "requested_revision": "master", "model_root": str(MODEL_ROOT),
            "listing_url": LIST_URL, "listing_fetched_at_utc": now(), "source_listing": listing,
            "files": files, "total_expected_bytes": sum(item["size_bytes"] for item in files),
            "status": "downloading", "root_disk_before": root_disk_check()}


def download_file(item, max_attempts):
    item = dict(item)
    destination = MODEL_ROOT / item["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    started = time.monotonic()
    errors = []
    if destination.exists() and destination.stat().st_size == item["size_bytes"] and sha256(destination) == item["sha256"]:
        if "downloaded_at_utc" not in item:
            item["downloaded_at_utc"] = datetime.fromtimestamp(destination.stat().st_mtime, timezone.utc).isoformat()
            item["download_time_source"] = "verified existing file mtime"
        item.update(status="verified", verified_at_utc=now(), actual_size_bytes=destination.stat().st_size,
                    actual_sha256=item["sha256"], reused_verified_file=True)
        return item
    attempt = 0
    failures = 0
    while failures < max_attempts:
        attempt += 1
        offset = temporary.stat().st_size if temporary.exists() else 0
        try:
            root_disk_check()
            if offset > item["size_bytes"]:
                temporary.unlink()
                offset = 0
            if offset < item["size_bytes"]:
                headers = {"User-Agent": "ctx-compressor-task03/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(item["url"], headers=headers)
                with opener().open(request, timeout=90) as response:
                    status = response.status
                    if offset and status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                        mode = "ab"
                    else:
                        if status != 200:
                            raise RuntimeError(f"Unexpected HTTP status: {status}")
                        mode = "wb"
                    with temporary.open(mode) as output:
                        last_disk_check = time.monotonic()
                        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            output.write(chunk)
                            if time.monotonic() - last_disk_check > 30:
                                root_disk_check()
                                last_disk_check = time.monotonic()
            actual_size = temporary.stat().st_size
            if actual_size != item["size_bytes"]:
                if offset < actual_size < item["size_bytes"]:
                    print(json.dumps({"path": item["path"], "status": "partial_resume",
                                      "downloaded_bytes": actual_size, "expected_bytes": item["size_bytes"]}), flush=True)
                    continue
                raise RuntimeError(f"Size mismatch: {actual_size} != {item['size_bytes']}")
            actual_hash = sha256(temporary)
            if actual_hash != item["sha256"]:
                temporary.unlink()
                raise RuntimeError(f"SHA256 mismatch: {actual_hash} != {item['sha256']}; discarded partial")
            temporary.replace(destination)
            item.update(status="verified", downloaded_at_utc=now(), verified_at_utc=now(),
                        elapsed_seconds=round(time.monotonic() - started, 3), attempts=attempt,
                        actual_size_bytes=actual_size, actual_sha256=actual_hash, retry_errors=errors)
            print(json.dumps({"path": item["path"], "status": "verified", "bytes": actual_size,
                              "elapsed_seconds": item["elapsed_seconds"]}), flush=True)
            return item
        except Exception as error:
            errors.append({"attempt": attempt, "at_utc": now(), "error": f"{type(error).__name__}: {error}"})
            print(json.dumps({"path": item["path"], **errors[-1]}), flush=True)
            if "Root disk growth" in str(error):
                break
            # Successful bytes are resumable progress, including a timeout after
            # a partial write. Only no-progress or failed validation consumes
            # the finite failure budget; bounded server responses do not.
            current_size = temporary.stat().st_size if temporary.exists() else 0
            if current_size <= offset or "SHA256 mismatch" in str(error):
                failures += 1
            if failures < max_attempts:
                time.sleep(min(30, 3 * max(1, failures)))
    item.update(status="failed", retry_errors=errors, elapsed_seconds=round(time.monotonic() - started, 3))
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("eval/reader/model_manifest.json"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    root_disk_check()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        assert manifest["model_id"] == MODEL_ID and manifest["model_root"] == str(MODEL_ROOT)
    else:
        manifest = discover()
    save_manifest(args.manifest, manifest)
    print(json.dumps({"model_id": MODEL_ID, "files": len(manifest["files"]),
                      "total_expected_bytes": manifest["total_expected_bytes"]}), flush=True)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_file, item, args.attempts): index for index, item in enumerate(manifest["files"])}
        for future in concurrent.futures.as_completed(futures):
            manifest["files"][futures[future]] = future.result()
            save_manifest(args.manifest, manifest)
    manifest["verified_file_count"] = sum(item["status"] == "verified" for item in manifest["files"])
    manifest["status"] = "verified" if manifest["verified_file_count"] == len(manifest["files"]) else "failed"
    manifest["finished_at_utc"] = now()
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    manifest["root_disk_after"] = root_disk_check()
    save_manifest(args.manifest, manifest)
    print(json.dumps({key: manifest[key] for key in ["status", "verified_file_count", "elapsed_seconds", "root_disk_after"]}), flush=True)
    return 0 if manifest["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
