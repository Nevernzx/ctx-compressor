#!/usr/bin/env python3
"""Offline final verification; run only after the main downloader has exited."""
import argparse
import json
from pathlib import Path
import time
from datetime import datetime
from download_reader_model import MODEL_ROOT, now, root_disk_check, save_manifest, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("eval/reader/model_manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "verified":
        raise RuntimeError("Wait for the main downloader to finish successfully before final verification")
    started = time.monotonic()
    receipt_path = args.manifest.with_name("model_small_files_receipt.json")
    receipts = {}
    if receipt_path.exists():
        receipts = {item["path"]: item for item in json.loads(receipt_path.read_text())["files"]}
    if len(manifest["files"]) != 16:
        raise RuntimeError("Task 03 expects the fixed 16-file ModelScope snapshot")
    for item in manifest["files"]:
        path = MODEL_ROOT / item["path"]
        actual_hash = sha256(path)
        if path.stat().st_size != item["size_bytes"] or actual_hash != item["sha256"]:
            raise RuntimeError(f"Offline re-verification failed: {item['path']}")
        receipt = receipts.get(item["path"])
        if receipt and ("downloaded_at_utc" not in item or item.get("download_time_source") == "verified existing file mtime"):
            if receipt["actual_sha256"] != actual_hash or receipt["revision"] != item["revision"]:
                raise RuntimeError(f"Small-file receipt mismatch: {item['path']}")
            item["downloaded_at_utc"] = receipt["downloaded_at_utc"]
            item["download_time_source"] = "separate priority small-file download receipt"
        if "downloaded_at_utc" not in item:
            raise RuntimeError(f"Missing download time: {item['path']}")
        item["offline_reverified_at_utc"] = now()
        print(json.dumps({"path": item["path"], "size_bytes": path.stat().st_size, "sha256": actual_hash}), flush=True)
    actual_paths = {str(path.relative_to(MODEL_ROOT)) for path in MODEL_ROOT.rglob("*") if path.is_file()}
    expected_paths = {item["path"] for item in manifest["files"]}
    if actual_paths != expected_paths:
        raise RuntimeError(f"Unexpected model directory contents: {sorted(actual_paths ^ expected_paths)}")
    manifest["offline_verification"] = {"verified_at_utc": now(), "file_count": len(expected_paths),
                                        "elapsed_seconds": round(time.monotonic() - started, 3),
                                        "all_sizes_and_sha256_match_official_listing": True,
                                        "root_disk": root_disk_check()}
    manifest["total_wall_seconds_since_listing"] = round((datetime.fromisoformat(manifest["finished_at_utc"]) - datetime.fromisoformat(manifest["listing_fetched_at_utc"])).total_seconds(), 3)
    save_manifest(args.manifest, manifest)
    print(json.dumps(manifest["offline_verification"]), flush=True)


if __name__ == "__main__":
    main()
