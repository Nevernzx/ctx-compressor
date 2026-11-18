#!/usr/bin/env python3
"""Fetch pending tokenizer files without modifying the active main manifest.

Run only while main download workers are busy with the three first shards.
Main downloader later verifies/reuses these files. The independent receipt
preserves actual download times for final manifest reconciliation.
"""
import json
from pathlib import Path
from download_reader_model import download_file, now, save_manifest


def main():
    path = Path("eval/reader/model_manifest.json")
    manifest = json.loads(path.read_text())
    records = []
    wanted = {"model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
    for item in manifest["files"]:
        if item["path"] in wanted:
            records.append(download_file(item, 6))
    receipt = {"created_at_utc": now(), "model_id": manifest["model_id"], "files": records}
    save_manifest(Path("eval/reader/model_small_files_receipt.json"), receipt)
    return 0 if all(item["status"] == "verified" for item in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
