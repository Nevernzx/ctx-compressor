"""Fetch the pinned Task 17 ModelScope snapshot and verify each published digest."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import urlencode
from urllib.request import build_opener, ProxyHandler

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("/root/userfolder/ctx-compressor")
MODEL = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "0311d7b88fd535395d154b17fe08f85327306508"
DESTINATION = ROOT / "models/Qwen3-Reranker-0.6B"
CACHE = ROOT / "cache/modelscope"
BASE = "https://modelscope.cn/api/v1/models/" + MODEL
CARD = PROJECT / "eval/reranker_v1/model_card.json"


def get(path, params, timeout=120):
    return build_opener(ProxyHandler({})).open(BASE + path + "?" + urlencode(params), timeout=timeout)


def download(file, cache):
    path = PurePosixPath(file["Path"])
    if path.is_absolute() or ".." in path.parts or "." in path.parts or ".git" in path.parts:
        raise ValueError("unsafe snapshot path")
    sha256 = file["Sha256"]
    if not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError(f"no published file hash for {path}")
    staged = cache / path
    staged.parent.mkdir(parents=True, exist_ok=True)
    hashed = hashlib.sha256()
    size = 0
    with get("/repo", {"Revision": REVISION, "FilePath": str(path)}) as remote, staged.open("xb") as output:
        while chunk := remote.read(4 << 20):
            size += len(chunk)
            if size > file["Size"]:
                raise ValueError(f"snapshot returned a file larger than declared: {path}")
            hashed.update(chunk)
            output.write(chunk)
    if hashed.hexdigest() != sha256 or size != file["Size"]:
        raise ValueError(f"snapshot file digest or byte count mismatch: {path}")
    target = DESTINATION / path
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(staged, target)
    print(str(path), size, sha256, flush=True)
    return {"file": str(path), "bytes": size, "sha256": sha256, "file_revision": file["Revision"]}


def main():
    if Path.cwd() != PROJECT or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("run the download on 5880 with CUDA hidden")
    if any(os.path.lexists(p) for p in (CARD, DESTINATION)):
        raise FileExistsError("Task 17 model receipt or destination already exists")
    os.environ["MODELSCOPE_CACHE"] = str(CACHE)
    if CACHE.resolve() != CACHE or DESTINATION.resolve() != DESTINATION:
        raise ValueError("model directory or cache was redirected")
    with get("/repo/files", {"Revision": REVISION, "Recursive": "true"}) as remote:
        envelope = json.load(remote)
    data = envelope["Data"]
    files = [f for f in data["Files"] if f["Type"] == "blob"]
    if envelope["Code"] != 200 or data["LatestCommitter"]["ShortId"] != REVISION[:8] or len(files) != 15:
        raise ValueError("unexpected ModelScope snapshot metadata")
    DESTINATION.mkdir()
    cache = CACHE / ("task17_qwen3_reranker_" + REVISION)
    cache.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        inventory = sorted(pool.map(lambda file: download(file, cache), files), key=lambda item: item["file"])
    document = (DESTINATION/"README.md").read_text(encoding="utf-8")
    if not re.search(r"(?im)^license:[ \t]*apache-2\.0[ \t]*$", document):
        raise ValueError("ModelScope README metadata does not declare Apache-2.0")
    CARD.parent.mkdir(exist_ok=True)
    with CARD.open("x", encoding="utf-8") as stream:
        json.dump({"model": MODEL, "source": "https://modelscope.cn/models/" + MODEL,
                   "revision": REVISION, "license": "Apache-2.0", "license_source": "snapshot README.md",
                   "directory": str(DESTINATION), "cache": str(cache), "files": inventory,
                   "completed_utc": datetime.now(timezone.utc).isoformat()}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NEED: Task17 ModelScope model snapshot: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

