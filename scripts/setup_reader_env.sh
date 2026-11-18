#!/usr/bin/env bash
set -euo pipefail
cd /root/workplace/ctx-compressor
source scripts/reader_environment.sh
mkdir -p logs eval/reader
printf '%s\n' "$$" > logs/setup_reader_env.pid
date -u +%FT%TZ > logs/setup_reader_env.started
READER_ROOT_BASELINE=931190407168
check_root_disk() {
    local reader_root_used
    reader_root_used="$(df -B1 --output=used / | tail -n 1 | tr -d ' ')"
    printf 'root_used_bytes=%s baseline=%s delta=%s\n' "$reader_root_used" "$READER_ROOT_BASELINE" "$((reader_root_used - READER_ROOT_BASELINE))"
    if (( reader_root_used - READER_ROOT_BASELINE > 300000000 )); then
        echo 'Root disk increase exceeds 300 MB; stop and investigate.' >&2
        return 1
    fi
}
check_root_disk
df -h / /root/userfolder
CTXC_PREFIX="$(cat logs/environment.path)"
CTXC_PY="$PWD/$CTXC_PREFIX/bin/python"
"$CTXC_PY" -B -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
if ! test -x "$READER_ENV/bin/python"; then
    "$CTXC_PY" -B -m venv --copies "$READER_ENV"
fi
"$READER_ENV/bin/python" -B -c 'import sys; assert sys.prefix == "/root/userfolder/ctx-compressor/envs/reader"; assert sys.version_info[:2] == (3, 11)'
# v0.11.0 official CUDA wheel targets CUDA 12.8 / PyTorch 2.8.0.
# numpy 2.2.6 satisfies the required numba 0.61.2 constraint (<2.3).
if test -s requirements.reader.lock.txt; then
    "$READER_ENV/bin/python" -m pip install \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 90 --retries 5 --prefer-binary \
        -r requirements.reader.lock.txt
else
    "$READER_ENV/bin/python" -m pip install \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 90 --retries 5 --prefer-binary \
        'vllm==0.11.0' 'transformers==4.57.6' 'tokenizers==0.22.2' \
        'numpy==2.2.6' 'bm25s==0.3.16' 'scipy==1.17.1'
fi
check_root_disk
"$READER_ENV/bin/python" -m pip check
"$READER_ENV/bin/python" -m pip freeze > requirements.reader.lock.txt
"$READER_ENV/bin/python" -B - <<'PY'
import importlib.metadata as m
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
required = ["vllm", "torch", "torchaudio", "torchvision", "xformers", "transformers", "tokenizers", "numpy", "bm25s", "scipy"]
result = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "prefix": sys.prefix,
    "platform": platform.platform(),
    "packages": {name: m.version(name) for name in required},
    "check": "importlib.metadata only; no GPU/library initialization",
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "cache_root": os.environ["READER_CACHE"],
}
assert result["packages"]["vllm"] == "0.11.0"
assert result["packages"]["torch"] == "2.8.0"
Path("eval/reader/environment_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
"$CTXC_PY" -B -c 'import importlib.util; assert importlib.util.find_spec("torch") is None, "old ctxc must remain torch-free"'
check_root_disk
df -h / /root/userfolder
date -u +%FT%TZ > logs/setup_reader_env.finished
