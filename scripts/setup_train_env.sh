#!/usr/bin/env bash
# Task 04: independent Python 3.11 environment; no CUDA initialization.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/train_environment.sh
export CUDA_VISIBLE_DEVICES=''
mkdir -p logs eval/sft_prep
printf '%s\n' "$$" > logs/task04_setup_train_env.pid
date -u +%FT%TZ > logs/task04_setup_train_env.started
CTXC_PREFIX="$(cat logs/environment.path)"
CTXC_PY="$PWD/$CTXC_PREFIX/bin/python"
check_root_disk() {
    "$CTXC_PY" -B - <<'PY'
import json, os
from pathlib import Path
preflight = json.loads(Path('eval/sft_prep/preflight.json').read_text())
st = os.statvfs('/')
used = (st.f_blocks - st.f_bfree) * st.f_frsize
delta = used - preflight['root_df_used_bytes']
print(f'root_used_bytes={used} delta_bytes={delta}', flush=True)
if delta > preflight['root_final_growth_limit_bytes']:
    raise RuntimeError('Root disk growth exceeds preflight limit')
PY
}
check_root_disk
"$CTXC_PY" -B -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
if ! test -x "$TRAIN_ENV/bin/python"; then
    "$CTXC_PY" -B -m venv --copies "$TRAIN_ENV"
fi
"$TRAIN_ENV/bin/python" -B -c 'import os,sys; assert sys.prefix == os.environ["TRAIN_ENV"]; assert sys.version_info[:2] == (3, 11)'
# v3.12.6 explicitly supports Qwen3 and pins transformers<4.58, peft<0.19,
# trl<0.25. Torch 2.8.0 reuses the Task 03 CUDA 12.8 wheel cache.
if test -s requirements.train.lock.txt; then
    "$TRAIN_ENV/bin/python" -m pip install --no-compile \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 90 --retries 5 --prefer-binary \
        -r requirements.train.lock.txt
else
    "$TRAIN_ENV/bin/python" -m pip install --no-compile \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 90 --retries 5 --prefer-binary \
        'ms-swift==3.12.6' 'torch==2.8.0' 'transformers==4.57.6' 'tokenizers==0.22.2' \
        'peft==0.18.0' 'trl==0.24.0' 'accelerate==1.12.0' 'datasets==3.6.0' \
        'numpy==2.2.6' 'scipy==1.17.1' 'pandas==2.3.3' 'bm25s==0.3.16'
fi
check_root_disk
"$TRAIN_ENV/bin/python" -m pip check
"$TRAIN_ENV/bin/python" -m pip freeze > requirements.train.lock.txt
"$TRAIN_ENV/bin/python" -B - <<'PY'
import importlib.metadata as m
import json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
required = ['ms-swift', 'torch', 'transformers', 'tokenizers', 'peft', 'trl', 'accelerate', 'datasets', 'numpy', 'scipy', 'pandas', 'bm25s']
result = {
    'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'python': sys.version, 'prefix': sys.prefix, 'platform': platform.platform(),
    'packages': {name: m.version(name) for name in required},
    'check': 'importlib.metadata only; no GPU initialization',
    'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
    'cache_root': os.environ['TRAIN_CACHE'],
    'attention_implementation': 'sdpa',
}
assert result['packages']['torch'] == '2.8.0'
assert result['packages']['ms-swift'] == '3.12.6'
for package in ['vllm', 'flash-attn']:
    try:
        m.version(package)
    except m.PackageNotFoundError:
        continue
    raise RuntimeError(f'Unexpected optional package installed: {package}')
Path('eval/sft_prep/environment_metadata.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
"$CTXC_PY" -B -c 'import importlib.util; assert importlib.util.find_spec("torch") is None, "old ctxc must remain torch-free"'
check_root_disk
date -u +%FT%TZ > logs/task04_setup_train_env.finished
