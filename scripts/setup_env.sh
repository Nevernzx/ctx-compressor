#!/usr/bin/env bash
set -uo pipefail
cd /root/workplace/ctx-compressor
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*' CUDA_VISIBLE_DEVICES=''
export CONDA_ENVS_PATH="$PWD/.conda/envs" CONDA_PKGS_DIRS="$PWD/.cache/conda/pkgs"
export PIP_CACHE_DIR="$PWD/.cache/pip" HF_HOME="$PWD/.cache/huggingface"
export TMPDIR="$PWD/.tmp"
mkdir -p "$TMPDIR" .cache logs
printf '%s\n' "$$" > logs/setup_env.pid
date -u +%FT%TZ > logs/environment.started
if test -x "$PWD/.conda/envs/ctxc/bin/python"; then
    CTXC_PY="$PWD/.conda/envs/ctxc/bin/python"
    printf '%s\n' '.conda/envs/ctxc' > logs/environment.path
elif test -x "$PWD/.venv/bin/python"; then
    CTXC_PY="$PWD/.venv/bin/python"
    printf '%s\n' '.venv' > logs/environment.path
elif timeout 150 /root/anaconda3/bin/conda create -y -n ctxc python=3.11 pip --no-default-packages; then
    CTXC_PY="$PWD/.conda/envs/ctxc/bin/python"
    printf '%s\n' '.conda/envs/ctxc' > logs/environment.path
else
    echo 'Conda unavailable; creating isolated venv using base Python, without installing into base.'
    /root/anaconda3/bin/python -m venv "$PWD/.venv" || exit 1
    CTXC_PY="$PWD/.venv/bin/python"
    printf '%s\n' '.venv' > logs/environment.path
fi
"$CTXC_PY" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 45 --retries 2 -r requirements.txt || exit 1
"$CTXC_PY" -m pip freeze > requirements.lock.txt
"$CTXC_PY" -c 'import importlib.util; import sys; print(sys.version); assert importlib.util.find_spec("torch") is None, "torch must not be installed"; import bm25s, transformers, numpy, scipy, pytest; print("CPU tokenizer/retrieval/test dependencies OK")' || exit 1
date -u +%FT%TZ > logs/environment.finished
