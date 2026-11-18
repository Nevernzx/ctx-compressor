#!/usr/bin/env bash
# Run only on 5880. Keep caches and temporary output within this project.
set -euo pipefail
cd /root/workplace/ctx-compressor
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*' CUDA_VISIBLE_DEVICES=''
export HF_HOME="$PWD/.cache/huggingface" TMPDIR="$PWD/.tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$HF_HOME"
CTXC_ENV="$(cat logs/environment.path)"
exec "$PWD/$CTXC_ENV/bin/python" "$@"
