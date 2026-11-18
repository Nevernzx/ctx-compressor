#!/usr/bin/env bash
# Start once with nohup and a fresh POSIX session. A selects; B starts immediately.
set -euo pipefail
cd /root/workplace/ctx-compressor
source scripts/train_environment.sh
export CUDA_VISIBLE_DEVICES=1
if [[ "${1:-}" == "--resume-initialization" ]]; then
    "$TRAIN_ENV/bin/python" -u -B scripts/bench_batch.py resume-initialization
else
    "$TRAIN_ENV/bin/python" -u -B scripts/bench_batch.py run
fi
exec bash scripts/train_sft.sh > /root/userfolder/ctx-compressor/runs/sft_20k_20261117/train.log 2>&1
