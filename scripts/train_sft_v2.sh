#!/usr/bin/env bash
set -euo pipefail
cd /root/workplace/ctx-compressor
source scripts/train_environment.sh
export CUDA_VISIBLE_DEVICES=1
exec "$TRAIN_ENV/bin/python" -u -B scripts/train_sft_v2_worker.py
