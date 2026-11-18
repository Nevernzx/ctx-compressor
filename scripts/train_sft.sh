#!/usr/bin/env bash
# Detach this wrapper with setsid nohup; all long-running state is external.
set -euo pipefail
cd /root/workplace/ctx-compressor
source scripts/train_environment.sh
export CUDA_VISIBLE_DEVICES=1
exec "$TRAIN_ENV/bin/python" -u -B scripts/train_sft_worker.py train
