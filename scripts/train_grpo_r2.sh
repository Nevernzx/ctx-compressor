#!/usr/bin/env bash
# Same arguments as the frozen Task 08 train_grpo invocation, with the round-two overrides.
set -euo pipefail
cd /root/workplace/ctx-compressor
source scripts/grpo_r2_environment.sh
export MERGED="${MERGED:?merged start model}" DATA="${DATA:?training JSONL}" OUT="${OUT:?fresh training directory}"
export LR="${LR:-1.5e-5}" SCALE_REWARDS="${SCALE_REWARDS:-none}" GRPO_SEED="${GRPO_SEED:-20261123}"
[[ -x "$GRPO_ENV/bin/python" ]] || { echo "NEED: existing GRPO Python missing" >&2; exit 2; }
exec "$GRPO_ENV/bin/python" -u -B scripts/run_grpo_r2_training.py

