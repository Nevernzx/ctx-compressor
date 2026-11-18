#!/usr/bin/env bash
set -euo pipefail
cd /root/workplace/ctx-compressor
source scripts/train_environment.sh
# Independent host compiler for Triton's LoRA extensions; sysroot is built in.
TOOLCHAIN_PREFIX=/root/userfolder/ctx-compressor/toolchains/gcc13_20261118/envs/gcc
export CC="$TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$TOOLCHAIN_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export PATH="$TOOLCHAIN_PREFIX/bin:$PATH"
test -x "$CC"
test -x "$CXX"
export CUDA_VISIBLE_DEVICES=1
exec /root/userfolder/ctx-compressor/envs/reader/bin/python -u -B scripts/rollout_model.py run
