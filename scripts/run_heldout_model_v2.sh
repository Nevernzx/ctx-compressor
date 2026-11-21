#!/usr/bin/env bash
set -euo pipefail
cd /root/workplace/ctx-compressor
source scripts/train_environment.sh
toolchain=/root/userfolder/ctx-compressor/toolchains/gcc13_20261118/envs/gcc
export CC="$toolchain/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$toolchain/bin/x86_64-conda-linux-gnu-g++"
export PATH="$toolchain/bin:$PATH"
test -x "$CC"
test -x "$CXX"
export CUDA_VISIBLE_DEVICES=1
/root/userfolder/ctx-compressor/envs/reader/bin/python -u -B scripts/heldout_model_v2.py run
export CUDA_VISIBLE_DEVICES=''
exec /usr/bin/python3 -u -B scripts/heldout_diagnostics_v2.py v2
