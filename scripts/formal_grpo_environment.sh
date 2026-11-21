#!/usr/bin/env bash
# Task 08 environment. GRPO_CPU_ONLY=1 hides CUDA for CPU-only stages.
export GRPO_ROOT=/root/userfolder/ctx-compressor
export GRPO_ENV="$GRPO_ROOT/envs/grpo"
export GRPO_V2="$GRPO_ROOT/grpo_v2"
export GRPO_PREP="$GRPO_V2"
export GRPO_CACHE="$GRPO_PREP/cache"
export GRPO_GPU="${GRPO_GPU:-1}"
if [[ "$GRPO_GPU" != 1 ]]; then
    echo "Task 08 requires physical GPU 1" >&2
    return 2
fi
export CUDA_VISIBLE_DEVICES="$GRPO_GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "${GRPO_CPU_ONLY:-0}" == 1 ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*' PIP_CONFIG_FILE=/dev/null
export TMPDIR="$GRPO_PREP/tmp" TMP="$GRPO_PREP/tmp" TEMP="$GRPO_PREP/tmp"
export PIP_CACHE_DIR="$GRPO_CACHE/pip" PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
export XDG_CACHE_HOME="$GRPO_CACHE/xdg" XDG_CONFIG_HOME="$GRPO_CACHE/config"
export XDG_STATE_HOME="$GRPO_CACHE/state" XDG_DATA_HOME="$GRPO_CACHE/data"
export HF_HOME="$GRPO_CACHE/huggingface" HF_HUB_CACHE="$GRPO_CACHE/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_ASSETS_CACHE="$HF_HOME/assets"
export MODELSCOPE_CACHE="$GRPO_CACHE/modelscope" SWIFT_HOME="$GRPO_CACHE/swift"
export SWIFT_CACHE_DIR="$SWIFT_HOME" MPLCONFIGDIR="$GRPO_CACHE/matplotlib"
export VLLM_CACHE_ROOT="$GRPO_CACHE/vllm" VLLM_CONFIG_ROOT="$GRPO_CACHE/vllm_config"
export RAY_TMPDIR="$GRPO_CACHE/ray" FLASHINFER_WORKSPACE_BASE="$GRPO_CACHE/flashinfer"
export TRITON_CACHE_DIR="$GRPO_CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$GRPO_CACHE/torchinductor"
export TORCH_HOME="$GRPO_CACHE/torch" TORCH_EXTENSIONS_DIR="$GRPO_CACHE/torch_extensions"
export CUDA_CACHE_PATH="$GRPO_CACHE/cuda" CCACHE_DIR="$GRPO_CACHE/ccache"
export NUMBA_CACHE_DIR="$GRPO_CACHE/numba" CUPY_CACHE_DIR="$GRPO_CACHE/cupy"
export WANDB_DIR="$GRPO_CACHE/wandb" WANDB_CACHE_DIR="$GRPO_CACHE/wandb_cache"
export WANDB_CONFIG_DIR="$GRPO_CACHE/wandb_config" WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX="$GRPO_CACHE/pycache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export RAYON_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MAX_JOBS=4
GRPO_TOOLCHAIN="$GRPO_ROOT/toolchains/gcc13_20261118/envs/gcc"
export CC="$GRPO_TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$GRPO_TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++"
export PATH="$GRPO_ENV/bin:$GRPO_TOOLCHAIN/bin:$PATH"
mkdir -p "$TMPDIR" "$GRPO_CACHE"/{pip,xdg,config,state,data,huggingface/hub,huggingface/datasets,huggingface/assets,modelscope,swift,matplotlib,vllm,vllm_config,ray,flashinfer,triton,torchinductor,torch,torch_extensions,cuda,ccache,numba,cupy,wandb,wandb_cache,wandb_config,pycache}
