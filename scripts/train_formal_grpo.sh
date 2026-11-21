#!/usr/bin/env bash
# Formal Task 08 entry point. CLI arguments below follow the verified Task 07 run.
set -euo pipefail
cd /root/workplace/ctx-compressor
source scripts/formal_grpo_environment.sh
: "${MERGED:?merged SFT model dir}" "${DATA:?train jsonl}" "${OUT:?new train output directory}"
[[ "$CUDA_VISIBLE_DEVICES" == "1" && "$CUDA_DEVICE_ORDER" == "PCI_BUS_ID" ]]
[[ -x "$GRPO_ENV/bin/python" ]] || { echo "NEED: existing GRPO Python missing" >&2; exit 2; }
DATA=$(realpath -e "$DATA")
MERGED=$(realpath -e "$MERGED")
case "$DATA" in "$GRPO_V2"/*) ;; *) echo "dataset outside grpo_v2" >&2; exit 2;; esac
[[ "$MERGED" == "$GRPO_V2/merged_v2_ckpt2000" ]]
[[ "$(realpath -m "$OUT")" == "$GRPO_V2/train" && ! -L "$OUT" ]]
OUT=$(realpath -m "$OUT")
prompts=$(awk 'END {print NR}' "$DATA")
expected_steps=$((prompts / 4))
: "${LR:=5e-6}" "${BETA:=0.04}" "${NGEN:=8}" "${GEN_TEMP:=1.0}" "${TOP_P:=1.0}" "${LAMBDA:=0.3}"
: "${PER_DEVICE:=4}" "${GRAD_ACC:=8}" "${MAX_STEPS:=$expected_steps}" "${SAVE_STEPS:=125}" "${EPOCHS:=1}"
: "${VLLM_UTIL:=0.30}" "${VLLM_MAX_LEN:=9216}" "${GRPO_SEED:=20261120}" "${SLEEP_LEVEL:=0}"
[[ "$prompts" -le 2000 && "$expected_steps" -gt 0 && "$MAX_STEPS" -eq "$expected_steps" ]]
used=$(nvidia-smi -i "$GRPO_GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$used" -lt 2048 ]] || { echo "NEED: GPU $GRPO_GPU busy ($used MiB)"; exit 3; }
mkdir -- "$OUT"
export CTX_GRPO_LAMBDA="$LAMBDA" OUT
exec "$GRPO_ENV/bin/python" -u -B scripts/run_formal_swift_grpo.py \
  --rlhf_type grpo \
  --model "$MERGED" --model_type qwen3 --template qwen3 \
  --enable_thinking false --add_non_thinking_prefix true \
  --train_type lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --target_modules all-linear \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --dataset "$DATA" --remove_unused_columns false --dataset_num_proc 1 --dataloader_num_workers 0 \
  --external_plugins scripts/grpo_swift_plugin.py --reward_funcs ctx_step_reward \
  --num_generations "$NGEN" --temperature "$GEN_TEMP" --top_p "$TOP_P" --top_k -1 \
  --max_completion_length 256 --max_length 8392 \
  --per_device_train_batch_size "$PER_DEVICE" --gradient_accumulation_steps "$GRAD_ACC" \
  --learning_rate "$LR" --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --beta "$BETA" --num_iterations 1 --dynamic_sample false --num_train_epochs "$EPOCHS" --max_steps "$MAX_STEPS" \
  --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization "$VLLM_UTIL" \
  --vllm_max_model_len "$VLLM_MAX_LEN" --vllm_enforce_eager true --sleep_level "$SLEEP_LEVEL" \
  --gradient_checkpointing true --bf16 true \
  --save_steps "$SAVE_STEPS" --save_total_limit 100 --logging_steps 1 --logging_nan_inf_filter false --log_completions true \
  --report_to none --seed "$GRPO_SEED" --data_seed "$GRPO_SEED" --output_dir "$OUT" --add_version false
