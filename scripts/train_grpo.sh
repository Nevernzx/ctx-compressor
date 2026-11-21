#!/usr/bin/env bash
# Task 07 (Claude): swift GRPO on a merged SFT model with a NEW LoRA; GPU 0 only.
# All knobs are env vars so Task 08 reuses this file. Provisional smoke defaults below.
set -euo pipefail
cd /root/workplace/ctx-compressor
source scripts/grpo_environment.sh
: "${MERGED:?merged SFT model dir}" "${DATA:?swift GRPO dataset jsonl}" "${OUT:?output dir}"
: "${LR:=5e-6}" "${BETA:=0.04}" "${NGEN:=8}" "${GEN_TEMP:=1.0}" "${TOP_P:=1.0}" "${LAMBDA:=0.3}"
: "${PER_DEVICE:=4}" "${GRAD_ACC:=8}" "${MAX_STEPS:=25}" "${SAVE_STEPS:=25}" "${EPOCHS:=1}"
: "${VLLM_UTIL:=0.30}" "${VLLM_MAX_LEN:=9216}" "${GRPO_SEED:=20261120}" "${SLEEP_LEVEL:=0}"
[ "$CUDA_VISIBLE_DEVICES" = "0" ] || { echo "GPU 0 only"; exit 2; }
used=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 2048 ] || { echo "NEED: GPU 0 busy ($used MiB)"; exit 3; }
[ ! -e "$OUT" ] || { echo "refusing to overwrite $OUT"; exit 4; }
mkdir -p "$OUT"
export CTX_GRPO_LAMBDA="$LAMBDA"
( while true; do nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
      | sed "s/^/$(date +%s),/" >> "$OUT/gpu0_monitor.csv"; sleep 10; done ) &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT
python -u -B -m swift.cli.rlhf \
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
  --beta "$BETA" --num_iterations 1 --num_train_epochs "$EPOCHS" --max_steps "$MAX_STEPS" \
  --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization "$VLLM_UTIL" \
  --vllm_max_model_len "$VLLM_MAX_LEN" --vllm_enforce_eager true --sleep_level "$SLEEP_LEVEL" \
  --gradient_checkpointing true --bf16 true \
  --save_steps "$SAVE_STEPS" --save_total_limit 100 --logging_steps 1 --log_completions true \
  --report_to none --seed "$GRPO_SEED" --data_seed "$GRPO_SEED" --output_dir "$OUT" --add_version false
echo GRPO_TRAIN_DONE
