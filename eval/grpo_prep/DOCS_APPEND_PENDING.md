# 待任务 06 收尾后追加到 README / NOTES / HANDOFF 的任务 07 段落

## README（追加）
### 任务 07：GRPO 准备（2026-11-20，5880 GPU 0）
- 环境：`/root/userfolder/ctx-compressor/envs/grpo`（ms-swift 3.12.6 + vLLM 0.11.0），`scripts/grpo_environment.sh`。
- 流程：`build_grpo_states.py`（训练任务闭环状态）→ `make_grpo_dataset.py` → `merge_sft_lora.py`（SFT LoRA 合并）→ `train_grpo.sh`（新 LoRA，KL 参照 = SFT）→ 评测链路复用 `build_grpo_states.py --adapter`。
- 结果与推算：`eval/grpo_prep/grpo_prep_report.md`（替身 v1 checkpoint-2000，非正式结果）。

## NOTES（追加）
- Claude 的解释：奖励前置评测解析器，奖励 −1 ⇔ 评测非法；插件从生成 token id 解码（swift 补全文本带 `<|im_end|>`）。
- Claude 的解释：GRPO 在合并后的 SFT 模型上训新 LoRA，使 KL 参照为 SFT 而非基座。
- Claude 的改动：`train_grpo.sh` 的温度变量改名 `GEN_TEMP`（`TEMP` 与临时目录变量冲突导致首次启动失败）；任务 07 的哈希单独记在 `eval/grpo_prep/TASK07_SHA256SUMS`。
- 发现：替身模型下 77.6%（149/192）的 8 样本组奖励全相同，正式 GRPO 需决定动态采样 / 状态筛选。

## HANDOFF（追加）
- 下一步：任务 06 选出 v2 checkpoint 后，在 v1 / v2 中定 GRPO 起点；在起点上重跑探针（30 任务约 10 分钟）；按 `eval/grpo_prep/projection.json` 定正式规模（任务 08）。
