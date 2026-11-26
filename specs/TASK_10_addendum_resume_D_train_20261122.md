# 任务10 补充单：从 D_train 续跑（2026-11-22 04:50，Claude）

## 0. 发生了什么
- 11-22 04:41 流水线在 D_train 的前置检查处退出：`D_train: GPU 1 busy (6608 MiB)`。D_train **没有启动**（status 里 pid=null），没有任何训练产物。
- C（22,158 步，hedged 6,565）、D_data（train 30,000 / eval 1,000 行）**都已 complete**，receipt 记录在 `/root/userfolder/ctx-compressor/dagger_v3/pipeline_status.json`。
- GPU 1 被用户另一条线 **MRI_SAM** 的全序列推理占用（medsam3 env；worker PID 2511814 / 2516615，queue 脚本 PID 2415185 / 2415783，目录 `/root/userfolder/medsam2_v2_20261116/full_seq_20261121/`）。该线自己估计 15:00–17:00 前后结束。**这是用户优先的任务，任何情况下都不要碰它的进程或目录。**

## 1. 要做的（只改续跑能力，不改任何预注册项）
1. 给 `run_sft_v3_pipeline.py` 加续跑入口（例如 `--resume-from D_train`），启动时必须先校验：
   - C、D_data 的 receipt（path / bytes / sha256）与旧 `pipeline_status.json` 一致；`steps.jsonl` 的 sha256 与 `dagger_rollouts.json` 一致；D 用到的训练 / 评估数据文件与 `sample_manifest.json` 一致；
   - D_train 及之后各 stage 的输出路径都不存在（不覆盖规则不变）。
   任何一项不符就拒绝启动并说明原因。
2. **旧的 `pipeline_status.json`、旧日志一律不改不删。** 续跑写新状态文件 `pipeline_status_resume1.json`（同 schema；C / D_data 标 `reused` 并带原 receipt），日志用新文件名。
3. GPU 等待：**每个用 GPU 的 stage（D_train、各 E、F）开始前**都按同一规则等：每 5 分钟读一次 GPU 1，连续 3 次 `memory_used < 2048 MiB` 才开始；等待期间状态写 `waiting_gpu` 和最近一次读数；单次最长等 36 h，超时记 failed。阈值 2048 MiB 不许改，不许用 GPU 0。
4. 不改：3,750 步 / 60,000 样本、Task 06 超参、ckpt 1000/2000/3000/final 的评估设置与选点规则、F 的探针设置、DONE 判据。
5. 加测试：receipt 不符拒绝、输出已存在拒绝、GPU 等待逻辑（mock 读数：忙→闲不足 3 次→连续 3 次闲）。全量 CPU 测试要全过。
6. 测完后在 GPU 1 上后台启动续跑（它会先进入 `waiting_gpu`），确认状态文件正常后打印 `WAITING: 任务10 续跑已挂上，等 GPU 1 空出`，然后像原规格 §8 一样等全部跑完写报告，最后 DONE 行 summary 里带"任务10"。

## 2. 禁止
- 杀任何非本任务进程；`pkill -f`；动 MRI_SAM 的目录或进程；commit；碰 `mirror/.git`。
- 在 `/` 下新增大文件（大文件只放 `/root/userfolder/ctx-compressor/`）。
