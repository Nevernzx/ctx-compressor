# 任务 04：独立训练环境与策略模型

仅在 5880 的 `/root/userfolder/ctx-compressor` 创建训练环境、下载模型及写入缓存；3080 mirror 保存新脚本、文档与小型结果。任务 01–03 原产物、旧 `ctxc` / `reader` 环境与 `scripts/reader_environment.sh` 均不修改。本子任务只做 CPU 环境检查；训练与 GPU 1 基准由主 runner 执行。

## 隔离与磁盘

- 环境：`/root/userfolder/ctx-compressor/envs/train`，只读使用旧 ctxc 的 Python 3.11 执行 `venv --copies`，不向旧环境安装包。
- 模型：`/root/userfolder/ctx-compressor/models/Qwen3-1.7B`。
- `scripts/train_environment.sh` 复用任务 03 外置 cache 路径，补充 HF datasets、Swift、Matplotlib、WandB cache/config 路径；全部临时文件、pip cache、Python bytecode 和运行缓存落在 userfolder。
- helper 固定 GPU 1；安装、下载和 CPU 检查在 source 后显式 `CUDA_VISIBLE_DEVICES=''`。不安装 vLLM 或 flash-attn，attention 选择 SDPA。
- 清空所有大小写代理变量，`NO_PROXY=*`，pip 源固定清华；模型下载器使用 `urllib.request.ProxyHandler({})`，不用推理 API。
- root 盘保护读取新 `eval/sft_prep/preflight.json`：基线 **931307966464 bytes**，允许最终增量 **300000000 bytes**。新脚本没有硬编码旧任务的基线。

## 版本选择与官方依据

选择稳定发布 **ms-swift 3.12.6**（2026-02-28 发布），固定 **torch 2.8.0、transformers 4.57.6、tokenizers 0.22.2、peft 0.18.0、trl 0.24.0、accelerate 1.12.0、datasets 3.6.0**。该 3.x 维护版本的官方依赖范围是 `transformers>=4.33,<4.58`、`peft>=0.11,<0.19`、`trl>=0.15,<0.25`、`datasets>=3,<4`；固定组合符合这些范围。保留任务 03 已缓存的 torch 2.8.0 CUDA 12.8 wheel，避免改动 reader 或系统 CUDA。使用 numpy 2.2.6、scipy 1.17.1、pandas 2.3.3、bm25s 0.3.16（与旧 ctxc 一致，供 SFT 复用 src.build_traj）。最终传递依赖由 `pip freeze` 写入 `requirements.train.lock.txt`，复现安装优先读取锁文件。

已在 5880 直接读取并保存官方证据到 `eval/sft_prep/environment_sources.json`：

- [官方 PyPI 3.12.6 元数据](https://pypi.org/pypi/ms-swift/3.12.6/json)：wheel `ms_swift-3.12.6-py3-none-any.whl`，1029428 bytes，SHA256 `fce40e661c8520014748e12dd7eb5fe5a22bcc34bcd2b03968d8b91372f4a9a9`。
- [v3.12.6 官方 requirements/framework.txt](https://github.com/modelscope/ms-swift/blob/v3.12.6/requirements/framework.txt)，Git blob SHA `874ae2bb4604627f9d5903c2e5decaae971e324e`。
- [v3.12.6 官方 Qwen 模型注册](https://github.com/modelscope/ms-swift/blob/v3.12.6/swift/llm/model/model/qwen.py)，Git blob SHA `bb31c1e18fe0e9ba82031e00b59186642fbcfc30`，明确注册 `Qwen/Qwen3-1.7B`、`Qwen3ForCausalLM` 和 qwen3 template，要求 `transformers>=4.51`，与 4.57.6 一致。

## ModelScope 固定清单

`scripts/download_policy_model.py` 首次读取 [Qwen/Qwen3-1.7B 官方清单](https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-1.7B/repo/files?Revision=master&Recursive=true)，保存每个文件自己的固定 revision、大小、SHA256 及下载 URL；重跑使用冻结清单，不重新跟随 master。共 **13 个文件，4079450740 bytes**，包括两片 safetensors；权重 revision 为 `980712f58bdf09497308d37d0e30b535064cde04`。

先并行完成 ≤32 MiB 的 11 个小文件，随后下载权重。`.part` 只放模型目录，HTTP Range 校验 Content-Range；已有有效字节的进度不消耗有限失败重试预算。大文件下载每 30 秒检查 root 用量；每个文件先核验完整大小与 SHA256，才把 `.part` 改名为正式文件。`--verify-only` 只离线重算大小/SHA256，并检查目录没有非清单文件。

小文件于 `2026-11-17T12:21:43.896044+00:00` 就绪。新模型 `tokenizer.json` 与项目 `data/raw/qwen3_tokenizer/tokenizer.json` **逐字节一致**：SHA256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`，旧 ntok 计数无变化。已通知数据生成 agent 可在 CPU 使用本地 tokenizer。

## 执行记录

安装日志：`logs/task04_setup_train_env.log`，启动 / 完成时间和 PID 文件同名前缀。下载日志：`logs/task04_download_policy_model.log`。两进程通过 `nohup setsid` 脱离终端启动，实际执行均在 5880；下载与安装并行。

复现（5880 项目目录，长任务同样建议 nohup / setsid）：

```bash
bash scripts/setup_train_env.sh
source scripts/train_environment.sh
export CUDA_VISIBLE_DEVICES=''
.conda/envs/ctxc/bin/python -B scripts/download_policy_model.py --workers 3
.conda/envs/ctxc/bin/python -B scripts/download_policy_model.py --verify-only
```

## 环境已完成

- 安装 `2026-11-17T12:21:11Z` 至 `12:24:48Z`，**217 秒**；`pip check` 通过，最终锁文件 **152 项**。环境约 **7.4 GiB**，全部在 userfolder。
- 版本与选定组合一致，未安装 vLLM / flash-attn。旧 ctxc 仍找不到 torch，证明没有向旧环境安装训练依赖。
- 完成时 root Used **931313152000 bytes**，比本次 preflight 增加 **5185536 bytes**，低于 300 MB。
- CPU 导入 `swift.llm.get_model_tokenizer/get_template`、`swift.trainers.Seq2SeqTrainer`、Transformers、PEFT、TRL、Accelerate 均通过；`environment_cpu_smoke.json` 保存结果。
- 额外以随机初始化的一层微型 Qwen3 在 CPU 做一次前向/反向，验证 bf16 + SDPA + gradient checkpointing + LoRA r16/alpha32/dropout0.05/all-linear 的软件组合，loss 和梯度有限。未加载预训练权重、未做 optimizer step、未跑吞吐基准；前后 `torch.cuda.is_initialized()` 均为 false。
- 环境锁文件与 CPU 检查小结果已同步 mirror。

为供 SFT helper 直接复用 `src.build_traj` 的渲染与 oracle 函数，12:27 UTC 在 train 环境补入 bm25s 0.3.16；`pip check` 与模块导入再次通过。新 setup 已纳入同版本依赖，锁文件和 metadata 同步更新。

## 模型已完成并离线复核

- 官方清单获取：`2026-11-17T12:21:22.305475+00:00`；下载完成：`12:33:44.300811+00:00`，墙钟 **741.995 秒**。
- **13/13** 文件、**4079450740 bytes**，大小与 SHA256 全部吻合官方冻结清单。两个权重分片分别为 **3441185608** / **622329984 bytes**。
- 下载进程结束后再次运行 `--verify-only`；`12:34:05.240087+00:00` 完成，重算全部文件用时 **2.834 秒**。目录内容与清单精确一致，没有 `.part` 或其他额外文件。
- 最终 `model_manifest.json` SHA256：`c0990b13faa0b56fedd0528fe16c97b8b81d9f61b2b41484a4b2cdd65d91b676`；`requirements.train.lock.txt` SHA256：`f1d904777b0416ef60d00ecb6bab3e8e6032f4bb9b794f912e816044c15756e9`。
- 离线复核后 root Used **931322171392 bytes**，相对本次 preflight 增长 **14204928 bytes**，低于 300 MB。该数是环境子任务结束时的检查值，任务 04 最终占用由主 runner 在全流程结束时复核。
- 下载、安装和校验均已退出；只把新代码、锁文件、说明和 `eval/sft_prep/` 小 metadata 同步到 mirror。没有同步任何权重、语料或 SFT 大数据。

## 基准脚本只读兼容复核

受主 runner 委托，只读审查 `scripts/bench_sft.py` 与实际安装的 ms-swift 3.12.6 源码：所有 `TrainArguments` 字段均存在；`torch_dtype=bfloat16` 在 `_init_mixed_precision` 自动映射为 `bf16=True/fp16=False`；`all-linear` 由 `get_target_modules` 展开。`SwiftSft.run` 确实调用覆写的 `train(trainer)`，模板处于 train 模式，实际 `Seq2SeqTrainer.training_step` 包含 `template.forward_context`。100 步由 `max_steps=100`、callback 停止与第 100 步后拒绝进入 training_step 共同限制。最长样本在优化器状态驻留时做一次前向/反向，无 optimizer / scheduler step。没有发现兼容阻断；此复核只读源码，未启动 GPU 或运行 benchmark，实际训练由主 runner 完成。
