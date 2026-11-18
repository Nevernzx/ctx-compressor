# 任务 03：环境与模型准备记录

本记录仅描述 reader 新环境、直接下载与校验；任务 01/02 的环境和数据不作修改。所有 HTTP/PIP 访问均由 `ssh 5880` 在远端发起，清空代理环境并使用 urllib `ProxyHandler({})`；3080 只保存 mirror 中的代码和小型记录。

## 路径与运行隔离

- 新环境：`/root/userfolder/ctx-compressor/envs/reader`。使用旧 ctxc 的 Python 3.11 只读执行 `venv --copies`，不向旧环境安装任何包。
- 模型：`/root/userfolder/ctx-compressor/models/Qwen3-8B`。
- 所有临时与缓存目录：`/root/userfolder/ctx-compressor/cache`。统一入口 `scripts/reader_environment.sh` 设置 TMP/TEMP/TMPDIR、PIP、XDG cache/config/data/state、HF、ModelScope、vLLM、Triton、TorchInductor、Torch extensions、CUDA、Numba、Ray、CCache、Conda、Python pycache 和 FlashInfer 路径；`PYTHONNOUSERSITE=1`。
- `CUDA_VISIBLE_DEVICES=1`，各线程环境变量为 4，关闭模型 Hub 网络访问和遥测。准备脚本只检查 `importlib.metadata`；不 import vLLM、不初始化 GPU。
- root 盘基线 Used = 931190407168 bytes，超过基线 300000000 bytes 时脚本停止检查。首次安装 guard delta = 4419584 bytes。

## 版本选择与官方依据

选择稳定标签 **vLLM 0.11.0**，使用其官方 PyPI CUDA wheel 对应依赖组合：PyTorch/torchaudio 2.8.0、torchvision 0.23.0、xformers 0.0.32.post1。

官方 tagged CUDA 安装文档明确默认二进制使用 **CUDA 12.8**，要求计算能力 ≥ 7.0；它与已确认的 RTX 5880 Ada、驱动 580.95.05 的 CUDA 13.0 驱动能力相容。本任务不安装或修改系统 CUDA。实际运行兼容性由后续 GPU 1 smoke 验证。

- [vLLM 0.11.0 官方 CUDA 文档](https://github.com/vllm-project/vllm/blob/v0.11.0/docs/getting_started/installation/gpu/cuda.inc.md)，远端 GitHub API 读取 blob SHA `9e64c6f2540af92c92501f9ac3438e62325a14d1`。
- [官方依赖 cuda.txt](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt)，逐项确认 torch/torchaudio/torchvision/xformers 版本。
- [官方 PyPI metadata](https://pypi.org/pypi/vllm/0.11.0/json)：Linux x86_64 wheel `vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl`，438217982 bytes，SHA256 `3861c75ff2b12e24f6d179ff5c084d791b42ded8675d76c8706697c79f68cd62`。
- 清华 pip 索引：`https://pypi.tuna.tsinghua.edu.cn/simple`。显式固定 transformers 4.57.6、tokenizers 0.22.2、bm25s 0.3.16、scipy 1.17.1；numpy 使用 2.2.6 以满足 vLLM 所需 numba 0.61.2 的 `<2.3` 约束。最终所有传递依赖写入 `requirements.reader.lock.txt`。

官方 raw.githubusercontent.com GPU 文档首次读取超时，随后改读同官方仓库 GitHub contents API 成功；没有使用代理或非官方包源。

## ModelScope 固定清单与下载

[官方文件列表 API](https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-8B/repo/files?Revision=master&Recursive=true)返回 **16 个文件，共 16397461896 bytes**，包括 5 个 safetensors 分片。首次 master 清单完整保存在 `eval/reader/model_manifest.json`，后续续跑复用同一清单，不再次跟随 master。各文件实际 URL 使用清单自己的固定 revision，清单 SHA256 与大小逐一核对；不以 git blob hash 代替文件 SHA256。

下载不调用 ModelScope SDK 或任何推理 API。初始 3 个下载线程，之后确认旧进程退出并保留所有 `.part`、固定 manifest，再以 `--workers 4` 恢复，让第 4 个分片提前开始；没有两个 manifest writer。HTTP Range 续传，`.part` 只存模型目录。观察到大文件首次无 Range 请求在 4 MiB/16 MiB 后结束，Range 请求可正常返回 HTTP 206 和正确 Content-Range；因此把可续传的有效字节进展与失败分开计数，仅无进展/校验失败消耗有限重试预算。已有 `.part` 保留并续传；哈希不符则删除该部分重新下载。每 30 秒检查 root 占用，失败返回非零。

进程由 setsid/nohup 启动，日志 `logs/setup_reader_env.log`、`logs/download_reader_model.log`；各自 PID 文件同名前缀。准备完成状态见下方追加记录。

## MuSiQue 官方评分核对

远端只读 [MuSiQue 官方 metrics/answer.py](https://github.com/stonybrooknlp/musique/blob/main/metrics/answer.py)，blob SHA `7f8249369b79954733bd53cd93a0edb737c9b1c0`。其 `compute_f1` 在任一归一化 token 列表为空时返回两列表是否相等，故空/空 **EM=1，F1=1**；这与某些早期 SQuAD v1 实现不同。已通知指标实现代理按任务指定的 MuSiQue 官方口径执行。

## 环境准备完成

安装从 `2026-11-17T06:54:53Z` 到 `2026-11-17T07:14:32Z`，耗时 **1179 秒**。`pip check` 通过，完整锁文件及 `eval/reader/environment_metadata.json` 已同步 mirror。vLLM 0.11.0、torch/torchaudio 2.8.0、torchvision 0.23.0、xformers 0.0.32.post1 与选择一致；旧 ctxc 的 `find_spec("torch") is None` 复查通过。安装阶段 root Used = 931212263424 bytes，比基线增加 **21856256 bytes**；环境目录约 9.4 GiB，位于 userfolder。

后续运行先 source `scripts/reader_environment.sh`；纯 CPU prompt 预检再覆盖 `CUDA_VISIBLE_DEVICES=''`。该子任务没有执行任何 GPU 初始化，GPU smoke 由主 runner 单独完成。setup 再次执行时优先使用已存在的完整 `requirements.reader.lock.txt`，保证传递依赖复现。

为提前解锁 CPU prompt 预检，`scripts/download_reader_small_files.py` 在主下载工作线程仍处理前三个权重分片时，按主 manifest 的固定 URL 独立补齐 index/tokenizer/vocab 四个小文件，全部 SHA256 通过。它不写主 manifest，单独保存 `eval/reader/model_small_files_receipt.json` 记录下载时间；主下载结束后会复用并补齐时间字段。模型原生配置确认 `max_position_embeddings=40960`、`torch_dtype=bfloat16`、`rope_scaling=null`；LICENSE 是 Apache 2.0。

环境 helper 后续补充 `CUPY_CACHE_DIR` 与 `HUGGINGFACE_ASSETS_CACHE`，也全部指到 userfolder/cache。复现按顺序运行（均在 5880 项目目录；长命令用 setsid/nohup 和独立日志）：

```bash
bash scripts/setup_reader_env.sh
source scripts/reader_environment.sh
.conda/envs/ctxc/bin/python -B scripts/download_reader_model.py --workers 4
.conda/envs/ctxc/bin/python -B scripts/verify_reader_model.py
```

离线 verifier 只在主下载成功退出后执行，重新读取全部 16 个文件核对官方 size/SHA256，确认模型目录正好只有清单文件、没有残留 part，并按独立小文件 receipt 补齐精确下载时间。它不访问网络、不初始化 GPU。

## 模型完成与冻结

- 下载清单获取：`2026-11-17T06:55:04.582258Z`；最后文件完成：`2026-11-17T07:35:47.070955Z`，墙钟 **2442.489 秒**（含断点续跑，不把各次进程耗时相加作为墙钟）。
- 16/16 文件、总计 **16397461896 bytes**，全部大小和 SHA256 与 ModelScope 固定清单一致。再次离线校验耗时 **37.540 秒**，于 `2026-11-17T07:36:35.720460Z` 完成；模型目录精确包含清单的 16 个文件，无 `.part` 残留。
- `eval/reader/model_manifest.json` 已冻结并同步 mirror，SHA256：`e4d78679a6b360e59592d3afb87ddde6aa34fb31dd8fbb6f064fc66cf0519a2e`；后续 runner 可记录该摘要，准备脚本不再改此文件。
- `requirements.reader.lock.txt` 共 156 项，SHA256：`8ca5918e9e314eb1c37508bae693848a2e4a101f5eb55852c185507855119998`。
- 下载结束及离线校验后的 root Used = **931233673216 bytes**，相对基线增加 **43266048 bytes**，低于 300 MB；root Available = 23463542784 bytes。userfolder 下环境约 9.4 GiB、模型约 16 GiB（du 向上取整）、缓存约 5.1 GiB，项目子目录总计约 30 GiB。
- 所有安装、下载和离线校验进程已结束；GPU 小样和全量推理由主 runner 接续，此准备子任务未初始化 GPU。任务 01/02 的环境与产物未作修改。
