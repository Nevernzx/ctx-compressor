#!/usr/bin/env python3
"""Append final TASK_03 documentation once, preserving every existing byte.

This script performs no download, inference, report generation or test run.
It requires completed, validated reader results and the existing test report.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from reader_report import render_criteria_markdown

ROOT = Path(__file__).resolve().parents[1]
MARKER = "## 任务03：§6 阅读模型 F1（Qwen3-8B）"
DOCUMENTS = ("README.md", "NOTES.md", "HANDOFF.md")
# The investigation did not identify a writer. Preserve that uncertainty and
# the original interruption; never reinterpret a transient as proven external.
DISK_GUARD_RESOLUTION = (
    "空闲120点的root增量约55.8–60.2MB；CPU-only完整prepare复检的1334个0.1秒采样约60.1–64.6MB，"
    "没有超过100MB，tempfile实际目录为userfolder/cache/tmp。没有定位具体写入者，未证实属于他人I/O。"
    "保留原始磁盘中止run_config与0行文件；新guard保留300MB门槛，超限时采集可见writable root FD元数据并暂停后续提交："
    "在提交边界检查到本任务存在>100MB的root可写普通文件即停止；超过300MB持续60秒则停止；"
    "仅在连续两次、间隔2秒的root读数均≤300MB后记录incident并恢复。4个合成control-flow检查通过。"
    "最终验收依据成功完成、最终增量≤300MB和所有超限事件均已完成上述复查，不宣称全程峰值≤300MB。"
)

# Verbatim TASK_03 §3, including its heading and literal backslash-n strings.
TASK03_PROTOCOL = r'''## 3. 阅读协议（Claude 的理解，照此实现，不要改模板）

- **上下文**：每个任务 × 每种配置的最终保留集合。片段按 `keep_after` 的到达顺序排列，每段是 `title + "\n" + paragraph_text`（与压缩器看到的渲染串一致，不带编号），段与段之间用一个空行（`"\n\n"`）连接。
  - 语料文本用项目的 `build_corpus` 从原始 dev 重建，并用 `corpus_dev_ntok.jsonl` 的 sha1 逐条核对 pid。
- **4 道题分开回答**：每题一个独立 prompt。同一任务同一配置的 4 题共用同一段上下文，开 vLLM prefix caching。
- **user 消息逐字如下**（`{passages}`、`{question}` 为占位符）：

```
Answer the question based on the passages below. Reply with only the final answer (a short phrase such as a name, date, place or number), with no explanation.

Passages:
{passages}

Question: {question}
```

- **闭卷对照 `no_context`**（Claude 加的对照：用来区分"靠上下文答对"和"模型本来就知道"）：

```
Answer the question. Reply with only the final answer (a short phrase such as a name, date, place or number), with no explanation.

Question: {question}
```

- **生成设置**：
  - 只有一条 user 消息，无 system；`apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`。
  - greedy（temperature=0），`max_tokens=48`；dtype bf16；`max_model_len=40960`（模型 config 原生值，**不开 YaRN**）；`gpu_memory_utilization` ≤ 0.85。
  - 每个 prompt 的 token 数 + 48 必须 ≤ 40960。超出就停下报告，**不许截断上下文**。
- **答案解析**：
  1. 去掉 `<think>…</think>`；
  2. strip；
  3. 去掉开头的 `Answer:`（不分大小写）；
  4. 取第一个非空行；
  5. 去掉首尾引号和结尾句号。
  - 原始输出与解析结果都保存。另统计解析结果超过 20 个词的比例，作为格式检查。
- **评分**：MuSiQue 官方口径，即 SQuAD 归一化（小写、去标点、去 a/an/the、合并空白）后，对 `answer` 和全部 `answer_aliases` 取 EM / F1 的最大值。
  - `src/qa_metrics.py` 自己实现，配单元测试（手算样例，含 alias 取 max、空预测的情况）。
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_summary():
    tree = ET.parse(ROOT / "eval/reader/tests.xml").getroot()
    cases = list(tree.iter("testcase"))
    require(len(cases) >= 150, "The complete old/new test report is required")
    for tag in ("failure", "error", "skipped"):
        require(not list(tree.iter(tag)), f"Tests contain {tag}")
    return {"n": len(cases), "failures": 0, "errors": 0, "skipped": 0}


def validate_results(results, run):
    require(DISK_GUARD_RESOLUTION is not None, "Root disk transient investigation is still pending; do not append documents")
    require(run.get("completed") is True and run.get("stage") == "complete", "Reader run is not complete")
    require(run["predictions"] == results["n_predictions"] == 12000, "Expected 12,000 full predictions")
    require(len(results["configs"]) == 20 and all(row["n"] == 600 for row in results["configs"].values()),
            "Expected 20 configurations with 600 questions each")
    validation = results["validation"]
    require(validation["passed"] is True, "Reader result validation failed")
    for field in ("prompt_plus_48_over_limit", "input_truncated_count", "input_truncated_flag_missing_count", "finish_reason_missing_count"):
        require(validation[field] == 0, f"Nonzero integrity problem: {field}")
    require(run["input_truncations"] == 0, "Input truncation was observed")
    require(sum(run["full_finish_reasons"].values()) == 12000, "Incomplete finish-reason accounting")
    require(run["output_limit_hits"] == validation["generation_length_finish_count"], "Output-limit statistics disagree")
    require(run["generation"]["temperature"] == 0 and run["generation"]["max_tokens"] == 48
            and run["generation"]["enable_thinking"] is False, "Generation protocol changed")
    parameters = run["parameters"]
    require(parameters["max_model_len"] == 40960 and parameters["dtype"] == "bfloat16"
            and parameters["gpu_memory_utilization"] <= 0.85, "Model runtime protocol changed")
    require(run["gpu"]["physical_index"] == 1 and run["cuda_visible_devices"] == "1", "GPU 1 binding is missing")
    observed = run["observed_own_gpu_processes"]
    require(bool(observed) and set(observed.values()) == {run["gpu"]["uuid"]}, "Observed GPU process identity mismatch")
    require(not run["resource_monitor_problems"], "Runtime resource monitor reported a problem")
    limit = run["df_preflight"]["root_growth_limit_bytes"]
    require(run["final_root_delta_bytes"] <= limit, "Final root filesystem growth exceeded the configured limit")
    incidents = run["root_disk_incidents"]
    require(sum(len(incident["alerts"]) for incident in incidents) == run["root_disk_alert_samples"],
            "A root-disk alert has not been investigated/resolved")
    for incident in incidents:
        checks = incident["recheck_measurements"]
        require(len(checks) >= 2 and all(row["delta_bytes"] <= limit for row in checks[-2:]),
                "A root-disk incident lacks two consecutive below-limit rechecks")
        require(not any(fd["bytes"] > 100_000_000 for alert in incident["alerts"]
                        for fd in alert["own_writable_root_files"]),
                "A disk incident contains a large root file owned by this task")
    manifest = ROOT / "eval/reader/model_manifest.json"
    require(digest_bytes(manifest.read_bytes()) == run["model_manifest_sha256"], "Frozen model manifest changed")
    require(run.get("smoke_reused_without_regeneration") is True, "This delivery expects the documented reviewed-smoke reuse")
    require(run["smoke_checks"]["n"] == 40 and run["smoke_checks"]["length_finished"] == 1,
            "The recorded smoke result differs from the reviewed 40 samples")


def percentage(value):
    return f"{100 * value:.3f}%"


def build_sections(results, run, tests):
    validation, parameters = results["validation"], run["parameters"]
    max_prompt = max(row["prompt_tokens"]["max"] for row in results["configs"].values())
    max_no_compress = validation["no_compress_max_prompt_tokens"]
    cap_count = validation["generation_length_finish_count"]
    cap_rate = percentage(validation["generation_length_finish_rate"])
    long_count = sum(row["format_checks"]["long_answers_over_20_words"] for row in results["configs"].values())
    empty_count = sum(row["format_checks"]["empty_predictions"] for row in results["configs"].values())
    baseline = run["df_preflight"]["root_df_used_bytes"]
    final_used = run["df_after_reader"]["used_bytes"]
    resource_text = (f"物理 GPU 1（{run['gpu']['name']}，UUID `{run['gpu']['uuid']}`），"
                     f"记录到的本任务 GPU 进程均属于该 UUID；峰值显存 {run['peak_gpu_memory_used_mib']:,} MiB。"
                     f"root Used 基线 {baseline:,} bytes，推理收尾 {final_used:,} bytes；"
                     f"最终增量 {run['final_root_delta_bytes']:,} bytes（验收门槛 {run['df_preflight']['root_growth_limit_bytes']:,} bytes）。"
                     f"另保留成功全量进程的全盘计数瞬时峰值 {run['peak_root_delta_bytes']:,} bytes及"
                     f"{len(run['root_disk_incidents'])}次已复查事件；历史超限和未能归因的调查经过见NOTES。")
    key_names = ["oracle|reselect|mid", "oracle|append|mid", "no_compress", "no_context",
                 "bm25|reselect|mid", "bm25|append|mid", "latest|reselect|mid", "latest|append|mid"]
    score_rows = []
    for name in key_names:
        metric = results["configs"][name]
        escaped_name = name.replace("|", "\\|")
        score_rows.append(f"| {escaped_name} | {metric['n']} | {percentage(metric['em'])} | {percentage(metric['f1'])} |")
    readme = f"""\n\n{MARKER}

已在任务02固定的150个任务、600道不同问题上完成本地Qwen3-8B阅读：20种配置，各n=600题，共12,000条预测记录。复用原19种最终上下文，增加闭卷no_context；没有重跑检索、压缩或预算，没有训练或在线模型API调用。全量一次，报告均值和n，不计算CI，不作“开工/不开工”判断。

### 主要结果

| 配置 | n 题 | EM | F1 |
| --- | --- | --- | --- |
{chr(10).join(score_rows)}

{render_criteria_markdown(results)}
### 任务03目录与结果

正式代码、小结果和文档仍在 `/root/workplace/ctx-compressor`。独立reader环境为 `/root/userfolder/ctx-compressor/envs/reader`，模型为 `/root/userfolder/ctx-compressor/models/Qwen3-8B`，所有缓存和临时文件为 `/root/userfolder/ctx-compressor/cache`。启动前 source `scripts/reader_environment.sh`。

- [完整中文报告](eval/reader/reader_report.md)：20配置EM/F1、位置/跳数/证据齐全分组、prompt分布和逐题配对，所有分组附n。
- [机器结果](eval/reader/reader_results.json)、[运行参数与资源](eval/reader/run_config.json)、[原始40条小样](eval/reader/smoke_samples.md)、[小样审阅](eval/reader/smoke_review.json)。
- [模型manifest](eval/reader/model_manifest.json)：16文件、16,397,461,896 bytes，官方SHA256和离线复核均通过；[环境锁定](requirements.reader.lock.txt)与[准备记录](notes_task03_environment.md)。
- 逐题原始输出和评分为 `data/reader/predictions.jsonl`；该文件仅留在5880，3080的mirror只同步代码、文档和eval/reader小结果，不同步data/reader或权重。

### 验证与运行记录

旧新测试共 {tests['n']} 项全部通过（[测试记录](eval/reader/tests.xml)）；CPU报告重新解析、评分和核对全部12,000条元数据/user消息SHA1。输入截断0，prompt+48超限0；全配置最大prompt {max_prompt:,} tokens，no_compress最大 {max_no_compress:,}，加48后 {max_no_compress + 48:,}≤40,960。固定48-token输出上限命中 {cap_count}/12,000（{cap_rate}），原样保留并评分；>20词解析结果 {long_count}/12,000，空解析 {empty_count}/12,000。40条smoke中的一次输出上限命中及实现guard修正见NOTES，未重新生成smoke。

{resource_text} 全量生成耗时 {run['full_seconds']:.3f} 秒；本次全量进程（含准备、引擎初始化）{run['elapsed_seconds']:.3f} 秒。保留任务01/02所有不变文件和三份文档的原字节前缀；最终验收使用 `scripts/checksums_task03.py --verify`，覆盖新代码、结果和data/reader，外置权重由manifest独立记录。
"""
    notes = f"""\n\n{MARKER}

### 任务书§3原文：Claude的理解

下段逐字保留任务书 `TASK_03_reader_f1_20261117.md` §3，作为本次阅读协议；其中闭卷对照也是任务书标注的Claude理解。

{TASK03_PROTOCOL}
### 官方评分口径与环境来源

采用[MuSiQue官方metrics/answer.py](https://github.com/stonybrooknlp/musique/blob/main/metrics/answer.py)，已读取的Git blob为 `7f8249369b79954733bd53cd93a0edb737c9b1c0`。任一归一化token列表为空时，官方F1返回两列表是否相等，因此空/空EM=1、F1=1；空预测对非空标准答案为0。对answer及全部aliases分别取最大EM、最大F1，重复词按词频交集计算。

环境为独立Python 3.11 venv，使用旧ctxc的Python只读执行 `venv --copies` 创建；旧ctxc及其他已有环境未安装或修改torch/vLLM。任务01/02原有src、tests、scripts和数据不修改；任务03新增reader/qa_metrics等代码，三份主文档仅追加。只在5880直连下载、清空代理，pip使用清华源；3080没有下载数据或权重。

最初尝试在旧ctxc里做完整chat template预检时发现缺少jinja2；没有补装旧环境，改在新reader环境中以CUDA_VISIBLE_DEVICES为空做CPU预检并通过。纯Python的解析/评分测试仍在原ctxc执行。解析保留固定步骤，边界引号包含直引号和弯引号，结尾句号包含`.`和`。`；只删除完整小写think块，未闭合块保留以供格式检查。

选择vLLM {run['package_versions']['vllm']}，官方v0.11.0[CUDA文档](https://github.com/vllm-project/vllm/blob/v0.11.0/docs/getting_started/installation/gpu/cuda.inc.md)明确使用CUDA12.8 wheel；配套torch {run['package_versions']['torch']}、xformers0.0.32.post1，与Ada和驱动580.95.05相容并经本地smoke验证。numpy {run['package_versions']['numpy']}满足numba0.61.2的<2.3限制。全部传递依赖锁定在requirements.reader.lock.txt。ModelScope官方master清单16文件、16,397,461,896 bytes，下载URL固定各文件revision，逐文件核对官方Sha256后再次离线复核。模型manifest固定摘要 `{run['model_manifest_sha256']}`。

环境、模型、TMP/PIP/XDG/HF/ModelScope/vLLM/Triton/TorchInductor/CUDA/CuPy/Ray/Conda/pycache等缓存全部在userfolder。完整变量、来源、安装1179秒、模型下载2442.489秒和离线校验37.540秒见[准备记录](notes_task03_environment.md)。下载的有效partial EOF改为Range继续，只有无进展/校验错误消耗重试预算；进程先退出后再从3改4线程续传，没有并发manifest writer。

### 本次smoke审阅及guard修正

最初40条smoke为oracle|reselect|mid和no_compress各20题（前5任务）。所有输入完整，最大no_compress prompt为32,382 tokens；解析与固定规则逐条一致，空答案0、think片段0。

no_compress/task4/question1（0-based）产生35词、48 tokens的解释性单行，finish_reason=length，末尾为“the 2001 Honda”。它未遵从短答案格式；固定规则没有可删除的think或Answer前缀，因此保留原文，EM=F1=0。该项占1/40=2.5%，也是唯一>20词样本。其余39条均stop且≤20词，但no_compress/task4/question3仍是一条16词解释句（23 tokens），解析只去末尾句号，EM=F1=0，不能把其余39条一概称为短语。

初始实现额外把任意finish_reason=length当作smoke失败，所以在生成任何全量预测前退出。按任务固定max_tokens=48，length属于输出预算命中；任务明确禁止的是输入上下文截断。修正这一额外guard后，保留原始smoke_predictions.jsonl、smoke_samples.md、smoke_checks.json和[初次运行配置](eval/reader/run_config_smoke_initial.json)，保存[独立审阅](eval/reader/smoke_review.json)，通过 `--reuse-reviewed-smoke` 核对摘要和输入后复用原40条小样进行校验，未重新生成smoke。随后只执行一次12,000题全量；没有改变模板、48-token上限、解析规则，也没有按得分或输出长度筛除、挑选或补跑样本。初始smoke进程仅记录到物理GPU1 UUID，未出现其他GPU进程归属。

一次尝试启动全量时，root的overlay全盘df计数相对基线出现341,770,240 bytes的单采样增量，触发磁盘guard；2秒后回落至约55.8MB。该尝试在第一条全量预测前停止，predictions为0行，GPU已释放。初次smoke的CPU准备期（GPU仍375MiB）也曾记录205,033,472 bytes的单采样尖峰。不能只凭回落认定是外部I/O；先停下调查，不把原始峰值删去或改低。调查处置：{DISK_GUARD_RESOLUTION} [CPU诊断摘要](eval/reader/root_disk_cpu_diagnostic.json)保留实际采样范围和外置诊断trace摘要。

全量完成后的交付审阅又加固了暂停期间新增告警的再次检查，避免只检查进入暂停前的告警。本次所有实际告警已由独立验收检查，均无本任务大文件写入；该代码加固未重新运行推理、未修改任何预测或评分。实际执行源码保存在 `specs/task03_executed_run_reader.py.txt`，摘要见 `eval/reader/execution_source_receipt.json`；新增竞态检查的验证见 `eval/reader/disk_guard_late_alert_check.json`。

### 额外运行参数与完整性

`enforce_eager={parameters['enforce_eager']}`使用eager执行，控制CUDA graph相关显存和编译开销；`enable_chunked_prefill={parameters['enable_chunked_prefill']}`、`max_num_batched_tokens={parameters['max_num_batched_tokens']}`、`max_num_seqs={parameters['max_num_seqs']}`只控制调度与并发，长prompt分块预填充后完整参与注意力。它们没有把输入截断到8192；每条生成返回的prompt_token_ids都与提交序列逐项相同。tensor_parallel_size=1、prefix caching开启、seed=20261117、bf16、无YaRN；仅scripts/run_reader.py导入vLLM。模型generation_config中的采样建议不会覆盖显式temperature=0/max_tokens=48。

全量12,000条、每配置600题；报告重新解析评分并核对user原文SHA1，输入截断0、prompt+48超限0。最大prompt {max_prompt:,}、no_compress最大 {max_no_compress:,}，保留48 token后仍≤40,960。48-token输出上限命中 {cap_count}/12,000（{cap_rate}）；>20词答案 {long_count}/12,000、空答案 {empty_count}/12,000，均原样纳入评分。所有分组n与逐题配对均保留，不定义“接近”“明显更低”阈值，不作开工决策。

{resource_text} 全量耗时 {run['full_seconds']:.3f} 秒；初次smoke和成功全量两个进程的耗时合计 {run['smoke_and_full_process_seconds']:.3f} 秒（不含另次0预测中止和独立诊断），从首次smoke开始至成功全量结束、包含中间审阅/诊断等待的墙钟 {run.get('wallclock_from_first_reader_start_seconds', run['elapsed_seconds']):.3f} 秒。测试 {tests['n']} 项通过。README/NOTES/HANDOFF追加前缀保护和旧任务文件保护由checksums_task03执行；本节仅描述本次已有结果，不反向修改任务02数据。
"""
    handoff = f"""\n\n{MARKER}

现有交付已包含20×600条全量预测和报告；所有命令在5880的 `/root/workplace/ctx-compressor` 执行。环境为 `/root/userfolder/ctx-compressor/envs/reader`，模型为 `/root/userfolder/ctx-compressor/models/Qwen3-8B`，缓存和临时文件为 `/root/userfolder/ctx-compressor/cache`。先校验当前快照：

```bash
cd /root/workplace/ctx-compressor
source scripts/reader_environment.sh
export CUDA_VISIBLE_DEVICES=''
.conda/envs/ctxc/bin/python -B scripts/checksums_task03.py --verify
.conda/envs/ctxc/bin/python -B -m pytest -q
```

测试仅用旧ctxc，不需要torch/vLLM；当前测试报告为 {tests['n']} 项全通过。上面的只读hash核验覆盖新旧交付，外置权重由已冻结model_manifest记录。不要为了复核再次运行会改写manifest时间字段的下载/离线verify脚本。

### 首次启动与本次smoke复用

下面是尚无reader产物的隔离运行/输出副本中的启动流程。默认先40条smoke再全量；需要助理在引擎保留时读样本可加 `--await-smoke-review`，由助理记录本地smoke_review.json，不需要用户再次确认。GPU脚本自身source外置环境并显式设置CUDA_VISIBLE_DEVICES=1；启动前再次检查物理GPU1显存<2GiB。

```bash
cd /root/workplace/ctx-compressor
setsid nohup bash scripts/run_reader_job.sh > logs/reader.log 2>&1 < /dev/null &
```

本次最初使用 `--await-smoke-review`，因额外length guard在全量前退出；40条小样原文已保留并审阅通过。此后一次尝试因磁盘guard在第一条预测前停止，原配置与0行文件归档并完成调查；成功的唯一12,000条全量使用以下参数启动：

```bash
setsid nohup bash scripts/run_reader_job.sh --reuse-reviewed-smoke > logs/reader.log 2>&1 < /dev/null &
```

复用分支必须有passed审阅、原始smoke摘要匹配、初次run_config的全量predictions=0，并核对参数、模型manifest和输入文件摘要；它不会重新生成smoke。**当前已完成目录不能再次运行上述推理命令**：runner默认拒绝覆盖已有predictions，即使用复用参数也拒绝。真正重现实验须先安排新的隔离运行/输出副本，并针对固定正式路径审阅配置；不删除已有快照解除guard。

### CPU报告、文档与新摘要

下面的写入命令用于一个结果尚未发布的完整运行，按顺序执行。CPU报告不导入模型或分词器，重新核对12,000条user SHA1、解析、评分和分组；报告脚本、文档追加脚本都拒绝重复覆盖或追加。

```bash
source scripts/reader_environment.sh
export CUDA_VISIBLE_DEVICES=''
.conda/envs/ctxc/bin/python -B scripts/reader_report.py
.conda/envs/ctxc/bin/python -B scripts/append_reader_documents.py
.conda/envs/ctxc/bin/python -B scripts/checksums_task03.py
.conda/envs/ctxc/bin/python -B scripts/checksums_task03.py --verify
```

只检查旧任务保护可用 `scripts/checksums_task03.py --verify-previous`；只检查文档追加前置条件可用 `scripts/append_reader_documents.py --check-only`（已有任务03marker时同样拒绝）。权重不进入SHA256SUMS，数据预测进入；任务01/02不变文件和三份主文档原字节前缀持续核对。

主要结果：[reader_report.md](eval/reader/reader_report.md)、[reader_results.json](eval/reader/reader_results.json)、[run_config.json](eval/reader/run_config.json)、[smoke_samples.md](eval/reader/smoke_samples.md)。逐题预测位于 `data/reader/predictions.jsonl`，不复制到3080。mirror仅用rsync -a同步代码、文档、requirements.reader.lock.txt和eval/reader小结果，不使用--delete、不同步data/reader、模型、环境或cache。
"""
    return {"README.md": readme, "NOTES.md": notes, "HANDOFF.md": handoff}


def append_once(sections, check_only=False):
    baselines = load_json("specs/TASK02_document_baselines.json")
    marker = MARKER.encode("utf-8")
    with ExitStack() as stack:
        handles = {}
        before = {}
        # Lock every existing document before checking any marker or writing.
        for name in DOCUMENTS:
            handle = stack.enter_context((ROOT / name).open("r+b"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles[name] = handle
            before[name] = handle.read()
        for name in DOCUMENTS:
            require(marker not in before[name], f"Task03 documentation already appended: {name}")
            baseline = baselines[name]
            prefix = before[name][:baseline["size_bytes"]]
            require(len(prefix) == baseline["size_bytes"] and digest_bytes(prefix) == baseline["sha256"],
                    f"TASK_02 document prefix changed: {name}")
        if check_only:
            print(json.dumps({"check_only": True, "append_bytes": {name: len(sections[name].encode("utf-8")) for name in DOCUMENTS}}))
            return
        for name in DOCUMENTS:
            handle = handles[name]
            handle.seek(0, os.SEEK_END)
            require(handle.tell() == len(before[name]), f"Document changed concurrently: {name}")
            handle.write(sections[name].encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            require(handle.read(len(before[name])) == before[name], f"Original bytes changed: {name}")
    print("Appended TASK_03 to README/NOTES/HANDOFF; all existing bytes preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require(ROOT == Path("/root/workplace/ctx-compressor"), "Run this helper only on 5880 in the formal project directory")
    results = load_json("eval/reader/reader_results.json")
    run = load_json("eval/reader/run_config.json")
    validate_results(results, run)
    sections = build_sections(results, run, test_summary())
    append_once(sections, check_only=args.check_only)


if __name__ == "__main__":
    main()
