#!/usr/bin/env python3
"""Append the completed Task 04 delivery to three protected documents once."""
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
MARKER = '## 任务04：SFT 数据与吞吐基准'


def main():
    if PROJECT != Path('/root/workplace/ctx-compressor') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('run on 5880 with CUDA_VISIBLE_DEVICES empty')
    out = PROJECT / 'eval/sft_prep'
    baseline = json.loads((PROJECT / 'specs/TASK03_document_baselines.json').read_text())
    before = {name: (PROJECT / name).read_bytes() for name in ('README.md', 'NOTES.md', 'HANDOFF.md')}
    for name, data in before.items():
        entry = baseline[name]
        if hashlib.sha256(data[:entry['size_bytes']]).hexdigest() != entry['sha256']:
            raise ValueError(f'{name}: previous document prefix changed')
        if MARKER.encode() in data:
            raise ValueError(f'{name}: refusing duplicate Task 04 appendix')
    load = lambda name: json.loads((out / name).read_text())
    sft, bench, resources = map(load, ('sft_data_stats.json', 'bench_results.json', 'final_resource_check.json'))
    report = load('sft_prep_results.json')
    if (bench['status'] != 'ok' or report['status'] != 'ok' or not resources['root_limit_passed']
            or not resources['gpu_released'] or not resources['only_gpu1']):
        raise ValueError('complete benchmark and resource acceptance before appending success')
    test_cases = list(ET.parse(out / 'tests.xml').getroot().iter('testcase'))
    if not test_cases or any(c.tag in ('failure', 'error', 'skipped') for node in test_cases for c in node):
        raise ValueError('tests must all pass')
    lengths = sft['overall']['lengths']['total_tokens']
    paragraphs = {
        'README.md': f'''\n\n{MARKER}\n\n
任务04完成 PLAN §7 的训练准备和一次吞吐基准；没有启动正式训练、GRPO 或闭环评测，没有调用外部推理 API。#9 Dream 的两道 train 题恢复可用，全部 dev 查询与旧结果逐字节不变。

训练部分 **4,735 个任务、88,562 条 SFT 样本**，5% 留出 **249 个任务、4,652 条样本**；两部分共 n=93,214 条，全部使用 ms-swift 实际 Qwen3 模板编码，裁切和丢弃均为 0。完整序列 P50/P99/max 为 **{lengths['p50']:g}/{lengths['p99']:g}/{lengths['max']:g} token**，`max_length={sft['max_length']}`。

一次基准使用均匀抽样的 reselect/append 各800条，共1,600条，**100个 optimizer step**；实际吞吐 **{bench['tokens_per_second']:.2f} token/s**（完整未padding序列token），每条 **{bench['seconds_per_sample']:.4f} 秒**。首10/末10步loss均值 **{bench['first_10_loss_mean']:.6f}/{bench['last_10_loss_mean']:.6f}**。最长8,390-token样本另做一次带驻留Adam状态的前向/反向检查，额外optimizer step=0，未OOM。只使用物理GPU 1，结束已释放；root增量低于300 MB，最终读数见[资源检查](eval/sft_prep/final_resource_check.json)。

- [完整中文报告和估时表](eval/sft_prep/sft_prep_report.md)、[机器结果](eval/sft_prep/sft_prep_results.json)。估时只列全量×1/2/3 epoch、20k×2、40k×2，尚未选择正式训练方案。
- [模板及labels核验](eval/sft_prep/template_check.md)、[模型逐文件校验](eval/sft_prep/model_manifest.json)、[基准配置](eval/sft_prep/bench_run_config.json)、[基准结果](eval/sft_prep/bench_results.json)。
- [环境准备记录](notes_task04_environment.md)、[依赖锁定](requirements.train.lock.txt)、[旧结果保护核验](eval/sft_prep/reference_fix_checks.json)。旧202项和新增测试共 **{len(test_cases)} 项通过**。

大数据、模型、独立环境和基准输出均在 `/root/userfolder/ctx-compressor/`。轨迹 `data/traj_train/`，SFT `data/sft/`，环境 `envs/train`，模型 `models/Qwen3-1.7B`，基准 `runs/bench_task04_20261117`；正式项目仅保留代码、文档和小结果。外置文件路径、字节/行数和SHA256见各manifest，不进入项目SHA256SUMS。
''',
        'NOTES.md': f'''\n\n{MARKER}\n\n
### 任务书“Claude 的理解”原文

CLAUDE_QUOTES

### 保守处理与边界

任务04 §2 明确授权对 `src/build_traj.py` 的 `hop_queries` 和旧越界测试作语义修复，作为“不改任务01–03产物”的两项特定例外；其余函数外字节、旧data/eval文件和三份文档原字节前缀继续保护。旧source/test全文件归档在 `specs/task03_pre_task04/`，旧Task03摘要在 `specs/TASK03_SHA256SUMS`。原题中的越界#k计数后保留，替换插入的answer文字不再次扫描；#9 Dream影响ID为 `2hop__444285_737459` 和 `2hop__169108_61463`，共2次原始引用保留。

修复后全19,938题可用。5%按题向下取整留996题，原raw顺序保持于两部分，再各自 `sample_tasks(..., seed=20261118)` 无放回拼4题任务。train剩余2题不拼任务；这是任务书允许的拼任务余数，不是SFT裁切或丢样本。语料/BM25使用全train段落，包括留出题段落；检索每任务仅模拟一次、两模式复用同一steps。任务序号、检索种子、噪声种子均是部分内0-based序号；预算Random(20261118+序号)固定一档。reselect状态走oracle-noise闭环，append走纯oracle，所有SFT目标均是纯oracle。具体余数ID和划分见 `data/splits/train_heldout_ids.json`。

实际ms-swift3.12.6模板采用 `qwen3 / enable_thinking=False / loss_scale=last_round / add_non_thinking_prefix=True`。n_input_tokens定义为labels=-100的前缀长度，n_target_tokens定义为监督后缀长度，二者之和是完整序列；assistant前4-token空think块和末尾2-token结束符计loss，user与角色头不计loss。未来HF推理generation prefix包含空think块，因此比masked前缀长4token；实际训练串和推理前缀的字符串及token IDs均逐一核对。该定义避免把4个think token重复计数。max_length向8对齐为8392，没有修改检索top20或预算来缩短输入。

吞吐基准使用两个独立reservoir：reselect seed20261118、append seed20261119，各800条，再用seed20261118打乱；两模式各自均匀无放回，不使用heldout。训练仅执行一次100步，未重复选种子或补步改善loss。LoRA r16/alpha32/dropout0.05、all-linear、bf16、SDPA、gradient checkpointing（use_reentrant=False）、batch1×累积16、lr1e-4/cosine/warmup0.03，无packing。ms-swift标准all-linear展开LoRA目标；最长样本检查放在100步之后，使Adam优化器状态真实驻留，仅前向/反向，不执行optimizer或scheduler step。

吞吐分子是实际参与训练的完整序列token，分母是同步计时的on_train_begin至on_train_end，含优化器和步级日志，不含模型初始化、CPU编码或单独最长样本probe。显存分别保存torch allocated/reserved峰值和nvidia-smi整卡占用峰值；整卡口径包含启动前375MiB背景占用。估时使用全train平均长度与实测token/s，并单次加实测初始化/编码开销；20k/40k尚未确定具体样本，假设长度分布同全train。墙钟与GPU占用按单卡独占保守取同值，不包括未来正式checkpoint/eval等未测开销，不据估时选择方案。

新train环境独立安装，旧ctxc/reader未更改；pip清华源、模型从ModelScope直连且逐文件官方SHA256及离线重算一致。tokenizer.json与任务01逐字节相同，ntok不变。没有安装flash-attn或vLLM到train环境。全部缓存/临时文件外置；只在5880下载，3080只同步mirror小产物，无代理和外部推理API。资源计数与检查详情见 `eval/sft_prep/final_resource_check.json`；未进行正式训练。
''',
        'HANDOFF.md': f'''\n\n{MARKER}\n\n
任务04已完成数据、环境、模型和唯一100-step吞吐基准。正式训练样本量尚未选择；不要启动全量SFT、额外benchmark或GRPO。结果入口为 [sft_prep_report.md](eval/sft_prep/sft_prep_report.md)，每个统计附分母，附5个方案估时；[sft_prep_results.json](eval/sft_prep/sft_prep_results.json)供程序读取。

### CPU复核

在5880正式项目运行以下只读检查（pytest仅更新指定新结果时须重建SHA）：

```bash
cd /root/workplace/ctx-compressor
source scripts/train_environment.sh
export CUDA_VISIBLE_DEVICES=''
.conda/envs/ctxc/bin/python -B scripts/checksums_task04.py check --slow
```

`--slow` 重算外置轨迹、SFT、模型与split清单中的SHA256、大小及JSONL行数；默认check只核对外部manifest自洽/大小，不声称重读外部payload。`--verify-previous` 单独校验Task01/02/03旧文件和文档前缀。两项旧函数变化由历史完整SHA加函数外逐字节一致核验；不能用旧Task03 checker直接判新授权修改为损坏。

### 已执行命令与重跑边界

```bash
# 历史执行顺序；当前完整输出存在，勿直接重跑或删结果解除guard。
source scripts/train_environment.sh
export CUDA_VISIBLE_DEVICES=''
.conda/envs/ctxc/bin/python -B scripts/make_train_traj.py
"$TRAIN_ENV/bin/python" -B scripts/make_sft_data.py --workers 4
.conda/envs/ctxc/bin/python -B scripts/bench_sft.py prepare
export CUDA_VISIBLE_DEVICES=1
"$TRAIN_ENV/bin/python" -u -B scripts/bench_sft.py run
```

bench的started标记拒绝自动重试，max_steps=100和每步guard共同限制；没有保存adapter/模型checkpoint。外置 `runs/bench_task04_20261117/` 保存1600条抽样JSONL、最长样本、原始loss曲线、资源日志和ms-swift参数；小配置、全部100步loss记录与统计保存在 `eval/sft_prep/`。CPU报告脚本 `scripts/sft_prep_report.py` 可从现有文件重新核对SHA/行数和重建报告；若重建或更新tests结果，随后运行 `scripts/checksums_task04.py write` 和 `check --slow`。

文档追加脚本 `scripts/append_sft_documents.py` 只追加一次，校验Task03原前缀并拒绝重复marker。mirror只同步src/scripts/tests/specs、文档/锁文件、SHA256SUMS和eval/sft_prep小结果，不同步任何新轨迹、SFT、模型、环境或运行大文件。split ID文件虽小，正式副本在项目data/splits及外置data/splits，不属于本次mirror同步范围。
''',
    }
    book = (PROJECT / 'specs/TASK_04_sft_data_and_benchmark_20261117.md').read_text()
    quotes = [line for line in book.splitlines() if '（Claude 的理解）**' in line]
    if len(quotes) != 2:
        raise ValueError('expected exactly two verbatim Claude interpretation lines')
    paragraphs['NOTES.md'] = paragraphs['NOTES.md'].replace('CLAUDE_QUOTES', '\n\n'.join(quotes))
    for name, suffix in paragraphs.items():
        with (PROJECT / name).open('ab') as stream:
            stream.write(suffix.encode('utf-8'))
        if not (PROJECT / name).read_bytes().startswith(before[name]):
            raise AssertionError('append-only postcondition failed')
    print(json.dumps({'appended': list(paragraphs), 'previous_prefixes_preserved': True,
                      'tests_passed': len(test_cases)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
