# 任务 01 交付给 Claude 的核查记录

任务已完成，范围止于任务书规定的框架、数据和验收。正式产物在 **5880 `/root/workplace/ctx-compressor`**；3080 仅有 `mirror/` 代码与文档。没有继续 PLAN §6，没有训练、GPU、模型权重或付费 API 调用。

## 验收结果

| 项目 | 实际结果 | 证据 |
|---|---|---|
| MuSiQue-Ans train / dev | 19,938 / 2,417，符合预期 | `data/raw/MANIFEST.json` |
| 全部下载文件 | 18/18 通过；1,057,170,141 字节；错误 0 | `eval/acceptance/data_verification.json` |
| 单元测试 | **80/80 通过**，失败/错误/跳过均 0 | `eval/acceptance/unit_tests.xml`、`logs/pytest.log` |
| 随机状态 | 两模式共 4,000 个可行状态 | `tests/test_build_traj.py` |
| 三任务两模式 | 12 题不重复；8/10/11 步；共 **58 个状态全通过** | `eval/acceptance/traj_samples.md`、`acceptance_checks.json` |
| 占位符、证据候选、预算、SFT 合法性、稳定编号 | 各 58 次通过 | 同上 |
| append 前缀不变 | 29/29 次通过 | 同上 |
| 独立复核 | 58 状态、489 候选重新分词均一致 | `eval/acceptance/independent_review.md` |
| 成本模型 | 165000/44250；20000/25000；16500/4425 | `tests/test_cost.py` |

验收 seed 为 **20261117**，每个任务检索打乱 seed 为 20261117 + 任务号。预算 **3000 token 为占位，正式值由 PLAN §6 试验定**。只有 dev 建了 BM25；train 仅用于原始字段与重叠核对。

train/dev 分别有 84,559 / 21,100 个精确去重段落；共有完全一致段落 3,697，占 dev 17.521327%。共有唯一标题 4,260，对应 dev 7,453 段（35.322275%）。dev token P50/P90/P99 为 103 / 205 / 363.01。Ans dev 没有检测到疑似其他引用、缺失/空支撑索引，跳过 0 题。统计分母和 15 条原始 decomposition 详见 README / DATA_CARD。

## Claude 核查清单

- [x] 正式目录新建前不存在；本机只写 mirror，下载只在 5880 直连且禁代理。
- [x] 独立 Python 3.11 环境及缓存位于项目内；已有环境未安装依赖；torch 未安装。
- [x] 三个核心文件及 PLAN 函数签名保留，变化逐条写入 NOTES。
- [x] 模型片段与 token 计数都使用“标题 + 换行 + 原正文”；编号稳定，append 前缀不变。
- [x] 噪声策略仅作用于重选状态，SFT 目标始终纯 oracle；“这是 Claude 的理解”已记录。
- [x] 输入列出全部问题、模式、预算、已用 token 和两组候选；目标 JSON 键顺序固定 add、drop。
- [x] 3 个四题任务覆盖 2/3/4 跳，各自两模式都完成逐步记录及自动检查。
- [x] MuSiQue、HotpotQA、2Wiki 保留原格式，所有下载文件与 MANIFEST 一一对应，无 `.part` 残留。
- [x] 镜像摘要与官方摘要区别清楚；MuSiQue 镜像许可误标采用官方 CC BY 4.0。
- [x] train/dev 重叠两种口径、去重数、dev token 分位数及占位符核对均入 README / DATA_CARD。
- [x] README、DATA_CARD、NOTES、HANDOFF、代码、测试和环境锁定文件同步至 mirror。
- [x] 所有内容定稿后生成 SHA256SUMS，并在 5880 全量验证；数据未回传本机。

建议人工重点看 NOTES 中的严格动作格式、非正预算拒绝、稳定平局顺序与噪声解释，再阅读 `traj_samples.md` 的任务 2 第 5 步和任务 3 的四跳链。源标注中看似间接的证据均保留原样，没有改写标签。

## 确切复现命令

先校验原始交付快照（不会重写报告）：

```bash
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh scripts/checksums.py --verify'
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh -m pytest tests -q'
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh scripts/review_acceptance.py'
```

离线重跑数据完整性核验、审计和三任务验收（会更新带生成时间的报告）：

```bash
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh scripts/verify_data.py'
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh scripts/audit_data.py'
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh scripts/acceptance.py --seed 20261117 --budget 3000'
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh scripts/data_card.py'
```

重新下载时只使用18文件白名单，不会下载模型权重或整个模型仓库。该操作会重新写入本项目数据文件，不要与其他下载进程同时写 MANIFEST：

```bash
ssh 5880 'cd /root/workplace/ctx-compressor; setsid nohup bash scripts/run_job.sh download_reproduce scripts/download_data.py --spec scripts/data_sources.json > logs/download_reproduce.log 2>&1 < /dev/null &'
ssh 5880 'cd /root/workplace/ctx-compressor && cat logs/download_reproduce.pid && tail -n 10 logs/download_reproduce.log'
ssh 5880 'cd /root/workplace/ctx-compressor && cat logs/download_reproduce.exit'
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh scripts/verify_data.py'
```

`run_job.sh` 保存 `.pid/.started/.finished/.seconds/.exit`；超过两分钟的其他命令也可套用同样的 setsid/nohup 方式。进程是否完成以退出记录和 PID 判断，不用 `pgrep -f`。重建环境、激活路径与依赖锁文件见 README；全部命令都在 5880 执行。

重跑后保留新的交付版本时，先同步更新 README 中 `DATA_AUDIT_BEGIN/END` 间的报告和 DATA_CARD，所有修改结束后再生成及验证摘要：

```bash
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh scripts/checksums.py && bash scripts/run_python.sh scripts/checksums.py --verify'
```

SHA256SUMS 覆盖代码、文档、全部 raw、验收结果和轨迹 JSONL；自身、运行日志和环境缓存不纳入。3080 没有 raw 数据，不能在 mirror 中全量验证该清单。

## 未取得的数据与失败来源

**没有缺失的必需数据。** MuSiQue 六文件与 HotpotQA 两个 dev 均取得。2Wiki dev 可以单独下载，故按任务书未获取 train/test；HotpotQA train 不在本任务范围。MuSiQue test 公开文件无答案和 decomposition，按原样保留，不能用于带标签验收。

源查找的失败 URL、报错、官方文件大小的已知边界及成功替代来源附在本文后部；最终数据来源及许可见 DATA_CARD。失败尝试没有触发代理或付费中转。

## 各步用时

时间均发生于 2026-11-17。下表的进程时间为实测，人工开发、寻源与文档时间为墙钟近似；并行阶段不能相加作为总耗时。

| 阶段 | 用时 / 时间范围 |
|---|---|
| 目录检查与初始化 | 正式目录创建时间 03:19:21 UTC；检查为秒级 |
| 独立环境与依赖 | 03:20:54–03:22:30 UTC，**96 秒** |
| 框架、测试编写与修正 | 初始化至最终测试 03:27:36 UTC，约 **8 分钟**，包含并行工作 |
| 最终80项单测 | pytest 展示 **0.51 秒**；JUnit总时间 **0.517 秒** |
| 寻源、下载、许可核查 | 约 **12 分钟**；最后文件完成 03:31:08.511182 UTC |
| 下载MuSiQue Ans train / dev | **92.025 / 11.474 秒**；Full train最慢127.809秒，文件并行 |
| 下载Hotpot两dev / 2Wiki dev | **26.390 / 26.999 / 23.720 秒** |
| 全部18文件最终完整性核验 | **7.198 秒** |
| 三任务两模式真实验收 | **7.525 秒**；包装器墙钟8秒 |
| 全量数据核对 | 首轮8.242秒；补充标题段落比例后的包装器墙钟 **9 秒** |
| 独立复核、文档整理、最终同步与摘要 | 框架验收后并行完成；以本次任务起止时间及文件时间核查，不伪造未单独计时的人工耗时 |

环境、下载和验收日志位于远程 `logs/`，每个下载文件的精确耗时在 MANIFEST。无需继续任何训练或可行性试验。

## 失败来源详情（已全部解决）


所有必需数据均已取得，下面是源查找过程中的失败，不构成剩余阻塞：

1. `https://www.modelscope.cn/api/v1/datasets?Search=musique&PageNumber=1&PageSize=20`（以及其 `Name`、`SortBy` 参数变体）返回 HTTP 500；`POST https://www.modelscope.cn/api/v1/datasets/list` 返回参数错误。阅读公开网页前端源码后找到可用只读搜索 API：`https://www.modelscope.cn/api/v1/dolphin/datasets?PageNumber=1&PageSize=20&Query=musique`。其中搜索必须用 `Query`，`Search/Name/Keyword` 无效；2Wiki 需完整检索词 `2WikiMultiHopQA`。
2. `https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/musique`、`LLM-Research/musique`、`extraordinarylab/musique`、`dgslibisey/MuSiQue` 均 404；`AI-ModelScope/hotpot_qa` 和 `AI-ModelScope/2WikiMultihopQA` 也 404。按可用公开镜像继续查找，没有用代理。
3. `https://raw.githubusercontent.com/Alab-NII/2wikimultihop/main/LICENSE` 首次探测 SSL 握手超时，正式下载又读取超时，已写入 MANIFEST 的 `failed_attempts`。通过同官方 GitHub 的不可变 blob API `https://api.github.com/repos/Alab-NII/2wikimultihop/git/blobs/342ce459a34de236d05b2be3f01e5f08f6c712e8`，设置 `Accept: application/vnd.github.raw+json` 后成功获取 11,337 字节的许可证。无任何代理或第三方中转。
4. 官方 MuSiQue `download_data.sh` 指向 `https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing`；按任务书已有不可达实测，优先寻找直连镜像。本任务未再次下载官方 zip，其官方字节大小未核实。Hotpot 官方 CMU、2Wiki 官方 Dropbox 同样依据任务书不可达信息选择镜像，不声称本次重新测过这些域名。



## 任务02：§6 可行性试验（无阅读模型部分）

已完成任务02的单次CPU运行：预算1876/2814/5628，自然命中91.343%，正文非证据段含答案比例4.214%；中档差值：reselect 任务末差 64.409 pp、单步按步差 21.066 pp；append 任务末差 85.717 pp、单步按步差 40.928 pp。全部150项测试通过（包含原80项），所有30286状态合法且不超预算，no_compress召回1、丢失率0。F1/EM未测。

正式运行命令（本次已执行一次；脚本拒绝覆盖已有实验文件）：

```bash
ssh 5880 'cd /root/workplace/ctx-compressor && setsid nohup bash scripts/run_job.sh feasibility scripts/feasibility.py --seed 20261117 --n-tasks 150 > logs/feasibility.log 2>&1 < /dev/null &'
```

核查命令（不重跑实验）：

```bash
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh scripts/checksums_task02.py --verify'
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh -m pytest tests -q'
ssh 5880 'cd /root/workplace/ctx-compressor && bash scripts/run_python.sh scripts/check_feasibility_independent.py'
```

需要用户另行决定复现时，可在原命令后添加`--output-dir eval/feasibility/reproduce_NAME --data-dir data/feasibility/reproduce_NAME`使用尚不存在文件的子目录。此次未执行任何复现或按结果调参。命令仅使用已有包；不得下载、安装或调用API。

实际运行时间：2026-11-17T06:19:11.309064+00:00至2026-11-17T06:25:34.857687+00:00，脚本记录 **383.549秒**。逐阶段耗时：

```json
{
  "dev_corpus_and_filter_seconds": 7.430762059986591,
  "one_time_dev_retrieval_seconds": 1.118157914141193,
  "dev_statistics_seconds": 1.8835783200338483,
  "train_reference_seconds": 2.3442048518918455,
  "all_closed_loop_configurations_seconds": 369.97132883011363,
  "input_hashes_seconds": 0.24404932418838143
}
```

进程数1，观测线程5（限制16）；峰值RSS 0.585 GiB；data/feasibility产物 45.912 MB，低于200MB预期与1GB停止阈值。pid/退出码/日志见logs/feasibility.*。测试JUnit报告在eval/feasibility/tests.xml，没有改动eval/acceptance/unit_tests.xml。

审核重点：预算由dev每任务全部去重gold的P90定档；泄漏排除整任务gold；任务末召回区别于当前可用gold的单步召回；丢失拆分分母及重复gold关联不能直接相加；缓存按全局pid及保留顺序计算并逐任务重置。查看报告末尾oracle比较项，若有个体反例其task/config/pid均保留，不能据此擅改oracle。独立复核与主实验输出在新eval/feasibility下。

旧SHA清单在specs/TASK01_SHA256SUMS；任务01的62个其他文件保持原摘要，三份主文档保留原字节前缀。当前SHA256SUMS使用新checksums_task02.py生成并--verify核对，纳入data/feasibility及所有新代码、小结果文件。本机没有data/feasibility，须在5880核验。

任务02独立复核完成：`eval/feasibility/independent_check.json`与`.md`记录5,687,736项比较、0差异，用时17.640秒；覆盖全部21,100段重分词、30,286步及2,850最终上下文，并抽查295条完整输入模板。所有12组oracle汇总比较和1,800次逐任务比较均无反例。此脚本不重新构造正式实验，但会更新自身独立报告的生成时间；重跑核查后若保留新的交付快照，最后执行`bash scripts/run_python.sh scripts/checksums_task02.py`再加`--verify`核对。


## 任务03：§6 阅读模型 F1（Qwen3-8B）

现有交付已包含20×600条全量预测和报告；所有命令在5880的 `/root/workplace/ctx-compressor` 执行。环境为 `/root/userfolder/ctx-compressor/envs/reader`，模型为 `/root/userfolder/ctx-compressor/models/Qwen3-8B`，缓存和临时文件为 `/root/userfolder/ctx-compressor/cache`。先校验当前快照：

```bash
cd /root/workplace/ctx-compressor
source scripts/reader_environment.sh
export CUDA_VISIBLE_DEVICES=''
.conda/envs/ctxc/bin/python -B scripts/checksums_task03.py --verify
.conda/envs/ctxc/bin/python -B -m pytest -q
```

测试仅用旧ctxc，不需要torch/vLLM；当前测试报告为 202 项全通过。上面的只读hash核验覆盖新旧交付，外置权重由已冻结model_manifest记录。不要为了复核再次运行会改写manifest时间字段的下载/离线verify脚本。

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


## 任务04：SFT 数据与吞吐基准


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


## 任务05后台执行快照（2026-11-17T18:32:11.516002+00:00）

任务05尚未完成。本快照时为1125/2500步、18000次训练样本暴露，已保存checkpoint-500与checkpoint-1000，无NaN/OOM。正式训练PID724147、独立等待器PID724148，均为nohup及独立session，训练结束后等待器会自动启动留出集闭环评测，不要重复启动。只允许物理GPU1。

训练目录：`/root/userfolder/ctx-compressor/runs/sft_20k_20261117/`；进度看`progress.json`，日志看`train.log`，接续状态看`continuation_status.json`。全部实操经`ssh 5880`，3080仅同步mirror。代码和CPU基线已就绪，436项全套测试通过；墙钟报告格式更新后受影响23项测试也通过。

剩余工作：完成2500步/两轮及五个checkpoint；等待自动C完成base与五个adapter的1494轨迹/模型闭环；运行`audit_model_rollouts.py`、`finalize_sft_artifacts.py`；生成最终报告与loss图，追加三份主文档，更新/慢核SHA256SUMS并同步mirror。不要在dev任务评测，不调用外部模型推理API，不以eval loss选checkpoint。最终磁盘门槛仍是相对Task05基线的原始df增量≤300000000 bytes；调查中未归因的宿主共享增量不扣除。最后按任务书§6汇报并输出任务05完成哨兵。

## 任务05运行中交接补充：最终根盘门槛可能失败

记录时间：2026-11-17T21:32:47.898324+00:00。任务尚未完成，禁止提前输出完成标记。训练PID724147、独立等待器PID724148；训练记录和日志均在/root/userfolder/ctx-compressor/runs/sft_20k_20261117。当前资源采样step=2076，root原始delta=342253568 bytes；已保存500/1000/1500/2000四个checkpoint，最后500步继续训练。

原始root基线仍为931440451584 bytes，最终硬限额300000000 bytes。20:42:47 UTC首次超限，来源未归因，不扣除未知增长，不清理其他任务文件。按任务书的最终限制继续B和C，C运行中仅记录root超限；最终仍硬验收。训练NaN/OOM门禁和C启动前GPU1<2GB门禁不变。自动等待器会在B确认为2500步/40000样本且退出后启动C，禁止重复启动。

收尾：先等六模型C完成、watcher退出、所有任务PID退出及GPU释放，跑独立model rollout audit。若最终root通过，照原scripts/finalize_sft_artifacts.py all；若唯一剩余失败为原始root超限，运行新增scripts/export_sft_resource_failure.py（CPU环境ctxc，CUDA_VISIBLE_DEVICES为空）。其成功导出仍exit1，必须核对stdout status=failed/export_complete=true和所有非root门槛通过；根盘回落则拒绝失败导出并指示严格finalizer。manifest.complete仅表示文件穷举，task_acceptance_status必须failed；报告保留全部指标和最佳checkpoint，整体failed。

失败分支报告顺序仍build→plot→build，预期返回1/0/1，不用&&中断；成功分支首次build可因旧图返回2，仍需plot再build。最终追加README/NOTES/HANDOFF，写并慢核SHA256SUMS，原四代产物/文档前缀保持不变，mirror只同步小文件。最终若root仍超限必须status=fail，不能把完整性通过当资源通过。

新增失败导出代码通过26项CPU fixture和独立只读复核；随后全套462项通过（包含原264项），最新证据eval/sft_run/tests.xml、final_tests_receipt.json。除非源码再变或发现新问题，不需要重复全套测试，只核对对应源码SHA。正式训练worker和C runner均未改；C runner SHA为649ecd4685a484d976a0f0015b7eb0780989f05809c157757176643097c7d7d5。详细决策见notes_task05_execution.md，最终全文追加到NOTES。


## 任务05运行快照（2026-11-18T01:22:21.830148+00:00，尚未最终验收）

- A 已选 per-device 1 × gradient accumulation 16，group_by_length=False；实测 5695.282197 token/s。
- B 已完成 20,000 条 × 2 epoch：2500 optimizer steps、40,000 次样本暴露，无 NaN/OOM；checkpoint-500/1000/1500/2000/2500 均已保存。训练循环实测 25774.874464 秒（含验证/保存）；worker 计时 26053.937670 秒（含其计时范围内的启动/编码，不等同完整 OS 进程寿命）。无需重跑训练。
- C 后台继续运行，已完成策略：{"base": 13956, "checkpoint-500": 13956}；当前 checkpoint-1000，已落盘 1494 个动作。runner PID 898553、engine PID 898637、watcher PID 898552，均属于本次既有 detached 流程。
- 原始模型独立全量 CPU 审计已通过，13956 个 prompt 全部重编码一致；其合并证据丢失率 1494/1494，任务末召回 0。收据：eval/sft_run/model_rollout_audit.base_only.provisional.20261118T011410095015Z.json。这是 base-only 收据，不代表六模型全部通过或完成选优。
- checkpoint-500 完整指标：证据丢失 1428/1494，任务末召回 0.4310829955408281；非法 JSON/ID/append-drop/超预算尝试分别为 1/37/0/9（各 n=13956），输出上限命中 0。尚不据此选优。
- 本地工具连接曾中断；重连实查远程同一后台流程继续推进，没有重启训练或推理。全量结果仍以远程 eval/sft_run/model_rollout_run.json 和 runs/sft_20k_20261117/continuation_status.json 为准。
- `/` 的原始增量仍高于 300,000,000 bytes，最终按实测硬门槛判断，未知来源占用不扣除。待六模型全部结束且 PID/GPU 释放后，再做最终独立闭环审计、资源验收/文件清单、完整报告、文档追加与 SHA 校验；不可提前宣布任务完成。


## 任务05最终交付：计算完成，资源验收未通过

记录时间：2026-11-18T04:02:22.428837+00:00。A/B/C 计算与独立审计均已结束；任务整体状态为 **failed**，原因是资源验收采样中 `/` 的原始增量为 **834,052,096 bytes**，超过 **300,000,000 bytes**。未扣除未知来源占用。只使用物理 GPU 1；16 个记录到的本任务 PID 已退出，终态 GPU 占用 375 MiB，本任务 GPU 进程为 0。

A/B/C及全量闭环审计已结束，现有模型和结果可直接使用。最佳adapter目录：`/root/userfolder/ctx-compressor/runs/sft_20k_20261117/checkpoints/checkpoint-2000`。模型原始权重位于 `/root/userfolder/ctx-compressor/models/Qwen3-1.7B`，采用原生LoRA加载，rank16。五checkpoint及训练日志保存在 `/root/userfolder/ctx-compressor/runs/sft_20k_20261117/`；逐步原始输出位于 `/root/userfolder/ctx-compressor/eval_rollouts/<策略>/steps.jsonl`。

最终主指标：500=1428/1494；1000=1334/1494；1500=1257/1494；2000=1238/1494；2500=1243/1494。独立排序为2000→2500→1500→1000→500；选中的2000合并任务末召回0.5795890633240051。完整10策略×模式×预算指标与中预算8行对照见 `eval/sft_run/sft_run_report.md`。

当前未满足的任务条件是根盘最终增量≤300MB。若外部存储状态改变，可复用本次结果执行严格资源核验、重新build报告并更新选择指针/校验清单；无需重新训练或生成。CPU命令在正式项目目录先source scripts/train_environment.sh、export CUDA_VISIBLE_DEVICES=""；严格资源核验为 `.conda/envs/ctxc/bin/python -B scripts/finalize_sft_artifacts.py all`。现有失败资源证据、pre_resource报告快照和完整模型审计都应保留。

任务01–04文件与README/NOTES/HANDOFF历史前缀继续保护，后续校验使用 `scripts/checksums_task05.py check --slow`。最终SHA256SUMS与mirror包含代码、文档、小结果与外置文件路径/SHA；大权重、数据、原始rollout保持在userfolder。


## 任务06　CPU 数据完成，训练正在等卡

5880 上等卡脚本 PID/SID 均为 1141340，PPID 为 1。请先读取 `/root/userfolder/ctx-compressor/runs/sft_20k_v2_20261118/continuation_status.json` 确认当前阶段。运行目录为其父目录，其中的 card_polls.jsonl 记录每十分钟的显存观测，resources.jsonl 记录训练/评测期间每 30 秒的资源和进度。若 status=failed，读取该状态及 training.log / evaluating.log 的首个异常。脚本每个阶段只执行一次。

状态进入 evaluation_complete 后，检查 worker_result.json、resolved_args_diff.json、eval/sft_run_v2/model_rollout_run.json、diagnostics_v2.json 以及 GPU 释放情况。用现有 ctxc Python、CUDA_VISIBLE_DEVICES 置空运行 `scripts/sft_v2_report.py build`，用现有 train Python 以 CPU 运行 `scripts/sft_v2_report.py plot`。报告使用五个 v2 checkpoint 的真实合并丢失率、末召回和训练步数排序，并引用已核对 SHA 的 v1 结果。最后执行 `scripts/checksums_task06.py resources` 记录任务归属文件与原始 df，`external` 记录外置大文件的路径与 SHA，更新文档后执行 `write` 和 `check --verify-old-external`。用 rsync -a 按文件列表同步 mirror/，排除 .git。当前等待阶段的报告与资源记录是启动时快照。


## 任务06　留出集闭环评测完成

5880 的流水线已于北京时间 2026-11-20 20:51 正常退出。v2 选中的 LoRA 位于 `/root/userfolder/ctx-compressor/runs/sft_20k_v2_20261118/checkpoints/checkpoint-2000`，基础模型位于 `/root/userfolder/ctx-compressor/models/Qwen3-1.7B`。adapter_model.safetensors 的 SHA-256 为 `7253eae08072782971fa8b7874e97fc3e76dee4ebd11ced5c014147090cf1f9c`。选点按 n=5 个完整 checkpoint、每个 n=1,494 条轨迹计算。

五个 checkpoint 的动作原文与逐步状态位于 `/root/userfolder/ctx-compressor/eval_rollouts_v2/<checkpoint>/steps.jsonl`。固定数据、训练输出和原始闭环文件的路径、大小及 SHA-256 见[外置清单](eval/sft_run_v2/external_manifest_v2.json)。[机器可读结果](eval/sft_run_v2/sft_v2_results.json)和[中文报告](eval/sft_run_v2/sft_v2_report.md)包含中档基线及全部 checkpoint 诊断。[测试记录](eval/sft_run_v2/final_tests_v2.json)及[资源记录](eval/sft_run_v2/final_resource_v2.json)保留了本次收尾的核验依据。


## 任务07　GRPO 准备　已完成，正式 GRPO 待定

- 任务07 的环境、状态构建、奖励插件、采样探针和冒烟训练均已在 0 号卡核验，结束时 0 号卡已释放。之后 0 号卡又被外部进程占用（2026-11-20 21:00 约 19.5 GB），1 号卡在任务06 收尾后空闲。
- 下一步：在 v1 checkpoint-2000 与任务06 选出的 v2 checkpoint-2000 中定 GRPO 起点；在起点上合并 LoRA、重建训练任务闭环状态、重跑采样探针（30 个任务约 10 分钟）；按 `eval/grpo_prep/projection.json` 定正式规模，写任务08。
- 外置产物在 `/root/userfolder/ctx-compressor/grpo_prep/`，路径与 SHA-256 见 `eval/grpo_prep/TASK07_SHA256SUMS`。


## 任务08　正式 GRPO 已启动

5880 的独立会话 PID/SID 为 1665710，启动核验时 F0 子进程 PID 为 1665731。实时状态在 `/root/userfolder/ctx-compressor/grpo_v2/pipeline_status.json`，包括阶段、子 PID 与每 30 秒更新的监控。每阶段命令及日志位于 `/root/userfolder/ctx-compressor/grpo_v2/logs/`；小结果在 `eval/grpo_v2/`，闭环原始记录在 `/root/userfolder/ctx-compressor/eval_rollouts_grpo_v2/`。流水线记录首次错误并停止，自动重试次数为 0。完整结果到齐后按任务书第 4 节收尾。


## 任务08　正式 GRPO 训练与留出评测完成

流水线于北京时间 2026-11-21 08:55:52 结束，状态为 `evaluation_complete`。选中的 LoRA 在 5880 的 `/root/userfolder/ctx-compressor/grpo_v2/train/checkpoint-500`，加载时使用合并后的基座 `/root/userfolder/ctx-compressor/grpo_v2/merged_v2_ckpt2000`。适配器权重 SHA-256 为 `e1f77a2077a14eb85ae03edd71f3800d42986faeaafbbcd3b038f33fad14c33c`。[选点记录](eval/grpo_v2/selected_checkpoint.json)

四个 GRPO checkpoint 和 F0 各完成 1,494 条留出轨迹、13,956 个决策步；逐步输出在 `/root/userfolder/ctx-compressor/eval_rollouts_grpo_v2/<策略>/steps.jsonl`。训练与采样的原始文件在 `/root/userfolder/ctx-compressor/grpo_v2/`。[完整清单](eval/grpo_v2/TASK08_SHA256SUMS)记录外置路径和哈希。[优化日志 CSV](eval/grpo_v2/training_curve.csv)与[训练核验](eval/grpo_v2/training_checks.json)保留每步数值和奖励复算结果。

## 任务09　阈值选取与留出集回执

5880 流水线于北京时间 2026-11-21 22:02:42 完成，训练状态打分与 H0、H1、H2 共 4/4 个阶段退出码为 0。输入为原始 Qwen3-1.7B 加任务06的 SFT v2 checkpoint-2000。τ=0.99 由训练状态的最高平均单步奖励选定（1.204460，n=22,158），三个留出运行均引用同一份[选点回执](eval/hedge_v2/train_selection.json)。

逐状态候选分数和九档开放环动作在 5880 的 `/root/userfolder/ctx-compressor/hedge_v2/train/state_scores.jsonl`。三个策略的逐步贪心原文、实际动作及加段记录在 `/root/userfolder/ctx-compressor/eval_rollouts_hedge_v2/{nohedge,tau_star,always}/steps.jsonl`，每个文件 n=13,956 步。[报告](eval/hedge_v2/hedge_v2_report.md) · [独立复算](eval/hedge_v2/final_audit.json) · [任务09校验清单](eval/hedge_v2/TASK09_SHA256SUMS)

## 任务10 后台流水线

5880 在 2026-11-21 23:13（北京时间）启动独立会话，流水线 PID 为 2378893，C 阶段 PID 为 2378915。状态文件 `/root/userfolder/ctx-compressor/dagger_v3/pipeline_status.json`，阶段日志 `dagger_v3/logs/`，资源、存活及训练步监控 `dagger_v3/resource_monitor.jsonl` 均在外盘。C、D 数据、D 训练、四次 E 评测、E 选优和两次 F 探针串行执行，失败会记录首个错误并退出。

`eval/sft_run_v3/launch_inputs.json` 固定交互阶段输入与回执。全量 C 仅用任务08的 400 个训练任务。E 仅用历史 249 个 train-heldout 任务，F 从未被本轮 DAgger 训练选中的状态中抽取 300 个。等 `status=evaluation_complete` 后按任务书 §8 汇总比较表、训练曲线和最终文件清单。

## 任务10 从 D_train 续跑

初次运行完成 C（22,158 步）和 D_data（训练 30,000 行、验证 1,000 行），在 D_train 启动前遇到显存占用。原状态及日志保存在原位置。续跑入口为 `scripts/run_sft_v3_pipeline.py --resume-from D_train`，独立会话 PID 为 2541782，状态文件为 `/root/userfolder/ctx-compressor/dagger_v3/pipeline_status_resume1.json`。本轮每个 GPU 阶段都按物理卡 1 每隔 5 分钟采样，连续 3 次显存低于 2,048 MiB 后启动，单阶段最长等待 36 小时。新阶段日志和监控位于 `/root/userfolder/ctx-compressor/dagger_v3/resume1/`。恢复回执、旧文件哈希及历史清单验证路径见 `eval/sft_run_v3/resume1_verification.json`。

## 任务10　续跑收尾与选中权重

5880 上的续跑于北京时间 2026-11-23 01:06:48 完成。外盘 `/root/userfolder/ctx-compressor/dagger_v3/pipeline_status_resume1.json` 登记 `evaluation_complete`。10/10 个阶段的回执路径、字节数及 SHA256 均已核验。所选适配器是 `/root/userfolder/ctx-compressor/runs/sft_v3_20261121/checkpoints/checkpoint-3000`，配合原始 Qwen3-1.7B 基座加载；权重 SHA256 为 `97c85137ad8531ef79c26cdb049630ac10ef88650c937f98ae8cf013a08820a5`。[选点回执](eval/sft_run_v3/selected_checkpoint.json)

四次评测的逐步记录在外盘 `eval_rollouts_v3/<checkpoint>/steps.jsonl`（各 n=13,956 步）。两次探针原始记录在 `dagger_v3/probe/{v2,v3}/sampler/probe_outputs.jsonl`（各 n=2,400 次生成）。锁定的 dev 集保留 451 个任务，首次策略使用仍遵循既有锁定记录。[报告](eval/sft_run_v3/sft_v3_report.md) · [测试回执](eval/sft_run_v3/final_tests.json) · [任务10清单](eval/sft_run_v3/TASK10_SHA256SUMS) · [测试集锁定记录](eval/test_dev_v1/LOCK.md)

## 任务11　正式 GRPO 归档

5880 上的流水线在北京时间 2026-11-23 14:37:32 写入 `evaluation_complete`。10/10 个阶段的退出码为零；阶段回执、字节数与 SHA256 已核验。加载所选策略时，基座为 `/root/userfolder/ctx-compressor/grpo_r2/merged_start`，其上加载新 LoRA `/root/userfolder/ctx-compressor/grpo_r2/train/checkpoint-125`。适配器权重 SHA256 为 `84612222c59733d2652dba101c526f448f3f9ea525ce80696489f4a2ebf05886`。[选点回执](eval/grpo_r2/selected_checkpoint.json) · [流水线核验](eval/grpo_r2/final_audit.json)

F0 和四个 checkpoint 的逐步动作位于 `/root/userfolder/ctx-compressor/eval_rollouts_grpo_r2/<策略>/steps.jsonl`（各 n=13,956 步）；训练状态、采样原文、数据集及优化日志保存在外盘 `grpo_r2/{states,filter,data,train}/`。[报告](eval/grpo_r2/grpo_r2_report.md) · [结果 JSON](eval/grpo_r2/grpo_r2_results.json) · [逐步训练 CSV](eval/grpo_r2/training_curve.csv) · [文件清单](eval/grpo_r2/TASK11_SHA256SUMS)


## 任务12　Qwen3-8B 留出集阅读评测

读者评测的主比较为 r2 对 SFT v3 F1 -0.019 个百分点、EM +0.318 个百分点（n=5,976 个配对）、r2 对 H1 F1 +1.145 个百分点、EM +1.355 个百分点（n=5,976 个配对）。CPU 测试通过 729/729 项，历史清单回执在 [history_post.json](eval/reader_heldout/history_post.json)。逐题预测保存在 5880 的 /root/userfolder/ctx-compressor/reader_heldout/predictions.jsonl，对应 [TASK12_SHA256SUMS](eval/reader_heldout/TASK12_SHA256SUMS)。任务13 G 通过这份清单校验 r2 的 5,976 条观测。[报告](eval/reader_heldout/reader_heldout_report.md) · [结构化结果](eval/reader_heldout/reader_heldout_results.json)


## 任务13 · GRPO 第三轮（2026-11-24）

任务13 已完成。原 pipeline_status.json 记录第一次 G 退出；grpo_r3/continuation_status.json 记录第二次 G 和全流程终态 evaluation_complete。所选适配器为 /root/userfolder/ctx-compressor/grpo_r3/train/checkpoint-250，读者预测在 /root/userfolder/ctx-compressor/reader_heldout/grpo_r3_checkpoint-250_try2/predictions.jsonl（n=5,976）。[结果](eval/grpo_r3/grpo_r3_results.json) · [训练曲线](eval/grpo_r3/training_curve.svg) · [文件清单](eval/grpo_r3/TASK13_SHA256SUMS)

任务13 的读者 F1（r3 50.709）待复测，见任务15。


## 任务14 · A、B 全量完成

API 子集预测和闭环轨迹位于 /root/userfolder/ctx-compressor/deepseek_api/full/。C 阶段等待读者配置决定。[结果](eval/deepseek_api/deepseek_api_results.json) · [清单](eval/deepseek_api/TASK14_SHA256SUMS)

任务12 / 13 的读者 F1 被任务16（关 cascade）取代，旧数保留。


## 任务17 · 归档交接

北京时间 2026-11-25 07:59:32 完成（n=10 阶段）。原始轨迹、预测、延迟和失败日志位于 5880 的 /root/userfolder/ctx-compressor/reranker_v1/。[运行配置](eval/reranker_v1/run_config.json) · [结果](eval/reranker_v1/reranker_v1_results.json) · [清单](eval/reranker_v1/TASK17_SHA256SUMS)

## 任务18 · 归档交接

北京时间 2026-11-25 02:28:45 完成（n=12 阶段）。checkpoint-500 的平均无关段数为 0.910（n=1,494 轨迹）。5880 的 grpo_r4/、eval_rollouts_grpo_r4/ 和 reader_t18/ 保留原始训练、闭环与阅读产物。[结果](eval/grpo_r4/grpo_r4_results.json) · [清单](eval/grpo_r4/TASK18_SHA256SUMS)


## 任务19 · 测试交接

锁定测试集已于任务19第一次使用。GPU 流水线在北京时间 2026-11-25 18:16:29 完成。5880 的 /root/userfolder/ctx-compressor/test_v1/ 留存轨迹、分数、请求、预测和阶段回执。[结构化结果](eval/test_v1/test_v1_results.json) · [签名清单](eval/test_v1/TASK19_SHA256SUMS)


## 任务20 · 真实 Agent 交接

正式推理在 UTC 2026-11-25T16:38:36.113557+00:00 的控制器退出前完成，后续经授权以现有归档做 CPU 收尾。5880 的 agent_v1/main_try2 留存全部正式逐轮对话和监控。[结构化结果](eval/agent_v1/agent_v1_results.json) · [交付清单](eval/agent_v1/TASK20_SHA256SUMS)

## 2026-11-26 任务21 交接（Claude）

- 任务21 已完成并核查：大文件在 `/root/userfolder/ctx-compressor/agent_ds/`，小结果在 `eval/agent_ds/`，签名清单是 `eval/agent_ds/TASK21_SHA256SUMS` 和 `TASK21_POSTHOC_SHA256SUMS`。
- 根目录 README 已改成项目总览，逐任务记录改写在 `docs/LAB_NOTEBOOK.md`。
- 事后分析文件里，`bridge_memory_posthoc.json` 的口径有误（HotpotQA / 2Wiki 没有 `gold_hop`），已作废，但保留文件。
