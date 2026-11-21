# 任务 01 实施记录

1. 本次范围为 PLAN §5 框架、数据下载与 §3 数据核对；PLAN §2 文献精读、§6 可行性试验及后续训练/评测留待下一任务，任务 01 未声称方法差异已经验证。预算 3000 token 仅为验收占位值。
2. 2026-11-17 开始前通过 `ssh 5880` 确认 `/root/workplace/ctx-compressor` 不存在后才创建。根分区初始剩余约 25 GB；原始数据预计不足 10 GB，先保存在项目 `data/raw/`，下载期间检查空间；如低于 15 GB 则按任务书迁往新建的 `/root/userfolder/ctx-compressor_data/raw/`。
3. 为避免修改已有环境并使环境与缓存都位于正式目录，用 `CONDA_ENVS_PATH=$PWD/.conda/envs`、`CONDA_PKGS_DIRS=$PWD/.cache/conda/pkgs` 执行 `conda create -n ctxc python=3.11 pip`；150 秒内未成功则按任务书回退 `.venv`。最终实际环境见 README。
4. 所有下载都从 5880 直连；远程脚本清除大小写代理变量，运行时设置 `NO_PROXY=*`，不读取第三方中转配置。3080 只在 `mirror/` 写代码、规格副本及文档；原始数据、依赖包、索引与机器可读验收数据均留在 5880。


5. 按任务书指定，将片段渲染为 `标题 + "\n" + 原正文`；`text` 与 `ntok` 使用完全相同的渲染串。`paragraph_text` 另存原正文，去重键仍是原始 `(title, paragraph_text)`。BM25 对渲染串分词，标题仅出现一次。预算只约束被选片段的 token 数之和，不包括输入模板、编号标签和任务问题；与 PLAN 的片段预算口径一致。
6. 随机操作只使用显式种子的 `random.Random(seed)`。`simulate_task` 加可选 `seed=0`；`sample_tasks(examples, n_tasks, seed, questions_per_task=4)` 一次无放回抽样后切成任务，任务内与任务间都不重复，保留抽样顺序。重复数据集 ID 或数量不足时报错，不偷偷复用 dev 题目。
7. 原 `#0` 会错误引用最后一个答案，前向引用会原样漏入查询。现在 `hop_queries` 只接受 `1 <= k <= 已完成跳数`；缺失/空/无效 `paragraph_support_idx`、无效答案和证据不在语料库中均报 `ValueError`，调用方可整题跳过并记录原因。`实体 >> 关系` 保留原字符串作为检索查询，不擅自改写。其他引用格式由真实数据核对层识别并保守跳过。
8. `simulate_task` 的候选数取 `min(top_n, 语料库段数)`，修复合成小语料检索 top-20 越界；空语料或非正 top_n 显式拒绝。每步增加原始/替换后查询、题目 ID、零起始题号和跳号等审计字段，原有 new/gold/natural_hit 保留。
9. 原贪心排序在 token 数和已保留状态相同时依赖集合迭代顺序；最后再按稳定编号排序，使结果可复现。首要顺序仍是 token 从少到多，重选模式同长度仍优先已有证据。
10. `apply_action` 与奖励共用严格格式语义：add 必须存在，add/drop 必须为列表，编号必须为 Python int（拒绝 bool/float）；drop 缺省仍视为空列表。重复编号仍按集合处理，与 PLAN 相同。原 apply_action 对非字典、不可哈希元素等可能抛异常，现统一返回 None，让 rollout 按 PLAN 保持原状态。非法模式、非正整数预算也拒绝；零预算原会在奖励里除零。
11. `make_sample` 对 kept/new 各自去重并去掉 new 中已保留项，补齐直接调用时 kept 的稳定编号；正常 PLAN 轨迹行为不变。增加 pid_by_lid 供验收追溯，不渲染给模型。
12. `make_oracle_noise_policy(seed)`：这是 Claude 的理解。只在 reselect 的理想动作上从本步新返回中随机额外加入 1–3 个非证据段，预算不足时少加或不加；旧噪声仍由下一步纯 oracle 目标丢弃。append 返回纯 oracle。rollout 的 sft_target 始终独立使用 oracle_action(s)，不含噪声。
13. 实现 render_input(s) 与 render_target(s)：输入依次列全部问题、模式、预算、已用 token、已保留/新返回两组片段与编号/token 数；不泄漏 gold、oracle、recall、pid_by_lid。目标 JSON 紧凑输出且键固定 add、drop。
14. rollout 增加 action/action_valid/used_after/recall 审计字段，不改变 PLAN 预算、可获得金标准或非法动作保持原状的规则。预算必须正整数；append 已有状态若超预算则显式拒绝，因为在不能 drop 的规则下不可能修复。
15. reward.py 使用包内相对导入 `from .build_traj import apply_action`，以便从仓库根目录执行 `python -m pytest` 或导入 `src.reward`；奖励数值公式不变。eval/cost.py 与 PLAN 公式一致，倍率仍是参数；未增加供应商价格查询、GPU 或模型调用。
16. 测试直接使用任务书要求的奖励分支及 PLAN §5 的三组精确成本数值。随机状态测试使用 `random.Random(seed)`，10 个种子 × 200 个状态 × 2 种模式，共 4,000 个可行状态；状态生成器保证已保留 token 未超预算，符合真实 rollout 的前置条件。重选状态额外检查噪声策略合法性。
17. `audit_data.py` 对 train/dev 原始 `(title, paragraph_text)` 元组做精确集合统计，不做大小写或空白归一化。标题口径同时报告唯一标题交集比例、共享标题对应段落数及其占各 split 去重段落数的比例，以避免把同标题不同正文误称为精确泄漏。
18. dev 的 P50/P90/P99 针对去重语料段落，使用原始 `tokenizer.json` 的 `Tokenizer.encode(..., add_special_tokens=False)`；渲染格式统一调用 `render_paragraph`，分位数采用 NumPy 线性插值。输入模板、编号元数据不计入“保留片段”预算。
19. 对 `#k` 以外引用格式采用保守词法扫描：非标准 `#`、`[n]`、`$n`、`<n>` / ref/answer/hop XML 标签、花括号引用。命中是“疑似”而非语义判定；相关题跳过并在 JSON 报告记录 ID、查询、原因。不声称有限正则能证明不存在任意自然语言引用。`>>` 为关系分隔符，保留原样。
20. 真实验收固定默认 seed=20261117，跨任务无放回抽 12 道 dev 题形成 3 个任务；同一任务两种模式复用相同检索步骤。各任务检索打乱种子为 seed+任务号。验收只建 dev BM25 索引，预算默认 3000，明确标为占位，不执行 PLAN §6。
21. `data_audit.md` 内打印 2/3/4 跳各 5 条完整原始 question_decomposition，供主 README / DATA_CARD 嵌入。机器可读 `data_audit.json`、`acceptance_checks.json` 及 `data/traj/acceptance.jsonl` 仅保留在 5880；镜像同步 Markdown 文档及代码。
22. MuSiQue 使用原始 JSONL 格式的 ModelScope 镜像，HotpotQA/2Wiki 使用原始 JSON 数组，均无格式转换。镜像提供的 SHA256 与官方摘要严格区分：MuSiQue、HotpotQA、2Wiki 未取得官方原始包摘要，不声称官方逐字节一致；以完整解析、原始字段逐条核对、条数、镜像大小和摘要核验。Qwen3 六个 tokenizer/config/许可文件已核官方 ModelScope SHA256。
23. MuSiQue 镜像页面的 Apache 许可标签与官方 LICENSE 不一致，采用官方 CC BY 4.0。HotpotQA 数据采用官方 README 的 CC BY-SA 4.0，区别于代码 Apache 2.0；2Wiki 官方仓库 LICENSE 为 Apache 2.0，但未另行区分数据专用许可。许可出处与快照见 DATA_CARD。
24. MuSiQue test 公共文件缺少答案、decomposition 和支持标注，按原始格式保存，不补造标签。Full train/dev 的不可回答题有 null 支撑索引，分别为 29,341 / 3,675 个；允许这些真实字段，并与用于验收的 Ans dev 区分。Ans dev 全部 2,417 题可用，疑似其他引用、缺失/空支撑索引和跳过数均为 0。
25. 2Wiki dev 能独立下载，按任务书不取其 train/test；HotpotQA train 未下载。其余指定可选文件均取得。下载后根盘仍约 23 GB 空闲，原始文件合计 1,057,170,141 字节，故无需外部磁盘软链。
26. 2Wiki 官方 LICENSE 的 raw URL 读取超时，改用同官方 GitHub 的不可变 blob API 与 raw Accept 头成功取得；失败记录保留于 MANIFEST/HANDOFF。下载脚本遇到任何失败返回非零状态，随后独立 verifier 再检查全部 raw 文件，避免把下载进程退出当成完整性证明。
27. 增加不依赖生产轨迹构造函数的只读复核脚本，验证全部 58 个真实状态及 489 个不同候选的 token 数。人工审阅覆盖 2/3/4 跳；此项不属于 PLAN §6 的 150 任务试验，不确定正式预算、不调用阅读模型。
28. SHA256SUMS 在所有内容定稿后生成，覆盖代码、文档、全部 data/raw 文件（包含 MANIFEST）、验收结果与 data/traj，排除 SHA256SUMS 自身、环境/缓存/字节码和运行日志。MANIFEST 无自引用摘要，最终由 SHA256SUMS 覆盖；本机镜像不含原始数据，完整校验只在 5880 进行。
29. 最终文档审阅发现首条样例结构摘要原先只保留列表首项，无法展示 HotpotQA/2Wiki 的异构元组位置。已修正摘要函数，仅读取每个原始文件首条重建 MANIFEST 中的结构元数据；context 保留标题与句子数组、supporting_facts 保留标题与整数索引、evidences 保留三个字符串位置。未修改原始数据，之后重新核验18文件通过。MANIFEST 作为元数据文档同步到 mirror/data/raw，未同步任何原始样本。


## 任务02：§6 可行性试验（无阅读模型部分）

以下补充只适用于任务02，出处为`tasks/TASK_02_feasibility_nomodel_20261117.md`（镜像副本在specs）。任务01记录保留原样。

1. 按任务书§5照抄并标明出处：

> **装满规则（Claude 的理解，三个非 oracle 策略共用）**：按优先级 first-fit，放得下就放，放不下就**跳过、继续看下一个**，不在第一个放不下时停。

2. 按任务书§6照抄并标明出处：

> **任务末召回（Claude 的理解）**：最后一步的保留集合中含有的该任务全部金标准段数 ÷ 该任务全部金标准段数，再对任务平均。阅读模型看到的是这个集合，所以把它当作 §6 判据里"证据召回"的主口径；单步召回并列报告。

3. 任务索引采用0–149，simulate_task种子就是该索引；不是任务01的seed+任务号。sample_tasks仍使用20261117，四题拼接及top-20未改变。每任务只检索一次，所有19配置复用。no_compress以全体任务最大去重返回token数+1作为统一非约束预算，只跑append一次，两模式等价；其占用比例仅作描述。
4. train参照逐题用临时原始段落键映射校验hop_queries，不建train全局语料或BM25；只对150个抽中任务的gold文本分词。对“#9 Dream”相关题沿用既有拒绝行为，未修代码或原始数据，拒绝ID及原因写入结果；dev被拒数也如实报告。
5. 答案匹配主口径用原正文paragraph_text，与正文重复项一致；为明确标题边界，另报告title+换行+body敏感性口径。两者均在一次统计中完成，不用于改任务或定预算。去ASCII标点、小写、去a/an/the、合并空白后按词边界匹配，空答案不匹配。≥2词子集单独重算段级和跳级分母，排除的gold是整个四题任务的全部证据。
6. 归一化正文相同总数与其中“标题不同”子集分开给出；同标题的标点、冠词或空白异文也可能归一化相同。样例从正文主口径命中出现中用random.Random(20261117)抽20条，跨跳重复出现可以多次入样。
7. latest按当前kept+new的逆序排序；BM25用四题问题拼接及同一个dev全语料IDF，同分按当前呈现位置较新者优先。预计算每任务全语料分数只为复用，候选通过pid_by_lid查分；策略本身不读取gold/sft_target等。真实dev的20跳查询已经用测试与index.retrieve对照，同分允许顺序差异。
8. 单步召回按步均值与任务内先平均两种方式并列；任务末召回和丢失率使用所有唯一gold。按跳位/题位拆分时，任务分母是拥有该组证据的任务数，证据计数按gold出现关联；重复pid可关联多个跳。丢失方式按每任务去重gold，只有最终缺失才分类为never_kept或曾保留后drop，重新取回后最终保留的不计丢失。各组任务率不可相加。
9. 成本使用corpus pid保持跨步身份稳定，各任务重置缓存；相对no_compress在同一缓存条件下比较。输入长度对render_input(s)完整字符串重新分词，并列候选片段ntok和；no_compress没有压缩器，只报累计保留上下文长度。所有策略保留内容按既有apply_action的到达顺序执行。
10. 新增分位数手算测试首次暴露3.0000000000000004与3.0的浮点精确比较差异；仅将测试浮点断言改用pytest.approx，n仍精确比较，没有更改或舍入指标计算。
11. 单进程运行，包装器维持BLAS/OpenMP四线程并禁用tokenizer并行；另外将CPU affinity限制至最多16个CPU，每任务检查线程数≤16。每任务检查新产物总量，达到1GB立即停止。实际资源、大小和耗时见run_config。
12. 不改任何任务01核心文件、测试或验收数据。README/NOTES/HANDOFF只追加，SHA清单按任务书更新；新脚本checksums_task02.py扩展data/feasibility覆盖并核对62个不变文件和三份文档的原字节前缀，避免修改旧校验脚本。镜像仅同步代码、文档和小结果，不同步data/feasibility。
13. PLAN原四条判据的条件文字保留，包括其中预设的阈值；该阈值不是本次F1观测。所有阅读模型格写“未测（需阅读模型）”，不存在实测F1/EM数值，不作开工/不开工判断，不自行定义“接近”或开展第二轮调整。oracle保持原有在线贪心语义，个体/聚合反例若出现，保留记录并解释，不据此改代码凑数。


## 任务03：§6 阅读模型 F1（Qwen3-8B）

### 任务书§3原文：Claude的理解

下段逐字保留任务书 `TASK_03_reader_f1_20261117.md` §3，作为本次阅读协议；其中闭卷对照也是任务书标注的Claude理解。

## 3. 阅读协议（Claude 的理解，照此实现，不要改模板）

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

### 官方评分口径与环境来源

采用[MuSiQue官方metrics/answer.py](https://github.com/stonybrooknlp/musique/blob/main/metrics/answer.py)，已读取的Git blob为 `7f8249369b79954733bd53cd93a0edb737c9b1c0`。任一归一化token列表为空时，官方F1返回两列表是否相等，因此空/空EM=1、F1=1；空预测对非空标准答案为0。对answer及全部aliases分别取最大EM、最大F1，重复词按词频交集计算。

环境为独立Python 3.11 venv，使用旧ctxc的Python只读执行 `venv --copies` 创建；旧ctxc及其他已有环境未安装或修改torch/vLLM。任务01/02原有src、tests、scripts和数据不修改；任务03新增reader/qa_metrics等代码，三份主文档仅追加。只在5880直连下载、清空代理，pip使用清华源；3080没有下载数据或权重。

最初尝试在旧ctxc里做完整chat template预检时发现缺少jinja2；没有补装旧环境，改在新reader环境中以CUDA_VISIBLE_DEVICES为空做CPU预检并通过。纯Python的解析/评分测试仍在原ctxc执行。解析保留固定步骤，边界引号包含直引号和弯引号，结尾句号包含`.`和`。`；只删除完整小写think块，未闭合块保留以供格式检查。

选择vLLM 0.11.0，官方v0.11.0[CUDA文档](https://github.com/vllm-project/vllm/blob/v0.11.0/docs/getting_started/installation/gpu/cuda.inc.md)明确使用CUDA12.8 wheel；配套torch 2.8.0、xformers0.0.32.post1，与Ada和驱动580.95.05相容并经本地smoke验证。numpy 2.2.6满足numba0.61.2的<2.3限制。全部传递依赖锁定在requirements.reader.lock.txt。ModelScope官方master清单16文件、16,397,461,896 bytes，下载URL固定各文件revision，逐文件核对官方Sha256后再次离线复核。模型manifest固定摘要 `e4d78679a6b360e59592d3afb87ddde6aa34fb31dd8fbb6f064fc66cf0519a2e`。

环境、模型、TMP/PIP/XDG/HF/ModelScope/vLLM/Triton/TorchInductor/CUDA/CuPy/Ray/Conda/pycache等缓存全部在userfolder。完整变量、来源、安装1179秒、模型下载2442.489秒和离线校验37.540秒见[准备记录](notes_task03_environment.md)。下载的有效partial EOF改为Range继续，只有无进展/校验错误消耗重试预算；进程先退出后再从3改4线程续传，没有并发manifest writer。

### 本次smoke审阅及guard修正

最初40条smoke为oracle|reselect|mid和no_compress各20题（前5任务）。所有输入完整，最大no_compress prompt为32,382 tokens；解析与固定规则逐条一致，空答案0、think片段0。

no_compress/task4/question1（0-based）产生35词、48 tokens的解释性单行，finish_reason=length，末尾为“the 2001 Honda”。它未遵从短答案格式；固定规则没有可删除的think或Answer前缀，因此保留原文，EM=F1=0。该项占1/40=2.5%，也是唯一>20词样本。其余39条均stop且≤20词，但no_compress/task4/question3仍是一条16词解释句（23 tokens），解析只去末尾句号，EM=F1=0，不能把其余39条一概称为短语。

初始实现额外把任意finish_reason=length当作smoke失败，所以在生成任何全量预测前退出。按任务固定max_tokens=48，length属于输出预算命中；任务明确禁止的是输入上下文截断。修正这一额外guard后，保留原始smoke_predictions.jsonl、smoke_samples.md、smoke_checks.json和[初次运行配置](eval/reader/run_config_smoke_initial.json)，保存[独立审阅](eval/reader/smoke_review.json)，通过 `--reuse-reviewed-smoke` 核对摘要和输入后复用原40条小样进行校验，未重新生成smoke。随后只执行一次12,000题全量；没有改变模板、48-token上限、解析规则，也没有按得分或输出长度筛除、挑选或补跑样本。初始smoke进程仅记录到物理GPU1 UUID，未出现其他GPU进程归属。

一次尝试启动全量时，root的overlay全盘df计数相对基线出现341,770,240 bytes的单采样增量，触发磁盘guard；2秒后回落至约55.8MB。该尝试在第一条全量预测前停止，predictions为0行，GPU已释放。初次smoke的CPU准备期（GPU仍375MiB）也曾记录205,033,472 bytes的单采样尖峰。不能只凭回落认定是外部I/O；先停下调查，不把原始峰值删去或改低。调查处置：空闲120点的root增量约55.8–60.2MB；CPU-only完整prepare复检的1334个0.1秒采样约60.1–64.6MB，没有超过100MB，tempfile实际目录为userfolder/cache/tmp。没有定位具体写入者，未证实属于他人I/O。保留原始磁盘中止run_config与0行文件；新guard保留300MB门槛，超限时采集可见writable root FD元数据并暂停后续提交：在提交边界检查到本任务存在>100MB的root可写普通文件即停止；超过300MB持续60秒则停止；仅在连续两次、间隔2秒的root读数均≤300MB后记录incident并恢复。4个合成control-flow检查通过。最终验收依据成功完成、最终增量≤300MB和所有超限事件均已完成上述复查，不宣称全程峰值≤300MB。 [CPU诊断摘要](eval/reader/root_disk_cpu_diagnostic.json)保留实际采样范围和外置诊断trace摘要。

全量完成后的交付审阅又加固了暂停期间新增告警的再次检查，避免只检查进入暂停前的告警。本次所有实际告警已由独立验收检查，均无本任务大文件写入；该代码加固未重新运行推理、未修改任何预测或评分。实际执行源码保存在 `specs/task03_executed_run_reader.py.txt`，摘要见 `eval/reader/execution_source_receipt.json`；新增竞态检查的验证见 `eval/reader/disk_guard_late_alert_check.json`。

### 额外运行参数与完整性

`enforce_eager=True`使用eager执行，控制CUDA graph相关显存和编译开销；`enable_chunked_prefill=True`、`max_num_batched_tokens=8192`、`max_num_seqs=16`只控制调度与并发，长prompt分块预填充后完整参与注意力。它们没有把输入截断到8192；每条生成返回的prompt_token_ids都与提交序列逐项相同。tensor_parallel_size=1、prefix caching开启、seed=20261117、bf16、无YaRN；仅scripts/run_reader.py导入vLLM。模型generation_config中的采样建议不会覆盖显式temperature=0/max_tokens=48。

全量12,000条、每配置600题；报告重新解析评分并核对user原文SHA1，输入截断0、prompt+48超限0。最大prompt 32,382、no_compress最大 32,382，保留48 token后仍≤40,960。48-token输出上限命中 207/12,000（1.725%）；>20词答案 708/12,000、空答案 0/12,000，均原样纳入评分。所有分组n与逐题配对均保留，不定义“接近”“明显更低”阈值，不作开工决策。

物理 GPU 1（NVIDIA RTX 5880 Ada Generation，UUID `GPU-4318dc90-4126-038a-9402-1323533db94f`），记录到的本任务 GPU 进程均属于该 UUID；峰值显存 42,423 MiB。root Used 基线 931,190,407,168 bytes，推理收尾 931,367,436,288 bytes；最终增量 177,029,120 bytes（验收门槛 300,000,000 bytes）。另保留成功全量进程的全盘计数瞬时峰值 501,190,656 bytes及3次已复查事件；历史超限和未能归因的调查经过见NOTES。 全量耗时 1750.871 秒；初次smoke和成功全量两个进程的耗时合计 2170.673 秒（不含另次0预测中止和独立诊断），从首次smoke开始至成功全量结束、包含中间审阅/诊断等待的墙钟 3736.857 秒。测试 202 项通过。README/NOTES/HANDOFF追加前缀保护和旧任务文件保护由checksums_task03执行；本节仅描述本次已有结果，不反向修改任务02数据。


## 任务04：SFT 数据与吞吐基准


### 任务书“Claude 的理解”原文

- **改法（Claude 的理解）**：`hop_queries` 只把 `1 ≤ k ≤ 已完成跳数` 的 `#k` 当作引用并替换；超出范围的 `#k` 保留原文，不抛错，同时计数。在结果里记录受影响题的 id。

- **预算档（Claude 的理解）**：每条轨迹固定一档，用 `random.Random(20261118 + 任务序号)` 在 tight / mid / loose（1876 / 2814 / 5628）中均匀抽一档。

### 保守处理与边界

任务04 §2 明确授权对 `src/build_traj.py` 的 `hop_queries` 和旧越界测试作语义修复，作为“不改任务01–03产物”的两项特定例外；其余函数外字节、旧data/eval文件和三份文档原字节前缀继续保护。旧source/test全文件归档在 `specs/task03_pre_task04/`，旧Task03摘要在 `specs/TASK03_SHA256SUMS`。原题中的越界#k计数后保留，替换插入的answer文字不再次扫描；#9 Dream影响ID为 `2hop__444285_737459` 和 `2hop__169108_61463`，共2次原始引用保留。

修复后全19,938题可用。5%按题向下取整留996题，原raw顺序保持于两部分，再各自 `sample_tasks(..., seed=20261118)` 无放回拼4题任务。train剩余2题不拼任务；这是任务书允许的拼任务余数，不是SFT裁切或丢样本。语料/BM25使用全train段落，包括留出题段落；检索每任务仅模拟一次、两模式复用同一steps。任务序号、检索种子、噪声种子均是部分内0-based序号；预算Random(20261118+序号)固定一档。reselect状态走oracle-noise闭环，append走纯oracle，所有SFT目标均是纯oracle。具体余数ID和划分见 `data/splits/train_heldout_ids.json`。

实际ms-swift3.12.6模板采用 `qwen3 / enable_thinking=False / loss_scale=last_round / add_non_thinking_prefix=True`。n_input_tokens定义为labels=-100的前缀长度，n_target_tokens定义为监督后缀长度，二者之和是完整序列；assistant前4-token空think块和末尾2-token结束符计loss，user与角色头不计loss。未来HF推理generation prefix包含空think块，因此比masked前缀长4token；实际训练串和推理前缀的字符串及token IDs均逐一核对。该定义避免把4个think token重复计数。max_length向8对齐为8392，没有修改检索top20或预算来缩短输入。

吞吐基准使用两个独立reservoir：reselect seed20261118、append seed20261119，各800条，再用seed20261118打乱；两模式各自均匀无放回，不使用heldout。训练仅执行一次100步，未重复选种子或补步改善loss。LoRA r16/alpha32/dropout0.05、all-linear、bf16、SDPA、gradient checkpointing（use_reentrant=False）、batch1×累积16、lr1e-4/cosine/warmup0.03，无packing。ms-swift标准all-linear展开LoRA目标；最长样本检查放在100步之后，使Adam优化器状态真实驻留，仅前向/反向，不执行optimizer或scheduler step。

吞吐分子是实际参与训练的完整序列token，分母是同步计时的on_train_begin至on_train_end，含优化器和步级日志，不含模型初始化、CPU编码或单独最长样本probe。显存分别保存torch allocated/reserved峰值和nvidia-smi整卡占用峰值；整卡口径包含启动前375MiB背景占用。估时使用全train平均长度与实测token/s，并单次加实测初始化/编码开销；20k/40k尚未确定具体样本，假设长度分布同全train。墙钟与GPU占用按单卡独占保守取同值，不包括未来正式checkpoint/eval等未测开销，不据估时选择方案。

新train环境独立安装，旧ctxc/reader未更改；pip清华源、模型从ModelScope直连且逐文件官方SHA256及离线重算一致。tokenizer.json与任务01逐字节相同，ntok不变。没有安装flash-attn或vLLM到train环境。全部缓存/临时文件外置；只在5880下载，3080只同步mirror小产物，无代理和外部推理API。资源计数与检查详情见 `eval/sft_prep/final_resource_check.json`；未进行正式训练。


## 任务05：任务书中的 Claude 的理解（原文照录）

- **抽样（Claude 的理解）**：从 `data/sft/train.jsonl` 抽 20,000 条，reselect、append 各 10,000 条。
  - 每种模式内先保证每个任务至少有 2 条样本（不足 2 条的任务全取），剩下的名额从其余样本中均匀随机补齐；
  - 用 `random.Random(20261118)`；
  - 抽中的 (task, step, mode) 写进 `runs/sft_20k_*/sample_ids.json`，并统计每种 tier 的条数。

**选 checkpoint（Claude 的理解）**：
- 主指标：把 6 个"模式 × 预算"格合起来（每格 249 个任务）算证据丢失率，最低者胜出；
- 打平时，比合起来的任务末召回；
- 还打平，选步数靠前的 checkpoint。

实测结果与状态见 `eval/sft_run/sft_run_report.md`；执行决策和异常处理详见 `notes_task05_execution.md`。此段只确定任务口径，不提前宣称训练或评测完成。


## 任务05最终交付：计算完成，资源验收未通过

记录时间：2026-11-18T04:02:22.428837+00:00。A/B/C 计算与独立审计均已结束；任务整体状态为 **failed**，原因是资源验收采样中 `/` 的原始增量为 **834,052,096 bytes**，超过 **300,000,000 bytes**。未扣除未知来源占用。只使用物理 GPU 1；16 个记录到的本任务 PID 已退出，终态 GPU 占用 375 MiB，本任务 GPU 进程为 0。

### 口径、保守处理与证据

抽样沿用上方照录的任务书条目：两模式各10,000条，每模式覆盖4,735个任务，至少2条，再按固定Random补齐，抽样ID和tier统计已冻结。正式训练使用seeded随机打乱，group_by_length=False；A的8×2试验采用group_by_length=True并保留其排序影响记录。最长8条使用真实驻留优化器状态做前向/反向，额外optimizer更新为0；40GB按40,000,000,000 bytes保守判定。A选择、实际吞吐、估时与实际耗时均保留原始来源。

固定1,000条heldout SFT样本的5次eval loss仅用于曲线。五checkpoint选择依据6格、每格249条轨迹的合并证据丢失率；依次以合并任务末召回和更早step解平，使用实际未舍入数值。模型输入沿用任务04检索steps，反馈模型自己的保留状态；完整JSON严格解析，非法JSON/候选ID/append-drop/超预算按优先级互斥计数并执行NoOp。超预算尝试率取invalid_outputs，执行后状态未超预算的action_validity计数与它区别保留。greedy、max_tokens=256，vLLM原生LoRA、输入裁切0。

完整独立审计覆盖6×13956状态链与6份模型指标；base的13956个prompt另作全量重新编码，checkpoint-500首轮1494个prompt与生成token解码也已核对。主报告与预资源快照、独立排名和最终报告之间的SHA关系保存在final_report_audit_consistency.json；eval loss不参与排名。

A8曾在初始化阶段因缺length字段退出，0 optimizer step；原日志归档，补齐元数据后才进行该配置的30步测量。C首次启动因系统缺C编译器在warmup阶段失败，真实生成0条，非NaN/OOM；该次原始文件已归档。编译器安装在独立userfolder/toolchains/gcc13_20261118，未往ctxc或reader安装内容；仅本任务wrapper增加CC/CXX/PATH。完整工具链9887个普通文件路径、230个链接及784个目录已校验，74项关键校验另保留。历史462测试覆盖的82个源码均未变，新inventory_sft_toolchain.py不伪称被历史测试覆盖，其真实CPU全树运行和独立复核另有收据。

早期A1的ms-swift默认check_model逻辑存在模型仓库元数据GET尝试；既有日志不足以确认是否送达，相关审计如实保留。其后A4/A8/B显式check_model=False；未调用外部模型推理API。下载仅发生在5880的独立工具链准备，未使用代理；3080仅同步mirror小产物。

根盘计数采用原始基线931440451584 bytes和终态df，不以“可能为其他进程”扣减。此前仅将确认为本任务所有的pytest-7/8临时树迁至userfolder，逐文件/链接验证保留；没有清理未知或他人文件。C期间曾观测非本任务GPU进程共享占用，仅留元数据证据，未管理其他进程。详细时间线、来源边界与收据索引见[执行日志](notes_task05_execution.md)。

严格finalizer的300MB门槛保持不变。独立失败导出器仅在其余完成/PID/GPU门禁全部通过且根盘仍超限时，导出failed资源收据与稳定完整外置清单，返回码仍为1。最终报告无缺项，整体failed；这与模型计算完整、文件清单完整及审计通过分别记录。


## 任务06　reselect 噪声与资源计量

- **`/` disk-space rule (Claude's change; reason below)**:
  - Task 05 established that `/` is an overlay sharing the same host NVMe as `/etc/hosts`, so raw `df` growth includes writes from other users.
  - The criterion therefore becomes: **the total size of files this task creates or modifies under `/` (overlay) ≤ 300 MB**, measured with `find -xdev -newer <start marker file>` and similar.
  - Still record raw `df /` at start and end, side by side, but it is no longer pass/fail.

- **Policy (Claude's interpretation)**: add a parameter to `make_oracle_noise_policy`, `noise_range=(1, 3)`.
  - The default keeps the old behaviour (RNG call order unchanged).
  - v2 uses `noise_range=(0, 2)`, i.e. `want = rng.randint(0, 2)`.
  - Everything else stays the same: seed = task index; `sft_target` is always the pure oracle; append still uses the pure oracle.

v2 使用 src/oracle_noise_v2.py 中的同名策略工厂。旧模块与历史 SHA 保留。默认参数在 4,984 个已保存任务上复演出逐字节相同的 v1 reselect 轨迹。v2 只将范围改为含端点的 0–2，每个 split 的 task index 作为随机种子。append 和全部 SFT 标签沿用纯 oracle。

SFT 数据统计使用 step ≥ 1 动作前的保留区，空集按 gold 的子集计数。训练 reselect 的非空 drop 为 v1 39,536/39,546 和 v2 26,300/39,546。留出为 2,076/2,077 和 1,379/2,077。闭环诊断中的旧 gold 则按当前检索前已揭示的证据定义，冻结的 v1 对照可复现任务书所列全 gold 保留区 5,111/6,197。每项的分子与分母见 eval/sft_run_v2/。

等卡脚本按十进制 2 GB 换算 nvidia-smi 的 MiB 值，最多 1,907 MiB 才算一次空闲，两次相邻观测间隔 600 秒。每个训练样本 ID 和顺序、留出 loss 的 1,000 个 ID 及顺序均与任务05一致。配置比较记录设置、输入输出路径，以及旧运行的输入哈希和实测字段。正式启动后另比较 Swift 实际解析的参数。


## 任务06　留出集闭环评测完成

每条轨迹在最后一步少保留任何一段已揭示的证据，即计一次证据丢失；任务末召回则先按轨迹计算再取平均。选择顺序沿用任务05，依次看合并丢失率最低、合并末召回最高和 checkpoint 步数最早。评测使用同一批 249 个训练留出任务、2,326 个检索步骤。两版保存的教师轨迹不同，剥离教师动作后，任务身份、题目和检索步骤的序列哈希一致；首步提示逐字节和 token ID 均相同（n=1,494）。[输入核验](eval/sft_run_v2/heldout_v2_preflight.json)

中档重选模式下，已保留旧证据的逐步存活率为 v1 的 2,247/3,681 与 v2 的 8,742/8,764。第 1 步起、输出有效且保留区全为已知证据时，仍有删除动作的步数分别为 1,313/1,699 与 10/1,328。两项统计对应的模型分别为各版本选中的 checkpoint-2000。只追加模式按协议不能删除。完整分母和首错分类见[留出集报告](eval/sft_run_v2/sft_v2_report.md)。

后台等卡 130,812.48 秒，优化循环 24,081.22 秒，含启动、保存和退出的训练阶段墙钟为 24,365.28 秒（各 n=1）。训练记录覆盖 2,500 个优化步、40,000 次样本使用和 5 次固定子集验证。训练与评测共记录 1,140 次资源观测，最大相邻间隔 30.61 秒。


## 任务07　GRPO 准备

- Claude 的解释：奖励前置评测解析器，奖励为 −1 当且仅当评测判为非法输出；插件从生成的 token id 解码，因为 swift 给的补全文本末尾带 `<|im_end|>`。
- Claude 的解释：GRPO 在合并后的 SFT 模型上训练新 LoRA，这样 swift 关掉 adapter 得到的参照模型是 SFT 模型，而不是基座。
- Claude 的改动：`train_grpo.sh` 的采样温度变量改名为 `GEN_TEMP`，种子改为 `GRPO_SEED`。原名 `TEMP` 和临时目录环境变量冲突，导致第一次启动时 argparse 报错，失败目录已改名保留为 `smoke_train_failed_argparse_TEMP_collision`。
- 发现：替身模型下 77.6%（149/192）的 8 样本组奖励全相同（只追加 84–91%，重选 66–69%），只有 7.3% 的组里有采样结果超过 greedy。正式 GRPO 需要先定动态采样或状态筛选。
- 任务07 的哈希单独记在 `eval/grpo_prep/TASK07_SHA256SUMS`（含外置大文件路径），相对路径条目已并入顶层 SHA256SUMS。


## 任务08　正式 GRPO 已启动

任务08沿用任务07的评测、状态构造、采样和奖励实现，用独立入口绑定 GPU 1 与正式输出目录，保留历史脚本的内容和校验和。两版训练文件共有 4,735 个任务、44,281 个检索步骤。去除教师轨迹字段后其余内容逐条一致；训练与留出题目 ID 交集为 0（18,940 个训练题目 ID，996 个留出题目 ID）。

有效训练状态按每个提示 8 次采样的总体奖励标准差大于 1e-9 筛选；预定选择至多 2,000 个提示。正式评测首看紧预算下分模式的证据丢失率，再看六格合并丢失率与任务末召回率。


任务08阶段 C 续跑。vLLM 0.11.0 的默认 `top_k=0` 和登记值 `-1` 均关闭 top-k 截断。采样汇总保留登记值，并在 `generation.top_k_actual` 记录实际值。续跑入口为 `scripts/run_formal_grpo_pipeline.py --resume-from C`；启动前重新验证 F0 的 13,956 条原始动作及 B 的 22,158 条状态，分母与 SHA-256 均与原回执相符。首次失败的状态和 C 日志以 `.failed_C_topk` 后缀保存在外置运行目录。


## 任务08　正式 GRPO 训练与留出评测完成

A 的 30 条固定提示在权重合并前后，生成文本和 token ID 的差异均为 0/30。F0 留出闭环按完全相同输入状态对齐后，原始输出不同的有 98/13,601 步。F0 的六格合并证据丢失为 966/1,494，任务06选中模型为 956/1,494。两项结果分别列入报告。

新增片段数按有效动作步统计。任务06的 SFT v2 每步新增 1.000000 段（13,926 段 / 13,926 步），选中的 GRPO 每步新增 0.999785 段（13,931 段 / 13,934 步）。正式训练的 16,000 次采样对应 16,032 条奖励日志，其中末批 32 条重复记录；逐条复算共 0/16,032 差异。

Claude 只读失败分析（3080）：`claude_checks/grpo_failure_analysis.py`、`claude_checks/grpo_failure_out_20261121.txt`。
