# ctx-compressor：任务 01 框架与数据

本目录实现 PLAN v2 §5 的预算感知片段选择框架：对每次检索返回的段落输出 `{"add":[],"drop":[]}`，支持重选（reselect）与只追加（append）两种模式。奖励采用可获得证据召回，全部召回时才给予预算节省奖励；成本模型按相邻步骤的最长公共片段前缀计算缓存读写费用。

任务 01 只完成框架、数据下载、字段核对和三任务验收。独立小模型、可溯源选择、可验证密集奖励及含缓存成本是后续项目定位；与 [Provence](https://arxiv.org/abs/2501.16214)、[Acon](https://arxiv.org/abs/2510.00615)、[ContextBudget](https://arxiv.org/abs/2604.01664) 的差异尚待精读核实。唯一规格保存在 [specs/PLAN_v2.md](specs/PLAN_v2.md)，本次范围见 [任务书](specs/TASK_01_framework_and_data_20261117.md)。

## 机器、环境和目录

正式目录为 **5880 的 `/root/workplace/ctx-compressor`**；3080 的 `mirror/` 只保存代码与文档。原始样本不回传 3080；镜像中的 `data/raw/MANIFEST.json` 仅为来源、摘要和条数等元数据文档。5880 上的数据及分词器位于 `data/raw/`，未使用外部磁盘软链；空间与下载结果见 DATA_CARD 和 HANDOFF。

```text
src/build_traj.py       去重、BM25、逐跳查询、四题拼接、闭环展开及输入模板
src/reward.py           两种模式共用的证据召回奖励
eval/cost.py            含缓存成本模型
tests/                 奖励、oracle、噪声、稳定编号、闭环和成本单元测试
scripts/               环境、下载、数据核对、验收、校验和工具
data/raw/              原始数据与 tokenizer（仅在 5880）
data/raw/MANIFEST.json  来源 URL、下载时间、大小、SHA256、条数与验证结果
data/traj/             验收轨迹 JSONL（仅在 5880）
eval/acceptance/        人工可读验收文档与远程机器可读检查结果
specs/                 原始任务书与唯一规格的只读副本
```

独立 conda 环境位于 `.conda/envs/ctxc`，Python **3.11.18**。已安装 bm25s、transformers/tokenizers、NumPy、SciPy、pytest 及其依赖；未安装 torch。完整版本见 [requirements.lock.txt](requirements.lock.txt)。激活方式：

```bash
ssh 5880
cd /root/workplace/ctx-compressor
source /root/anaconda3/etc/profile.d/conda.sh
conda activate /root/workplace/ctx-compressor/.conda/envs/ctxc
```

也可直接使用 `bash scripts/run_python.sh ...`，该包装器选择本项目环境，禁用代理、GPU 与在线 Hugging Face 请求，并限制 CPU 线程。环境重建在 5880 运行 `bash scripts/setup_env.sh`；首次创建环境时已清除代理变量，conda 超时则回退独立 venv，实际本次无需回退。重建精确依赖时在激活后执行：

```bash
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NO_PROXY='*' no_proxy='*' PIP_CACHE_DIR="$PWD/.cache/pip" python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.lock.txt
```

## 测试和真实数据验收

本次 **80 / 80 单元测试通过**（含 4,000 个随机可行状态）；三个任务分别为 8、10、11 步，两模式合计 58 个状态全部通过自动检查。

以下命令在 5880 的正式目录执行：

```bash
bash scripts/run_python.sh scripts/checksums.py --verify
bash scripts/run_python.sh -m pytest tests -q
bash scripts/run_python.sh scripts/audit_data.py
bash scripts/run_python.sh scripts/acceptance.py
```

先校验收到的交付快照，再复现计算。审计和验收报告包含生成时间，重跑后会改变其摘要；若将重跑结果作为新交付版本，应同步更新 README 中的审计报告、运行 `scripts/data_card.py`，最后重新执行 `scripts/checksums.py` 生成清单并用 `--verify` 核对。完整命令见 HANDOFF。

验收仅用 MuSiQue-Ans dev 的段落建语料库与 BM25 索引。固定种子抽取互不重复的 12 题，分成 3 个四题任务，按题和跳的原顺序执行；同一个任务分别运行 reselect 与 append。预算 **3000 token 为占位，正式值由 PLAN §6 试验定**。输出见 [三任务两模式轨迹](eval/acceptance/traj_samples.md)；原始/替换查询、证据片段、自然命中、编号、oracle 动作、执行结果与 recall 都逐步列出。

数据核对报告见 [eval/acceptance/data_audit.md](eval/acceptance/data_audit.md)。该报告全文附在本文后部，以便直接核查每种跳数的 5 条分解问题。

[独立复核](eval/acceptance/independent_review.md) 从原始 dev 重新核对了全部 58 个状态，并对其中 489 个不同候选片段重新分词，结果一致。

## 片段、状态和 SFT 约定

模型可选片段采用 **`标题 + "\n" + 原正文`**。Qwen3-1.7B 的本地 `tokenizer.json` 对这个完全相同的字符串分词，`add_special_tokens=False`；预算为被保留片段 token 数的和，不含问题、编号标签和输入模板。BM25 也索引同一渲染串，标题只计一次。段落去重键保留原始 `(title, paragraph_text)`，不作大小写或空白归一化。

`render_input(s)` 依次列全部任务问题、模式、预算、已用 token、“已保留”和“新返回”片段；每段标编号及 token 数。编号在同一轨迹内稳定，输出按到达顺序执行。`render_target(s)` 输出紧凑 JSON，键顺序固定为 `add`、`drop`。`gold`、oracle 及审计字段不放入模型输入。

所有随机均使用显式种子的 `random.Random(seed)`。默认策略为纯 oracle；`rollout(..., policy=make_oracle_noise_policy(seed))` 可在重选模式中让理想动作额外保留预算内的 1–3 个新非证据段，形成下一步需要 drop 的状态，`sft_target` 始终保持纯 oracle。追加模式不引入噪声。这是任务书指定的 Claude 理解，详见 NOTES。

无效输出在 rollout 中按“本步保持原状”处理；重复 add 编号按集合处理。oracle 先按 token 数贪心，重选时同长度优先保留旧证据，再以稳定编号打破平局。当前奖励分母只含已见过且仍在候选或保留集合中的证据；已丢失且未重新检索到的证据不在分母内，这是 PLAN 的原口径。

Qwen3 后续训练和推理须关闭 thinking，例如使用对应 chat template 的 `enable_thinking=False`；本任务不实现训练或模型推理。缓存读写倍率默认 0.05 / 1.25，是本规格中的可调参数，不代表已核实的供应商现行价格。

## 下一步

- 精读 PLAN §2 指定文献，核实差异与引用。
- 按 PLAN §6 开展可行性试验，统计任务长度并确定正式三档预算。
- 安装训练框架后封装 ms-swift ORM 奖励插件；本任务未安装 ms-swift。
- 通过可行性检查后再考虑 SFT / GRPO、同数据基线与真实 Agent 评测。

## 交付与复现说明

数据来源、许可、实际条数与字段结构见 [DATA_CARD.md](DATA_CARD.md)；全部改动及保守处理见 [NOTES.md](NOTES.md)；给审核者的检查清单、确切命令、测试摘要与耗时见 [HANDOFF.md](HANDOFF.md)。全部文件定稿后才生成 `SHA256SUMS`；本机镜像没有数据，因此完整校验必须在 5880 执行。

<!-- DATA_AUDIT_BEGIN -->
## MuSiQue 原始数据核对

核对时间（UTC）：2026-11-17T03:31:12.032983+00:00。仅做任务 01 规定的字段、重叠及段落长度核对。

MuSiQue-Ans train / dev：19,938 / 2,417 题。
按 `(title, paragraph_text)` 精确去重后：train **84,559** 段，dev **21,100** 段。

| 重叠口径 | 交集数 | train 分母及比例 | dev 分母及比例 |
|---|---:|---:|---:|
| 标题 + 正文完全一致 | 3,697 | 84,559；4.372095% | 21,100；17.521327% |
| 标题完全一致（唯一标题） | 4,260 | 71,090；5.992404% | 17,629；24.164729% |

共有标题对应 train 13,896 段（占 train 去重段落 16.433496%）、dev 7,453 段（占 dev 去重段落 35.322275%）；标题可对应不同正文。去重和重叠均不做大小写、空白或 Unicode 归一化。

dev 去重段落按 `标题 + 换行 + 正文` 渲染；Qwen3-1.7B tokenizer.json 离线分词，`add_special_tokens=False`。P50 / P90 / P99 = **103 / 205 / 363.01 token**；共 21,100 段。分位数采用 NumPy 线性插值；预算是这些片段 token 数的和，不含输入模板元数据。

### 子问题占位符核对

dev 跳数分布：`{"2": 1252, "3": 760, "4": 405}`。
`#k` 出现在 2,417 题、3,636 跳，共 3,987 次；`>>` 出现在 1,594 题、2,304 跳。
`#k` 按题内第 k 个已完成子问题的答案替换；`实体 >> 关系` 保留为普通检索字符串。
疑似其他引用写法：0 题 / 0 跳。检测范围：非标准 #、[n]、$n、数字或 ref/answer/hop XML 标签、花括号引用；无法由有限正则证明不存在任意自然语言引用。
`paragraph_support_idx` 缺字段 0 跳、空值 0 跳，涉及 0 题。
经过引用、证据索引和语料映射校验，可用于模拟 2,417 题，保守跳过 0 题；原因计数：`{}`。

### 2 / 3 / 4 跳各 5 条原始 question_decomposition

按 dev 文件顺序取各类前 5 题；以下保留原始字段及占位符，供人工核对。

#### 2 跳

题号：`2hop__460946_294723`

问题：Who is the spouse of the Green performer?

```json
[
  {
    "id": 460946,
    "question": "Green >> performer",
    "answer": "Steve Hillage",
    "paragraph_support_idx": 10
  },
  {
    "id": 294723,
    "question": "#1 >> spouse",
    "answer": "Miquette Giraudy",
    "paragraph_support_idx": 5
  }
]
```

题号：`2hop__252311_366220`

问题：Who founded the company that distributed the film UHF?

```json
[
  {
    "id": 252311,
    "question": "UHF >> distributed by",
    "answer": "Orion Pictures",
    "paragraph_support_idx": 10
  },
  {
    "id": 366220,
    "question": "#1 >> founded by",
    "answer": "Mike Medavoy",
    "paragraph_support_idx": 6
  }
]
```

题号：`2hop__701895_752697`

问题：What administrative territorial entity is the owner of Ciudad Deportiva located?

```json
[
  {
    "id": 701895,
    "question": "Ciudad Deportiva >> owned by",
    "answer": "Nuevo Laredo",
    "paragraph_support_idx": 11
  },
  {
    "id": 752697,
    "question": "#1 >> located in the administrative territorial entity",
    "answer": "Tamaulipas",
    "paragraph_support_idx": 3
  }
]
```

题号：`2hop__259228_793698`

问题：Where is Ulrich Walter's employer headquartered?

```json
[
  {
    "id": 259228,
    "question": "Ulrich Walter >> employer",
    "answer": "German Aerospace Center",
    "paragraph_support_idx": 14
  },
  {
    "id": 793698,
    "question": "#1 >> headquarters location",
    "answer": "Cologne",
    "paragraph_support_idx": 11
  }
]
```

题号：`2hop__481349_302087`

问题：Which company owns the manufacturer of Learjet 60?

```json
[
  {
    "id": 481349,
    "question": "Learjet 60 >> manufacturer",
    "answer": "Bombardier Aerospace",
    "paragraph_support_idx": 10
  },
  {
    "id": 302087,
    "question": "#1 >> owned by",
    "answer": "Bombardier Inc.",
    "paragraph_support_idx": 5
  }
]
```

#### 3 跳

题号：`3hop1__454441_55349_651302`

问题：What is the birthplace of the man who does the voice of Stan on the series that includes the episode The Hobbit?

```json
[
  {
    "id": 454441,
    "question": "The Hobbit >> part of the series",
    "answer": "South Park",
    "paragraph_support_idx": 0
  },
  {
    "id": 55349,
    "question": "who does the voice of stan on #1",
    "answer": "Trey Parker",
    "paragraph_support_idx": 7
  },
  {
    "id": 651302,
    "question": "#2 >> place of birth",
    "answer": "Denver",
    "paragraph_support_idx": 9
  }
]
```

题号：`3hop1__145290_443779_52195`

问题：Who is the president of the new country who has jointly established a Commission on Truth and Friendship with the birth country of Satyo Husodo?

```json
[
  {
    "id": 145290,
    "question": "Where was Satyo Husodo born?",
    "answer": "Indonesia",
    "paragraph_support_idx": 8
  },
  {
    "id": 443779,
    "question": "#1 –Timor Leste Commission of Truth and Friendship >> country",
    "answer": "East Timor",
    "paragraph_support_idx": 7
  },
  {
    "id": 52195,
    "question": "who is the president of newly declared independent country #2",
    "answer": "Francisco Guterres",
    "paragraph_support_idx": 16
  }
]
```

题号：`3hop1__105767_443779_52195`

问题：Who is the president of the newly declared independent country that is part of the commission of truth and friendship with the country that eats Kemplang?

```json
[
  {
    "id": 105767,
    "question": "Which was the country for Kemplang?",
    "answer": "Indonesia",
    "paragraph_support_idx": 0
  },
  {
    "id": 443779,
    "question": "#1 –Timor Leste Commission of Truth and Friendship >> country",
    "answer": "East Timor",
    "paragraph_support_idx": 3
  },
  {
    "id": 52195,
    "question": "who is the president of newly declared independent country #2",
    "answer": "Francisco Guterres",
    "paragraph_support_idx": 19
  }
]
```

题号：`3hop1__105126_443779_52195`

问题：Who is the president of the newly declared independent country part of the Commission of Truth and Friendship with the country where Gado-gado is found?

```json
[
  {
    "id": 105126,
    "question": "The country for Gado-gado was what?",
    "answer": "Indonesia",
    "paragraph_support_idx": 13
  },
  {
    "id": 443779,
    "question": "#1 –Timor Leste Commission of Truth and Friendship >> country",
    "answer": "East Timor",
    "paragraph_support_idx": 9
  },
  {
    "id": 52195,
    "question": "who is the president of newly declared independent country #2",
    "answer": "Francisco Guterres",
    "paragraph_support_idx": 2
  }
]
```

题号：`3hop1__103881_443779_52195`

问题：Who is the president of the newly declared independent country that is part of the Commission of Truth and Friendship with the country that Tony Gunawan is from?

```json
[
  {
    "id": 103881,
    "question": "Where was Tony Gunawan from?",
    "answer": "Indonesia",
    "paragraph_support_idx": 17
  },
  {
    "id": 443779,
    "question": "#1 –Timor Leste Commission of Truth and Friendship >> country",
    "answer": "East Timor",
    "paragraph_support_idx": 7
  },
  {
    "id": 52195,
    "question": "who is the president of newly declared independent country #2",
    "answer": "Francisco Guterres",
    "paragraph_support_idx": 16
  }
]
```

#### 4 跳

题号：`4hop1__93963_170667_443779_52195`

问题：Who is the president of the newly declared independent country, that established the Timor Leste Commission of Truth and Friendship, with the country containing the airport that includes Lion Air?

```json
[
  {
    "id": 93963,
    "question": "What airport is Lion Air part of?",
    "answer": "Juanda International Airport",
    "paragraph_support_idx": 13
  },
  {
    "id": 170667,
    "question": "#1 >> country",
    "answer": "Indonesia",
    "paragraph_support_idx": 3
  },
  {
    "id": 443779,
    "question": "#2 –Timor Leste Commission of Truth and Friendship >> country",
    "answer": "East Timor",
    "paragraph_support_idx": 14
  },
  {
    "id": 52195,
    "question": "who is the president of newly declared independent country #3",
    "answer": "Francisco Guterres",
    "paragraph_support_idx": 11
  }
]
```

题号：`4hop3__862_846_326964_7713`

问题：How long was the city where the Yongle emperor greeted the person the Ming court thought the representatives were sent by the capitol of Yaxing Coach's headquarters location?

```json
[
  {
    "id": 862,
    "question": "Who did the Ming court think the representatives were sent by?",
    "answer": "the Karmapa",
    "paragraph_support_idx": 0
  },
  {
    "id": 846,
    "question": "Where did the Yongle Emperor greet the #1 ?",
    "answer": "Nanjing",
    "paragraph_support_idx": 3
  },
  {
    "id": 326964,
    "question": "Yaxing Coach >> headquarters location",
    "answer": "Yangzhou",
    "paragraph_support_idx": 6
  },
  {
    "id": 7713,
    "question": "How long had #2 been the capital city of #3 ?",
    "answer": "about 400 years",
    "paragraph_support_idx": 15
  }
]
```

题号：`4hop3__862_846_613770_7713`

问题：How long was the place where the Yongle Emperor greeted who the Ming court thought representatives were sent by the capitol of the city in which Guangling District is located?

```json
[
  {
    "id": 862,
    "question": "Who did the Ming court think the representatives were sent by?",
    "answer": "the Karmapa",
    "paragraph_support_idx": 2
  },
  {
    "id": 846,
    "question": "Where did the Yongle Emperor greet the #1 ?",
    "answer": "Nanjing",
    "paragraph_support_idx": 6
  },
  {
    "id": 613770,
    "question": "Guangling District >> located in the administrative territorial entity",
    "answer": "Yangzhou",
    "paragraph_support_idx": 17
  },
  {
    "id": 7713,
    "question": "How long had #2 been the capital city of #3 ?",
    "answer": "about 400 years",
    "paragraph_support_idx": 4
  }
]
```

题号：`4hop3__857_846_613770_7713`

问题：How long was the place where the Yongle Emperor greeted the person to whom the edict was addressed the capitol of the area where Guangling District was located?

```json
[
  {
    "id": 857,
    "question": "Who was the edict addressed to?",
    "answer": "the Karmapa",
    "paragraph_support_idx": 11
  },
  {
    "id": 846,
    "question": "Where did the Yongle Emperor greet the #1 ?",
    "answer": "Nanjing",
    "paragraph_support_idx": 3
  },
  {
    "id": 613770,
    "question": "Guangling District >> located in the administrative territorial entity",
    "answer": "Yangzhou",
    "paragraph_support_idx": 19
  },
  {
    "id": 7713,
    "question": "How long had #2 been the capital city of #3 ?",
    "answer": "about 400 years",
    "paragraph_support_idx": 17
  }
]
```

题号：`4hop3__857_846_326964_7713`

问题：How long had the city where the Yongle emperor greeted the person to whom the edict was addressed been the capitol of Yaxing Coach's headquarters location?

```json
[
  {
    "id": 857,
    "question": "Who was the edict addressed to?",
    "answer": "the Karmapa",
    "paragraph_support_idx": 7
  },
  {
    "id": 846,
    "question": "Where did the Yongle Emperor greet the #1 ?",
    "answer": "Nanjing",
    "paragraph_support_idx": 13
  },
  {
    "id": 326964,
    "question": "Yaxing Coach >> headquarters location",
    "answer": "Yangzhou",
    "paragraph_support_idx": 5
  },
  {
    "id": 7713,
    "question": "How long had #2 been the capital city of #3 ?",
    "answer": "about 400 years",
    "paragraph_support_idx": 4
  }
]
```


<!-- DATA_AUDIT_END -->


## 任务02：§6 可行性试验（无阅读模型部分）

任务02在5880使用现有数据和环境完成一次纯CPU实验：固定seed=20261117，150个四题任务、600道互不重复dev题；每跳top-20，19种配置共用同一次生成的检索步骤。没有下载、安装、GPU或API调用，没有按结果调整任务。**阅读模型F1 / EM未测（需阅读模型）**；证据完整题比例是代理指标，不是F1。

紧/中/松预算为 **1876 / 2814 / 5628 token**；dev每任务去重证据token P90=1875.600（n=150），紧档向上取整，中档ceil(1.5×紧档)，松档3×紧档。train参照P90=1691.800（150任务），不参与定预算、不建train索引。

dev查询校验：2417/2417通过，拒绝0题。检索自然命中 **1456/1594 = 91.343%**。非证据段先排除整任务全部gold，再用归一化后的本跳答案匹配正文：段级 **1253/29732 = 4.214%**，跳级 **473/1594 = 29.674%**；≥2词答案的段级为 **2.815%**（n=18544），跳级为 **21.888%**（n=996）。标题+正文敏感性口径：段级 **4.359%**（n=29732），跳级 **30.678%**（n=1594）。

中档的任务末召回用该任务全部唯一gold为分母；下表差值为oracle减BM25，单位为百分点。每配置n=150任务/1594步。

| 模式 | oracle任务末召回 | BM25任务末召回 | 任务末差pp | 单步按步差pp | oracle丢失率 | BM25丢失率 |
|---|---:|---:|---:|---:|---:|---:|
| reselect | 100.000% | 35.591% | 64.409 | 21.066 | 0.000% | 100.000% |
| append | 100.000% | 14.283% | 85.717 | 40.928 | 0.000% | 100.000% |

全量原始轨迹token均值=24804.687、P90=29887.100（n=150）；去重轨迹均值=23411.533、P90=28772.000。全部 **150/150** 测试通过；30286个状态中非法动作和超预算均为0。no_compress任务末召回为1、丢失率为0。

完整的19配置指标、损失拆分、输入长度、缓存成本及PLAN原四行判据见[实验报告](eval/feasibility/feasibility_report.md)，所有分布与比例均有n。阅读模型相关格标为“未测（需阅读模型）”，本任务不作开工决策，不定义“接近”阈值。[机器结果](eval/feasibility/feasibility_results.json)、[运行配置](eval/feasibility/run_config.json)和[20条答案命中样例](eval/feasibility/answer_leak_samples.md)已同步镜像。

原始实验产物仅在5880的`data/feasibility/`：`tasks_dev150.jsonl`、`steps.jsonl`、`final_contexts.jsonl`和`corpus_dev_ntok.jsonl`。最终保留集合及四题的question/answer/answer_aliases均保存，以后可直接补阅读模型评测。任务01的验收目录、轨迹和3000占位预算保持原样。

任务02扩展了摘要覆盖范围，因此从本节起使用 `bash scripts/run_python.sh scripts/checksums_task02.py --verify`；任务01的原校验脚本未改动，旧清单保存在`specs/TASK01_SHA256SUMS`。原文档前缀与62个其他任务01文件也由新校验器逐项保护。确切命令见HANDOFF的任务02追加节。

[独立落盘复核](eval/feasibility/independent_check.md)已通过：21,100段重新分词、30,286状态重放、2,850最终上下文核对，5,687,736项比较无差异；全部12组oracle汇总比较及1,800次逐任务比较均无低于其他策略的反例。


## 任务03：§6 阅读模型 F1（Qwen3-8B）

已在任务02固定的150个任务、600道不同问题上完成本地Qwen3-8B阅读：20种配置，各n=600题，共12,000条预测记录。复用原19种最终上下文，增加闭卷no_context；没有重跑检索、压缩或预算，没有训练或在线模型API调用。全量一次，报告均值和n，不计算CI，不作“开工/不开工”判断。

### 主要结果

| 配置 | n 题 | EM | F1 |
| --- | --- | --- | --- |
| oracle\|reselect\|mid | 600 | 45.500% | 57.564% |
| oracle\|append\|mid | 600 | 45.333% | 57.479% |
| no_compress | 600 | 15.667% | 26.416% |
| no_context | 600 | 4.000% | 12.485% |
| bm25\|reselect\|mid | 600 | 12.000% | 19.673% |
| bm25\|append\|mid | 600 | 4.333% | 10.349% |
| latest\|reselect\|mid | 600 | 7.667% | 14.806% |
| latest\|append\|mid | 600 | 4.000% | 10.083% |

### PLAN §6 判据表（填数，不作开工判断）

前四行观察条件保留 PLAN 原文。oracle − BM25 和 oracle − no_compress 的差值均以百分点（pp）表示；不对‘接近’或‘明显更低’另定阈值，不据本表调整任务、检索或预算。

| PLAN 原观察条件 | 证据指标 / 配对范围 | 阅读模型观测 |
| --- | --- | --- |
| 中档预算下，BM25 的证据召回比 oracle 低 15 个点以上，F1 低 5 个点以上 | reselect/mid：任务末召回差 64.409 pp（≥15：是，n=150任务）；单步按步差 21.066 pp（≥15：是，n=1594步）；单步按任务差 20.867 pp（n=150任务）<br>append/mid：任务末召回差 85.717 pp（≥15：是，n=150任务）；单步按步差 40.928 pp（≥15：是，n=1594步）；单步按任务差 40.857 pp（n=150任务） | reselect/mid：oracle 57.564% − BM25 19.673% = 37.891 pp（≥5：是；各 n=600题）<br>append/mid：oracle 57.479% − BM25 10.349% = 47.129 pp（≥5：是；各 n=600题） |
| BM25 已接近 oracle | reselect/tight：任务末差 68.894 pp（n=150任务）；单步按步差 26.357 pp（n=1594步）；单步按任务差 26.093 pp（n=150任务）<br>reselect/mid：任务末差 64.409 pp（n=150任务）；单步按步差 21.066 pp（n=1594步）；单步按任务差 20.867 pp（n=150任务）<br>reselect/loose：任务末差 51.753 pp（n=150任务）；单步按步差 12.033 pp（n=1594步）；单步按任务差 11.835 pp（n=150任务）<br>append/tight：任务末差 88.041 pp（n=150任务）；单步按步差 48.921 pp（n=1594步）；单步按任务差 48.776 pp（n=150任务）<br>append/mid：任务末差 85.717 pp（n=150任务）；单步按步差 40.928 pp（n=1594步）；单步按任务差 40.857 pp（n=150任务）<br>append/loose：任务末差 72.683 pp（n=150任务）；单步按步差 23.168 pp（n=1594步）；单步按任务差 22.982 pp（n=150任务）<br>差值仅作描述，不新增‘接近’阈值。 | reselect/tight：38.944 pp（各 n=600题）<br>reselect/mid：37.891 pp（各 n=600题）<br>reselect/loose：36.114 pp（各 n=600题）<br>append/tight：48.667 pp（各 n=600题）<br>append/mid：47.129 pp（各 n=600题）<br>append/loose：42.446 pp（各 n=600题） |
| 不压缩的 F1 与 oracle 接近 | 两种 oracle 中档，使用相同 600 题逐题配对。 | reselect/mid，n=600：oracle − no_compress F1=31.148 pp；逐题 oracle 更高/相等/no_compress 更高=263/310/27；EM 配对 2×2（行=oracle 对/错，列=no_compress 对/错）=[[87,186],[7,320]]<br>append/mid，n=600：oracle − no_compress F1=31.062 pp；逐题 oracle 更高/相等/no_compress 更高=263/310/27；EM 配对 2×2（行=oracle 对/错，列=no_compress 对/错）=[[87,185],[7,321]] |
| 不压缩的 F1 明显低于 oracle | 两种 oracle 中档，使用相同 600 题逐题配对。 | reselect/mid，n=600：oracle − no_compress F1=31.148 pp；逐题 oracle 更高/相等/no_compress 更高=263/310/27；EM 配对 2×2（行=oracle 对/错，列=no_compress 对/错）=[[87,186],[7,320]]<br>append/mid，n=600：oracle − no_compress F1=31.062 pp；逐题 oracle 更高/相等/no_compress 更高=263/310/27；EM 配对 2×2（行=oracle 对/错，列=no_compress 对/错）=[[87,185],[7,321]] |
| 附加闭卷对照 no_context | 无段落上下文，同一批 n=600 题。 | EM 4.000%；F1 12.485%；n=600题 |

### 任务03目录与结果

正式代码、小结果和文档仍在 `/root/workplace/ctx-compressor`。独立reader环境为 `/root/userfolder/ctx-compressor/envs/reader`，模型为 `/root/userfolder/ctx-compressor/models/Qwen3-8B`，所有缓存和临时文件为 `/root/userfolder/ctx-compressor/cache`。启动前 source `scripts/reader_environment.sh`。

- [完整中文报告](eval/reader/reader_report.md)：20配置EM/F1、位置/跳数/证据齐全分组、prompt分布和逐题配对，所有分组附n。
- [机器结果](eval/reader/reader_results.json)、[运行参数与资源](eval/reader/run_config.json)、[原始40条小样](eval/reader/smoke_samples.md)、[小样审阅](eval/reader/smoke_review.json)。
- [模型manifest](eval/reader/model_manifest.json)：16文件、16,397,461,896 bytes，官方SHA256和离线复核均通过；[环境锁定](requirements.reader.lock.txt)与[准备记录](notes_task03_environment.md)。
- 逐题原始输出和评分为 `data/reader/predictions.jsonl`；该文件仅留在5880，3080的mirror只同步代码、文档和eval/reader小结果，不同步data/reader或权重。

### 验证与运行记录

旧新测试共 202 项全部通过（[测试记录](eval/reader/tests.xml)）；CPU报告重新解析、评分和核对全部12,000条元数据/user消息SHA1。输入截断0，prompt+48超限0；全配置最大prompt 32,382 tokens，no_compress最大 32,382，加48后 32,430≤40,960。固定48-token输出上限命中 207/12,000（1.725%），原样保留并评分；>20词解析结果 708/12,000，空解析 0/12,000。40条smoke中的一次输出上限命中及实现guard修正见NOTES，未重新生成smoke。

物理 GPU 1（NVIDIA RTX 5880 Ada Generation，UUID `GPU-4318dc90-4126-038a-9402-1323533db94f`），记录到的本任务 GPU 进程均属于该 UUID；峰值显存 42,423 MiB。root Used 基线 931,190,407,168 bytes，推理收尾 931,367,436,288 bytes；最终增量 177,029,120 bytes（验收门槛 300,000,000 bytes）。另保留成功全量进程的全盘计数瞬时峰值 501,190,656 bytes及3次已复查事件；历史超限和未能归因的调查经过见NOTES。 全量生成耗时 1750.871 秒；本次全量进程（含准备、引擎初始化）1910.128 秒。保留任务01/02所有不变文件和三份文档的原字节前缀；最终验收使用 `scripts/checksums_task03.py --verify`，覆盖新代码、结果和data/reader，外置权重由manifest独立记录。


## 任务04：SFT 数据与吞吐基准


任务04完成 PLAN §7 的训练准备和一次吞吐基准；没有启动正式训练、GRPO 或闭环评测，没有调用外部推理 API。#9 Dream 的两道 train 题恢复可用，全部 dev 查询与旧结果逐字节不变。

训练部分 **4,735 个任务、88,562 条 SFT 样本**，5% 留出 **249 个任务、4,652 条样本**；两部分共 n=93,214 条，全部使用 ms-swift 实际 Qwen3 模板编码，裁切和丢弃均为 0。完整序列 P50/P99/max 为 **3194/5308/8390 token**，`max_length=8392`。

一次基准使用均匀抽样的 reselect/append 各800条，共1,600条，**100个 optimizer step**；实际吞吐 **5602.89 token/s**（完整未padding序列token），每条 **0.5790 秒**。首10/末10步loss均值 **3.998265/0.163077**。最长8,390-token样本另做一次带驻留Adam状态的前向/反向检查，额外optimizer step=0，未OOM。只使用物理GPU 1，结束已释放；root增量低于300 MB，最终读数见[资源检查](eval/sft_prep/final_resource_check.json)。

- [完整中文报告和估时表](eval/sft_prep/sft_prep_report.md)、[机器结果](eval/sft_prep/sft_prep_results.json)。估时只列全量×1/2/3 epoch、20k×2、40k×2，尚未选择正式训练方案。
- [模板及labels核验](eval/sft_prep/template_check.md)、[模型逐文件校验](eval/sft_prep/model_manifest.json)、[基准配置](eval/sft_prep/bench_run_config.json)、[基准结果](eval/sft_prep/bench_results.json)。
- [环境准备记录](notes_task04_environment.md)、[依赖锁定](requirements.train.lock.txt)、[旧结果保护核验](eval/sft_prep/reference_fix_checks.json)。旧202项和新增测试共 **264 项通过**。

大数据、模型、独立环境和基准输出均在 `/root/userfolder/ctx-compressor/`。轨迹 `data/traj_train/`，SFT `data/sft/`，环境 `envs/train`，模型 `models/Qwen3-1.7B`，基准 `runs/bench_task04_20261117`；正式项目仅保留代码、文档和小结果。外置文件路径、字节/行数和SHA256见各manifest，不进入项目SHA256SUMS。


## 任务05最终交付：计算完成，资源验收未通过

记录时间：2026-11-18T04:02:22.428837+00:00。A/B/C 计算与独立审计均已结束；任务整体状态为 **failed**，原因是资源验收采样中 `/` 的原始增量为 **834,052,096 bytes**，超过 **300,000,000 bytes**。未扣除未知来源占用。只使用物理 GPU 1；16 个记录到的本任务 PID 已退出，终态 GPU 占用 375 MiB，本任务 GPU 进程为 0。

A 选定 per-device **1 × 累积16**，group_by_length=False，实测 **5695.282197 token/s**（n=30 optimizer steps、480样本暴露）。B 完成 **20,000条 × 2 epoch**，2500 optimizer steps、40,000次样本暴露，无NaN/OOM。训练循环 **25774.874464秒**（含5次验证/保存），worker计时 **26053.937670秒**（含其计时区间内启动/编码）。

C 完成原始模型与5个checkpoint，共 **6×13956=83736** 个动作、**6×1494=8964** 条轨迹，另完成4项CPU基线。按任务书合并证据丢失率排序选中 **checkpoint-2000**（1238/1494，任务末召回0.5795890633240051）。其路径为 `/root/userfolder/ctx-compressor/runs/sft_20k_20261117/checkpoints/checkpoint-2000`；[选中checkpoint指针](eval/sft_run/selected_checkpoint.json)记录adapter SHA及选择规则。

- [完整中文报告](eval/sft_run/sft_run_report.md)、[结构化结果](eval/sft_run/sft_run_results.json)、[loss曲线](eval/sft_run/loss_curve.svg)。
- [全量闭环审计](eval/sft_run/model_rollout_audit.json)、[最终报告一致性](eval/sft_run/final_report_audit_consistency.json)、[源码/依赖终检](eval/sft_run/final_dependency_verification.json)。
- [失败资源验收](eval/sft_run/final_resource_check.json)、[151项外置文件清单](eval/sft_run/external_artifact_manifest.json)、[额外工具链完整清单](eval/sft_run/toolchain_full_artifact_manifest.json)。外置清单complete只表示文件覆盖完整，任务验收仍失败。

原264项及新增测试共462项通过；历史82个Python源码仍与测试收据一致。新增纯审计工具的真实全树验证单独留证。数据仅使用train及其249个留出任务；未在dev测试任务评测，未运行GRPO或阅读模型F1。报告只列实测数值与checkpoint选择。


## 任务06　CPU 数据完成，训练正在等卡

v2 将 reselect 的每步噪声改为随机 0–2 个片段。训练数据 88,562 行、留出数据 4,652 行。step ≥ 1 的训练 reselect 目标中，非空 drop 从 39,536/39,546 变为 26,300/39,546。抽中的 20,000 个训练样本与 1,000 个验证样本的 ID 和顺序均与任务05一致。472 项 CPU 测试通过。每个拆分、模式和预算的统计见[当前报告](eval/sft_run_v2/sft_v2_report.md)。

5880 的后台脚本 PID 1141340 每 600 秒检查一次物理 GPU 1。首次观测为 42088 MiB，连续两次低于 2 GB 后启动训练。训练计划为 2 个 epoch、2,500 个 optimizer step，随后对五个 v2 checkpoint 评测。实时状态文件为 `/root/userfolder/ctx-compressor/runs/sft_20k_v2_20261118/continuation_status.json`。[启动验证](eval/sft_run_v2/waiting_launch.json)记录了 SSH 重连后读取的 PID、独立会话和日志文件。


## 任务06　留出集闭环评测完成

五个 v2 checkpoint 各完成 1,494 条留出轨迹、13,956 个动作。按六个模式与预算组合的合并证据丢失率，选中 **checkpoint-2000**，丢失 956/1,494（63.989%），任务末平均证据召回为 89.006%（n=1,494）。中档预算为 2,814 token，重选模式的任务末召回为 88.506%，只追加模式为 89.586%（各 n=249）。

[完整中文报告](eval/sft_run_v2/sft_v2_report.md) · [结构化结果](eval/sft_run_v2/sft_v2_results.json) · [选中的 LoRA](eval/sft_run_v2/selected_checkpoint.json) · [损失曲线](eval/sft_run_v2/loss_curve.svg)


## 任务07　GRPO 准备（2026-11-20，5880 GPU 0，Claude 完成）

- 环境：`/root/userfolder/ctx-compressor/envs/grpo`（ms-swift 3.12.6 + vLLM 0.11.0），入口 `scripts/grpo_environment.sh`。
- 流程：`build_grpo_states.py`（训练任务上的闭环状态）→ `make_grpo_dataset.py` → `merge_sft_lora.py`（SFT LoRA 合并进基座）→ `train_grpo.sh`（在合并模型上训新 LoRA，KL 参照 = SFT 模型）→ 评测链路复用 `build_grpo_states.py --adapter`。
- 所有模型输出数字来自替身 v1 checkpoint-2000，不是正式结果。关键数：奖励与评测解析器在 13,956 条输出上 0 处不一致；冒烟训练 25 步，每步约 40.4 秒（4 个 prompt × 8 个补全）；替身模型下 77.6%（149/192）的 8 样本组奖励全相同。

[GRPO 准备报告](eval/grpo_prep/grpo_prep_report.md) · [规模推算](eval/grpo_prep/projection.json) · [任务07 校验清单](eval/grpo_prep/TASK07_SHA256SUMS)


## 任务08　正式 GRPO 已启动

从任务06选中的 v2 checkpoint-2000 启动正式 GRPO。CPU 合并后，GPU 1 上的固定训练提示中，贪心文本和生成 token ID 均为 0/30 不一致。完整 CPU 测试通过 527/527。后台阶段依次运行合并模型留出评测、训练状态生成、八次采样过滤、均衡抽样、优化与各 checkpoint 的留出评测。[起点核验](eval/grpo_v2/a_checks.json) · [后台启动记录](eval/grpo_v2/launch_verification.json)


## 任务08　正式 GRPO 训练与留出评测完成

从 SFT v2 checkpoint-2000 构建的 22,158 个训练状态中，有 4,002 个状态的八次采样奖励存在差异。分层抽取 2,000 个提示后完成 500 个优化步。按六格合并证据丢失率，从四个 GRPO checkpoint 中选中 **checkpoint-500**，丢失 947/1,494 条轨迹，任务末平均召回 89.30%（n=1,494）。同一留出集上，SFT v2 为 956/1,494 和 89.01%（n=1,494）。

紧预算下，重选模式的证据丢失为 SFT 164/249、GRPO 161/249，只追加模式为 SFT 155/249、GRPO 159/249。[中文报告](eval/grpo_v2/grpo_v2_report.md) · [结构化结果](eval/grpo_v2/grpo_v2_results.json) · [训练曲线](eval/grpo_v2/training_curve.svg)

## 任务09　置信度阈值与第二个候选片段

使用任务06的 SFT v2 checkpoint-2000，在 400 个训练任务的 22,158 个状态上按平均单步奖励选出 τ=0.99。每个留出策略覆盖 249 个任务的六种模式与预算组合（各 n=1,494 条轨迹）。H0（不加段）、H1（τ=0.99）、H2（预算允许时总加段）的证据丢失数依次为 957、806、916，末步平均召回依次为 88.99%、90.62%、88.14%，平均缓存成本依次为 1,800.30、2,478.70、3,475.33 个等价输入 token（以上各 n=1,494）。[中文报告](eval/hedge_v2/hedge_v2_report.md) · [结果 JSON](eval/hedge_v2/hedge_v2_results.json) · [成本与证据丢失率](eval/hedge_v2/cost_vs_loss.svg)

## 任务10 SFT v3 与锁定测试集

锁定的 MuSiQue dev 最终测试集有 451 个任务（1,804 题、4,771 个检索步骤）。首次运行策略须另有 Claude 编写的任务书。锁定记录见 `eval/test_dev_v1/LOCK.md`，结构统计见 `eval/test_dev_v1/structural_stats.json`。

奖励 v2 的离线扫描使用任务09的 22,158 个训练状态。五个 token 价格的完整阈值表见 `eval/reward_v2/offline_tau_table.md`。两训练任务的闭环冒烟覆盖 12 条轨迹、120 步；全部训练输入和目标通过核对。正式流水线使用 5880 物理 GPU 1，从 SFT v2 的训练轨迹构造数据，再从 Qwen3-1.7B 基座新建 LoRA。状态见 `/root/userfolder/ctx-compressor/dagger_v3/pipeline_status.json`。

## 任务10　SFT v3 留出集结果

在 30,000 条样本上训练两轮后，四个候选中按留出集选中 checkpoint-3000。证据丢失为 886/1,494 条轨迹（59.30%），最终平均召回为 90.46%（n=1,494）。任务06的 SFT v2 在只追加和重选模式分别丢失 466/747 和 490/747 条，SFT v3 分别为 396/747 和 490/747 条。两个模型在同一批训练状态上各采样 2,400 次，单次请求新增至少两段的次数分别为 SFT v2 的 0 次和 SFT v3 的 420 次（各 n=300 个状态）。[任务10报告](eval/sft_run_v3/sft_v3_report.md) · [机器结果](eval/sft_run_v3/sft_v3_results.json) · [训练曲线](eval/sft_run_v3/loss_curve.svg)

## 任务11　第二轮 GRPO 留出评测

从任务10 SFT v3 checkpoint-3000 出发，按丢失率、最终召回、训练步数的预登记顺序，从四个 GRPO 候选中选中 checkpoint-125。起点和所选模型的轨迹证据丢失分别为 886/1,494（59.30%）和 714/1,494（47.79%），最终平均召回分别为 90.46% 和 92.66%（各 n=1,494 条轨迹）。紧预算只追加模式为 137/249 和 115/249，紧预算重选模式为 165/249 和 144/249。平均单步奖励分别为 −0.469241 和 −0.494518（各 n=13,956 步）。[中文报告](eval/grpo_r2/grpo_r2_report.md) · [结构化结果](eval/grpo_r2/grpo_r2_results.json) · [训练曲线](eval/grpo_r2/training_curve.svg)


## 任务12　Qwen3-8B 留出集阅读评测

在 249 个留出任务的 996 道不同问题上，Qwen3-8B 共给出 49,800 条回答。GRPO r2 checkpoint-125 的合并结果为 F1 56.118%、EM 46.034%（n=5,976），SFT v3 checkpoint-3000 为 F1 56.137%、EM 45.716%（n=5,976），H1 为 F1 54.972%、EM 44.679%（n=5,976）。

r2 与 SFT v3 的配对差为 F1 -0.019 个百分点、EM +0.318 个百分点（n=5,976 个配对），与 H1 的配对差为 F1 +1.145 个百分点、EM +1.355 个百分点（n=5,976 个配对）。报告逐格列出紧、中、松预算下的读者分数及轨迹缓存成本。[报告](eval/reader_heldout/reader_heldout_report.md) · [结构化结果](eval/reader_heldout/reader_heldout_results.json)


## 任务13 · GRPO 第三轮（2026-11-24）

任务13 从 r2 checkpoint-125 出发，使用 2,000/2,000 条 gold-varying 训练状态。留出集选中 checkpoint-250，丢证据 592/1,494，最终召回 93.702%（n=1,494 条轨迹）。Qwen3-8B 读者 F1 为 50.709%，相对 r2 为 -5.409 个百分点（n=5,976 个配对题次）。[完整报告](eval/grpo_r3/grpo_r3_report.md)


## 任务14 · DeepSeek API 子集实验

全量调用 n=2,588，小样 n=36，两阶段账本合计 ¥3.03074106。[报告](eval/deepseek_api/deepseek_api_report.md) · [结果](eval/deepseek_api/deepseek_api_results.json)


## 任务17 · 同数据重排基线

val100 选出 RR-τ 的 τ=0.5（n=100 个验证任务）。同批留出比较 r2 − RR-τ 的 ΔF1 为 +0.536 个百分点（n=5,976）。[报告](eval/reranker_v1/reranker_v1_report.md) · [按预算的 F1](eval/reranker_v1/budget_f1_with_reranker.svg) · [按成本的 F1](eval/reranker_v1/cost_f1_with_reranker.svg)

## 任务18 · GRPO 第四轮

reward v3 选中 checkpoint-500（n=4 个候选）。同批留出比较 r4 − r2 的 ΔF1 为 +0.401 个百分点（n=5,976）。[报告](eval/grpo_r4/grpo_r4_report.md) · [训练曲线](eval/grpo_r4/training_curve.svg)


## 任务19 · 锁定测试集

锁定测试集已于任务19第一次使用。r4 F1 为 44.866%（n=10,824 题次），r4 − RR-τ 的 ΔF1 为 +20.841 个百分点（n=10,824 对）。[报告](eval/test_v1/test_v1_report.md)


## 任务20 · 真实 Agent 测试

本地 Qwen3-8B 在 MuSiQue、HotpotQA 和 2Wiki 上完成 n=300 个多题任务。合并 r4 重选 − RR-τ 重选 ΔF1 为 +0.993 个百分点（n=900 题对）。[任务20报告](eval/agent_v1/agent_v1_report.md)

---

## 任务21：真实 ReAct Agent 换 DeepSeek 主模型（2026-11-26，Claude 实现并执行）

- 任务书 `specs/TASK_21_deepseek_agent_20261126.md`，报告 `eval/agent_ds/main/agent_ds_report.md`，事后分析 `eval/agent_ds/main/posthoc_report.md`。代码：`scripts/agent_ds_client.py`、`scripts/agent_ds_run.py`、`scripts/agent_ds_report.py`、`tests/test_agent_ds.py`（13 项 CPU 测试）。
- **设置**：沿用任务20 的 Agent、提示词、检索和压缩器，只把主模型换成 `deepseek-flash`（关思考，用前缀续写接口），行数取 6 行。
- **范围**：任务20 正式题里每个数据集的前 40 个任务。7 个任务被内容审核拒绝，按规则整体排除，剩 113 个任务、339 道题。
- **主结果（ΔF1）**：
  - r4 reselect − RR-τ +12.6（同题 Qwen +2.7）；
  - r4 reselect − 不压缩 +0.4，r4 append − 不压缩 +1.2；
  - r4 reselect − LLMLingua-2 +7.4；
  - oracle − r4 reselect +5.9。
- **实际账单**：r4 约 ¥0.008 / 任务，不压缩 ¥0.025 / 任务。
- **偏差**：
  - 停止符加了 `\nThought:`；
  - 被内容审核拒绝的任务整体排除；
  - 允许高峰时段调用；
  - 总上限改为 ¥20。
- **花费**：账本合计 ¥11.29。账户余额从 ¥24.94 降到 ¥15.26，账户与 ctx-compressor 之外的线共用。
