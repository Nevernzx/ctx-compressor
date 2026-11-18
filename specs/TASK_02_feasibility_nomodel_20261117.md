# 任务 02：PLAN §6 可行性试验（只做不需要阅读模型的那半）

日期：2026-11-17　派发：Claude（大脑 / 核查）　执行：Codex（3080 上通过 `ssh 5880` 操作）
唯一规格：`tasks/PLAN_v2.md` §3、§4、§6、§7（指标定义）。本任务书只补 PLAN 没写死的操作细节；凡标"Claude 的理解"的条目，在 NOTES 里照抄并注明出处。

## 0. 范围

用户 11-17 选择："先跑不需要模型的那半——统计、定预算、四个策略的召回率和丢失率，纯 CPU，不下载任何东西。"

**做**：PLAN §6 的三件事中的
1. 统计（全部）；
2. 定三档预算；
3. 闭环跑四个不训练的策略，记录证据召回、证据丢失率（**不算 F1**）。

**不做（硬性）**：
- 不下载任何模型 / 数据，不 pip / conda 安装任何包（缺包就停，最后一行用 `NEED:` 说明）。
- 不用 GPU（`CUDA_VISIBLE_DEVICES=""`），不调用任何 API（免费的也不调）。
- 不算阅读模型 F1 / EM。
- 不修 "#9 Dream" 那 2 道 train 题（用户没选这项），只报告 dev 有没有被拒的题。
- **不按结果调任务**：即使 BM25 接近 oracle，也不改 top-20、不改每任务 4 题、不改预算公式、不重跑第二轮。PLAN §6 的"调整后再测"由用户决定。
- 不判"开工 / 不开工"：只把 §6 表格每一行需要的数字填出来，需要 F1 的格子写"未测（需阅读模型）"。
- 不删除、不覆盖任务 01 的任何产物（`eval/acceptance/`、`data/traj/acceptance.jsonl`、占位预算 3000 的验收保持原样）。README / NOTES / HANDOFF 只追加。

## 1. 位置与环境

- 正式目录：5880 `/root/workplace/ctx-compressor`，一律用 `bash scripts/run_python.sh ...`（已关代理、关 GPU、离线）。
- CPU 并行不超过 16 个进程 / 线程（宿主机共享）。开跑前 `df -h /`，新产物预计 < 200 MB，超过 1 GB 先停。
- 3080 的 `mirror/` 只同步代码、文档和小结果文件（`eval/feasibility/` 下的 md / json），**不同步 `data/feasibility/`**。rsync `-a`，不加 `--delete`。

## 2. 任务构造

- 语料库与 BM25：只用 MuSiQue-Ans **dev**，沿用 `build_corpus`（title + "\n" + body，Qwen3 tokenizer，`add_special_tokens=False`）。
- 先对全部 dev 题跑 `hop_queries`，把被拒的题按原因计数（期望 0；不为 0 也照常继续，只报告）。
- 从**通过的 dev 题**中用 `sample_tasks(valid, n_tasks=150, seed=20261117)` 抽 600 道互不重复的题，拼 150 个四题任务。
- 每个任务 `simulate_task(..., top_n=20, seed=<task_index>)` 生成一次检索步骤。四个策略、两种模式、三档预算**共用同一份步骤**（检索不依赖保留状态）。
- 单次运行、点估计 + n，不算 CI、不多种子（LMM / agent 线约定）。

## 3. 统计（PLAN §6 第一项）

所有分布报告 n、mean、P10 / P50 / P90 / P99、max。

1. **整条轨迹 token**（每任务）：
   - (a) 原始：每步 20 个返回片段的 ntok 之和，跨步重复的也算；
   - (b) 去重：整条轨迹返回过的不同片段的 ntok 之和。
   - 同时报告每任务步数分布、每题跳数分布。
2. **压缩器单步输入 token**：对每个状态 `render_input(s)` 用同一 tokenizer 计数。按 策略 × 模式 × 预算 分别报告；另报只算片段的部分（Σ ntok of kept + new），以便区分模板开销。"不压缩"没有压缩器，只报它每步累计上下文的 token。
3. **证据 token**（每任务）：该任务全部跳的金标准段**去重后**的 ntok 之和。另附每题的证据 token 分布。
4. **检索自然命中率**：`natural_hit` 为真的跳数占比。总体、按跳位（第 1/2/3/4 跳）、按题目总跳数（2/3/4）分别报告。
5. **非证据段含答案的比例**（标注遗漏上界）：
   - 每一跳的 20 个返回里，去掉**该任务所有金标准段**（任何题、任何跳），剩下的叫"非证据段"。
   - 用 SQuAD 风格归一化（小写、去标点、去冠词 a/an/the、合并空白）后，按词边界判断非证据段是否包含**这一跳**的 `answer`。
   - 报告：
     - (i) 段级：含答案的非证据段数 ÷ 非证据段总数；
     - (ii) 跳级：至少一个非证据段含答案的跳的占比；
     - (iii) (i)(ii) 只限归一化后 ≥ 2 个词的答案再算一次（排除 "1990"、"UK" 这类巧合匹配）；
     - (iv) 非证据段正文（归一化后）与该跳金标准段正文完全相同的个数（标题不同的重复段）。
   - 随机抽 20 条命中样例（`random.Random(20261117)`），写进 `eval/feasibility/answer_leak_samples.md`，每条列：子问题、答案、金标准段标题与前 300 字、命中的非证据段标题与前 300 字。供人工判断是真遗漏还是巧合。

## 4. 定预算（PLAN §6 第二项）

- 紧档 `B_tight = ceil(P90(每任务证据 token))`，用 `numpy.percentile(..., 90)`（默认 linear）。
- 中档 `B_mid = ceil(1.5 × B_tight)`；松档 `B_loose = 3 × B_tight`。
- 参照（只报告，不用于定档）：用 `sample_tasks(train_valid, 150, seed=20261117)` 抽 150 个 train 任务，只算其证据 token 的 P90。只需金标准段的 ntok，**不建 train 语料库和 BM25**。

## 5. 四个策略（PLAN §6 第三项，闭环）

全部通过现有 `rollout(questions, steps, corpus, budget, mode, policy=...)` 运行。非法动作或超预算出现即视为 bug（应为 0，并断言）。

| 策略 | 定义 |
|---|---|
| `no_compress` | 什么都留：每步 `add` 全部 `new_ids`、`drop` 为空；预算取一个足够大的整数（如全部片段 ntok 和 + 1）。两种模式结果相同，只跑一次、报一次。 |
| `latest` 保留最新 | 优先级 = 到达顺序倒序，即对 `kept_ids + new_ids` 反向遍历（同一步内按呈现给压缩器的顺序反向）。 |
| `bm25` BM25 装满 | 查询 = 4 道任务问题原文用空格拼接（即 `s["questions"]`）。分数 = 同一个 dev 语料库 BM25 索引下该片段的分数（IDF 来自语料库），与检索用同一套分词。按分数降序，同分时较新的在前。 |
| `oracle` | 现有 `oracle_action`。 |

- **装满规则（Claude 的理解，三个非 oracle 策略共用）**：按优先级 first-fit，放得下就放，放不下就**跳过、继续看下一个**，不在第一个放不下时停。
- **重选模式**：从 `kept_ids + new_ids` 全体中按上面的规则重新挑，没挑中的旧片段 drop。
- **只追加模式**：旧片段全留，只在 `new_ids` 中按优先级填剩余预算。
- **`latest` 和 `bm25` 只能使用压缩器可见的信息**：`questions`、`mode`、`budget`、`kept_ids`、`new_ids`、`text`、`ntok`；`bm25` 另可读 `pid_by_lid`，仅用于查语料库 BM25 分数（它只是编号映射，不含标注）。严禁读 `gold`、`sft_target` 和其他审计字段。
  - 加单元测试：把 `s["gold"]` 置空或打乱后，两者输出不变。
- **BM25 打分正确性**加单元测试：随机取 20 个 dev 跳查询，用打分函数对全语料排序取 top-20，应与 `index.retrieve` 的结果一致（同分处允许顺序不同）。

运行矩阵：`no_compress` × 1，加上 {latest, bm25, oracle} × {reselect, append} × {tight, mid, loose}，共 19 种配置 × 150 任务。

## 6. 指标（PLAN §6、§7 口径）

每种配置都报告以下指标：

1. **单步召回**：所有步的 `s["recall"]`（PLAN §4 奖励口径：分母是当前还拿得到的金标准）的平均值。按步汇总一次，按任务先平均再汇总一次。
2. **任务末召回（Claude 的理解）**：最后一步的保留集合中含有的该任务全部金标准段数 ÷ 该任务全部金标准段数，再对任务平均。阅读模型看到的是这个集合，所以把它当作 §6 判据里"证据召回"的主口径；单步召回并列报告。
3. **证据丢失率**（PLAN §7 原文）：任务结束时至少丢了一段证据的任务比例。
   - 按丢失证据所在的跳位（题内第 1/2/3/4 跳）拆分；
   - 按所在题在任务中的位置（第 1–4 题）拆分；
   - 按丢失方式拆分：从未被保留，还是保留过后又被 drop。
4. **证据完整题比例**：四道题里，全部金标准段都在最终保留集合中的题的占比。它是 F1 的代理，**不是 F1**，报告里要写明。
5. **预算占用**：最终 `used / B` 的平均值和最大值；`action_valid` 为假的次数；超预算的次数。后两项都应为 0。
6. **成本**：用 `eval/cost.py` 的 `trajectory_cost`，contexts 取每步执行后的保留集合（`keep_after`，按到达顺序），分别报告不开缓存和开缓存时每任务的平均成本，以及相对 `no_compress` 的比例。

## 7. §6 判据表（只填数，不下结论）

在报告中原样列出 PLAN §6 的四行，每行填写：

- **第 1 行**：两种模式分别列出，中档预算下 oracle 与 bm25 的任务末召回差、单步召回差（单位：百分点），并标注差值是否 ≥ 15。"F1 低 5 个点以上"一格写"未测（需阅读模型）"。
- **第 2 行**：列出同样的差值，以及三档预算下的差值。"接近"PLAN 没有给出数值，**不要自己定阈值**。
- **第 3、4 行**：写"未测（需阅读模型）"。可附上 `no_compress` 与 `oracle` 的单步输入 token 对比，作为"长度"背景。

## 8. 产物

**代码**：

- `src/policies.py`：`latest_policy(mode)` / `bm25_policy(index, ...)` / `no_compress_policy`，或等价的工厂函数。
- `src/feasibility.py`：指标函数。
- `scripts/feasibility.py`：一键运行，参数包括 seed、n_tasks、输出目录。
- `tests/test_policies.py` 和 `tests/test_feasibility.py`：在小型合成状态上用手算值核对指标；覆盖 first-fit 跳过语义、append 从不 drop、预算永不超出、不读 gold。
- 原有 80 个测试必须继续全部通过。

**数据**（只放 5880 的 `data/feasibility/`）：

- `tasks_dev150.jsonl`：每任务的 example ids、步骤（new pid 列表、gold pid、natural_hit、原始 / 替换后的查询、跳答案、题号、跳位）。
- `steps.jsonl`：19 种配置 × 每一步，全部用**语料库 pid**。字段包括 task、config、step、kept_before、new、available_gold、keep_after、used_after、recall、action_valid；bm25 配置另加每个候选的分数。
- `final_contexts.jsonl`：每种配置 × 每任务的最终保留 pid（按到达顺序），加上 4 道题的问题、答案、answer_aliases。以后补 F1 时直接用这个文件，不必重跑。
- `corpus_dev_ntok.jsonl`：pid、title、ntok、渲染文本的 sha1。

**结果**（`eval/feasibility/`，同步到 mirror）：

- `feasibility_report.md`：中文。包含全部表格、每个数的 n、预算三档的数值，以及 §7 判据表。
- `feasibility_results.json`：同一份数字的机器可读版本。
- `answer_leak_samples.md`。
- `run_config.json`：seed、tokenizer 与数据的 sha256、各包版本、运行命令、耗时。

**文档**：

- README 追加"§6 可行性试验（无阅读模型部分）"一节，写入关键数字（PLAN 要求"试验的数字直接写进 README"），并注明 F1 未测。
- NOTES 追加本任务的理解与偏差。
- HANDOFF 追加复现命令。
- SHA256SUMS 重新生成，旧条目对应的文件保留，新增文件纳入；用 `--verify` 核对通过。

## 9. 完成标准

1. 新旧单元测试全部通过；`scripts/feasibility.py` 在 5880 上从头跑通一次。
2. 所有配置 `action_valid` 全真、超预算 0 次。
3. `no_compress` 的召回 = 1、丢失率 = 0。
4. oracle 在每档预算下的任务末召回 ≥ 其他两个策略（若不成立，查明原因写进报告，不要改代码去凑）。
5. 报告里没有任何 F1 数字。
6. 最后一条消息用中文汇报关键数字（三档预算值、自然命中率、非证据段含答案比例、中档下两种模式 oracle vs bm25 的召回差），并以下面这行结尾：

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务02 <一句话>
```

（summary 必须以"任务02"开头，Claude 的监控靠它识别。）
