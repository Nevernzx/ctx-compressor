# 任务 01：在 5880 上搭 ctx-compressor 框架（PLAN v2 第 5 节）+ 下载数据集

你负责动手，Claude 负责审核。全程不要中途提问（审批框没人能点）：拿不准的按保守方式处理，写进 NOTES.md；只有数据或安全问题才停下来。

## 0. 先完整读完（只读，不要改）
- `tasks/PLAN_v2.md`：项目方案 v2，是**唯一规格**，从用户的 Claude Docs 导出（表格是简化排版）。本任务书和它冲突时以 PLAN 为准，冲突处写进 NOTES。
- 本任务**只做**三件事：
  - PLAN §5 的框架和验收；
  - 下载数据；
  - PLAN §3 里两项数据核对：占位符格式、train/dev 段落重叠。
- **不做**：
  - §6 可行性试验、任何训练；
  - 下载模型权重、用 GPU、调用付费 API。

## 1. 位置与机器
- **正式目录**：5880 上的 `/root/workplace/ctx-compressor`（新建；已存在就停下来报告）。数据、环境、产物都在 5880。
- **本机镜像**：3080 上的 `~/workplace/experiments/ctx-compressor/mirror/`，只放代码和文档，不放数据。
  - 在这里写代码，再用 `rsync -a`（不加 `--delete`）同步到 5880。
  - 3080 上你只写 `mirror/` 这一个目录。
- **怎么操作 5880**：从 3080 执行 `ssh 5880 '...'`（别名已配好、免密，登录后是 root）。
  - 5880 本机连不上 OpenAI，所以你留在 3080 上远程操作。
  - 本会话的沙箱已开联网；如果 ssh 还是被沙箱挡住，不要绕过：把要执行的确切命令写进本机 `mirror/HANDOFF.md`，以 status=fail 结束。
- **5880 环境（11-17 实测）**：
  - Ubuntu 24.04 容器，128 核 / 125 GB；
  - `/root/anaconda3`：conda 25.5.1，base python 3.13.5；
  - pip 已配清华镜像；rsync、git、unzip 都有。
- **5880 已知的坑**：
  - 单引号 ssh 命令里写 heredoc，引号会被吃掉 → 文件在本机写好再 rsync / scp。
  - 没有 `/usr/bin/time`。
  - 超过 2 分钟的命令用 `setsid nohup ... > log 2>&1 &`，配 pid 文件，再轮询。
  - `pgrep -f` 会匹配到 ssh 自己那条命令，查进程用 pid 文件。
- **磁盘**：5880 的 `/`（含 `/root/workplace`）只剩约 25 GB。
  - 下载前先 `df -h /`。
  - 预计下完后 `/` 剩余低于 15 GB 时，把原始数据放到 `/root/userfolder/ctx-compressor_data/raw/`，再把 `data/raw` 软链过去，写进 README。
- **不碰别人的东西**：
  - 不改 5880 上其他目录（`/root/workplace` 下其他项目、`/root/userfolder` 下已有目录）；
  - 不往已有 conda 环境（base、medsam3、PTRNet、mil_mars）里装包；
  - 不删任何文件（你自己的临时文件除外）；
  - 不用 GPU：0 号卡是别人的，1 号卡留给别的任务。

## 2. 网络与下载规则（重要）
- **所有下载都在 5880 上做**（5880 直连，没有代理）。
  - 3080 shell 的环境变量里有按流量计费的代理，**禁止在 3080 上下载任何数据**，也禁止在任何地方用代理。
  - 5880 的 `~/.codex/config.toml` 里有第三方中转，不要用。
- **5880 能连**（11-17 实测）：github.com（慢，约 10 秒）、raw.githubusercontent.com、www.modelscope.cn、pypi.tuna。
- **5880 连不上**：drive.google.com、dropbox.com、huggingface.co、hf-mirror.com、curtis.ml.cmu.edu。
- **官方源**：
  - MuSiQue 官方包在 Google Drive（stonybrooknlp/musique 的 `download_data.sh`）；
  - 2Wiki 在 Dropbox；
  - HotpotQA 在 CMU。
- **找源的规则**：
  - 先找 5880 能免费直连的镜像，比如 ModelScope 数据集、GitHub release / 仓库文件、其他国内镜像。
  - 镜像要满足：
    - 字段和官方原始格式一致；只有转换格式（比如 parquet）时，必须能无损还原成原始字段，并且核对过；
    - 条数和下面的预期值核对。
  - 一个必需数据集找不到免费源时：
    - 不要用代理；
    - 把试过的地址、报错和官方文件大小（知道的话）写进 HANDOFF；
    - 继续做别的。
- **"下载完成"不算数**：每个文件都要核对。
  - 有官方 sha256 就核 sha256；
  - 没有就核大小、能否完整解析、条数。

## 3. 要下载的东西（`data/raw/` 下按数据集分目录）
| 数据 | 文件 | 是否必需 | 预期条数（Claude 的记忆值，用来核对；对不上就如实记录，不要硬凑） |
|---|---|---|---|
| MuSiQue v1.0 | `musique_ans_v1.0_{train,dev}.jsonl` | **必需** | Ans train 19,938 / dev 2,417 |
| MuSiQue v1.0 | `musique_ans_v1.0_test`、`musique_full_v1.0_{train,dev,test}` | 能拿到就拿 | Ans test 2,459；Full train 39,876 / dev 4,834 |
| HotpotQA | `hotpot_dev_distractor_v1.json` | **必需** | 7,405 |
| HotpotQA | dev fullwiki | 能拿到就拿；train 不要 | 7,405 |
| 2WikiMultiHopQA | dev | **必需** | 12,576 |
| 2WikiMultiHopQA | train / test | 只能整包下载时才一起留（总大小 < 2 GB） | 167,454 / 12,576 |
| Qwen3-1.7B 分词器 | 只下 tokenizer 相关文件和 config，**不下 `*.safetensors`** | **必需** | — |

- **MuSiQue 原始字段**：`id`、`paragraphs[{idx, title, paragraph_text, is_supporting}]`、`question`、`question_decomposition[{id, question, answer, paragraph_support_idx}]`、`answer`、`answer_aliases`、`answerable`。
- **Qwen3 分词器**：用 ModelScope 的 `Qwen/Qwen3-1.7B`，放 `data/raw/qwen3_tokenizer/`（或 `models/qwen3-1.7b-tokenizer/`，选一个写进 README）。
- **两份清单**：
  - `data/raw/MANIFEST.json`：每个文件的来源 URL、大小、sha256、条数、下载时间。
  - `DATA_CARD.md`：中文，写来源（官方还是镜像，镜像的话怎么核对的）、许可、条数与预期对照、每个数据集一条样例的字段结构。
- **许可**：
  - 能从源页面核实的写出处；
  - 核实不了的标"未核实"；
  - 参考值：MuSiQue CC BY 4.0、HotpotQA CC BY-SA 4.0、Qwen3 Apache 2.0；2Wiki 看仓库 LICENSE。

## 4. 环境
- 新建一个独立环境。优先 `conda create -n ctxc python=3.11`；conda 源连不上，就用 `/root/anaconda3/bin/python -m venv /root/workplace/ctx-compressor/.venv`。
- 只装框架要用的包：`bm25s`、`transformers` / `tokenizers`（只用分词器，**不装 torch**）、`numpy`、`scipy`、`pytest`，以及确实需要的小包（比如 `modelscope`，下载用）。
- 把 `pip freeze` 存成 `requirements.lock.txt`，README 里写清怎么激活环境。

## 5. 框架代码（PLAN §5）
目录按 PLAN §5，`src/build_traj.py`、`src/reward.py`、`eval/cost.py` 的文件名、函数名、函数签名都保留（可以加可选参数），另加 `tests/`、`scripts/`。PLAN 里的三段代码照实现，只有三种情况可以改：
- 真实字段名和 PLAN 不同；
- 发现 bug；
- 下面 Claude 指定的改动。

每处改动都写进 NOTES.md：改了什么、为什么。

**Claude 读 PLAN 代码时发现的问题，按下面的方式处理：**
1. **标题没算进去**：
   - 现状：PLAN 里 `ntok` 只数 `paragraph_text`，`text` 也不带标题，但 BM25 索引用的是"标题 + 正文"。MuSiQue 的实体名常常只出现在标题里，不带标题的话，阅读模型可能看不出这一段在讲谁。
   - 改法：给模型看的片段渲染成"标题 + 正文"（具体格式你定，写进 README），`ntok` 数渲染后的片段串，保证预算和模型实际看到的 token 一致。
2. **随机数**：所有随机都用显式种子的 `random.Random(seed)`，不用全局 `random`。`simulate_task` 里的 shuffle 也一样。
3. **占位符格式**：PLAN 说后几跳子问题用 `#k` 指代前面的答案，有的是"实体 >> 关系"的形式，仓库 README 没写明。
   - 在真实 dev 上，2 / 3 / 4 跳各打印 5 条 `question_decomposition` 核对，写进 README。
   - 发现 `#k` 以外的引用写法、`paragraph_support_idx` 缺失或为空的，都统计条数；能处理就处理，处理不了的题跳过，记下数目。
4. **"在重选模式的 SFT 数据里随机多加 1–3 个非证据段"（PLAN §4）**：Claude 的理解是——
   - 生成**状态**时，用"oracle + 噪声"策略推进轨迹：在理想动作上随机多加 1–3 个非证据段，加了会超预算就少加；
   - 每一步的 **`sft_target` 仍然是 `oracle_action(s)`**，它会把这些多余的段丢掉，模型由此学会 drop；
   - 只用在重选模式。
   
   按这个理解实现成可选参数（`rollout` 的 `policy` 参数就能接），写测试，并在 NOTES 里标明"这是 Claude 的理解"。
5. **输入模板**（PLAN §5 正文）：实现 `render_input(s)`。
   - 内容：先列任务的全部问题、模式、预算、已用 token，再分"已保留""新返回"两组列片段，每个片段前标编号和 token 数。
   - SFT 目标串是 `sft_target` 的紧凑 JSON，键顺序固定为 add、drop。
   - 注意 Qwen3 的思考模式：以后训练和推理要关掉 thinking，在 README 记一句，现在不用实现。
6. **多题拼接**（PLAN §3）：每个任务用带种子的随机数抽 4 道题，按题依次执行，步数等于各题跳数之和，任务描述列出全部问题。
7. **不做**：ms-swift 奖励插件（ms-swift 还没装），在 README 里列成"下一步"。

**单元测试**（`tests/`，PLAN §5 验收第 1 条，全部要过）：
- **奖励的每个分支**：
  - JSON 解析失败、缺 `add`、编号不是 int；
  - add 不在新返回里、drop 不在已保留里；
  - 只追加模式下 drop 非空；
  - 超预算；
  - recall < 1 时没有奖励项；
  - recall = 1 时奖励项的数值；
  - 金标准为空；
  - add 里有重复编号（按 PLAN 用集合处理，测试里写明这个行为）。
- **SFT 目标一定合法**：在大量随机合成状态上（两种模式），`apply_action(s, oracle_action(s))` 不返回 None、不超预算；带噪声的策略也不超预算。
- **整条 rollout**：每一步保留的总 token ≤ 预算；只追加模式下，每一步的 `keep_after` 都以上一步的 `keep_after` 开头（前缀不变）。
- **成本模型**：复现 PLAN §5 表格的三行——10 步，每步工具返回 3k：
  - 不压缩：165,000 / 44,250；
  - 重选，每步留 2k 且每步内容不同：20,000 / 25,000；
  - 只追加，每步留 300：16,500 / 4,425。
  
  斜杠前是不开缓存，斜杠后是开缓存；读取倍率 0.05、写入倍率 1.25。
- **其他**：`#k` 替换、语料库去重、编号在一条轨迹内稳定、oracle 的贪心顺序。

## 6. 用真实数据验收（PLAN §5 验收第 2 条）+ 数据核对
1. **3 条示例轨迹**：
   - 用 MuSiQue-Ans **dev** 建语料库和 BM25 索引（只用 dev 的段落），用固定种子拼 3 个任务（每个 4 道题），两种模式都跑一遍。
   - 预算先用一个**占位值**（比如 3,000 token），在输出里标明"占位，正式值由 §6 试验定"。
   - 写成 `eval/acceptance/traj_samples.md`。每一步列出：
     - 原始子问题和替换后的查询；
     - 证据段的标题和前 200 字符；
     - 是否自然命中；
     - 新返回的编号、已保留的编号、oracle 动作、执行后的保留结果和 recall。
   - 另写一个自动检查：查询里不再有 `#\d`；每一步证据段都在候选里；保留的 token 不超预算。结果写进同一个文件。
2. **数据核对**，写进 README 和 DATA_CARD：
   - 占位符格式的核对结果（上面第 3 条）；
   - train 和 dev 的段落重叠：分别按"标题 + 正文"完全相同、只按标题相同两种口径统计，给出数目和比例；
   - train / dev 语料库去重后的段落数；
   - dev 段落渲染后 token 数的 P50 / P90 / P99。

## 7. 交付物（都在 5880 的 `/root/workplace/ctx-compressor/`，代码和文档同步一份到本机 `mirror/`）
- `README.md`（中文）：
  - 目录说明、环境激活、怎么跑测试、怎么生成示例轨迹；
  - 数据放在哪里；
  - §3 要求写进 README 的两项（占位符格式、train/dev 重叠）；
  - 下一步清单（§6 可行性试验、ms-swift 插件等）。
- `DATA_CARD.md`、`data/raw/MANIFEST.json`。
- `NOTES.md`：所有偏离 PLAN 的地方和保守处理，逐条编号。
- `HANDOFF.md`：
  - 给 Claude 的核查清单；
  - 确切的复现命令；
  - 测试结果摘要（通过数 / 总数）；
  - 没拿到的数据和原因；
  - 各步用时。
- `SHA256SUMS`：覆盖代码、文档和 `data/raw` 下所有文件。**所有文件都改完之后最后再生成。**

## 8. 结束
全部做完就停，不要接着做 §6。最后一行输出：
`===AGENTDECK_DONE=== status=<ok|fail> summary=<一句话>`
- status=ok：MuSiQue-Ans train/dev 已下载并核对，测试全过，3 条示例轨迹已生成。
- 否则 status=fail，summary 写明卡在哪里。
