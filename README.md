# ctx-compressor

> 给多步搜索 Agent 用的 1.7B 上下文压缩器：每次搜索后只留下有用的段落，省 token，也不丢答案。

搜索型 Agent 每一步都把检索结果塞进上下文，一条多跳任务很快就积累几万 token，其中大部分是噪声。本项目基于 Qwen3-1.7B，用 LoRA SFT + GRPO 训练一个**抽取式**压缩器。Agent 每次搜索后，压缩器根据问题、已保留的段落、新返回的段落和 token 预算，决定留哪些、丢哪些：

```json
{"add": [12, 15], "drop": [3]}
```

压缩器只输出段落编号、不改写原文，所以每条保留下来的内容都能追溯到来源。它和主模型解耦，主模型可以换成任意 API 模型。

## 项目特点

- **抽取式、可溯源**：只选编号，不生成摘要，不会编造内容。
- **即插即用**：压缩器是独立的小模型，不需要训练主模型。
- **两种模式**：
  - `reselect`：可以删掉旧段落，压得更狠；
  - `append`：只追加，前缀保持不变，prompt cache 的命中率最高。
- **密集且便宜的奖励**：用数据集自带的证据标注逐步打分，训练循环里不调用大模型。
- **严格的评测**：
  - 和训练零重叠的锁定测试集，只用一次；
  - 真实的 ReAct Agent；
  - 成本按含 prompt caching 的价格算，并用 DeepSeek API 的实际账单做了核对。

## 结果

### 锁定测试集（固定轨迹，MuSiQue dev）

固定的多步检索轨迹，由 Qwen3-8B 在最后作答，共 451 个任务、10,824 个配对题次。

| 方法 | F1 ↑ | 输入成本 / 轨迹（开缓存）↓ |
|---|---|---|
| 不压缩 | 23.5 | 34,649 |
| BM25 | 14.7 | 17,598 |
| 微调重排模型（Qwen3-Reranker-0.6B，同数据训练） | 24.0 | **1,676** |
| 本项目：SFT | 43.8 | 2,530 |
| **本项目：SFT + GRPO** | **44.9** | 2,316 |
| oracle（上限） | 54.3 | 2,109 |

> 留出集里 83% 的证据段在训练中出现过，重排模型靠记忆就能和压缩器打平。所以另建了证据段与训练零重叠的测试集，所有设置在测试前定死。

### 真实 ReAct Agent（主模型 DeepSeek）

自建的 ReAct 搜索 Agent，只有一个 BM25 检索工具，在 MuSiQue / HotpotQA / 2Wiki 上测试，共 339 道题。HotpotQA 和 2Wiki 没参与训练。

| 方法 | F1 ↑ | 实际 API 成本（元 / 任务）↓ |
|---|---|---|
| 不压缩 | 61.9 | 0.0250 |
| LLMLingua-2 | 54.9 | 0.0334 |
| 微调重排模型 | 49.6 | 0.0084 |
| 本项目（reselect） | 62.3 | 0.0081 |
| **本项目（append）** | **63.1** | **0.0079** |
| oracle（上限） | 68.2 | 0.0064 |

> 以上均为单次运行。主模型换成 Qwen3-8B 后（900 道题），本项目和重排模型基本打平（+1.0 / +1.7），比不压缩低 1.0，成本约为不压缩的 1/6。主模型越强、搜得越全，压缩器的优势越明显。详见 [`docs/RESULTS.md`](docs/RESULTS.md)。

## 整体流程

```text
MuSiQue 多跳题
   │  每 4 题拼成一个任务，逐跳 BM25 top-20
   ▼
多步轨迹（约 10 步 / 2.3 万 token）
   │  oracle 保留动作 + 噪声段 + DAgger
   ▼
SFT（LoRA，Qwen3-1.7B）
   │  逐步奖励：−丢失证据 −无关段 −token 用量
   ▼
GRPO
   ├──► 锁定测试集：固定轨迹 + Qwen3-8B 作答
   └──► 真实 ReAct Agent：Qwen3-8B / DeepSeek 作主模型
```

Agent 中的一步：

```text
Thought → search[query] → BM25 top-20 → 压缩器：留哪些 / 丢哪些 → 写回上下文 → … → finish[答案]
```

## 项目结构

```text
ctx-compressor/
├── src/            核心逻辑：轨迹构造、oracle 动作、奖励、Agent 循环、指标
├── scripts/        数据、训练、评测流水线（按任务前缀命名）
├── tests/          CPU 测试
├── eval/           每个实验的报告、结果 JSON 和图；cost.py 是成本模型
├── specs/          实验方案和各任务的任务书
├── docs/           结果汇总、数据说明、完整实验记录
├── requirements/   各环境的精确版本锁
├── requirements.txt
└── LICENSE
```

## 环境安装

```bash
git clone git@github.com:Nevernzx/ctx-compressor.git
cd ctx-compressor
pip install -r requirements.txt                  # 数据处理与 CPU 测试
# 训练与推理环境的精确版本见：
#   requirements/train.lock.txt   (torch 2.8, transformers 4.57, TRL, PEFT)
#   requirements/reader.lock.txt  (vLLM 0.11)
```

## 训练

```bash
# 1. SFT：构造 oracle 动作标签 + 噪声段，DAgger 一轮，LoRA 训练，留出集选 checkpoint
python -m scripts.run_sft_v3_pipeline
# 2. 把 SFT LoRA 合并进基座
python -m scripts.merge_sft_lora
# 3. GRPO（ms-swift）：闭环状态重打分、导出数据、训练、留出集选 checkpoint
python -m scripts.grpo_r4_prepare prepare
python -m scripts.run_grpo_r4_pipeline
# 基线：同数据微调的重排模型
python -m scripts.train_reranker_v1
```

## 评测

```bash
# 锁定测试集（固定轨迹 + Qwen3-8B 作答）
python -m scripts.test_v1_pipeline detach
# 真实 ReAct Agent：本地 Qwen3-8B / DeepSeek API
python -m scripts.agent_v1_main launch
python -m scripts.agent_ds_run launch main --approved
# CPU 测试
CUDA_VISIBLE_DEVICES= python -m pytest tests/
```

> 这些流水线是在 2 × RTX 5880 Ada 的服务器上跑的，模型、数据和输出路径写在各脚本开头，换机器需要先改路径。模型权重和轨迹数据不在仓库里。

## 数据来源

| 数据集 | 用途 | 许可 |
|---|---|---|
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | 训练、留出集、锁定测试集 | CC BY 4.0 |
| [HotpotQA](https://hotpotqa.github.io/) | 真实 Agent 的跨数据集测试 | CC BY-SA 4.0 |
| [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) | 真实 Agent 的跨数据集测试 | Apache-2.0 |

下载和校验方式见 [`docs/DATA_CARD.md`](docs/DATA_CARD.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | 全部结果、配对比较和局限 |
| [`docs/LAB_NOTEBOOK.md`](docs/LAB_NOTEBOOK.md) | 21 个任务的完整实验记录 |
| [`specs/PLAN_v2.md`](specs/PLAN_v2.md) | 方案设计与相关工作对比 |
| [`eval/`](eval/) | 每个任务的报告与原始结果 |

## 许可

- 代码：[MIT](LICENSE)。
- 模型权重：基于 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)，遵循 Apache-2.0。
- 数据集：遵循各自的许可证，见上方"数据来源"。
