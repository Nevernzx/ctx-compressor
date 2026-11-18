# 数据卡：任务 01

数据只存放在 5880；本卡不包含原始数据副本。所有文件来自免费直连来源，原始格式未转换。
最终离线核验：**18/18 文件通过**，错误 0。详见 `data/raw/MANIFEST.json` 与 `eval/acceptance/data_verification.json`。

## 最终结果

原始数据目录：5880 `/root/workplace/ctx-compressor/data/raw/`。根分区下载前约 25 GB 空闲，下载后约 23 GB，未触发低于 15 GB 的迁移条件，无软链。18 个下载文件共 **1,057,170,141 字节**，不含 MANIFEST 本身；完整解析、逐条原始字段检查、大小与摘要复核全部通过。

| 文件 | 字节 | 条数 | 任务书预期 |
|---|---:|---:|---:|
| musique_ans_v1.0_train.jsonl | 241,046,755 | 19,938 | 19,938 |
| musique_ans_v1.0_dev.jsonl | 30,439,728 | 2,417 | 2,417 |
| musique_ans_v1.0_test.jsonl | 28,366,572 | 2,459 | 2,459 |
| musique_full_v1.0_train.jsonl | 476,696,984 | 39,876 | 39,876 |
| musique_full_v1.0_dev.jsonl | 59,422,562 | 4,834 | 4,834 |
| musique_full_v1.0_test.jsonl | 55,558,292 | 4,918 | 未给出；此处为实测 |
| hotpot_dev_distractor_v1.json | 46,320,117 | 7,405 | 7,405 |
| hotpot_dev_fullwiki_v1.json | 47,454,698 | 7,405 | 7,405 |
| 2wikimultihopqa/dev.json | 55,934,464 | 12,576 | 12,576 |

Qwen3 分词器放 `data/raw/qwen3_tokenizer/`：`tokenizer.json`、`tokenizer_config.json`、`config.json`、`vocab.json`、`merges.txt`、`LICENSE`。其余三个文件为 MuSiQue 官方 LICENSE、HotpotQA 官方 README 许可声明快照、2Wiki 官方 LICENSE。

所有必需数据、所有任务书要求尽力获取的 MuSiQue/Hotpot 可选文件均已获取。2Wiki 的 dev 可单独下载，因此按任务书未下载其 train/test；未下载 HotpotQA train。

## 来源与核验边界

- MuSiQue：ModelScope 镜像 [voidful/MuSiQue](https://www.modelscope.cn/datasets/voidful/MuSiQue)，revision `aa92b9e2a0bb07120885f620e6f9116ea4f62376`，原始 JSONL，无格式转换。6 个文件核对镜像 API 给出的 SHA256 和大小；Ans train/dev 和 Full train/dev 的大小及 SHA256 还与独立镜像 [qboodp/musique](https://www.modelscope.cn/datasets/qboodp/musique) 的元数据一致。没有取得官方原始包的 SHA256，**不宣称完成与官方包逐字节比对**。所有 train/dev 逐条核对原始字段与嵌套结构；Ans train/dev 的 `paragraph_support_idx` 均为存在于本题段落的整数。
- MuSiQue test 原始文件公开部分只有 `id`、`paragraphs[{idx,title,paragraph_text}]`、`question`，没有答案或 decomposition；保留原样，未补造标签。MuSiQue Full 的不可回答题允许 null 支撑索引：Full train 29,341 个 null、Full dev 3,675 个 null，均在完整校验时记录；任务验收仅使用 Ans dev。
- HotpotQA：ModelScope 镜像 [OpenDataLab/HotpotQA](https://www.modelscope.cn/datasets/OpenDataLab/HotpotQA)，revision `8f4804ba8c47c8d911bdf8d5fb6dabb23535f544`。两个原始 JSON 数组直接下载，核对镜像 SHA256、大小及 7,405 条；每条核对 `_id/question/answer/type/level/context/supporting_facts` 和段落、句子、证据索引的结构。未获取官方 SHA256。
- 2Wiki：ModelScope 镜像 [voidful/2WikiMultihopQA](https://www.modelscope.cn/datasets/voidful/2WikiMultihopQA)，revision `75d023a1becf3e2e73401639536eae3d0bc82d2c`。原始 dev JSON 数组，核对镜像 SHA256、大小、12,576 条和每条 `_id/question/answer/type/context/supporting_facts/evidences` 字段。未获取官方 SHA256。
- Qwen3：官方 [Qwen/Qwen3-1.7B](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B)，只下载明确白名单中的 tokenizer/config/许可文件。六个文件均核对官方 ModelScope API 的 SHA256 和大小，逐文件 URL 固定到各自 revision。未下载任何权重或权重索引。
- 每个文件的来源 URL、大小、SHA256、下载时间、条数和样例字段结构在 `data/raw/MANIFEST.json`。MANIFEST 不记录自己的 hash，最终 `SHA256SUMS` 应覆盖 MANIFEST。

## 许可

| 数据 | 核实结果 | 核实出处 |
|---|---|---|
| MuSiQue | CC BY 4.0 | [官方 LICENSE](https://github.com/StonyBrookNLP/musique/blob/main/LICENSE)，已下载快照 |
| HotpotQA | 数据 CC BY-SA 4.0；仓库代码 Apache 2.0 | [官方 README 的 License 节](https://github.com/hotpotqa/hotpot#license)，已下载快照 |
| 2Wiki | 官方仓库 LICENSE 为 Apache 2.0，版权 2020 Xanh Ho | [官方 LICENSE](https://github.com/Alab-NII/2wikimultihop/blob/main/LICENSE)，已下载快照；仓库未另行区分数据专用许可 |
| Qwen3 | Apache 2.0 | [官方 ModelScope LICENSE](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B/file/view/master/LICENSE)，已下载快照 |

MuSiQue 两个镜像的页面元数据标注 Apache，但官方 LICENSE 明确 CC BY 4.0。以官方为准，在 NOTES 中记录这一不一致；不采用镜像的许可标签。


## 单条样例的实际字段结构

下列结构直接取自 MANIFEST 中相应文件第一条记录的解析结果；值为实际类型名，空数组保留为空，未臆造其中的元素类型。

### `musique/musique_ans_v1.0_train.jsonl`

```json
{
  "id": "str",
  "paragraphs": [
    {
      "idx": "int",
      "title": "str",
      "paragraph_text": "str",
      "is_supporting": "bool"
    }
  ],
  "question": "str",
  "question_decomposition": [
    {
      "id": "int",
      "question": "str",
      "answer": "str",
      "paragraph_support_idx": "int"
    }
  ],
  "answer": "str",
  "answer_aliases": [],
  "answerable": "bool"
}
```

### `musique/musique_ans_v1.0_test.jsonl`

```json
{
  "id": "str",
  "paragraphs": [
    {
      "idx": "int",
      "title": "str",
      "paragraph_text": "str"
    }
  ],
  "question": "str"
}
```

### `hotpotqa/hotpot_dev_distractor_v1.json`

```json
{
  "_id": "str",
  "answer": "str",
  "question": "str",
  "supporting_facts": [
    [
      "str",
      "int"
    ]
  ],
  "context": [
    [
      "str",
      [
        "str",
        "str",
        "str"
      ]
    ]
  ],
  "type": "str",
  "level": "str"
}
```

### `hotpotqa/hotpot_dev_fullwiki_v1.json`

```json
{
  "_id": "str",
  "answer": "str",
  "question": "str",
  "supporting_facts": [
    [
      "str",
      "int"
    ]
  ],
  "context": [
    [
      "str",
      [
        "str",
        "str",
        "str",
        "str",
        "str"
      ]
    ]
  ],
  "type": "str",
  "level": "str"
}
```

### `2wikimultihopqa/dev.json`

```json
{
  "_id": "str",
  "type": "str",
  "question": "str",
  "context": [
    [
      "str",
      [
        "str",
        "str",
        "str",
        "str"
      ]
    ]
  ],
  "supporting_facts": [
    [
      "str",
      "int"
    ]
  ],
  "evidences": [
    [
      "str",
      "str",
      "str"
    ]
  ],
  "answer": "str"
}
```

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


2 / 3 / 4 跳各 5 条完整原始分解样例已逐条打印在 [README](README.md) 和 [数据核对全文](eval/acceptance/data_audit.md)。

下载失败尝试、可复现命令与各步耗时见 [HANDOFF](HANDOFF.md)；镜像许可元数据冲突与保守处理见 [NOTES](NOTES.md)。
