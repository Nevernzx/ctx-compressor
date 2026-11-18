# 数据下载与核验记录（给主代理汇总）

全部操作通过 `ssh 5880` 在 `/root/workplace/ctx-compressor` 执行。网络请求统一使用 Python 标准库 `urllib.request.ProxyHandler({})` 或 `curl --noproxy '*'`；未使用代理、付费 API、GPU、模型权重、已有环境安装。3080 只写 `mirror/` 的代码和文档；`scripts/data_sources.json` 仅为 URL、摘要和预期大小组成的下载配置，不含原始数据。

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

## 失败地址与保守处理

所有必需数据均已取得，下面是源查找过程中的失败，不构成剩余阻塞：

1. `https://www.modelscope.cn/api/v1/datasets?Search=musique&PageNumber=1&PageSize=20`（以及其 `Name`、`SortBy` 参数变体）返回 HTTP 500；`POST https://www.modelscope.cn/api/v1/datasets/list` 返回参数错误。阅读公开网页前端源码后找到可用只读搜索 API：`https://www.modelscope.cn/api/v1/dolphin/datasets?PageNumber=1&PageSize=20&Query=musique`。其中搜索必须用 `Query`，`Search/Name/Keyword` 无效；2Wiki 需完整检索词 `2WikiMultiHopQA`。
2. `https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/musique`、`LLM-Research/musique`、`extraordinarylab/musique`、`dgslibisey/MuSiQue` 均 404；`AI-ModelScope/hotpot_qa` 和 `AI-ModelScope/2WikiMultihopQA` 也 404。按可用公开镜像继续查找，没有用代理。
3. `https://raw.githubusercontent.com/Alab-NII/2wikimultihop/main/LICENSE` 首次探测 SSL 握手超时，正式下载又读取超时，已写入 MANIFEST 的 `failed_attempts`。通过同官方 GitHub 的不可变 blob API `https://api.github.com/repos/Alab-NII/2wikimultihop/git/blobs/342ce459a34de236d05b2be3f01e5f08f6c712e8`，设置 `Accept: application/vnd.github.raw+json` 后成功获取 11,337 字节的许可证。无任何代理或第三方中转。
4. 官方 MuSiQue `download_data.sh` 指向 `https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing`；按任务书已有不可达实测，优先寻找直连镜像。本任务未再次下载官方 zip，其官方字节大小未核实。Hotpot 官方 CMU、2Wiki 官方 Dropbox 同样依据任务书不可达信息选择镜像，不声称本次重新测过这些域名。

## 复现命令与耗时

仅在 5880 执行；下载脚本本身禁用代理，无需 ModelScope SDK：

```bash
ssh 5880 'cd /root/workplace/ctx-compressor; setsid nohup /root/anaconda3/bin/python scripts/download_data.py --spec scripts/data_sources.json > logs/download_reproduce.log 2>&1 < /dev/null & echo $! > logs/download_reproduce.pid'
ssh 5880 'cd /root/workplace/ctx-compressor; /root/anaconda3/bin/python scripts/verify_data.py'
```

`scripts/data_sources.json` 是18个文件的完整静态来源配置；下载会重新拉取明确列出的文件，不进行目录快照或下载模型权重。不要与另一个下载进程并发写同一 MANIFEST。

- 数据子任务于北京时间 11:19 左右开始阅读与源发现；Qwen3 于 11:22:33–11:22:39 完成，最慢单文件 8.572 秒。
- 约 11:26 找到 MuSiQue 完整镜像；Ans dev 11.474 秒、Ans train 92.025 秒；Full train 最慢 127.809 秒，其余9.594–30.726秒，并行下载。
- Hotpot/2Wiki 数据文件分别耗时 26.390/26.999/23.720 秒，于 11:29 左右完成；官方许可的读取超时留档后，GitHub blob API 兜底耗时 0.588 秒，最后文件完成时间为 **2026-11-17 11:31:08.511182 +08:00**。
- 最终离线核验 **18/18 文件通过**，错误 0，实测 **7.198 秒**。报告 `eval/acceptance/data_verification.json`，终端日志 `logs/data_verification.log`。
- 寻源、编写下载/逐条验证脚本、并行下载及许可核实总耗时约 12 分钟；各文件精确下载耗时见 MANIFEST。

3条示例轨迹、段落重叠、token分布、占位符审计由主代理验收脚本生成，本记录不重复重新统计。
