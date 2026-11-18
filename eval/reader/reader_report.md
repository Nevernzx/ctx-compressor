# §6 阅读模型 EM / F1：Qwen3-8B

20 种配置 × 同一批 600 道题，共 12,000 个独立 prompt。每任务 4 道题分别回答，复用任务 02 已保存的最终上下文；本任务没有重跑压缩、检索或预算。单次运行，报告均值和 n，不计算置信区间。
模型为本地 Qwen3-8B；单条 user 消息，关闭 thinking，greedy，最多生成 48 token。答案按任务书固定流程解析，对标准答案及全部 aliases 计算 SQuAD 归一化后的最大 EM / F1。报告脚本重新解析原始输出、重新评分，并核对逐题元数据、证据齐全标记和 user 原文 SHA1。

## 各配置整体表现

| 配置 | n 题 | EM | F1 |
| --- | --- | --- | --- |
| no_compress | 600 | 15.667% | 26.416% |
| no_context | 600 | 4.000% | 12.485% |
| latest\|reselect\|tight | 600 | 6.833% | 13.527% |
| latest\|reselect\|mid | 600 | 7.667% | 14.806% |
| latest\|reselect\|loose | 600 | 8.500% | 17.399% |
| latest\|append\|tight | 600 | 3.167% | 8.506% |
| latest\|append\|mid | 600 | 4.000% | 10.083% |
| latest\|append\|loose | 600 | 8.000% | 14.726% |
| bm25\|reselect\|tight | 600 | 10.000% | 17.691% |
| bm25\|reselect\|mid | 600 | 12.000% | 19.673% |
| bm25\|reselect\|loose | 600 | 12.333% | 21.699% |
| bm25\|append\|tight | 600 | 2.833% | 8.601% |
| bm25\|append\|mid | 600 | 4.333% | 10.349% |
| bm25\|append\|loose | 600 | 8.000% | 15.005% |
| oracle\|reselect\|tight | 600 | 44.667% | 56.635% |
| oracle\|reselect\|mid | 600 | 45.500% | 57.564% |
| oracle\|reselect\|loose | 600 | 45.667% | 57.812% |
| oracle\|append\|tight | 600 | 45.333% | 57.269% |
| oracle\|append\|mid | 600 | 45.333% | 57.479% |
| oracle\|append\|loose | 600 | 45.500% | 57.451% |

## 按任务内题目位置的 F1

| 配置 | 第 1 题 | 第 2 题 | 第 3 题 | 第 4 题 |
| --- | --- | --- | --- | --- |
| no_compress | 30.605%（n=150） | 27.237%（n=150） | 21.161%（n=150） | 26.663%（n=150） |
| no_context | 15.685%（n=150） | 12.551%（n=150） | 10.192%（n=150） | 11.512%（n=150） |
| latest\|reselect\|tight | 8.443%（n=150） | 7.721%（n=150） | 7.076%（n=150） | 30.870%（n=150） |
| latest\|reselect\|mid | 8.825%（n=150） | 9.251%（n=150） | 8.281%（n=150） | 32.866%（n=150） |
| latest\|reselect\|loose | 10.365%（n=150） | 11.509%（n=150） | 15.399%（n=150） | 32.322%（n=150） |
| latest\|append\|tight | 13.044%（n=150） | 8.936%（n=150） | 6.565%（n=150） | 5.478%（n=150） |
| latest\|append\|mid | 20.440%（n=150） | 8.117%（n=150） | 4.649%（n=150） | 7.127%（n=150） |
| latest\|append\|loose | 34.246%（n=150） | 12.326%（n=150） | 5.164%（n=150） | 7.170%（n=150） |
| bm25\|reselect\|tight | 20.774%（n=150） | 18.108%（n=150） | 13.424%（n=150） | 18.458%（n=150） |
| bm25\|reselect\|mid | 25.722%（n=150） | 20.072%（n=150） | 11.916%（n=150） | 20.982%（n=150） |
| bm25\|reselect\|loose | 27.183%（n=150） | 19.687%（n=150） | 17.540%（n=150） | 22.385%（n=150） |
| bm25\|append\|tight | 15.834%（n=150） | 8.059%（n=150） | 4.432%（n=150） | 6.081%（n=150） |
| bm25\|append\|mid | 19.948%（n=150） | 8.894%（n=150） | 5.995%（n=150） | 6.560%（n=150） |
| bm25\|append\|loose | 33.172%（n=150） | 14.012%（n=150） | 4.994%（n=150） | 7.843%（n=150） |
| oracle\|reselect\|tight | 64.695%（n=150） | 54.666%（n=150） | 54.359%（n=150） | 52.819%（n=150） |
| oracle\|reselect\|mid | 66.363%（n=150） | 54.922%（n=150） | 55.407%（n=150） | 53.565%（n=150） |
| oracle\|reselect\|loose | 66.688%（n=150） | 54.922%（n=150） | 55.407%（n=150） | 54.232%（n=150） |
| oracle\|append\|tight | 67.354%（n=150） | 55.589%（n=150） | 54.359%（n=150） | 51.773%（n=150） |
| oracle\|append\|mid | 66.021%（n=150） | 54.922%（n=150） | 55.407%（n=150） | 53.565%（n=150） |
| oracle\|append\|loose | 66.353%（n=150） | 54.922%（n=150） | 55.407%（n=150） | 53.121%（n=150） |

## 按题目跳数的 F1

| 配置 | 2 跳 | 3 跳 | 4 跳 |
| --- | --- | --- | --- |
| no_compress | 30.568%（n=306） | 23.825%（n=194） | 18.740%（n=100） |
| no_context | 13.530%（n=306） | 11.012%（n=194） | 12.143%（n=100） |
| latest\|reselect\|tight | 13.945%（n=306） | 13.391%（n=194） | 12.515%（n=100） |
| latest\|reselect\|mid | 16.207%（n=306） | 13.345%（n=194） | 13.354%（n=100） |
| latest\|reselect\|loose | 18.767%（n=306） | 15.816%（n=194） | 16.282%（n=100） |
| latest\|append\|tight | 11.103%（n=306） | 4.913%（n=194） | 7.528%（n=100） |
| latest\|append\|mid | 14.210%（n=306） | 5.629%（n=194） | 6.096%（n=100） |
| latest\|append\|loose | 19.388%（n=306） | 10.782%（n=194） | 8.113%（n=100） |
| bm25\|reselect\|tight | 20.874%（n=306） | 15.152%（n=194） | 12.878%（n=100） |
| bm25\|reselect\|mid | 23.090%（n=306） | 17.820%（n=194） | 12.813%（n=100） |
| bm25\|reselect\|loose | 24.826%（n=306） | 20.631%（n=194） | 14.202%（n=100） |
| bm25\|append\|tight | 11.367%（n=306） | 4.971%（n=194） | 7.182%（n=100） |
| bm25\|append\|mid | 14.446%（n=306） | 5.277%（n=194） | 7.655%（n=100） |
| bm25\|append\|loose | 19.899%（n=306） | 10.040%（n=194） | 9.661%（n=100） |
| oracle\|reselect\|tight | 64.831%（n=306） | 50.506%（n=194） | 43.443%（n=100） |
| oracle\|reselect\|mid | 65.648%（n=306） | 51.601%（n=194） | 44.397%（n=100） |
| oracle\|reselect\|loose | 65.975%（n=306） | 51.807%（n=194） | 44.484%（n=100） |
| oracle\|append\|tight | 65.267%（n=306） | 51.906%（n=194） | 43.198%（n=100） |
| oracle\|append\|mid | 65.321%（n=306） | 51.807%（n=194） | 44.484%（n=100） |
| oracle\|append\|loose | 65.648%（n=306） | 51.205%（n=194） | 44.482%（n=100） |

## 按证据是否齐全的 F1

证据齐全指该题全部金标准段 pid 都出现在该任务该配置的最终保留集合中；沿用任务 02 的定义。no_context 没有上下文，全部归入证据不齐全。空分组显示未适用，不记作零分。

| 配置 | 证据齐全 | 证据不齐全 |
| --- | --- | --- |
| no_compress | 26.416%（n=600） | 未适用（n=0） |
| no_context | 未适用（n=0） | 12.485%（n=600） |
| latest\|reselect\|tight | 36.250%（n=8） | 13.220%（n=592） |
| latest\|reselect\|mid | 50.343%（n=25） | 13.261%（n=575） |
| latest\|reselect\|loose | 35.490%（n=95） | 13.995%（n=505） |
| latest\|append\|tight | 33.333%（n=6） | 8.255%（n=594） |
| latest\|append\|mid | 45.599%（n=26） | 8.475%（n=574） |
| latest\|append\|loose | 45.938%（n=106） | 8.029%（n=494） |
| bm25\|reselect\|tight | 63.205%（n=39） | 14.527%（n=561） |
| bm25\|reselect\|mid | 60.030%（n=56） | 15.519%（n=544） |
| bm25\|reselect\|loose | 47.458%（n=99） | 16.608%（n=501） |
| bm25\|append\|tight | 46.154%（n=13） | 7.770%（n=587） |
| bm25\|append\|mid | 47.778%（n=34） | 8.101%（n=566） |
| bm25\|append\|loose | 41.648%（n=108） | 9.157%（n=492） |
| oracle\|reselect\|tight | 57.660%（n=585） | 16.667%（n=15） |
| oracle\|reselect\|mid | 57.564%（n=600） | 未适用（n=0） |
| oracle\|reselect\|loose | 57.812%（n=600） | 未适用（n=0） |
| oracle\|append\|tight | 58.401%（n=585） | 13.111%（n=15） |
| oracle\|append\|mid | 57.479%（n=600） | 未适用（n=0） |
| oracle\|append\|loose | 57.451%（n=600） | 未适用（n=0） |

## 每题 prompt token 分布

这里使用实际提交给 vLLM 的 chat-template 后 prompt token 数；不是任务 02 的片段预算。输入不得截断，每条 prompt 加上 48 个生成 token 的预留必须不超过 40,960。

| 配置 | n | mean | P50 | P90 | max |
| --- | --- | --- | --- | --- | --- |
| no_compress | 600 | 23,490.703 | 23,564 | 28,856.3 | 32,382 |
| no_context | 600 | 64.997 | 63.5 | 76 | 100 |
| latest\|reselect\|tight | 600 | 1,931.123 | 1,932 | 1,950.1 | 1,981 |
| latest\|reselect\|mid | 600 | 2,872.243 | 2,874 | 2,890 | 2,922 |
| latest\|reselect\|loose | 600 | 5,686.037 | 5,688 | 5,704 | 5,727 |
| latest\|append\|tight | 600 | 1,938.73 | 1,939 | 1,954 | 1,979 |
| latest\|append\|mid | 600 | 2,876.457 | 2,877 | 2,892 | 2,917 |
| latest\|append\|loose | 600 | 5,690.39 | 5,691 | 5,707 | 5,736 |
| bm25\|reselect\|tight | 600 | 1,933.59 | 1,934.5 | 1,953 | 1,981 |
| bm25\|reselect\|mid | 600 | 2,873.997 | 2,876 | 2,890.1 | 2,920 |
| bm25\|reselect\|loose | 600 | 5,683.503 | 5,686 | 5,703 | 5,724 |
| bm25\|append\|tight | 600 | 1,938.557 | 1,939 | 1,954 | 1,980 |
| bm25\|append\|mid | 600 | 2,876.957 | 2,878 | 2,892 | 2,912 |
| bm25\|append\|loose | 600 | 5,689.197 | 5,689 | 5,705 | 5,731 |
| oracle\|reselect\|tight | 600 | 1,458.13 | 1,483.5 | 1,830 | 1,963 |
| oracle\|reselect\|mid | 600 | 1,510.07 | 1,492.5 | 1,960.2 | 2,827 |
| oracle\|reselect\|loose | 600 | 1,510.07 | 1,492.5 | 1,960.2 | 2,827 |
| oracle\|append\|tight | 600 | 1,474.31 | 1,490.5 | 1,851 | 1,963 |
| oracle\|append\|mid | 600 | 1,510.07 | 1,492.5 | 1,960.2 | 2,827 |
| oracle\|append\|loose | 600 | 1,510.07 | 1,492.5 | 1,960.2 | 2,827 |

**no_compress 最大 prompt：32,382 token**；加 48 token 预留后为 32,430，上限 40,960。分位数采用线性插值，n 均为实际题数。

### PLAN §6 判据表（填数，不作开工判断）

前四行观察条件保留 PLAN 原文。oracle − BM25 和 oracle − no_compress 的差值均以百分点（pp）表示；不对‘接近’或‘明显更低’另定阈值，不据本表调整任务、检索或预算。

| PLAN 原观察条件 | 证据指标 / 配对范围 | 阅读模型观测 |
| --- | --- | --- |
| 中档预算下，BM25 的证据召回比 oracle 低 15 个点以上，F1 低 5 个点以上 | reselect/mid：任务末召回差 64.409 pp（≥15：是，n=150任务）；单步按步差 21.066 pp（≥15：是，n=1594步）；单步按任务差 20.867 pp（n=150任务）<br>append/mid：任务末召回差 85.717 pp（≥15：是，n=150任务）；单步按步差 40.928 pp（≥15：是，n=1594步）；单步按任务差 40.857 pp（n=150任务） | reselect/mid：oracle 57.564% − BM25 19.673% = 37.891 pp（≥5：是；各 n=600题）<br>append/mid：oracle 57.479% − BM25 10.349% = 47.129 pp（≥5：是；各 n=600题） |
| BM25 已接近 oracle | reselect/tight：任务末差 68.894 pp（n=150任务）；单步按步差 26.357 pp（n=1594步）；单步按任务差 26.093 pp（n=150任务）<br>reselect/mid：任务末差 64.409 pp（n=150任务）；单步按步差 21.066 pp（n=1594步）；单步按任务差 20.867 pp（n=150任务）<br>reselect/loose：任务末差 51.753 pp（n=150任务）；单步按步差 12.033 pp（n=1594步）；单步按任务差 11.835 pp（n=150任务）<br>append/tight：任务末差 88.041 pp（n=150任务）；单步按步差 48.921 pp（n=1594步）；单步按任务差 48.776 pp（n=150任务）<br>append/mid：任务末差 85.717 pp（n=150任务）；单步按步差 40.928 pp（n=1594步）；单步按任务差 40.857 pp（n=150任务）<br>append/loose：任务末差 72.683 pp（n=150任务）；单步按步差 23.168 pp（n=1594步）；单步按任务差 22.982 pp（n=150任务）<br>差值仅作描述，不新增‘接近’阈值。 | reselect/tight：38.944 pp（各 n=600题）<br>reselect/mid：37.891 pp（各 n=600题）<br>reselect/loose：36.114 pp（各 n=600题）<br>append/tight：48.667 pp（各 n=600题）<br>append/mid：47.129 pp（各 n=600题）<br>append/loose：42.446 pp（各 n=600题） |
| 不压缩的 F1 与 oracle 接近 | 两种 oracle 中档，使用相同 600 题逐题配对。 | reselect/mid，n=600：oracle − no_compress F1=31.148 pp；逐题 oracle 更高/相等/no_compress 更高=263/310/27；EM 配对 2×2（行=oracle 对/错，列=no_compress 对/错）=[[87,186],[7,320]]<br>append/mid，n=600：oracle − no_compress F1=31.062 pp；逐题 oracle 更高/相等/no_compress 更高=263/310/27；EM 配对 2×2（行=oracle 对/错，列=no_compress 对/错）=[[87,185],[7,321]] |
| 不压缩的 F1 明显低于 oracle | 两种 oracle 中档，使用相同 600 题逐题配对。 | reselect/mid，n=600：oracle − no_compress F1=31.148 pp；逐题 oracle 更高/相等/no_compress 更高=263/310/27；EM 配对 2×2（行=oracle 对/错，列=no_compress 对/错）=[[87,186],[7,320]]<br>append/mid，n=600：oracle − no_compress F1=31.062 pp；逐题 oracle 更高/相等/no_compress 更高=263/310/27；EM 配对 2×2（行=oracle 对/错，列=no_compress 对/错）=[[87,185],[7,321]] |
| 附加闭卷对照 no_context | 无段落上下文，同一批 n=600 题。 | EM 4.000%；F1 12.485%；n=600题 |


## no_compress 与 oracle 中档的逐题配对

### oracle|reselect|mid 对 no_compress

同一批 n=600 题；oracle F1 57.564%，no_compress F1 26.416%；oracle − no_compress = 31.148 pp。
逐题 F1：oracle 更高 263，相等 310，no_compress 更高 27。

| EM 配对（n=600） | no_compress 正确 | no_compress 错误 |
| --- | --- | --- |
| oracle 正确 | 87 | 186 |
| oracle 错误 | 7 | 320 |

### oracle|append|mid 对 no_compress

同一批 n=600 题；oracle F1 57.479%，no_compress F1 26.416%；oracle − no_compress = 31.062 pp。
逐题 F1：oracle 更高 263，相等 310，no_compress 更高 27。

| EM 配对（n=600） | no_compress 正确 | no_compress 错误 |
| --- | --- | --- |
| oracle 正确 | 87 | 185 |
| oracle 错误 | 7 | 321 |

## 输出格式与完整性检查

`finish_reason=length` 表示用满任务书固定的 48-token 生成额度，单独统计；它不等于输入上下文被截断。这些原始输出保留、照常解析评分，不剔除、不延长生成、不选择性补跑。输入完整性由每条 prompt+48 长度检查和推理时 token 序列一致性断言核验。

| 配置 | n | >20 词答案数 / 比例 | 空解析答案 | 结束原因 | prompt+48 超限 |
| --- | --- | --- | --- | --- | --- |
| no_compress | 600 | 15 / 2.500% | 0 | {"length": 11, "stop": 589} | 0 |
| no_context | 600 | 9 / 1.500% | 0 | {"length": 9, "stop": 591} | 0 |
| latest\|reselect\|tight | 600 | 77 / 12.833% | 0 | {"length": 17, "stop": 583} | 0 |
| latest\|reselect\|mid | 600 | 73 / 12.167% | 0 | {"length": 16, "stop": 584} | 0 |
| latest\|reselect\|loose | 600 | 40 / 6.667% | 0 | {"length": 9, "stop": 591} | 0 |
| latest\|append\|tight | 600 | 86 / 14.333% | 0 | {"length": 18, "stop": 582} | 0 |
| latest\|append\|mid | 600 | 79 / 13.167% | 0 | {"length": 18, "stop": 582} | 0 |
| latest\|append\|loose | 600 | 29 / 4.833% | 0 | {"length": 13, "stop": 587} | 0 |
| bm25\|reselect\|tight | 600 | 32 / 5.333% | 0 | {"length": 8, "stop": 592} | 0 |
| bm25\|reselect\|mid | 600 | 26 / 4.333% | 0 | {"length": 8, "stop": 592} | 0 |
| bm25\|reselect\|loose | 600 | 11 / 1.833% | 0 | {"length": 8, "stop": 592} | 0 |
| bm25\|append\|tight | 600 | 89 / 14.833% | 0 | {"length": 16, "stop": 584} | 0 |
| bm25\|append\|mid | 600 | 77 / 12.833% | 0 | {"length": 19, "stop": 581} | 0 |
| bm25\|append\|loose | 600 | 29 / 4.833% | 0 | {"length": 14, "stop": 586} | 0 |
| oracle\|reselect\|tight | 600 | 6 / 1.000% | 0 | {"length": 5, "stop": 595} | 0 |
| oracle\|reselect\|mid | 600 | 6 / 1.000% | 0 | {"length": 4, "stop": 596} | 0 |
| oracle\|reselect\|loose | 600 | 6 / 1.000% | 0 | {"length": 3, "stop": 597} | 0 |
| oracle\|append\|tight | 600 | 6 / 1.000% | 0 | {"length": 3, "stop": 597} | 0 |
| oracle\|append\|mid | 600 | 6 / 1.000% | 0 | {"length": 3, "stop": 597} | 0 |
| oracle\|append\|loose | 600 | 6 / 1.000% | 0 | {"length": 5, "stop": 595} | 0 |

| 检查项 | 结果 |
| --- | --- |
| passed | true |
| prediction_matrix | "20 configurations × 600 distinct questions" |
| reparsed_and_rescored_records | 12000 |
| metadata_and_user_sha1_checked_records | 12000 |
| corpus_sha1_checked_paragraphs | 21100 |
| task02_final_contexts_checked | 2850 |
| prompt_plus_48_over_limit | 0 |
| input_truncated_count | 0 |
| input_truncated_flag_missing_count | 0 |
| generation_length_finish_count | 207 |
| generation_length_finish_rate | 0.01725 |
| output_limit_interpretation | "The fixed 48-token generation limit is reported without filtering or reruns; it is not input-context truncation" |
| finish_reason_missing_count | 0 |
| prompt_token_verification | "Recorded prompt counts checked against limit; no model/tokenizer loaded by reporting script" |
| no_compress_max_prompt_tokens | 32382 |

报告脚本不加载模型或分词器：prompt token 数来自预测记录，并检查长度上限；user 消息原文 SHA1 则由任务 02 上下文和原始 dev 独立重建核对。[机器可读结果](reader_results.json)保存同一份数字；[推理运行配置](run_config.json)记录 GPU、环境、生成参数与资源用量。
