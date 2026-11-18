"""Append TASK_02 results to existing documentation without changing its prefix."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MARKER = "## 任务02：§6 可行性试验（无阅读模型部分）"


def rate(value):
    return f"{100 * value:.3f}%"


def main():
    results = json.loads((ROOT / "eval/feasibility/feasibility_results.json").read_text())
    run = json.loads((ROOT / "eval/feasibility/run_config.json").read_text())
    suite = ET.parse(ROOT / "eval/feasibility/tests.xml").getroot().find("testsuite")
    tests = int(suite.attrib["tests"])
    assert int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == int(suite.attrib["skipped"]) == 0
    budgets = results["budgets"]
    stats = results["distributions"]
    natural = results["retrieval"]["overall"]
    leakage, multi = results["answer_leak"]["all"], results["answer_leak"]["multiword"]
    rendered = results["answer_leak"]["rendered_all"]
    configs = results["configs"]
    rows, differences = [], []
    for mode in ("reselect", "append"):
        oracle = configs[f"oracle|{mode}|mid"]["metrics"]
        bm25 = configs[f"bm25|{mode}|mid"]["metrics"]
        delta = 100 * (oracle["final_recall"]["mean"] - bm25["final_recall"]["mean"])
        step_delta = 100 * (oracle["step_recall"]["micro"]["mean"] - bm25["step_recall"]["micro"]["mean"])
        rows.append(f"| {mode} | {rate(oracle['final_recall']['mean'])} | {rate(bm25['final_recall']['mean'])} | {delta:.3f} | {step_delta:.3f} | {rate(oracle['evidence_loss']['rate'])} | {rate(bm25['evidence_loss']['rate'])} |")
        differences.append(f"{mode} 任务末差 {delta:.3f} pp、单步按步差 {step_delta:.3f} pp")
    prefix = "\n\n" + MARKER + "\n\n"
    readme = prefix + f"""任务02在5880使用现有数据和环境完成一次纯CPU实验：固定seed={results['settings']['seed']}，150个四题任务、600道互不重复dev题；每跳top-20，19种配置共用同一次生成的检索步骤。没有下载、安装、GPU或API调用，没有按结果调整任务。**阅读模型F1 / EM未测（需阅读模型）**；证据完整题比例是代理指标，不是F1。

紧/中/松预算为 **{budgets['tight']} / {budgets['mid']} / {budgets['loose']} token**；dev每任务去重证据token P90={stats['task_evidence_tokens']['p90']:.3f}（n=150），紧档向上取整，中档ceil(1.5×紧档)，松档3×紧档。train参照P90={budgets['train_reference_p90']:.3f}（150任务），不参与定预算、不建train索引。

dev查询校验：{results['dev_filter']['valid']}/{results['dev_filter']['total']}通过，拒绝{results['dev_filter']['rejected']}题。检索自然命中 **{natural['hits']}/{natural['n']} = {rate(natural['rate'])}**。非证据段先排除整任务全部gold，再用归一化后的本跳答案匹配正文：段级 **{leakage['paragraph']['hits']}/{leakage['paragraph']['n']} = {rate(leakage['paragraph']['rate'])}**，跳级 **{leakage['hop']['hits']}/{leakage['hop']['n']} = {rate(leakage['hop']['rate'])}**；≥2词答案的段级为 **{rate(multi['paragraph']['rate'])}**（n={multi['paragraph']['n']}），跳级为 **{rate(multi['hop']['rate'])}**（n={multi['hop']['n']}）。标题+正文敏感性口径：段级 **{rate(rendered['paragraph']['rate'])}**（n={rendered['paragraph']['n']}），跳级 **{rate(rendered['hop']['rate'])}**（n={rendered['hop']['n']}）。

中档的任务末召回用该任务全部唯一gold为分母；下表差值为oracle减BM25，单位为百分点。每配置n=150任务/{natural['n']}步。

| 模式 | oracle任务末召回 | BM25任务末召回 | 任务末差pp | 单步按步差pp | oracle丢失率 | BM25丢失率 |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + f"""

全量原始轨迹token均值={stats['task_raw_tokens']['mean']:.3f}、P90={stats['task_raw_tokens']['p90']:.3f}（n=150）；去重轨迹均值={stats['task_unique_tokens']['mean']:.3f}、P90={stats['task_unique_tokens']['p90']:.3f}。全部 **{tests}/{tests}** 测试通过；{results['validation']['step_records']}个状态中非法动作和超预算均为0。no_compress任务末召回为1、丢失率为0。

完整的19配置指标、损失拆分、输入长度、缓存成本及PLAN原四行判据见[实验报告](eval/feasibility/feasibility_report.md)，所有分布与比例均有n。阅读模型相关格标为“未测（需阅读模型）”，本任务不作开工决策，不定义“接近”阈值。[机器结果](eval/feasibility/feasibility_results.json)、[运行配置](eval/feasibility/run_config.json)和[20条答案命中样例](eval/feasibility/answer_leak_samples.md)已同步镜像。

原始实验产物仅在5880的`data/feasibility/`：`tasks_dev150.jsonl`、`steps.jsonl`、`final_contexts.jsonl`和`corpus_dev_ntok.jsonl`。最终保留集合及四题的question/answer/answer_aliases均保存，以后可直接补阅读模型评测。任务01的验收目录、轨迹和3000占位预算保持原样。

任务02扩展了摘要覆盖范围，因此从本节起使用 `bash scripts/run_python.sh scripts/checksums_task02.py --verify`；任务01的原校验脚本未改动，旧清单保存在`specs/TASK01_SHA256SUMS`。原文档前缀与62个其他任务01文件也由新校验器逐项保护。确切命令见HANDOFF的任务02追加节。
"""
    notes = prefix + """以下补充只适用于任务02，出处为`tasks/TASK_02_feasibility_nomodel_20261117.md`（镜像副本在specs）。任务01记录保留原样。

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
"""
    handoff = prefix + f"""已完成任务02的单次CPU运行：预算{budgets['tight']}/{budgets['mid']}/{budgets['loose']}，自然命中{rate(natural['rate'])}，正文非证据段含答案比例{rate(leakage['paragraph']['rate'])}；中档差值：{'；'.join(differences)}。全部{tests}项测试通过（包含原80项），所有{results['validation']['step_records']}状态合法且不超预算，no_compress召回1、丢失率0。F1/EM未测。

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

实际运行时间：{run['started_at_utc']}至{run['finished_at_utc']}，脚本记录 **{run['elapsed_seconds']:.3f}秒**。逐阶段耗时：

```json
""" + json.dumps(run["timings_seconds"], ensure_ascii=False, indent=2) + f"""
```

进程数1，观测线程{run['cpu']['threads_observed']}（限制16）；峰值RSS {run['maximum_rss_bytes'] / 1024**3:.3f} GiB；data/feasibility产物 {run['data_bytes'] / 1e6:.3f} MB，低于200MB预期与1GB停止阈值。pid/退出码/日志见logs/feasibility.*。测试JUnit报告在eval/feasibility/tests.xml，没有改动eval/acceptance/unit_tests.xml。

审核重点：预算由dev每任务全部去重gold的P90定档；泄漏排除整任务gold；任务末召回区别于当前可用gold的单步召回；丢失拆分分母及重复gold关联不能直接相加；缓存按全局pid及保留顺序计算并逐任务重置。查看报告末尾oracle比较项，若有个体反例其task/config/pid均保留，不能据此擅改oracle。独立复核与主实验输出在新eval/feasibility下。

旧SHA清单在specs/TASK01_SHA256SUMS；任务01的62个其他文件保持原摘要，三份主文档保留原字节前缀。当前SHA256SUMS使用新checksums_task02.py生成并--verify核对，纳入data/feasibility及所有新代码、小结果文件。本机没有data/feasibility，须在5880核验。
"""
    for name, suffix in (("README.md", readme), ("NOTES.md", notes), ("HANDOFF.md", handoff)):
        path = ROOT / name
        if MARKER in path.read_text():
            raise SystemExit(f"Task02 documentation already appended: {name}")
    for name, suffix in (("README.md", readme), ("NOTES.md", notes), ("HANDOFF.md", handoff)):
        with (ROOT / name).open("a") as stream:
            stream.write(suffix)
    print("Appended TASK_02 documentation; all TASK_01 prefixes preserved")


if __name__ == "__main__":
    main()
