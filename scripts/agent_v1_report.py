"""Verify archived Task 20 episodes and write the preregistered report and plots."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from io import BytesIO
from math import isclose
import os
from pathlib import Path

from scripts import agent_v1_main as main, agent_v1_smoke as smoke
from src import agent_v1 as agent, agent_v1_metrics as metrics
from src.reader_heldout_inputs import need, read, rows, write_new
from src.reranker_v1 import record, require_new, verify
from scripts import test_v1_seal as sums

S = smoke.SMALL
SCOPES = ("musique", "hotpotqa", "2wiki", "pooled")
DISPLAY = {"musique": "MuSiQue", "hotpotqa": "HotpotQA", "2wiki": "2Wiki", "pooled": "合并"}
REPORT = S / "agent_v1_report.md"
RESULT = S / "agent_v1_results.json"
TASK19 = smoke.PROJECT / "eval/test_v1/test_v1_results.json"
SVG = {scope: S / ("cost_f1_" + scope + ".svg") for scope in SCOPES[:3]}
CST = timezone(timedelta(hours=8))


def checked(item):
    p = Path(item["path"])
    need(p.is_absolute() and p.resolve() == p and p.is_file() and p.stat().st_size == item["size"],
         f"Invalid artifact path or size: {p}")
    verify(p, item["sha256"])
    return p


def close_inputs():
    status = read(main.STATUS)
    need(status["status"] == "evaluation_complete" and status["stage"] == "finished" and not status["lock"]["held"]
         and status["stage_pid"] is None and not status["automatic_retry"], "Main controller is not terminal")
    need(not Path(f"/proc/{status['pipeline_pid']}").exists(), "Main controller still present")
    stage, = status["stages"]
    need(stage["status"] == "complete" and stage["exit_code"] == 0 and stage["uses_gpu"]
         and stage["run_config"] == record(main.CONFIG), "Incomplete main stage")
    cfg = read(main.CONFIG)
    need(cfg["status"] == "evaluation_complete" and cfg["gpu_index"] == 1 and cfg["rows"] == list(agent.ROWS),
         "Completed config has the wrong GPU or rows")
    for key in ("ipc_check", "preflight", "data_preflight", "prompt", "try3", "gpu_ownership"):
        checked(cfg[key])
    need(cfg["frozen_code"] == smoke.frozen_code() and cfg["main_code"] == main.current_source(),
         "Code changed during the run")
    frozen = read(main.FROZEN)
    need(cfg["assets"] == main.effective_assets(frozen["assets"]) and cfg["prompt"] == frozen["prompt_file"],
         "Completed settings differ from approved configuration")
    need(cfg["scope"] == "formal" and cfg["n_tasks"] == 300
         and cfg["agent_worker_extension"] == main.residency.EXTENSION, "Wrong formal scheduler scope")
    pilot = cfg["pilot"]
    pilot_config = read(checked(pilot["config"]))
    pilot_state = read(checked(pilot["status"]))
    for name in ("cache", "episodes"):
        checked(pilot[name])
    need(pilot_config["scope"] == "pilot" and pilot_config["runtime"]["n_episodes"] == 16
         and pilot_state["status"] == "evaluation_complete" and pilot_config["assets"] == cfg["assets"],
         "Two-task pilot proof differs from formal deployment")
    repair = cfg["rpc_repair"]
    original = read(checked(repair["original_preflight"]))
    checked(repair["previous_launch"])
    need(repair["source_sha256"] == {
        key: {"old": original["main_code"][key]["sha256"], "new": value["sha256"]}
        for key, value in cfg["main_code"].items()
    }, "Main RPC repair source digests differ")
    runtime = cfg["runtime"]
    for key in ("episodes", "gpu_snapshots", "root_usage_check", "weight_transfers", "residency_check"):
        checked(runtime[key])
    need(runtime == read(checked(stage["receipt"])), "GPU receipt changed")
    need(any(engine["engine"] == "agent" and engine["effective_disable_cascade_attn"] is True
             for engine in runtime["engine_loads"]), "Missing effective reader configuration")
    check = read(Path(runtime["residency_check"]["path"]))
    need(check["status"] == "passed" and check["next_cached_tokens"] > 0, "Native cache handoff unproven")
    storage = read(Path(runtime["root_usage_check"]["path"]))
    for name in ("root_telemetry", "workplace_telemetry"):
        checked(storage[name])
    need(storage["status"] == "passed" and storage["sustained_growth_bytes"] <= 300_000_000
         and storage["peak_attributed_growth_bytes"] <= 300_000_000 and storage["min_root_free_bytes"] >= 5_000_000_000,
         "Main storage receipt differs from approved policy")
    return status, cfg, check, storage


def gold_distribution(tally):
    buckets = dict.fromkeys(("zero", "one_incomplete", "multiple_incomplete", "all"), 0)
    by_total = defaultdict(dict)
    for (total, kept), n in sorted(tally.items()):
        need(0 <= kept <= total and total > 0, "Invalid question gold count")
        bucket = ("all" if kept == total else "zero" if kept == 0 else
                  "one_incomplete" if kept == 1 else "multiple_incomplete")
        buckets[bucket] += n
        by_total[str(total)][str(kept)] = n
    return {"n": sum(buckets.values()), "counts": buckets, "by_total_gold_and_kept": dict(by_total)}


def rescore(cfg):
    data = read(main.data.RECEIPT)["datasets"]
    registered = {(name, x["task"]): x for name, info in data.items()
                  for x in rows(checked(info["main"]["file"]))}
    need(len(registered) == 300, "Expected 300 frozen formal tasks")
    expected = {(ds, row, task) for ds, task in registered for row in agent.ROWS}
    seen, data_rows = set(), []
    cache, gold = defaultdict(Counter), defaultdict(Counter)
    n_turns = 0
    for ep in rows(checked(cfg["runtime"]["episodes"])):
        ds, row, task = (ep[k] for k in ("dataset", "row", "task"))
        identity = ds, row, task
        need(identity in expected and identity not in seen, f"Duplicate or unregistered episode: {identity}")
        seen.add(identity)
        original = registered[ds, task]
        need(ep["questions"] == original["questions"] and ep["budget"] == original["budget"]
             and len(ep["answers"]) == agent.K and ep["turns"], f"Episode differs from registered task: {identity}")
        retained, retrieved, queries = [], set(), []
        last_count = 0
        for i, turn in enumerate(ep["turns"]):
            raw = turn["raw_output"]
            reply = agent.parse_reply(raw)
            invalid = reply.malformed_json or (reply.kind != "finish" if turn["forced"] else reply.kind == "invalid")
            need(raw.startswith(agent.THOUGHT_PREFILL) and reply.kind == turn["action"]
                 and reply.query == turn["query"] and invalid == turn["format_invalid"]
                 and reply.malformed_json == turn["malformed_finish_json"] and turn["round"] == i,
                 f"Archived format mismatch: {identity}")
            need(turn["keep_before"] == retained, f"Retained history does not join: {identity}")
            terminal = i == len(ep["turns"]) - 1
            need(terminal == bool(turn["forced"] or reply.kind == "finish"), f"Episode finish mismatch: {identity}")
            new = turn.get("retrieved", [])
            if reply.kind == "search" and not turn["forced"]:
                need("retrieved" in turn and len(new) <= agent.TOP_K and len(new) == len(set(new)) and
                     all(type(pid) is int and 0 <= pid < data[ds]["n_paragraphs"] for pid in new),
                     f"Invalid search results: {identity}")
                queries.append(reply.query)
            else:
                need("retrieved" not in turn and turn["keep_after"] == retained,
                     f"Non-search turn changed the context: {identity}")
            retrieved.update(new)
            retained = turn["keep_after"]
            need(len(retained) == len(set(retained)) and set(retained) <= retrieved,
                 f"Retained passage did not appear in search results: {identity}")
            if terminal:
                need(list(reply.answers) == ep["answers"] and turn["forced"] == ep["force_reason"],
                     f"Final answers or forced-finish reason differ: {identity}")
            tokens, prefix = turn["prompt_tokens"], turn["prefix_hit_tokens"]
            need(0 <= prefix <= min(last_count, tokens) and 0 <= turn["output_tokens"] <= agent.MAX_NEW
                 and isclose(turn["cached_input_token_cost"], (prefix + 25 * (tokens - prefix)) / 20),
                 f"Archived cache cost mismatch: {identity}")
            if row.endswith(" append") and not turn["forced"]:
                need(prefix == last_count, f"Append prefix changed: {identity}")
            if i:
                observed = turn["vllm_cached_tokens"]
                for scope in (ds, "pooled"):
                    counts = cache[scope, row]
                    counts["adjacent_rounds"] += 1
                    counts["native_cached_rounds"] += bool(observed)
                    counts["native_cached_input_tokens"] += observed or 0
                    counts["adjacent_common_tokens"] += prefix
            last_count = tokens
            n_turns += 1
        need(ep["searched_pids"] == sorted(retrieved) and ep["final_keep"] == retained
             and ep["queries"] == queries, f"Final context/search summary differs: {identity}")
        final = set(retained)
        for q in ep["questions"]:
            targets = set(q["gold"])
            bucket = len(targets), len(targets & final)
            gold[ds, row][bucket] += 1
            gold["pooled", row][bucket] += 1
        data_rows.append(ep)
    need(seen == expected and len(seen) == cfg["runtime"]["n_episodes"] == 2400, "Incomplete formal archive")
    result = metrics.analyze(data_rows, 100)
    cache_summary = {scope: {row: dict(cache[scope, row]) for row in agent.ROWS} for scope in SCOPES}
    diagnostics = {scope: {row: gold_distribution(gold[scope, row]) for row in agent.ROWS} for scope in SCOPES}
    for scope, methods in diagnostics.items():
        for row, stats in methods.items():
            official = result[scope]["rows"][row]["evidence"]["evidence_complete"]
            need(stats["n"] == official["n"] and stats["counts"]["all"] == official["count"],
                 f"Post hoc gold counts differ from complete-evidence metric: {scope}/{row}")
    return result, cache_summary, n_turns, diagnostics


def comparison_against_simulation(ag):
    index = sums.load_sums(sums.MANIFEST)
    verify(TASK19, index[str(TASK19.relative_to(smoke.PROJECT))])
    source = read(TASK19)
    aliases = {
        "no_compress": ("no_compress", None), "latest": ("latest", "reselect"), "oracle": ("oracle", "reselect"),
        "RR-τ reselect": ("RR-τ", "reselect"), "RR-τ append": ("RR-τ", "append"),
        "r4 reselect": ("r4", "reselect"), "r4 append": ("r4", "append"),
    }
    out = {}
    for row, (registered, mode) in aliases.items():
        cell = source["reader"][registered]
        simulated = cell["cells"][f"{mode}|mid"] if mode else cell["by_tier"]["mid"]
        out[row] = {"task19": {"f1": simulated["f1"], "n": simulated["n"], "cell": (mode or "shared") + "|mid"},
                    "task20": {"f1": ag[row]["f1"], "n": ag[row]["n_questions"]}}
    return out


def percent(x):
    return "—" if x is None else f"{100*x:.3f}%"


def ratio(x):
    return f"{percent(x['rate'])} ({x['count']:,}/{x['n']:,})"


def avg(d, digits=1):
    return f"{d['mean']:,.{digits}f} (n={d['n']:,})"


def signed(x):
    return f"{100*x:+.3f}"


def tab(headers, values):
    def line(row):
        return "| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(value) for value in values)])


def qa_and_cost(results):
    values = []
    for scope in SCOPES:
        for row in agent.ROWS:
            d = results[scope]["rows"][row]
            cost = d["cost"]
            values.append((DISPLAY[scope], row, d["n_questions"], percent(d["em"]), percent(d["f1"]),
                           avg(cost["uncached"]), avg(cost["cached"]), avg(cost["output"])))
    return tab(("数据集", "策略", "题 n", "EM", "F1", "无缓存输入 token", "开缓存折合输入 token", "输出 token"), values)


def pairs(results):
    values = []
    for scope in SCOPES:
        for key, pair in results[scope]["paired"].items():
            m = pair["em_2x2_left_rows_right_columns"]
            values.append((DISPLAY[scope], key, pair["n"], signed(pair["delta_f1"]), signed(pair["delta_em"]),
                           f"{pair['f1_higher']} / {pair['f1_equal']} / {pair['f1_lower']}",
                           f"{m[1][1]} / {m[1][0]} / {m[0][1]} / {m[0][0]}"))
    return tab(("数据集", "配对", "题 n", "ΔF1（百分点）", "ΔEM（百分点）", "F1 高 / 平 / 低",
                "EM 双对 / 仅左对 / 仅右对 / 双错"), values)


def process_table(results):
    values = []
    for scope in SCOPES:
        for row in agent.ROWS:
            p = results[scope]["rows"][row]["process"]
            searches = p["searches"]
            hist = " / ".join(f"{k}次 {v}" for k, v in searches["histogram"].items())
            values.append((DISPLAY[scope], row, avg(searches, 2), hist, ratio(p["self_finished"]),
                           ratio(p["forced_finish"]), ratio(p["invalid_format"]), ratio(p["overlength"]),
                           ratio(p["duplicate_queries"])))
    return tab(("数据集", "策略", "平均搜索次数", "搜索次数分布（次→任务数）", "自己 finish", "强制 finish",
                "格式无效轮", "超长强制 finish", "重复查询"), values)


def evidence_table(results):
    values, hops, substring = [], [], []
    for scope in SCOPES:
        for row in agent.ROWS:
            e = results[scope]["rows"][row]["evidence"]
            values.append((DISPLAY[scope], row, ratio(e["retrieved"]), ratio(e["retrieval_complete"]), ratio(e["retained"]),
                           ratio(e["evidence_complete"]), ratio(e["lost_given_retrieved"]), ratio(e["questions_lost_any_retrieved"])))
            if scope == "musique":
                hops.extend((row, hop, ratio(rate)) for hop, rate in e.get("musique_by_hop_index", {}).items())
            if "compressed_context_contains_verbatim_canonical_answer" in e:
                substring.append((DISPLAY[scope], ratio(e["compressed_context_contains_verbatim_canonical_answer"])))
    return (tab(("数据集", "策略", "搜到的 gold", "搜齐的题", "保留的 gold", "证据齐全的题",
                 "已搜到 gold 中的丢失", "丢过已检索 gold 的题"), values),
            tab(("MuSiQue 策略", "hop_index", "gold 保留"), hops),
            tab(("LLMLingua-2 数据集", "保留上下文含标准答案字符串的题"), substring))


def compression_table(results):
    retained, diagnostics = [], []
    for scope in SCOPES:
        for row in agent.ROWS:
            c = results[scope]["rows"][row]["compression"]
            retained.append((DISPLAY[scope], row, avg(c["retained_passages"], 2), avg(c["nongold_passages"], 2)))
            if row.startswith(("RR-τ", "LLMLingua-2", "r4")):
                latency = c["seconds_per_call"]
                dt = ("— (n=0)", "— (n=0)") if latency is None else tuple(
                    f"{1000*latency[key]:.2f} (n={latency['n']:,})" for key in ("p50", "p90"))
                diagnostics.append((DISPLAY[scope], row, c["calls_with_new"],
                                    ratio(c["invalid"]) if "invalid" in c else "—",
                                    ratio(c["over_budget"]) if "over_budget" in c else "—", *dt))
    return (tab(("数据集", "策略", "任务末平均保留段数", "任务末平均无关段数"), retained),
            tab(("数据集", "策略", "压缩调用 n", "非法", "超预算", "调用 p50（ms）", "调用 p90（ms）"), diagnostics))


def native_cache_table(diagnostics):
    values = []
    for row in agent.ROWS:
        counts = diagnostics["pooled"][row]
        n = counts["adjacent_rounds"]
        matched = counts["native_cached_rounds"]
        values.append((row, ratio(metrics.ratio(matched, n)), counts["native_cached_input_tokens"],
                       counts["adjacent_common_tokens"]))
    return tab(("策略", "相邻后轮产生物理命中", "物理缓存 token 合计", "同 episode 公共前缀 token 合计"), values)


def posthoc_gold_table(stats):
    values = []
    for scope in SCOPES:
        for row in agent.ROWS:
            cell = stats[scope][row]
            values.append((DISPLAY[scope], row, cell["n"],
                           *(ratio(metrics.ratio(cell["counts"][bucket], cell["n"]))
                             for bucket in ("zero", "one_incomplete", "multiple_incomplete", "all"))))
    return tab(("数据集", "策略", "题 n", "保留 0 段", "恰留 1 段且未齐", "保留 ≥2 段且未齐", "全部保留"), values)


def render(bundle):
    r = bundle["results"]
    sections = [
        "\n## 正式运行",
        f"同一模型 Qwen3-8B 执行 300 个冻结多题任务，在 8 行上归档 n={bundle['episodes']:,} 个 episode、"
        f"n={bundle['rounds']:,} 轮。MuSiQue 正式题来自锁定测试集。"
        f"正式运行监控窗口 {bundle['wall_seconds']/60:.2f} 分钟（n=1 次运行）。"
        "冻结的 Thought: 预填和解码参数沿用 try3。唯一的引擎资源调整是 §3 允许的 Agent 显存比例 0.50。"
        "r4 使用任务18 F 的原配置。Agent 权重在 r4 阶段暂移至主存，KV 块留在 GPU 1，随后恢复权重。"
        "正式题启动前，独立 pilot 用 n=2 个旧冒烟任务覆盖全部 n=8 个策略，归档 n=16 个 episode。"
        "本机 vLLM worker 扩展调用具名方法。新旧源文件 SHA256 已写进运行配置。"
        f"合成检查下一轮实际命中 {bundle['cache_preflight']['next_cached_tokens']} 个 token（n=1 相邻轮对）。"
        f"GPU 1 实测峰值 {bundle['peak_observed_gpu1_mib']:,} MiB（n={bundle['gpu_samples']:,} 监控和快照）。"
        "[冻结配置](run_config.json) · [结构化结果](agent_v1_results.json)",
        "### 数据",
        tab(("语料", "段数", "每任务 gold token P50", "P90", "题型（各 n/300）", "yes/no", "预算 token"),
            ((DISPLAY[name], d["n_paragraphs"], f"{d['gold_tokens']['p50']:.1f} (n={d['gold_tokens']['n']})",
              f"{d['gold_tokens']['p90']:.1f} (n={d['gold_tokens']['n']})",
              " / ".join(f"{key} {value}/300" for key, value in d["main"]["question_type"].items()) or "—",
              ratio(d["main"]["yes_no"]), f"{d['budget']} (n={d['main']['n_tasks']} 任务)")
             for name, d in bundle["datasets"].items())),
        "### 答案与成本",
        "每个成本均值的 n 为 episode 数。开缓存成本按相邻 prompt token 的最长公共前缀计算，命中倍率 0.05，其余倍率 1.25。",
        qa_and_cost(r),
        " ".join(f"[{DISPLAY[name]} 成本–F1]({SVG[name].name})" for name in SCOPES[:3]),
        "### 逐题配对",
        pairs(r),
        "### ReAct 过程",
        process_table(r),
        "### 检索覆盖与保留",
        *evidence_table(r),
        "### 压缩诊断",
        *compression_table(r),
        "r4 每次请求的墙钟时间采用其所在 vLLM 批次从提交至返回的时间。"
        "任务20 try3 冒烟的 r4 只追加非法 7/143 次，重选 0/134 次。"
        "open-provence 为第三方复现，现有环境缺少 NLTK punkt 与 punkt_tab，本轮未跑。"
        "cl100k_base.tiktoken 是 LLMLingua-2 的本地依赖，已按记录核验。[模型卡](model_cards.json)",
        "### MuSiQue 模拟与真实对照",
        "任务19 使用锁定测试集中档，用同策略及相同模式。任务20 使用本任务抽样的真实 Agent 题。两列分别汇总。",
        tab(("策略", "任务19 中档格", "任务19 F1", "题次 n", "任务20 F1", "题 n"),
            ((row, d["task19"]["cell"], percent(d["task19"]["f1"]), d["task19"]["n"],
              percent(d["task20"]["f1"]), d["task20"]["n"])
             for row, d in bundle["musique_simulation"].items())),
        "### 物理前缀缓存",
        "对每个 episode 的相邻后轮读取 vLLM 的实际缓存计数。开缓存成本仍取同 episode token 最长公共前缀。",
        native_cache_table(bundle["physical_prefix_diagnostics"]),
        "### 根分区和验收",
    ]
    storage = bundle["storage"]
    sections.append(
        f"根分区全局首尾差值 {storage['sustained_growth_bytes']:,} 字节，可归因增长峰值 "
        f"{storage['peak_attributed_growth_bytes']:,} 字节，最少可用 {storage['min_root_free_bytes']:,} 字节"
        f"（n={storage['samples']:,} 次根分区采样）。"
        f"规则 (d) 记录 n={len(storage['transient_unattributed_peaks'])} 个瞬时全局高点。"
    )
    if storage["transient_unattributed_peaks"]:
        sections.append("\n".join(
            f"- 北京时间 {datetime.fromisoformat(row['utc']).astimezone(CST).strftime('%Y-%m-%d %H:%M:%S')}，"
            f"引擎 {row['engine'] or '切换间隔'}，增长 {row['global_growth_bytes']:,} 字节（n=1 个峰值）。"
            for row in storage["transient_unattributed_peaks"]))
    if "cpu_archive_recovery" in bundle:
        recovery = bundle["cpu_archive_recovery"]
        a = storage["violation"]
        facts = recovery["attribution_evidence"]
        trace = recovery["trace_audit"]
        failed_at = datetime.fromisoformat(a["utc"]).astimezone(CST).strftime("%Y-%m-%d %H:%M:%S")
        sections.extend((
            f"北京时间 {failed_at}，完成全部 n={bundle['episodes']:,} 个 episode 后，"
            f"全局首尾差值触发规则 (a)，GPU 控制器以 exit 1 退出（n=1 次），GPU 1 和共享锁已释放。"
            "原状态、根分区回执和失败日志按原字节保留。"
            "经授权，依据逐轮归档做纯 CPU 收尾。核对了任务组合、逐轮动作和答案、"
            "检索记录与最终保留列表。故障后未再次推理。[CPU 恢复回执](main_try2/cpu_archive_recovery.json)",
            f"Claude 的只读核查显示，北京时间 22:00 后根分区新增 n={facts['new_root_files']} 个文件，"
            f"合计约 {facts['new_root_files_approx_kb']} KB。其检查时持有已删除文件的进程为 "
            f"n={facts['processes_holding_deleted_files']}，可归因增长为 4 KiB，"
            f"根分区可用 {facts['root_available_gb']:.2f} GB（n=1 次核查），高于 5 GB 阈值。"
            f"正式监控的 n={trace['fd_samples']:,} 次有效文件描述符采样均未找到本任务在根分区的可写文件，"
            f"另有 n={trace['samples']-trace['fd_samples']} 次快速扫描失败。",
            "归因假设：根分区 df 反映宿主共享盘的计数，规则 (a) 使用全局首尾差会计入外部写入。"
            "所给观察无法确认具体写入方。",
        ))
    posthoc = bundle["posthoc_gold_retention"]
    hotpot = posthoc["hotpotqa"]
    sections.extend((
        "### 事后诊断（非预注册）",
        "每题将去重后的 gold PID 与任务末保留段落取交集。表中各档互斥。"
        "含三段或四段 gold 的题可能留两段以上但未齐，另列该档。JSON 同时记录按 gold 总段数和保留段数分层的精确计数。",
        posthoc_gold_table(posthoc),
        "HotpotQA 每题有两段 gold（每行 n=300）。r4 重选恰留一段的题为 "
        f"{hotpot['r4 reselect']['counts']['one_incomplete']}/{hotpot['r4 reselect']['n']}，"
        "r4 只追加为 "
        f"{hotpot['r4 append']['counts']['one_incomplete']}/{hotpot['r4 append']['n']}。",
    ))
    return "\n\n".join(sections) + "\n"


def plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = ("o", "s", "D", "^", "v", "P", "X", "<")
    palette = ("#677580", "#8299a5", "#7955aa", "#b96c32", "#ab5c24", "#248785", "#1d5ba1", "#143b67")
    for scope, p in SVG.items():
        fig, ax = plt.subplots(figsize=(8.0, 4.7), layout="constrained")
        for row, marker, color in zip(agent.ROWS, markers, palette, strict=True):
            stat = results[scope]["rows"][row]
            x, y = stat["cost"]["cached"]["mean"], stat["f1"] * 100
            ax.scatter(x, y, s=45, color=color, marker=marker, label=row)
        ax.margins(x=.2, y=.17)
        ax.set_title(f"{DISPLAY[scope]} | 300 questions and 100 tasks per strategy", loc="left")
        ax.set_xlabel("Mean cached input cost per task (weighted tokens)")
        ax.set_ylabel("Answer F1 (%)")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, .5), frameon=False, fontsize=8)
        ax.grid(color="#e0e5eb", linewidth=.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        with BytesIO() as out:
            fig.savefig(out, format="svg", metadata={"Date": None})
            yield p, out.getvalue()
        plt.close(fig)


def assemble(state, cfg, check, storage, scored):
    result, physical, turns, gold = scored
    n_gpu = sum(1 for _ in rows(main.ROOT / "resource_monitor.jsonl"))
    n_gpu += sum(1 for _ in rows(main.ROOT / "gpu_snapshots.jsonl"))
    return {
        "status": "complete", "task": 20, "episodes": 2400, "rounds": turns,
        "posthoc_gold_retention": gold,
        "wall_seconds": cfg["runtime"]["wall_seconds"], "peak_observed_gpu1_mib": cfg["peak_observed_gpu1_mib"],
        "gpu_samples": n_gpu, "cache_preflight": check, "physical_prefix_diagnostics": physical, "storage": storage,
        "datasets": read(main.data.RECEIPT)["datasets"], "results": result,
        "musique_simulation": comparison_against_simulation(result["musique"]["rows"]),
        "terminal_status": record(main.STATUS),
        "model_cards": record(S / "model_cards.json"), "ended_utc": state["ended_utc"],
    }


def deliver(bundle):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Report is CPU only")
    require_new(RESULT, *SVG.values())
    report_text = render(bundle)
    charts = list(plots(bundle["results"]))
    for p, svg in charts:
        with p.open("xb") as out:
            out.write(svg)
    with REPORT.open("a", encoding="utf-8") as out:
        out.write(report_text)
    bundle["run_config"] = record(main.CONFIG)
    bundle["delivery"] = {p.name: record(p) for p in (REPORT, *SVG.values())}
    write_new(RESULT, bundle)
    print("Task 20 report, scores, and three SVG figures verified", flush=True)


def main_report():
    state, cfg, check, storage = close_inputs()
    deliver(assemble(state, cfg, check, storage, rescore(cfg)))


if __name__ == "__main__":
    main_report()

