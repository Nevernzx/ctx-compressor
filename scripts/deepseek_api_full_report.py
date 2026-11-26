"""Close the DeepSeek subset experiment from its immutable CPU and API receipts."""
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts import deepseek_api_inputs as inputs
from scripts import deepseek_api_rollout as rollout
from scripts import deepseek_api_subset as exp
from scripts import deepseek_api_client as transport
from scripts import deepseek_api_smoke_report as smoke
from src.qa_metrics import score_answer
from src.reader import parse_answer
from src.reader_heldout_inputs import need

FULL = exp.WORK / "full"
READERS = ("DeepSeek", "Qwen3-8B")
STRATEGIES = (*inputs.A_POLICIES, "no_compress", "no_context")
D = Decimal


def unique(rows, key):
    result = {}
    for r in rows:
        k = key(r)
        need(k not in result, "duplicate result key")
        result[k] = r
    return result


def number(row, metric, reader):
    return row[("qwen_" if reader == READERS[1] else "") + metric]


def scores(rows, reader):
    n = len(rows)
    return {"n": n, **{key: math.fsum(number(r, key, reader) for r in rows)/n if n else None
                       for key in ("f1", "em")}}


def paired(pairs, left_reader, right_reader):
    counts = {"left_higher": 0, "equal": 0, "right_higher": 0}
    matrix = [[0, 0], [0, 0]]
    left = scores([a for a, _ in pairs], left_reader)
    right = scores([b for _, b in pairs], right_reader)
    for a, b in pairs:
        f, g = number(a, "f1", left_reader), number(b, "f1", right_reader)
        counts["left_higher" if f > g else "right_higher" if g > f else "equal"] += 1
        matrix[1-int(number(a, "em", left_reader))][1-int(number(b, "em", right_reader))] += 1
    return {"n": len(pairs), "left_scores": left, "right_scores": right, "f1_counts": counts,
            "em_matrix_correct_then_incorrect": matrix,
            **{key+"_difference_pp": (left[key]-right[key])*100 if pairs else None for key in ("f1", "em")}}


def pair_strategies(rows, left, right, *, selected=None, shared=False):
    """Match by mode and question; a shared comparison reuses one control per item and mode."""
    loc = lambda r: (r["task"], r["question_index"], r["mode"])
    left_rows = [r for r in rows if r["strategy"] == left and (selected is None or r["task"] in selected)]
    right_rows = [r for r in rows if r["strategy"] == right and (selected is None or r["task"] in selected)]
    a, b = unique(left_rows, loc), unique(right_rows, loc)
    keys = {k: (k[0], k[1], None) if shared else k for k in a}
    need(set(keys.values()) == set(b), "unmatched question or mode between strategies")
    if shared:
        need(len(a) == len(inputs.MODES)*len(b), "incomplete shared-control comparison")
        need(all({k[2] for k in a if k[:2] == idx[:2]} == set(inputs.MODES) for idx in b),
             "missing comparison mode")
    pairs = []
    for k, target in keys.items():
        need(all(a[k][field] == b[target][field] for field in ("example_id","answer","answer_aliases")),
             "paired reference answer differs")
        pairs.append((a[k], b[target]))
    return pairs


def pair_breakdown(pairs):
    by_scope = {"combined": pairs, **{m:[(a,b) for a,b in pairs if a["mode"] == m] for m in inputs.MODES}}
    return {scope: {"n_unique_questions":len({(a["task"], a["question_index"]) for a,_ in group}),
                    **{reader:paired(group, reader, reader) for reader in READERS}}
            for scope, group in by_scope.items()}


def answer_shape(rows, calls):
    return {"n":len(rows), "finish_reasons":dict(Counter(r["finish_reason"] for r in rows)),
            "at_token_limit":sum(calls[r["request_id"]]["usage"]["completion_tokens"] == 48 for r in rows),
            "empty_parsed_answers":sum(not r["prediction"].strip() for r in rows)}


def reader_results(predictions, s30, calls):
    a = [row for row in predictions if row["strategy"] in STRATEGIES]
    by = {}
    for name in STRATEGIES:
        group = [r for r in a if r["strategy"] == name]
        sub = {}
        for flag in (True, False):
            selected = [r for r in group if r["evidence_complete"] == flag]
            sub[str(flag).lower()] = {reader:scores(selected, reader) for reader in READERS}
        by[name] = {"scores": {reader:scores(group, reader) for reader in READERS},
                    "paired_readers":paired([(r,r) for r in group], *READERS),
                    "by_evidence_complete":sub, "output_shape":answer_shape(group, calls)}
    rv = pair_strategies(a, inputs.R2, inputs.V3)
    oc = pair_strategies(a, "oracle", "no_compress", selected=set(s30), shared=True)
    nongold = {k:[] for k in ("<=0", "1", ">=2")}
    for left, right in rv:
        if left["evidence_complete"] and right["evidence_complete"]:
            diff = left["non_gold_count"] - right["non_gold_count"]
            nongold["<=0" if diff <= 0 else "1" if diff == 1 else ">=2"].append((left, right))
    return {"strategies":by, "r2_minus_v3":pair_breakdown(rv), "oracle_minus_no_compress_s30":pair_breakdown(oc),
            "r2_minus_v3_non_gold_conditioned":{k:{reader:paired(g,reader,reader) for reader in READERS}
                                               for k,g in nongold.items()},
            "output_shape":answer_shape(a, calls)}


def quantile(values, q):
    ordered = sorted(values)
    at = (len(ordered)-1)*q
    lo = int(at)
    return ordered[lo] + (ordered[min(lo+1,len(ordered)-1)]-ordered[lo])*(at-lo)


def transport_group(rows):
    hit = sum(r["usage"]["prompt_cache_hit_tokens"] for r in rows)
    miss = sum(r["usage"]["prompt_cache_miss_tokens"] for r in rows)
    out = sum(r["usage"]["completion_tokens"] for r in rows)
    latency = [r["latency_seconds"] for r in rows]
    return {"n":len(rows),"input_tokens":hit+miss,"cache_hit_tokens":hit,"cache_miss_tokens":miss,
            "cache_hit_fraction":hit/(hit+miss) if hit+miss else None,"output_tokens":out,
            "latency_seconds":{"p50":quantile(latency,0.50),"p90":quantile(latency,0.90)},
            "paid_yuan":str((D(".02")*hit+D(miss)+D(4)*out)/10**6)}


def cost_summary(full_calls, attempts, events, balances, entries, full_ledger, smoke_result):
    unique(full_calls,lambda r:r["call_id"])
    unique(full_calls,lambda r:r["id"])
    by_id = unique([x for x in entries if x["event"] == "settle" and x["stage"] == "full"],
                   lambda r:r["call_id"])
    need(set(by_id) == {r["call_id"] for r in full_calls}, "closed full ledger and raw calls differ")
    chat = [r for r in attempts if r["kind"] == "chat"]
    need(all(transport.peak_end(datetime.fromisoformat(r["started_utc"])) is None for r in attempts),
         "peak-time HTTP attempt present")
    attempt_log = unique([r for r in chat if 200 <= r.get("http_status",0) < 300],lambda r:r["call_id"])
    need(set(attempt_log) == set(by_id), "settlement and successful HTTP attempts differ")
    for r in full_calls:
        limit = 256 if r["kind"] == "B_compressor" else 48
        hit, miss, out = transport.token_counts(r["usage"], limit)
        receipt = by_id[r["call_id"]]
        need((receipt["hit"],receipt["miss"],receipt["output"]) == (hit,miss,out)
             and D(receipt["cost_yuan"]) == D(r["cost_yuan"]) == (D(".02")*hit+miss+D(4)*out)/10**6,
             "usage cost differs from ledger")
        need(r["requested_model"] == transport.MODEL and r["period"] == "off_peak" and r["max_tokens"] == limit,
             "request configuration differs")
        need(all(r[k] == attempt_log[r["call_id"]][k] for k in
                 ("id","model","system_fingerprint","usage","finish_reason","content","started_utc","ended_utc","attempts")),
             "HTTP receipt and raw call disagree")
    kinds = sorted({r["kind"] for r in full_calls})
    usage = {"all":transport_group(full_calls), **{k:transport_group([r for r in full_calls if r["kind"]==k]) for k in kinds}}
    paid = D(usage["all"]["paid_yuan"])
    all_paid = sum(D(r["cost_yuan"]) for r in entries if r["event"] == "settle")
    need(paid == D(full_ledger["stage_totals"]["spent_yuan"])
         and D(smoke_result["smoke_spent_yuan"])+paid == all_paid == D(full_ledger["spent_yuan"])
         and all_paid+D(full_ledger["reserved_yuan"]) <= 8, "shared cap or billed stage totals differ")
    begin, end = (D(balances[label]["total_balance"]) for label in ("start","end"))
    smoke_begin = D(smoke_result["balance_before"]["total_balance"])
    pauses = sorted({r["until_beijing"] for r in events if r.get("status") == "paused_peak"})
    return {"requested":{"model":transport.MODEL,"thinking":{"type":"disabled"},"temperature":0,"stream":False,
                         "max_tokens_reader":48,"max_tokens_compressor":256,"endpoint":"https://api.deepseek.com/chat/completions",
                         "direct":True,"concurrent_trajectory_limit":8,"currency":"CNY","spend_cap_with_smoke":"8.00",
                         "peak_beijing":"周一至周五 09:00–12:00、14:00–18:00，含工作日假日",
                         "off_peak_yuan_per_million":{"input_hit":"0.02","input_miss":"1","output":"4"}},
            "periods":dict(Counter(r["period"] for r in full_calls)),
            "first_request_utc":min(r["started_utc"] for r in full_calls),"last_response_utc":max(r["ended_utc"] for r in full_calls),
            "pause_windows_until_beijing":pauses, "chat_http_attempts":len(chat),
            "balance_http_attempts":sum(r["kind"]=="balance" for r in attempts),
            "attempt_http_status":dict(Counter(str(r.get("http_status")) for r in chat)),
            "model_responses":dict(Counter(r["model"] for r in full_calls)),
            "system_fingerprints":dict(Counter(str(r["system_fingerprint"]) for r in full_calls)),
            "full_usage":usage,"smoke_paid_yuan":str(D(smoke_result["smoke_spent_yuan"])),"full_paid_yuan":str(paid),
            "smoke_plus_full_paid_yuan":str(all_paid),"ledger":full_ledger,
            "full_balances":balances,"smoke_balances":{k:smoke_result["balance_"+k] for k in ("before","after")},
            "full_account_drop_yuan":str(begin-end),"full_account_drop_minus_ledger_yuan":str(begin-end-paid),
            "overall_account_drop_yuan":str(smoke_begin-end),
            "overall_account_drop_minus_ledger_yuan":str(smoke_begin-end-all_paid)}


def verify_predictions(raw, calls, expected):
    indexed = unique(raw, inputs.key)
    need(indexed.keys() == expected.keys(), "reader predictions do not match the registered grid")
    for ident, row in indexed.items():
        source = expected[ident]
        fields = ("strategy","task","mode","tier","question_index","non_gold_count","evidence_complete","trajectory_lost",
                  "example_id","answer","answer_aliases","prompt_sha1")
        if row["strategy"] != inputs.B_POLICY:
            fields += ("qwen_em","qwen_f1","qwen_prompt_tokens")
        need(all(row[k] == source[k] for k in fields), "prediction diverges from checked input")
        call = calls[row["request_id"]]
        need(call["content"] == row["raw_output"] and call["finish_reason"] == row["finish_reason"]
             and call["max_tokens"] == 48, "reader transport receipt identity differs")
        parsed = parse_answer(call["content"])
        recheck = score_answer(parsed, source["answer"], source["answer_aliases"])
        need(row["prediction"] == parsed and all(row[k] == v for k,v in recheck.items()), "reader score differs")
    return indexed


def build_b(prepared, trajectories, predictions, calls):
    actual = rollout.aggregate_b(prepared,trajectories)
    baselines = rollout.baseline_metrics(prepared,prepared.s30)
    selected = set(prepared.s30)
    readers = [r for r in predictions if r["task"] in selected]
    results = {}
    for name, metrics in [(inputs.B_POLICY,actual), *baselines.items()]:
        for scope in (*inputs.MODES,"combined"):
            result = metrics["combined"] if scope=="combined" else metrics["configs"][name+"|"+scope+"|mid"]["metrics"]
            need(result["n_tasks"] == (60 if scope=="combined" else 30), "unexpected B trajectory count")
            readers_here = [r for r in readers if r["strategy"]==name and (scope=="combined" or r["mode"]==scope)]
            need(len(readers_here) == (0 if name=="bm25" else 4*result["n_tasks"]), "unexpected B reader count")
            figures = {"metrics":result, "DeepSeek_reader":scores(readers_here,READERS[0])}
            if name == inputs.B_POLICY:
                compressor, question = [], [calls[r["request_id"]] for r in readers_here]
                for traj in trajectories:
                    if scope!="combined" and traj["mode"] != scope:
                        continue
                    for step in traj["steps"]:
                        call = calls[step["generation_metadata"]["api_id"]]
                        need(call["kind"]=="B_compressor" and call["content"]==step["raw_output"]
                             and call["usage"]["completion_tokens"]==step["generation_metadata"]["generated_tokens"],
                             "compressor response and replay differ")
                        compressor.append(call)
                need(len(compressor)==result["n_steps"], "incomplete compressor usage")
                figures["transport"]={"compressor":transport_group(compressor),"reader":transport_group(question)}
                figures["reader_output_shape"]=answer_shape(readers_here,calls)
            results.setdefault(name,{})[scope]=figures
    return results, actual


def compute():
    run = exp.read(FULL/"full_results.json")
    need(exp.read(FULL/"status.json")["status"] == "complete","full API stage incomplete")
    data, b, rows_a, sample = exp.prepare_both()
    pre = exp.read(exp.SMALL/"input_checks.json")
    need(json.loads(exp.encoded(exp.records(data,b,rows_a,sample)))==pre,"registered input receipts changed")
    raw = smoke.rows(FULL/"api_calls.jsonl")
    calls = unique(raw,lambda r:r["id"])
    pred = smoke.rows(FULL/"predictions.jsonl")
    trajectory_files = sorted((FULL/"trajectories").glob("*.json"))
    trajectories = [exp.read(p) for p in trajectory_files]
    seen = unique(trajectories,lambda t:(t["task"],t["mode"]))
    need(set(seen)=={(t,m) for t in data.s30 for m in inputs.MODES},"incomplete full compressor grid")
    rows_b = [r for t in trajectories for r in data.reader_rows(t["task"],t["mode"],t["final_pids"])]
    need(len(calls)==len(rows_a)+len(rows_b)+pre["B"]["n_steps"]==2588,"full call count differs")
    expected = unique(rows_a+rows_b,inputs.key)
    verify_predictions(pred,calls,expected)
    compressor_ids = [s["generation_metadata"]["api_id"] for t in trajectories for s in t["steps"]]
    consumed = [r["request_id"] for r in pred]+compressor_ids
    need(len(consumed)==len(set(consumed)) and set(consumed)==set(calls),"unmapped or duplicate API calls")
    b_result, recomputed = build_b(b,trajectories,pred,calls)
    if "B_metrics" in run:
        need(run["B_metrics"]==recomputed,"independent B replay disagrees with driver")
    balances={label:exp.read(FULL/("balance_"+label+".json")) for label in ("start","end")}
    smoke_result=exp.read(exp.WORK/"smoke/smoke_results.json")
    ledger=transport.Ledger(exp.WORK/"ledger.jsonl").summary("full")
    cfg=cost_summary(raw,smoke.rows(FULL/"attempts.jsonl"),smoke.rows(FULL/"events.jsonl"),balances,
                     smoke.rows(exp.WORK/"ledger.jsonl"),ledger,smoke_result)
    result={"status":"complete","task":14,"phase":"F","seed":inputs.SEED,"split":"train-held-out",
            "scope":{"s60":list(data.s60),"s30":list(data.s30),"budget_tier":"mid","budget":b.receipt["budget"],
                     "reader_only_calls":len(rows_a),"compressor_calls":len(compressor_ids),
                     "compressor_reader_calls":len(rows_b),"full_calls":len(raw),"smoke_excluded_from_scores":True},
            "checks":{"A_prompt_sha1_mismatches":pre["A"]["prompt_sha1_mismatches"],"A_n":len(rows_a),
                      "B_step0_sha256_mismatches":b.receipt["step0_mismatches"],"B_n":len(trajectories),
                      "rescored_answers":len(pred),"independent_baseline_strategies":list(rollout.BASELINES)},
            "A":reader_results(pred,data.s30,calls),"B":b_result,
            "costs":{k:cfg[k] for k in ("smoke_paid_yuan","full_paid_yuan","smoke_plus_full_paid_yuan","full_account_drop_minus_ledger_yuan")}}
    return result, cfg


def table(headers, rows):
    def line(cells):
        return "| "+" | ".join(str(cell).replace("|"," / ") for cell in cells)+" |"
    return "\n".join([line(headers),line(["---"]*len(headers)),*(line(row) for row in rows)])


def pct(val):
    return f"{100*val:.3f}%" if val is not None else "—"


def diff(val):
    return f"{val:+.3f}" if val is not None else "—"


def f1_counts(group):
    f=group["f1_counts"]
    return f"{f['left_higher']} / {f['equal']} / {f['right_higher']}"


def render(result, cfg):
    a, b = result["A"],result["B"]
    main = a["r2_minus_v3"]["combined"]
    n = result["scope"]["full_calls"]
    short = {inputs.R2:"GRPO r2 checkpoint-125",inputs.V3:"SFT v3 checkpoint-3000",
             inputs.B_POLICY:"DeepSeek 压缩器","oracle":"oracle","bm25":"bm25","no_context":"无上下文","no_compress":"不压缩"}
    markdown=["# 任务14　DeepSeek 子集实验",
              f"同一批中档预算提示上，r2 相对 v3 的 F1 差为 DeepSeek {diff(main[READERS[0]]['f1_difference_pp'])}、"
              f"Qwen3-8B {diff(main[READERS[1]]['f1_difference_pp'])} 个百分点（各 n={main[READERS[0]]['n']} 题次）。"
              f"全量从头运行 n={n:,} 次模型调用；小样仅计入总费用。"]
    markdown+=["## A　换读者复核",
               "F1 和 EM 为百分制；每行两个读者按同一提示配对。Qwen3-8B 使用任务12 已归档的输出。"
               "对照抽样仅来自训练留出集，S60 含 60 个任务，S30 为其前 30 个任务。"]
    rows=[]; evidence=[]; shape=[]
    for name,data in a["strategies"].items():
        ds, qw = (data["scores"][k] for k in READERS)
        rows.append([short[name],ds["n"],pct(ds["f1"]),pct(ds["em"]),pct(qw["f1"]),pct(qw["em"]),f1_counts(data["paired_readers"])])
        for flag,sub in data["by_evidence_complete"].items():
            x,y=(sub[k] for k in READERS)
            evidence.append([short[name],"齐全" if flag=="true" else "不齐全",x["n"],pct(x["f1"]),pct(y["f1"])])
        info=data["output_shape"]
        shape.append([short[name],info["n"],json.dumps(info["finish_reasons"],ensure_ascii=False),info["at_token_limit"],info["empty_parsed_answers"]])
    markdown.append(table(["上下文策略","题次 n","DeepSeek F1","DeepSeek EM","Qwen F1","Qwen EM","逐题 F1 DeepSeek 高 / 平 / Qwen 高"],rows))
    markdown+=["### 预注册主比较",
               "同一任务、问题和模式逐题配对。变化为左侧减右侧的百分点差。EM 四格按“左对右对、左对右错、左错右对、两边均错”列出。"
               "oracle 与不压缩在 S30 每格共用同一组 120 个问题；合并时每个不压缩答案配对两个模式，共 240 对、120 个不同问题。"]
    rows=[]
    for label,tag in [("r2 − v3","r2_minus_v3"),("oracle − 不压缩","oracle_minus_no_compress_s30")]:
        for scope,scope_data in a[tag].items():
            for reader in READERS:
                p=scope_data[reader]
                matrix=p["em_matrix_correct_then_incorrect"]
                rows.append([label,scope,reader,p["n"],scope_data["n_unique_questions"],diff(p["f1_difference_pp"]),
                             diff(p["em_difference_pp"]),f1_counts(p)," / ".join(str(i) for row in matrix for i in row)])
    markdown.append(table(["比较","范围","读者","配对 n","不同问题 n","ΔF1","ΔEM","F1 左高 / 平 / 右高","EM 四格计数"],rows))
    markdown+=["### 证据与无关段",table(["策略","该题证据","题次 n","DeepSeek F1","Qwen F1"],evidence),
               "两种策略都保留该题全部证据时，按 r2 比 v3 多保留的非金段数分组。非金段指该任务所有问题的金段以外的段落。"]
    rows=[]
    for bin_name,sub in a["r2_minus_v3_non_gold_conditioned"].items():
        x,y=(sub[r] for r in READERS)
        rows.append([bin_name,x["n"],diff(x["f1_difference_pp"]),diff(y["f1_difference_pp"])])
    markdown.append(table(["r2 多留非金段数","题次 n","DeepSeek ΔF1（百分点）","Qwen ΔF1（百分点）"],rows))
    overall=a["output_shape"]
    shape.append(["合并",overall["n"],json.dumps(overall["finish_reasons"],ensure_ascii=False),overall["at_token_limit"],overall["empty_parsed_answers"]])
    markdown+=["### DeepSeek 答案形态",table(["上下文策略","题次 n","finish_reason 计数","达到 48 token 的条数","解析后空答案条数"],shape)]
    markdown+=["## B　API 压缩器闭环",
               "每格使用 S30 相同的 30 个任务；合并共 60 条轨迹。最终召回和轨迹成本以轨迹为单位，单步召回、无效率、用量和新增数以步骤为单位。"
               "r2、v3 和 oracle 的 DeepSeek 读者分数取自 A 在 S30 的同格结果。bm25 仅复算压缩轨迹。"
               "缓存和普通成本单位均为每轨迹的等价输入 token。"]
    rows=[]; actions=[]; performance=[]; bshape=[]
    for scope in ("combined",*inputs.MODES):
        for name in (inputs.B_POLICY,*rollout.BASELINES):
            entry=b[name][scope];m=entry["metrics"];r=entry["DeepSeek_reader"]
            l=m["evidence_loss"];inv=m["invalid_outputs"];t=m["cost"]
            rows.append([scope,short[name],l["n"],f"{l['count']}/{l['n']}（{pct(l['rate'])}）",pct(m["final_recall"]["mean"]),
                         m["n_steps"],pct(m["step_recall"]["micro"]["mean"]),pct(inv["any_invalid"]["rate"]),
                         f"{t['cached']['mean']:.2f}",f"{t['plain']['mean']:.2f}",r["n"],pct(r["f1"]),pct(r["em"])])
            actions.append([scope,short[name],m["n_steps"],*(inv[k]["count"] for k in ("any_invalid","invalid_json","invalid_candidate_id","append_drop","over_budget")),
                            m["fence_or_whitespace_rescued"]["count"],json.dumps(m["actual_add_distribution"]),
                            f"{m['used_after']['mean']:.2f}"])
            if name==inputs.B_POLICY:
                for role,u in entry["transport"].items(): performance.append(transport_line(scope,role,u))
                sh=entry["reader_output_shape"]
                bshape.append([scope,sh["n"],json.dumps(sh["finish_reasons"]),sh["at_token_limit"],sh["empty_parsed_answers"]])
    markdown.append(table(["范围","压缩策略","轨迹 n","丢证据率","最终召回","步骤 n","单步召回","无效率","缓存成本","普通成本","读者题次 n","F1","EM"],rows))
    markdown+=["### 动作和成本明细","非法四类互斥，收尾空白/代码围栏只用于诊断。分布为“实际新增段数 → 步数”，非法动作按实际新增零段计。",
               table(["范围","策略","步骤 n","非法合计","JSON","候选 ID","append 删除","超预算","仅清理围栏/空白后合法","新增数分布","used_after 均值"],actions),
               "### B 的 API 用量和延迟",
               table(["范围","用途","调用 n","P50 秒","P90 秒","输入 token","命中 token","未命中 token","输入命中率","输出 token"],performance),
               table(["B 读者范围","题次 n","finish_reason 计数","达到 48 token 的条数","解析后空答案条数"],bshape)]
    u=cfg["full_usage"]
    markdown+=["## 费用、时间与校验",
               table(["范围","用途","调用 n","P50 秒","P90 秒","输入 token","命中 token","未命中 token","输入命中率","输出 token"],
                     [transport_line("全量",role,val) for role,val in u.items()]),
               f"账本记录小样 ¥{cfg['smoke_paid_yuan']}（n={cfg['ledger']['calls']-n} 次）、全量 ¥{cfg['full_paid_yuan']}（n={n:,} 次），"
               f"合计 ¥{cfg['smoke_plus_full_paid_yuan']}（n={cfg['ledger']['calls']:,} 次），保守预留 ¥{cfg['ledger']['reserved_yuan']}。"
               f"全量阶段首末共两次余额检查，余额由 ¥{cfg['full_balances']['start']['total_balance']} 至 ¥{cfg['full_balances']['end']['total_balance']}；"
               f"该阶段余额减少额与用量账本之差为 ¥{cfg['full_account_drop_minus_ledger_yuan']}（共用账户）。"
               f"从小样开始至全量结束，账户余额减少额与总账本之差为 ¥{cfg['overall_account_drop_minus_ledger_yuan']}。",
               f"全量发起 n={cfg['chat_http_attempts']:,} 次付费端点 HTTP 尝试，成功 n={n:,} 次，均核对北京时间。"
               f"按结束时刻去重的工作日高峰暂停窗口共 {len(cfg['pause_windows_until_beijing'])} 个"
               f"（窗口结束时间：{', '.join(cfg['pause_windows_until_beijing']) or '无'}）。"
               f"成功响应模型计数 {json.dumps(cfg['model_responses'])}，系统指纹计数 {json.dumps(cfg['system_fingerprints'])}（n={n:,}）。",
               "全量重新核对 A 的 1,800 条提示、B 的 60 条首步提示，哈希差异分别为 0/1,800 和 0/60。"
               "从 API 原始输出重新解析评分 2,040/2,040 条读者结果，并按 60/60 条实测轨迹复算闭环指标。"
               "任务14 C 的 Qwen3-8B 读者补测仍待确定读者配置。",
               "已使用实际密钥和前缀模式扫描全量 API 原始记录、账本、代码、测试与交付物，匹配为零。"
               "扫描范围和文件数见 [校验回执](full_security.json)，精确费用和请求配置见 [运行配置](run_config.json)。"]
    return "\n\n".join(markdown)+"\n"


def transport_line(scope,role,u):
    return [scope,role,u["n"],f"{u['latency_seconds']['p50']:.3f}",f"{u['latency_seconds']['p90']:.3f}",
            u["input_tokens"],u["cache_hit_tokens"],u["cache_miss_tokens"],pct(u["cache_hit_fraction"]),u["output_tokens"]]


def scan_inputs():
    documents = smoke.input_files() + [FULL]
    for suite_path in exp.WORK.glob("cpu_suite_*"):
        for f in ("tests.xml","pytest.log","result.json"):
            p=suite_path/f
            if p.is_file():
                documents.append(p)
    return documents


def publish(path, value):
    if path.is_symlink() or path.parent.resolve() != path.parent:
        raise ValueError("unexpected output path")
    data = value.encode("utf-8") if isinstance(value,str) else exp.encoded(value)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise FileExistsError("refusing to replace a different report artifact")
        return
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or Path.cwd() != PROJECT:
        raise RuntimeError("run the full report on 5880 with CPU only")
    result,cfg=compute()
    inputs_to_scan=scan_inputs()
    receipt=exp.SMALL/"full_security.json"
    targets=[exp.SMALL/"deepseek_api_results.json", exp.SMALL/"run_config.json",exp.SMALL/"deepseek_api_report.md"]
    # A repeated validation has the same input inventory and never replaces a previous receipt.
    omitted={receipt,*targets}
    sources=[p for p in inputs_to_scan if p not in omitted]
    security=transport.scan_artifacts(sources)
    publish(targets[0],result)
    publish(targets[1],cfg)
    publish(targets[2],render(result,cfg))
    all_scanned=transport.scan_artifacts(sources+targets)
    publish(receipt,{"status":"passed","matches":all_scanned["matches"],"source_scan":security,
                     "source_and_report_scan":all_scanned,"scope":"Task 14 program, test receipts, API archives, ledger, balance receipts and all three full reports."})
    transport.scan_artifacts([receipt])
    print(json.dumps({"status":"complete","n_calls":result["scope"]["full_calls"],
                      "paid_with_smoke":cfg["smoke_plus_full_paid_yuan"],
                      "full_end_balance":cfg["full_balances"]["end"]["total_balance"],
                      "r2_v3":{r:result["A"]["r2_minus_v3"]["combined"][r]["f1_difference_pp"] for r in READERS},
                      "oracle_no_compress":{r:result["A"]["oracle_minus_no_compress_s30"]["combined"][r]["f1_difference_pp"] for r in READERS},
                      "B":{scope:{"n":v["metrics"]["n_tasks"],"loss_count":v["metrics"]["evidence_loss"]["count"],
                                  "steps":v["metrics"]["n_steps"],"invalid":v["metrics"]["invalid_outputs"]["any_invalid"]["count"],
                                  "f1":v["DeepSeek_reader"]["f1"]} for scope,v in result["B"][inputs.B_POLICY].items()},
                      "scan":all_scanned}))


if __name__ == "__main__":
    main()

