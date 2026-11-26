"""CPU fixtures for paired answers, account reconciliation and immutable reporting."""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import deepseek_api_full_report as rep


def row(strategy, mode, task, q, f1, em, **extra):
    return {"strategy":strategy,"mode":mode,"task":task,"question_index":q,"example_id":str((task,q)),
            "answer":"title","answer_aliases":[],"f1":f1,"em":em,"qwen_f1":f1,"qwen_em":em,**extra}


def test_oracle_shared_control_has_two_explicit_pairs_per_question():
    left=[row("oracle",mode,11,0,f1,em) for mode,f1,em in
          [("append",1.0,1),("reselect",0.0,0)]]
    control=[row("no_compress",None,11,0,0.5,0)]
    other=[row("oracle","append",12,0,0,0)]
    pairs=rep.pair_strategies(left+other+control,"oracle","no_compress",selected={11},shared=True)
    summaries=rep.pair_breakdown(pairs)
    assert summaries["combined"]["n_unique_questions"]==1
    both=summaries["combined"]["DeepSeek"]
    assert both["n"]==2 and both["em_difference_pp"]==50 and both["f1_difference_pp"]==0
    assert both["f1_counts"]=={"left_higher":1,"equal":0,"right_higher":1}
    assert both["em_matrix_correct_then_incorrect"]==[[0,1],[0,1]]
    assert summaries["append"]["Qwen3-8B"]["f1_difference_pp"]==50
    with pytest.raises((ValueError,RuntimeError)):
        rep.pair_strategies(left[:1]+control,"oracle","no_compress",shared=True)


def test_match_strategy_uses_mode_and_question_and_rejects_duplicate():
    left=[row("r","append",1,0,.5,0),row("r","reselect",1,0,1,1)]
    right=[row("v","reselect",1,0,0,0),row("v","append",1,0,.5,0)]
    pairs=rep.pair_strategies(left+right,"r","v")
    assert all(a["mode"] == b["mode"] for a,b in pairs)
    p=rep.paired(pairs,"DeepSeek","DeepSeek")
    assert p["f1_counts"] == {"left_higher":1,"equal":1,"right_higher":0}
    assert p["em_matrix_correct_then_incorrect"]==[[0,1],[0,1]]
    with pytest.raises((ValueError,RuntimeError)):
        rep.pair_strategies(left+right+[right[0]],"r","v")


def test_discordant_deepseek_qwen_and_empty_condition():
    q=row("r","append",1,0,.5,0,qwen_em=1,qwen_f1=1)
    d=rep.paired([(q,q)],*rep.READERS)
    assert d["f1_difference_pp"]==-50 and d["f1_counts"]["right_higher"]==1
    assert rep.paired([], *rep.READERS)["f1_difference_pp"] is None


def test_token_denominator_and_percentile():
    def call(hit,miss,latency):
        return {"usage":{"prompt_cache_hit_tokens":hit,"prompt_cache_miss_tokens":miss,
                         "completion_tokens":2},"latency_seconds":latency}
    v=rep.transport_group([call(10,0,1),call(0,90,3)])
    assert v["cache_hit_fraction"]==.1 and v["latency_seconds"]=={"p50":2,"p90":2.8}
    assert rep.D(v["paid_yuan"])==rep.D(".00010620")


def financial_fixture():
    ts="2026-11-24T05:10:00+00:00"
    usage={"prompt_cache_hit_tokens":10,"prompt_cache_miss_tokens":30,"completion_tokens":4,
           "prompt_tokens":40,"total_tokens":44}
    cost=".00004620"
    rec={"call_id":"one","id":"resp","kind":"A_compressed","requested_model":"deepseek-flash","model":"server",
         "period":"off_peak","latency_seconds":1.5,"started_utc":ts,"ended_utc":"2026-11-24T05:10:01.5+00:00",
         "attempts":1,"system_fingerprint":"fp","content":"ok","finish_reason":"stop","usage":usage,
         "cost_yuan":cost,"max_tokens":48}
    attempt={**rec,"kind":"chat","http_status":200}
    settles=[{"event":"settle","stage":"full","call_id":"one","hit":10,"miss":30,"output":4,"cost_yuan":cost},
             {"event":"settle","stage":"smoke","call_id":"one","cost_yuan":".005"}]
    ledger={"stage_totals":{"spent_yuan":cost},"spent_yuan":".00504620","reserved_yuan":"0"}
    balances={key:{"total_balance":v} for key,v in (("start","5"),("end","4.98"))}
    small={"balance_before":{"total_balance":"5.10"},"balance_after":{"total_balance":"5.09"},
           "smoke_spent_yuan":".005"}
    return [rec],[attempt],[],balances,settles,ledger,small


def test_ledger_accounts_for_shared_smoke_and_unrelated_account_spending():
    args=financial_fixture(); cfg=rep.cost_summary(*args)
    assert rep.D(cfg["full_account_drop_minus_ledger_yuan"])==rep.D(".0199538")
    assert rep.D(cfg["smoke_plus_full_paid_yuan"])==rep.D(".0050462")
    assert rep.D(cfg["overall_account_drop_minus_ledger_yuan"])==rep.D(".1149538")
    assert cfg["chat_http_attempts"]==1


def test_reject_peak_http_attempt_and_corrupt_settlement():
    args=financial_fixture()
    args[1][0]["started_utc"]="2026-11-24T03:10:00+00:00"
    with pytest.raises((ValueError,RuntimeError)):
        rep.cost_summary(*args)
    args=financial_fixture(); args[4][0]["miss"]=29
    with pytest.raises((ValueError,RuntimeError)):
        rep.cost_summary(*args)


def test_refuses_overwrite_but_accepts_identical_delivery(tmp_path):
    dest=tmp_path/"receipt.json"
    rep.publish(dest,{"count":2})
    initial=dest.read_bytes()
    rep.publish(dest,{"count":2})
    with pytest.raises(FileExistsError):
        rep.publish(dest,{"count":3})
    assert dest.read_bytes()==initial

