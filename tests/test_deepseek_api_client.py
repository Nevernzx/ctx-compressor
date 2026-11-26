from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import threading

import pytest

from scripts import deepseek_api_client as api


KEY = "private" + "Unit" + "ValueXYZ"
MESSAGES = [{"role": "user", "content": "What is the answer?"}]


class Clock:
    def __init__(self, start=None):
        self.value = start or datetime(2026, 11, 23, 14, tzinfo=timezone.utc)
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += timedelta(seconds=seconds)


def cash(value="9.40"):
    return {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": value}]}


def answer(hit=11, miss=4, out=1):
    return {"id": "unit-id", "model": "provider-version", "system_fingerprint": "fingerprint",
            "choices": [{"finish_reason": "stop", "message": {"content": "A"}}],
            "usage": {"prompt_cache_hit_tokens": hit, "prompt_cache_miss_tokens": miss,
                      "completion_tokens": out, "total_tokens": hit+miss+out, "prompt_tokens": hit+miss,
                      "completion_tokens_details": {"reasoning_tokens": 0}}}


def setup_client(root, replies=(), start=None):
    (root/"smoke").mkdir(exist_ok=True)
    queue = iter(replies)
    calls, statuses = [], []
    clock = Clock(start)
    def send(path, payload, secret):
        assert secret == KEY
        if payload is None:
            return 200, cash()
        calls.append((path, payload, clock.now()))
        response = next(queue)
        if isinstance(response, Exception):
            raise response
        return response
    client = api.DeepSeekClient(root, "smoke", statuses.append, key=KEY, transport=send,
                                clock=clock.now, sleep=clock.sleep)
    return client, calls, statuses, clock


@pytest.mark.parametrize("weekday,hour,minute,peak", [
    (0,8,59,False),(0,9,0,True),(0,11,59,True),(0,12,0,False),(0,13,59,False),
    (0,14,0,True),(0,17,59,True),(0,18,0,False),(5,10,0,False)])
def test_registered_beijing_peak_boundaries(weekday,hour,minute,peak):
    t = datetime(2026,11,23+weekday,hour,minute,tzinfo=api.BEIJING)
    end = api.peak_end(t.astimezone(timezone.utc))
    assert (end is not None) == peak
    if peak:
        assert end.hour == (12 if hour < 12 else 18) and end.minute == 0


def test_usage_cost_raw_receipt_and_single_stage_spend(tmp_path):
    c, calls, _, _ = setup_client(tmp_path,[(200,answer())])
    assert c.balance("start")["total_balance"] == "9.40"
    r = c.complete("reader/oracle:2",MESSAGES,48)
    assert r["usage"] == answer()["usage"] and r["requested_model"] == api.MODEL
    assert r["content"] == "A" and r["model"] == "provider-version" and r["attempts"] == 1
    assert calls[0][1] == {"model":"deepseek-flash","stream":False,"temperature":0,"max_tokens":48,
                           "thinking":{"type":"disabled"},"messages":MESSAGES}
    assert r["cost_yuan"] == "0.00000822" and c.summary()["spent_yuan"] == r["cost_yuan"]
    assert c.summary()["reserved_yuan"] == "0" and c.summary()["stage_totals"]["calls"] == 1
    other = api.Ledger(tmp_path/"ledger.jsonl").summary("full")
    assert other["calls"] == 1 and other["stage_totals"]["calls"] == 0
    c.balance("end")
    assert api.scan_artifacts([tmp_path],key=KEY)["matches"] == 0


def test_cost_cap_uses_durable_accounts_before_network(tmp_path,monkeypatch):
    c, calls, statuses, _ = setup_client(tmp_path,[(200,answer())])
    c.balance("start")
    maximum = 100*api.upper_input_tokens(MESSAGES)+400*48
    previous_charge = api.CAP - maximum
    with c.ledger.locked() as add:
        reserve={"stage":"full","attempt":1,"call_id":"historical"}
        add({**reserve,"event":"reserve","worst_units":previous_charge})
        add({**reserve,"event":"settle","cost_units":previous_charge,"hit":0,"miss":0,"output":0})
    c.complete("a",MESSAGES)
    with pytest.raises(api.APIStop, match="stopped_cost_cap"):
        c.complete("b",MESSAGES)
    assert len(calls) == 1 and statuses[-1]["status"] == "stopped_cost_cap"


def test_429_backoff_rechecks_beijing_time_before_retry(tmp_path):
    t=datetime(2026,11,23,0,59,59,tzinfo=timezone.utc)
    c,calls,statuses,clock=setup_client(tmp_path,[(429,None),(200,answer())],t)
    c.balance("start")
    r=c.complete("a",MESSAGES)
    assert r["attempts"]==2 and [s for s in clock.sleeps]==[2,10799]
    assert [x[2].astimezone(api.BEIJING).hour for x in calls]==[8,12]
    assert statuses[0]["status"]=="paused_peak" and c.summary()["attempts"]==2
    assert c.summary()["reserved_yuan"]=="0"


def test_no_retry_on_402_and_state_sealed(tmp_path):
    c,calls,_,clock=setup_client(tmp_path,[(402,None)])
    c.balance("start")
    with pytest.raises(api.APIStop, match="stopped_balance"):
        c.complete("a",MESSAGES)
    with pytest.raises(api.APIStop, match="stopped_balance"):
        c.complete("b",MESSAGES)
    assert len(calls)==1 and not clock.sleeps and c.summary()["spent_yuan"]=="0"


def test_timeout_and_server_error_each_hold_worst_case(tmp_path):
    c,calls,_,clock=setup_client(tmp_path,[(500,None),TimeoutError(),(200,answer())])
    c.balance("start")
    r=c.complete("a",MESSAGES)
    assert len(calls)==3 and r["attempts"]==3 and clock.sleeps==[2,4]
    assert c.summary()["reserved_yuan"]==api.yuan(2*(100*api.upper_input_tokens(MESSAGES)+400*48))
    assert c.summary()["spent_yuan"]==r["cost_yuan"]


def test_max_attempts_and_all_backoffs(tmp_path):
    c,calls,_,clock=setup_client(tmp_path,[(429,None)]*6)
    c.balance("start")
    with pytest.raises(api.APIStop, match="stopped_retries"):
        c.complete("a",MESSAGES)
    assert len(calls)==6 and clock.sleeps==list(api.BACKOFF)


def test_duplicate_call_does_not_issue_network_and_overwrite(tmp_path):
    c,calls,_,_=setup_client(tmp_path,[(200,answer())])
    c.balance("start")
    r=c.complete("a",MESSAGES)
    with pytest.raises(api.APIStop,match="stopped_duplicate_call"):
        c.complete("a",MESSAGES)
    with pytest.raises(FileExistsError):
        c.balance("start")
    assert len(calls)==1 and r["usage"]==answer()["usage"]


def test_two_clients_cannot_both_reserve_last_capacity(tmp_path,monkeypatch):
    c,*_=setup_client(tmp_path)
    d,*_=setup_client(tmp_path)
    monkeypatch.setattr(api,"CAP",100)
    barrier=threading.Barrier(2)
    def reserve(client):
        barrier.wait()
        try:
            client.reserve(str(id(client)),1,75)
            return True
        except api.APIStop as e:
            assert e.status=="stopped_cost_cap"
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(reserve,(c,d)))==[False,True]
    assert c.summary()["reserved_yuan"]==api.yuan(75)


def test_end_balance_low_is_recorded_and_seals_stage(tmp_path):
    c,*_=setup_client(tmp_path)
    c.balance("start")
    c.transport=lambda *args:(200,cash("0.99"))
    assert c.balance("end")["total_balance"]=="0.99"
    assert (tmp_path/"smoke/balance_end.json").is_file()
    with pytest.raises(api.APIStop):
        c.complete("a",MESSAGES)


@pytest.mark.parametrize("bad_value",[KEY, "s" + "k" + chr(45) + "1234testing"],ids=["exact","prefix"])
def test_output_secret_never_persisted(tmp_path,bad_value):
    reply=answer()
    reply["choices"][0]["message"]["content"]=bad_value
    c,*_=setup_client(tmp_path,[(200,reply)])
    c.balance("start")
    with pytest.raises(api.APIStop,match="stopped_secret_scan"):
        c.complete("a",MESSAGES)
    assert c.summary()["spent_yuan"]=="0.00000822"
    assert api.scan_artifacts([tmp_path],key=KEY)["matches"]==0


def test_read_key_requires_exact_permissions_and_refuses_symlink(tmp_path):
    file=tmp_path/"unit.env"
    test_bytes=b"DEEPSEEK_API_KEY=" + b"in-memory-only-for-unit"
    file.write_bytes(test_bytes)
    file.chmod(0o644)
    with pytest.raises(api.APIStop,match="permissions"):
        api.read_key(file)
    file.chmod(0o600)
    assert api.read_key(file).startswith("in-memory")
    link=tmp_path/"alias"
    link.symlink_to(file)
    with pytest.raises(OSError):
        api.read_key(link)


def test_artifact_scanner_sees_byte_split_and_secret_like_strings():
    # Detection uses only synthetic in-memory values.
    assert api.contains_secret(b"x " + b"s" + b"k" + bytes([45]) + b"demo",KEY)
    assert api.contains_secret(("masked "+KEY).encode(),KEY)
    assert not api.contains_secret(b"ordinary text",KEY)

def test_prep_has_only_free_balance_requests(tmp_path):
    (tmp_path/"prep").mkdir()
    calls=[]
    def send(path,payload,key):
        calls.append(path)
        return 200,cash()
    c=api.DeepSeekClient(tmp_path,"prep",transport=send,key=KEY,clock=Clock().now)
    assert c.balance("end")["currency"]=="CNY"
    with pytest.raises(api.APIStop,match="stopped_prep_chat_forbidden"):
        c.complete("anything",MESSAGES)
    assert calls==["/user/balance"]
