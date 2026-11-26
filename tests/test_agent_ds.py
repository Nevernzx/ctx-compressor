"""CPU contracts for Task 21 (DeepSeek main model). Run with CUDA hidden:
CUDA_VISIBLE_DEVICES= .conda/envs/ctxc/bin/python -B -m pytest -p no:cacheprovider --basetemp <RT>/pytest tests/test_agent_ds.py
"""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
from unittest.mock import patch

import pytest

from scripts import agent_ds_client as api, agent_ds_report as report, agent_ds_run as run
from scripts import agent_v1_data as data, agent_v1_engines as engines, agent_v1_smoke as smoke
from scripts.deepseek_api_client import APIStop
from src import agent_v1 as agent
from src.model_inference import compact
from src.policies import latest_policy
from src.reader_heldout_inputs import EXTERNAL, read, rows

KEY = "testkey" + "0123456789abcdef"
BEIJING = timezone(timedelta(hours=8))
PROJECT = run.PROJECT


@pytest.fixture(scope="module")
def tokenizer():
    assets = engines.registered_assets(verify_models=False)
    return engines.offline_tokenizer(assets["agent"]["params"]["tokenizer"])


@pytest.fixture(scope="module")
def smoke_data():
    tasks, corpora, index_dirs = run.selection("smoke")
    return tasks, corpora


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


# ---------- prompt text equals what Task 20 tokenised ----------

def check_prompt(tok, ep, tokens, msgs):
    head = decode(tok, ep.prompt)
    assert msgs[0] == {"role": "system", "content": agent.SYSTEM}
    assert msgs[1] == {"role": "user", "content": agent.user_text(ep.question_text)}
    assert agent.SYSTEM in head and agent.user_text(ep.question_text) in head
    api.check_messages(msgs)
    if ep.force_reason is None:
        assert len(msgs) == 3
        assert decode(tok, tokens) == head + msgs[2]["content"]
    else:
        assert len(msgs) == 5 and msgs[3]["content"] == agent.FORCE_MESSAGE
        assert decode(tok, tokens) == head + msgs[2]["content"] + decode(tok, ep.force_tail) + msgs[4]["content"]


@pytest.mark.parametrize("row", ["no_compress", "oracle", "r4 append"])
def test_messages_reproduce_task20_prompt_text(tokenizer, smoke_data, row):
    tasks, corpora = smoke_data
    task = tasks["musique"][0]
    ep = agent.Episode(task, row, corpora["musique"], tokenizer)
    gold = sorted(ep.gold)
    noise = [pid for pid in range(40) if pid not in ep.gold]
    replies = [" find the first entity.\nAction: search[" + task["questions"][0]["question"] + "]",
               " look again.\nAction: search[second query]",
               "no action here", " still wrong",
               " answer now.\nAction: finish[{\"1\":\"a\",\"2\":\"b\",\"3\":\"c\"}]"]
    results = [gold[:2] + noise[:10], noise[10:20] + gold[2:4]]
    for reply in replies:
        tokens, msgs, cid = run.prepare(ep, "test")
        check_prompt(tokenizer, ep, tokens, msgs)
        assert re.fullmatch(r"[A-Za-z0-9_:.|/-]{1,240}", cid)
        call = {"content": reply, "output": 10, "call_id": cid, "id": "x", "model": api.MODEL, "system_fingerprint": None,
                "finish_reason": "stop", "hit": 0, "miss": 1, "cost_yuan": "0", "latency_seconds": 0.1, "attempts": 1,
                "messages_sha256": "0"}
        request = run.accept(ep, tokens, call, lambda q: results.pop(0))
        if request:
            if row == "r4 append":
                agent.execute_compression(request, lambda item: agent.apply_compression(item, compact(latest_policy()(item.state))))
            else:
                agent.execute_compression(request, agent.baseline_compression)
        if ep.completed():
            break
    assert ep.completed() and ep.force_reason == "invalid_format" and ep.answers == ("a", "b", "c")
    assert all("deepseek" in t for t in ep.turns)
    # The history-size variant used when the forced prompt would overflow.
    k = len(ep.blocks) - 1
    expected = decode(tokenizer, ep.prompt) + run.transcript(ep, k) + decode(tokenizer, ep.force_tail) + agent.THOUGHT_PREFILL
    assert decode(tokenizer, ep.prompt_tokens(history_size=k, force=True)) == expected
    if row == "oracle":  # reselect dropped non-gold passages, so the rendered history must have changed
        assert "[P3]" not in run.transcript(ep)


def test_message_shape_validation():
    good = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "Thought:", "prefix": True}]
    api.check_messages(good)
    forced = good[:2] + [{"role": "assistant", "content": "x"}, {"role": "user", "content": agent.FORCE_MESSAGE},
                         {"role": "assistant", "content": "Thought:", "prefix": True}]
    api.check_messages(forced)
    for bad in (good[:2], good[:2] + [{"role": "assistant", "content": "Thought:"}],
                good[:2] + [{"role": "assistant", "content": "x", "prefix": True}],
                [good[1], good[0], good[2]]):
        with pytest.raises(ValueError):
            api.check_messages(bad)


# ---------- client: price, caps, peak gate, secrets ----------

class Clock:
    def __init__(self, when):
        self.t = when
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += timedelta(seconds=seconds)


def usage(hit=60, miss=40, out=10):
    return {"prompt_tokens": hit + miss, "prompt_cache_hit_tokens": hit, "prompt_cache_miss_tokens": miss,
            "completion_tokens": out, "total_tokens": hit + miss + out}


def fake_transport(log, content=" ok\nAction: search[q]", status=200, use=None, clock=None):
    def send(path, body, key):
        assert key == KEY
        log.append((clock() if clock else None, path, body))
        if path == api.BALANCE_PATH:
            return 200, {"is_available": True, "balance_infos": [
                {"currency": "CNY", "total_balance": "24.94", "granted_balance": "0", "topped_up_balance": "24.94"}]}
        if status != 200:
            return status, None
        return 200, {"id": "c1", "model": api.MODEL, "system_fingerprint": "fp",
                     "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                     "usage": use or usage()}
    return send


def msgs(text="Thought:"):
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": text, "prefix": True}]


def client_for(tmp, stage, log, *, when=datetime(2026, 11, 26, 20, 0, tzinfo=BEIJING), **kw):
    work = tmp.resolve()
    (work / stage).mkdir(parents=True, exist_ok=True)
    clock = Clock(when)
    c = api.AgentClient(work, stage, key=KEY, transport=fake_transport(log, clock=clock, **kw), clock=clock, sleep=clock.sleep)
    c.balance("start")
    return c, clock


def test_request_shape_and_price(tmp_path):
    log = []
    c, _ = client_for(tmp_path, "smoke", log)
    row = c.complete("smoke|musique|t0|nc|r0", msgs())
    (_, path, body), = [x for x in log if x[1] != api.BALANCE_PATH]
    assert path == "/beta/chat/completions" and body["stop"] == ["\nObservation", "\nThought:"] and body["max_tokens"] == 256
    assert body["thinking"] == {"type": "disabled"} and body["temperature"] == 0 and body["model"] == "deepseek-flash"
    assert body["messages"][-1]["prefix"] is True
    assert row["cost_units"] == 2 * 60 + 100 * 40 + 400 * 10 and row["content"] == " ok\nAction: search[q]"
    assert c.summary()["stage_totals"]["spent_yuan"] == "0.0000812"


def test_peak_hours_pause_before_any_chat(tmp_path):
    log = []
    c, clock = client_for(tmp_path, "smoke", log, when=datetime(2026, 11, 26, 17, 30, tzinfo=BEIJING))
    c.complete("smoke|x|t0|nc|r0", msgs())
    chats = [t for t, p, _ in log if p == api.CHAT_PATH]
    assert len(chats) == 1 and chats[0].astimezone(BEIJING).hour == 18 and c.peak_pauses >= 1
    assert all((t is None or api.base.peak_end(t) is None) for t, p, _ in log)


def test_peak_allowed_runs_now_at_double_price(tmp_path):
    log = []
    work = tmp_path.resolve()
    (work / "smoke").mkdir(parents=True)
    clock = Clock(datetime(2026, 11, 26, 15, 0, tzinfo=BEIJING))
    c = api.AgentClient(work, "smoke", key=KEY, transport=fake_transport(log, clock=clock), clock=clock,
                        sleep=clock.sleep, allow_peak=True)
    c.balance("start")
    row = c.complete("smoke|x|t0|nc|r0", msgs())
    assert clock.slept == [] and row["period"] == "peak"
    assert row["cost_units"] == 2 * (2 * 60 + 100 * 40 + 400 * 10)
    clock.t = datetime(2026, 11, 26, 19, 0, tzinfo=BEIJING)
    row = c.complete("smoke|x|t0|nc|r1", msgs())
    assert row["period"] == "off_peak" and row["cost_units"] == 2 * 60 + 100 * 40 + 400 * 10


def test_stage_and_total_caps(tmp_path):
    log = []
    c, _ = client_for(tmp_path, "smoke", log)
    huge = "Thought:".rjust(300_000)  # worst case reserved at peak price: ~300k x 2e-6 CNY = 0.6 > 0.5 smoke cap
    with pytest.raises(APIStop, match="stopped_cost_cap"):
        c.complete("smoke|x|t0|nc|r0", msgs(huge))
    assert not [p for _, p, _ in log if p == api.CHAT_PATH]
    # total cap across stages: pretend main already spent 19.99 CNY in the shared ledger
    log2 = []
    m, _ = client_for(tmp_path, "main", log2)
    with m.ledger.locked() as add:
        add({"event": "reserve", "stage": "main", "call_id": "seed", "attempt": 1, "utc": "x", "worst_units": 1_999_000_000, "worst_yuan": "19.99"})
        add({"event": "settle", "stage": "main", "call_id": "seed", "attempt": 1, "utc": "x", "cost_units": 1_999_000_000,
             "hit": 0, "miss": 0, "output": 0})
    with pytest.raises(APIStop, match="stopped_cost_cap"):
        m.complete("main|x|t0|nc|r0", msgs("Thought:".rjust(10_000)))


def test_http_402_and_secret_and_scope(tmp_path):
    with pytest.raises(APIStop, match="stopped_balance"):
        client_for(tmp_path / "a", "smoke", [], status=402)[0].complete("smoke|x|t0|nc|r0", msgs())
    with pytest.raises(APIStop, match="stopped_secret_scan"):
        client_for(tmp_path / "b", "smoke", [], content="leak " + KEY)[0].complete("smoke|x|t0|nc|r0", msgs())
    with pytest.raises(APIStop, match="stopped_transport_scope"):
        api.transport("/chat/completions", {"x": 1}, KEY)
    with pytest.raises(ValueError):
        api.AgentClient(tmp_path, "full", key=KEY)


def test_content_risk_refusal_does_not_halt(tmp_path):
    log, state = [], {"n": 0}
    work = tmp_path.resolve()
    (work / "smoke2").mkdir(parents=True)
    clock = Clock(datetime(2026, 11, 26, 20, 0, tzinfo=BEIJING))
    ok = fake_transport(log, clock=clock)
    def send(path, body, key):
        if path == api.CHAT_PATH and state["n"] == 0:
            state["n"] += 1
            return 400, {"content_risk": True, "message": "Content Exists Risk (request_id: x)"}
        return ok(path, body, key)
    c = api.AgentClient(work, "smoke2", key=KEY, transport=send, clock=clock, sleep=clock.sleep)
    c.balance("start")
    with pytest.raises(api.ContentRefused):
        c.complete("smoke2|x|t0|nc|r0", msgs())
    assert "smoke2" not in c.ledger.halts and c.summary()["stage_totals"]["spent_yuan"] == "0"
    assert c.complete("smoke2|x|t0|nc|r1", msgs())["content"]


# ---------- paths and a fake end-to-end run ----------

def test_caps():
    assert api.TOTAL_CAP == 20 * api.CNY and api.STAGE_CAP == {"probe": api.CNY // 10, "smoke": api.CNY // 2,
                                                                "smoke2": api.CNY, "main": 20 * api.CNY}


def test_paths_are_new_task21_locations(tmp_path):
    assert run.WORK == EXTERNAL / "agent_ds" == api.WORK
    assert run.SMALL == PROJECT / "eval/agent_ds" and run.RT == EXTERNAL / "rt/ads"
    for old in (data.WORK, data.SMALL, EXTERNAL / "deepseek_api", EXTERNAL / "dagger_v3", EXTERNAL / "test_v1"):
        assert old not in (run.WORK, run.SMALL) and old not in run.WORK.parents and old not in run.SMALL.parents
    import secrets
    rt = EXTERNAL / "rt/ads" / ("u" + secrets.token_hex(2))  # short: the vLLM IPC path must stay < 100 bytes
    assert not rt.exists()
    with patch.object(run, "RT", rt):
        env = run.environment("smoke")
    assert all(rt in Path(env[k]).resolve().parents for k in smoke.CACHE_KEYS)
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert not any(k.lower() in ("http_proxy", "https_proxy", "all_proxy") for k in env)


def all_pids(state):
    pids = state["pid_by_lid"]
    return list(pids.values()) if isinstance(pids, dict) else list(pids)


def snapshot(roots):
    seen = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if "agent_ds" in Path(dirpath).parts:
                dirnames[:] = []
                continue
            for name in filenames:
                p = Path(dirpath) / name
                st = p.lstat()
                seen[str(p)] = (st.st_size, st.st_mtime_ns)
    return seen


def scripted(log, clock, refuse_when=None):
    def send(path, body, key):
        log.append((clock(), path, body))
        if path == api.BALANCE_PATH:
            return 200, {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "24.94"}]}
        if refuse_when and refuse_when(body):
            return 400, {"content_risk": True, "message": "Content Exists Risk (request_id: y)"}
        history = body["messages"][2]["content"]
        q = body["messages"][1]["content"].splitlines()[0].split(". ", 1)[1]
        text = (" done.\nAction: finish[{\"1\":\"x\",\"2\":\"y\",\"3\":\"z\"}]" if "Observation:" in history
                else f" search it.\nAction: search[{q}]")
        return 200, {"id": "c", "model": api.MODEL, "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                     "usage": usage(10, 90, 12)}
    return send


def test_fake_end_to_end_writes_only_new_dirs(tmp_path, tokenizer, smoke_data):
    import bm25s
    tasks, corpora = smoke_data
    _, _, index_dirs = run.selection("smoke")
    indexes = {k: bm25s.BM25.load(str(v)) for k, v in index_dirs.items()}
    searchers = {k: (lambda i: (lambda q: agent.search_bm25(i, q)))(v) for k, v in indexes.items()}
    roots = [PROJECT / "eval", data.WORK, EXTERNAL / "deepseek_api", EXTERNAL / "dagger_v3"]
    before = snapshot(roots)
    work = tmp_path.resolve() / "agent_ds"
    (work / "smoke").mkdir(parents=True)
    log, clock = [], Clock(datetime(2026, 11, 28, 10, 0, tzinfo=BEIJING))  # Saturday: off-peak
    # refuse the second turn of the no_compress episode of the hotpotqa task: its task must leave the comparisons
    hq = tasks["hotpotqa"][0]["questions"][0]["question"]
    refuse_when = lambda body: (hq in body["messages"][1]["content"] and "Observation:" in body["messages"][2]["content"]
                                and body["messages"][2]["content"].count("\n[P") >= 18)
    client = api.AgentClient(work, "smoke", key=KEY, transport=scripted(log, clock, refuse_when), clock=clock, sleep=clock.sleep)
    client.balance("start")
    archive = smoke.SmokeArchive(work / "smoke/episodes.jsonl")
    fake_r4 = lambda pending: [agent.execute_compression(p, lambda item: agent.apply_compression(
        item, compact(latest_policy()(item.state)))) for p in pending]
    codecs = {"no_compress": agent.baseline_compression, "oracle": agent.baseline_compression,
              "RR-τ reselect": lambda item: agent.reranker_compression(item, {pid: 0.9 for pid in all_pids(item.state)}),
              "LLMLingua-2": lambda item: agent.llmlingua_compression(item, lambda text: text[: len(text) // 3], len)}
    group = run.chunks(tasks, "smoke")[0][:3]
    assert {t["dataset"] for t in group} == {"musique"}  # first three of the dataset-major chunk
    group = [tasks[d][0] for d in run.DATASETS]
    episodes = [agent.Episode(t, row, corpora[t["dataset"]], tokenizer) for t in group for row in run.ROWS]
    run.run_episodes(episodes, client, codecs, fake_r4, searchers, archive, "smoke", workers=4)
    archive.file.close()
    assert snapshot(roots) == before
    written = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file())
    assert written == ["ledger.jsonl", "smoke/attempts.jsonl", "smoke/balance_start.json", "smoke/episodes.jsonl"]
    eps = list(rows(work / "smoke/episodes.jsonl"))
    refused = [e for e in eps if e["force_reason"] == "content_refused"]
    assert len(eps) == 18 and [(e["dataset"], e["row"]) for e in refused] == [("hotpotqa", "no_compress")]
    assert refused[0]["answers"] == ["", "", ""] and refused[0]["turns"][-1]["deepseek"]["refused"] is True
    rest = [e for e in eps if e not in refused]
    assert all(e["answers"] == ["x", "y", "z"] and len(e["turns"]) == 2 for e in rest)
    assert all(t["deepseek"]["hit"] == 10 and t["deepseek"]["miss"] == 90 for e in rest for t in e["turns"])
    assert len([1 for _, p, _ in log if p == api.CHAT_PATH]) == 36
    assert api.base.scan_artifacts([work], key=KEY)["matches"] == 0
    # report on the same episodes, with Qwen patched to the same file: every DeepSeek − Qwen delta is zero
    small = tmp_path.resolve() / "small"
    (small / "smoke").mkdir(parents=True)
    with patch.object(report, "WORK", work), patch.object(report, "SMALL", small), \
            patch.object(report, "QWEN", {"smoke": work / "smoke/episodes.jsonl"}):
        report.main("smoke")
    result = read(small / "smoke/agent_ds_results.json")
    assert result["complete_tasks"] == {"musique": 1, "hotpotqa": 0, "2wiki": 1}
    assert result["content_refused_tasks"] == [["hotpotqa", tasks["hotpotqa"][0]["task"]]]
    assert result["content_refusals_by_row"]["no_compress"] == 1 and result["sensitivity_refused_as_zero"] is not None
    assert all(v is None or v["delta_f1"] == 0 for v in result["deepseek_minus_qwen"].values())
    assert result["deepseek"]["no_compress|ALL"]["deepseek_calls"] == 4
