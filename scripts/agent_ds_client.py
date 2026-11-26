"""Task 21 DeepSeek prefix-completion client for the ReAct agent turn.

Reuses the Task 14 key reader, off-peak gate, flock ledger and secret scan.
Only the request shape changes: the beta endpoint, a prefilled assistant
message and the agent stop sequence. Task 14 files are imported, not changed.
"""
from collections import Counter
import hashlib
from http.client import HTTPException, HTTPSConnection
import json
from pathlib import Path
import re
import ssl
import threading
import time

from scripts import deepseek_api_client as base
from scripts.deepseek_api_client import APIStop, yuan

MODEL = base.MODEL
HOST = base.HOST
CHAT_PATH = "/beta/chat/completions"
BALANCE_PATH = "/user/balance"
MAX_TOKENS = 256
STOP = ["\nObservation", "\nThought:"]  # smoke (11-26 14:41): DeepSeek chained several Thought/Action pairs in 12/170 turns
WORK = Path("/root/userfolder/ctx-compressor/agent_ds")
STAGES = ("probe", "smoke", "smoke2", "main")
CNY = base.CNY_UNIT
TOTAL_CAP = 20 * CNY  # user 2026-11-26: "调整上限到20" (was 8)
STAGE_CAP = {"probe": CNY // 10, "smoke": CNY // 2, "smoke2": CNY, "main": TOTAL_CAP}
# Off-peak CNY per million tokens: hit 0.02, miss 1, output 4 -> units of 1e-8 CNY per token.
# Peak (weekdays 9-12, 14-18 Beijing) is twice that; the user allowed peak calls on 11-26 ("直接开始，不用错峰").
HIT_UNITS, MISS_UNITS, OUT_UNITS = 2, 100, 400
PEAK_MULTIPLIER = 2
SLOTS = 16
CONTENT_RISK = "Content Exists Risk"  # DeepSeek moderation refusal (HTTP 400), seen in smoke2 on a Falun Gong passage


class ContentRefused(Exception):
    """The provider refused this prompt's content. Ends the episode; the run continues."""


def transport(path, body, key):
    """Same as Task 14 `direct`, scoped to the beta chat path and the free balance query."""
    if (path, body is None) not in ((CHAT_PATH, False), (BALANCE_PATH, True)):
        raise APIStop("stopped_transport_scope")
    conn = HTTPSConnection(HOST, timeout=180, context=ssl.create_default_context())
    try:
        headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        conn.request("GET" if data is None else "POST", path, body=data, headers=headers)
        response = conn.getresponse()
        raw = response.read(base.MAX_RESPONSE + 1)
        if not 200 <= response.status < 300:
            # Error bodies may echo the authorization header: keep only a key-free moderation message.
            if response.status == 400 and not base.contains_secret(raw, key):
                try:
                    message = json.loads(raw)["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    message = ""
                if isinstance(message, str) and message.startswith(CONTENT_RISK):
                    return 400, {"content_risk": True, "message": message[:300]}
            return response.status, None
        if len(raw) > base.MAX_RESPONSE:
            raise APIStop("stopped_invalid_response")
        try:
            return response.status, json.loads(raw)
        except (ValueError, UnicodeError):
            raise APIStop("stopped_invalid_response") from None
    finally:
        conn.close()


def check_messages(messages):
    """system, user, then either a prefilled assistant, or assistant + forced user + prefilled assistant."""
    roles = [m.get("role") for m in messages] if isinstance(messages, list) else None
    if roles not in (["system", "user", "assistant"], ["system", "user", "assistant", "user", "assistant"]):
        raise ValueError("invalid agent message roles")
    for i, m in enumerate(messages):
        last = i == len(messages) - 1
        keys = {"role", "content", "prefix"} if last else {"role", "content"}
        if set(m) != keys or not isinstance(m["content"], str) or (last and m["prefix"] is not True):
            raise ValueError("invalid agent message fields")
    if not messages[-1]["content"].endswith("Thought:"):
        raise ValueError("agent prefill must end with Thought:")


class AgentClient(base.DeepSeekClient):
    def __init__(self, work=WORK, stage="smoke", publish=None, *, key=None, transport=transport,
                 clock=base.now, sleep=time.sleep, allow_peak=False):
        if stage not in STAGES:
            raise ValueError("invalid Task 21 stage")
        self.work, self.stage = Path(work), stage
        self.directory = self.work / stage
        if self.work.resolve() != self.work or self.directory.resolve() != self.directory or not self.directory.is_dir():
            raise APIStop("stopped_output_scope")
        self.key = key if key is not None else base.read_key()
        self.transport, self.clock, self.sleep = transport, clock, sleep
        self.publish = publish or (lambda event: None)
        self.ledger = base.Ledger(self.work / "ledger.jsonl")
        self.peak_pauses = 0
        self.slots = threading.BoundedSemaphore(SLOTS)
        self.balance_ok = False
        self.allow_peak = allow_peak

    def gate(self):
        return self.clock() if self.allow_peak else super().gate()

    def multiplier(self, when):
        return PEAK_MULTIPLIER if base.peak_end(when) is not None else 1

    def request(self, path, body):
        if path == "/user/balance":
            path = BALANCE_PATH
        try:
            return self.transport(path, body, self.key)
        except (OSError, HTTPException):
            return 0, None

    def reserve(self, cid, attempt, worst):
        while True:
            self.gate()
            with self.ledger.locked() as add:
                dt = self.clock()
                if base.peak_end(dt) is not None and not self.allow_peak:
                    continue
                if self.stage in self.ledger.halts:
                    raise APIStop(self.ledger.halts[self.stage])
                if (self.stage, cid, attempt) in self.ledger.seen:
                    raise APIStop("stopped_duplicate_call")
                total = sum(t["reserved"] + t["spent"] for t in self.ledger.totals.values())
                mine = self.ledger.totals.get(self.stage, Counter())
                if total + worst <= TOTAL_CAP and mine["reserved"] + mine["spent"] + worst <= STAGE_CAP[self.stage]:
                    add({"event": "reserve", "stage": self.stage, "call_id": cid, "attempt": attempt,
                         "utc": dt.isoformat(), "worst_units": worst, "worst_yuan": yuan(worst)})
                    return dt
                add({"event": "halt", "stage": self.stage, "status": "stopped_cost_cap", "utc": dt.isoformat()})
            self.publish({"status": "stopped_cost_cap"})
            raise APIStop("stopped_cost_cap")

    def complete(self, call_id, messages, max_tokens=MAX_TOKENS, input_upper=None):
        if not isinstance(call_id, str) or not re.fullmatch(r"[A-Za-z0-9_:.|/-]{1,240}", call_id):
            raise ValueError("invalid call identifier")
        if max_tokens != MAX_TOKENS:
            raise ValueError("invalid completion length")
        check_messages(messages)
        self.safe(call_id)
        if self.key.encode() in base.compact(messages).encode():
            raise self.halt("stopped_secret_scan")
        if not self.balance_ok:
            raise APIStop("stopped_balance_unchecked")
        payload = {"model": MODEL, "thinking": {"type": "disabled"}, "temperature": 0, "stream": False,
                   "messages": messages, "max_tokens": max_tokens, "stop": STOP}
        estimate = base.upper_input_tokens(messages)
        if input_upper is not None:  # caller bound from the Qwen3 token count (DeepSeek used fewer tokens in the probe)
            estimate = min(estimate, int(input_upper))
        worst = (estimate * MISS_UNITS + max_tokens * OUT_UNITS) * PEAK_MULTIPLIER  # reserve at peak price
        digest = hashlib.sha256(base.compact(messages).encode("utf-8")).hexdigest()
        with self.slots:
            for attempt in range(1, 7):
                started = self.reserve(call_id, attempt, worst)
                info = {"kind": "chat", "stage": self.stage, "call_id": call_id, "attempt": attempt,
                        "requested_model": MODEL, "max_tokens": max_tokens, "estimated_input_tokens": estimate,
                        "messages_sha256": digest, "started_utc": started.isoformat(),
                        "period": "peak" if self.multiplier(started) > 1 else "off_peak"}
                try:
                    status, raw = self.request(CHAT_PATH, payload)
                except APIStop as exc:
                    self.finish(call_id, attempt, "uncertain")
                    self.attempt_log({**info, "ended_utc": self.clock().isoformat(), "error": exc.status})
                    raise self.halt(exc.status) from None
                ended = self.clock()
                info.update(ended_utc=ended.isoformat(), http_status=status,
                            latency_seconds=(ended - started).total_seconds())
                if status == 400 and isinstance(raw, dict) and raw.get("content_risk") is True:
                    self.finish(call_id, attempt, "release")
                    self.attempt_log({**info, "error": "content_risk", "message": str(raw.get("message", ""))[:300]})
                    raise ContentRefused(call_id)
                if not 200 <= status < 300:
                    self.finish(call_id, attempt, "uncertain" if status == 0 or 500 <= status < 600 else "release")
                    self.attempt_log(info)
                    self.retry_or_stop(status, attempt)
                    continue
                try:
                    usage = raw["usage"]
                    hit, miss, out = base.token_counts(usage, max_tokens)
                except (KeyError, TypeError, APIStop):
                    self.finish(call_id, attempt, "uncertain")
                    self.attempt_log({**info, "error": "stopped_usage_invalid"})
                    raise self.halt("stopped_usage_invalid") from None
                cost = (HIT_UNITS * hit + MISS_UNITS * miss + OUT_UNITS * out) * self.multiplier(started)
                self.finish(call_id, attempt, "settle", cost_units=cost, cost_yuan=yuan(cost),
                            hit=hit, miss=miss, output=out)
                try:
                    choice, = raw["choices"]
                    content = choice["message"]["content"]
                    if not all(isinstance(v, str) for v in (content, choice["finish_reason"], raw["id"], raw["model"])):
                        raise ValueError()
                    row = {**info, "attempts": attempt, "id": raw["id"], "model": raw["model"],
                           "system_fingerprint": raw.get("system_fingerprint"), "usage": usage,
                           "finish_reason": choice["finish_reason"], "content": content,
                           "cost_units": cost, "cost_yuan": yuan(cost), "hit": hit, "miss": miss, "output": out}
                    self.safe(row)
                except (KeyError, TypeError, ValueError, APIStop) as exc:
                    code = exc.status if isinstance(exc, APIStop) else "stopped_invalid_response"
                    self.attempt_log({**info, "error": code, "cost_yuan": yuan(cost)})
                    raise self.halt(code) from None
                self.attempt_log(row)
                if hit + miss > estimate or cost > worst:
                    raise self.halt("stopped_estimate_exceeded")
                return row
