"""Direct DeepSeek client for the frozen Task 14 CNY pricing schedule."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import fcntl
from http.client import HTTPException, HTTPSConnection
import json
import os
from pathlib import Path
import re
import ssl
import stat
import threading
import time

MODEL = "deepseek-flash"
HOST = "api.deepseek.com"
KEY_FILE = Path("/root/.config/ctx-compressor/deepseek.env")
WORK = Path("/root/userfolder/ctx-compressor/deepseek_api")
CNY_UNIT = 100_000_000
CAP = 8 * CNY_UNIT
BEIJING = timezone(timedelta(hours=8))
BACKOFF = (2, 4, 8, 16, 32)
KEYLIKE = re.compile(rb"(?<![a-zA-Z0-9_])" + b"sk" + bytes([45]) + rb"[a-zA-Z0-9_-]")
MAX_RESPONSE = 4 * 1024 * 1024


class APIStop(RuntimeError):
    def __init__(self, status, detail=""):
        self.status = status
        super().__init__(status + (" " + detail if detail else ""))


def now():
    return datetime.now(timezone.utc)


def peak_end(when):
    if when.tzinfo is None:
        raise ValueError("timezone-aware time required")
    local = when.astimezone(BEIJING)
    if local.weekday() < 5:
        for start, end in ((9, 12), (14, 18)):
            if start <= local.hour < end:
                return local.replace(hour=end, minute=0, second=0, microsecond=0)
    return None


def read_key(path=KEY_FILE):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        meta = os.fstat(fd)
        if (not stat.S_ISREG(meta.st_mode) or stat.S_IMODE(meta.st_mode) != 0o600
                or meta.st_uid != os.geteuid() or meta.st_size > 8192):
            raise APIStop("stopped_key_permissions")
        parts = []
        for line in stream.read().splitlines():
            line = line.strip().removeprefix(b"export ").strip()
            if line.startswith(b"DEEPSEEK_API_KEY="):
                val = line.split(b"=", 1)[1].strip()
                if len(val) >= 2 and val[:1] == val[-1:] and val[:1] in (b"'", b'"'):
                    val = val[1:-1]
                parts.append(val)
        if len(parts) != 1 or not re.fullmatch(rb"[a-zA-Z0-9._-]{8,1024}", parts[0]):
            raise APIStop("stopped_key_format")
        return parts[0].decode("ascii")


def contains_secret(raw, key):
    return key.encode() in raw or bool(KEYLIKE.search(raw))


def scan_artifacts(paths, key=None):
    key = key or read_key()
    files = set()
    for path in map(Path, paths):
        if path.is_symlink() or any(p in path.parts for p in (".git", "test_dev_v1", "grpo_r3")):
            raise APIStop("stopped_scan_scope")
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if candidate.is_symlink() or any(p in candidate.parts for p in (".git", "test_dev_v1", "grpo_r3")):
                raise APIStop("stopped_scan_scope")
            if candidate.is_file():
                files.add(candidate)
            elif not candidate.is_dir():
                raise APIStop("stopped_scan_missing_file")
    size = 0
    for path in files:
        with path.open("rb") as f:
            tail = b""
            for chunk in iter(lambda: f.read(1 << 20), b""):
                raw = tail + chunk
                if contains_secret(raw, key):
                    raise APIStop("stopped_secret_scan")
                tail = raw[-max(len(key), 8):]
                size += len(chunk)
    return {"files": len(files), "bytes": size, "matches": 0}


def direct(path, body, key):
    if (path, body is None) not in (("/chat/completions", False), ("/user/balance", True)):
        raise APIStop("stopped_transport_scope")
    conn = HTTPSConnection(HOST, timeout=120, context=ssl.create_default_context())
    conn.set_debuglevel(0)
    try:
        headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        conn.request("GET" if data is None else "POST", path, body=data, headers=headers)
        response = conn.getresponse()
        raw = response.read(MAX_RESPONSE + 1)
        if not 200 <= response.status < 300:
            # Server error messages may echo authorization. Keep only the HTTP status.
            return response.status, None
        if len(raw) > MAX_RESPONSE:
            raise APIStop("stopped_invalid_response")
        try:
            return response.status, json.loads(raw)
        except (ValueError, UnicodeError):
            raise APIStop("stopped_invalid_response") from None
    finally:
        conn.close()


def upper_input_tokens(messages):
    # The model's byte-level tokenizer cannot need more ordinary tokens than UTF-8 bytes.
    # Also reserve room for message boundaries, the chat template and model control tokens.
    return 256 + sum(64 + len(m["content"].encode("utf-8")) for m in messages)


def token_counts(usage, max_tokens):
    fields = ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in fields):
        raise APIStop("stopped_usage_invalid")
    hit, miss, out = (usage[k] for k in fields)
    if usage.get("prompt_tokens") != hit + miss or usage.get("total_tokens") != hit + miss + out or out > max_tokens:
        raise APIStop("stopped_usage_invalid")
    return hit, miss, out


def yuan(value):
    return format(Decimal(value) / CNY_UNIT, "f")


def compact(row):
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def append_line(stream, row):
    stream.seek(0, os.SEEK_END)
    stream.write((compact(row) + "\n").encode("utf-8"))
    stream.flush()
    os.fsync(stream.fileno())


class Ledger:
    """One process-local incremental view of the append-only, flock-protected journal."""
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.seen = {}
        self.pending = {}
        self.totals = {}
        self.halts = {}
        self.mutex = threading.Lock()

    def apply(self, row):
        event, stage = row["event"], row["stage"]
        t = self.totals.setdefault(stage, Counter())
        if event == "halt":
            self.halts[stage] = row["status"]
            return
        identity = (stage, row["call_id"], row["attempt"])
        if event == "reserve":
            if identity in self.seen:
                raise APIStop("stopped_duplicate_call")
            self.seen[identity] = event
            hold = row["worst_units"]
            self.pending[identity] = hold
            t.update(attempts=1, reserved=hold)
        else:
            hold = self.pending.pop(identity)
            self.seen[identity] = event
            if event == "uncertain":
                return
            if event not in ("release", "settle"):
                raise APIStop("stopped_ledger_invalid")
            t["reserved"] -= hold
            if event == "settle":
                t.update(spent=row["cost_units"], calls=1, input_cache_hit_tokens=row["hit"],
                         input_cache_miss_tokens=row["miss"], completion_tokens=row["output"])

    @contextmanager
    def locked(self):
        with self.mutex, self.path.open("a+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if os.fstat(f.fileno()).st_size < self.offset:
                raise APIStop("stopped_ledger_shortened")
            f.seek(self.offset)
            for line in f:
                if not line.endswith(b"\n"):
                    raise APIStop("stopped_incomplete_ledger")
                self.apply(json.loads(line))
                self.offset = f.tell()
            def add(row):
                append_line(f, row)
                self.apply(row)
                self.offset = f.tell()
            yield add

    @staticmethod
    def format_total(t):
        return {"spent_yuan": yuan(t["spent"]), "reserved_yuan": yuan(t["reserved"]),
                **{k: t[k] for k in ("calls", "attempts", "input_cache_hit_tokens",
                                     "input_cache_miss_tokens", "completion_tokens")}}

    def summary(self, stage):
        with self.locked():
            total = sum(self.totals.values(), Counter())
            return {**self.format_total(total), "stage_totals": self.format_total(self.totals.get(stage, Counter()))}


class DeepSeekClient:
    def __init__(self, work=WORK, stage="smoke", publish=None, *, key=None, transport=direct, clock=now, sleep=time.sleep):
        if stage not in ("prep", "smoke", "full"):
            raise ValueError("invalid experiment stage")
        self.work, self.stage = Path(work), stage
        self.directory = self.work / stage
        if self.work.resolve() != self.work or self.directory.resolve() != self.directory or not self.directory.is_dir():
            raise APIStop("stopped_output_scope")
        self.key = key if key is not None else read_key()
        self.transport, self.clock, self.sleep = transport, clock, sleep
        self.publish = publish or (lambda event: None)
        self.ledger = Ledger(self.work / "ledger.jsonl")
        self.peak_pauses = 0
        self.slots = threading.BoundedSemaphore(8)
        self.balance_ok = False

    def safe(self, row):
        if contains_secret(compact(row).encode("utf-8"), self.key):
            raise APIStop("stopped_secret_scan")

    def gate(self):
        while True:
            dt = self.clock()
            end = peak_end(dt)
            if end is None:
                return dt
            self.peak_pauses += 1
            self.publish({"status": "paused_peak", "until_beijing": end.isoformat()})
            self.sleep((end - dt).total_seconds())

    def attempt_log(self, row):
        self.safe(row)
        with (self.directory / "attempts.jsonl").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            append_line(stream, row)

    def halt(self, code):
        with self.ledger.locked() as add:
            if self.stage not in self.ledger.halts:
                add({"stage": self.stage, "event": "halt", "status": code, "utc": self.clock().isoformat()})
        self.publish({"status": code})
        return APIStop(code)

    def reserve(self, cid, attempt, worst):
        while True:
            self.gate()
            with self.ledger.locked() as add:
                dt = self.clock()
                if peak_end(dt) is not None:
                    continue
                if self.stage in self.ledger.halts:
                    raise APIStop(self.ledger.halts[self.stage])
                if (self.stage, cid, attempt) in self.ledger.seen:
                    raise APIStop("stopped_duplicate_call")
                total = sum(t["reserved"] + t["spent"] for t in self.ledger.totals.values())
                if total + worst <= CAP:
                    add({"event": "reserve", "stage": self.stage, "call_id": cid, "attempt": attempt,
                         "utc": dt.isoformat(), "worst_units": worst, "worst_yuan": yuan(worst)})
                    return dt
                add({"event": "halt", "stage": self.stage, "status": "stopped_cost_cap", "utc": dt.isoformat()})
            self.publish({"status": "stopped_cost_cap"})
            raise APIStop("stopped_cost_cap")

    def finish(self, cid, attempt, event, **facts):
        with self.ledger.locked() as add:
            total = sum(t["spent"] for t in self.ledger.totals.values())
            add({"event": event, "stage": self.stage, "call_id": cid, "attempt": attempt,
                 "utc": self.clock().isoformat(), "cumulative_spent_yuan": yuan(total + facts.get("cost_units", 0)),
                 **facts})

    def request(self, path, body):
        try:
            return self.transport(path, body, self.key)
        except (OSError, HTTPException):
            return 0, None  # Network errors may have happened after submission.

    def balance(self, label):
        if label not in ("start", "end"):
            raise ValueError("invalid balance checkpoint")
        target = self.directory / ("balance_" + label + ".json")
        if os.path.lexists(target):
            raise FileExistsError(target)
        for attempt in range(1, 7):
            started = self.gate()
            status, raw = self.request("/user/balance", None)
            ended = self.clock()
            self.attempt_log({"kind": "balance", "label": label, "attempt": attempt, "http_status": status,
                              "started_utc": started.isoformat(), "ended_utc": ended.isoformat()})
            if 200 <= status < 300:
                break
            self.retry_or_stop(status, attempt)
        try:
            self.safe(raw)
            cny, = [item for item in raw["balance_infos"] if item["currency"] == "CNY"]
            value = Decimal(str(cny["total_balance"]))
            if not value.is_finite():
                raise ValueError()
            receipt = {"currency": "CNY", "total_balance": str(value), "response": raw,
                       "started_utc": started.isoformat(), "ended_utc": ended.isoformat(), "attempts": attempt}
        except APIStop as error:
            raise self.halt(error.status) from None
        except (ValueError, TypeError, KeyError, ArithmeticError):
            raise self.halt("stopped_balance_format") from None
        with target.open("x") as f:
            print(compact(receipt), file=f)
        self.balance_ok = raw.get("is_available") is True and value >= 1
        if not self.balance_ok:
            error = self.halt("stopped_balance")
            if label == "start":
                raise error
        return receipt

    def retry_or_stop(self, status, attempt):
        if status not in (0, 429) and not 500 <= status < 600:
            category = "stopped_balance" if status == 402 else "stopped_http_4xx" if 400 <= status < 500 else "stopped_http"
            raise self.halt(category) from None
        if attempt == 6:
            raise self.halt("stopped_retries") from None
        self.sleep(BACKOFF[attempt - 1])

    def complete(self, call_id, messages, max_tokens=48):
        if self.stage == "prep":
            raise APIStop("stopped_prep_chat_forbidden")
        if not isinstance(call_id, str) or not re.fullmatch(r"[A-Za-z0-9_:.|/-]{1,240}", call_id):
            raise ValueError("invalid call identifier")
        if max_tokens not in (48, 256):
            raise ValueError("invalid completion length")
        roles = ["user"] if max_tokens == 48 else ["system", "user"]
        if not isinstance(messages, list) or [m.get("role") for m in messages] != roles or any(
                set(m) != {"role", "content"} or not isinstance(m["content"], str) for m in messages):
            raise ValueError("invalid messages")
        self.safe(call_id)
        if self.key.encode() in compact(messages).encode():
            raise self.halt("stopped_secret_scan")
        if not self.balance_ok:
            raise APIStop("stopped_balance_unchecked")
        payload = {"model": MODEL, "thinking": {"type": "disabled"}, "temperature": 0,
                   "stream": False, "messages": messages, "max_tokens": max_tokens}
        estimate = upper_input_tokens(messages)
        worst = estimate * 100 + max_tokens * 400
        with self.slots:
            for attempt in range(1, 7):
                started = self.reserve(call_id, attempt, worst)
                base = {"kind": "chat", "stage": self.stage, "call_id": call_id, "attempt": attempt,
                        "requested_model": MODEL, "max_tokens": max_tokens, "estimated_input_tokens": estimate,
                        "started_utc": started.isoformat(), "period": "off_peak"}
                try:
                    status, raw = self.request("/chat/completions", payload)
                except APIStop as exc:
                    self.finish(call_id, attempt, "uncertain")
                    self.attempt_log({**base, "ended_utc": self.clock().isoformat(), "error": exc.status})
                    raise self.halt(exc.status) from None
                ended = self.clock()
                base.update(ended_utc=ended.isoformat(), http_status=status,
                            latency_seconds=(ended - started).total_seconds())
                if not 200 <= status < 300:
                    event = "uncertain" if status == 0 or 500 <= status < 600 else "release"
                    self.finish(call_id, attempt, event)
                    self.attempt_log(base)
                    self.retry_or_stop(status, attempt)
                    continue
                try:
                    usage = raw["usage"]
                    hit, miss, out = token_counts(usage, max_tokens)
                except (KeyError, TypeError, APIStop):
                    self.finish(call_id, attempt, "uncertain")
                    self.attempt_log({**base, "error": "stopped_usage_invalid"})
                    raise self.halt("stopped_usage_invalid") from None
                cost = 2 * hit + 100 * miss + 400 * out
                self.finish(call_id, attempt, "settle", cost_units=cost, cost_yuan=yuan(cost),
                            hit=hit, miss=miss, output=out)
                try:
                    choice, = raw["choices"]
                    content = choice["message"]["content"]
                    if not all(isinstance(v, str) for v in (content, choice["finish_reason"], raw["id"], raw["model"])):
                        raise ValueError()
                    row = {**base, "attempts": attempt, "id": raw["id"], "model": raw["model"],
                           "system_fingerprint": raw.get("system_fingerprint"), "usage": usage,
                           "finish_reason": choice["finish_reason"], "content": content, "cost_yuan": yuan(cost)}
                    self.safe(row)
                except (KeyError, TypeError, ValueError, APIStop) as exc:
                    code = exc.status if isinstance(exc, APIStop) else "stopped_invalid_response"
                    self.attempt_log({**base, "error": code, "cost_yuan": yuan(cost)})
                    raise self.halt(code) from None
                self.attempt_log(row)
                if hit + miss > estimate or cost > worst:
                    raise self.halt("stopped_estimate_exceeded")
                return row

    def summary(self):
        return {**self.ledger.summary(self.stage), "peak_pauses": self.peak_pauses}
