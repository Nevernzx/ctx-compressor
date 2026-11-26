"""Root filesystem accounting for Task 20."""
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from threading import Event, Lock, Thread
import time

ROOT_LIMIT_BYTES = 300_000_000
MIN_ROOT_FREE_BYTES = 5_000_000_000
FAST_DIRS = ("/tmp", "/root/.cache")
WORKPLACE = "/root/workplace"
PERIOD_SECONDS = 1


class RootLimitError(RuntimeError):
    pass


def root_usage():
    return shutil.disk_usage("/")


def utc():
    return datetime.now(timezone.utc).isoformat()


def allocated_bytes(paths, timeout):
    command = ["du", "--summarize", "--one-file-system", "--block-size=1", "--exclude=MRI_SAM*", "--", *paths]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
    sizes = {}
    for line in result.stdout.splitlines():
        size, path = line.split("\t", 1)
        sizes[path] = int(size)
    if set(sizes) != set(paths):
        raise ValueError("Incomplete directory sizes")
    return sizes


def writable_root_files(pid, device):
    """Read the regular writable descriptors held by this process tree."""
    todo, visited, found = [pid], set(), {}
    while todo:
        pid = todo.pop()
        if pid in visited:
            continue
        visited.add(pid)
        proc = Path("/proc") / str(pid)
        try:
            tasks = list((proc / "task").iterdir())
            descriptors = list((proc / "fd").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        for task in tasks:
            try:
                todo.extend(map(int, (task / "children").read_text().split()))
            except (FileNotFoundError, ProcessLookupError):
                continue
        for entry in descriptors:
            try:
                info = entry.stat()
                if info.st_dev != device or not stat.S_ISREG(info.st_mode):
                    continue
                flags = next(row for row in (proc / "fdinfo" / entry.name).read_text().splitlines()
                             if row.startswith("flags:"))
                if int(flags.split()[1], 8) & os.O_ACCMODE:
                    path = os.readlink(entry)
                    found[pid, info.st_ino, path] = {
                        "pid": pid, "path": path, "inode": info.st_ino, "allocated_bytes": info.st_blocks * 512,
                    }
            except (FileNotFoundError, ProcessLookupError, StopIteration):
                continue
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EPERM, errno.ESRCH):
                    raise
    return [found[key] for key in sorted(found)]


def tracked_sizes(sample):
    sizes = dict(sample.get("directories", {}))
    workplace = sample.get("workplace")
    if workplace and "bytes" in workplace:
        sizes[WORKPLACE] = workplace["bytes"]
    return sizes


def fd_growth(sample, baseline, directories):
    """Count each inode once and do not add linked files to overlapping directory totals."""
    previous = {fd["inode"]: fd["allocated_bytes"] for fd in baseline}
    all_files, uncovered = {}, {}
    for fd in sample:
        inode, path = fd["inode"], fd["path"]
        grew = max(0, fd["allocated_bytes"] - previous.get(inode, 0))
        all_files[inode] = max(all_files.get(inode, 0), grew)
        if path.endswith(" (deleted)") or not any(path.startswith(p + "/") for p in directories):
            uncovered[inode] = max(uncovered.get(inode, 0), grew)
    return sum(all_files.values()), sum(uncovered.values())


class RootPolicy:
    """Apply the four root storage rules to measurements from one attempt."""

    def __init__(self):
        self.opening = None
        self.peak = None
        self.minimum_free = None
        self.attributed_peak = 0
        self.violation = None
        self.high_point = None
        self.transients = []
        self.closing = None

    def stop_for(self, rule, value, sample, source):
        if self.violation is None:
            self.violation = {"rule": rule, "bytes": value, "source": source, "utc": sample["utc"], "engine": sample["engine"]}

    def observe(self, sample):
        if self.opening is None:
            self.opening = dict(sample)
        sizes = {
            path: max(0, value - self.opening_sizes[path])
            for path, value in tracked_sizes(sample).items() if path in self.opening_sizes
        }
        all_fds, uncovered = fd_growth(sample.get("root_writable_fds", []),
                                      self.opening.get("root_writable_fds", []), sizes)
        attributed = max(sum(sizes.values()) + uncovered, all_fds)
        growth = sample["root_used_bytes"] - self.opening["root_used_bytes"]
        free = sample["root_free_bytes"]
        sample.update(directory_growth_bytes=sizes, fd_growth_bytes=all_fds, attributed_growth_bytes=attributed,
                      global_growth_bytes=growth)
        self.attributed_peak = max(self.attributed_peak, attributed)
        self.minimum_free = free if self.minimum_free is None else min(free, self.minimum_free)
        if self.peak is None or growth > self.peak["global_growth_bytes"]:
            self.peak = sample
        if free < MIN_ROOT_FREE_BYTES:
            self.stop_for("c", free, sample, "available root bytes")
        if attributed > ROOT_LIMIT_BYTES:
            self.stop_for("b", attributed, sample, "tracked directories and writable process-tree inodes")
        if growth > ROOT_LIMIT_BYTES and attributed <= ROOT_LIMIT_BYTES:
            if self.high_point is None or growth > self.high_point["global_growth_bytes"]:
                self.high_point = {k: sample[k] for k in ("utc", "engine", "boundary", "global_growth_bytes",
                                                          "attributed_growth_bytes", "root_free_bytes")}
        elif self.high_point is not None:
            if growth <= ROOT_LIMIT_BYTES:
                self.transients.append({**self.high_point, "returned_utc": sample["utc"]})
            self.high_point = None

    @property
    def opening_sizes(self):
        return tracked_sizes(self.opening)

    def finish(self, closing):
        self.closing = closing
        sustained = closing["root_used_bytes"] - self.opening["root_used_bytes"]
        if sustained > ROOT_LIMIT_BYTES:
            self.stop_for("a", sustained, closing, "closing root use minus opening root use")

    def check(self):
        if self.violation is not None:
            v = self.violation
            raise RootLimitError(f"Task 20 storage rule {v['rule']} triggered at {v['utc']} ({v['source']}: {v['bytes']} bytes)")

    def report(self):
        return {
            "status": "failed" if self.violation else "passed",
            "root_before_bytes": self.opening["root_used_bytes"],
            "root_after_bytes": self.closing["root_used_bytes"],
            "peak_root_bytes": self.peak["root_used_bytes"],
            "sustained_growth_bytes": self.closing["global_growth_bytes"],
            "peak_growth_bytes": self.peak["global_growth_bytes"],
            "peak_attributed_growth_bytes": self.attributed_peak,
            "min_root_free_bytes": self.minimum_free,
            "limit_bytes": ROOT_LIMIT_BYTES,
            "min_free_limit_bytes": MIN_ROOT_FREE_BYTES,
            "transient_unattributed_peaks": self.transients,
            "peak_1s_sample": self.peak,
            "violation": self.violation,
        }


class RootTelemetry:
    """Sample root each second and measure the slower workplace census concurrently."""

    def __init__(self, directory):
        self.trace = directory / "root_telemetry.jsonl"
        self.workplace_trace = directory / "workplace_telemetry.jsonl"
        self.policy = RootPolicy()
        self.samples = 0
        self.active_engine = None
        self.workplace = None
        self.stop = Event()
        self.serial = Lock()
        self.device = Path("/").stat().st_dev
        self.errors = []
        self.threads = ()

    def workplace_sample(self):
        row = {"directory": WORKPLACE, "started_utc": utc()}
        try:
            row["bytes"] = allocated_bytes((WORKPLACE,), timeout=25)[WORKPLACE]
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            row["error"] = str(exc)
        row["ended_utc"] = utc()
        with self.workplace_trace.open("a", encoding="utf-8") as out:
            print(json.dumps(row), file=out)
        if "bytes" in row:
            with self.serial:
                self.workplace = row
                if self.policy.opening is not None:
                    growth = row["bytes"] - self.policy.opening_sizes[WORKPLACE]
                    if growth > ROOT_LIMIT_BYTES:
                        self.policy.attributed_peak = max(self.policy.attributed_peak, growth)
                        self.policy.stop_for("b", growth, {"utc": row["ended_utc"], "engine": self.active_engine}, WORKPLACE)
        return row

    def scan_workplace(self):
        while not self.stop.is_set():
            self.workplace_sample()
            self.stop.wait(PERIOD_SECONDS)

    def sample(self, boundary, engine):
        with self.serial:
            reading = root_usage()
            row = {"utc": utc(), "root_used_bytes": reading.used, "root_free_bytes": reading.free,
                   "engine": engine, "boundary": boundary}
            try:
                row["directories"] = allocated_bytes(FAST_DIRS, timeout=0.7)
                row["root_writable_fds"] = writable_root_files(os.getpid(), self.device)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                row["error"] = str(exc)
                self.errors.append(row["error"])
            row["fast_scan_ended_utc"] = utc()
            row["workplace"] = self.workplace
            self.policy.observe(row)
            with self.trace.open("a", encoding="utf-8") as out:
                print(json.dumps(row), file=out)
            self.samples += 1
        return row

    def poll(self):
        deadline = time.monotonic() + PERIOD_SECONDS
        while not self.stop.wait(max(0, deadline - time.monotonic())):
            self.sample("periodic", self.active_engine)
            deadline = max(deadline + PERIOD_SECONDS, time.monotonic())

    def start(self):
        for path in (self.trace, self.workplace_trace):
            with path.open("x"):
                pass
        if "bytes" not in self.workplace_sample():
            raise RuntimeError("Cannot establish the Task 20 /root/workplace baseline")
        opening = self.sample("opening", None)
        if "error" in opening:
            raise RuntimeError(f"Cannot establish the Task 20 root baseline: {opening['error']}")
        self.policy.check()
        self.threads = (Thread(target=self.scan_workplace), Thread(target=self.poll))
        for thread in self.threads:
            thread.start()

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join()
        if self.policy.opening is not None:
            self.policy.finish(self.sample("closing", None))

    @contextmanager
    def engine_start(self, name):
        self.sample("before_engine", name)
        self.policy.check()
        try:
            yield
        finally:
            self.sample("after_engine", name)
