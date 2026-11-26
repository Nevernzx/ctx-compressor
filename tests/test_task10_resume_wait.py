import pytest

from scripts import sft_v3_wait_gpu as gate


class Clock:
    def __init__(self):
        self.t = 0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.t += delay


def reading(memory):
    return {"index": 1, "uuid": gate.controller.GPU_UUID, "memory_used_mib": memory, "utilization_pct": 0}


def test_busy_and_exact_threshold_restart_three_observation_streak():
    clock, events = Clock(), []
    memory = iter([6608, 2047, 0, 2048, 375, 375, 2047])
    sample = gate.wait_for_gpu("D_train", events.append, reader=lambda: reading(next(memory)),
                               clock=clock.now, sleep=clock.sleep)
    assert [e["consecutive_free"] for e in events] == [0, 1, 2, 0, 1, 2, 3]
    assert [e["elapsed_seconds"] for e in events] == [0, 300, 600, 900, 1200, 1500, 1800]
    assert clock.sleeps == [300] * 6
    assert sample == events[-1]["gpu"] == reading(2047)
    assert all(e["status"] == "waiting_gpu" and e["stage"] == "D_train" and e["utc"] for e in events)


def test_each_stage_observes_its_own_three_idle_samples():
    clock = Clock()
    for name in ("E_1000", "F_v3"):
        events = []
        gate.wait_for_gpu(name, events.append, reader=lambda: reading(0), clock=clock.now, sleep=clock.sleep)
        assert [e["consecutive_free"] for e in events] == [1, 2, 3]
        assert [e["elapsed_seconds"] for e in events] == [0, 300, 600]


def test_timeout_after_36_hours_even_with_two_last_idle_observations():
    clock, events = Clock(), []
    def read():
        remaining = gate.TIMEOUT_SECONDS - clock.now()
        return reading(100 if remaining <= 600 else 6000)
    with pytest.raises(TimeoutError, match="E_1000.*36 h"):
        gate.wait_for_gpu("E_1000", events.append, reader=read, clock=clock.now, sleep=clock.sleep)
    assert clock.now() == 36 * 60 * 60
    assert len(events) == 432
    assert [e["consecutive_free"] for e in events[-3:]] == [0, 1, 2]


def test_slow_last_read_cannot_admit_after_deadline():
    clock, events = Clock(), []
    def read():
        if len(events) == 2:
            clock.t = gate.TIMEOUT_SECONDS
        return reading(0)
    with pytest.raises(TimeoutError):
        gate.wait_for_gpu("F_v2", events.append, reader=read, clock=clock.now, sleep=clock.sleep)
    assert events[-1]["elapsed_seconds"] == gate.TIMEOUT_SECONDS


def test_only_physical_gpu_one_accepted():
    clock = Clock()
    def wrong():
        return {**reading(0), "index": 0}
    with pytest.raises(ValueError, match="physical GPU 1"):
        gate.wait_for_gpu("F_v3", lambda record: pytest.fail("wrong card published"),
                          reader=wrong, clock=clock.now, sleep=clock.sleep)
    assert clock.sleeps == []
