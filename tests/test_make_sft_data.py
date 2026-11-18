"""Streaming order and metric denominators for the full SFT conversion."""
from concurrent.futures import ThreadPoolExecutor
import threading

from scripts import make_sft_data as sft


def test_bounded_encoding_preserves_order_despite_worker_completion_order(monkeypatch):
    second_done = threading.Event()
    completion, produced, consumed = [], [], []

    def encode(task, audit):
        if task == 0:
            assert second_done.wait(5)
        completion.append(task)
        if task == 1:
            second_done.set()
        return task

    def tasks():
        for i in range(12):
            produced.append(i)
            assert len(produced) - len(consumed) <= 2
            yield i, False

    monkeypatch.setattr(sft, "encode_task", encode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in sft.bounded_results(pool, tasks(), capacity=2):
            consumed.append(result)
    assert consumed == list(range(12))
    assert completion[0] == 1


def test_statistics_use_every_sample_and_all_length_components():
    group = sft.group_factory()
    for size, add_empty, drop, binding in [(10, True, False, False), (20, False, True, True)]:
        sft.update_group(group, {"n_input_tokens": size - 2, "n_target_tokens": 2,
                                "total_tokens": size, "loss_tokens": 2,
                                "target_add_empty": add_empty,
                                "target_drop_nonempty": drop, "budget_binding": binding})
    stats = sft.summarize_group(group)
    assert stats["samples"] == 2
    assert stats["lengths"]["total_tokens"]["p50"] == 15
    assert stats["lengths"]["n_target_tokens"]["max"] == 2
    assert all(stats[field] == {"count": 1, "n": 2, "rate": 0.5} for field in sft.FLAG_FIELDS)
