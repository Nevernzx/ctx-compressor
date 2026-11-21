"""CPU-only regression tests for Task 06 sampling, config equality and detached scheduling."""
import json
import os
from pathlib import Path
import pytest

from scripts import sample_sft_v2_subset as sample
from scripts import sft_v2_train_config as cfg
from scripts import wait_sft_v2 as waiter


def test_poll_reset_and_strict_threshold():
    readings = iter([900, 1908, 1907, 1907])
    seen, slept = [], []
    waiter.wait_for_card(lambda: {'index': 1, 'uuid': waiter.GPU_UUID, 'memory_used_mib': next(readings)},
                         slept.append, lambda row, streak: seen.append(streak))
    assert seen == [1, 0, 1, 2]
    assert slept == [600, 600, 600]


def test_wrong_physical_gpu_aborts_before_training():
    with pytest.raises(ValueError, match='wrong physical GPU'):
        waiter.wait_for_card(lambda: {'index': 0, 'uuid': waiter.GPU_UUID, 'memory_used_mib': 0},
                             pytest.fail, pytest.fail)


@pytest.mark.parametrize('train_ok', [True, False])
def test_no_retry_and_eval_after_success_only(monkeypatch, tmp_path, train_ok):
    monkeypatch.setattr(waiter, 'RUN', tmp_path)
    monkeypatch.setattr(waiter, 'OUT', tmp_path)
    monkeypatch.setattr(waiter.os, 'getsid', lambda _: os.getpid())
    monkeypatch.setattr(waiter.time, 'sleep', lambda _: None)
    monkeypatch.setattr(waiter, 'gpu_sample', lambda: {'index': 1, 'uuid': waiter.GPU_UUID, 'memory_used_mib': 100})
    phases = []
    def child(stage, script, status, publish):
        phases.append(stage)
        if stage == 'training':
            if not train_ok:
                raise ValueError('out of memory')
            result = {'status': 'ok', 'nonfinite': False, 'oom': False, 'optimizer_steps': 2500,
                      'consumed_samples': 40000, 'epoch': 2., 'train_seconds': 4.,
                      'checkpoints': [f'checkpoint-{n}' for n in range(500, 2501, 500)]}
            (tmp_path / 'worker_result.json').write_text(json.dumps(result))
    monkeypatch.setattr(waiter, 'monitored_child', child)
    if train_ok:
        waiter.run()
        assert phases == ['training', 'evaluating']
    else:
        with pytest.raises(ValueError, match='out of memory'):
            waiter.run()
        assert phases == ['training']
    state = json.loads((tmp_path / 'continuation_status.json').read_text())
    assert state['status'] == ('evaluation_complete' if train_ok else 'failed')
    assert state['poll_count'] == 2
    with pytest.raises(FileExistsError):
        waiter.run()


def test_config_only_rewrites_paths():
    assert set(cfg.compare_fields({'path': str(cfg.OLD_RUN / 'train'), 'lr': 1e-4},
        {'path': str(cfg.RUN / 'train'), 'lr': 1e-4}, {'path'})) == {'path'}
    with pytest.raises(ValueError, match='non-path'):
        cfg.compare_fields({'lr': 1e-4}, {'lr': 2e-4}, set())
    with pytest.raises(ValueError, match='unexpected path'):
        cfg.compare_fields({'path': str(cfg.OLD_RUN / 'heldout')}, {'path': str(cfg.RUN / 'dev')}, {'path'})


def test_sample_append_mismatch_detected(tmp_path):
    row = {'meta': {'task': 1, 'step': 3, 'mode': 'append'}}
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.write_text(json.dumps(row) + '\n')
    new.write_text(json.dumps(row) + ' \n')
    with pytest.raises(ValueError, match='bytes changed'):
        sample.verify_rows(new, old, [row])
    new.write_bytes(old.read_bytes())
    assert sample.verify_rows(new, old, [row]) == {'rows': 1, 'append': 1, 'reselect_step0': 0}
