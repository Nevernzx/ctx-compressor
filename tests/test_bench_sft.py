"""CPU tests for benchmark isolation and the no-truncation stop condition."""
import pytest
from scripts.bench_sft import ensure_length_gate, select_balanced


def rows(n=900):
    return [{'meta': {'split': 'train', 'task': i, 'step': 0, 'mode': mode}}
            for i in range(n) for mode in ('reselect', 'append')]


def test_balanced_sample_is_unique_reproducible_and_not_a_prefix():
    selected, population = select_balanced(rows())
    assert population == {'reselect': 900, 'append': 900}
    assert selected == select_balanced(rows())[0]
    for mode in population:
        ids = [r['meta']['task'] for r in selected if r['meta']['mode'] == mode]
        assert len(ids) == len(set(ids)) == 800
        assert max(ids) >= 800


def test_heldout_is_rejected():
    records = rows(1)
    records[0]['meta']['split'] = 'heldout'
    with pytest.raises(ValueError, match='train only'):
        select_balanced(records, per_mode=1)


@pytest.mark.parametrize('maximum,cap,allowed', [(16385,16392,False), (16384,16384,False),
                                              (100,96,True), (100,20000,True)])
def test_length_stop_cannot_be_overridden(maximum, cap, allowed):
    with pytest.raises(ValueError, match='length stop'):
        ensure_length_gate({'observed_max_tokens': maximum, 'max_length': cap,
                            'benchmark_allowed': allowed})


def test_length_gate_covers_every_sample_without_discarding():
    stats = {'observed_max_tokens': 8002, 'max_length': 8008, 'benchmark_allowed': True,
             'validation': {'truncated_samples': 0, 'dropped_samples': 0}}
    assert ensure_length_gate(stats) == 8008
    stats['validation']['dropped_samples'] = 1
    with pytest.raises(ValueError, match='zero truncated/dropped'):
        ensure_length_gate(stats)
