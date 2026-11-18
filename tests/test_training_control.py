import pytest
from src.training_control import choose_batch, checkpoint_steps, validate_training_counts


def trial(batch,speed,memory=20_000_000_000):
    return dict(batch_size=batch,status='ok',optimizer_steps=30,consumed_samples=480,
                tokens_per_second=speed,selection_peak_bytes=memory,
                longest_probe=dict(oom=False,finite_gradients=True,optimizer_steps=0))


def test_fastest_eligible_configuration_wins():
    assert choose_batch([trial(1,50),trial(4,70),trial(8,90,40_000_000_001)])['batch_size']==4


def test_oom_or_missing_longest_probe_cannot_win():
    failed=trial(8,100); failed['longest_probe']['oom']=True
    assert choose_batch([trial(1,50),failed])['batch_size']==1
    with pytest.raises(ValueError): choose_batch([failed])


def test_equal_speed_has_deterministic_smaller_batch_tie_break():
    assert choose_batch([trial(8,70),trial(4,70)])['batch_size']==4


def test_exact_two_epoch_and_five_checkpoint_contract():
    validate_training_counts(2500,40000,2.0)
    checkpoint_steps([f'checkpoint-{s}' for s in (2500,500,1000,1500,2000)])
    with pytest.raises(ValueError): validate_training_counts(2500,39984,2.0)
    with pytest.raises(ValueError): checkpoint_steps(['checkpoint-2500'])
