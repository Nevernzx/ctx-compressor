"""Task 11 Phase A smoke controls and non-scaled reward evidence."""
from pathlib import Path
import json

import pytest

from scripts import smoke_grpo_r2 as smoke
from scripts import run_grpo_r2_training as train


def test_smoke_count_does_not_extend_formal_training_range():
    assert train.validate_count(8,True) is None
    with pytest.raises(ValueError):
        train.validate_count(8,False)
    with pytest.raises(ValueError):
        train.validate_count(500,True)
    assert train.validate_count(500,False) is None


@pytest.mark.parametrize('attempt',['../train','1/../train','0','-1','01'])
def test_attempt_paths_cannot_alias_formal_output(attempt):
    with pytest.raises(ValueError):
        train.smoke_output(attempt)


def test_smoke_paths_are_confined_to_separate_attempts():
    assert train.smoke_output(1) == train.ROOT/'smoke_train/attempt-1/train'
    assert train.smoke_output(1) != train.smoke_output(2)


def completion_table(step):
    return {'step':[str(step)]*32,'advantages':[-.5,.5]*16,'CtxStepRewardV2':[0.,1.]*16}


def write_tables(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def test_smoke_completion_tables_preserve_the_terminal_copy(tmp_path):
    path = tmp_path/'log'
    batches=[completion_table(1), completion_table(2)]
    counts = smoke.check_completion_tables(write_tables(path,batches+[batches[-1]]))
    assert counts == {'new_completions':64,'new_completions_per_step':[32,32],
                      'terminal_duplicate_rows':32,'informative_groups':8,'n_groups':8,'nonzero_advantages':64}
    assert smoke.check_completion_tables(write_tables(path,batches))['terminal_duplicate_rows'] == 0
    with pytest.raises(ValueError,match='terminal'):
        smoke.check_completion_tables(write_tables(path,batches+[batches[0]]))
    with pytest.raises(ValueError,match='32'):
        smoke.check_completion_tables(write_tables(path,[batches[0],{**batches[1],'step':['2']*24}]))


def test_smoke_reward_receipt_must_match_recomputed_evidence(tmp_path,monkeypatch):
    path = tmp_path/'proof'
    monkeypatch.setattr(smoke,'RECEIPT',path)
    monkeypatch.setattr(smoke,'checked_result',lambda attempt:{'status':'ok','attempt':attempt,'peak':80})
    path.write_text('{"status":"ok","attempt":1,"peak":80}')
    assert smoke.verify_smoke()['status'] == 'ok'
    path.write_text('{"status":"ok","attempt":1,"peak":20}')
    with pytest.raises(ValueError,match='receipt'):
        smoke.verify_smoke()
