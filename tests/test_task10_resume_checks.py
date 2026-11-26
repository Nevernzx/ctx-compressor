import json

import pytest

from scripts import sft_v3_resume_checks as checks
from scripts.sample_sft_subset import digest


def record(path):
    return {'path':str(path), 'bytes':path.stat().st_size, 'sha256':digest(path)}


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents))
    return record(path)


@pytest.fixture
def prior(tmp_path, monkeypatch):
    cfg = checks.cfg
    root = tmp_path / 'external'
    out = tmp_path / 'eval/sft_run_v3'
    train = root / 'data/sft_v3/train_30k.jsonl'
    old_run = root / 'runs/v2'
    for key, val in [('ROOT',root), ('RUN',root / 'runs/v3'), ('OUT',out), ('TRAIN',train)]:
        monkeypatch.setattr(cfg,key,val)
    monkeypatch.setattr(cfg.v2, 'RUN', old_run)
    rollout = {'status':'complete', 'n_steps':22158,
               'steps_file':write(root/'dagger_v3/rollouts/steps.jsonl', 'sample step')}
    c = write(out/'dagger_rollouts.json', rollout)
    files = {'train':{**write(train,'train'), 'lines':30000},
             'eval_loss':{**write(old_run/'heldout_eval_loss_1k.jsonl','eval'), 'lines':1000},
             'sample_ids':write(train.parent/'sample_ids.json', ['sample'])}
    d = write(out/'sample_manifest.json', {'status':'complete', 'c_receipt':c, 'files':files})
    old = {'status':'failed','stage':'D_train','stage_pid':None,'first_error':'D_train: GPU 1 busy (6608 MiB)',
           'stages':[{'stage':'C','status':'complete','pid':101,'exit_code':0,'receipt':c},
                     {'stage':'D_data','status':'complete','pid':102,'exit_code':0,'receipt':d},
                     {'stage':'D_train','status':'failed','pid':None,'gpu_before':{'index':1,'memory_used_mib':6608}}]}
    path = root / 'dagger_v3/pipeline_status.json'
    write(path,old)
    return path


def test_returns_original_receipts_without_changing_status(prior):
    original = prior.read_bytes()
    reused = checks.verify_resume(prior)
    assert [s['status'] for s in reused] == ['reused', 'reused']
    assert [{**s, 'status':'complete'} for s in reused] == json.loads(original)['stages'][:2]
    assert prior.read_bytes() == original


@pytest.mark.parametrize('stage', [0, 1], ids=['C','D_data'])
@pytest.mark.parametrize('field,value,reason', [('path','/copied','recorded path'),
                                               ('bytes',0,'byte count'),('sha256','0'*64,'SHA256')])
def test_rejects_inconsistent_receipt(prior,stage,field,value,reason):
    status = checks.cfg.read(prior)
    status['stages'][stage]['receipt'][field] = value
    write(prior,status)
    with pytest.raises(ValueError, match=reason):
        checks.verify_resume(prior)


@pytest.mark.parametrize('filename', ['steps.jsonl','train_30k.jsonl','heldout_eval_loss_1k.jsonl','sample_ids.json'])
def test_verifies_raw_artifact_bytes(prior,filename):
    (artifact,) = checks.cfg.ROOT.rglob(filename)
    data = bytearray(artifact.read_bytes())
    data[-1] ^= 1
    artifact.write_bytes(data)
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        checks.verify_resume(prior)


@pytest.mark.parametrize('where,filename', [('external','runs/v3'),('project','training_stop.json'),
            ('external','eval_rollouts_v3'),('project','checkpoint-3000'),
            ('project','selected_checkpoint.json'),('external','dagger_v3/probe'),
            ('project','probe_v3.json'),('project','probe_v2.json')])
def test_preexisting_output_rejected_including_broken_symlinks(prior,where,filename):
    root = checks.cfg.ROOT if where == 'external' else checks.cfg.OUT
    output = root / filename
    output.parent.mkdir(parents=True,exist_ok=True)
    output.symlink_to(root/'missing')
    with pytest.raises(ValueError,match='downstream output already exists'):
        checks.verify_resume(prior)
    assert output.is_symlink()


def test_rejects_previous_train_child(prior):
    status = checks.cfg.read(prior)
    status['stages'][2]['pid'] = 103
    write(prior,status)
    with pytest.raises(ValueError,match='D_train already started'):
        checks.verify_resume(prior)

