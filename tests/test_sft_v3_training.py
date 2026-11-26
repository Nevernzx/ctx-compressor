import json

import pytest
from scripts import run_sft_v3_pipeline as pipeline
from scripts import sft_v3_config as config


def test_archive_preserves_each_checkpoint_and_final_step(tmp_path):
    active = tmp_path / 'checkpoints'
    active.mkdir()
    for step in config.SAVE:
        path = active / f'checkpoint-{step}'
        path.mkdir()
        (path / 'adapter').write_text(str(step))
        config.archive_oldest(active)
    assert len(list(active.iterdir())) == 5
    for step in config.SAVE:
        assert (config.checkpoint(step, tmp_path) / 'adapter').read_text() == str(step)
    assert [config.checkpoint(s, tmp_path).parent.name for s in config.EVAL] == [
        'checkpoint_archive', 'checkpoints', 'checkpoints', 'checkpoints']


def test_archive_collision_leaves_original(tmp_path):
    active, archive = (tmp_path / n for n in ('checkpoints', 'checkpoint_archive'))
    active.mkdir()
    archive.mkdir()
    for i in (1, 2):
        (active / f'checkpoint-{i}').mkdir()
    (archive / 'checkpoint-1').mkdir()
    with pytest.raises(ValueError, match='collision'):
        config.archive_oldest(active, keep=1)
    assert (active / 'checkpoint-1').is_dir()


def test_resolved_configuration_accepts_only_registered_changes(tmp_path, monkeypatch):
    original = {'output_dir':'/old', 'logging_dir':'/old/runs', 'run_name':'/old', 'max_steps':2500,
                'dataset':['/old/train'], 'training_args':'(max_steps=2500, output=/old)',
                'model_meta':'<function f at 0xDeF>', 'learning_rate':.0001}
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(original))
    monkeypatch.setattr(config, 'RESOLVED', path)
    other = str(config.RUN / 'checkpoints')
    actual = {**original, 'output_dir':other, 'run_name':other, 'logging_dir':other+'/runs', 'max_steps':3750,
              'dataset':[str(config.TRAIN)], 'training_args':f'(max_steps=3750, output={other})',
              'model_meta':'<function f at 0xABC>'}
    new_path = tmp_path / 'new.json'
    new_path.write_text(json.dumps(actual))
    assert config.verify_resolved_args(new_path)['compared_fields'] == len(original)
    actual['learning_rate'] *= 2
    new_path.write_text(json.dumps(actual))
    with pytest.raises(ValueError, match='unexpected configuration'):
        config.verify_resolved_args(new_path)


def test_training_monitor_records_actual_step_and_disk(tmp_path, monkeypatch):
    (tmp_path / 'progress.json').write_text(json.dumps({'optimizer_steps':40, 'epoch':.02, 'nonfinite':False}))
    for module, key, path in [(pipeline,'TRAIN_RUN',tmp_path),(pipeline,'OUT',tmp_path),(pipeline,'ROOT',tmp_path),
                              (pipeline,'RUN',tmp_path),(pipeline.controller,'STATUS',tmp_path/'status.json')]:
        monkeypatch.setattr(module, key, path)
    controller = pipeline.Pipeline()
    controller.record['last_monitor'] = {'utc':'a','stage':'D_train','oom':False}
    controller.save()
    controller.save()
    sample = json.loads((tmp_path / 'resource_monitor.jsonl').read_text())
    assert sample['optimizer_step'] == 40 and sample['external_free_bytes'] > 0 and not sample['nonfinite']

