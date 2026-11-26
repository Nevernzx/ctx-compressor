"""Verify the original Task 10 C and D-data artifacts before resuming at D-train."""
import os
from pathlib import Path

from scripts import sft_v3_config as cfg
from scripts.sample_sft_subset import digest, require

OLD_STATUS = cfg.ROOT / 'dagger_v3/pipeline_status.json'


def check_file(record, path):
    """The configured path is authoritative even when another file has identical bytes."""
    require(isinstance(record, dict) and record.get('path') == str(path),
            f'unexpected recorded path for {path}')
    require(path.is_file(), f'missing recorded file: {path}')
    require(record.get('bytes') == path.stat().st_size, f'byte count mismatch: {path}')
    require(record.get('sha256') == digest(path), f'SHA256 mismatch: {path}')


def check_downstream_outputs():
    paths = [
        cfg.RUN,
        cfg.OUT / 'training_stop.json',
        cfg.OUT / 'training_stop.json.tmp',
        cfg.ROOT / 'eval_rollouts_v3',
        *(cfg.OUT / f'checkpoint-{step}' for step in cfg.EVAL),
        cfg.OUT / 'selected_checkpoint.json',
        cfg.ROOT / 'dagger_v3/probe',
        cfg.OUT / 'probe_v3.json',
        cfg.OUT / 'probe_v2.json',
    ]
    for path in paths:
        require(not os.path.lexists(path), f'downstream output already exists: {path}')


def verify_resume(old_status_path=OLD_STATUS):
    """Return the two verified historical stage entries with status set to reused."""
    old = cfg.read(Path(old_status_path))
    require(old.get('status') == 'failed' and old.get('stage') == 'D_train'
            and old['stage_pid'] is None, 'expected a terminated original run at D_train')
    stages = old['stages']
    require([stage['stage'] for stage in stages] == ['C', 'D_data', 'D_train'],
            'unexpected original stage sequence')
    c_stage, d_stage, stopped = stages
    require(stopped['status'] == 'failed' and stopped['pid'] is None
            and stopped.get('exit_code') is None, 'D_train already started or did not terminate before launch')
    require(stopped['gpu_before']['index'] == 1 and stopped['gpu_before']['memory_used_mib'] >= 2048
            and old['first_error'].startswith('D_train: GPU 1 busy ('), 'original failure was not the GPU 1 gate')

    check_downstream_outputs()
    rollout_path = cfg.OUT / 'dagger_rollouts.json'
    manifest_path = cfg.OUT / 'sample_manifest.json'
    for stage, path in ((c_stage, rollout_path), (d_stage, manifest_path)):
        require(stage.get('status') == 'complete' and stage.get('exit_code') == 0,
                f'{stage["stage"]} did not complete successfully')
        check_file(stage.get('receipt'), path)

    rollout = cfg.read(rollout_path)
    require(rollout['status'] == 'complete' and rollout['n_steps'] == 22158, 'incomplete C rollout receipt')
    check_file(rollout['steps_file'], cfg.ROOT / 'dagger_v3/rollouts/steps.jsonl')

    data = cfg.read(manifest_path)
    require(data['status'] == 'complete' and data['c_receipt'] == c_stage['receipt'],
            'D_data references a different C receipt or is incomplete')
    inputs = data['files']
    require(inputs['train']['lines'] == 30000 and inputs['eval_loss']['lines'] == 1000,
            'registered training or validation row counts differ')
    check_file(inputs['train'], cfg.TRAIN)
    check_file(inputs['eval_loss'], cfg.v2.RUN / 'heldout_eval_loss_1k.jsonl')
    check_file(inputs['sample_ids'], cfg.TRAIN.parent / 'sample_ids.json')
    return [{**stage, 'status':'reused'} for stage in (c_stage, d_stage)]

