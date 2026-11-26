"""Frozen identities, paths and seeds for the third GRPO round."""
from functools import lru_cache
from pathlib import Path

from scripts.grpo_r2_start import PROJECT, ROOT, read
from scripts.rollout_model import file_record
from scripts.sample_sft_subset import digest, require
from scripts.sft_run_report import validate_strategy

RUN = ROOT / 'grpo_r3'
OUT = PROJECT / 'eval/grpo_r3'
RAW = ROOT / 'eval_rollouts_grpo_r3'
MERGED = RUN / 'merged_start'
START = OUT / 'start_model.json'
PREVIOUS = ROOT / 'grpo_r2'
BASE = PREVIOUS / 'merged_start'
ADAPTER = PREVIOUS / 'train/checkpoint-125'
R2 = PROJECT / 'eval/grpo_r2'
STATES = RUN / 'states/states.jsonl'
FILTER = RUN / 'filter'
GROUPS = FILTER / 'groups.jsonl'
DATA = RUN / 'data/train.jsonl'
TRAIN = RUN / 'train'
SEED = 20261125
DATA_SEED = 20261126
MANIFEST_SHA = 'de9a401c02fc12ba4441fdb18ebb6e5c9cb40def56ef5da2b70ac05d62c1989e'


@lru_cache(maxsize=1)
def previous_manifest():
    path = R2 / 'TASK11_SHA256SUMS'
    require(digest(path) == MANIFEST_SHA, 'Task 11 checksum list changed')
    return {name:sha for sha,name in (line.split('  ', 1) for line in path.read_text().splitlines())}


def previous_file(path):
    name = str(path.relative_to(PROJECT) if path.is_relative_to(PROJECT) else path)
    require(path.resolve() == path and 'test_dev_v1' not in path.parts, 'unexpected reference path')
    record = file_record(path)
    require(previous_manifest().get(name) == record['sha256'], f'Task 11 identity differs: {path}')
    return record


def resolve_start():
    name = ADAPTER.name
    selected = R2 / 'selected_checkpoint.json'
    metrics = R2 / name / (name + '_results.json')
    steps = ROOT / 'eval_rollouts_grpo_r2' / name / 'steps.jsonl'
    previous_file(selected)
    metric = previous_file(metrics)
    trace = previous_file(steps)
    weights = previous_file(ADAPTER / 'adapter_model.safetensors')
    previous_file(BASE / 'merge_record.json')
    prior = read(selected)
    score = validate_strategy(read(metrics), name)
    require(prior['path'] == str(ADAPTER) and prior['strategy'] == name and prior['checkpoint_step'] == 125
            and prior['adapter_sha256'] == weights['sha256'] and prior['evidence_loss'] == score['combined']['evidence_loss']
            and (prior['evidence_loss']['count'], prior['evidence_loss']['n']) == (714,1494)
            and trace == score['steps_artifact'], 'Task 11 selection differs')
    return {'version':'r2','strategy':name,'path':str(ADAPTER),'base':str(BASE),
            'adapter_sha256':weights['sha256'],'metrics_path':str(metrics),'metrics_sha256':metric['sha256'],
            'steps_path':str(steps),'steps_sha256':trace['sha256'], 'evidence_loss':prior['evidence_loss'],
            'n_trajectories':1494,'selection_receipt':str(selected)}


def recorded_start():
    live = resolve_start()
    require(live == read(START), 'recorded round-three start differs from Task 11')
    return live

