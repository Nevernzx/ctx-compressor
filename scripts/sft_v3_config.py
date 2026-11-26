"""Task 10 schedule and field-by-field comparison with the frozen Task 06 configuration."""
from importlib.metadata import version
from pathlib import Path
import os
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import sft_v2_train_config as v2
from scripts.sample_sft_subset import require, digest, write_json_exclusive

ROOT, GPU_UUID, read = v2.ROOT, v2.GPU_UUID, v2.read
RUN = ROOT / 'runs/sft_v3_20261121'
OUT = PROJECT / 'eval/sft_run_v3'
TRAIN = ROOT / 'data/sft_v3/train_30k.jsonl'
CONFIG = v2.OUT / 'train_run_config.json'
RESOLVED = v2.RUN / 'checkpoints/args.json'
SAVE = (500, 1000, 1500, 2000, 2500, 3000, 3500, 3750)
EVAL = (1000, 2000, 3000, 3750)


def effective_arguments():
    kwargs = v2.effective_arguments()
    require(kwargs['max_steps'] == 2500 and kwargs['model'] == str(v2.MODEL)
            and not kwargs.get('adapters') and not kwargs.get('resume_from_checkpoint'),
            'expected Task 06 base model with a new adapter')
    kwargs.update(dataset=[str(TRAIN)], output_dir=str(RUN / 'checkpoints'), max_steps=3750)
    return kwargs


def diff(old, new, expected):
    require(old.keys() == new.keys(), 'configuration field names differ')
    differences = {k: {'task06': old[k], 'task10': new[k]} for k in sorted(old) if old[k] != new[k]}
    require({k: v['task10'] for k, v in differences.items()} == expected,
            f'unexpected configuration differences: {differences.keys()}, expected {expected.keys()}')
    return differences


def verify_resolved_args(path):
    old, new = read(RESOLVED), read(path)
    for values in (old, new):
        values['model_meta'] = re.sub(r'( at )0x[0-9a-fA-F]+(?=>)', r'\1<address>', values['model_meta'])
    old_dir, new_dir = old['output_dir'], str(RUN / 'checkpoints')
    expected = {k: old[k].replace(old_dir, new_dir) for k in ('output_dir', 'run_name', 'logging_dir')}
    expected.update(dataset=[str(TRAIN)], max_steps=3750,
                    training_args=old['training_args'].replace(old_dir, new_dir).replace('max_steps=2500,', 'max_steps=3750,'))
    differences = diff(old, new, expected)
    return {'compared_fields': len(old), 'differing_fields': sorted(differences),
            'source_sha256': digest(RESOLVED), 'actual_sha256': digest(path)}


def checkpoint(step, run=RUN):
    paths = [run / parent / f'checkpoint-{step}' for parent in ('checkpoints', 'checkpoint_archive')]
    found = [p for p in paths if p.is_dir()]
    require(len(found) == 1, f'expected a single saved checkpoint for step {step}: {found}')
    return found[0]


def archive_oldest(output_dir, keep=5):
    """Preserve all saved adapters while keeping Task 06's active checkpoint limit."""
    base = Path(output_dir)
    active = sorted((p for p in base.glob('checkpoint-*') if re.fullmatch(r'checkpoint-\d+', p.name)
                     and p.is_dir()), key=lambda p: int(p.name.split('-')[-1]))
    archive = base.parent / 'checkpoint_archive'
    for path in active[:-keep]:
        archive.mkdir(exist_ok=True)
        target = archive / path.name
        require(not target.exists(), f'checkpoint archive collision: {target}')
        path.rename(target)


def prepare():
    old = read(CONFIG)
    kwargs = effective_arguments()
    old_args = read(RESOLVED)
    actual = {k: old_args[k] for k in kwargs}
    explicit = diff(actual, kwargs, dict(dataset=kwargs['dataset'], output_dir=kwargs['output_dir'], max_steps=3750))
    allowed = {'run_dir': str(RUN), 'output_dir': kwargs['output_dir'], 'optimizer_steps_planned': 3750,
               'train_input': str(TRAIN)}
    new = {**old, **allowed}
    require(new['versions'] == {k: version(k) for k in old['versions']}
            and new['environment_lock_sha256'] == digest(PROJECT / 'requirements.train.lock.txt'),
            'training environment differs from Task 06')
    require(new['eval_input'] == kwargs['val_dataset'][0] and new['epochs_planned'] == 2
            and 30000 * new['epochs_planned'] == new['effective_batch'] * kwargs['max_steps'],
            'data and schedule mismatch')
    receipt = {'compared_config_fields': sorted(old), 'differences': diff(old, new, allowed),
               'explicit_swift_fields': sorted(kwargs), 'explicit_swift_argument_differences': explicit,
               'rows': {'task06': 20000, 'task10': 30000}, 'task06_config_sha256': digest(CONFIG),
               'task06_resolved_sha256': digest(RESOLVED), 'saved_steps': list(SAVE), 'eval_checkpoints': list(EVAL),
               'runtime_comparison': str(RUN / 'resolved_args_diff.json')}
    write_json_exclusive(OUT / 'train_run_config.json', new)
    write_json_exclusive(OUT / 'config_diff_vs_task06.json', receipt)


if __name__ == '__main__':
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and Path(sys.prefix) == ROOT / 'envs/train',
            'configuration check requires the CPU-only training environment')
    prepare()

