"""Freeze Task 05's effective SFT arguments, changing only the data and output paths."""
import ast
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.sample_sft_subset import digest, require, write_json_exclusive

ROOT = Path('/root/userfolder/ctx-compressor')
RUN = ROOT / 'runs/sft_20k_v2_20261118'
OLD_RUN = ROOT / 'runs/sft_20k_20261117'
OUT = PROJECT / 'eval/sft_run_v2'
OLD_CONFIG = PROJECT / 'eval/sft_run/train_run_config.json'
OLD_SWIFT = OLD_RUN / 'checkpoints/args.json'
MODEL = ROOT / 'models/Qwen3-1.7B'
GPU_UUID = 'GPU-4318dc90-4126-038a-9402-1323533db94f'
# These are measured by the previous run. Keep their originals and v2 receipts separate from the hyperparameters.
OBSERVED = {'pid', 'initial_gpu', 'worker_source_sha256', 'status', 'runtime', 'final_gpu_before_process_exit'}


def read(path):
    return json.loads(path.read_bytes())


def effective_arguments():
    """Pass exactly the keywords the Task 05 worker passed, using its recorded resolved values."""
    archived = OLD_RUN / 'worker_source.py'
    require(digest(archived) == read(OLD_CONFIG)['worker_source_sha256'], 'Task 05 worker SHA256 mismatch')
    calls = [node for node in ast.walk(ast.parse(archived.read_bytes())) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'TrainArguments']
    require(len(calls) == 1, 'ambiguous Task 05 TrainArguments call')
    old_args = read(OLD_SWIFT)
    kwargs = {key.arg: old_args[key.arg] for key in calls[0].keywords}
    kwargs.update(dataset=[str(RUN / 'train_20k.jsonl')], val_dataset=[str(RUN / 'heldout_eval_loss_1k.jsonl')],
                  output_dir=str(RUN / 'checkpoints'))
    return kwargs


def compare_fields(old, new, allowed_paths):
    require(old.keys() == new.keys(), f'field set mismatch: {old.keys() ^ new.keys()}')
    diffs = {k: {'task05': old[k], 'task06': new[k]} for k in sorted(old) if old[k] != new[k]}
    require(diffs.keys() <= allowed_paths, f'unexpected non-path argument difference: {diffs.keys() - allowed_paths}')
    for difference in diffs.values():
        expected = json.dumps(difference['task05']).replace(str(OLD_RUN), str(RUN))
        require(expected == json.dumps(difference['task06']), f'unexpected path change: {difference}')
    return diffs


def verify_resolved_args(path):
    old, new = read(OLD_SWIFT), read(path)
    # ModelMeta's repr embeds the address of a Python function in each fresh process.
    for values in (old, new):
        values['model_meta'] = re.sub(r'( at )0x[0-9a-fA-F]+(?=>)', r'\1<process address>', values['model_meta'])
    diff = compare_fields(old, new, {'dataset', 'val_dataset', '_val_dataset_exists',
                        'output_dir', 'logging_dir', 'run_name', 'training_args'})
    return {'compared_fields': len(old), 'changed_path_fields': sorted(diff),
            'nonpath_differences': [], 'task05_sha256': digest(OLD_SWIFT), 'v2_sha256': digest(path)}


def prepare():
    kwargs = effective_arguments()
    old = read(OLD_CONFIG)
    old_args = read(OLD_SWIFT)
    call_diff = compare_fields({k: old_args[k] for k in kwargs}, kwargs, {'dataset', 'val_dataset', 'output_dir'})
    # The historical config mixes settings, data checksums and measurements. Retain every
    # setting, normalize the two input records to paths, and explicitly list all other fields.
    baseline = {k: v for k, v in old.items() if k not in OBSERVED}
    v2 = dict(baseline)
    for k, arg in [('train_input', 'dataset'), ('eval_input', 'val_dataset')]:
        baseline[k] = old[k]['path']
        v2[k] = kwargs[arg][0]
    v2.update(run_dir=str(RUN), output_dir=kwargs['output_dir'],
              versions={n: version(n) for n in old['versions']},
              environment_lock_sha256=digest(PROJECT / 'requirements.train.lock.txt'))
    mappings = {'batch_size': 'per_device_train_batch_size', 'optimizer_steps_planned': 'max_steps',
                'epochs_planned': 'num_train_epochs', 'attention': 'attn_impl'}
    for k in v2.keys() & kwargs.keys():
        if k != 'target_modules':
            v2[k] = kwargs[k]
    for k, arg in mappings.items():
        v2[k] = kwargs[arg]
    v2.update(target_modules=kwargs['target_modules'][0],
              effective_batch=kwargs['per_device_train_batch_size'] * kwargs['gradient_accumulation_steps'],
              gradient_checkpointing_use_reentrant=kwargs['gradient_checkpointing_kwargs']['use_reentrant'],
              checkpoint_selection_uses_eval_loss=kwargs['load_best_model_at_end'],
              cuda_visible_devices='1', gpu_uuid=GPU_UUID, monitor_interval_seconds=30)
    require((v2['batch_size'], v2['gradient_accumulation_steps'], v2['group_by_length'], v2['epochs_planned'],
             v2['optimizer_steps_planned'], v2['save_steps'], v2['eval_steps'], v2['max_length'], v2['seed']) ==
            (1, 16, False, 2, 2500, 500, 500, 8392, 20261118), 'unexpected formal SFT schedule')
    diff = compare_fields(baseline, v2, {'run_dir', 'output_dir', 'train_input', 'eval_input'})
    receipt = {'task05_config': str(OLD_CONFIG), 'task05_config_sha256': digest(OLD_CONFIG),
               'compared_config_fields': sorted(baseline), 'differences': diff,
               'explicit_swift_fields': sorted(kwargs), 'explicit_swift_argument_differences': call_diff,
               'nonpath_hyperparameter_differences': [], 'runtime_fields_separate_from_configuration': sorted(OBSERVED),
               'input_receipt_fields_separate_from_configuration': ['bytes', 'lines', 'sha256'],
               'task05_input_receipt': {k: old[k] for k in ('train_input', 'eval_input')},
               'v2_input_receipt': str(OUT / 'sample_manifest.json'),
               'actual_runtime': str(RUN / 'worker_result.json'),
               'actual_resolved_swift_comparison': str(RUN / 'resolved_args_diff.json')}
    write_json_exclusive(OUT / 'train_run_config.json', v2)
    write_json_exclusive(OUT / 'config_diff_vs_task05.json', receipt)
    return receipt


if __name__ == '__main__':
    require(PROJECT == Path('/root/workplace/ctx-compressor') and Path(sys.prefix) == ROOT / 'envs/train'
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'run in train Python on 5880 with CPU only')
    receipt = prepare()
    print('config fields:', len(receipt['compared_config_fields']), 'Swift keywords:', len(receipt['explicit_swift_fields']))
