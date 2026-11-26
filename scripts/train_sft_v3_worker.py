#!/usr/bin/env python3
"""Task 10 formal training, retaining the Task 06 template, callbacks and sample/loss accounting."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.bench_sft import gpu_sample, jsonlines, sha256
from scripts.sample_sft_subset import require, write_json_exclusive
from scripts.sft_v3_config import (
    GPU_UUID, OUT, ROOT, RUN, SAVE, archive_oldest, checkpoint, effective_arguments, read, verify_resolved_args,
)
from scripts.train_sft_worker import save
from src.training_control import MAX_LENGTH


def execute():
    require(read(OUT / 'train_run_config.json')['epochs_planned'] == 2, 'incorrect formal config')
    kwargs = effective_arguments()
    receipt = read(OUT / 'config_diff_vs_task06.json')['explicit_swift_argument_differences']
    require(all(kwargs[key] == entry['task10'] for key, entry in receipt.items()), 'formal arguments changed')
    require(not (RUN / 'checkpoints').exists() and not (OUT / 'training_stop.json').exists(),
            'prior formal training artifacts exist')
    RUN.mkdir(parents=True, exist_ok=False)
    run_dir = RUN
    write_json_exclusive(RUN / 'worker_started.json', {'pid': os.getpid(), 'ppid': os.getppid(),
                         'session_id': os.getsid(0), 'utc': datetime.now(timezone.utc).isoformat()})
    manifest = read(OUT / 'sample_manifest.json')
    train_record, eval_record = manifest['files']['train'], manifest['files']['eval_loss']
    for key, record in [('dataset', train_record), ('val_dataset', eval_record)]:
        require(kwargs[key] == [record['path']] and sha256(Path(record['path'])) == record['sha256'],
                'training subset path or bytes differ from verified manifest')
    selected = list(jsonlines(Path(train_record['path'])))
    evaluation = list(jsonlines(Path(eval_record['path'])))
    require((len(selected), len(evaluation)) == (30000, 1000), 'incorrect subset sizes')
    maximum = kwargs['max_steps']
    initial = gpu_sample()
    require(initial['index'] == 1 and initial['uuid'] == GPU_UUID
            and initial['memory_used_mib'] < 2048, f'GPU 1 busy before training: {initial}')
    result = {'status': 'running', 'optimizer_steps': 0, 'consumed_samples': 0, 'consumed_tokens': 0,
              'padded_tokens': 0, 'oom': False, 'nonfinite': False, 'initial_gpu': initial, 'pid': os.getpid(),
              'checkpoints': [], 'loss_curve': [], 'eval_curve': [], 'worker_source_sha256': sha256(Path(__file__)),
              'config_sha256': sha256(OUT / 'train_run_config.json')}
    wall_start = time.monotonic()
    try:
        import torch
        from datasets import Dataset
        from transformers import TrainerCallback
        from swift.llm import TrainArguments
        from swift.llm.train import SwiftSft
        from src.sft_protocol import encode_sft_record, make_sft_template
        if torch.cuda.device_count() != 1:
            raise ValueError('exactly one visible GPU is required')
        torch.set_num_threads(4)

        def check_finite(value, label):
            if not math.isfinite(float(value)):
                result['nonfinite'] = True
                raise FloatingPointError(f'nonfinite {label}; stopping without retry')

        class Recorder(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                result['train_start_monotonic'] = time.monotonic()
                result['startup_seconds'] = time.monotonic() - wall_start

            def on_step_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                if state.global_step > maximum:
                    raise RuntimeError('optimizer-step ceiling exceeded')
                result['optimizer_steps'] = state.global_step
                result['epoch'] = state.epoch
                result['seconds_so_far'] = time.monotonic() - result['train_start_monotonic']
                progress = {k: v for k, v in result.items() if k not in ('loss_curve', 'eval_curve')}
                save(run_dir / 'progress.json', progress)
                if state.global_step == maximum:
                    control.should_save = True
                    control.should_training_stop = True

            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                # Detect bad gradients before a NaN optimizer update can propagate.
                for parameter in model.parameters():
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        result['nonfinite'] = True
                        raise FloatingPointError('nonfinite gradient; stopping without retry')

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                row = {'step': state.global_step, 'elapsed_seconds': result.get('seconds_so_far'), **logs}
                if 'loss' in logs:
                    check_finite(logs['loss'], 'train loss')
                    result['loss_curve'].append(row)
                if 'eval_loss' in logs:
                    check_finite(logs['eval_loss'], 'eval loss')
                    result['eval_curve'].append(row)
                with (run_dir / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')

            def on_save(self, args, state, control, **kwargs):
                saved = [int(Path(p).name.split('-')[-1]) for p in result['checkpoints']]
                saved.append(state.global_step)
                result['checkpoints'] = [str(checkpoint(step)) for step in saved]
                save(run_dir / 'checkpoint_status.json', {'paths': result['checkpoints'],
                     'last_saved_step': state.global_step, 'utc': datetime.now(timezone.utc).isoformat()})

            def on_train_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                result['train_seconds'] = time.monotonic() - result['train_start_monotonic']

        class Runner(SwiftSft):
            def _prepare_dataset(self):
                reference = make_sft_template(max_length=MAX_LENGTH, processor=self.processor)
                def encode(rows, split):
                    encoded_rows = []
                    for i, row in enumerate(rows):
                        item = encode_sft_record(self.template, row['messages'])
                        if any(item[field] != row['meta'][field] for field in
                               ('n_input_tokens', 'n_target_tokens', 'total_tokens')):
                            raise ValueError('template length differs from Task 04 metadata')
                        if i == 0:
                            proof = encode_sft_record(reference, row['messages'])
                            if any(item[key] != proof[key] for key in ('input_ids', 'labels')):
                                raise ValueError('actual SwiftSft template differs from Task 04')
                        encoded_rows.append({**{key: item[key] for key in ('input_ids', 'labels')},
                                             'length': item['total_tokens']})
                        if (i + 1) % 2000 == 0:
                            print(f'ENCODE {split} {i+1}/{len(rows)}', flush=True)
                    return Dataset.from_list(encoded_rows) if rows else None
                return encode(selected, 'train'), encode(evaluation, 'eval_loss')

            def train(self, trainer):
                write_json_exclusive(RUN / 'resolved_args_diff.json',
                                     verify_resolved_args(RUN / 'checkpoints/args.json'))
                trainer.model.config.use_cache = False
                trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                def preserve_checkpoints(use_mtime=False, output_dir=None):
                    archive_oldest(output_dir or trainer.args.output_dir, trainer.args.save_total_limit)
                trainer._rotate_checkpoints = preserve_checkpoints
                original = trainer.training_step
                def counted(model, inputs, *args, **kwargs):
                    if trainer.state.global_step >= maximum:
                        raise RuntimeError('optimizer-step ceiling guard')
                    result['consumed_samples'] += inputs['input_ids'].shape[0]
                    result['consumed_tokens'] += int(inputs['attention_mask'].sum())
                    result['padded_tokens'] += inputs['input_ids'].numel()
                    loss = original(model, inputs, *args, **kwargs)
                    check_finite(loss.detach().float().cpu(), 'microbatch loss')
                    return loss
                trainer.training_step = counted
                trainer.add_callback(Recorder())
                torch.cuda.reset_peak_memory_stats()
                trained = trainer.train()
                result['trainer_metrics'] = trained.metrics
                result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                return result

        pipeline = Runner(TrainArguments(**kwargs))
        pipeline.run()
        require((result['optimizer_steps'], result['consumed_samples'], result['epoch']) == (3750, 60000, 2),
                'optimizer steps, sample exposure or epochs differ from the plan')
        require(tuple(int(Path(p).name.split('-')[-1]) for p in result['checkpoints']) == SAVE,
                'incomplete checkpoint schedule')
        result['checkpoints'] = [str(checkpoint(step)) for step in SAVE]
        eval_steps = tuple(r['step'] for r in result['eval_curve'])
        require(eval_steps == SAVE, f'incorrect fixed eval-loss schedule: {eval_steps}')
        for path in result['checkpoints']:
            for filename in ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json'):
                require((Path(path) / filename).stat().st_size > 0, f'incomplete adapter: {path}/{filename}')
        expected_tokens = sum(row['meta']['total_tokens'] for row in selected) * 2
        require(result['consumed_tokens'] == expected_tokens, 'token exposure differs from two full epochs')
        result['tokens_per_second'] = result['consumed_tokens'] / result['train_seconds']
        result['status'] = 'ok'
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                      oom='out of memory' in str(error).lower())
        save(OUT / 'training_stop.json', {'pid': os.getpid(), 'error': repr(error), 'oom': result['oom'],
             'nonfinite': result['nonfinite'], 'utc': datetime.now(timezone.utc).isoformat(),
             'automatic_retry_allowed': False})
        raise
    finally:
        result['wall_seconds'] = time.monotonic() - wall_start
        save(run_dir / 'worker_result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'optimizer_steps', 'epoch', 'tokens_per_second')}), flush=True)


if __name__ == '__main__':
    require(PROJECT == Path('/root/workplace/ctx-compressor') and Path(sys.prefix) == ROOT / 'envs/train',
            'expected the existing train Python on 5880')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '1' and os.environ.get('HF_HUB_OFFLINE') == '1',
            'GPU 1 only, offline')
    require(not any(os.environ.get(k) for k in ('http_proxy', 'https_proxy', 'all_proxy',
                              'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')), 'proxy is enabled')
    execute()

