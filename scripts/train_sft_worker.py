#!/usr/bin/env python3
"""Task 05 GPU worker: one 30-step batch trial or the authorized 2-epoch SFT.

Run via bench_batch.py or the detached train_sft.sh wrapper. Never auto-resume,
retry a NaN/OOM, modify old artifacts, or use a GPU other than physical GPU 1.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path('/root/userfolder/ctx-compressor')
OUT = PROJECT / 'eval/sft_run'
BATCH_ROOT = EXTERNAL / 'runs/batch_bench_20261117'
TRAIN_ROOT = EXTERNAL / 'runs/sft_20k_20261117'
MODEL = EXTERNAL / 'models/Qwen3-1.7B'
sys.path.insert(0, str(PROJECT))
from scripts.bench_sft import gpu_sample, jsonlines, sha256
from src.training_control import MAX_LENGTH, SEED, checkpoint_steps, validate_training_counts


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def execute(stage, batch_size):
    formal = stage == 'train'
    stop_file = OUT / 'training_stop.json'
    if stop_file.exists():
        raise RuntimeError('Task05 stop marker exists; do not start another trial or formal training')
    if formal:
        comparison = json.loads((OUT / 'batch_bench.json').read_text())
        chosen = comparison['selected']
        batch_size = chosen['batch_size']
        # The already-detached orchestrator predates the optional A8 grouping.
        # Preserve its declaration and reconcile only metadata to actual trials.
        if 'orchestrator_initial_group_by_length' not in comparison:
            comparison['orchestrator_initial_group_by_length'] = comparison['group_by_length']
        actual_flags = {}
        for trial in comparison['trials']:
            cfg_path = BATCH_ROOT / f"batch_{trial['batch_size']}" / 'worker_config.json'
            cfg = json.loads(cfg_path.read_text())
            trial['group_by_length'] = cfg['group_by_length']
            actual_flags[str(trial['batch_size'])] = cfg['group_by_length']
        grouped = actual_flags[str(batch_size)]
        comparison['group_by_length_by_batch'] = actual_flags
        comparison['group_by_length'] = 'varies_by_trial'
        comparison['selected']['group_by_length'] = grouped
        comparison['configuration_note'] = ('A1/A4 use original sample order; A8 enables the explicitly '
            'permitted length grouping before its first run. Compare complete configurations, not batch size alone. '
            'Formal training inherits the selected trial. Initial orchestrator metadata preserved separately.')
        save(OUT / 'batch_bench.json', comparison)
    else:
        grouped = batch_size == 8
    if batch_size not in (1, 4, 8):
        raise ValueError('batch must be one of the three authorized configurations')
    accumulation, maximum = 16 // batch_size, 2500 if formal else 30
    run_dir = TRAIN_ROOT if formal else BATCH_ROOT / f'batch_{batch_size}'
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / 'worker_started.json'
    if marker.exists():
        raise ValueError('worker already attempted; automatic training retries are forbidden')
    initial = gpu_sample()
    preflight = json.loads((OUT / 'preflight.json').read_text())
    if initial['uuid'] != preflight['gpu']['uuid'] or initial['memory_used_mib'] >= 2048:
        raise ValueError(f'NEED: GPU 1 must use <2 GiB at launch: {initial}')
    if formal:
        manifest = json.loads((OUT / 'sample_manifest.json').read_text())
        train_record, eval_record = manifest['files']['train'], manifest['files']['eval_loss']
    else:
        manifest = json.loads((BATCH_ROOT / 'input_manifest.json').read_text())
        train_record, eval_record = manifest['samples'], None
    for record in (train_record, eval_record):
        if record and sha256(Path(record['path'])) != record['sha256']:
            raise ValueError('dataset bytes differ from verified manifest')
    selected = list(jsonlines(Path(train_record['path'])))
    evaluation = list(jsonlines(Path(eval_record['path']))) if eval_record else []
    if len(selected) != (20000 if formal else 480) or len(evaluation) != (1000 if formal else 0):
        raise ValueError('wrong sample counts')
    save(marker, {'pid': os.getpid(), 'ppid': os.getppid(), 'session_id': os.getsid(0),
                  'stage': stage, 'initial_gpu': initial, 'utc': datetime.now(timezone.utc).isoformat()})
    config = {'stage': stage, 'batch_size': batch_size, 'gradient_accumulation_steps': accumulation,
              'effective_batch': 16, 'optimizer_steps_planned': maximum,
              'epochs_planned': 2 if formal else 1, 'model': str(MODEL), 'run_dir': str(run_dir),
              'output_dir': str(run_dir / ('checkpoints' if formal else 'output')),
              'train_input': train_record, 'eval_input': eval_record,
              'seed': SEED, 'max_length': MAX_LENGTH, 'truncation_strategy': 'raise',
              'train_type': 'lora', 'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': .05,
              'target_modules': 'all-linear', 'bf16': True, 'attention': 'sdpa',
              'gradient_checkpointing': True, 'gradient_checkpointing_use_reentrant': False,
              'learning_rate': 1e-4, 'lr_scheduler_type': 'cosine', 'warmup_ratio': .03,
              'weight_decay': 0.0, 'optim': 'adamw_torch', 'max_grad_norm': 1.0,
              'packing': False, 'group_by_length': grouped, 'loss_scale': 'last_round',
              'template': 'qwen3', 'enable_thinking': False, 'add_non_thinking_prefix': True,
              'save_steps': 500 if formal else None, 'eval_steps': 500 if formal else None,
              'checkpoint_selection_uses_eval_loss': False, 'cuda_visible_devices': '1',
              'gpu_uuid': initial['uuid'], 'pid': os.getpid(), 'initial_gpu': initial,
              'worker_source_sha256': sha256(Path(__file__)),
              'environment_lock_sha256': sha256(PROJECT / 'requirements.train.lock.txt')}
    config['versions'] = {name: version(name) for name in
                          ('ms-swift','torch','transformers','tokenizers','peft','accelerate','datasets')}
    config['driver_version'] = preflight['gpu']['driver_version']
    (run_dir / 'worker_source.py').write_bytes(Path(__file__).read_bytes())
    save(run_dir / 'worker_config.json', config)
    if formal:
        save(OUT / 'train_run_config.json', config)
    result = {'status': 'running', 'stage': stage, 'batch_size': batch_size,
              'group_by_length': grouped,
              'gradient_accumulation_steps': accumulation, 'optimizer_steps': 0,
              'consumed_samples': 0, 'consumed_tokens': 0, 'padded_tokens': 0,
              'oom': False, 'nonfinite': False, 'initial_gpu': initial, 'pid': os.getpid(),
              'checkpoints': [], 'loss_curve': [], 'eval_curve': []}
    wall_start = time.monotonic()
    monitor_stop, observations = threading.Event(), []

    def monitor():
        with (run_dir / 'resources.jsonl').open('x') as stream:
            while not monitor_stop.is_set():
                try:
                    row = gpu_sample()
                    row.update(pid=os.getpid(), alive=Path(f'/proc/{os.getpid()}').exists(),
                               optimizer_step=result['optimizer_steps'], nonfinite=result['nonfinite'],
                               root_delta_bytes=row['root_used_bytes']-preflight['root_df_used_bytes'])
                    observations.append(row)
                    stream.write(json.dumps(row) + '\n'); stream.flush()
                except Exception as error:
                    stream.write(json.dumps({'monitor_error': repr(error)}) + '\n'); stream.flush()
                monitor_stop.wait(30 if formal else 2)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
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
                path = str(Path(args.output_dir) / f'checkpoint-{state.global_step}')
                result['checkpoints'].append(path)
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
                trainer.model.config.use_cache = False
                trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
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
                if not formal:
                    # Adam states remain resident; probe has no optimizer/scheduler step.
                    rows = json.loads((BATCH_ROOT / 'longest_8.json').read_text())[:batch_size]
                    encoded = [encode_sft_record(self.template, row['messages']) for row in rows]
                    batch = self.template.data_collator([{key: row[key] for key in ('input_ids', 'labels')}
                                                         for row in encoded])
                    trainer.model.train(); trainer.model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                    inputs = trainer._prepare_inputs(batch)
                    with self.template.forward_context(trainer.model, inputs), trainer.compute_loss_context_manager():
                        loss = trainer.compute_loss(trainer.model, inputs)
                    trainer.accelerator.backward(loss)
                    torch.cuda.synchronize()
                    check_finite(loss.detach().float().cpu(), 'longest-batch loss')
                    finite = all(bool(torch.isfinite(p.grad).all()) for p in trainer.model.parameters() if p.grad is not None)
                    if not finite:
                        raise FloatingPointError('nonfinite longest-batch gradients')
                    result['longest_probe'] = {'samples': len(rows), 'meta': [row['meta'] for row in rows],
                        'sequence_lengths': [row['total_tokens'] for row in encoded],
                        'oom': False, 'finite_gradients': True, 'optimizer_steps': 0,
                        'optimizer_states_resident': True, 'seconds': time.monotonic()-started,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                    trainer.model.zero_grad(set_to_none=True)
                return result

        arguments = TrainArguments(
            model=str(MODEL), model_type='qwen3', template='qwen3', train_type='lora',
            lora_rank=16, lora_alpha=32, lora_dropout=.05, target_modules=['all-linear'],
            torch_dtype='bfloat16', bf16=True, fp16=False, attn_impl='sdpa',
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
            per_device_eval_batch_size=batch_size, prediction_loss_only=True,
            learning_rate=1e-4, lr_scheduler_type='cosine', warmup_ratio=.03,
            max_steps=maximum, num_train_epochs=2 if formal else 1, packing=False, padding_free=False,
            max_length=MAX_LENGTH, truncation_strategy='raise', strict=True,
            enable_thinking=False, add_non_thinking_prefix=True, loss_scale='last_round',
            dataset=[train_record['path']], val_dataset=[eval_record['path']] if eval_record else [],
            split_dataset_ratio=0., dataset_shuffle=False, dataset_num_proc=1,
            dataloader_num_workers=0, seed=SEED, data_seed=SEED, output_dir=config['output_dir'],
            add_version=False, save_strategy='steps' if formal else 'no', save_steps=500,
            save_total_limit=5, save_only_model=False, eval_strategy='steps' if formal else 'no',
            eval_steps=500, report_to=[], logging_steps=1, logging_first_step=True,
            logging_nan_inf_filter=False, disable_tqdm=True, use_hf=False, check_model=False,
            optim='adamw_torch', weight_decay=0., max_grad_norm=1.,
            load_from_cache_file=False, group_by_length=grouped, load_best_model_at_end=False)
        pipeline = Runner(arguments)
        pipeline.run()
        if formal:
            validate_training_counts(result['optimizer_steps'], result['consumed_samples'], result['epoch'])
            checkpoint_steps(result['checkpoints'])
            if [r['step'] for r in result['eval_curve']] != [500,1000,1500,2000,2500]:
                raise ValueError('expected five fixed heldout eval-loss measurements')
        elif result['optimizer_steps'] != 30 or result['consumed_samples'] != 480:
            raise ValueError('batch benchmark must have exactly 30 optimizer steps / 480 samples')
        expected_tokens = sum(row['meta']['total_tokens'] for row in selected) * (2 if formal else 1)
        if result['consumed_tokens'] != expected_tokens:
            raise ValueError('actual consumed token count differs from full sample exposure')
        result['tokens_per_second'] = result['consumed_tokens'] / result['train_seconds']
        result['padded_tokens_per_second'] = result['padded_tokens'] / result['train_seconds']
        result['seconds_per_sample'] = result['train_seconds'] / result['consumed_samples']
        result['status'] = 'ok'
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                      oom='out of memory' in str(error).lower())
        save(stop_file, {'stage':stage, 'batch_size':batch_size, 'pid':os.getpid(),
                        'error':repr(error), 'oom':result['oom'], 'nonfinite':result['nonfinite'],
                        'utc':datetime.now(timezone.utc).isoformat(), 'automatic_retry_allowed':False})
        raise
    finally:
        monitor_stop.set(); monitor_thread.join(timeout=10)
        result['wall_seconds'] = time.monotonic() - wall_start
        result['resource_samples'] = len(observations)
        result['monitor_peak_memory_mib'] = max((x['memory_used_mib'] for x in observations), default=0)
        result['root_peak_delta_bytes'] = max((x['root_delta_bytes'] for x in observations), default=0)
        result['selection_peak_bytes'] = max(result.get('peak_reserved_bytes', 0),
            result.get('longest_probe', {}).get('peak_reserved_bytes', 0)) + initial['memory_used_mib']*2**20
        result['selection_peak_bytes'] = max(result['selection_peak_bytes'], result['monitor_peak_memory_mib']*2**20)
        result['timing_definition'] = 'synchronized train_begin to train_end, includes scheduled eval/save for formal training; numerator actual unpadded sequence tokens'
        save(run_dir / 'worker_result.json', result)
        if formal:
            config.update(status=result['status'], runtime=result,
                          final_gpu_before_process_exit=gpu_sample(), monitor_interval_seconds=30)
            save(OUT / 'train_run_config.json', config)
    print(json.dumps({key:result.get(key) for key in ('status','batch_size','optimizer_steps','tokens_per_second','selection_peak_bytes')}), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('benchmark','train'))
    parser.add_argument('--batch-size', type=int, choices=(1,4,8), default=1)
    args=parser.parse_args()
    if PROJECT != Path('/root/workplace/ctx-compressor') or Path(sys.prefix) != EXTERNAL/'envs/train':
        parser.error('use the existing train environment on 5880')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        parser.error('only physical GPU 1 is authorized')
    if any(os.environ.get(k) for k in ('http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY')):
        parser.error('proxies must be disabled')
    execute(args.stage,args.batch_size)


if __name__=='__main__':
    main()
