#!/usr/bin/env python3
"""Task 04 only: one 100-optimizer-step ms-swift LoRA benchmark on GPU 1.

Source scripts/train_environment.sh first. No network, checkpoint, evaluation,
packing, truncation, model export, or automatic retries. Preparation is CPU-only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = Path('/root/userfolder/ctx-compressor')
OUT = PROJECT / 'eval/sft_prep'
RUN = EXTERNAL / 'runs/bench_task04_20261117'
MODEL = EXTERNAL / 'models/Qwen3-1.7B'
SEED = 20261118
GPU_UUID = 'GPU-4318dc90-4126-038a-9402-1323533db94f'
sys.path.insert(0, str(PROJECT))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jsonlines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def select_balanced(rows, per_mode=800, seed=SEED):
    """Uniform reservoir sampling separately within each training mode."""
    reservoirs = {mode: [] for mode in ('reselect', 'append')}
    rng = {mode: random.Random(seed + i) for i, mode in enumerate(reservoirs)}
    seen = Counter()
    for row in rows:
        meta = row['meta']
        if meta['split'] != 'train':
            raise ValueError('benchmark samples must be from train only')
        mode = meta['mode']
        if mode not in reservoirs:
            raise ValueError('unknown mode')
        seen[mode] += 1
        if len(reservoirs[mode]) < per_mode:
            reservoirs[mode].append(row)
        else:
            pos = rng[mode].randrange(seen[mode])
            if pos < per_mode:
                reservoirs[mode][pos] = row
    if any(len(rows) != per_mode for rows in reservoirs.values()):
        raise ValueError('not enough training samples in both modes')
    selected = reservoirs['reselect'] + reservoirs['append']
    random.Random(seed).shuffle(selected)
    ids = [(r['meta']['task'], r['meta']['step'], r['meta']['mode']) for r in selected]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate benchmark sample identity')
    return selected, dict(seen)


def ensure_length_gate(stats):
    maximum = stats['observed_max_tokens']
    cap = stats['max_length']
    if not stats['benchmark_allowed'] or maximum > 16384 or not maximum <= cap <= 16384:
        raise ValueError(f'Task04 length stop: observed={maximum}, max_length={cap}; no truncation or GPU run')
    if stats['validation']['truncated_samples'] or stats['validation']['dropped_samples']:
        raise ValueError('SFT data must have zero truncated/dropped states')
    return cap


def root_used():
    return int(subprocess.check_output(['df', '-B1', '--output=used', '/'], text=True).splitlines()[-1])


def gpu_sample():
    row = subprocess.check_output([
        'nvidia-smi', '-i', '1', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    return {'index': int(row[0]), 'uuid': row[1], 'memory_used_mib': int(row[2]),
            'utilization_percent': int(row[3]), 'root_used_bytes': root_used(),
            'utc': datetime.now(timezone.utc).isoformat()}


def prepare():
    stats = json.loads((OUT / 'sft_data_stats.json').read_text())
    cap = ensure_length_gate(stats)
    if RUN.exists():
        raise ValueError('benchmark directory exists; automatic repeat is forbidden')
    started = time.monotonic()
    selected, population = select_balanced(jsonlines(Path(stats['files']['train']['path'])))
    info = stats['longest_sample']
    longest = next(row for number, row in enumerate(jsonlines(Path(info['path'])), 1)
                   if number == info['line_1based'])
    if longest['meta']['total_tokens'] != stats['observed_max_tokens']:
        raise ValueError('longest-sample locator does not match global maximum')
    RUN.mkdir(parents=True)
    selected_path = RUN / 'benchmark_samples.jsonl'
    with selected_path.open('x') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    write_json(RUN / 'longest_sample.json', longest)
    config = {
        'status': 'prepared_not_started', 'run_dir': str(RUN), 'model': str(MODEL),
        'seed': SEED, 'selection': 'Uniform reservoir: reselect seed, append seed+1; final shuffle seed',
        'samples': 1600, 'per_mode': 800, 'training_population_by_mode': population,
        'benchmark_input': {'path': str(selected_path), 'lines': 1600,
                            'bytes': selected_path.stat().st_size, 'sha256': sha256(selected_path)},
        'selection_meta': [row['meta'] for row in selected],
        'max_length': cap, 'observed_max_tokens': stats['observed_max_tokens'],
        'longest_sample': info, 'optimizer_steps': 100, 'longest_probe_optimizer_steps': 0,
        'cuda_visible_devices': '1', 'physical_gpu_uuid': GPU_UUID,
        'train_type': 'lora', 'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': 0.05,
        'target_modules': 'all-linear', 'torch_dtype': 'bfloat16', 'attn_impl': 'sdpa',
        'gradient_checkpointing': True, 'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 16, 'learning_rate': 1e-4,
        'lr_scheduler_type': 'cosine', 'warmup_ratio': 0.03, 'packing': False,
        'template': 'qwen3', 'enable_thinking': False, 'loss_scale': 'last_round',
        'truncation_strategy': 'raise', 'save_strategy': 'no', 'eval_strategy': 'no',
        'optim': 'adamw_torch', 'max_grad_norm': 1.0, 'weight_decay': 0.0,
        'selection_seconds': time.monotonic() - started,
        'sft_stats_sha256': sha256(OUT / 'sft_data_stats.json'),
    }
    write_json(OUT / 'bench_run_config.json', config)
    print(json.dumps({'prepared': True, 'max_length': cap, 'samples': 1600}), flush=True)


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('CUDA_VISIBLE_DEVICES must equal physical GPU 1')
    if Path(sys.prefix) != EXTERNAL / 'envs/train':
        raise ValueError('use only the dedicated train environment')
    if any(os.environ.get(name) for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                                            'http_proxy', 'https_proxy', 'all_proxy')):
        raise ValueError('proxy environment must be cleared')
    config = json.loads((OUT / 'bench_run_config.json').read_text())
    stats = json.loads((OUT / 'sft_data_stats.json').read_text())
    ensure_length_gate(stats)
    if config['status'] != 'prepared_not_started' or (RUN / 'started.json').exists():
        raise ValueError('benchmark already attempted: no automatic training repeat')
    if sha256(OUT / 'sft_data_stats.json') != config['sft_stats_sha256']:
        raise ValueError('SFT statistics changed after selection')
    if sha256(Path(config['benchmark_input']['path'])) != config['benchmark_input']['sha256']:
        raise ValueError('benchmark selection file changed')
    initial = gpu_sample()
    if initial['uuid'] != GPU_UUID or initial['memory_used_mib'] >= 2048:
        raise ValueError(f'NEED: GPU 1 must have <2048 MiB used before launch: {initial}')
    write_json(RUN / 'started.json', initial)
    result = {'status': 'running', 'oom': False, 'optimizer_steps': 0,
              'consumed_samples': 0, 'consumed_tokens': 0, 'loss_tokens': 0,
              'initial_gpu': initial, 'config_sha256': sha256(OUT / 'bench_run_config.json')}
    monitor_stop = threading.Event()
    monitor_rows = []

    def monitor():
        with (RUN / 'resources.jsonl').open('x') as stream:
            while not monitor_stop.is_set():
                try:
                    row = gpu_sample()
                    monitor_rows.append(row)
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                except Exception as error:
                    stream.write(json.dumps({'monitor_error': repr(error)}) + '\n')
                monitor_stop.wait(2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    wall_started = time.monotonic()
    gpu_ready_at = None
    try:
        import torch
        from datasets import Dataset
        from transformers import TrainerCallback
        from swift.llm import TrainArguments
        from swift.llm.train import SwiftSft
        from src.sft_protocol import encode_sft_record, make_sft_template
        if torch.cuda.device_count() != 1:
            raise ValueError('exactly one visible CUDA device is required')
        torch.set_num_threads(4)
        samples = list(jsonlines(Path(config['benchmark_input']['path'])))
        longest = json.loads((RUN / 'longest_sample.json').read_text())
        loss_rows = []

        class TimingCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                result['train_start_monotonic'] = time.monotonic()
                result['training_startup_seconds'] = result['train_start_monotonic'] - wall_started

            def on_step_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                if state.global_step > 100:
                    raise RuntimeError('optimizer-step ceiling exceeded')
                result['optimizer_steps'] = state.global_step
                result['training_seconds_so_far'] = time.monotonic() - result['train_start_monotonic']
                write_json(OUT / 'bench_progress.json', result)
                if state.global_step == 100:
                    control.should_training_stop = True

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and 'loss' in logs:
                    loss_rows.append({'step': state.global_step, **logs,
                                      'elapsed_seconds': result.get('training_seconds_so_far')})
                    with (RUN / 'loss_curve.jsonl').open('a') as stream:
                        stream.write(json.dumps(loss_rows[-1]) + '\n')

            def on_train_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                result['train_seconds'] = time.monotonic() - result['train_start_monotonic']

        class BenchmarkSft(SwiftSft):
            def _prepare_dataset(self):
                reference = make_sft_template(max_length=config['max_length'], processor=self.processor)
                encoded_rows = []
                for row in samples:
                    encoded = encode_sft_record(self.template, row['messages'])
                    expected = encode_sft_record(reference, row['messages'])
                    if any(encoded[key] != expected[key] for key in ('input_ids', 'labels')):
                        raise ValueError('benchmark SwiftSft differs from SFT data template')
                    for field in ('n_input_tokens', 'n_target_tokens', 'total_tokens'):
                        if encoded[field] != row['meta'][field]:
                            raise ValueError('actual training encoding differs from measured length')
                    encoded_rows.append({key: encoded[key] for key in ('input_ids', 'labels')})
                result['selected_total_tokens'] = sum(len(row['input_ids']) for row in encoded_rows)
                return Dataset.from_list(encoded_rows), None

            def train(self, trainer):
                trainer.model.config.use_cache = False
                trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                trainer.model.train()
                torch.cuda.reset_peak_memory_stats()
                original_step = trainer.training_step

                def counted_step(model, inputs, *args, **kwargs):
                    if trainer.state.global_step >= 100:
                        raise RuntimeError('no more than 100 optimizer steps are permitted')
                    result['consumed_samples'] += inputs['input_ids'].shape[0]
                    result['consumed_tokens'] += int(inputs['attention_mask'].sum())
                    result['loss_tokens'] += int((inputs['labels'] != -100).sum())
                    return original_step(model, inputs, *args, **kwargs)

                trainer.training_step = counted_step
                trainer.add_callback(TimingCallback())
                output = trainer.train()
                result['trainer_metrics'] = output.metrics
                result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                result['loss_curve'] = loss_rows
                # Probe after training so Adam optimizer states remain resident.
                # It is one forward/backward only, with no optimizer/scheduler step.
                trainer.model.train()
                trainer.model.zero_grad(set_to_none=True)
                encoded = encode_sft_record(self.template, longest['messages'])
                batch = self.template.data_collator([
                    {key: encoded[key] for key in ('input_ids', 'labels')}])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                probe_start = time.monotonic()
                prepared = trainer._prepare_inputs(batch)
                with self.template.forward_context(trainer.model, prepared), trainer.compute_loss_context_manager():
                    loss = trainer.compute_loss(trainer.model, prepared)
                trainer.accelerator.backward(loss)
                torch.cuda.synchronize()
                finite_grads = all(bool(torch.isfinite(p.grad).all()) for p in trainer.model.parameters()
                                   if p.grad is not None)
                if not torch.isfinite(loss) or not finite_grads:
                    raise ValueError('longest-sample loss/gradients not finite')
                result['longest_probe'] = {
                    'meta': longest['meta'], 'sequence_tokens': encoded['total_tokens'],
                    'seconds': time.monotonic() - probe_start, 'loss': float(loss.detach().cpu()),
                    'optimizer_steps': 0, 'finite_gradients': finite_grads, 'oom': False,
                    'optimizer_states_resident': trainer.optimizer is not None,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                trainer.model.zero_grad(set_to_none=True)
                del loss, prepared, batch, encoded
                torch.cuda.empty_cache()
                return result

        arguments = TrainArguments(
            model=str(MODEL), model_type='qwen3', template='qwen3',
            train_type='lora', lora_rank=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=['all-linear'], torch_dtype='bfloat16', bf16=True, fp16=False, attn_impl='sdpa',
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            learning_rate=1e-4, lr_scheduler_type='cosine', warmup_ratio=0.03,
            max_steps=100, num_train_epochs=1, packing=False, padding_free=False,
            max_length=config['max_length'], truncation_strategy='raise', strict=True,
            enable_thinking=False, add_non_thinking_prefix=True, loss_scale='last_round',
            dataset=[config['benchmark_input']['path']], split_dataset_ratio=0.0,
            dataset_shuffle=False, dataset_num_proc=1, dataloader_num_workers=0,
            seed=SEED, data_seed=SEED, output_dir=str(RUN / 'output'), add_version=False,
            save_strategy='no', eval_strategy='no', report_to=[], logging_steps=1,
            logging_first_step=True, disable_tqdm=True, use_hf=False,
            optim='adamw_torch', weight_decay=0.0, max_grad_norm=1.0,
            load_from_cache_file=False, group_by_length=False)
        pipeline = BenchmarkSft(arguments)
        gpu_ready_at = time.monotonic()
        result['load_and_initialization_seconds'] = gpu_ready_at - wall_started
        pipeline.run()
        if (result['optimizer_steps'] != 100 or result['consumed_samples'] != 1600
                or result['consumed_tokens'] != result['selected_total_tokens'] or len(loss_rows) != 100):
            raise ValueError('benchmark did not consume exactly 1600 samples and 100 optimizer steps')
        losses = [row['loss'] for row in loss_rows]
        if not all(math.isfinite(value) for value in losses):
            raise ValueError('nonfinite benchmark loss')
        result['first_10_loss_mean'] = statistics.mean(losses[:10])
        result['last_10_loss_mean'] = statistics.mean(losses[-10:])
        result['loss_decreased'] = result['last_10_loss_mean'] < result['first_10_loss_mean']
        result['tokens_per_second'] = result['consumed_tokens'] / result['train_seconds']
        result['seconds_per_sample'] = result['train_seconds'] / 1600
        result['mean_tokens_per_sample'] = result['consumed_tokens'] / 1600
        result['status'] = 'ok' if result['loss_decreased'] else 'fail_loss_did_not_decrease'
    except BaseException as error:
        result['status'] = 'failed'
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
        result['oom'] = 'out of memory' in str(error).lower()
        raise
    finally:
        monitor_stop.set()
        thread.join(timeout=10)
        result['wall_seconds'] = time.monotonic() - wall_started
        result['monitor_samples'] = len(monitor_rows)
        result['monitor_peak_memory_used_mib'] = max((row['memory_used_mib'] for row in monitor_rows), default=None)
        result['root_peak_used_bytes'] = max((row['root_used_bytes'] for row in monitor_rows), default=root_used())
        result['root_final_used_bytes'] = root_used()
        result['gpu_occupancy_seconds_upper_bound'] = result['wall_seconds']
        result['timing_definition'] = ('train_seconds: synchronized on_train_begin to on_train_end, includes optimizer/logging; '
            'token numerator: actual consumed unpadded full sequence tokens; single device batch1; '
            'wall_seconds includes imports/model load/encoding/probe/training, GPU occupancy conservatively uses same upper bound')
        write_json(OUT / 'bench_results.json', result)
        config['status'] = result['status']
        config['completed_optimizer_steps'] = result['optimizer_steps']
        write_json(OUT / 'bench_run_config.json', config)
    print(json.dumps({key: result[key] for key in ('status', 'optimizer_steps', 'tokens_per_second',
                                                 'peak_allocated_bytes', 'first_10_loss_mean', 'last_10_loss_mean')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    if PROJECT != Path('/root/workplace/ctx-compressor'):
        parser.error('run only on 5880 in the formal project')
    if args.command == 'prepare':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            parser.error('prepare with CUDA_VISIBLE_DEVICES empty')
        prepare()
    else:
        run()


if __name__ == '__main__':
    main()
