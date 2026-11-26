"""Train GRPO round two using Task 08's signed CLI and the installed Swift optimizer."""
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
sys.path.insert(0, str(PROJECT))
from scripts.check_formal_grpo_training import METRICS
from scripts.run_formal_swift_grpo import check_logged_finite, utc, write_json
from scripts.sample_sft_subset import digest
from src.grpo_r2_rewards import MU, take_components

ROOT = Path('/root/userfolder/ctx-compressor/grpo_r2')
BASELINE = ROOT.parent / 'grpo_v2/train'
CLI_SHA = '8a1e7aa22f81cd64984eb9bc86ce5c66bc61bf3b76f6ead40fa46e2896c2ffca'
CONFIG_SHA = '6c74f1a9fb39ab5e02ea4e09c35e71fdbeef0444ca8b303db221ec03cc180721'


def signed_json(name, expected):
    path = BASELINE / name
    if digest(path) != expected:
        raise ValueError(f'Task 08 reference changed: {path}')
    return json.loads(path.read_bytes())


def smoke_output(attempt):
    if not str(attempt).isascii() or not str(attempt).isdecimal() or str(int(attempt)) != str(attempt) or int(attempt) < 1:
        raise ValueError('invalid smoke attempt')
    return ROOT/'smoke_train'/f'attempt-{attempt}'/'train'


def validate_count(n, smoke):
    if (smoke and n != 8) or (not smoke and not 500 <= n <= 2000):
        raise ValueError('invalid registered prompt count')


def swift_argv(reference, model, dataset, out, n, *, lr=5e-6, scale_rewards='group', seed=20261123, smoke=False):
    """Keep Task 08's LR/scaling defaults here; the round-two shell passes its registered pair."""
    validate_count(n, smoke)
    if not math.isfinite(float(lr)) or float(lr) <= 0:
        raise ValueError('invalid registered prompt count or learning rate')
    if scale_rewards not in ('group', 'batch', 'none'):
        raise ValueError('unsupported reward scaling')
    argv = list(reference)
    if not argv or argv[0] != 'scripts/run_formal_swift_grpo.py' or len(argv) % 2 != 1:
        raise ValueError('unexpected Task 08 CLI')
    keys = argv[1::2]
    if len(set(keys)) != len(keys) or any(not k.startswith('--') for k in keys) or '--scale_rewards' in keys:
        raise ValueError('ambiguous Task 08 command')
    replacements = {'--model':model, '--dataset':dataset, '--output_dir':out, '--max_steps':n//4,
                    '--external_plugins':'scripts/grpo_r2_swift_plugin.py', '--reward_funcs':'ctx_step_reward_v2',
                    '--learning_rate':lr, '--seed':seed, '--data_seed':seed}
    for key, value in replacements.items():
        argv[argv.index(key)+1] = str(value)
    argv[0] = 'scripts/run_grpo_r2_training.py'
    return argv + ['--scale_rewards', scale_rewards]


def monitor_card(out, stop, errors):
    try:
        with (out/'gpu1_monitor.csv').open('x', buffering=1) as stream:
            stream.write('unix_seconds,memory_used_mib,utilization_percent\n')
            while not stop.is_set():
                data = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True, timeout=8)
                memory, util = map(int,data.strip().split(','))
                stream.write(f'{time.time():.3f},{memory},{util}\n')
                stop.wait(10)
    except Exception as e:
        errors.append(e)


def train(out, argv):
    started = time.monotonic()
    start_utc = utc()
    with (out/'trainer.pid').open('x') as f:
        f.write(f'{os.getpid()}\n')
    write_json(out/'launch_args.json', {'argv':argv,'pid':os.getpid(),'start_utc':start_utc,'mu':MU,
               'cuda_visible_devices':'1','cuda_device_order':'PCI_BUS_ID','task08_argv_sha256':CLI_SHA,
               'task08_config_sha256':CONFIG_SHA})
    stop = threading.Event()
    errors = []
    thread = threading.Thread(target=monitor_card,args=(out,stop,errors),name='gpu1-monitor',daemon=True)
    thread.start()
    try:
        import torch
        from swift.cli.utils import try_use_single_device_mode
        try_use_single_device_mode()
        from transformers import TrainerCallback
        from swift.plugin import extra_callbacks
        from swift.llm import rlhf_main

        with (out/'optimizer_steps.jsonl').open('x',buffering=1) as stream:
            class StepMonitor(TrainerCallback):
                def on_step_begin(self, args, state, control, **kwargs):
                    self.started = time.monotonic()
                    if errors:
                        raise RuntimeError('GPU 1 monitoring failed') from errors[0]

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs is None or 'loss' not in logs:
                        return
                    check_logged_finite(logs, state.global_step)
                    row = {'step':state.global_step,'utc':utc(),'step_seconds':time.monotonic()-self.started,
                           'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                           'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(),**take_components(state)}
                    # Swift appends memory(GiB) in its own logging callback.
                    row.update({k:float(logs[k]) for k in METRICS if k != 'memory(GiB)'})
                    stream.write(json.dumps(row,allow_nan=False)+'\n')

            extra_callbacks.append(StepMonitor())
            sys.argv = argv
            rlhf_main()
        if errors:
            raise RuntimeError('GPU 1 monitoring failed') from errors[0]
    except BaseException as exc:
        detail = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if isinstance(exc, ImportError):
            print('NEED: Swift training dependency unavailable:',exc,file=sys.stderr,flush=True)
        write_json(out/'training_failure.json', {'utc':utc(),'error':repr(exc),'traceback':detail,
                   'oom':any(x in detail.lower() for x in ('out of memory','outofmemoryerror','cudaerrormemoryallocation')),
                   'nonfinite':isinstance(exc,FloatingPointError)})
        raise
    finally:
        stop.set()
        thread.join()
    write_json(out/'training_runtime.json',{'status':'ok','start_utc':start_utc,'end_utc':utc(),
               'wall_seconds':time.monotonic()-started,'pid':os.getpid(),'nonfinite':False,'oom':False})


def main():
    model, data, out = (Path(os.environ[k]) for k in ('MERGED','DATA','OUT'))
    if (Path(sys.prefix).resolve() != ROOT.parent/'envs/grpo' or os.environ.get('CUDA_VISIBLE_DEVICES') != '1'
            or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or os.environ.get('HF_HUB_OFFLINE') != '1'):
        raise ValueError('round-two training requires the offline 5880 grpo environment on GPU 1')
    attempt = os.environ.get('GRPO_R2_SMOKE_ATTEMPT')
    smoke = attempt is not None
    expected_data = ROOT/'smoke_train/train.jsonl' if smoke else ROOT/'data/train.jsonl'
    expected_out = smoke_output(attempt) if smoke else ROOT/'train'
    if (model != ROOT/'merged_start' or data != expected_data or out != expected_out
            or any(p.resolve()!=p for p in (model,data,out)) or os.path.lexists(out)):
        raise ValueError('invalid inputs or existing training output')
    with data.open('rb') as stream:
        n = sum(1 for _ in stream)
    steps = n//4
    if int(os.environ.get('MAX_STEPS',steps)) != steps or int(os.environ['GRPO_SEED']) != 20261123:
        raise ValueError('registered step count or generation seed changed')
    reference = signed_json('launch_args.json',CLI_SHA)['argv']
    argv = swift_argv(reference, model,data,out,n,lr=os.environ['LR'],scale_rewards=os.environ['SCALE_REWARDS'],smoke=smoke)
    from scripts.run_formal_grpo_pipeline import gpu
    if gpu()['memory_used_mib'] >= 2048:
        raise RuntimeError('physical GPU 1 became busy before training')
    out.mkdir()
    train(out,argv)


if __name__ == '__main__':
    main()

