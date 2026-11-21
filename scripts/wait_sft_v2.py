#!/usr/bin/env python3
"""Launch once in a new session; wait for two free-card polls, train once, then evaluate once."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.bench_sft import gpu_sample
from scripts.sample_sft_subset import require, write_json_exclusive
from scripts.sft_v2_train_config import GPU_UUID, OUT, ROOT, RUN, read
from scripts.train_sft_worker import save
from src.training_control import checkpoint_steps, validate_training_counts

POLL_SECONDS = 600
MONITOR_SECONDS = 30
GPU_LIMIT_BYTES = 2_000_000_000


def utc():
    return datetime.now(timezone.utc).isoformat()


def wait_for_card(observe, sleep, report):
    streak = 0
    while streak < 2:
        snapshot = observe()
        require(snapshot['index'] == 1 and snapshot['uuid'] == GPU_UUID, 'wrong physical GPU')
        streak = streak + 1 if snapshot['memory_used_mib'] * 2**20 < GPU_LIMIT_BYTES else 0
        report(snapshot, streak)
        if streak < 2:
            sleep(POLL_SECONDS)


def log_event(name, record):
    with (RUN / name).open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def monitored_child(stage, script, status, publish):
    # Popen owns the exact child; check its exit code rather than reusing a system PID.
    started = time.monotonic()
    with (RUN / f'{stage}.log').open('xb') as stream:
        child = subprocess.Popen(['bash', str(PROJECT / 'scripts' / script)], cwd=str(PROJECT),
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        status.update(status=stage, child_pid=child.pid, **{f'{stage}_pid': child.pid},
                      child_log=str(RUN / f'{stage}.log'))
        publish()
        while child.poll() is None:
            snap = {'utc': utc(), 'phase': stage, 'child_pid': child.pid, 'alive': True}
            try:
                snap.update(gpu_sample())
                filesystem = os.statvfs(ROOT)
                snap['external_available_bytes'] = filesystem.f_bavail * filesystem.f_frsize
                if stage == 'training':
                    path = RUN / 'progress.json'
                    progress = read(path) if path.exists() else {}
                    snap.update(optimizer_step=progress.get('optimizer_steps', 0),
                                nonfinite=progress.get('nonfinite', False), oom=progress.get('oom', False))
            except (OSError, ValueError, subprocess.SubprocessError) as err:
                snap['monitor_error'] = repr(err)
            log_event('resources.jsonl', snap)
            status['last_monitor'] = snap
            publish()
            try:
                child.wait(timeout=MONITOR_SECONDS)
            except subprocess.TimeoutExpired:
                continue
        status.update(child_exit_code=child.returncode, child_pid=None)
        status['train_wall_seconds' if stage == 'training' else 'eval_wall_seconds'] = time.monotonic() - started
        publish()
        require(child.returncode == 0, f'{stage} failed with exit code {child.returncode}; see {stage}.log')


def run():
    require(os.getsid(0) == os.getpid(), 'waiter must run in a new session')
    started = time.monotonic()
    status = {'status': 'starting', 'waiter_pid': os.getpid(), 'session_id': os.getsid(0), 'started_utc': utc(),
              'child_pid': None, 'training_pid': None, 'evaluating_pid': None, 'wait_seconds': 0,
              'poll_interval_seconds': POLL_SECONDS, 'free_if_used_bytes_strictly_less_than': GPU_LIMIT_BYTES,
              'checks_required': 2, 'monitor_interval_seconds': MONITOR_SECONDS, 'automatic_retry': False}
    write_json_exclusive(RUN / 'continuation_started.json', status)
    def publish():
        status['updated_utc'] = utc()
        save(RUN / 'continuation_status.json', status)
    def report(snapshot, streak):
        status.update(status='waiting_for_gpu', last_gpu=snapshot, consecutive_free_polls=streak,
                      wait_seconds=time.monotonic() - started, poll_count=status.get('poll_count', 0) + 1)
        log_event('card_polls.jsonl', {'poll': status['poll_count'], 'consecutive_free': streak, **snapshot})
        publish()
    publish()
    try:
        wait_for_card(gpu_sample, time.sleep, report)
        status['training_start_utc'] = utc()
        monitored_child('training', 'train_sft_v2.sh', status, publish)
        result = read(RUN / 'worker_result.json')
        require(result['status'] == 'ok' and not result['nonfinite'] and not result['oom'], 'formal training failed')
        validate_training_counts(result['optimizer_steps'], result['consumed_samples'], result['epoch'])
        checkpoint_steps(result['checkpoints'])
        require(not (OUT / 'training_stop.json').exists(), 'explicit training failure marker')
        status['train_seconds'] = result['train_seconds']
        status['evaluation_start_utc'] = utc()
        monitored_child('evaluating', 'run_heldout_model_v2.sh', status, publish)
        status.update(status='evaluation_complete', completed_utc=utc())
    except BaseException as err:
        status.update(status='failed', error=repr(err), traceback=traceback.format_exc())
        failed = RUN / 'worker_result.json'
        if failed.exists():
            result = read(failed)
            status['train_final'] = {k: result.get(k) for k in ('status', 'optimizer_steps', 'oom', 'nonfinite', 'error')}
        publish()
        raise
    publish()


def launch():
    require(PROJECT == Path('/root/workplace/ctx-compressor') and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'launch on 5880 with CUDA_VISIBLE_DEVICES empty')
    require(read(OUT / 'sample_manifest.json')['ordered_id_lists_and_source_lines_identical'],
            'Task 05 sample identities not verified')
    require(read(OUT / 'config_diff_vs_task05.json')['nonpath_hyperparameter_differences'] == [],
            'training config differs')
    for name in ('train_sft_v2.sh', 'run_heldout_model_v2.sh'):
        require((PROJECT / 'scripts' / name).is_file(), f'missing script: {name}')
    require(not (RUN / 'continuation_started.json').exists() and not (RUN / 'worker_started.json').exists(),
            'prior attempt found; automatic retries are disabled')
    with (RUN / 'continuation.log').open('xb') as stream:
        child = subprocess.Popen([sys.executable, '-u', '-B', str(Path(__file__).resolve()), 'run'], cwd=str(PROJECT),
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    print(json.dumps({'waiter_pid': child.pid, 'session_id': child.pid, 'status_file': str(RUN / 'continuation_status.json')}))


if __name__ == '__main__':
    if sys.argv[1:] == ['launch']:
        launch()
    elif sys.argv[1:] == ['run']:
        run()
    else:
        raise SystemExit('usage: wait_sft_v2.py launch')
