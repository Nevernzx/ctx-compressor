"""Run the registered Task 10 C → D-data → D-train → E → F sequence once."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import traceback
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_formal_grpo_pipeline as controller
from scripts.rollout_model import file_record
from scripts.sft_v3_config import EVAL, OUT, ROOT, RUN as TRAIN_RUN, SAVE, checkpoint, read
from scripts.sample_sft_subset import require, digest

RUN = ROOT / 'dagger_v3'
STATUS = RUN / 'pipeline_status.json'
FROZEN = OUT / 'launch_inputs.json'


def certify_file(record):
    path = Path(record['path'])
    require(record['sha256'] == digest(path), f'changed file: {path}')
    return record


def interactive_gates(archived_controller=None):
    frozen = read(FROZEN)
    for item in frozen['files']:
        if archived_controller is not None and Path(item['path']) == PROJECT / 'scripts/run_sft_v3_pipeline.py':
            certify_file({**item, 'path': str(archived_controller)})
        else:
            certify_file(item)
    pre, smoke, targets, tests = [read(OUT / name) for name in
        ('dagger_precheck.json', 'dagger_smoke.json', 'dagger_smoke_targets.json', 'cpu_tests.json')]
    require(pre['task08_id_list_equal'] and pre['heldout_overlap'] == 0 and pre['n_tasks'] == 400,
            'C task registration or split changed')
    require(smoke['n_tasks'] == 2 and smoke['n_trajectories'] == 12 and smoke['status'] == 'complete'
            and targets['status'] == 'complete' and targets['n_states'] == smoke['n_steps']
            and targets['user_text_checked'] == targets['prompt_ids_checked'] == smoke['n_steps'],
            'smoke inputs, transitions or training targets incomplete')
    certify_file(smoke['steps_file'])
    table = read(PROJECT / 'eval/reward_v2/offline_tau_table.json')
    require(table['n_states'] == 22158 and table['v1_parity']['exact'] and table['v1_parity']['n_table_cells'] == 63
            and table['v1_parity']['n_state_tau'] == 199422, 'offline reward replay inconsistent')
    require(tests['status'] == 'passed' and tests['passed'] == tests['tests']
            and tests['legacy_tests_matched'] == 568, 'full CPU regression suite incomplete')
    return file_record(FROZEN)


def verify_rollouts():
    path = OUT / 'dagger_rollouts.json'
    roll = read(path)
    require(roll['status'] == 'complete' and roll['n_tasks'] == 400 and roll['n_steps'] == 22158
            and roll['n_trajectories'] == 2400 and len(roll['cells']) == 6 and roll['tau'] == .99
            and roll['task_ids'] == read(OUT / 'dagger_precheck.json')['task_ids']
            and roll['adapter_sha256'] == controller.SHA, 'full rollout grid incomplete')
    certify_file(roll['steps_file'])
    return file_record(path)


def verify_data():
    path = OUT / 'sample_manifest.json'
    data = read(path)
    require(data['status'] == 'complete' and data['n_states'] == data['user_text_checked'] == 22158
            and data['overlength']['rate'] <= .01 and data['prompt_ids_checked'] * 100 >= data['n_states']
            and data['selected_by_mode'] == {'append':5000, 'reselect':5000}
            and data['frozen_rows_source_id_and_bytes_checked'] == 20000 and data['fixed_eval_loss_ids_checked'] == 1000
            and data['files']['train']['lines'] == 30000 and data['files']['eval_loss']['lines'] == 1000,
            'sample coverage, identity or length gate failed')
    for key in ('train', 'eval_loss', 'sample_ids'):
        certify_file(data['files'][key])
    return file_record(path)


def verify_training():
    path = TRAIN_RUN / 'worker_result.json'
    trained = read(path)
    require(trained['status'] == 'ok' and trained['optimizer_steps'] == 3750 and trained['consumed_samples'] == 60000
            and trained['epoch'] == 2 and not trained['nonfinite'] and not trained['oom']
            and [str(checkpoint(s)) for s in SAVE] == trained['checkpoints']
            and read(TRAIN_RUN / 'resolved_args_diff.json')['compared_fields'] > 0, 'training checks failed')
    return file_record(path)


def verify_evaluation(step):
    name = f'checkpoint-{step}'
    directory = OUT / name
    run = read(directory / 'run.json')
    metrics = read(directory / (name + '_results.json'))
    require(run['status'] == 'complete' and run['engine_shutdown']['completed']
            and run['step0']['n'] == 1494 and run['step0']['token_id_mismatches'] == run['step0']['user_byte_mismatches'] == 0
            and metrics['checkpoint_step'] == step and metrics['combined']['n_tasks'] == 1494
            and metrics['combined']['n_steps'] == 13956 and len(metrics['configs']) == 6, 'held-out replay incomplete')
    for key in ('extra','diagnostics'):
        require(bool(read(directory / (key + '.json'))), f'missing {key} for {name}')
    certify_file(metrics['steps_artifact'])
    return file_record(directory / 'run.json')


def verify_selection():
    path = OUT / 'selected_checkpoint.json'
    selected = read(path)
    step = selected['checkpoint_step']
    require(step in EVAL and selected['strategy'] == f'checkpoint-{step}' and selected['path'] == str(checkpoint(step))
            and selected['adapter_sha256'] == digest(checkpoint(step) / 'adapter_model.safetensors'), 'wrong E adapter')
    return file_record(path)


def verify_probe(model):
    path = OUT / f'probe_{model}.json'
    p = read(path)
    require(p['status'] == 'complete' and p['model'] == model and p['n_groups'] == 300 and p['n_completions'] == 2400
            and p['per_mode'] == {'append':150, 'reselect':150} and p['training_overlap'] == 0
            and p['adapter_sha256'] == (controller.SHA if model == 'v2' else read(OUT / 'selected_checkpoint.json')['adapter_sha256']),
            'sampling probe incomplete')
    if model == 'v2':
        require(p['selected_ids_sha256'] == read(OUT / 'probe_v3.json')['selected_ids_sha256'], 'probe cohort differs')
    return file_record(path)


class Pipeline(controller.Pipeline):
    def __init__(self, monitor_dir=None):
        super().__init__()
        self.monitor_utc = None
        self.monitor_dir = RUN if monitor_dir is None else monitor_dir

    def save(self):
        sample = self.record.get('last_monitor')
        if sample and sample['utc'] != self.monitor_utc:
            self.monitor_utc = sample['utc']
            sample['external_free_bytes'] = shutil.disk_usage(ROOT).free
            sample['overlay_free_bytes'] = shutil.disk_usage('/').free
            if sample['stage'] == 'D_train':
                progress = TRAIN_RUN / 'progress.json'
                stats = read(progress) if progress.exists() else {}
                sample.update(optimizer_step=stats.get('optimizer_steps', 0), epoch=stats.get('epoch', 0),
                              nonfinite=stats.get('nonfinite', False))
                incident = OUT / 'training_stop.json'
                if incident.exists():
                    stop = read(incident)
                    sample.update(nonfinite=stop['nonfinite'], oom=sample['oom'] or stop['oom'])
            with (self.monitor_dir / 'resource_monitor.jsonl').open('a') as stream:
                stream.write(json.dumps(sample) + '\n')
        super().save()

    def execute_stage(self, name, env_name, script, args, use_gpu, verify):
        env_path = ROOT / 'envs' / env_name
        env = {'PATH':str(env_path / 'bin') + os.pathsep + os.environ['PATH'],
               'TRAIN_ROOT':str(ROOT), 'TRAIN_MODEL':str(ROOT / 'models/Qwen3-1.7B'), 'TRAIN_ENV':str(ROOT / 'envs/train')}
        self.stage(name, [str(env_path / 'bin/python'), '-B', '-u', str(PROJECT / 'scripts' / script), *args],
                      use_gpu, env)
        self.active['receipt'] = verify()
        self.save()

    def run(self):
        history = RUN / 'history_before_resume1/run_sft_v3_pipeline.py'
        self.record['interactive_receipt'] = interactive_gates(history if history.is_file() else None)
        self.execute_stage('C', 'grpo', 'rollout_dagger_v3.py', ['run'], True, verify_rollouts)
        self.execute_stage('D_data', 'train', 'make_sft_v3_data.py', [], False, verify_data)
        self.run_remaining()

    def run_remaining(self):
        self.execute_stage('D_train', 'train', 'train_sft_v3_worker.py', [], True, verify_training)
        for step in EVAL:
            self.execute_stage(f'E_{step}', 'reader', 'eval_sft_v3.py', ['--step', str(step)], True,
                               lambda: verify_evaluation(step))
        self.execute_stage('E_select', 'reader', 'eval_sft_v3.py', ['--select'], False, verify_selection)
        for model in ('v3', 'v2'):
            self.execute_stage(f'F_{model}', 'grpo', 'probe_sft_v3.py', ['--model', model], True,
                               lambda: verify_probe(model))
        self.record.update(status='evaluation_complete', stage='finished', selected=read(OUT / 'selected_checkpoint.json'))


class ResumedPipeline(Pipeline):
    """Route the original controller's admission read through the three-sample GPU gate."""

    def publish_wait(self, event):
        self.active.update(status='waiting_gpu', gpu_wait=event)
        last = {'utc': event['utc'], 'stage': event['stage'], 'pid': None, 'alive': False,
                'oom': False, 'nonfinite': False, **event['gpu'], 'consecutive_free': event['consecutive_free']}
        self.record.update(status='waiting_gpu', last_monitor=last)
        self.save()

    def stage(self, name, argv, on_gpu, env):
        if not on_gpu:
            return super().stage(name, argv, on_gpu, env)
        from scripts.sft_v3_wait_gpu import wait_for_gpu
        physical_gpu = controller.gpu
        awaiting_admission = True

        def stage_gpu():
            nonlocal awaiting_admission
            if not awaiting_admission:
                return physical_gpu()
            awaiting_admission = False
            sample = wait_for_gpu(name, self.publish_wait, reader=physical_gpu)
            self.active.update(status='starting', admitted_utc=controller.now())
            self.record['status'] = 'running'
            self.save()
            return sample

        # The inherited stage calls gpu() once before spawning and then once per monitoring tick.
        # Capture the physical reader above so monitoring still measures the live GPU after admission.
        with patch.object(controller, 'gpu', stage_gpu):
            return super().stage(name, argv, on_gpu, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', choices=('D_train',))
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '1' and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
            and os.environ.get('HF_HUB_OFFLINE') == '1' and Path(sys.prefix) == ROOT / 'envs/grpo',
            'Task 10 requires the existing offline environment and physical GPU 1')
    resuming = args.resume_from is not None
    if resuming:
        from scripts.sft_v3_resume_checks import verify_resume
        directory = RUN / 'resume1'
        status = RUN / 'pipeline_status_resume1.json'
        reused = verify_resume(STATUS)
        frozen = interactive_gates(RUN / 'history_before_resume1/run_sft_v3_pipeline.py')
        require(frozen == read(STATUS)['interactive_receipt'], 'original interactive receipt changed')
        for path in (directory, status, status.with_suffix('.writing')):
            require(not os.path.lexists(path), f'resume output already exists: {path}')
        directory.mkdir()
        (directory / 'logs').mkdir()
        workflow = ResumedPipeline(directory)
        workflow.record.update(resume_from=args.resume_from, source_status=file_record(STATUS),
                               interactive_receipt=frozen, stages=reused)
    else:
        directory, status = RUN, STATUS
        for path in (status, directory / 'pipeline.pid', directory / 'resource_monitor.jsonl',
                     directory / 'pipeline_monitor.jsonl', RUN / 'rollouts', ROOT / 'data/sft_v3',
                     TRAIN_RUN, ROOT / 'eval_rollouts_v3'):
            require(not os.path.lexists(path), f'output already exists: {path}')
        workflow = Pipeline(directory)
    # These controller parameters are private to this interpreter.
    controller.RUN, controller.LOGS, controller.STATUS, controller.TRAIN = directory, directory / 'logs', status, TRAIN_RUN
    with (directory / 'pipeline.pid').open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    def stop(sig, frame):
        raise InterruptedError(f'pipeline received signal {sig}')
    signal.signal(signal.SIGTERM, stop)
    try:
        workflow.save()
        if resuming:
            workflow.run_remaining()
        else:
            workflow.run()
    except BaseException as e:
        workflow.record.update(status='failed', first_error=str(e), traceback=traceback.format_exc())
        if workflow.active:
            workflow.active.update(status='failed', ended_utc=controller.now())
        raise
    finally:
        workflow.record.update(ended_utc=controller.now(), stage_pid=None)
        workflow.save()


if __name__ == '__main__':
    main()

