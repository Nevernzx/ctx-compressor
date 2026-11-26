"""Run the one-shot Task 11 GRPO pipeline after the interactive start-model checks."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_formal_grpo_pipeline as legacy
from scripts.grpo_r2_start import MERGED, OUT, ROOT, RUN, START, phase_a_evidence, read
from scripts.rollout_model import file_record
from scripts.run_sft_v3_pipeline import ResumedPipeline
from scripts.sample_sft_subset import digest, require, write_json_exclusive
from scripts.sft_run_report import validate_strategy
from src.heldout_eval import select_checkpoint

RAW = ROOT / 'eval_rollouts_grpo_r2'
STATUS = RUN / 'pipeline_status.json'
STATES = RUN / 'states/states.jsonl'
GROUPS = RUN / 'filter/groups.jsonl'
DATA = RUN / 'data/train.jsonl'
TRAIN = RUN / 'train'
SEED = 20261123


def command(script, *options, environment='grpo'):
    return [str(ROOT / 'envs' / environment / 'bin/python'), '-B', '-u', str(PROJECT / 'scripts' / script), *options]


def verify_states(adapter_sha):
    path = STATES.parent / 'state_checks.json'
    info = read(path)
    require(info['status'] == 'complete' and info['scope'] == 'train' and info['n_tasks'] == 400
            and info['n_trajectories'] == 2400 and info['available_checks_pass'] and len(info['cells']) == 6
            and info['source_adapter_sha256'] == adapter_sha, 'B split, policy or coverage differs')
    require(all(info[k] == 0 for k in ('gold_mismatches','reencode_mismatches','rerender_mismatches',
            'heldout_overlap','locked_test_overlap','task08_overlap','task10_overlap')), 'B input validation failed')
    require(info['n_states'] > 0 and info['state_file'] == file_record(STATES), 'B states changed')
    return info, path


def verify_filter(states):
    path = GROUPS.parent / 'sampling_probe.json'
    info = read(path)
    generation = info['generation']
    require(info['status'] == 'complete' and info['state_file'] == states['state_file']
            and info['n_states'] == states['n_states'] and len(info['cells']) == 6
            and info['mu'] == 0.5 and 0 <= info['n_informative'] <= info['n_states'], 'C coverage or reward differs')
    require(all(generation[k] == v for k,v in {'n':8,'temperature':1.0,'top_p':1.0,'top_k':-1,
            'max_tokens':256,'seed':SEED}.items()), 'C sampling settings differ')
    require(info['group_file'] == file_record(GROUPS)
            and info['informative_file'] == file_record(GROUPS.with_name('informative.jsonl')), 'C artifacts changed')
    return info, path


def verify_dataset(probe):
    path = DATA.parent / 'dataset_stats.json'
    info = read(path)
    n = info['n']
    require(info['n_informative'] == probe['n_informative'] and info['n_states'] == probe['n_states']
            and n == min(2000, probe['n_informative'])
            and info['source_states'] == probe['state_file'] and info['source_groups'] == probe['group_file']
            and info['output'] == file_record(DATA), 'D dataset lineage or sample size differs')
    require(info['rows_checked'] == info['plugin_greedy_checked'] == n and all(info[k] == 0 for k in
            ('plugin_greedy_mismatches','user_text_mismatches','heldout_rows')), 'D greedy parity or split failed')
    status = 'stopped_insufficient_signal' if n < 500 else 'complete'
    require(info['status'] == status, 'incorrect insufficient-signal decision')
    return info, path


def verify_training(n):
    path = OUT / 'training_checks.json'
    info = read(path)
    cfg = info['resolved_config']
    require(info['status'] == 'ok' and info['n_prompts'] == n and info['optimizer_steps'] == n // 4
            and info['data_sha256'] == digest(DATA) and info['reward_recheck']['mismatches'] == 0
            and info['reward_recheck']['n'] >= (n//4)*32, 'E sample count or logged reward validation failed')
    require(cfg['scale_rewards'] == 'none' and float(cfg['learning_rate']) == 1.5e-5
            and cfg['advantage_estimator'] == cfg['loss_type'] == 'grpo', 'E resolved settings differ')
    # The Task 08 verifier validates the saved checkpoint directory set and step order.
    with patch.object(legacy, 'TRAIN', TRAIN):
        paths = legacy.checkpoint_paths(info)
    require(all(info['adapter_sha256'][p.name] == digest(p/'adapter_model.safetensors') for p in paths),
            'trained adapter digest differs')
    return paths, path


def verify_eval(name, start):
    small = OUT / name
    record = small / 'run.json'
    run = read(record)
    require(run['status'] == 'complete' and run['engine_shutdown']['completed']
            and run['start_adapter_sha256'] == start['adapter_sha256'] and run['checks']['step0_matched'] == 1494
            and run['step0']['token_id_mismatches'] == run['step0']['user_byte_mismatches'] == 0,
            f'{name}: inference did not pass held-out input checks')
    metrics = validate_strategy(read(small/(name+'_results.json')),name)
    require(metrics['combined']['n_steps'] == 13956 and metrics['combined']['n_tasks'] == 1494
            and metrics['steps_artifact'] == file_record(RAW/name/'steps.jsonl'), f'{name}: raw eval differs')
    require(read(small/'extra.json')['combined']['heldout_reward_v2'] and read(small/'diagnostics.json'),
            f'{name}: incomplete diagnostics or held-out reward breakdown')
    return metrics, record


class Pipeline(ResumedPipeline):
    def __init__(self):
        super().__init__(RUN)

    def step(self, label, argv, gpu, verifier, env=None):
        stage_env = dict(env or {})
        if argv[0].endswith('/bin/python'):
            stage_env['PATH'] = str(Path(argv[0]).parent) + os.pathsep + os.environ['PATH']
        self.stage(label, argv, gpu, stage_env)
        result, path = verifier()
        self.active['receipt'] = file_record(path)
        self.save()
        return result

    def evaluate(self, name, start):
        label = 'F0' if name == 'merged_start' else 'F_'+name
        return self.step(label, command('eval_grpo_r2.py','--name',name,environment='reader'), True,
                         lambda: verify_eval(name,start))

    def run(self):
        self.record['interactive_preflight'] = phase_a_evidence()
        start = read(START)
        self.evaluate('merged_start',start)
        states = self.step('B', command('build_grpo_r2_states.py'), True, lambda: verify_states(start['adapter_sha256']))
        probe = self.step('C', command('probe_grpo_r2.py'), True, lambda: verify_filter(states))
        data = self.step('D', command('make_grpo_r2_dataset.py'), False, lambda: verify_dataset(probe))
        n = data['n']
        if n < 500:
            self.record.update(stage='D', status='stopped_insufficient_signal', n_informative=probe['n_informative'],
                               n_prompts=n, minimum_informative=500)
            return
        runtime_path = TRAIN/'training_runtime.json'
        def training_finished():
            runtime = read(runtime_path)
            require(runtime['status']=='ok' and not runtime['oom'] and not runtime['nonfinite'], 'E training failed')
            return runtime,runtime_path
        self.step('E',['bash',str(PROJECT/'scripts/train_grpo_r2.sh')], True, training_finished,
                  {'MERGED':str(MERGED),'DATA':str(DATA),'OUT':str(TRAIN),'MAX_STEPS':str(n//4),
                   'LR':'1.5e-5','SCALE_REWARDS':'none','GRPO_SEED':str(SEED),'MU':'0.5'})
        adapters = self.step('E_check',command('check_grpo_r2_training.py','--data',str(DATA),'--train-dir',str(TRAIN)),
                             False, lambda: verify_training(n))
        results = [self.evaluate(adapter.name,start) for adapter in adapters]
        selected = select_checkpoint(results)
        name = selected['strategy']
        summary = {'strategy':name, 'path':str(TRAIN/name), 'checkpoint_step':selected['checkpoint_step'],
                   'n_candidates':len(results),'evidence_loss':selected['combined']['evidence_loss'],
                   'final_recall':selected['combined']['final_recall'],
                   'adapter_sha256':digest(TRAIN/name/'adapter_model.safetensors'),
                   'rule':'Lowest pooled loss, highest pooled final recall, earliest optimizer step (GRPO only).'}
        write_json_exclusive(OUT/'selected_checkpoint.json',summary)
        self.record.update(stage='finished', status='evaluation_complete', selected_checkpoint=name)


def require_new_run():
    for path in (STATUS, STATUS.with_suffix('.writing'), RUN/'pipeline.pid', RUN/'pipeline_monitor.jsonl',
                 RUN/'resource_monitor.jsonl', RUN/'logs', STATES.parent, GROUPS.parent, DATA.parent, TRAIN, RAW,
                 OUT/'merged_start', OUT/'selected_checkpoint.json', OUT/'training_checks.json', OUT/'training_curve.csv'):
        require(not os.path.lexists(path), f'output already exists: {path}')
    require(not any(OUT.glob('checkpoint-*')), 'held-out checkpoint output already exists')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true',help='start the validated pipeline in an independent session')
    args=parser.parse_args()
    require(PROJECT == Path('/root/workplace/ctx-compressor') and Path(sys.prefix).resolve() == ROOT/'envs/grpo'
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '1' and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
            and os.environ.get('HF_HUB_OFFLINE') == '1', 'use the Task 11 offline environment on physical GPU 1')
    require_new_run()
    require(START.is_file() and (OUT/'phase_a_tests.json').is_file(), 'interactive Phase A receipts required')
    if args.detach:
        # Verify before creating the launcher log. The independent child repeats all checks in Pipeline.run().
        phase_a_evidence()
        log = RUN/'launcher.log'
        with log.open('xb') as output:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())], cwd=PROJECT,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True)
        print(f'pipeline PID {child.pid}, status {STATUS}', flush=True)
        return
    with (RUN/'pipeline.pid').open('x') as pid_file:
        pid_file.write(f'{os.getpid()}\n')
    (RUN/'logs').mkdir()
    def stop(signum, frame):
        raise InterruptedError(f'pipeline received signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    workflow = Pipeline()
    # Paths and callbacks in the historical supervisor are rebound only in this child process.
    with patch.multiple(legacy, RUN=RUN, LOGS=RUN/'logs', STATUS=STATUS, TRAIN=TRAIN):
        try:
            workflow.save()
            workflow.run()
        except BaseException as exc:
            workflow.record.update(status='failed',first_error=str(exc),traceback=traceback.format_exc())
            if workflow.active is not None:
                workflow.active.update(status='failed',ended_utc=legacy.now())
            raise
        finally:
            workflow.record.update(ended_utc=legacy.now(),stage_pid=None)
            workflow.save()


if __name__ == '__main__':
    main()

