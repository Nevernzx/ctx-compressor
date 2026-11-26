"""Detached F0 → B → C → D → E → F → reader evaluation for GRPO round three."""
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
from scripts import grpo_r3_config as cfg
from scripts import run_formal_grpo_pipeline as controller
from scripts import run_grpo_r2_pipeline as r2
from scripts.rollout_model import file_record
from scripts.sample_sft_subset import digest, require, write_json_exclusive
from src.heldout_eval import select_checkpoint

STATUS = cfg.RUN / 'pipeline_status.json'


def stage_command(action, *options, env='grpo'):
    return r2.command('grpo_r3_stages.py', action, *options, environment=env)


def phase_a_evidence():
    from scripts.grpo_r3_phase_a import evidence
    return evidence()


def verify_states(start):
    info, path = r2.verify_states(start['adapter_sha256'])
    require(info['seed'] == cfg.SEED and info['base_model'] == str(cfg.BASE)
            and info['inference_adapter'] == str(cfg.ADAPTER) and info['task11_overlap'] == 0
            and info['task11_source'] == cfg.previous_file(cfg.PREVIOUS / 'states/state_checks.json'),
            'B did not use the registered adapter, population and seed')
    return info, path


def verify_dataset(probe):
    info, path = r2.verify_dataset(probe)
    gold = info['gold_varying']
    n = info['n']
    counts = info['per_layer_counts']
    available = info['per_layer_available']
    left = n
    from scripts.make_formal_grpo_dataset import quotas
    for layer in ('gold_varying','token_only_varying'):
        expected = quotas(available[layer],left)
        require(counts[layer] == expected, f'D layer quota mismatch: {layer}')
        left -= sum(expected.values())
    require(left == 0 and gold['count'] == min(n,sum(available['gold_varying'].values()))
            and gold['n'] == n and info['seed'] == cfg.DATA_SEED, 'D violated gold-first priority or seed')
    return info, path


def training_runtime():
    path = cfg.TRAIN / 'training_runtime.json'
    value = cfg.read(path)
    require(value['status']=='ok' and not value['oom'] and not value['nonfinite'],'E stopped')
    return value,path


def verify_training(n):
    adapters, path = r2.verify_training(n)
    report = cfg.read(path)
    args = report['resolved_config']
    require(args['seed'] == args['data_seed'] == cfg.SEED and args['model'] == str(cfg.MERGED)
            and report['reward_recheck']['advantage_mismatches'] == 0, 'E audit failed')
    return adapters,path


def reader_command(name, metric):
    directory = cfg.RAW / name
    result = cfg.OUT/name/f'{name}_results.json'
    receipt = cfg.read(cfg.OUT / name / 'run.json')
    require(receipt['artifacts'][result.name] == file_record(result)
            and metric['steps_artifact'] == file_record(directory/'steps.jsonl'), 'G lineage changed')
    # The reader worker uses its own offline runtime and performs the admission after CPU tokenization.
    output = cfg.ROOT / 'reader_heldout' / ('grpo_r3_' + name)
    args = r2.command('run_reader_heldout.py','extra','--rollout',str(directory),
             '--strategy','grpo_r3:'+name,'--output',str(output),'--small-output',str(cfg.OUT),
             '--metrics',str(result),'--steps-sha256',metric['steps_artifact']['sha256'],
             '--metrics-sha256',receipt['artifacts'][result.name]['sha256'], environment='reader')
    return ['bash','-c','source scripts/reader_environment.sh && exec "$@"','reader',*args]


def reader_result(name):
    from scripts.grpo_r3_reader_check import verify
    path = cfg.OUT / 'reader_f1_results.json'
    return verify(name, path),path


class Pipeline(r2.Pipeline):
    def __init__(self):
        super().__init__()

    def evaluate(self, name, start):
        label = 'F0' if name == 'merged_start' else 'F_'+name
        return self.step(label,stage_command('eval','--name',name,env='reader'),True,
                         lambda:r2.verify_eval(name,start))

    def run(self):
        self.record['interactive_preflight'] = phase_a_evidence()
        start = cfg.recorded_start()
        self.evaluate('merged_start',start)
        states = self.step('B',stage_command('states'),True,lambda:verify_states(start))
        probe = self.step('C',stage_command('filter'),True,lambda:r2.verify_filter(states))
        data = self.step('D',stage_command('dataset'),False,lambda:verify_dataset(probe))
        n = data['n']
        if n < 500:
            self.record.update(stage='D',status='stopped_insufficient_signal',n_informative=probe['n_informative'],
                               n_prompts=n,minimum_informative=500)
            return
        training_env = {'MERGED':str(cfg.MERGED),'DATA':str(cfg.DATA),'OUT':str(cfg.TRAIN),'MAX_STEPS':str(n//4)}
        self.step('E',r2.command('grpo_r3_training.py','train'),True,training_runtime,training_env)
        adapters = self.step('E_check',r2.command('grpo_r3_training.py','check'),False,lambda:verify_training(n))
        results = [self.evaluate(path.name,start) for path in adapters]
        selected = select_checkpoint(results)
        name = selected['strategy']
        choice = {'strategy':name,'path':str(cfg.TRAIN/name),'checkpoint_step':selected['checkpoint_step'],
                  'n_candidates':len(results),'evidence_loss':selected['combined']['evidence_loss'],
                  'final_recall':selected['combined']['final_recall'],
                  'adapter_sha256':digest(cfg.TRAIN/name/'adapter_model.safetensors'),
                  'rule':'Minimum pooled evidence loss, maximum pooled final recall, earliest optimizer step.'}
        write_json_exclusive(cfg.OUT/'selected_checkpoint.json',choice)
        self.record['selected_checkpoint'] = name
        self.record['reader_gpu_gate'] = 'The G reader driver gates physical GPU 1 after its CPU token preflight.'
        self.step('G',reader_command(name,selected),False,lambda:reader_result(name))
        self.record.update(stage='finished',status='evaluation_complete')


def launch(detach):
    require(Path(sys.prefix).resolve() == cfg.ROOT/'envs/grpo' and os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
            and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID' and os.environ.get('HF_HUB_OFFLINE') == '1',
            'use the round-three offline GRPO environment on GPU 1')
    r2.require_new_run()
    if detach:
        phase_a_evidence()
        with (cfg.RUN/'launcher.log').open('xb') as stream:
            child = subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],cwd=PROJECT,
                        stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                        start_new_session=True,close_fds=True)
        print(f'PID {child.pid}; status {STATUS}', flush=True)
        return
    with (cfg.RUN/'pipeline.pid').open('x') as pid:
        pid.write(f'{os.getpid()}\n')
    (cfg.RUN/'logs').mkdir()
    def stop(signum, frame):
        raise InterruptedError(f'pipeline received signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    workflow = Pipeline()
    try:
        workflow.save()
        workflow.run()
    except BaseException as exc:
        workflow.record.update(status='failed',first_error=str(exc),traceback=traceback.format_exc())
        if workflow.active is not None:
            workflow.active.update(status='failed',ended_utc=controller.now())
        raise
    finally:
        workflow.record.update(ended_utc=controller.now(),stage_pid=None)
        workflow.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach',action='store_true')
    args = parser.parse_args()
    # Both inherited supervisors bind their monitoring and checks to module constants.
    # Keep these bindings process-local for the lifetime of this independent controller.
    with patch.multiple(r2,RUN=cfg.RUN,OUT=cfg.OUT,START=cfg.START,STATUS=STATUS,STATES=cfg.STATES,
                        GROUPS=cfg.GROUPS,DATA=cfg.DATA,TRAIN=cfg.TRAIN,RAW=cfg.RAW,SEED=cfg.SEED), \
         patch.multiple(controller,RUN=cfg.RUN,LOGS=cfg.RUN/'logs',STATUS=STATUS,TRAIN=cfg.TRAIN):
        launch(args.detach)


if __name__ == '__main__':
    main()

