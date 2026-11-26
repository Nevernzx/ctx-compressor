#!/usr/bin/env python3
"""Run the Task 03 reader on signed, CPU-preflighted held-out inputs."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import groupby
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import run_reader as original
from scripts.run_formal_grpo_pipeline import gpu
from scripts.sft_v3_wait_gpu import wait_for_gpu
from scripts.sample_sft_subset import digest, require, write_json_exclusive

ROOT = Path('/root/userfolder/ctx-compressor')
STANDARD = ROOT / 'reader_heldout'
SMALL = PROJECT / 'eval/reader_heldout'
FIELDS = ('strategy', 'mode', 'tier', 'trajectory_lost', 'answer', 'answer_aliases')
PYTHON = ROOT / 'envs/reader/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def require_new(paths):
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f'refusing to overwrite {path}')


def check_locations(args):
    if args.action == 'extra':
        name = args.rollout.name
        require(name.startswith('checkpoint-') and name.removeprefix('checkpoint-').isdigit()
                and args.strategy == 'grpo_r3:' + name, 'invalid r3 strategy')
        require(args.rollout == ROOT/'eval_rollouts_grpo_r3'/name
                and args.output == STANDARD/('grpo_r3_'+name)
                and args.small_output == PROJECT/'eval/grpo_r3', 'unexpected G output or rollout')
    for path in [args.output,args.small_output]:
        require(path.resolve() == path, f'redirected output {path}')


def reader_environment(output, argv):
    if os.environ.get('READER_HELDOUT_BOOTSTRAP') != str(output):
        env = dict(os.environ, READER_HELDOUT_BOOTSTRAP=str(output))
        script = 'source scripts/reader_environment.sh && exec "$READER_ENV/bin/python" -B -u scripts/run_reader_heldout.py "$@"'
        os.execve('/bin/bash',['bash','-c',script,'reader-heldout',*argv],env)
    require(Path(sys.prefix).resolve() == ROOT/'envs/reader'
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '1', 'use the task03 reader environment')
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                      PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', VLLM_NO_USAGE_STATS='1')
    cache = str(ROOT/'cache')
    for key,value in list(os.environ.items()):
        if value == cache or value.startswith(cache+'/'):
            dest = output/'runtime_cache'/Path(value).relative_to(cache)
            dest.mkdir(parents=True,exist_ok=True)
            os.environ[key] = str(dest)
    os.environ['PYTHONPATH']=str(PROJECT)+os.pathsep+os.environ.get('PYTHONPATH','')


def generate(llm,sampling,rows,monitor):
    result = original.generate_batch(llm,sampling,rows,monitor)
    for row, source in zip(result, rows, strict=True):
        row.update({k:source[k] for k in FIELDS if k in source})
    return result


def lines(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


class Monitor:
    def __init__(self, output):
        self.output = output
        self.stop_event = threading.Event()
        self.failure = None
        self.peak = 0
        self.samples = 0

    def collect(self):
        try:
            with (self.output/'resources.jsonl').open('x',buffering=1) as f:
                while True:
                    g = gpu()
                    self.peak = max(self.peak,g['memory_used_mib'])
                    self.samples += 1
                    print(json.dumps(dict(utc=now(),gpu=g,root=original.root_df())),file=f)
                    if self.stop_event.wait(2):
                        break
        except Exception as error:
            self.failure = error

    def start(self):
        self.thread = threading.Thread(target=self.collect,daemon=True)
        self.thread.start()

    def check(self):
        if self.failure is not None:
            raise RuntimeError('GPU 1 / disk monitoring failed') from self.failure

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.check()


def publish(directory, data):
    record = {'utc':now(), 'pid':os.getpid(), **data}
    with (directory/'events.jsonl').open('a') as log:
        print(json.dumps(record),file=log)
    draft = directory/'status.next'
    write_json_exclusive(draft,record)
    os.replace(draft,directory/'status.json')


def receipts_write(directory, stage, **fields):
    write_json_exclusive(directory/('receipt_'+stage+'.json'),dict(stage=stage,utc=now(),**fields))


def verify_inputs(inp,standard):
    info=read(inp/'preflight.json')
    require(info['status']=='passed','input preflight did not pass')
    for key in (['requests','smoke'] if standard else ['requests']):
        path=inp/(key+'.jsonl')
        require(info[key]['sha256']==digest(path) and info[key]['n']>0,'input spool changed: '+key)
    require(info['requests']['n'] in ((49800,54780) if standard else (5976,)),
            'unexpected held-out request count')
    if standard:
        require(info['smoke']['n']==40,'wrong historical smoke size')
    return info


def work(args):
    standard=args.action=='run'
    directory=args.output
    inp=directory/'input_preflight'
    small=args.small_output
    name='run_config.json' if standard else 'reader_run_config.json'
    archive=directory/'predictions.jsonl'
    require_new([archive,small/name,directory/'resources.jsonl'])
    if not standard:
        cmd=[str(PYTHON),'-B','-m','src.reader_heldout_inputs','--out-dir',str(inp),'--strategy',args.strategy,
             '--rollout',str(args.rollout),'--metrics',str(args.metrics),'--steps-sha256',args.steps_sha256,
             '--metrics-sha256',args.metrics_sha256]
        subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    info=verify_inputs(inp,standard)
    if standard:
        test=read(small/'cpu_tests.json')
        require(test['status']=='passed' and test['tests']==test['passed']>=705,'incomplete CPU suite')
    receipts_write(directory,'inputs',sha256=digest(inp/'preflight.json'),n=info['requests']['n'])
    admitted=wait_for_gpu('R' if standard else 'G',lambda row:publish(directory,row))
    receipts_write(directory,'gate',gpu=admitted)
    historic=read(PROJECT/'eval/reader/run_config.json')
    params=historic['parameters']
    gen=historic['generation']
    require(params['max_model_len']==40960 and gen['max_tokens']==48 and not gen['enable_thinking']
            and params['seed']==gen['seed']==20261117 and gen['temperature']==0,'reader config changed')
    cfg=dict(parameters=params,generation=gen,source_sha256=digest(PROJECT/'eval/reader/run_config.json'),
             before=original.root_df(),preflight_sha256=digest(inp/'preflight.json'),
             preflight_max_prompt_tokens=info['requests']['max_prompt_tokens'],
             package_versions={k:version(k) for k in historic['package_versions']},
             started_utc=now(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    require(cfg['package_versions']==historic['package_versions'],'reader environment versions changed')
    monitor=Monitor(directory)
    monitor.start()
    llm=None
    n=0
    began=time.monotonic()
    try:
        from vllm import LLM, SamplingParams
        require(gpu()['memory_used_mib']<2048,'GPU 1 occupied after admission')
        publish(directory,dict(status='running',stage='engine'))
        llm=LLM(**params)
        sampling=SamplingParams(n=1,temperature=gen['temperature'],max_tokens=gen['max_tokens'],seed=gen['seed'])
        cfg['sampling_params_repr']=repr(sampling)
        if standard:
            sources=list(lines(inp/'smoke.jsonl'))
            smoke=generate(llm,sampling,sources,monitor)
            mismatches=sum(r['raw_output']!=s['expected_raw_output'] for r,s in zip(smoke,sources,strict=True))
            with (directory/'smoke_predictions.jsonl').open('x') as f:
                for r,s in zip(smoke,sources,strict=True):
                    print(original.compact(dict(r,expected_raw_output=s['expected_raw_output'])),file=f)
            receipts_write(directory,'smoke',n=len(smoke),raw_output_mismatches=mismatches,
                           predictions_sha256=digest(directory/'smoke_predictions.jsonl'))
            cfg['smoke']={'n':len(smoke),'raw_output_mismatches':mismatches}
        seen_tasks=set()
        with archive.open('x') as f:
            for task, requests in groupby(lines(inp/'requests.jsonl'),key=lambda row:row['task']):
                batch=list(requests)
                require(task not in seen_tasks and len(batch)%4==0 and
                        all(row['question_index']==i%4 and row['prompt_tokens']==len(row['token_ids'])
                            and row['prompt_tokens']+48<=40960 for i,row in enumerate(batch)),
                        'unexpected order or input length')
                seen_tasks.add(task)
                for r in generate(llm,sampling,batch,monitor):
                    print(original.compact(r),file=f)
                n+=len(batch)
                f.flush()
                if len(seen_tasks)%5==0 or len(seen_tasks)==1:
                    publish(directory,dict(status='running',stage='inference',tasks=len(seen_tasks),predictions=n))
        require(n==info['requests']['n'] and len(seen_tasks)==249,'incomplete reader run')
    finally:
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
        finally:
            monitor.stop()
    cfg.update(status='complete',predictions=n,engine_shutdown=True,elapsed_seconds=time.monotonic()-began,
               peak_gpu1_mib=monitor.peak,n_resource_samples=monitor.samples,df_after_reader=original.root_df(),
               predictions_sha256=digest(archive))
    if standard:
        cfg['df_opening']=read(directory/'before.json')
        baseline=cfg['df_opening']['first_observed_root_used_bytes']
        cfg['peak_observed_root_delta_bytes']=max(x['used_bytes'] for x in lines(directory/'disk_samples.jsonl'))-baseline
    write_json_exclusive(small/name,cfg)
    receipts_write(directory,'inference',n=n,run_config_sha256=digest(small/name),predictions_sha256=cfg['predictions_sha256'])
    if not standard:
        cmd=[str(PYTHON),'-B','scripts/reader_heldout_report.py','--predictions',str(archive),'--preflight',str(inp/'preflight.json'),
             '--output-dir',str(small),'--baseline-predictions',str(STANDARD/'predictions.jsonl'),'--extra-strategy',args.strategy]
        subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        receipts_write(directory,'report',results_sha256=digest(small/'reader_f1_results.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('run').set_defaults(output=STANDARD,small_output=SMALL)
    extra=sub.add_parser('extra')
    for key in ('rollout','metrics','output','small-output'):
        extra.add_argument('--'+key,type=Path,required=True)
    for key in ('strategy','steps-sha256','metrics-sha256'):
        extra.add_argument('--'+key,required=True)
    args=parser.parse_args()
    check_locations(args)
    reader_environment(args.output,sys.argv[1:])
    directory=args.output
    require_new([directory/'status.json',directory/'events.jsonl',directory/'reader.pid',directory/'status.next'])
    with (directory/'reader.pid').open('x') as f:
        print(os.getpid(),file=f)
    try:
        publish(directory,dict(stage='preflight',status='running'))
        work(args)
    except BaseException as exc:
        publish(directory,dict(status='failed',error=repr(exc)))
        raise
    publish(directory,dict(status='complete',stage='finished'))


if __name__=='__main__':
    main()
