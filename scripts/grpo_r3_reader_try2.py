"""Run only the authorized second G attempt with Task 12's frozen reader."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import run_reader_heldout as reader
from scripts.grpo_r3_reader_check import verify
from scripts.sample_sft_subset import digest, require, write_json_exclusive
from src import reader_heldout_inputs as inputs

NAME='checkpoint-250'
ROOT=reader.ROOT
SHORT=ROOT/'rt/g13'
FAILED=reader.STANDARD/('grpo_r3_'+NAME)
OUT=FAILED.with_name(FAILED.name+'_try2')
SMALL=PROJECT/'eval/grpo_r3'
RAW=ROOT/'eval_rollouts_grpo_r3'/NAME
METRIC=SMALL/NAME/(NAME+'_results.json')
RUN=ROOT/'grpo_r3'


def identity():
    choice=reader.read(SMALL/'selected_checkpoint.json')
    require(choice['strategy']==NAME and choice['checkpoint_step']==250,'selected checkpoint changed')
    receipt=reader.read(SMALL/NAME/'run.json')
    require(receipt['status']=='complete' and receipt['artifacts'][METRIC.name]['sha256']==digest(METRIC),
            'F metric checksum changed')
    metrics=reader.read(METRIC)
    steps=metrics['steps_artifact']
    require(steps['path']==str(RAW/'steps.jsonl') and steps['sha256']==digest(RAW/'steps.jsonl'),'F trace changed')
    return argparse.Namespace(action='extra',strategy='grpo_r3:'+NAME,rollout=RAW,metrics=METRIC,
           steps_sha256=steps['sha256'],metrics_sha256=receipt['artifacts'][METRIC.name]['sha256'],
           output=OUT,small_output=SMALL)


def input_check():
    args=identity()
    source=inputs.Source(args.strategy,args.metrics,args.metrics_sha256,RAW/'steps.jsonl',args.steps_sha256,
                         'Task 13 F stage receipt')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='' and OUT.resolve()==OUT,'use CPU and the dedicated try2 output')
    report=inputs.prepare(OUT/'input_preflight',source)
    first=reader.verify_inputs(FAILED/'input_preflight',False)
    require(report['requests']['sha256']==first['requests']['sha256'] and report['sources']==first['sources']
            and report['costs']==first['costs'] and report['requests']['n']==5976,
            'try2 input spool differs from first attempt')


def run():
    args=identity()
    require(OUT==OUT.resolve() and SHORT==SHORT.resolve() and args.rollout==args.rollout.resolve()
            and METRIC.parent==SMALL/NAME and os.environ['READER_HELDOUT_BOOTSTRAP']==str(OUT),
            'unexpected G paths or launch environment')
    reader.reader_environment(OUT,[])
    keys=('TMPDIR','TMP','TEMP','RAY_TMPDIR')
    require(all(os.environ[k]==str(SHORT) for k in keys) and Path(tempfile.gettempdir())==SHORT,
            'short temporary directory is not active')
    require(os.environ.get('VLLM_RPC_BASE_PATH',str(SHORT))==str(SHORT),'unexpected IPC base')
    sample=str(SHORT/'00000000-0000-0000-0000-000000000000')
    length=len(os.fsencode(sample))
    print(f'G IPC path={sample}, bytes={length}, URI bytes={length+6}, required <100',flush=True)
    require(length+6<100,'IPC URI too long')
    reader.require_new([OUT/'reader.pid',OUT/'status.json',OUT/'events.jsonl',OUT/'predictions.jsonl',
                       SMALL/'reader_run_config.json', SMALL/'reader_f1_results.json', SMALL/'reader_f1_report.md'])
    with (OUT/'reader.pid').open('x') as f:print(os.getpid(),file=f)
    reader.publish(OUT,{'stage':'preflight','status':'running'})
    write_json_exclusive(OUT/'ipc_check.json',{'sample':sample,'path_bytes':length,'uri_bytes':length+6,
                         'environment':{k:os.environ[k] for k in keys},'limit_exclusive':100})
    # Only the old CLI path check assumes the first attempt name. The second
    # attempt calls that same input builder with the exact F-signed source.
    def dispatch(argv,**kwargs):
        if argv[1:4]==['-B','-m','src.reader_heldout_inputs']:
            require(Path(argv[argv.index('--out-dir')+1])==OUT/'input_preflight','unexpected input target')
            argv=[str(reader.PYTHON),'-B','-u',str(Path(__file__).resolve()),'input-check']
        return subprocess.run(argv,**kwargs)
    reader.subprocess=SimpleNamespace(run=dispatch)
    try:
        reader.work(args)
        result=verify(NAME,SMALL/'reader_f1_results.json')
        require(result['n_predictions']==5976,'incomplete G results')
        reader.publish(OUT,{'stage':'finished','status':'complete'})
    except BaseException as exc:
        reader.publish(OUT,{'status':'failed','error':repr(exc)})
        raise


if __name__=='__main__':
    if sys.argv[1:]==['input-check']:input_check()
    elif not sys.argv[1:]:run()
    else:raise SystemExit('unexpected arguments')

