#!/usr/bin/env python3
"""Prepare one fixed 480-sample comparison, run 1/4/8 trials, select by policy."""
import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
EXTERNAL=Path('/root/userfolder/ctx-compressor')
RUN=EXTERNAL/'runs/batch_bench_20261117'
OUT=PROJECT/'eval/sft_run'
sys.path.insert(0,str(PROJECT))
from scripts.bench_sft import select_balanced, sha256
from src.training_control import choose_batch, MEMORY_LIMIT_BYTES, SEED


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prepare():
    if RUN.exists():
        raise ValueError('batch comparison already prepared; no overwrite')
    sft=json.loads((PROJECT/'eval/sft_prep/sft_data_stats.json').read_text())
    heap=[]; serial=0
    def scan(split):
        nonlocal serial
        record=sft['files'][split]; digest=hashlib.sha256(); n=0
        with Path(record['path']).open('rb') as stream:
            for line in stream:
                digest.update(line); n+=1; serial+=1
                row=json.loads(line)
                entry=(row['meta']['total_tokens'],-serial,row)
                if len(heap)<8: heapq.heappush(heap,entry)
                elif entry[:2]>heap[0][:2]: heapq.heapreplace(heap,entry)
                yield row
        if n!=record['lines'] or digest.hexdigest()!=record['sha256']:
            raise ValueError('SFT source hash/line mismatch')
    samples,population=select_balanced(scan('train'),per_mode=240,seed=SEED)
    for _ in scan('heldout'): pass
    longest=[entry[2] for entry in sorted(heap,reverse=True)]
    RUN.mkdir(parents=True)
    data=RUN/'samples_480.jsonl'
    with data.open('x') as stream:
        for row in samples: stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    save(RUN/'longest_8.json',longest)
    manifest={'seed':SEED,'samples':{'path':str(data),'lines':480,'bytes':data.stat().st_size,'sha256':sha256(data)},
              'population_by_mode':population,'samples_per_mode':240,'total_tokens':sum(r['meta']['total_tokens'] for r in samples),
              'longest_8_meta':[row['meta'] for row in longest],
              'group_by_length':False,'trials':[{'batch_size':b,'gradient_accumulation_steps':16//b,'optimizer_steps':30} for b in (1,4,8)],
              'longest_probe_optimizer_steps':0,'memory_limit_bytes':MEMORY_LIMIT_BYTES}
    save(RUN/'input_manifest.json',manifest)
    print(json.dumps({'prepared':True,'samples':480,'longest_tokens':[r['meta']['total_tokens'] for r in longest]}),flush=True)


def run(resume_initialization=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('GPU1 only')
    if resume_initialization:
        evidence=json.loads((RUN/'initialization_attempts/batch_8_missing_length/worker_result.json').read_text())
        if (evidence['optimizer_steps']!=0 or evidence['consumed_samples']!=0 or evidence['oom']
                or evidence['nonfinite'] or 'Column length not in the dataset' not in evidence['error']):
            raise ValueError('only the preserved zero-step missing-length initialization may resume')
        if (RUN/'orchestrator_resumed.json').exists(): raise ValueError('initialization already resumed')
        save(RUN/'orchestrator_resumed.json',{'pid':os.getpid(),'session_id':os.getsid(0),
             'reason':'length column added; no optimizer step or data selection repeated'})
    else:
        if (RUN/'orchestrator_started.json').exists(): raise ValueError('no automatic repetition of batch trials')
        save(RUN/'orchestrator_started.json',{'pid':os.getpid(),'session_id':os.getsid(0)})
    results=[]
    for batch in (1,4,8):
        folder=RUN/f'batch_{batch}'; folder.mkdir(exist_ok=True)
        if resume_initialization and batch in (1,4):
            result=json.loads((folder/'worker_result.json').read_text())
            cfg=json.loads((folder/'worker_config.json').read_text())
            source=folder/'worker_source.py'
            data=json.loads((RUN/'input_manifest.json').read_text())['samples']
            if (result['status']!='ok' or result['optimizer_steps']!=30 or result['consumed_samples']!=480
                    or cfg['train_input']!=data or sha256(source)!=cfg['worker_source_sha256']):
                raise ValueError('completed trial provenance is not valid for reuse')
            result['exit_code']=0
            results.append(result)
            print(json.dumps({'reuse_completed_batch':batch,'optimizer_steps':30}),flush=True)
            continue
        with (folder/'console.log').open('xb') as log:
            worker=subprocess.Popen([str(EXTERNAL/'envs/train/bin/python'),'-u','-B',str(PROJECT/'scripts/train_sft_worker.py'),
                                     'benchmark','--batch-size',str(batch)],stdout=log,stderr=subprocess.STDOUT)
            save(folder/'launch.json',{'pid':worker.pid,'parent_pid':os.getpid()})
            code=worker.wait()
        result_path=folder/'worker_result.json'
        if not result_path.exists(): raise RuntimeError(f'batch {batch} failed before result; inspect console, no retry')
        result=json.loads(result_path.read_text()); result['exit_code']=code
        results.append(result)
        save(OUT/'batch_bench_progress.json',{'completed':results,'next_batch':{1:4,4:8,8:None}[batch]})
        if code: raise RuntimeError(f'batch {batch} failed; stopping without another trial or formal training')
        time.sleep(2)
    selected=choose_batch(results)
    subset=json.loads((OUT/'sample_manifest.json').read_text())
    tokens=subset['statistics']['train']['tokens']['total_tokens']['sum']*2
    report={'status':'ok','trials':results,'selected':{k:selected[k] for k in
            ('batch_size','gradient_accumulation_steps','tokens_per_second','selection_peak_bytes','seconds_per_sample')},
            'selection_rule':'highest actual unpadded token/s among completed 30-step trials whose resident-Adam longest batch does not OOM and peak<=40,000,000,000 bytes; tie smaller batch',
            'memory_limit_bytes':MEMORY_LIMIT_BYTES,'memory_definition':'max(torch reserved peak plus pre-existing background, sampled whole-card memory)',
            'group_by_length':False,'formal_estimate':{'sample_exposures':40000,'sequence_tokens':tokens,
              'train_loop_seconds':tokens/selected['tokens_per_second'],
              'train_loop_hours':tokens/selected['tokens_per_second']/3600,
              'overhead':'model startup, 21k CPU encodings, five eval-loss passes and checkpoint saves are additional; final report records actual wall time'},
            'task04_reference':{'path':str(PROJECT/'eval/sft_prep/bench_results.json'),
                                'note':'reference only; Task05 measured a fresh matched 30-step batch1 control'}}
    save(OUT/'batch_bench.json',report)
    print(json.dumps({'selected':report['selected'],'estimate':report['formal_estimate']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','run','resume-initialization'))
    args=p.parse_args()
    if PROJECT!=Path('/root/workplace/ctx-compressor'): p.error('5880 formal project only')
    if args.command=='prepare':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='': p.error('prepare CPU only')
        prepare()
    else: run(resume_initialization=args.command=='resume-initialization')


if __name__=='__main__': main()
