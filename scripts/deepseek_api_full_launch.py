"""Detach the full subset experiment, observing the registered Beijing admission window."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.deepseek_api_client import CAP, Ledger, now, peak_end
from scripts.deepseek_api_subset import SMALL, WORK, read, stage_directory, update_status

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(stream.read()).hexdigest()

def launch():
    if Path.cwd()!=PROJECT or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('expected CPU only on 5880')
    directory=stage_directory('full')
    with (directory/'full.pid').open('x') as f: print(os.getpid(),file=f)
    try:
        if os.getpid()!=os.getsid(0):
            raise RuntimeError('full stage must be detached')
        smoke=read(WORK/'smoke/smoke_results.json')
        before=Ledger(WORK/'ledger.jsonl').summary('full')
        if not (smoke['status']=='complete' and smoke['n_calls']==before['calls']==36 and
                smoke['smoke_spent_yuan']==before['spent_yuan'] and before['reserved_yuan']=='0' and
                before['stage_totals']['calls']==0 and read(SMALL/'cpu_tests.json')['status']=='passed'):
            raise RuntimeError('S or ledger preflight failed')
        while True:
            dt=now();end=peak_end(dt)
            if end is None:break
            update_status(directory,dict(stage='F',status='paused_peak',pid=os.getpid(),until_beijing=end.isoformat(),paid_calls_full=0,
                                         spent_including_smoke_yuan=before['spent_yuan']))
            time.sleep(max(0,(end-dt).total_seconds()))
        worker=PROJECT/'scripts/deepseek_api_full.py'
        ready=read(WORK/'full_close/driver_ready.json')
        if ready['status']!='passed' or ready['worker_sha256']!=sha(worker):
            raise RuntimeError('worker identity or regression tests differ')
        update_status(directory,dict(stage='F',status='running',pid=os.getpid(),phase='admission',paid_calls_full=0))
        os.execv(sys.executable,[sys.executable,'-B','-u',str(worker)])
    except Exception as exc:
        update_status(directory,dict(stage='F',status='failed',type=type(exc).__name__))
        raise SystemExit(1) from None

if __name__=='__main__':launch()

