"""Offline tests for the full-stage ownership, partitioning, and concurrent outputs."""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha1
import json
import os
from pathlib import Path
import threading

import pytest
from scripts import deepseek_api_full as full


def requests(task):
    user = 'Which city?'
    return [{'strategy':'oracle','mode':'reselect','tier':'mid','task':task,'question_index':q,
             'prompt_sha1':sha1(user.encode()).hexdigest(),'messages':[{'role':'user','content':user}],
             'answer':'Paris','answer_aliases':[]} for q in range(4)]


def launcher(directory):
    directory.mkdir()
    (directory/'full.pid').write_text(str(os.getpid())+'\n')
    full.subset.update_status(directory, {'stage':'F','status':'running','pid':os.getpid()})


def test_launcher_ownership_and_prior_outputs(tmp_path):
    directory=tmp_path/'full'
    launcher(directory)
    full.claim_full(directory)
    preserved=directory/'predictions.jsonl'
    preserved.write_bytes(b'old\n')
    with pytest.raises(FileExistsError):
        full.claim_full(directory)
    assert preserved.read_bytes()==b'old\n'


def test_reject_wrong_detached_pid(tmp_path):
    directory=tmp_path/'full'
    launcher(directory)
    (directory/'full.pid').write_text('0\n')
    with pytest.raises(ValueError,match='PID'):
        full.claim_full(directory)


def test_call_plan_distinguishes_smoke_independent_a_b_and_modes():
    a=requests(2)
    traj={'rows':[{'task':2,'mode':m,'n_steps':2} for m in ('append','reselect')]}
    keys=full.expected_calls(a,traj)
    assert len(keys)==4+4+8 and 'B_reader/deepseek-flash/append/mid/2/3' in keys
    assert 'B_compressor/reselect/mid/2/0' in keys and 'A_compressed/oracle/reselect/mid/2/0' in keys
    with pytest.raises(ValueError,match='duplicate'):
        full.expected_calls(a,{'rows':2*traj['rows']})
    with pytest.raises(ValueError):
        full.expected_calls(a[::-1],traj)


def test_per_trajectory_order_and_serialized_artifacts(tmp_path):
    # All slots are occupied in the first wave. The provider never sees more than eight.
    barrier=threading.Barrier(8)
    lock=threading.Lock()
    active=0
    high_water=0
    order=defaultdict(list)
    class Client:
        @staticmethod
        def clock():return datetime(2026,11,28,3,tzinfo=timezone.utc)
        def complete(self,ident,messages,max_tokens):
            nonlocal active,high_water
            _,_,_,_,task,q=ident.split('/')
            with lock:
                active+=1
                high_water=max(high_water,active)
                order[task].append(int(q))
            if int(task)<8 and q=='0':
                barrier.wait(timeout=10)
            with lock:
                active-=1
            return {'call_id':ident,'id':ident,'started_utc':self.clock().isoformat(),
                    'finish_reason':'stop','content':'Paris','cost_yuan':'0.00001'}
    a=tmp_path/'api_calls.jsonl'
    p=tmp_path/'predictions.jsonl'
    with a.open('x') as raw,p.open('x') as pred:
        stage=full.FullStage(tmp_path,Client(),None,None,raw,pred)
        full.run_parallel([('A',requests(t)) for t in range(16)],stage.job,stage.stop)
    assert high_water==8
    assert all(r==list(range(4)) for r in order.values())
    assert len(order)==16 and len(stage.predicted)==64
    api=[json.loads(s) for s in a.read_text().splitlines()]
    preds=[json.loads(s) for s in p.read_text().splitlines()]
    assert len(api)==len(preds)==64 and {r['id'] for r in api}=={r['request_id'] for r in preds}
    assert all(r['f1']==r['em']==1 for r in preds)
    assert all(s in stage.by_kind for s in ('A_compressed',))
    assert stage.paid==full.Decimal('0.00064')
    assert json.loads((tmp_path/'status.json').read_bytes())['paid_calls_full']==50


def test_stopped_worker_blocks_pending_and_notice_preserves_halt(tmp_path):
    stop=threading.Event()
    visited=[]
    def bad(i):
        visited.append(i)
        raise full.APIStop('stopped_cost_cap')
    with pytest.raises(full.APIStop,match='stopped_cost_cap'):
        full.run_parallel(list(range(50)),bad,stop)
    assert stop.is_set() and len(visited)<50
    stage=full.FullStage(tmp_path,None,None,None,None,None)
    stage.notify({'status':'stopped_cost_cap'})
    stage.notify({'status':'paused_peak'})
    assert json.loads((tmp_path/'status.json').read_bytes())['status']=='stopped_cost_cap'
    with pytest.raises(full.APIStop,match='stopped_cost_cap'):
        stage.query('reader','r',[],48)

