"""Preflight and run the registered DeepSeek subset smoke experiment on CPU."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.reader import parse_answer
from src.qa_metrics import score_answer

WORK = Path('/root/userfolder/ctx-compressor/deepseek_api')
SMALL = PROJECT / 'eval/deepseek_api'
MODES = ('reselect','append')
D = Decimal


def utc():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, default=str)+'\n').encode()


def write_new(path, obj):
    with path.open('xb') as f:
        f.write(encoded(obj))
        f.flush()
        os.fsync(f.fileno())


def read(path):
    return json.loads(path.read_bytes())


def update_status(directory, record):
    item={'utc':utc(),**record}
    with (directory/'events.jsonl').open('a') as f:
        print(json.dumps(item,ensure_ascii=False,default=str),file=f)
    pending=directory/'status.pending'
    write_new(pending,item)
    os.replace(pending,directory/'status.json')


def key(strategy,mode,tier,task,question):
    return '/'.join(map(str,(strategy,mode or 'shared',tier or 'shared',task,question)))


def write_row(stream, obj):
    stream.write(json.dumps(obj, ensure_ascii=False,allow_nan=False,default=str)+'\n')
    stream.flush()
    os.fsync(stream.fileno())


def validate_order(rows):
    seen=set()
    if len(rows)%4:
        raise ValueError('reader requests must contain complete trajectories')
    for start in range(0,len(rows),4):
        batch=rows[start:start+4]
        identity={(r['strategy'],r['mode'],r['tier'],r['task']) for r in batch}
        if len(identity)!=1 or seen.intersection(identity) or [r['question_index'] for r in batch]!=[0,1,2,3]:
            raise ValueError('duplicate, reordered, or incomplete reader trajectory')
        seen.update(identity)
        for row in batch:
            msgs=row['messages']
            if len(msgs)!=1 or msgs[0]['role']!='user' or hashlib.sha1(msgs[0]['content'].encode()).hexdigest()!=row['prompt_sha1']:
                raise ValueError('reader identity mismatch')


def prepare_both():
    from scripts import deepseek_api_inputs as a
    from scripts.deepseek_api_rollout import prepare_b
    data=a.prepare()
    all_rows=data.a_rows('full')
    sample=data.a_rows('smoke')
    validate_order(all_rows)
    validate_order(sample)
    if len(all_rows)!=1800 or len(sample)!=12 or list(data.s60[:30])!=list(data.s30):
        raise ValueError('unregistered held-out subset')
    b=prepare_b(data.corpus,data.tasks,data.s30)
    return data,b,all_rows,sample


def records(data,b,all_rows,sample):
    full_by=Counter(r['strategy'] for r in all_rows)
    full_tokens=Counter()
    for r in all_rows:
        full_tokens[r['strategy']]+=r['qwen_prompt_tokens']
    akeys=[(key(r['strategy'],r['mode'],r['tier'],r['task'],r['question_index']),r['prompt_sha1']) for r in all_rows]
    return {'status':'passed','split':'train-held-out','seed':20261127,'s60':list(data.s60),'s30':list(data.s30),
            'A':data.audit,'B':b.receipt,'A_full_n':len(all_rows),'A_smoke_n':len(sample),
            'A_full_counts':dict(full_by),'A_full_qwen_tokens':dict(full_tokens),
            'A_identities_sha256':hashlib.sha256(encoded(akeys)).hexdigest()}


def stage_directory(name):
    directory=WORK/name
    if directory.resolve()!=directory:
        raise ValueError('stage path redirected')
    directory.mkdir(parents=False,exist_ok=False)
    return directory


def prepare_stage():
    directory=stage_directory('prep')
    data,b,all_rows,sample=prepare_both()
    proof=records(data,b,all_rows,sample)
    if proof['B']['n_trajectories']!=60 or proof['B']['n_steps']<=0:
        raise ValueError('closed-loop preflight coverage differs')
    write_new(SMALL/'input_checks.json',proof)
    update_status(directory,{'status':'input_checks_passed','A_count':1800,'B_trajectories':60})
    return proof


def reader_kind(row):
    if row['strategy']=='no_compress':return 'A_no_compress'
    if row['strategy']=='no_context':return 'A_no_context'
    return 'A_compressed'


def estimates(call_rows,pre):
    """Scale the four smoke call types by preregistered full inputs and observed cache reuse."""
    measured=defaultdict(Counter)
    for row in call_rows:
        m=measured[row['kind']]
        use=row['usage']
        for k in ('prompt_cache_hit_tokens','prompt_cache_miss_tokens','completion_tokens'):
            m[k]+=use[k]
        m['n']+=1
        m['qwen_tokens']+=row.get('qwen_tokens',0)
        m['paid']+=D(str(row['cost_yuan']))
    counts=pre['A_full_counts']
    tokens=pre['A_full_qwen_tokens']
    names=[name for name in counts if name not in ('no_compress','no_context')]
    groups={'A_compressed':(sum(counts[n] for n in names),sum(tokens[n] for n in names)),
            'A_no_compress':(counts['no_compress'],tokens['no_compress']),
            'A_no_context':(counts['no_context'],tokens['no_context']),
            'B_compressor':(pre['B']['n_steps'],None),'B_reader':(240,None)}
    forecast={}
    for kind,(planned,qwen) in groups.items():
        m=measured['A_compressed' if kind=='A_no_context' else kind]
        if not m['n']:raise ValueError('missing measured call category for extrapolation')
        if qwen is None:
            pred=m['paid']*planned/m['n']
        else:
            if m['qwen_tokens']<=0:raise ValueError('missing historical tokenizer counts')
            hit,miss=m['prompt_cache_hit_tokens'],m['prompt_cache_miss_tokens']
            rate=(D('.02')*hit+miss if kind!='A_no_context' else D(hit+miss))
            pred=(rate*qwen/m['qwen_tokens']+D(4)*m['completion_tokens']*planned/m['n'])/10**6
        forecast[kind]={'full_calls':planned,'estimate_yuan':str(pred),'reference_smoke_calls':m['n']}
    return {'full_estimate_yuan':str(sum(D(v['estimate_yuan']) for v in forecast.values())),
            'by_kind':forecast,'measured':{k:{n:str(v) if isinstance(v,D) else v for n,v in m.items()} for k,m in measured.items()},
            'assumptions':'A scales observed DeepSeek input tokens and cache mix to Task 12 prompt lengths. Closed-book assumes no cache with compressed-reader output length. B scales measured cost per call to the exact step and reader counts. All costs off-peak.'}


def smoke():
    from scripts.deepseek_api_client import DeepSeekClient
    from scripts.deepseek_api_rollout import run_b_task,aggregate_b
    from scripts.deepseek_api_client import APIStop
    pre=read(SMALL/'input_checks.json')
    suite=read(SMALL/'cpu_tests.json')
    if suite['status']!='passed' or suite['previous_test_ids_matched']!=729 or suite['cpu_guard_events']:
        raise ValueError('CPU regression gate failed')
    if not (WORK/'prep/balance_end.json').is_file():
        raise ValueError('P end balance receipt missing')
    directory=stage_directory('smoke')
    def publish(row):
        update_status(directory,row)
    client=DeepSeekClient(work=WORK,stage='smoke',publish=publish)
    attempts=[]
    invalid=Counter()
    try:
        start=client.balance('start')
        data,b,all_rows,sample=prepare_both()
        if json.loads(encoded(records(data,b,all_rows,sample)))!=pre:
            raise ValueError('fresh P/S identity checks disagree')
        trajectories=[]
        with (directory/'predictions.jsonl').open('x') as predictions,(directory/'api_calls.jsonl').open('x') as raw:
            def query(kind,identifier,messages,max_tokens,qwen_tokens=0):
                r=client.complete(identifier,messages,max_tokens=max_tokens)
                event={**r,'call_id':identifier,'kind':kind,'qwen_tokens':qwen_tokens}
                write_row(raw,event)
                attempts.append(event)
                return r
            def answer(row,kind):
                identity=key(row['strategy'],row['mode'],row['tier'],row['task'],row['question_index'])
                response=query(kind,kind+'/'+identity,row['messages'],48,row.get('qwen_prompt_tokens',0))
                pred=parse_answer(response['content'])
                score=score_answer(pred,row['answer'],row['answer_aliases'])
                public={k:v for k,v in row.items() if k not in ('messages','token_ids','final_pids')}
                write_row(predictions,{**public,'request_id':response['id'],'prediction':pred,
                          'raw_output':response['content'],'finish_reason':response['finish_reason'],**score})
            for row in sample:
                answer(row,reader_kind(row))
            for mode in MODES:
                task=data.s30[0]
                def gen(messages,task_id,mode,step):
                    return query('B_compressor',key('B_compressor',mode,'mid',task_id,step),messages,256)
                trajectory=run_b_task(b,task,mode,gen)
                for step in trajectory['steps']:
                    invalid[step['invalid_reason'] or 'valid']+=1
                trajectories.append(trajectory)
                write_new(directory/('trajectory_'+mode+'.json'),trajectory)
                questions=data.reader_rows(task,mode,trajectory['final_pids'])
                validate_order(questions)
                for row in questions:answer(row,'B_reader')
        end=client.balance('end')
        total=client.summary()
        expected_steps=sum(r['n_steps'] for r in pre['B']['rows'] if r['task']==data.s30[0])
        if len(attempts)!=20+expected_steps or sum(invalid.values())!=expected_steps:
            raise ValueError('registered smoke count differs')
        for field in ('id','model','system_fingerprint','usage','finish_reason','content','started_utc','ended_utc','attempts'):
            if any(field not in r for r in attempts):raise ValueError('missing requested API field '+field)
        paid=sum(D(str(r['cost_yuan'])) for r in attempts)
        if paid!=D(str(total['stage_totals']['spent_yuan'])) or paid+D(str(total['reserved_yuan']))>8:
            raise ValueError('usage and ledger differ or cost exceeds cap')
        estimate=estimates(attempts,pre)
        delta=D(str(start['total_balance']))-D(str(end['total_balance']))
        result={'status':'complete','n_calls':len(attempts),'first_two_task_ids':list(data.s60[:2]),
                'A_n':len(sample),'B_reader_n':8,'B_steps':expected_steps,'B_invalid':expected_steps-invalid['valid'],
                'B_outcomes':dict(invalid),'B_metrics':aggregate_b(b,trajectories),
                'smoke_spent_yuan':str(paid),'balance_before':start,'balance_after':end,'account_drop_yuan':str(delta),
                'account_drop_minus_ledger_yuan':str(delta-paid),'ledger':total,
                'forecast':estimate,'total_estimate_including_smoke_yuan':str(D(estimate['full_estimate_yuan'])+paid),
                'model_names':dict(Counter(r['model'] for r in attempts)),
                'fingerprints':dict(Counter(str(r['system_fingerprint']) for r in attempts)),
                'finish_reasons':dict(Counter(r['finish_reason'] for r in attempts))}
        write_new(directory/'smoke_results.json',result)
        publish({'status':'complete','stage':'S','n_calls':len(attempts),'spent_yuan':str(paid)})
        return result
    except Exception as exc:
        if not (directory/'balance_end.json').exists():
            try:client.balance('end')
            except Exception:pass
        publish({'status':exc.status if isinstance(exc,APIStop) else 'failed','type':type(exc).__name__})
        raise


def main():
    from scripts.deepseek_api_client import DeepSeekClient
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','finish-p','smoke'))
    arg=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or Path.cwd()!=PROJECT or PROJECT!=Path('/root/workplace/ctx-compressor'):
        raise RuntimeError('task14 requires CPU on 5880')
    if arg.action=='prepare':
        proof=prepare_stage()
        print('P',proof['A_full_n'],proof['B']['n_trajectories'],proof['B']['n_steps'])
    elif arg.action=='finish-p':
        if read(SMALL/'cpu_tests.json')['status']!='passed' or read(SMALL/'input_checks.json')['status']!='passed':
            raise ValueError('P receipts missing')
        client=DeepSeekClient(work=WORK,stage='prep',publish=lambda row:update_status(WORK/'prep',row))
        paid=client.balance('end')
        print('P end CNY:',paid['total_balance'])
    else:
        r=smoke()
        print('S calls',r['n_calls'],'yuan',r['smoke_spent_yuan'])


if __name__=='__main__':
    main()

