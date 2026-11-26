import json
from pathlib import Path

import pytest
from scripts import grpo_r2_start as origin
from scripts import run_grpo_r2_pipeline as driver
from scripts import sft_v3_wait_gpu as gate


@pytest.mark.parametrize(('loss','expected'),[(0,'v3'),(956,'v3'),(957,'v2'),(1494,'v2')])
def test_start_cutoff(loss,expected):
    assert origin.start_version(loss,1494)==expected


@pytest.mark.parametrize(('loss','n'),[(-1,1494),(1495,1494),(956,249)])
def test_start_rejects_bad_denominators(loss,n):
    with pytest.raises(ValueError):
        origin.start_version(loss,n)


def test_fixture_records_mismatch_without_rejecting(monkeypatch,tmp_path):
    fixture = tmp_path/'fixture.jsonl'
    rows = [dict(task=i, mode='append', budget_tier='tight', prompt_sha256=str(i), split='train', step=0)
            for i in range(30)]
    fixture.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    source=origin.file_record(fixture)
    monkeypatch.setattr(origin, 'FIXTURE',fixture)
    monkeypatch.setattr(origin, 'read',lambda p:{'fixture_n':30,'fixture_sha256':source['sha256']})
    generated=[dict(r,raw_output='a',output_token_ids=[2]) for r in rows]
    base=dict(n=30,rows=generated,prompt_file=source,source_adapter_sha256='abc')
    other={**base,'rows':[{**generated[0],'raw_output':'b','output_token_ids':[3]},*generated[1:]]}
    assert origin.compare_fixture(base,other,'abc') == {'n_reference':30,'text_mismatches':1,'token_id_mismatches':1}
    other['rows']=list(reversed(other['rows']))
    with pytest.raises(ValueError,match='fixture'):
        origin.compare_fixture(base,other,'abc')


def aggregate(name, step, loss):
    from src.heldout_eval import BUDGETS, MODES
    return {'strategy':name,'checkpoint_step':step,'settings':{'n_tasks':249},
            'configs':{f'{m}|{t}':{'metrics':{'n_tasks':249}} for m in MODES for t in BUDGETS},
            'combined':{'n_tasks':1494, 'evidence_loss':{'count':loss},'final_recall':{'mean':.8}}}


@pytest.mark.parametrize('n',[0,499,500,2000])
def test_one_pass_stage_sequence_and_signal_boundary(n,monkeypatch,tmp_path):
    out=tmp_path/'results'
    out.mkdir()
    monkeypatch.setattr(driver,'OUT',out)
    monkeypatch.setattr(driver,'TRAIN',tmp_path/'train')
    monkeypatch.setattr(driver,'phase_a_evidence',lambda:{})
    monkeypatch.setattr(driver,'read',lambda _: {'adapter_sha256':'sft'})
    seen=[]
    p=driver.Pipeline()
    def evaluate(name,start):
        seen.append(name)
        return aggregate(name,None if name=='merged_start' else int(name.split('-')[1]),0 if name=='merged_start' else 3)
    def step(label,argv,use_gpu,verify,env=None):
        seen.append(label)
        if label=='B':return {'n_states':22001}
        if label=='C':return {'n_informative':n}
        if label=='D':return {'n':n}
        if label=='E':
            assert (env['MAX_STEPS'],env['LR'],env['SCALE_REWARDS'],env['GRPO_SEED'])==(
                str(n//4),'1.5e-5','none','20261123')
        if label=='E_check':
            steps=list(range(125,n//4+1,125))
            paths=[driver.TRAIN/f'checkpoint-{s}' for s in steps]
            for path in paths:
                path.mkdir(parents=True)
                (path/'adapter_model.safetensors').write_bytes(b'weight')
            return paths
    monkeypatch.setattr(p,'evaluate',evaluate)
    monkeypatch.setattr(p,'step',step)
    p.run()
    if n<500:
        assert seen==['merged_start','B','C','D']
        assert p.record['status']=='stopped_insufficient_signal'
        assert not (out/'selected_checkpoint.json').exists()
    else:
        assert seen==['merged_start','B','C','D','E','E_check',*[f'checkpoint-{s}' for s in range(125,n//4+1,125)]]
        assert json.loads((out/'selected_checkpoint.json').read_bytes())['strategy']=='checkpoint-125'
        assert p.record['status']=='evaluation_complete'


def test_driver_admission_and_monitor_stay_in_r2(monkeypatch,tmp_path):
    import sys
    run=tmp_path/'grpo_r2'
    (run/'logs').mkdir(parents=True)
    for field,value in {'RUN':run,'LOGS':run/'logs','STATUS':run/'pipeline_status.json','TRAIN':run/'train'}.items():
        monkeypatch.setattr(driver.legacy,field,value)
    from scripts import run_sft_v3_pipeline as previous
    monkeypatch.setattr(previous,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'RUN',run)
    samples=iter([4096,375,375,375])
    def gpu():
        return {'index':1,'uuid':driver.legacy.GPU_UUID,'memory_used_mib':next(samples,375),'utilization_pct':0}
    monkeypatch.setattr(driver.legacy,'gpu',gpu)
    real=gate.wait_for_gpu
    waits=[]
    def admission(stage,publish,reader):
        def observe(event):
            publish(event)
            state=json.loads((run/'pipeline_status.json').read_bytes())
            assert state['status']=='waiting_gpu' and state['stage_pid'] is None
            waits.append(state['last_monitor']['consecutive_free'])
        return real(stage,observe,reader=reader,clock=lambda:0,sleep=lambda _:None)
    monkeypatch.setattr(gate,'wait_for_gpu',admission)
    receipt=run/'sample.json'
    receipt.write_text('{}')
    p=driver.Pipeline()
    assert p.step('F0',[sys.executable,'-B','-c','pass'],True,lambda:('ok',receipt))=='ok'
    assert waits==[0,1,2,3] and p.active['receipt']==driver.file_record(receipt)
    assert (run/'resource_monitor.jsonl').is_file() and (run/'pipeline_monitor.jsonl').is_file()

