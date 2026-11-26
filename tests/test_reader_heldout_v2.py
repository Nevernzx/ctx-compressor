"""CPU checks for the signed joint reader batch and the strict cascade setting."""
from pathlib import Path

import pytest

from scripts import reader_heldout_v2 as job


def records(task, order):
    return [dict(strategy=s,mode=m,tier=t,task=task,question_index=q,
                 token_ids=[task,q],prompt_tokens=2) for s,m,t,q in order]


def fixture():
    r3_order = [(job.diag.R3,m,t,q) for m,t in job.inputs.CELLS for q in range(4)]
    base, r3 = [], []
    for t in (4,11):
        base.extend(records(t,job.BASE_ORDER))
        r3.extend(records(t,r3_order))
    return base,r3


def test_all_224_records_pass_through_in_original_order():
    base,r3=fixture()
    batches=list(job.main_batches(iter(base),iter(r3)))
    assert [task for task,_ in batches]==[4,11]
    for i,(_,batch) in enumerate(batches):
        assert len(batch)==224
        assert all(a is b for a,b in zip(batch[:200],base[i*200:(i+1)*200],strict=True))
        assert all(a is b for a,b in zip(batch[200:],r3[i*24:(i+1)*24],strict=True))
        assert batch[0]['strategy']=='no_compress' and batch[196]['strategy']=='no_context' and batch[200]['strategy']==job.diag.R3


@pytest.mark.parametrize('case',['short_prefix','reordered_prefix','different_task','missing_r3','wrong_length','too_long'])
def test_damaged_or_reordered_batches_stop(case):
    base,r3=fixture()
    if case=='short_prefix':base.pop()
    if case=='reordered_prefix':base[0],base[3]=base[3],base[0]
    if case=='different_task':
        for row in r3[24:]:row['task']=14
    if case=='missing_r3':r3=r3[:24]
    if case=='wrong_length':base[0]['prompt_tokens']=5
    if case=='too_long':base[0].update(token_ids=[0]*40913,prompt_tokens=40913)
    with pytest.raises(ValueError):list(job.main_batches(base,r3))


@pytest.mark.parametrize('changed',['historical','r3','m12','m13'])
def test_both_input_signatures_are_checked_before_reading(changed,tmp_path,monkeypatch):
    project=tmp_path/'project';project.mkdir()
    old=tmp_path/'task12';r3=tmp_path/'task13'
    for directory in (old,r3):(directory/'input_preflight').mkdir(parents=True)
    config=project/'historical.json';config.write_text('{}')
    m12=project/'M12';m13=project/'M13'
    req12=old/'input_preflight/requests.jsonl';req13=r3/'input_preflight/requests.jsonl'
    smoke=old/'input_preflight/smoke.jsonl'
    for m,files in [(m12,[req12,smoke,req12.with_name('preflight.json')]),(m13,[req13,req13.with_name('preflight.json')])]:
        for f in files:f.write_text(str(f.name)+'\n')
        m.write_text(''.join(f'{job.inputs.checksum(f)}  {f}\n' for f in files))
    (project/'SHA256SUMS').write_text(''.join(f'{job.inputs.checksum(f)}  {f.name}\n' for f in (config,m12,m13)))
    for name,p in dict(PROJECT=project,R12=req12,R13=req13,SMOKE=smoke,M12=m12,M13=m13,HIST_CONFIG=config).items():
        monkeypatch.setattr(job,name,p)
    monkeypatch.setattr(job.reader,'STANDARD',old);monkeypatch.setattr(job.diag,'TRY2',r3)
    monkeypatch.setattr(job.diag,'PROJECT',project)
    proof=job.verify_sources()
    assert proof[str(req12)]==job.inputs.checksum(req12) and proof[str(req13)]==job.inputs.checksum(req13)
    victim={'historical':req12,'r3':req13,'m12':m12,'m13':m13}[changed]
    victim.write_text(victim.read_text()+'corrupt\n')
    with pytest.raises(ValueError,match='SHA256 mismatch'):job.verify_sources()


@pytest.mark.parametrize('value,source',[(False,'llm_engine.vllm_config'),(None,'未取得'),(True,'engine startup config line'),(1,'llm_engine.vllm_config')])
def test_effective_flag_requires_engine_true(value,source):
    with pytest.raises(ValueError):job.require_effective_cascade_off(value,source)
    job.require_effective_cascade_off(True,'llm_engine.vllm_config.model_config.disable_cascade_attn')


def test_existing_archive_rejected_before_launch_or_wait(tmp_path,monkeypatch):
    target=tmp_path/'predictions.jsonl';target.write_bytes(b'saved')
    monkeypatch.setattr(job,'WORK',tmp_path)
    with pytest.raises(FileExistsError):job.detach()
    assert target.read_bytes()==b'saved' and list(tmp_path.iterdir())==[target]


def test_short_runtime_ipc_path_including_protocol_bytes(tmp_path,monkeypatch):
    monkeypatch.delenv('VLLM_RPC_BASE_PATH',raising=False)
    proof=job.deployable_ipc()
    assert proof['path_bytes']==75 and proof['uri_bytes']==81 and proof['limit_exclusive']==100
    monkeypatch.setattr(job,'SHORT',Path('/tmp')/('文'*25))
    with pytest.raises(ValueError,match='IPC'):job.deployable_ipc()

