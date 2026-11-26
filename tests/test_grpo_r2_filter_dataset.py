from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
from tokenizers import Tokenizer, decoders, models

from scripts import probe_grpo_r2 as filt
from scripts import make_grpo_r2_dataset as data

GOLD = '{"add":[1],"drop":[]}'
EXTRA = '{"add":[1,2],"drop":[]}'
MISS = '{"add":[2],"drop":[]}'
BAD = '???'


def policy():
    backend = Tokenizer(models.WordLevel(vocab={'[UNK]':0,'p':1,GOLD:2,EXTRA:3,MISS:4,BAD:5},unk_token='[UNK]'))
    backend.decoder = decoders.Fuse()
    return backend


def state(key,cell='append|tight'):
    mode,tier=cell.split('|')
    text=f'问题 {key}'
    reward={"mode":mode, "budget":1000,"kept_ids":[],"new_ids":[1,2],"gold":[1],
            "ntok":{"1":20,"2":40},"prompt_token_ids_tail":[1,1,1,1]}
    return {"split":"train","trajectory_id":f'{cell}:{key}',"task":key,"mode":mode,"budget_tier":tier,
            "step":0,"prompt_sha256":str(key),"prompt_token_ids":[1,1,1,1],"prompt_tokens":4,
            "user_text":text,"messages":[{"role":"user","content":text}],"greedy_output":GOLD,
            "greedy_output_token_ids":[2],"greedy_invalid_reason":None,"greedy_keep_after":[1],
            "reward_state_json":json.dumps(reward)}


def raw(row,texts):
    token={GOLD:2,EXTRA:3,MISS:4,BAD:5}
    return {**{k:row[k] for k in filt.COLUMNS},"samples":[{"index":i,"completion":txt,"token_ids":[token[txt]]}
                        for i,txt in enumerate(texts)]}


def write_rows(path,rows):
    with path.open('x') as f:
        for r in rows:f.write(filt.states.compact(r)+'\n')


@pytest.mark.parametrize('texts,gold,token,informative',[
    ([GOLD]*8,False,False,False),([GOLD,EXTRA]*4,False,True,True),
    ([MISS,GOLD]*4,True,False,True),([MISS,BAD]*4,True,False,True)])
def test_group_components(texts,gold,token,informative):
    r=state(0)
    g=filt.score_group(r,raw(r,texts))
    assert g['gold_varying'] is gold and g['token_only_varying'] is token and g['informative'] is informative
    assert g['greedy_reward']==-.01
    parts={GOLD:(0,20,0),EXTRA:(0,60,0),MISS:(1,40,0),BAD:(1,0,1)}
    assert [tuple(p[k] for k in ('n_lost','used_after','invalid')) for p in g['reward_parts']]==[parts[t] for t in texts]
    stats=filt.summarize([g],2)
    assert stats['completions_per_second']==4 and stats['states_per_second']==.5
    assert stats['invalid']['count']==texts.count(BAD)
    assert stats['multi_add_rate']['count']==texts.count(EXTRA)
    assert stats['sampled_add_counts_parsed']['2']['n']==8-texts.count(BAD)


def test_group_checks_identity_and_sample_indices():
    r=state(3)
    group=raw(r,[GOLD]*8)
    group['samples'].reverse()
    assert filt.score_group(r,group)['rewards']==[-.01]*8
    group['samples'][0]['index']=0
    with pytest.raises(ValueError,match='generation index'):
        filt.score_group(r,group)
    group=raw(r,[GOLD]*8)
    with pytest.raises(ValueError, match='identit'):
        filt.score_group({**r,'task':4},group)
    with pytest.raises(ValueError, match='execution'):
        filt.score_group({**r,'greedy_keep_after':[]},group)


class Engine:
    def __init__(self):
        self.closed=False
        self.llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=self.shutdown))
        self.batches=[]
    def shutdown(self):
        self.closed=True
    def generate(self,prompts,params,lora_request,use_tqdm):
        assert (params.temperature,params.n,params.top_p,params.top_k,params.seed,params.max_tokens)==(1,8,1,-1,20261123,256)
        self.batches.append(len(prompts))
        return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'],
            outputs=[SimpleNamespace(index=i,token_ids=[i%2+2],text=t,finish_reason='stop')
                     for i,t in enumerate([GOLD,EXTRA]*4)]) for p in prompts]


def test_cpu_filter_and_dataset_use_original_quota_and_actual_orm(tmp_path,monkeypatch):
    from src.grpo_r2_rewards import CtxStepRewardV2
    root=tmp_path/'grpo_r2'
    sample_dir=root/'states'
    sample_dir.mkdir(parents=True)
    adapter=tmp_path/'runs'/'checkpoint-1000'
    adapter.mkdir(parents=True)
    (adapter/'adapter_model.safetensors').write_bytes(b'fixture')
    base=tmp_path/'base'
    base.mkdir()
    monkeypatch.setattr(filt.states,'EXTERNAL',tmp_path)
    monkeypatch.setattr(filt.states,'DEFAULT_BASE',base)
    for k,v in [('ROOT',root),('FILTER',root/'filter'),('POOL',sample_dir/'states.jsonl'),('START',tmp_path/'selection.json')]:
        monkeypatch.setattr(filt,k,v)
    source=filt.states.source_info(SimpleNamespace(source_adapter=adapter,adapter=adapter,base_model=base,model_label=adapter.name))
    start={'path':str(adapter),'strategy':adapter.name,'version':'v3','adapter_sha256':source['source_adapter_sha256']}
    filt.START.write_text(json.dumps(start))
    rows=[{**state(k,c),**source} for c in filt.CELLS for k in range(2)]
    write_rows(filt.POOL,rows)
    (sample_dir/'state_checks.json').write_text(json.dumps({'scope':'train','available_checks_pass':True,
         'state_file':filt.states.file_record(filt.POOL),'n_states':len(rows),'source_adapter_sha256':start['adapter_sha256'],
         **dict.fromkeys(('heldout_overlap','locked_test_overlap','gold_mismatches','rerender_mismatches','reencode_mismatches'),0)}))
    fake=Engine()
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=SimpleNamespace))
    monkeypatch.setattr(filt.states,'create_engine',lambda args:(fake,'adapter',{'max_model_len':8192}))
    references=filt.tables({c:[filt.score_group(state(0,c),raw(state(0,c),[GOLD]*8))] for c in filt.CELLS},
                           {},6.0)
    monkeypatch.setattr(filt,'task10_reference',lambda start:{'reward_v2':references})
    monkeypatch.setattr(filt,'task08_reference',lambda :{'reward_v2':references})
    filt.run()
    assert fake.closed and fake.batches==[2]*6
    summary=filt.load(filt.FILTER/'sampling_probe.json')
    assert summary['n_informative']==12 and summary['pooled']['token_only_varying']==filt.states.rate(12,12)
    assert summary['cells']['append|tight']['n_completions']==16
    diff=summary['references']['task10_f']['delta_r2_minus_reference_v2']['pooled']
    assert diff['token_only_varying']=={'r2_n':12,'reference_n':6,'delta':1.0}
    assert (filt.FILTER/'informative.jsonl').read_bytes()==(filt.FILTER/'groups.jsonl').read_bytes()
    with pytest.raises((ValueError,FileExistsError),match='exists'):
        filt.run()
    orm=CtxStepRewardV2(SimpleNamespace(backend_tokenizer=policy()))
    monkeypatch.setattr(data,'make_orm',lambda:orm)
    data.run()
    result=filt.load(root/'data/dataset_stats.json')
    assert result['n']==result['plugin_greedy_checked']==12 and result['plugin_greedy_mismatches']==0
    assert result['status']=='stopped_insufficient_signal' and result['per_cell_counts']==dict.fromkeys(filt.CELLS,2)
    assert result['seed']==20261124


def test_sparse_and_zero_dataset_and_greedy_reward_mismatch(tmp_path):
    from src.grpo_r2_rewards import CtxStepRewardV2
    orm=CtxStepRewardV2(SimpleNamespace(backend_tokenizer=policy()))
    pool=tmp_path/'states.jsonl'
    groups=tmp_path/'groups.jsonl'
    subset=tmp_path/'informative.jsonl'
    rows=[state(k,c) for k,c in enumerate(filt.CELLS)]
    write_rows(pool,rows)
    lows=[filt.score_group(r,raw(r,[GOLD]*8)) for r in rows]
    write_rows(groups,lows)
    subset.write_bytes(b'')
    result=data.build(pool,groups,subset,tmp_path/'train.jsonl',orm)
    assert result['status']=='stopped_insufficient_signal' and result['n']==result['plugin_greedy_checked']==0
    high=filt.score_group(rows[-1],raw(rows[-1],[MISS,EXTRA]*4))
    high['greedy_reward']=-2
    all2,sub2=tmp_path/'groups2',tmp_path/'sub2'
    write_rows(all2,[*lows[:-1],high])
    write_rows(sub2,[high])
    with pytest.raises(ValueError,match='Plugin reward differs'):
        data.build(pool,all2,sub2,tmp_path/'wrong.jsonl',orm)
    with pytest.raises(ValueError,match='Informative stream'):
        data.greedy_rewards(all2,subset)


def test_r2_uses_exact_task08_quota_and_shortfall_rule():
    available=dict.fromkeys(filt.CELLS,1000)
    assert list(data.prior.quotas(available,2000).values()) == [334,334,333,333,333,333]
    available[filt.CELLS[0]]=0
    assert list(data.prior.quotas(available,2000).values()) == [0,401,400,400,400,399]

