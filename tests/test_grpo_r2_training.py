import json
from types import SimpleNamespace

import pytest

from scripts import check_grpo_r2_training as verify
from scripts import run_grpo_r2_training as worker
from src import grpo_r2_rewards as plugin
from src.grpo_reward_v2 import score_completion_v2


def state():
    return {'mode':'reselect','budget':12,'kept_ids':[1],'new_ids':[2,3],
            'gold':[1,2],'ntok':{'1':4,'2':5,'3':2},'prompt_token_ids_tail':[1,1,1,1]}


def test_actual_plugin_uses_generated_ids_and_component_clock(monkeypatch):
    raw=['{"add":[2]}','{"add":[2,3,3]}','{"add":[],"drop":[1]}','oops']
    monkeypatch.setattr(plugin,'decode_rollout',lambda tokenizer, tail, ids:raw[ids[0]])
    orm=plugin.CtxStepRewardV2(SimpleNamespace(backend_tokenizer=object()))
    trainer=SimpleNamespace(global_step=0)
    expected=[score_completion_v2(t,state(),.5)['reward'] for t in raw]
    for _ in range(8):
        assert orm(['displayed prefix']*4,[json.dumps(state())]*4,[[i] for i in range(4)],trainer_state=trainer)==expected
    trainer.global_step=1
    result=plugin.take_components(trainer)
    assert result=={'reward_component_n':32,'mean_n_lost':.75,'mean_used_per_1000':.006,
                    'invalid_rate':.25,'mean_add_count':.75,'multi_add_rate':.25}
    with pytest.raises(ValueError,match='missing'):
        plugin.take_components(trainer)


def test_plugin_refuses_misaligned_or_stale_samples(monkeypatch):
    with pytest.raises(ValueError,match='aligned'):
        plugin.rollout_scores(None,[''],[],[])
    monkeypatch.setattr(plugin,'decode_rollout',lambda *args:'oops')
    orm=plugin.CtxStepRewardV2(SimpleNamespace(backend_tokenizer=None))
    trainer=SimpleNamespace(global_step=0)
    orm([''],[json.dumps(state())],[[0]],trainer_state=trainer)
    trainer.global_step=1
    with pytest.raises(ValueError,match='prior'):
        orm([''],[json.dumps(state())],[[0]],trainer_state=trainer)
    with pytest.raises(ValueError,match='misaligned'):
        plugin.take_components(trainer)


def reference():
    return ['scripts/run_formal_swift_grpo.py','--model','old/merged','--dataset','old/data',
            '--output_dir','old/out','--max_steps','500','--external_plugins','old.py',
            '--reward_funcs','ctx_step_reward','--learning_rate','5e-6','--seed','20261120','--data_seed','20261120',
            '--dynamic_sample','false','--beta','0.04']


def test_cli_inherits_original_knobs_and_overrides_registered_changes():
    before=reference()
    native=worker.swift_argv(before,'new/merged','new/data','new/out',1131)
    r2=worker.swift_argv(before,'new/merged','new/data','new/out',1131,lr=1.5e-5,scale_rewards='none')
    args=dict(zip(r2[1::2],r2[2::2]))
    assert before==reference()
    assert native[native.index('--scale_rewards')+1]=='group'
    assert float(native[native.index('--learning_rate')+1])==5e-6
    assert args['--scale_rewards']=='none' and float(args['--learning_rate'])==1.5e-5
    assert args['--max_steps']=='282' and args['--beta']=='0.04' and args['--dynamic_sample']=='false'
    assert args['--seed']==args['--data_seed']=='20261123'
    assert args['--reward_funcs']=='ctx_step_reward_v2'


@pytest.mark.parametrize('n,scale,lr',[(499,'none',1.5e-5),(2001,'none',1.5e-5),(500,'invalid',1.5e-5),
                                        (500,'none',float('inf')),(500,'none',0)])
def test_training_count_lr_and_scale(n,scale,lr):
    with pytest.raises(ValueError):
        worker.swift_argv(reference(),'m','d','o',n,lr=lr,scale_rewards=scale)


def batch():
    outputs=['{"add":[2]}','{"add":[2,3]}','{}','{"add":[],"drop":[1]}']*2
    rewards=[score_completion_v2(t,state(),.5)['reward'] for t in outputs]
    mean=sum(rewards)/8
    return {'step':['1']*32,'prompt':[verify.old.USER_START+'user'+verify.old.RESPONSE_SUFFIX]*32,
            'completion':[t+'<|im_end|>' for t in outputs]*4,'CtxStepRewardV2':rewards*4,
            'advantages':[r-mean for r in rewards]*4}


def jsonl(path,rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def test_recheck_includes_terminal_repeats_and_components(tmp_path):
    b=batch()
    result,components=verify.reward_recheck(jsonl(tmp_path/'r',[b,b]),{'user':state()},1)
    assert result['n']==64 and result['extra_logged_completions']==32
    assert result['mismatches']==result['advantage_mismatches']==0
    assert components[1]['mean_n_lost']==.75 and components[1]['mean_add_count']==.75
    assert components[1]['multi_add_rate']==components[1]['invalid_rate']==.25


@pytest.mark.parametrize('defect',['reward','advantage','count','step','user','group'])
def test_corrupt_logged_completion_or_wrong_reward_scaling_rejected(tmp_path,defect):
    b=batch()
    if defect=='reward':
        b['CtxStepRewardV2'][5]=0
    elif defect=='advantage':
        b['advantages'][0]*=3
    elif defect=='count':
        for v in b.values():
            v.pop()
    elif defect=='step':
        b['step'][0]='2'
    elif defect=='user':
        b['prompt']=[x.replace('user\nuser','user\nnew') for x in b['prompt']]
    else:
        b['prompt'][0]+='extra'
    with pytest.raises(ValueError):
        verify.reward_recheck(jsonl(tmp_path/'r',[b]),{'user':state()},1)


def test_checks_all_resolved_params_and_records_actual_estimator(tmp_path):
    ref=reference()
    old={k[2:]:v for k,v in zip(ref[1::2],ref[2::2])}
    old.update(advantage_estimator='grpo',loss_type='grpo',kl_in_reward=False,
               gradient_accumulation_steps=8,steps_per_generation=None)
    saved={**old,'model':str(worker.ROOT/'merged_start'),'dataset':[str(tmp_path)],'output_dir':str(worker.ROOT/'train'),
           'learning_rate':1.5e-5,'max_steps':500,'scale_rewards':'none','seed':20261123,'data_seed':20261123,
           'external_plugins':['scripts/grpo_r2_swift_plugin.py'],'reward_funcs':['ctx_step_reward_v2']}
    _,cfg=verify.verify_config(saved,old,ref,tmp_path,2000)
    assert cfg['advantage_estimator']==cfg['loss_type']=='grpo'
    for field in ('dynamic_sample','loss_type','beta','scale_rewards','learning_rate'):
        with pytest.raises(ValueError,match='configuration'):
            verify.verify_config({**saved,field:None},old,ref,tmp_path,2000)


def test_optimizer_curve_crosschecks_live_plugin_against_parser(tmp_path):
    _,components=verify.reward_recheck(jsonl(tmp_path/'completions',[batch()]),{'user':state()},1)
    comp=components[1]
    score=-comp['mean_n_lost']-.5*comp['mean_used_per_1000']-comp['invalid_rate']
    values={k:0. for k in verify.old.METRICS}
    values['reward']=score
    own={'step':1,'step_seconds':1.5,'torch_peak_allocated_bytes':100,'torch_peak_reserved_bytes':130,**values,**comp}
    jsonl(tmp_path/'optimizer_steps.jsonl',[own])
    jsonl(tmp_path/'logging.jsonl',[{'global_step/max_steps':'1/1',**values}])
    assert verify.join_curve(tmp_path,1,components)[0]['mean_used_per_1000']==comp['mean_used_per_1000']
    jsonl(tmp_path/'optimizer_steps.jsonl',[{**own,'invalid_rate':1}])
    with pytest.raises(ValueError,match='component'):
        verify.join_curve(tmp_path,1,components)

