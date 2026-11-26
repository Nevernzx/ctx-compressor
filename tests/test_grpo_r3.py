"""CPU coverage for the round-three change in composition and the B–G wiring."""
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from scripts import grpo_r3_config as cfg
from scripts import grpo_r3_dataset as dataset
from scripts import grpo_r3_reader_check as reader
from scripts import grpo_r3_stages as stages
from scripts import grpo_r3_training as train
from scripts import run_grpo_r3_pipeline as pipeline
from src.grpo_r2_rewards import CtxStepRewardV2
from test_grpo_r2_filter_dataset import state, raw, policy, write_rows, GOLD, EXTRA, MISS


def make_pools(gold, tokens):
    return {layer:{c:[(layer,c,i) for i in range(n)] for c,n in zip(dataset.CELLS,counts,strict=True)}
            for layer,counts in zip(dataset.LAYERS,(gold,tokens),strict=True)}


def test_enough_gold_reproduces_quota_and_shuffle():
    pools = make_pools([400]*6,[300]*6)
    selected, counts = dataset.plan(pools,2000,cfg.DATA_SEED)
    assert list(counts['gold_varying'].values()) == [334,334,333,333,333,333]
    assert set(counts['token_only_varying'].values()) == {0}
    assert len(selected) == len(set(selected)) == 2000
    assert (selected,counts) == dataset.plan(pools,2000,cfg.DATA_SEED)
    assert selected != dataset.plan(pools,2000,cfg.DATA_SEED+1)[0]


def test_gold_shortfall_redistributes_to_other_gold_before_tokens():
    selected, counts = dataset.plan(make_pools([0,500,500,500,500,500],[100]*6),2000,cfg.DATA_SEED)
    assert list(counts['gold_varying'].values()) == [0,401,400,400,400,399]
    assert len(selected) == 2000 and all(s[0]=='gold_varying' for s in selected)


def test_token_topup_balanced_including_round_robin():
    selected, counts = dataset.plan(make_pools([5,0,2,0,0,0],[0,1,9,9,9,9]),20,cfg.DATA_SEED)
    assert list(counts['gold_varying'].values()) == [5,0,2,0,0,0]
    assert list(counts['token_only_varying'].values()) == [0,1,3,3,3,3]
    assert len(selected) == len(set(selected)) == 20


def test_small_dataset_exports_all_informative_with_orm_and_stops_before_training(tmp_path):
    rows = [state(i,c) for i,c in enumerate(dataset.CELLS)]
    groups = [stages.sampler.score_group(r,raw(r,[GOLD,MISS if i%2 else EXTRA]*4)) for i,r in enumerate(rows)]
    pool, all_groups, subset, output = [tmp_path/p for p in ('states','groups','informative','train')]
    write_rows(pool,rows)
    write_rows(all_groups,groups)
    write_rows(subset,groups)
    result = dataset.build(pool,all_groups,subset,output,CtxStepRewardV2(SimpleNamespace(backend_tokenizer=policy())))
    assert result['status'] == 'stopped_insufficient_signal'
    assert result['n'] == result['rows_checked'] == result['plugin_greedy_checked'] == 6
    assert result['gold_varying']['count'] == 3 and result['seed'] == 20261126
    assert result['heldout_rows'] == result['plugin_greedy_mismatches'] == 0
    with pytest.raises(ValueError,match='existing destination'):
        dataset.build(pool,all_groups,subset,output,None)


def test_flags_checked_against_recorded_components(tmp_path):
    r = state(4)
    group = stages.sampler.score_group(r,raw(r,[GOLD,MISS]*4))
    group['gold_varying'] = False
    path = tmp_path/'groups'
    write_rows(path,[group])
    signals,_ = dataset.formal.filter_signals(path)
    with pytest.raises(ValueError,match='components and flags'):
        dataset.group_layers(path,signals)


def test_state_sample_excludes_all_old_tasks():
    population = [{'split':'train','task':i} for i in range(120)]
    histories = [range(10),range(10,20),range(20,30)]
    got = stages.select_tasks(population,histories,n=80)
    assert len({t['task'] for t in got}) == 80 and min(t['task'] for t in got) >= 30
    assert got == stages.select_tasks(population,histories,n=80)


def test_eval_keeps_historical_integrity_checks_on_the_original_model(monkeypatch):
    original = stages.evaluation.v3.v1.MODEL
    def run(name):
        assert name=='merged_start' and stages.evaluation.MERGED==cfg.MERGED
        assert stages.evaluation.v1.MODEL==cfg.BASE
        assert stages.evaluation.v3.v1.MODEL==original
        assert stages.evaluation.RAW==cfg.RAW
    monkeypatch.setattr(stages.evaluation,'run',run)
    stages.evaluate('merged_start')
    assert stages.evaluation.v1.MODEL==original


def test_actual_train_cli_changes_only_provenance_and_seeds():
    original = train.training.signed_json('launch_args.json',train.training.CLI_SHA)['argv']
    changed = train.argv(original, cfg.DATA, cfg.TRAIN, 2000, False)
    values=dict(zip(changed[1::2],changed[2::2],strict=True))
    assert changed[0]=='scripts/grpo_r3_training.py'
    assert values['--seed'] == values['--data_seed'] == str(cfg.SEED)
    assert values['--model']==str(cfg.MERGED) and values['--scale_rewards']=='none'
    assert float(values['--learning_rate'])==1.5e-5 and values['--beta']=='0.04'
    assert values['--reward_funcs']=='ctx_step_reward_v2' and values['--max_steps']=='500'


def test_reader_compares_selected_round_three_with_round_two(tmp_path):
    key='grpo_r3:checkpoint-125'
    data={'n_predictions':5976,'strategies':{key:{'pooled':{'n':5976,'em':.4,'f1':.5}}},
          'comparisons':[{'left':key,'right':'grpo_r2:checkpoint-125','n':5976,
            'f1_counts':{'left':201,'equal':5657,'right':118},'f1_difference_pp':.2,
            'em_difference_pp':.1,'em_matrix_correct_then_incorrect':[[2421,11],[9,3535]]}]}
    path=tmp_path/'reader_f1_results.json'
    path.write_text(json.dumps(data))
    path.with_name('reader_f1_report.md').write_text('n=5976')
    assert reader.verify('checkpoint-125',path)==data
    data['comparisons'][0]['right']='sft_v3:checkpoint-3000'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='G reader pairing'):
        reader.verify('checkpoint-125',path)


def test_pipeline_stops_before_e_for_499_informative(monkeypatch):
    flow = pipeline.Pipeline()
    monkeypatch.setattr(pipeline,'phase_a_evidence',lambda:{})
    monkeypatch.setattr(cfg,'recorded_start',lambda:{})
    monkeypatch.setattr(flow,'evaluate',lambda *args:None)
    log=[]
    def fake(label,command,on_gpu,verify):
        log.append((label,on_gpu))
        return {'n':499,'n_informative':499}
    monkeypatch.setattr(flow,'step',fake)
    flow.run()
    assert log==[('B',True),('C',True),('D',False)]
    assert flow.record['status']=='stopped_insufficient_signal'

