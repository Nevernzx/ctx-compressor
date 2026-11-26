import pytest
from scripts import reader_heldout_v2_deepseek as mod


def sample(t=1):
    return [dict(strategy='deepseek-flash',mode=m,tier='mid',task=t,question_index=q)
            for m in ('reselect','append') for q in range(4)]


def test_c_trajectory_batches_keep_order_and_question_identity():
    rows=sample(4)+sample(2)
    assert list(mod.c_batches(rows))==[(4,rows[:8]),(2,rows[8:])]
    with pytest.raises(ValueError): list(mod.c_batches(rows[1:]))
    with pytest.raises(ValueError): list(mod.c_batches(rows+sample(4)))


def test_c_checks_f_reader_prompt_hash_and_evidence():
    row={k:None for k in ('strategy','mode','tier','task','question_index','example_id','answer','answer_aliases',
                          'question_hops','evidence_complete','trajectory_lost','prompt_sha1','non_gold_count')}
    row.update(prompt_sha1='abc',evidence_complete=True)
    mod.validate_question(row,dict(row))
    with pytest.raises(ValueError):mod.validate_question(row,dict(row,prompt_sha1='bad'))
    with pytest.raises(ValueError):mod.validate_question(row,dict(row,evidence_complete=False))


def test_only_manifested_files_are_read(tmp_path):
    f=tmp_path/'traj.json'
    f.write_text('input')
    with pytest.raises(ValueError):mod.signed(f,{})
    with pytest.raises(ValueError):mod.signed(f,{str(f):'0'*64})
    assert mod.signed(f,{str(f):mod.inp.checksum(f)})==mod.inp.checksum(f)


def prediction(**updates):
    base=dict(strategy='oracle',mode='append',tier='mid',task=2,question_index=0,example_id='test',answer='New York',answer_aliases=[],
              question_hops=2,evidence_complete=True,trajectory_lost=False,prompt_sha1='abc',config='oracle|append|mid',
              input_truncated=False,prompt_tokens=22,raw_output='New York',pred='New York',f1=1.,em=1.)
    return dict(base,**updates)


def test_column_rejoins_in_original_order_preserves_deepseek_and_rescores_qwen():
    new=prediction()
    old=dict(new,qwen_prompt_tokens=22,qwen_f1=0.,qwen_em=0.,f1=.2,em=0.)
    joined,=mod.aligned_predictions([prediction(strategy='irrelevant'),new],{mod.ds.key(old):old})
    assert (joined['qwen_f1'],joined['qwen_em'],joined['f1'],joined['em'])==(1.,1.,.2,0.)
    with pytest.raises(ValueError):mod.aligned_predictions([new,new],{mod.ds.key(old):old})
    with pytest.raises(ValueError):mod.aligned_predictions([],{mod.ds.key(old):old})


@pytest.mark.parametrize('field,value',[('prompt_sha1','changed'),('prompt_tokens',23),('raw_output','Chicago'),('tier','tight')])
def test_column_rejects_shifted_identity_or_score(field,value):
    row=prediction()
    old=dict(row,qwen_prompt_tokens=row['prompt_tokens'])
    with pytest.raises(ValueError):mod.aligned_predictions([dict(row,**{field:value})],{mod.ds.key(old):old})


def test_only_completed_cascade_disabled_inference_can_feed_reports(tmp_path):
    import json
    cfg=tmp_path/'cfg.json'
    cfg.write_text(json.dumps(dict(status='complete',engine_shutdown=True,predictions=1,effective_disable_cascade_attn=False,
                                   cuda_visible_devices='1')))
    (tmp_path/'status.json').write_text('{"status":"complete"}')
    with pytest.raises(ValueError):mod.read_inference(tmp_path,cfg,1)

