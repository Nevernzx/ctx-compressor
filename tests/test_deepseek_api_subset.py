import hashlib
from decimal import Decimal
import pytest
from scripts import deepseek_api_subset as subset


def test_registered_reader_order_and_exact_identity():
    batch=[]
    for i in range(4):
        text=f'q{i}'
        batch.append(dict(strategy='oracle',mode='reselect',tier='mid',task=9,question_index=i,
                     prompt_sha1=hashlib.sha1(text.encode()).hexdigest(),messages=[{'role':'user','content':text}]))
    subset.validate_order(batch)
    with pytest.raises(ValueError,match='reordered'):
        subset.validate_order(list(reversed(batch)))
    with pytest.raises(ValueError,match='identity mismatch'):
        subset.validate_order([dict(batch[0],prompt_sha1='0'*40),*batch[1:]])


def test_forecast_rescales_actual_token_usage_and_exact_full_trajectory_count():
    pre={'A_full_counts':{'oracle':480,'r2':480,'v3':480,'no_compress':120,'no_context':240},
         'A_full_qwen_tokens':{'oracle':48000,'r2':48000,'v3':48000,'no_compress':12000,'no_context':24000},
         'B':{'n_steps':548}}
    rows=[]
    for kind in ('A_compressed','A_no_compress','B_compressor','B_reader'):
        rows.append(dict(kind=kind,qwen_tokens=100,cost_yuan='0.000065',
               usage={'prompt_cache_hit_tokens':50,'prompt_cache_miss_tokens':60,'completion_tokens':1}))
    q=subset.estimates(rows,pre)
    assert q['by_kind']['B_compressor']['full_calls']==548
    assert sum(v['full_calls'] for v in q['by_kind'].values())==2588
    assert Decimal(q['by_kind']['A_compressed']['estimate_yuan'])==Decimal('.000065')*1440
    assert Decimal(q['by_kind']['A_no_context']['estimate_yuan'])==Decimal('0.000114')*240


def test_refuse_reuse_smoke_dir(tmp_path,monkeypatch):
    monkeypatch.setattr(subset,'WORK',tmp_path)
    d=subset.stage_directory('smoke')
    (d/'one.json').write_bytes(b'old')
    with pytest.raises(FileExistsError):subset.stage_directory('smoke')
    assert (d/'one.json').read_bytes()==b'old'

