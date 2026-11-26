import pytest
from scripts.reader_diag_metrics import comparison, inconsistent, paired, signature

def test_duplicate_token_groups_match_reference_raw_comparison():
    a=signature({'token_ids':[3,4]});b=signature({'token_ids':[4,3]})
    assert inconsistent([(a,'A'),(a,'a'),(a,'A'),(b,'x'),(b,'x'),(b'x','z')])=={'count':1,'n':2,'rate':.5}
    assert inconsistent([(b,'solo')])['n']==0

def test_only_same_input_keys_enter_cross_run_subset_and_changed_lengths():
    def row(raw,f1,n):return {'raw_output':raw,'f1':f1,'em':int(f1==1),'output_tokens':n}
    left={0:row('cat',1,1),1:row('DOG',1,3),2:row('zz',0,4)}
    right={0:row('cat',1,1),1:row('dog',1,2),2:row('zz',0,4)}
    x=comparison(left,right,[0,1])
    assert x['raw_differences']=={'count':1,'n':2,'rate':.5}
    assert x['length_changed']=={'left':{'n':1,'mean':3},'right':{'n':1,'mean':2}}
    assert x['left_scores']=={'n':2,'f1':1,'em':1}
    assert paired(left,right)['f1_counts']['equal']==3
    with pytest.raises(ValueError):paired(left,{})

