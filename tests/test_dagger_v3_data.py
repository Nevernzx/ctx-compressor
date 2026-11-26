"""Task 10 target invariants, prompt provenance, state identity and seeded sampling."""
import copy
import math
from pathlib import Path

import pytest

from scripts import make_sft_v3_data as data
from scripts import rollout_dagger_v3 as rollout
from scripts.hedge_action import candidate_ids, ranked_confidence
from src.build_traj import make_sample, render_input
from src.heldout_eval import HeldoutReplay
from src.model_inference import inference_prompt
from scripts.eval_hedge_v2 import wrap_batch


class Tokenizer:
    def apply_chat_template(self, messages, **_):
        return list(messages[0]["content"].encode("utf-8"))


def example(mode="reselect", budget=8, kept=(0,1), new=(2,3,4), gold=(0,2)):
    corpus = [{"pid":i,"ntok":n,"text":f"paragraph {i}"} for i,n in enumerate([2,2,2,3,2])]
    return make_sample(["Q"],list(kept),list(new),set(gold),corpus,budget,mode,{i:i for i in range(5)})


def decision(s, values):
    ids = candidate_ids(s)
    scores = dict(zip(ids,values))
    _,p1 = ranked_confidence(ids,scores)
    return {"tau":0.99,"p1":p1,"scores":{str(i):s for i,s in scores.items()}}


def test_reselect_hedge_after_correct_drop():
    s=example()
    target,result=data.build_target(s,decision(s,[8.,9.,8.]))
    assert target=={"add":[2,3],"drop":[1]}
    assert set(result["keep_after"])=={0,2,3} and result["used_after"]==7


def test_oracle_uses_top_remaining_new_and_keeps_old_append_prefix():
    s=example(mode="append",budget=10)
    target,result=data.build_target(s,decision(s,[10.,9.,8.]))
    assert target=={"add":[2,3],"drop":[]}
    assert result["keep_after"][:2]==[0,1]


def test_top_extra_over_budget_has_no_lower_score_fallback():
    s=example(budget=6)
    target,result=data.build_target(s,decision(s,[9.,8.,7.]))
    assert target=={"add":[2],"drop":[1]}
    assert result["used_after"]==4 and 4 not in result["keep_after"]


def test_high_confidence_and_missing_candidate():
    s=example()
    assert data.build_target(s,decision(s,[12.,0.,0.]))[0]["add"]==[2]
    s=example(new=(),gold=(0,))
    assert data.build_target(s,decision(s,[]))[0]=={"add":[],"drop":[1]}


def test_confidence_must_match_the_actual_scores():
    s=example()
    h=decision(s,[3.,2.,1.])
    h["p1"]=0.1
    with pytest.raises(ValueError,match="confidence"):
        data.build_target(s,h)


def test_budget_binding_preserves_oracle_gold_and_two_base_adds():
    s=example(kept=(0,1),gold=(0,2,4),budget=6)
    target,result=data.build_target(s,decision(s,[0.,2.,0.]))
    assert set(target["add"])=={2,4}
    assert set(result["keep_after"])=={0,2,4}


def test_tie_score_priority_is_stable_and_reproducible():
    s=example()
    target,_=data.build_target(s,decision(s,[0.,0.,0.]))
    assert target=={"add":[2,3],"drop":[1]}


def test_fixed_mode_balanced_sample_without_replacement():
    pools={m:[f"{m}:{i}" for i in range(30)] for m in data.MODES}
    picked=data.select_ids(pools,5)
    assert picked==data.select_ids(pools,5)
    assert picked!=data.select_ids(pools,5,44)
    assert len(set(picked))==10
    assert all(s.startswith(data.MODES[0]) for s in picked[:5])
    pools["append"][1]=pools["append"][0]
    with pytest.raises(ValueError,match="duplicate"):
        data.select_ids(pools,5)


def test_close_loop_preserves_policy_visible_text_local_and_global_keeps():
    corpus=[{"pid":i,"text":f"p{i}","ntok":2} for i in range(12)]
    tasks=[{"task":80,"questions":["q"],"split":"train","steps":[
        {"new":[7,8,9],"gold":7,"question_index":0,"hop_index":0},
        {"new":[6,10],"gold":10,"question_index":0,"hop_index":1}]}]
    replay=HeldoutReplay(tasks,corpus,rollout.POLICY,budgets={"tight":20},modes=("reselect",))
    items=replay.next_batch()
    s=items[0]["state"]
    prompt=inference_prompt(Tokenizer(),s)
    generated=[{**prompt,"raw_output":'{"add":[0],"drop":[]}',"output_token_ids":[5]}]
    scores=[{0:0.,1:0.,2:0.}]
    raw=wrap_batch(replay,items,generated,scores,.99)[0]
    row=rollout.complete_row(raw,items[0])
    assert row["user_text"]==render_input(s) and row["state"]["text"]==s["text"]
    assert row["greedy_keep_after"]==[0] and row["keep_after"]==[7,8] and row["state"]["keep_after"]==[0,1]
    assert row["state_id"]==row["trajectory_id"]+":0"
    assert replay.next_batch()[0]["state"]["kept_ids"]==row["state"]["keep_after"]
    assert set(row["hedge"]["scores"])=={0,1,2}
    tampered=copy.deepcopy(raw)
    tampered["generation_metadata"]["user_text_sha256"]="bad"
    with pytest.raises(ValueError,match="input"):
        rollout.complete_row(tampered,items[0])


def test_scoring_and_writer_paths_are_isolated():
    assert rollout.CAPTURE.parent == rollout.ROOT
    assert rollout.CAPTURE != rollout.scorer.EXTERNAL/"hedge_v2/score_logits"
    assert data.DATA==Path("/root/userfolder/ctx-compressor/data/sft_v3")



def test_threshold_equality_does_not_hedge():
    state=example(new=(2,3))
    hedge=decision(state,[math.log(99.),0.])
    assert hedge["p1"]==0.99
    assert data.build_target(state,hedge)[0]["add"]==[2]
