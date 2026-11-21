import json
import math

import pytest

from scripts.hedge_action import TAU_GRID, candidate_ids, execute_hedge, ranked_confidence
from scripts.score_hedge_train import Aggregate, local_current_gold, select_tau


@pytest.fixture
def state():
    return {"mode": "reselect", "budget": 12, "kept_ids": [10], "new_ids": [11, 12, 13],
            "ntok": {"10": 5, "11": 3, "12": 4, "13": 10}, "gold": [10, 12]}


def test_confidence_p1_is_global_and_logits_stabilized(state):
    scores = {11: -1010., 12: -1000., 13: -1002.}
    ranked, p1 = ranked_confidence([11, 12, 13], scores)
    assert ranked == [12, 13, 11]
    assert p1 == pytest.approx(1 / (1 + math.exp(-2) + math.exp(-10)))
    raw = '{"add":[11],"drop":[]}'
    assert execute_hedge(state, raw, scores, 0.8)["raw_output"] == raw
    assert execute_hedge(state, raw, scores, 0.9)["raw_output"] == '{"add":[11,12],"drop":[]}'


def test_greedy_invalid_is_noop(state):
    state["mode"] = "append"
    scores = {11: 1., 12: 0., 13: -2.}
    for raw, reason in [("oops", "invalid_json"), ('{"add":[888]}', "invalid_candidate_id"),
                        ('{"add":[],"drop":[10]}', "append_drop"), ('{"add":[13]}', "over_budget")]:
        result = execute_hedge(state, raw, scores, 1.01)
        assert (result["extra_id"], result["raw_output"], result["invalid_reason"]) == (None, raw, reason)
        assert result["keep_after"] == [10] and result["reward"] == -1.0


def test_free_budget_after_drops_and_no_fallback_to_cheaper_candidate(state):
    scores = {11: 2., 13: 1., 12: 0.}
    raw = '{"add":[11],"drop":[]}'
    result = execute_hedge(state, raw, scores, 1.01)
    assert not result["hedged"] and result["raw_output"] == raw
    state["budget"] = 13
    result = execute_hedge(state, '{"add":[11],"drop":[10]}', scores, 1.01)
    assert result["raw_output"] == '{"add":[11,13],"drop":[10]}' and result["used_after"] == 13
    assert result["extra_tokens"] == 10


def test_valid_scores_ties_empty_candidates_and_strict_threshold(state):
    assert candidate_ids({"new_ids":[10,12,11,12], "kept_ids":[10]}) == [12,11]
    assert ranked_confidence([12,11], {11: 0., 12: 0.}) == ([12,11], .5)
    state["new_ids"] = [11,12]
    raw = '{"add":[11]}'
    assert not execute_hedge(state,raw,{11:0.,12:0.},.5)["hedged"]
    assert execute_hedge(state,raw,{11:0.,12:0.},.6)["extra_id"] == 12
    assert execute_hedge(state,raw,None,0)["raw_output"] == raw
    state["new_ids"] = []
    result = execute_hedge(state,'{"add":[]}',{},1.01)
    assert result["p1"] is None and result["extra_id"] is None
    with pytest.raises(ValueError):
        execute_hedge(state,'{"add":[]}',None,1.01)


def test_invalid_score_inputs_fail_closed():
    for scores in ({1:0.}, {1:0.,2:1.,3:1.}, {1:0.,2:float("nan")}):
        with pytest.raises(ValueError):
            ranked_confidence([1,2],scores)


def test_selection_tie_and_denominators(state):
    raw = '{"add":[11]}'
    row = {"task": 100, "step": 0, "pid_by_lid_json": json.dumps({10:100,11:200,12:300,13:400})}
    assert local_current_gold(row,state,{100:[300]}) == 12
    assert local_current_gold(row,state,{100:[100]}) is None
    table = Aggregate()
    no_hedge = execute_hedge(state,raw,{11:1.,12:1.,13:0.},0)
    hedge = execute_hedge(state,raw,{11:1.,12:1.,13:0.},1.01)
    table.add(no_hedge, state["gold"], 12)
    table.add(hedge, state["gold"], 12)
    table.add(no_hedge, state["gold"], None)
    result = table.result()
    assert result["hedge_hit_rate"] == {"count":1,"n":1,"rate":1}
    assert result["current_gold_miss_rate"] == {"count":1,"n":2,"rate":.5}
    assert result["mean_extra_tokens"] == {"sum":4,"n":3,"mean":4/3}
    ranked = {str(t): {"mean_reward":{"mean": 2.0 if t in (.6,.7) else 1.0}} for t in TAU_GRID}
    assert select_tau(ranked) == .6

