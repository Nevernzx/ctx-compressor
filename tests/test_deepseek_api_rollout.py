"""Task 14 offline tests for signed identities, strict action replay and diagnostics."""
import json
from dataclasses import replace

import pytest

from scripts import deepseek_api_rollout as rollout


def fixture():
    corpus = [{"pid":i,"ntok":2000 if i == 4 else 100,"text":"passage "+str(i)} for i in range(5)]
    task = {"task":9,"split":"heldout","questions":["q"+str(i) for i in range(4)],
            "example_ids":[str(i) for i in range(4)],"steps":[
                {"question_index":i,"hop_index":0,"new":[i,4],"gold":i} for i in range(4)]}
    replay = rollout.HeldoutReplay([task],corpus,rollout.STRATEGY,budgets=rollout.BUDGET)
    reference = {(9,x["mode"]):rollout.text_sha(rollout.user_text(x["state"])) for x in replay.next_batch()}
    receipt = rollout.check_grid(corpus,{9:task},[9],reference)
    return rollout.BInputs(corpus,{9:task},[9],reference,receipt)


def reply(raw):
    return {"content":raw,"usage":{"completion_tokens":7,"prompt_tokens":268},
            "finish_reason":"stop","id":"test-call"}


def test_registered_system_exact():
    assert rollout.text_sha(rollout.SYSTEM) == "caef58ecd918771e5a83d071a0e03bf2dcf3d593b059da1c17a3c96786457a85"


def test_step0_mismatch_rejected_before_any_request():
    b = fixture()
    assert b.receipt["n_trajectories"] == 2 and b.receipt["n_steps"] == 8
    wrong = {**b.step0,(9,"append"):"0"*64}
    with pytest.raises(ValueError,match="step 0 user text differs"):
        rollout.check_grid(b.corpus,b.tasks,b.s30,wrong)
    with pytest.raises(ValueError,match="step 0 user text differs"):
        rollout.run_b_task(replace(b,step0=wrong),9,"append",
                           lambda *args:pytest.fail("prompt mismatch reached generation"))


def test_replay_and_metrics_match_production_for_sequential_responses():
    b = fixture()
    ll = []
    answers = ['{"add":[0]}', '```json\n{"add":[2]}\n```', '{"add":[999]}', '{"add":[],"drop":[0]}']
    def policy(messages,tid,mode,step):
        assert (tid,mode,step)==(9,"append",len(ll))
        assert messages[0]=={"role":"system","content":rollout.SYSTEM} and messages[1]["role"]=="user"
        ll.append(messages[1]["content"])
        return reply(answers[step])
    traj=rollout.run_b_task(b,9,"append",policy)
    rows=traj["steps"]
    assert [r["invalid_reason"] for r in rows] == [None,"invalid_json","invalid_candidate_id","append_drop"]
    assert [r["fence_or_whitespace_rescued"] for r in rows]==[False,True,False,False]
    assert all(r["keep_after"]==[0] and r["used_after"]==100 for r in rows)
    assert traj["final_pids"]==[0] and rows[1]["kept_before"]==[0]
    assert "已用 token：0" in ll[0] and "已用 token：100" in ll[1]
    assert rows[0]["generation_metadata"]["user_text_sha256"]==b.step0[9,"append"]
    agg=rollout.aggregate_b(b,[json.loads(json.dumps(traj))])["combined"]
    assert agg["n_steps"]==4 and agg["n_tasks"]==1 and agg["invalid_outputs"]["any_invalid"]["count"]==3
    assert agg["final_recall"]["mean"]==.25 and agg["evidence_loss"]["count"]==1
    assert agg["cost"]["plain"]["mean"]==400 and agg["cost"]["cached"]["mean"]==140
    assert agg["fence_or_whitespace_rescued"]=={"count":1,"n":4,"n_invalid":3}
    assert agg["actual_add_distribution"]=={"0":3,"1":1} and agg["used_after"]["mean"]==100


def test_diagnostic_never_salvages_illegal_fenced_action_or_changes_execution():
    state={"kept_ids":[],"new_ids":[0,1],"gold":[0],"ntok":{0:5,1:7},"mode":"reselect","budget":6}
    def rescue(s):
        return rollout.diagnostic_rescue(state,s,rollout.apply_model_output(state,s)["invalid_reason"])
    assert rescue('  ```json\n{"add":[0]}\n```  ')
    assert rescue('```json{"add":[0]}```')
    assert rescue('\u00a0{"add":[0]}\u00a0')
    assert not rescue('```json\n{"add":[1]}\n```')
    assert not rescue(' \"irreparable\" ')
    assert not rescue('  {"add":[0]}  ')


def test_aggregation_rejects_changed_final_context_and_missing_batch():
    b=fixture()
    traj=rollout.run_b_task(b,9,"reselect",lambda *args:reply('{"add":[]}'))
    with pytest.raises(ValueError,match="invalid B trajectory"):
        rollout.aggregate_b(b,[dict(traj,final_pids=[2])])
    with pytest.raises(ValueError,match="step count differs"):
        rollout.aggregate_b(b,[dict(traj,steps=traj["steps"][:-1])])

