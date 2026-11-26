"""CPU coverage of round-2 task isolation, rollout statistics and merged-start comparisons."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import build_grpo_r2_states as b
from scripts import eval_grpo_r2 as e
from src.grpo_reward_v2 import score_completion_v2


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_disjoint_task_sampling_stable_and_source_ordered():
    population = [{"split":"train", "task":i} for i in range(850)]
    old08, old10 = list(range(100)), list(range(50,150))
    result = b.select_tasks(population,old08,old10)
    ids = [r["task"] for r in result]
    assert len(set(ids)) == len(ids) == 400
    assert not (set(ids) & (set(old08) | set(old10)))
    assert b.select_tasks(population,old08,old10) == result
    import random
    assert ids == random.Random(20261123).sample(list(range(150,850)),400)
    with pytest.raises(ValueError,match="previous training task missing"):
        b.select_tasks(population,old08,[1000])
    with pytest.raises(ValueError,match="non-training"):
        b.select_tasks([*population,{"task":851,"split":"heldout"}],old08,old10)


@pytest.mark.parametrize("data,match", [
    (({"a","b","c","d"},{"z"},{"w"}),None),
    (({"a","b","c"},{"z"},{"w"}),"non-training"),
    (({"a","b","c","d"},{"a"},{"w"}),"raw dev"),
    (({"a","b","c","d"},{"x"},{"c"}),"held-out"),
])
def test_question_split_proof(data,match):
    tasks = [{"example_ids":list("abcd")}]
    if match:
        with pytest.raises(ValueError,match=match): b.check_question_ids(tasks,*data)
    else:
        checks = b.check_question_ids(tasks,*data)
        assert checks["locked_test_overlap"] == checks["heldout_overlap"] == 0
        assert checks["train_example_ids"] == 4 and checks["raw_dev_ids_compared"] == 1


def step(raw, mode="reselect", ids=(1,2), step_number=1):
    state = {"mode":mode, "budget":700, "kept_ids":[0], "new_ids":[1,2], "gold":[0,1],
             "ntok":{"0":150, "1":100, "2":300}, "questions":["q"],
             "pid_by_lid":{"0":30,"1":31,"2":32}}
    score = score_completion_v2(raw,state,.5)
    state["keep_after"] = score["keep_after"]
    return {"state":state,"used_after":score["used_after"],"raw_output":raw,"invalid_reason":score["invalid_reason"],
            "mode":mode,"step":step_number,"budget_tier":"tight","task":1,"gold":31,"returned":[31,32],
            "generation_metadata":{"prompt_token_ids":list(ids),"user_text_sha256":"user hash"}}


def test_objective_replay_pooled_component_denominators_and_raw_identity(tmp_path):
    good = step('{"add":[1],"drop":[]}')
    bad = step('garbled',mode="append")
    other = step('{"add":[1,2],"drop":[]}')
    diff = deepcopy(bad)
    diff["generation_metadata"]["prompt_token_ids"] = [3,4]
    raw, start = tmp_path/"steps", tmp_path/"start"
    write_rows(raw,[good,bad])
    write_rows(start,[other,diff])
    extra = {"combined":{"n_steps":2},"configs":{f"{mode}|tight":{"n_steps":1} for mode in ("append","reselect")}}
    e.augment_objective(extra,raw,start)
    result = extra["combined"]["heldout_reward_v2"]
    assert result["n_lost"]["sum"] == 1
    assert result["used_per_1000"]["mean"] == .2
    assert result["invalid"] == {"count":1,"n":2,"rate":.5}
    assert result["reward"]["mean"] == -1.1
    assert extra["combined"]["raw_diff_identical_state"] == {"count":1,"n":1,"rate":1}
    assert extra["combined"]["objective_deltas_vs_start"]["reward"] == pytest.approx(.075)
    assert extra["configs"]["append|tight"]["raw_diff_identical_state"]["n"] == 0
    assert extra["configs"]["reselect|tight"]["start_reward_v2"]["reward"]["mean"] == -.275
    write_rows(start,[other])
    with pytest.raises(ValueError,match="unaligned"): e.augment_objective(extra,raw,start)
    altered = {**good,"used_after":249}
    with pytest.raises(ValueError,match="scorer"): e.reward_observation(altered)


def test_state_statistics_count_executed_adds_and_long_prompts(tmp_path):
    rows = []
    for i, raw in enumerate(('{"add":[1,2],"drop":[]}', 'bad json')):
        item = step(raw)
        state = item["state"]
        rows.append({"split":"train","trajectory_id":"test","step":i,"reward_state_json":json.dumps(state),
                     "mode":item["mode"],"budget_tier":"tight","prompt_tokens":8392+i,
                     "greedy_output":raw,"greedy_keep_after":state["keep_after"],"greedy_invalid_reason":item["invalid_reason"]})
    path = tmp_path/"states"
    write_rows(path,rows)
    stats = b.rollout_stats(path,2)
    for name in ("pooled","reselect|tight"):
        assert stats[name]["add_counts"]["2"]["count"] == stats[name]["add_counts"]["0"]["count"] == 1
        lengths = stats[name]["prompt_lengths"]
        assert (lengths["min"], lengths["p50"], lengths["max"], lengths["exceeds_8392"]) == (8392,8392.5,8393,1)
    write_rows(path, [rows[0],rows[0]])
    with pytest.raises(ValueError,match="identity"): b.rollout_stats(path,2)


def test_adapter_provenance_and_no_lora_f0(tmp_path,monkeypatch):
    monkeypatch.setattr(e,"RUN",tmp_path)
    monkeypatch.setattr(e,"MERGED",tmp_path/"merged_start")
    assert e.model_files("merged_start") == (None,None)
    with pytest.raises(ValueError,match="cannot load"):
        e.model_files("merged_start",tmp_path/"a")
    with pytest.raises(ValueError,match="invalid evaluation name"):
        e.model_files("../checkpoint-125")
    folder = tmp_path/"train/checkpoint-125"
    folder.mkdir(parents=True)
    (folder/"adapter_config.json").write_text(json.dumps({"base_model_name_or_path":str(e.MERGED),"peft_type":"LORA","r":16}))
    (folder/"trainer_state.json").write_text('{"global_step":125}')
    assert e.model_files("checkpoint-125") == (125,folder)
    (folder/"trainer_state.json").write_text('{"global_step":500}')
    with pytest.raises(ValueError,match="provenance"):e.model_files("checkpoint-125")


def test_start_baseline_pinned_to_version_and_index(tmp_path,monkeypatch):
    dirs = [tmp_path/"metric",tmp_path/"raw"]
    source = {"strategy":"checkpoint-3000","version":"v3"}
    monkeypatch.setattr(e.v3.cfg,"OUT",dirs[0])
    monkeypatch.setattr(e.v3,"RAW",dirs[1])
    monkeypatch.setattr(e,"validate_strategy",lambda m,name:m)
    name=source["strategy"]
    metric=dirs[0]/name/f"{name}_results.json"
    steps=dirs[1]/name/"steps.jsonl"
    for file in (metric, steps):file.parent.mkdir(parents=True)
    steps.write_text("rollout")
    metric.write_text(json.dumps({"steps_artifact":e.v1.file_record(steps)}))
    source.update(metrics_path=str(metric),steps_path=str(steps))
    assert e.start_baseline(source)[1] == steps
    steps.write_text("changed")
    with pytest.raises(ValueError,match="rollout bytes"): e.start_baseline(source)
    with pytest.raises(ValueError,match="metrics path"): e.start_baseline({**source,"version":"v2"})

