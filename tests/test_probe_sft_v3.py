from collections import Counter
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import probe_sft_v3 as probe
from scripts.offline_reward_v2 import TAU_GRID, select, v2_cell
from src.model_inference import compact, sha256_json


def fixture_row(mode, i):
    state = {"kept_ids": [], "new_ids":[1,2], "ntok":{"1":10,"2":20}, "budget":100, "gold":[1], "mode":mode}
    return {"split":"train","task":i,"trajectory_id":f"test|{mode}|{i}","step":0,"budget_tier":"mid","mode":mode,
            "state_id":f"test|{mode}|{i}:0","user_text":"abc","state":state,"greedy_raw_output":'{"add":[1]}',
            "greedy_keep_after":[1],"hedge":{"tau":.99},"generation_metadata":{"prompt_tokens":2,
            "prompt_token_ids":[3,4],"prompt_sha256":sha256_json([3,4]),"user_text_sha256":hashlib.sha256(b"abc").hexdigest()}}


def test_reward_grid_exact_ties_and_token_price():
    counts = Counter(n=3, n_lost=2, invalid=1, used=3000, current_n=1, miss=1)
    table = {str(tau):v2_cell(counts,0.25) for tau in reversed(TAU_GRID)}
    assert table["0.5"]["mean_reward"]["mean"] == -1.25
    assert select(table) == 0.


def test_state_selection_excludes_exact_ids(tmp_path):
    rows = [fixture_row(mode,i) for mode in probe.MODES for i in range(9)]
    filename = tmp_path/"population.jsonl"
    filename.write_text("".join(compact(row)+"\n" for row in rows))
    excluded = {rows[0]["state_id"], rows[-1]["state_id"]}
    selected, record, n = probe.sample_states(filename,excluded,5)
    assert n == 18 and record == probe.state_ops.file_record(filename)
    assert not excluded.intersection(row["state_id"] for row in selected)
    assert Counter(row["mode"] for row in selected) == {m:5 for m in probe.MODES}
    assert selected == probe.sample_states(filename,excluded,5)[0]
    with pytest.raises(ValueError,match="Unknown"):
        probe.sample_states(filename,{"unknown"},5)
    with filename.open("a") as f:
        f.write(compact(rows[-1])+"\n")
    with pytest.raises(ValueError,match="duplicate"):
        probe.sample_states(filename,excluded,5)


@pytest.mark.parametrize("field,value", [("user_text","bad"),("greedy_keep_after",[]),("hedge",{"tau":0.5})])
def test_record_requires_original_prompt_and_greedy(field,value):
    row = fixture_row("append",1)
    row[field] = value
    with pytest.raises(ValueError):
        probe.probe_state(row,{})


class FakeLLM:
    def __init__(self):
        self.closed = False
        self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=self.shutdown))

    def shutdown(self):
        self.closed = True

    def generate(self,prompts,sampling,lora_request,use_tqdm):
        assert lora_request and not use_tqdm
        assert vars(sampling) == dict(n=8,temperature=1.,top_p=1.,max_tokens=256,seed=20261120)
        samples = ['{"add":[1]}','{"add":[2,1,1]}','{"add":[1,2]}',"malformed"] * 2
        items = [SimpleNamespace(index=i,text=raw,token_ids=[42],finish_reason="stop") for i,raw in enumerate(samples)]
        return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"],outputs=items) for p in prompts]


def test_two_models_reuse_task07_sampler_and_same_states(tmp_path,monkeypatch):
    model_paths = {}
    for m in ("v2","v3"):
        path = tmp_path/m
        path.mkdir()
        (path/"adapter_model.safetensors").write_bytes(m.encode())
        model_paths[m] = path
    ext = tmp_path/"external"
    out = tmp_path/"eval"
    out.mkdir()
    source = tmp_path/"source.jsonl"
    excluded = tmp_path/"train.json"
    excluded.write_text(json.dumps({"dagger_state_ids":[f"excluded{i}" for i in range(10000)]}))
    frozen = {"path":str(source),"bytes":1,"sha256":"abc"}
    policy_sha = probe.state_ops.sha256(model_paths["v2"]/"adapter_model.safetensors")
    (out/"dagger_rollouts.json").write_text(json.dumps({"steps_file":frozen,"n_steps":22158,
            "status":"complete","adapter_sha256":policy_sha}))
    (out/"sample_manifest.json").write_text(json.dumps({"files":{"sample_ids":probe.state_ops.file_record(excluded)},
                                                        "selected_by_mode":dict.fromkeys(probe.MODES,5000)}))
    rows = [fixture_row(m,i) for m in probe.MODES for i in range(150)]
    for key,value in dict(PROJECT=probe.Path("/root/workplace/ctx-compressor"),EXT=ext,OUT=out,
                          ROLLS=source,TRAIN_IDS=excluded,SHA=policy_sha,ADAPTER=model_paths["v2"]).items():
        monkeypatch.setattr(probe,key,value)
    monkeypatch.setattr(probe,"sample_states", lambda p, excluded:(rows,frozen,22158))
    monkeypatch.setattr(probe,"adapter_identity",lambda m:(model_paths[m],probe.state_ops.sha256(model_paths[m]/"adapter_model.safetensors")))
    monkeypatch.setattr(sys,"prefix",str(ext/"envs/grpo"))
    class SamplingParams(SimpleNamespace):
        top_k = 0
    monkeypatch.setitem(sys.modules,"vllm",SimpleNamespace(SamplingParams=SamplingParams))
    gates = []
    monkeypatch.setattr(probe,"gpu_gate",lambda index:gates.append(index))
    engines = []
    def make_engine(args):
        probe.state_ops.check_gpu(args.gpu)
        llm = FakeLLM()
        engines.append(llm)
        return llm,str(args.adapter),{"max_model_len":512}
    monkeypatch.setattr(probe.state_ops,"create_engine",make_engine)
    for m in ("v3","v2"):
        probe.run(m)
        receipt = probe.load(out/f"probe_{m}.json")
        stats = receipt["tables"]["pooled"]
        assert receipt["training_overlap"] == 0 and receipt["n_completions"] == 2400
        assert stats["multi_add_rate"]["count"] == 1200
        assert stats["invalid_rate"]["count"] == 600
        assert stats["mean_distinct_actions"]["mean"] == stats["mean_distinct_executed_actions"]["mean"] == 3
        assert stats["zero_std_v1"]["count"] == 0
        assert all(x["count"] == 0 and x["n"] == 300 for x in stats["zero_std_v2"].values())
        with pytest.raises(ValueError,match="outputs exist"):
            probe.run(m)
    assert len(engines) == 2 and all(llm.closed for llm in engines) and set(gates) == {1}

