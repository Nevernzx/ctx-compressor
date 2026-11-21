"""Task 08 checks for the device guard, population filter and informative-only export."""
import json

import pytest

from scripts import build_formal_grpo_states as build
from scripts import make_formal_grpo_dataset as dataset
from scripts import probe_formal_grpo as probe


def dump(path,rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


def fixture_rows():
    states, groups = [], []
    for n,name in enumerate(probe.CELLS):
        mode,tier=name.split("|")
        for i in range(3):
            tokens=list(range(32))
            user=f"question {name} {i}"
            ident={"split":"train","task":n,"step":i,"trajectory_id":f"trajectory{n}",
                   "mode":mode,"budget_tier":tier,"prompt_sha256":f"h{n}_{i}"}
            states.append({**ident,"messages":[{"role":"user","content":user}],"user_text":user,
                           "prompt_token_ids":tokens,
                           "reward_state_json":json.dumps({"prompt_token_ids_tail":tokens[-24:]})})
            delta=[0.,2e-9,0.4][i]
            groups.append(probe.group_record({**ident,"greedy_reward":0,
                "samples":[{"index":j,"reward":delta if j%2 else 0.,"invalid_reason":None,"action":{"add":[1]}}
                           for j in range(8)]}))
    return states,groups


def test_gpu_gate_only_queries_physical_one(monkeypatch):
    seen=[]
    def query(cmd,text):
        seen.append(cmd)
        return "2047"
    monkeypatch.setattr(build.subprocess,"check_output",query)
    monkeypatch.setenv("CUDA_DEVICE_ORDER","PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","1")
    build.gpu_gate(1)
    assert len(seen)==1 and seen[0][1:3]==["-i","1"]
    with pytest.raises(RuntimeError):
        build.gpu_gate(0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    with pytest.raises(RuntimeError):
        build.gpu_gate(1)
    assert len(seen)==1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","1")
    monkeypatch.setattr(build.subprocess,"check_output",lambda *a,**k:"2048")
    with pytest.raises(RuntimeError,match="occupied"):
        build.gpu_gate(1)


def test_question_overlap_uses_example_identity(tmp_path):
    train,held=tmp_path/"train",tmp_path/"held"
    dump(train,[{"task":0,"split":"train","example_ids":["a"]}])
    dump(held,[{"task":0,"split":"heldout","example_ids":["b"]}])
    assert build.train_overlap(train,held,[0])["heldout_overlap"]==0
    dump(held,[{"task":17,"split":"heldout","example_ids":["a"]}])
    with pytest.raises(ValueError,match="overlap|occur"):
        build.train_overlap(train,held,[0])


def test_budget_shortfall_sorted_round_robin():
    cells=probe.CELLS
    totals=dict.fromkeys(cells,1000)
    assert list(dataset.quotas(totals,2000).values())==[334,334,333,333,333,333]
    totals[cells[0]]=330
    assert list(dataset.quotas(totals,2000).values())==[330,335,334,334,334,333]
    assert dataset.quotas(dict.fromkeys(cells,2),2000)==dict.fromkeys(cells,2)


def test_all_states_filter_and_export(tmp_path):
    states,groups=fixture_rows()
    pool,filter_file=tmp_path/"states",tmp_path/"groups"
    dump(pool,states)
    dump(filter_file,groups)
    indexed=probe.index_states(pool)
    assert list(probe.IndexedRows(pool,indexed[probe.CELLS[0]]))==states[:3]
    assert probe.IndexedRows(pool,indexed[probe.CELLS[1]])[:2]==states[3:5]
    assert [g["informative"] for g in groups[:3]]==[False,False,True]
    assert groups[1]["population_std"]==probe.THRESHOLD
    first=tmp_path/"out1"
    stats=dataset.build(pool,filter_file,first,2000,20261121)
    assert stats["n_states"]==18 and stats["n"]==stats["n_informative"]==stats["rows_checked"]==6
    assert stats["heldout_rows"]==stats["user_text_mismatches"]==0
    assert [item["state_step"] for item in map(json.loads,first.read_text().splitlines())]==[2]*6
    second=tmp_path/"out2"
    dataset.build(pool,filter_file,second,2000,20261121)
    assert first.read_bytes()==second.read_bytes()
    with pytest.raises(FileExistsError):
        dataset.build(pool,filter_file,first,2000,20261121)


def test_filter_rejects_lost_state_or_leakage(tmp_path):
    states,groups=fixture_rows()
    pool,filters=tmp_path/"states",tmp_path/"groups"
    dump(pool,states)
    dump(filters,groups[1:])
    with pytest.raises(ValueError,match="one-to-one"):
        dataset.build(pool,filters,tmp_path/"out1",2000,20261121)
    groups[0]["informative"]=True
    dump(filters,groups)
    with pytest.raises(ValueError,match="threshold"):
        dataset.filter_signals(filters)
    states[1]["split"]="heldout"
    dump(pool,states)
    with pytest.raises(ValueError,match="train"):
        probe.index_states(pool)


def test_prompt_text_and_filter_output(tmp_path):
    states,groups=fixture_rows()
    pool,filters=tmp_path/"states",tmp_path/"groups"
    states[2]["user_text"]="other prompt"
    dump(pool,states)
    dump(filters,groups)
    with pytest.raises(ValueError,match="user_text"):
        dataset.build(pool,filters,tmp_path/"out",2000,20261121)
    groups[2]["add_counts"][0]=None
    groups[2]["invalid_reasons"][0]="invalid_json"
    summary=probe.describe(groups[:3],2.0)
    assert summary["n_completions"]==24 and summary["informative"]["count"]==1
    assert summary["unparseable_add_count"]["count"]==1
    assert summary["executed_add_counts_invalid_noop"]["0"]["count"]==1
    assert summary["sampled_add_counts_parsed"]["1"]["n"]==23



@pytest.mark.parametrize("top_k", [0, -1, 1, -2])
def test_existing_sampler_and_exporter_end_to_end_on_cpu(tmp_path,monkeypatch,top_k):
    """The formal path delegates generation and scoring to Task 07's real batch loop."""
    import sys
    from types import SimpleNamespace
    states,_=fixture_rows()
    pool=tmp_path/"states.jsonl"
    adapter=tmp_path/"checkpoint-2000"
    adapter.mkdir()
    (adapter/"adapter_model.safetensors").write_bytes(b"cpu fixture")
    args=SimpleNamespace(states=pool,adapter=adapter,source_adapter=None,base_model=tmp_path,
                         model_label="cpu fixture",seed=20261120,lam=.3,generations=8,gpu=1,
                         output_dir=tmp_path/"filter")
    source=probe.probe.source_info(args)
    for row in states:
        row.update(source,greedy_output='{"add":[0]}',greedy_invalid_reason=None,greedy_keep_after=[0])
        row["prompt_tokens"]=len(row["prompt_token_ids"])
        row["reward_state_json"]=json.dumps({"mode":row["mode"],"budget":10,"gold":[0],"kept_ids":[],
            "new_ids":[0],"ntok":{"0":1},"prompt_token_ids_tail":row["prompt_token_ids"][-24:]})
    dump(pool,states)
    (tmp_path/"state_checks.json").write_text(json.dumps({"state_file":probe.probe.file_record(pool),
        "available_checks_pass":True,"heldout_overlap":0,"n_states":len(states)}))
    class Params:
        def __init__(self,**options):
            self.top_k=top_k
            self.__dict__.update(options)
    class FakeLLM:
        def generate(self,prompts,sampling,lora_request,use_tqdm):
            assert sampling.n==8 and sampling.seed==20261120 and sampling.top_k==top_k
            return [SimpleNamespace(prompt_token_ids=p["prompt_token_ids"],outputs=[
                SimpleNamespace(index=i,text='{"add":[0]}' if i%2 else '{"add":[]}',token_ids=[i],
                                finish_reason="stop") for i in range(8)]) for p in prompts]
    monkeypatch.setitem(sys.modules,"vllm",SimpleNamespace(SamplingParams=Params))
    monkeypatch.setattr(build,"ROOT",tmp_path)
    monkeypatch.setattr(probe.state_ops,"create_engine", lambda a:(FakeLLM(),None,{"max_model_len":40960}))
    if top_k not in (0, -1):
        with pytest.raises(ValueError, match="Unexpected vLLM top_k default"):
            probe.run(args)
        assert not args.output_dir.exists()
        return
    probe.run(args)
    summary=json.loads((args.output_dir/"sampling_probe.json").read_text())
    assert summary["generation"]["top_k"] == -1
    assert summary["generation"]["top_k_actual"] == top_k
    assert "0 and -1 to disable top-k truncation" in summary["generation"]["top_k_interpretation"]
    assert summary["n_states"]==summary["pooled"]["informative"]["count"]==18
    assert summary["pooled"]["n_completions"]==144
    assert {k:v["n_groups"] for k,v in summary["cells"].items()}==dict.fromkeys(probe.CELLS,3)
    monkeypatch.setattr(sys,"argv",["formal","--states",str(pool),"--informative",str(args.output_dir/"groups.jsonl"),
                                    "--out",str(tmp_path/"data/train.jsonl")])
    dataset.main()
    stats=json.loads((tmp_path/"data/dataset_stats.json").read_text())
    assert stats["n"]==18 and stats["rows_checked"]==18 and stats["shortfall_from_target"]==1982

