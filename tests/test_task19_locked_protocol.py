"""Small CPU contracts for the Task 19 locked-set driver."""
import hashlib
from types import SimpleNamespace

import pytest

from scripts import test_v1_data as data, test_v1_rollout as rollout
from scripts.reader_heldout_v2 import require_effective_cascade_off
from src import reader_heldout_inputs as reuse
from src.build_traj import build_corpus


def test_lock_check_requires_every_bytes_signature(tmp_path):
    a,b=tmp_path/"steps.jsonl",tmp_path/"tasks.jsonl"
    a.write_text("step\n")
    b.write_text("task\n")
    expected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (a,b)}
    lock=tmp_path/"LOCK.md"
    lock.write_text("# Locked\n"+"".join(f"- {p}  {sha}\n" for p,sha in expected.items()))
    stamp=hashlib.sha256(lock.read_bytes()).hexdigest()
    assert set(data.lock_check(lock,stamp,expected))=={str(a),str(b)}
    b.write_text("different task\n")
    with pytest.raises(ValueError,match="SHA256"):
        data.lock_check(lock,stamp,expected)


def test_dev_rebuild_checks_dedup_order_and_exact_gold_text():
    one={"idx":0,"title":"Gold article","paragraph_text":"one paragraph,  exact whitespace"}
    two={"idx":1,"title":"Other","paragraph_text":"blue"}
    raw={"id":"q","question":"Question?","answer":"answer","answer_aliases":[],"paragraphs":[two,one,one],
         "question_decomposition":[{"question":"first","answer":"answer","paragraph_support_idx":0}]}
    corpus, _, keys=build_corpus([raw,raw],lambda s:len(s.split()))
    assert [(p["title"],p["paragraph_text"]) for p in corpus] == [
        (two["title"],two["paragraph_text"]),(one["title"],one["paragraph_text"])]
    row={"task":8,"metadata":[{"example_id":raw["id"],"question":raw["question"],"answer":raw["answer"],
                              "answer_aliases":[]}],"steps":[{"question_index":0,"hop_index":0,"gold":1}]}
    assert data.verify_gold([row],{"q":raw},corpus,keys)==[{"task":8,"steps":1,"gold_mismatches":0}]
    corpus[1]["paragraph_text"]=corpus[1]["paragraph_text"].replace("  "," ")
    with pytest.raises(ValueError,match="dev corpus gold bytes differ"):
        data.verify_gold([row],{"q":raw},corpus,keys)


def test_reader_real_request_order_and_exact_200_per_task():
    task=9
    corpus=[{"title":"Evidence","paragraph_text":"text","ntok":4}]
    meta=[{"example_id":str(q),"question":f"Q{q}?","answer":"a","answer_aliases":[],"question_hops":2,"gold":{0}}
          for q in range(4)]
    tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:[1,2,3])
    finals={name:{(cell,task):[0] for cell in reuse.CELLS} for name in data.NAMES if name!="no_context"}
    requests=list(data.requests(finals,{task:{"metadata":meta,"all_gold":{0}}},corpus,tokenizer))
    assert len(requests)==200
    assert [r["strategy"] for r in requests[::4]][:2]==["no_compress","no_context"]
    assert [r["strategy"] for r in requests[8::24]]==["latest","bm25","oracle","RR-fill","RR-τ","v3","r2","r4"]
    assert requests[4]["trajectory_lost"] is True
    assert next(data.batches(requests,[task]))==(task,requests)
    bad=list(requests)
    bad[4],bad[5]=bad[5],bad[4]
    with pytest.raises(ValueError,match="unexpected reader batch"):
        list(data.batches(bad,[task]))


def test_replay_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        rollout.run_loop({},[],"oracle",tmp_path,None)


@pytest.mark.parametrize("effective",[False,None,"True",1])
def test_reader_gate_is_boolean_true(effective):
    require_effective_cascade_off(True,"llm_engine.model_config")
    with pytest.raises(ValueError,match="disable_cascade_attn"):
        require_effective_cascade_off(effective,"llm_engine.model_config")



def test_task19_reader_prepare_includes_disk_snapshot_used_by_v16(monkeypatch,tmp_path):
    from scripts import test_v1_reader as run

    metadata=[{"example_id":str(i),"question":f"Q{i}?","answer":"a","answer_aliases":[],
               "question_hops":2,"gold":{0}} for i in range(4)]
    task=3
    tasks={task:{"metadata":metadata,"all_gold":{0}}}
    corpus=[{"title":"Evidence","paragraph_text":"text","ntok":4}]
    finals={name:{(cell,task):[0] for cell in reuse.CELLS} for name in data.NAMES if name!="no_context"}
    opening={"used_bytes":1234,"available_bytes":8766,"size_bytes":10000}
    path=tmp_path/"reader"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(run,"READER",path)
    monkeypatch.setattr(run,"PREFLIGHT",path/"input_preflight.json")
    monkeypatch.setattr(run,"N_TASKS",1)
    monkeypatch.setattr(run,"N_PREDICTIONS",200)
    monkeypatch.setattr(data,"load_test",lambda:(corpus,tasks))
    monkeypatch.setattr(run,"final_contexts",lambda *a:(finals,{},{}))
    monkeypatch.setattr(data,"reader_tokenizer",lambda:SimpleNamespace(apply_chat_template=lambda *a,**k:[1,2]))
    monkeypatch.setattr(run.v16.reader.original,"root_df",lambda:opening)
    run.prepare()

    assert reuse.read(run.PREFLIGHT)["df_opening"] == opening



def test_task19_reader_run_rejects_legacy_preflight_before_engine(monkeypatch,tmp_path):
    from scripts import test_v1_reader as run

    old=tmp_path/"input_preflight.json"
    old.write_text('{"status":"passed"}')
    monkeypatch.setattr(run,"PREFLIGHT",old)
    with pytest.raises(ValueError,match="missing its df_opening"):
        run.run()
