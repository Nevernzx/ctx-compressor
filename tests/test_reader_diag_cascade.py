"""CPU coverage of the fixed diagnostic arm inputs and launch guards."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from scripts import reader_diag_cascade as d


def rows(strategy, tasks=(7, 4)):
    return [dict(strategy=strategy, mode=m, tier=t, task=task, question_index=qi,
                 prompt_sha1=f"{task}-{m}-{t}-{qi}", token_ids=[task, qi], prompt_tokens=2)
            for task in tasks for m,t in d.inputs.CELLS for qi in range(4)]


def test_a_forwards_every_original_row_in_order():
    spool = rows(d.R3)
    got = list(d.batches(d.ARMS[0], iter(spool)))
    assert [task for task,_ in got] == [7,4]
    assert [len(batch) for _,batch in got] == [24,24]
    assert [id(row) for _,batch in got for row in batch] == list(map(id,spool))


def test_b_is_task_local_r2_then_r3_and_keeps_both_orders():
    r2, r3 = rows(d.R2), rows(d.R3)
    got = list(d.batches(d.ARMS[1], iter(r3), iter(r2)))
    assert [task for task,_ in got] == [7,4]
    for i,(_,batch) in enumerate(got):
        start=i*24
        assert list(map(id,batch)) == list(map(id,r2[start:start+24]+r3[start:start+24]))


@pytest.mark.parametrize("case", ["missing", "out_of_order", "different_tasks", "invalid_length", "overflow"])
def test_rejects_malformed_batches(case):
    data = rows(d.R3)
    if case == "missing":
        data.pop()
    elif case == "out_of_order":
        data[0], data[1] = data[1], data[0]
    elif case == "different_tasks":
        data = rows(d.R3, tasks=(4,7))
    elif case == "invalid_length":
        data[0]["prompt_tokens"] += 1
    else:
        data[0].update(token_ids=[1]*40913, prompt_tokens=40913)
    with pytest.raises(ValueError):
        list(d.batches(d.ARMS[1], data, rows(d.R2)))


@pytest.mark.parametrize("change", ["hash","token","order","missing"])
def test_r2_compares_signed_prompt_identities_and_tokens(change):
    rebuilt = rows(d.R2)
    archived = [r.copy() for r in rebuilt]
    preds = [r.copy() for r in archived]
    assert list(d.checked_r2_rows(rebuilt,archived,preds)) == rebuilt
    if change == "hash":
        preds[2]["prompt_sha1"] = "other"
    elif change == "token":
        archived[1]["token_ids"] = [0,1]
    elif change == "order":
        preds[2], preds[3] = preds[3], preds[2]
    else:
        preds.pop()
    with pytest.raises(ValueError):
        list(d.checked_r2_rows(rebuilt,archived,preds))


def test_short_ipc_and_utf8_limit():
    short = d.ipc_record(Path('/root/userfolder/ctx-compressor/rt/g15'))
    assert short["path_bytes"] == 75 and short["uri_bytes"] == 81
    with pytest.raises(ValueError,match="IPC"):
        d.ipc_record(Path('/tmp')/('测'*21))


def test_second_launch_does_not_replace_any_file(tmp_path,monkeypatch):
    monkeypatch.setattr(d,"WORK",tmp_path)
    monkeypatch.setattr(d,"SHORT",tmp_path/"short")
    monkeypatch.setattr(d,"SMALL",tmp_path/"small")
    monkeypatch.setattr(d,"check_tests",lambda:None)
    monkeypatch.setattr(d,"checked_inputs",lambda:None)
    old=tmp_path/"pipeline.pid"
    old.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        d.detach()
    assert old.read_bytes() == b"unchanged" and list(tmp_path.iterdir())==[old]


def test_effective_flag_prefers_model_config_to_logged_cli_args(tmp_path):
    logfile=tmp_path/"log"
    logfile.write_text("non-default args: {'disable_cascade_attn': True}\nmodel config:disable_cascade_attn=True")
    conf=NS(model_config=NS(disable_cascade_attn=False))
    assert d.effective_cascade(NS(vllm_config=conf),logfile)[:2] == (
        False,"llm_engine.vllm_config.model_config.disable_cascade_attn")
    assert d.effective_cascade(NS(get_vllm_config=lambda:conf),logfile)[0] is False
    assert d.effective_cascade(NS(),logfile)[:2] == (True,"engine startup config line")
    logfile.write_text("non-default args: config {'disable_cascade_attn': True}")
    assert d.effective_cascade(NS(),logfile) == (None,"未取得",None)

