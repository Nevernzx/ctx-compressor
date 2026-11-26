"""CPU regression tests for the reader runner and held-out event receipts."""
import json
from types import SimpleNamespace
import pytest
from scripts import run_reader_heldout as heldout


def test_preexisting_predictions_cannot_be_overwritten(tmp_path):
    path=tmp_path/'predictions.jsonl'
    path.write_bytes(b'archived')
    with pytest.raises(FileExistsError):
        heldout.require_new([path])
    assert path.read_bytes()==b'archived'


def test_gate_event_preserves_timestamp_and_history(tmp_path):
    heldout.publish(tmp_path,{'utc':'registered-time','stage':'R','status':'waiting_gpu'})
    assert json.loads((tmp_path/'status.json').read_text())['utc']=='registered-time'
    heldout.publish(tmp_path,{'status':'running','stage':'engine'})
    assert [json.loads(x)['status'] for x in (tmp_path/'events.jsonl').read_text().splitlines()]==['waiting_gpu','running']


def test_task03_alias_and_task12_metadata_are_preserved():
    row=dict(config='oracle|append|tight',task=72,question_index=0,example_id='e',
             prompt_sha1='a'*40,prompt_tokens=2,evidence_complete=True,question_hops=2,
             token_ids=[100,101],answer='New York City',answer_aliases=['NYC'],
             strategy='oracle',mode='append',tier='tight',trajectory_lost=False)
    class Model:
        def generate(self, prompts, **kwargs):
            assert prompts==[{'prompt_token_ids':[100,101]}]
            done=SimpleNamespace(text='Answer: NYC.',token_ids=[123],finish_reason='stop',stop_reason=None)
            return [SimpleNamespace(prompt_token_ids=[100,101],outputs=[done])]
    r,=heldout.generate(Model(),object(),[row],SimpleNamespace(check=lambda:None))
    assert r['f1']==r['em']==1 and r['raw_output']=='Answer: NYC.' and r['pred']=='NYC'
    assert not r['trajectory_lost'] and r['mode']=='append' and r['answer_aliases']==['NYC']


def test_changed_input_spool_is_rejected(tmp_path):
    (tmp_path/'preflight.json').write_text(json.dumps({'status':'passed','requests':{'sha256':'0'*64,'n':5976}}))
    (tmp_path/'requests.jsonl').write_text('{}\n')
    with pytest.raises((RuntimeError,ValueError),match='input spool changed'):
        heldout.verify_inputs(tmp_path,False)

