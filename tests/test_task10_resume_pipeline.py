import json
import sys


from scripts import run_sft_v3_pipeline as pipeline
from scripts import sft_v3_wait_gpu as wait


def test_resumed_stage_waits_before_spawning_and_preserves_old_logs(tmp_path, monkeypatch):
    old = tmp_path / 'original'
    root = tmp_path / 'resume1'
    for directory in (root, old):
        (directory / 'logs').mkdir(parents=True)
    (old / 'resource_monitor.jsonl').write_bytes(b'old monitor\n')
    status = tmp_path / 'pipeline_status_resume1.json'
    for name, path in [('RUN',root), ('LOGS',root/'logs'), ('STATUS',status), ('TRAIN',root/'train')]:
        monkeypatch.setattr(pipeline.controller, name, path)
    for name, path in [('OUT',root), ('ROOT',tmp_path), ('TRAIN_RUN',root/'train')]:
        monkeypatch.setattr(pipeline, name, path)
    (root/'train').mkdir()
    marker = root/'child_started'
    def gpu():
        return {'index':1, 'uuid':pipeline.controller.GPU_UUID, 'memory_used_mib':375, 'utilization_pct':0}
    monkeypatch.setattr(pipeline.controller, 'gpu', gpu)
    real_wait = wait.wait_for_gpu
    observed = []
    def admission(name, publish, reader):
        def emit(event):
            assert not marker.exists() and not (root/'logs'/f'{name}.log').exists()
            publish(event)
            snapshot = json.loads(status.read_bytes())
            assert snapshot['status'] == 'waiting_gpu' and snapshot['stages'][-1]['pid'] is None
            assert snapshot['stage_pid'] is None and snapshot['stages'][0]['status'] == 'reused'
            observed.append(snapshot['last_monitor']['consecutive_free'])
        return real_wait(name, emit, reader=reader, clock=lambda:0, sleep=lambda delay:None)
    monkeypatch.setattr(wait, 'wait_for_gpu', admission)
    workflow = pipeline.ResumedPipeline(root)
    workflow.record['stages'] = [{'stage':'D_data','status':'reused','receipt':{'path':'old.json'}}]
    cmd = [sys.executable,'-B','-c','import pathlib,sys; pathlib.Path(sys.argv[1]).touch()',str(marker)]
    workflow.stage('D_train', cmd, True, {})
    assert marker.is_file() and observed == [1,2,3]
    assert workflow.active['gpu_wait']['consecutive_free'] == 3 and workflow.active['status']=='complete'
    assert workflow.record['status']=='running'
    workflow.stage('E_select', [sys.executable,'-B','-c','pass'], False, {})
    assert observed == [1,2,3]
    assert (old/'resource_monitor.jsonl').read_bytes() == b'old monitor\n'
    assert sorted(p.name for p in (root/'logs').iterdir()) == ['D_train.log','E_select.log']


def test_remaining_stages_match_registered_schedule(tmp_path, monkeypatch):
    workflow = pipeline.ResumedPipeline(tmp_path)
    schedule = []
    monkeypatch.setattr(workflow, 'execute_stage',
                        lambda name, env, script, args, use_gpu, verify: schedule.append((name, use_gpu)))
    monkeypatch.setattr(pipeline, 'read', lambda path: {})
    workflow.run_remaining()
    assert schedule == [('D_train',True),('E_1000',True),('E_2000',True),('E_3000',True),('E_3750',True),
                        ('E_select',False),('F_v3',True),('F_v2',True)]

