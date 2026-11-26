#!/usr/bin/env python3
"""One-shot Task 13 merge, greedy equivalence, and two-step GRPO smoke checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import build_grpo_states as protocol
from scripts import merge_sft_lora as merge
from scripts import smoke_grpo_r2 as previous_smoke
from scripts.build_formal_grpo_states import gpu_gate
from scripts.grpo_r2_start import FIXTURE, compare_fixture
from scripts.grpo_r3_config import ROOT, RUN, OUT, BASE, MERGED, START, ADAPTER, previous_manifest, resolve_start
from scripts.rollout_model import file_record
from scripts.run_formal_grpo_pipeline import now
from scripts.sample_sft_subset import digest, require, write_json_exclusive
from scripts.sft_v3_wait_gpu import wait_for_gpu
from src.grpo_r2_rewards import COMPONENTS

SMOKE=RUN/'smoke_train'
DATA=SMOKE/'train.jsonl'
SMOKE_PROOF=SMOKE/'smoke_checks.json'
def read(path):
    return json.loads(path.read_bytes())


def smoke_source():
    signed=previous_manifest()
    source=ROOT/'grpo_r2/data/train.jsonl'
    stats=source.with_name('dataset_stats.json')
    require(digest(stats)==signed[str(stats)],'round-two data receipt changed')
    with patch.multiple(previous_smoke,SOURCE=source,STATS=stats,SOURCE_SHA=signed[str(source)]):
        lines,proof=previous_smoke.source_rows()
    proof['selection']='first eight rows of the frozen shuffled Task 11 dataset'
    return lines,proof


def check_reader_delivery():
    directory=PROJECT/'eval/reader_heldout'
    manifest=directory/'TASK12_SHA256SUMS'
    hashes={path:sha for sha,path in (line.split('  ',1) for line in manifest.read_text().splitlines())}
    for name in ('reader_heldout_report.md','reader_heldout_results.json','run_config.json'):
        p=directory/name
        require(hashes[str(p.relative_to(PROJECT))]==digest(p),'Task 12 report or run config changed')
    require(read(directory/'run_config.json')['status']=='complete','Task 12 reader incomplete')


def verified_merge():
    saved=read(MERGED/'merge_record.json')
    require(saved['base']==str(BASE) and saved['adapter']==str(ADAPTER)
            and saved['adapter_sha256']==read(START)['adapter_sha256'],'unexpected merge sources')
    weights={p.name:digest(p) for p in MERGED.glob('*.safetensors')}
    require(weights and weights==saved['weights'],'merged weights changed')
    return saved


def initial():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','merge requires CPU')
    check_reader_delivery()
    require(MERGED.parent==RUN and MERGED.resolve()==MERGED and not os.path.lexists(MERGED),'merge output exists')
    signed=previous_manifest()
    for p in [*BASE.glob('*.safetensors'),BASE/'config.json',ADAPTER/'adapter_model.safetensors',ADAPTER/'adapter_config.json']:
        require(digest(p)==signed[str(p)],f'round-two weight/config changed: {p}')
    start=resolve_start()
    require(start['path']==str(ADAPTER) and start['adapter_sha256']==signed[str(ADAPTER/'adapter_model.safetensors')],'wrong start model')
    write_json_exclusive(START,start)
    with patch.object(merge,'PREP',RUN), patch.object(sys,'argv',[
            'merge_sft_lora.py','--base',str(BASE),'--adapter',str(ADAPTER),'--out',str(MERGED)]):
        merge.main()
    verified_merge()
    lines,proof=smoke_source()
    SMOKE.mkdir(exist_ok=True)
    with DATA.open('xb') as f:
        f.writelines(lines)
    write_json_exclusive(SMOKE/'inputs.json',{**proof,'output':file_record(DATA)})


def references():
    paths=[RUN/sub/'greedy_reference.json' for sub in ('ref_adapter','ref_merged')]
    a,b=map(read,paths)
    require(a['base_model']==str(BASE) and a['inference_adapter']==str(ADAPTER)
            and b['base_model']==str(MERGED) and b['inference_adapter'] is None,'unexpected reference models')
    counts=compare_fixture(a,b,read(START)['adapter_sha256'])
    require(b['against']==file_record(paths[0]) and all(b[k]==counts[k] for k in
            ('text_mismatches','token_id_mismatches')),'reference mismatch counts changed')
    return dict(counts,references=[file_record(p) for p in paths])


def reference(stage):
    is_merged=stage=='ref_merged'
    output=RUN/stage
    output.mkdir()
    options=argparse.Namespace(output_dir=output,prompts=FIXTURE,compare=RUN/'ref_adapter/greedy_reference.json' if is_merged else None,
            base_model=MERGED if is_merged else BASE,source_adapter=ADAPTER,adapter=None if is_merged else ADAPTER,
            gpu=1,model_label='GRPO r2 checkpoint-125')
    engines=[]
    original=protocol.create_engine
    def tracked(args):
        result=original(args)
        engines.append(result[0])
        return result
    try:
        with patch.multiple(protocol,create_engine=tracked,check_gpu=gpu_gate):
            protocol.reference(options)
    finally:
        for engine in engines:
            engine.llm_engine.engine_core.shutdown()
    return references() if is_merged else {'reference':file_record(output/'greedy_reference.json')}


def smoke_evidence():
    from scripts.grpo_r3_training import validate
    lines,proof=smoke_source()
    require(DATA.read_bytes()==b''.join(lines) and read(SMOKE/'inputs.json')==dict(proof,output=file_record(DATA)),
            'smoke dataset lineage changed')
    out=SMOKE/'attempt-1/train'
    rows,report=validate(out,DATA,smoke=True)
    tables=previous_smoke.check_completion_tables(out/'completions.jsonl')
    require(report['optimizer_steps']==2 and report['n_prompts']==8 and report['status']=='ok'
            and report['checkpoint_steps']==[2] and len(rows)==2,'expected two successful optimizer steps')
    for i,row in enumerate(rows,1):
        require(row['step']==i and row['reward_component_n']==32 and all(k in row for k in COMPONENTS),'missing per-step reward components')
    check=report['reward_recheck']
    require(check['n']==64+tables['terminal_duplicate_rows'] and check['mismatches']==check['advantage_mismatches']==0,
            'reward/advantage recheck differs')
    return json.loads(json.dumps(dict(report,step_metrics=rows,logged_tables=tables,input=proof),allow_nan=False))


def evidence():
    require(read(START)==resolve_start(),'start model changed')
    verified_merge()
    ref=references()
    require(read(SMOKE_PROOF)==smoke_evidence(),'smoke receipt differs from raw evidence')
    test=OUT/'phase_p_tests.json'
    t=read(test)
    require(t['status']=='passed' and t['passed']==t['tests'] and t['tests']>=705
            and not any(t[k] for k in ('failures','skipped','errors')),'CPU suite incomplete')
    return dict(ref,smoke_training=file_record(SMOKE_PROOF),start_model=file_record(START),
                merge=file_record(MERGED/'merge_record.json'),cpu_tests=file_record(test))


def gated(stage):
    require(Path(sys.prefix).resolve()==ROOT/'envs/grpo' and os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
            and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID' and os.environ.get('HF_HUB_OFFLINE')=='1',
            'use offline GRPO environment on GPU 1')
    require(read(START)==resolve_start(),'start model changed')
    verified_merge()
    if stage=='smoke':
        references()
    target=RUN/'phase_a'/('gate_'+stage)
    target.mkdir(parents=True)
    status=target/'status.json'
    def publish(row):
        record={'utc':now(),'pid':os.getpid(),'stage':stage,**row}
        with (target/'events.jsonl').open('a') as f:
            print(json.dumps(record),file=f)
        draft=target/'status.next'
        write_json_exclusive(draft,record)
        os.replace(draft,status)
    try:
        admitted=wait_for_gpu(stage,publish)
        publish({'status':'running','gpu':admitted})
        if stage=='smoke':
            env=dict(os.environ,MERGED=str(MERGED),DATA=str(DATA),OUT=str(SMOKE/'attempt-1/train'),
                     MAX_STEPS='2',GRPO_R3_SMOKE_ATTEMPT='1',GRPO_SEED='20261125',SCALE_REWARDS='none',LR='1.5e-5',MU='0.5')
            subprocess.run(['/bin/bash','-c',
                'source scripts/grpo_r3_environment.sh && exec "$GRPO_ENV/bin/python" -B -u scripts/grpo_r3_training.py train'],
                env=env,check=True)
            result=smoke_evidence()
            write_json_exclusive(SMOKE_PROOF,result)
            proof={'smoke':file_record(SMOKE_PROOF)}
        else:
            proof=reference(stage)
        publish({'status':'complete',**proof})
    except BaseException as exc:
        publish({'status':'failed','error':repr(exc)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('merge','ref_adapter','ref_merged','smoke','verify'))
    args=parser.parse_args()
    if args.stage=='merge':
        initial()
    elif args.stage=='verify':
        print(json.dumps(evidence()))
    else:
        gated(args.stage)

