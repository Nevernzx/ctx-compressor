"""Starting-policy identity and interactive preflight evidence for Task 11."""
import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import eval_formal_grpo as formal
from scripts import prepare_formal_grpo as old
from scripts import sft_v3_config as v3
from scripts.rollout_model import file_record
from scripts.sample_sft_subset import digest, require, write_json_exclusive
from scripts.sft_run_report import validate_strategy

ROOT = old.ROOT
RUN = ROOT / 'grpo_r2'
OUT = PROJECT / 'eval/grpo_r2'
MERGED = RUN / 'merged_start'
START = OUT / 'start_model.json'
FIXTURE = ROOT / 'grpo_prep/step0_fixture/prompts.jsonl'
BASE = old.BASE
read = v3.read


def start_version(count, n):
    require(type(count) is int and n == 1494 and 0 <= count <= n, 'invalid pooled loss count or denominator')
    return 'v3' if count <= 956 else 'v2'


def resolve_start():
    selected = read(v3.OUT / 'selected_checkpoint.json')
    step, name = selected['checkpoint_step'], selected['strategy']
    require(step in v3.EVAL and name == f'checkpoint-{step}', 'unregistered SFT v3 checkpoint')
    metric = v3.OUT / name / f'{name}_results.json'
    values = validate_strategy(read(metric), name)
    lost = values['combined']['evidence_loss']
    require(lost == selected['evidence_loss'] and step == values['checkpoint_step'], 'v3 selection and metrics differ')
    version = start_version(lost['count'], lost['n'])
    if version == 'v3':
        adapter = v3.checkpoint(step)
        require(str(adapter) == selected['path'], 'v3 adapter directory differs from selection')
        expected_sha = selected['adapter_sha256']
    else:
        old.check_start()
        name, adapter, expected_sha = old.ADAPTER.name, old.ADAPTER, old.SHA
        metric = formal.SFT_RESULTS / f'{name}_results.json'
        values = validate_strategy(formal.load_signed(metric, formal.SFT_METRICS_SHA), name)
        baseline = values['combined']['evidence_loss']
        require((baseline['count'], baseline['n']) == (956,1494), 'SFT v2 reference changed')
    weights = adapter / 'adapter_model.safetensors'
    require(digest(weights) == expected_sha, 'starting adapter checksum changed')
    steps = ROOT / f'eval_rollouts_{version}' / name / 'steps.jsonl'
    require(values['steps_artifact'] == file_record(steps), 'start-model held-out trace differs from its metrics')
    return {'version':version,'strategy':name,'path':str(adapter),'adapter_sha256':expected_sha,
            'metrics_path':str(metric),'metrics_sha256':digest(metric),'steps_path':str(steps),
            'steps_sha256':values['steps_artifact']['sha256'],'sft_v3_selected':selected['strategy'],
            'sft_v3_loss_count':lost['count'],'threshold_loss_count':956,'n_trajectories':1494,
            'rule':'Use SFT v3 for at most 956 lost trajectories out of 1494; otherwise use SFT v2 checkpoint-2000.'}


def compare_fixture(adapter, merged, expected_sha):
    original = read(PROJECT / 'eval/grpo_v2/input_checks.json')
    expected = file_record(FIXTURE)
    require(expected['sha256'] == original['fixture_sha256'] and original['fixture_n'] == 30,
            'Task 08 step-zero fixture changed')
    with FIXTURE.open() as f:
        prompts = [json.loads(line) for line in f]
    require(len(prompts)==30 and all(p['step']==0 and p['split']=='train' for p in prompts),
            'expected 30 train step-zero prompts')
    keys = ('task','mode','budget_tier','prompt_sha256')
    ids = [tuple(p[k] for k in keys) for p in prompts]
    for record in (adapter, merged):
        require(record['source_adapter_sha256'] == expected_sha and record['prompt_file'] == expected
                and record['n'] == 30 and [tuple(r[k] for k in keys) for r in record['rows']] == ids,
                'greedy reference fixture or start adapter differs')
    return {'n_reference':30,
            'text_mismatches':sum(a['raw_output'] != b['raw_output'] for a,b in zip(adapter['rows'],merged['rows'])),
            'token_id_mismatches':sum(a['output_token_ids'] != b['output_token_ids'] for a,b in zip(adapter['rows'],merged['rows']))}


def phase_a_evidence():
    start = read(START)
    require(start == resolve_start(), 'recorded start does not follow the Task 11 selection rule')
    merge = read(MERGED / 'merge_record.json')
    require((merge['base'],merge['adapter'],merge['adapter_sha256']) == (str(BASE), start['path'],start['adapter_sha256'])
            and merge['weights'] and {p.name:digest(p) for p in MERGED.glob('*.safetensors')} == merge['weights'],
            'merged base or saved weight checksum differs')
    paths = [RUN / sub / 'greedy_reference.json' for sub in ('ref_adapter','ref_merged')]
    adapter, merged = map(read, paths)
    require(adapter['base_model'] == str(BASE) and adapter['inference_adapter'] == start['path']
            and merged['base_model'] == str(MERGED) and merged['inference_adapter'] is None,
            'unexpected models in greedy equivalence check')
    compared = compare_fixture(adapter, merged, start['adapter_sha256'])
    if 'against' in merged:
        require(merged['against'] == file_record(paths[0]) and all(merged[k] == compared[k] for k in
                ('text_mismatches','token_id_mismatches')), 'saved mismatch counts differ')
    tests = read(OUT / 'phase_a_tests.json')
    require(tests['status'] == 'passed' and tests['tests'] == tests['passed'] >= read(OUT/'phase_p_tests.json')['tests']
            and all(tests[k] == 0 for k in ('failures','skipped','errors')), 'Phase A CPU regression suite incomplete')
    from scripts.smoke_grpo_r2 import verify_smoke
    smoke = verify_smoke()
    require(smoke['status'] == 'ok' and smoke['optimizer_steps'] == 2 and smoke['n_prompts'] == 8
            and smoke['reward_recheck']['mismatches'] == smoke['reward_recheck']['advantage_mismatches'] == 0,
            'two-step GRPO smoke verification failed')
    return {**compared,'smoke_training':file_record(RUN/'smoke_train/smoke_checks.json'),
            'start_model':file_record(START),'merge':file_record(MERGED/'merge_record.json'),
            'references':[file_record(p) for p in paths],'cpu_tests':file_record(OUT/'phase_a_tests.json')}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('record-start','check'))
    args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and PROJECT == Path('/root/workplace/ctx-compressor'),
            'use the formal CPU environment')
    if args.command == 'record-start':
        write_json_exclusive(START, resolve_start())
    else:
        print(json.dumps(phase_a_evidence(), ensure_ascii=False))

