"""Prepare and audit the two-step Task 11 training smoke using sealed Task 08 train rows."""
import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import check_grpo_r2_training as check
from scripts.run_grpo_r2_training import ROOT, smoke_output
from scripts.rollout_model import file_record
from scripts.sample_sft_subset import require, write_json_exclusive
from src.grpo_r2_rewards import COMPONENTS, score_raw
from src.grpo_reward import STATE_FIELDS

SMOKE = ROOT / 'smoke_train'
SOURCE = ROOT.parent / 'grpo_v2/data/train.jsonl'
SOURCE_SHA = '7c752b2dac997d059de77c8fc9df7de3ebc63e8e5c99a3c8c5df6b7844d23495'
STATS = SOURCE.with_name('dataset_stats.json')
START = PROJECT/'eval/grpo_r2/start_model.json'
MERGE = ROOT/'merged_start/merge_record.json'
DATA = SMOKE/'train.jsonl'
INPUTS = SMOKE/'inputs.json'
RECEIPT = SMOKE/'smoke_checks.json'


def read(p):
    return json.loads(p.read_bytes())


def source_rows():
    signed = file_record(SOURCE)
    proof = read(STATS)
    require(signed['sha256'] == SOURCE_SHA and proof['output'] == signed and proof['heldout_rows'] == 0
            and proof['n'] == proof['rows_checked'] == 2000, 'archived Task 08 train dataset changed')
    with SOURCE.open('rb') as stream:
        lines = [next(stream) for _ in range(8)]
    ids, prompts = [], set()
    for line in lines:
        row = json.loads(line)
        state = json.loads(row['reward_state_json'])
        require(set(STATE_FIELDS) | {'prompt_token_ids_tail'} <= set(state)
                and len(state['prompt_token_ids_tail']) >= 4, 'reward v2 input missing')
        require(set(row) == {'trajectory_id','state_step','cell','messages','reward_state_json'}
                and len(row['messages']) == 1 and row['messages'][0]['role'] == 'user',
                'Task 08 row format changed')
        for raw in ('{"add":[],"drop":[]}', 'invalid'):
            score = score_raw(raw, state)
            require(score['invalid_penalty'] == int(raw == 'invalid'), 'reward guard differs for smoke input')
        ids.append([row['trajectory_id'], row['state_step']])
        prompts.add(row['messages'][0]['content'])
    require(len(prompts) == len({tuple(x) for x in ids}) == 8, 'smoke prompts must be distinct')
    return lines, {'source':signed,'source_stats':file_record(STATS),'source_rows_one_based':list(range(1,9)),
                   'identities':ids,'n_prompts':8,'selection':'first eight rows of the frozen shuffled Task 08 train dataset'}


def verify_data():
    lines, provenance = source_rows()
    require(DATA.read_bytes() == b''.join(lines), 'smoke dataset differs from signed source rows')
    expected = {**provenance,'output':file_record(DATA)}
    require(read(INPUTS) == expected, 'smoke data receipt changed')
    return expected


def prepare():
    lines, provenance = source_rows()
    require(SMOKE.resolve() == SMOKE, 'smoke directory redirected')
    SMOKE.mkdir(exist_ok=True)
    require(not any(os.path.lexists(p) for p in (INPUTS, DATA)), 'smoke dataset already exists')
    with DATA.open('xb') as f:
        f.writelines(lines)
    write_json_exclusive(INPUTS, {**provenance, 'output':file_record(DATA)})
    return verify_data()


def check_completion_tables(path):
    """Swift may print the last completion table a second time in its final log hook."""
    tables = list(check.old.json_rows(path))
    require(len(tables) in (2,3), 'expected two batches plus at most one terminal copy')
    if len(tables) == 3:
        require(tables[-1] == tables[-2], 'terminal Swift table differs from last optimizer step')
    nonzero, varying = 0, 0
    for step, table in enumerate(tables[:2],1):
        require(len(table['step']) == 32 and all(int(v) == step for v in table['step']),
                'expected exactly 32 new completions per optimizer step')
        values = list(map(float, table['advantages']))
        nonzero += sum(abs(v) > 1e-5 for v in values)
        rewards = list(map(float,table['CtxStepRewardV2']))
        varying += sum(len(set(rewards[i:i+8])) > 1 for i in range(0,32,8))
    return {'new_completions':64,'new_completions_per_step':[32,32],
            'terminal_duplicate_rows':32*(len(tables)-2),'informative_groups':varying,
            'n_groups':8,'nonzero_advantages':nonzero}


def checked_result(attempt):
    output = smoke_output(attempt)
    require(output.resolve() == output, 'redirected smoke training output')
    data = verify_data()
    start, merge = read(START),read(MERGE)
    require(start['version'] == 'v3' and start['strategy'] == 'checkpoint-3000'
            and start['adapter_sha256'] == merge['adapter_sha256']
            and start['path'] == merge['adapter'], 'smoke model is not the selected merged start')
    weights = {p.name:file_record(p)['sha256'] for p in MERGE.parent.glob('*.safetensors')}
    require(weights and weights == merge['weights'], 'merged starting weights changed')
    rows, report = check.validate(output, DATA, smoke=True)
    require(report['optimizer_steps'] == 2 and report['checkpoint_steps'] == [2] and report['n_prompts'] == 8,
            'incomplete two-step optimizer run')
    require(all(r['step'] == i and r['reward_component_n'] == 32 and all(k in r for k in COMPONENTS)
                for i,r in enumerate(rows, 1)) and len(rows) == 2, 'unaligned per-step reward components')
    log = output/'completions.jsonl'
    tables = check_completion_tables(log)
    require(report['reward_recheck']['n'] == tables['new_completions'] + tables['terminal_duplicate_rows']
            and report['reward_recheck']['advantage_mismatches'] == report['reward_recheck']['mismatches'] == 0,
            'incomplete logged completion audit')
    files = [START, MERGE, INPUTS, DATA] + [output/name for name in (
        'args.json','launch_args.json','training_runtime.json','optimizer_steps.jsonl','logging.jsonl',
        'completions.jsonl','gpu1_monitor.csv','checkpoint-2/adapter_config.json','checkpoint-2/trainer_state.json')]
    return {**report,'attempt':int(attempt),'input_provenance':data,'step_metrics':rows,'logged_tables':tables,
            'source_files':{str(p):file_record(p) for p in files}}


def verify_smoke():
    saved = read(RECEIPT)
    result = json.loads(json.dumps(checked_result(saved['attempt']), allow_nan=False))
    require(result == saved and saved['status'] == 'ok', 'smoke acceptance receipt differs from original artifacts')
    return saved


def launch(attempt):
    verify_data()
    require(not os.path.lexists(RECEIPT), 'a successful smoke already exists')
    output = smoke_output(attempt)
    require(output.parent.resolve() == output.parent, 'redirected smoke attempt')
    output.parent.mkdir(exist_ok=False)
    env = dict(os.environ, MERGED=str(MERGE.parent), DATA=str(DATA), OUT=str(output), MAX_STEPS='2',
               GRPO_R2_SMOKE_ATTEMPT=str(attempt), GRPO_SEED='20261123', LR='1.5e-5', SCALE_REWARDS='none',
               GRPO_CPU_ONLY='0',GRPO_GPU='1')
    os.execve('/bin/bash',['bash',str(PROJECT/'scripts/train_grpo_r2.sh')],env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    for cmd in ('train','check'):
        sub.add_parser(cmd).add_argument('--attempt',type=int,required=True)
    sub.add_parser('verify')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare()))
    elif args.command == 'train':
        launch(args.attempt)
    elif args.command == 'check':
        value = checked_result(args.attempt)
        write_json_exclusive(RECEIPT, value)
        print(json.dumps({'receipt':str(RECEIPT),'reward_recheck':value['reward_recheck']}))
    else:
        print(json.dumps({'status':verify_smoke()['status'],'receipt':str(RECEIPT)}))


if __name__ == '__main__':
    main()
