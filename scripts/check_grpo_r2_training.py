"""Reconcile every Swift completion and optimizer step with round two's plugin and frozen CLI."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import check_formal_grpo_training as old
from scripts.run_grpo_r2_training import ROOT, CLI_SHA, CONFIG_SHA, signed_json, swift_argv, validate_count
from scripts.run_formal_swift_grpo import write_json
from scripts.sample_sft_subset import digest
from src.grpo_r2_rewards import COMPONENTS, MU, RewardComponents, score_raw

RESULTS = PROJECT / 'eval/grpo_r2'
SCORE_TOLERANCE = 1e-5


def verify_config(saved, baseline, reference_argv, data, n, *, smoke_out=None):
    out = ROOT/'train' if smoke_out is None else smoke_out
    argv = swift_argv(reference_argv, ROOT/'merged_start',data,out,n,lr=1.5e-5,scale_rewards='none',
                      smoke=smoke_out is not None)
    keys = [flag[2:] for flag in reference_argv[1::2]]
    expected = {k:baseline[k] for k in keys}
    expected.update(model=str(ROOT/'merged_start'), dataset=[str(data)], output_dir=str(out),
                    learning_rate=1.5e-5, scale_rewards='none', seed=20261123,data_seed=20261123,
                    external_plugins=['scripts/grpo_r2_swift_plugin.py'],reward_funcs=['ctx_step_reward_v2'],
                    max_steps=n//4)
    for k in ('advantage_estimator','loss_type','kl_in_reward','gradient_accumulation_steps','steps_per_generation'):
        expected[k] = baseline[k]
    if (expected['advantage_estimator'],expected['loss_type'],expected['kl_in_reward']) != ('grpo','grpo',False):
        raise ValueError('Task 08 reference advantage estimator, loss or KL placement changed')
    differences = {k:(v,saved.get(k)) for k,v in expected.items() if k not in saved or saved[k]!=v}
    if differences:
        raise ValueError(f'unexpected resolved round-two configuration: {differences}')
    return argv,expected


def reward_recheck(log, users, steps):
    records = defaultdict(RewardComponents)
    count = mismatches = advantage_errors = 0
    for row in old.json_rows(log):
        columns = [row[k] for k in ('step','prompt','completion','CtxStepRewardV2','advantages')]
        if (not all(isinstance(col,list) for col in columns) or len({len(col) for col in columns})!=1
                or len(columns[0]) % 8 != 0):
            raise ValueError('misaligned Swift completion/reward columns')
        for start in range(0,len(columns[0]),8):
            group_steps, prompts, completions, rewards, advantages = [col[start:start+8] for col in columns]
            if len(set(group_steps))!=1 or len(set(prompts))!=1:
                raise ValueError('Swift group contains multiple steps or prompts')
            step = int(group_steps[0])
            prompt = prompts[0]
            if not prompt.startswith(old.USER_START) or not prompt.endswith(old.RESPONSE_SUFFIX):
                raise ValueError('unknown Swift prompt envelope')
            user = prompt[len(old.USER_START):-len(old.RESPONSE_SUFFIX)]
            if user not in users:
                raise ValueError('logged prompt missing from the training set')
            scores = []
            for text,reward in zip(completions,rewards):
                for eos in ('<|im_end|>','<|endoftext|>'):
                    if text.endswith(eos):
                        text=text[:-len(eos)]
                        break
                score=score_raw(text,users[user])
                records[step].observe(score)
                scores.append(score['reward'])
                mismatches += not math.isclose(float(reward),score['reward'],rel_tol=0,abs_tol=SCORE_TOLERANCE)
            mean=math.fsum(scores)/8
            advantage_errors += sum(not math.isclose(float(actual),r-mean,rel_tol=0,abs_tol=SCORE_TOLERANCE)
                                    for actual,r in zip(advantages,scores))
            count+=8
    if (mismatches or advantage_errors or set(records)!=set(range(1,steps+1))
            or any(a.n<32 for a in records.values())):
        raise ValueError(f'completion audit failed: {mismatches}/{count} reward, {advantage_errors}/{count} '
                         f'advantage mismatches, {len(records)}/{steps} steps')
    return ({'n':count,'mismatches':mismatches,'advantage_mismatches':advantage_errors,
             'step_counts':{s:c.n for s,c in sorted(records.items())},'optimizer_steps':steps,
             'n_planned_completions':32*steps,'extra_logged_completions':count-32*steps,
             'tolerance':SCORE_TOLERANCE,'mu':MU,
             'definition':'Every logged completion including the terminal repeated table; one trailing EOS removed. '
                          'Check reward against the eval parser and advantage against reward minus group mean.'},
            {s:c.result() for s,c in records.items()})


def join_curve(train,steps,from_completions):
    curve=old.curve(train,steps)
    for record,plugin in zip(curve,old.json_rows(train/'optimizer_steps.jsonl')):
        step=record['step']
        if plugin['reward_component_n']!=32:
            raise ValueError(f'expected 32 newly sampled plugin observations at step {step}')
        for k in COMPONENTS:
            if not math.isclose(plugin[k],from_completions[step][k],rel_tol=0,abs_tol=1e-10):
                raise ValueError(f'plugin/Swift component disagreement at step {step}: {k}')
            record[k]=plugin[k]
        record['reward_component_n']=plugin['reward_component_n']
        recomposed = -record['mean_n_lost'] - MU*record['mean_used_per_1000'] - record['invalid_rate']
        if not math.isclose(recomposed,record['reward'],abs_tol=SCORE_TOLERANCE,rel_tol=0):
            raise ValueError(f'mean reward components differ from Swift reward at step {step}')
    return curve


def validate(train,data, *, smoke=False):
    n,users=old.dataset_states(data)
    steps=n//4
    validate_count(n, smoke)
    runtime=json.loads((train/'training_runtime.json').read_bytes())
    launch=json.loads((train/'launch_args.json').read_bytes())
    if (runtime['status']!='ok' or runtime['nonfinite'] or runtime['oom'] or (train/'training_failure.json').exists()
            or launch['mu']!=MU or launch['cuda_visible_devices']!='1' or launch['cuda_device_order']!='PCI_BUS_ID'
            or launch['task08_argv_sha256']!=CLI_SHA or launch['task08_config_sha256']!=CONFIG_SHA):
        raise ValueError('training runtime/identity checks failed')
    argv,config=verify_config(json.loads((train/'args.json').read_bytes()),signed_json('args.json',CONFIG_SHA),
                              signed_json('launch_args.json',CLI_SHA)['argv'],data,n,smoke_out=train if smoke else None)
    actual=list(launch['argv'])
    # Numeric spelling in an env var may differ (1.5e-5 vs 1.5e-05).
    position=argv.index('--learning_rate')+1
    if float(actual[position])!=float(argv[position]):
        raise ValueError('incorrect learning rate')
    actual[position]=argv[position]
    if actual!=argv:
        raise ValueError('registered training CLI changed')
    audit,components=reward_recheck(train/'completions.jsonl',users,steps)
    rows=join_curve(train,steps,components)
    checkpoints,saved=old.stored_checkpoints(train,steps)
    with (train/'gpu1_monitor.csv').open() as stream:
        samples=list(csv.DictReader(stream))
    if not samples:
        raise ValueError('missing GPU 1 telemetry')
    result={'status':'ok','n_prompts':n,'optimizer_steps':steps,'runtime':runtime,'mu':MU,
            'resolved_config':config,'launch_args':launch,'data_sha256':digest(data),
            'reward_recheck':audit,'checkpoints':list(map(str,checkpoints)),'checkpoint_steps':saved,
            'adapter_sha256':{p.name:digest(p/'adapter_model.safetensors') for p in checkpoints},
            'n_gpu_samples':len(samples),'peak_gpu1_used_mib':max(int(s['memory_used_mib']) for s in samples),
            'peak_torch_allocated_bytes':max(r['torch_peak_allocated_bytes'] for r in rows),
            'peak_torch_reserved_bytes':max(r['torch_peak_reserved_bytes'] for r in rows)}
    return rows,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--train-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.train_dir.resolve()!=ROOT/'train' or args.data.resolve()!=ROOT/'data/train.jsonl':
        raise ValueError('checker requires round-two inputs')
    target=RESULTS/'training_checks.json'
    csv_path=RESULTS/'training_curve.csv'
    if target.exists() or csv_path.exists():
        raise FileExistsError('round-two training audit output exists')
    rows,result=validate(args.train_dir,args.data)
    RESULTS.mkdir(exist_ok=True,parents=True)
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result.update(curve=str(csv_path),curve_sha256=digest(csv_path))
    write_json(target,result)
    print(json.dumps({'status':'ok','n_prompts':result['n_prompts'],'optimizer_steps':result['optimizer_steps'],
                      'reward_records':result['reward_recheck']['n']}))


if __name__=='__main__':
    main()

