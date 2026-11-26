"""Train and audit round three with the frozen Task 11 optimizer and component logger."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import check_grpo_r2_training as audit
from scripts import grpo_r3_config as cfg
from scripts import run_grpo_r2_training as training
from scripts.sample_sft_subset import digest, require, write_json_exclusive


def argv(reference, data, out, n, smoke):
    args = training.swift_argv(reference,cfg.MERGED,data,out,n,lr=1.5e-5,scale_rewards='none',
                               seed=cfg.SEED,smoke=smoke)
    args[0] = 'scripts/grpo_r3_training.py'
    return args


def check_args(saved, _baseline, reference, data, n, *, smoke_out=None):
    output = smoke_out or cfg.TRAIN
    original = cfg.R2/'training_checks.json'
    cfg.previous_file(original)
    expected = dict(cfg.read(original)['resolved_config'],model=str(cfg.MERGED),dataset=[str(data)],
                    output_dir=str(output),max_steps=n//4,seed=cfg.SEED,data_seed=cfg.SEED)
    differences = {k:(saved.get(k),v) for k,v in expected.items() if k not in saved or saved[k]!=v}
    require(not differences, f'round-three resolved config differs: {differences}')
    return argv(reference,data,output,n,smoke_out is not None), expected


def validate(train, data, *, smoke=False):
    with patch.object(audit,'verify_config',check_args):
        return audit.validate(train,data,smoke=smoke)


def train():
    from scripts.run_formal_grpo_pipeline import gpu
    model,data,out = (Path(os.environ[k]) for k in ('MERGED','DATA','OUT'))
    attempt = os.environ.get('GRPO_R3_SMOKE_ATTEMPT')
    smoke = attempt is not None
    require(not smoke or (attempt.isascii() and attempt.isdecimal() and str(int(attempt)) == attempt
                          and int(attempt)>0), 'invalid smoke attempt')
    require(Path(sys.prefix).resolve() == cfg.ROOT/'envs/grpo' and os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
            and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID' and os.environ.get('HF_HUB_OFFLINE') == '1',
            'expected round-three offline environment on GPU 1')
    require(model == cfg.MERGED and data == (cfg.RUN/'smoke_train/train.jsonl' if smoke else cfg.DATA)
            and out == (cfg.RUN/'smoke_train'/f'attempt-{attempt}'/'train' if smoke else cfg.TRAIN)
            and all(p == p.resolve() for p in (model,data,out)) and not os.path.lexists(out),
            'unexpected model, data or output path')
    with data.open('rb') as stream:
        n = sum(1 for _ in stream)
    training.validate_count(n,smoke)
    require(int(os.environ['MAX_STEPS']) == n//4, 'registered optimizer steps differ')
    args = argv(training.signed_json('launch_args.json',training.CLI_SHA)['argv'],data,out,n,smoke)
    require(gpu()['memory_used_mib'] < 2048, 'GPU 1 occupied after gate')
    out.mkdir()
    training.train(out, args)


def check():
    target = cfg.OUT / 'training_checks.json'
    curve = cfg.OUT / 'training_curve.csv'
    require(not any(os.path.lexists(path) for path in (target,curve)), 'formal training audit already exists')
    rows, result = validate(cfg.TRAIN, cfg.DATA)
    with curve.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json_exclusive(target,{**result,'curve':str(curve),'curve_sha256':digest(curve)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('train','check'))
    args = parser.parse_args()
    if args.action == 'check':
        check()
    else:
        train()

