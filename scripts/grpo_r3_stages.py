"""Round-three on-policy states, reward sampling and held-out evaluation entry points."""
import argparse
from collections import defaultdict
from pathlib import Path
import os
import random
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import build_grpo_r2_states as building
from scripts import eval_grpo_r2 as evaluation
from scripts import grpo_r3_config as cfg
from scripts import probe_grpo_r2 as sampler
from scripts.sample_sft_subset import require, write_json_exclusive

old = building.old


def select_tasks(population, histories, n=400):
    ids = {t['task'] for t in population}
    require(len(ids) == len(population) and all(t['split'] == 'train' for t in population), 'invalid population')
    excluded = set().union(*histories)
    require(excluded <= ids, 'earlier task outside train')
    return random.Random(cfg.SEED).sample([t for t in population if t['task'] not in excluded], n)


def state_assets():
    previous = cfg.PREVIOUS / 'states/state_checks.json'
    signature = cfg.previous_file(previous)
    r2 = cfg.read(previous)
    ids = set(r2['task_ids'])
    require(len(ids) == r2['n_tasks'] == 400 and r2['scope'] == 'train', 'incomplete Task 11 B source')
    def sample(population, eight, ten):
        return select_tasks(population, (eight, ten, ids))
    with patch.object(building, 'select_tasks', sample):
        tasks, corpus, facts = building.select_assets()
    require(not ids.intersection(t['task'] for t in tasks), 'Task 11 train population overlap')
    return tasks, corpus, {**facts,'task11_source':signature,'task11_overlap':0}


def build_states():
    # Reuse B including its byte-level renderer/encoder and per-cell diagnostics.
    # select_assets is invoked here before installing the callback for building.run.
    assets = state_assets()
    def options(**kwargs):
        kwargs['model_label'] = 'GRPO r2 checkpoint-125'
        return argparse.Namespace(**kwargs)
    with patch.multiple(building, RUN=cfg.RUN, SMALL=cfg.OUT, R2_SEED=cfg.SEED,
                        start_model=cfg.recorded_start, select_assets=lambda:assets, Namespace=options), \
         patch.object(old, 'DEFAULT_BASE', cfg.BASE):
        building.run()


def run_probe():
    from vllm import SamplingParams
    start = cfg.recorded_start()
    args = argparse.Namespace(base_model=cfg.BASE, adapter=cfg.ADAPTER, source_adapter=cfg.ADAPTER,
                              model_label='GRPO r2 checkpoint-125', gpu=1)
    source = old.source_info(args)
    proof = cfg.read(cfg.STATES.with_name('state_checks.json'))
    require(proof['state_file'] == old.file_record(cfg.STATES) and proof['n_tasks'] == 400
            and proof['scope'] == 'train' and proof['available_checks_pass']
            and source['source_adapter_sha256'] == start['adapter_sha256'] == proof['source_adapter_sha256']
            and all(proof[k] == 0 for k in ('heldout_overlap','locked_test_overlap','task08_overlap',
                'task10_overlap','task11_overlap','gold_mismatches','reencode_mismatches','rerender_mismatches')),
            'B split, adapter or state integrity differs')
    offsets = sampler.prior.index_states(cfg.STATES)
    require(sum(map(len, offsets.values())) == proof['n_states'], 'wrong count of B states')
    params = SamplingParams(n=8, temperature=1., top_p=1., top_k=-1, max_tokens=256, seed=cfg.SEED)
    require(params.top_k in (0,-1), 'unexpected top_k truncation')
    require(cfg.FILTER == cfg.FILTER.resolve(), 'redirected C output')
    cfg.FILTER.mkdir()
    prior_path = cfg.PREVIOUS / 'filter/sampling_probe.json'
    ref_record = cfg.previous_file(prior_path)
    previous = cfg.read(prior_path)
    require(previous['mu'] == .5 and previous['seed'] == 20261123, 'incorrect C reference')
    with patch.object(old,'check_gpu',sampler.gpu_gate):
        llm, adapter, engine = old.create_engine(args)
    groups, times, raws = {}, {}, {}
    informative = cfg.FILTER / 'informative.jsonl'
    try:
        with cfg.GROUPS.open('x') as target, informative.open('x') as subset:
            for cell in sampler.CELLS:
                rows = sampler.prior.IndexedRows(cfg.STATES, offsets[cell])
                path = cfg.FILTER / (cell.replace('|','_') + '_samples.jsonl')
                groups[cell], times[cell] = sampler.sample_cell(rows, llm, adapter, engine, params, source,
                                                              path, target, subset)
                raws[cell] = old.file_record(path)
    finally:
        llm.llm_engine.engine_core.shutdown()
    stats = sampler.tables(groups,times,sum(times.values()))
    pooled = stats['pooled']
    require(pooled['n_groups'] == proof['n_states'], 'incomplete C sampling')
    # differences() uses "r2_n" for its first operand. Store the current round's proper name.
    delta = sampler.differences(stats, previous)
    def rename_denominator(value):
        if isinstance(value, dict):
            return {('r3_n' if k == 'r2_n' else k):rename_denominator(v) for k,v in value.items()}
        return value
    write_json_exclusive(cfg.FILTER/'sampling_probe.json', {
        'status':'complete','n_states':pooled['n_groups'],'n_informative':pooled['informative']['count'],
        'seed':cfg.SEED,'mu':.5,'ddof':0,'split':'train','informative_threshold':sampler.prior.THRESHOLD,
        'generation':{'n':8,'temperature':1.0,'top_p':1.0,'top_k':-1,'top_k_actual':params.top_k,
                      'max_tokens':256,'seed':cfg.SEED},'state_file':proof['state_file'],
        'start_receipt':old.file_record(cfg.START),'group_file':old.file_record(cfg.GROUPS),
        'informative_file':old.file_record(informative),'raw_samples_by_cell':raws,
        'engine':engine, **source, **stats, 'references':{'task11_c':{'receipt':ref_record,
        'reward_v2':{k:previous[k] for k in ('cells','pooled')},'delta_r3_minus_r2':rename_denominator(delta)}}})


def baseline(start):
    metrics = cfg.read(Path(start['metrics_path']))
    return evaluation.validate_strategy(metrics, start['strategy']), Path(start['steps_path'])


def evaluate(name):
    # v1.MODEL used by the outer evaluator identifies the base in the *merge receipt*.
    # Frozen Task 05/06 input checks still use their own original model/tokenizer globals.
    api = SimpleNamespace(**{**vars(evaluation.v1),'MODEL':cfg.BASE})
    def against_r2(tasks, corpus, tokenizer, proof):
        step_reference = cfg.ROOT / 'eval_rollouts_grpo_r2' / cfg.ADAPTER.name / 'steps.jsonl'
        return evaluation.formal.compare_step0(tasks, corpus, tokenizer, step_reference,
                                              proof['step0']['identity_and_hashes_sha256'])
    with patch.multiple(evaluation, RUN=cfg.RUN, SMALL=cfg.OUT, RAW=cfg.RAW, MERGED=cfg.MERGED,
                        v1=api, start_model=cfg.recorded_start, start_baseline=baseline,
                        step0_for_task10=against_r2):
        evaluation.run(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    for name in ('states','filter','dataset'):
        subs.add_parser(name)
    subs.add_parser('eval').add_argument('--name', required=True)
    args = parser.parse_args()
    if args.action == 'states':
        build_states()
    elif args.action == 'filter':
        run_probe()
    elif args.action == 'dataset':
        from scripts.grpo_r3_dataset import run
        run()
    else:
        evaluate(args.name)

