"""Build the seeded round-three dataset, prioritizing informative changes to gold loss or validity."""
import json
from pathlib import Path
import random

from scripts import grpo_r3_config as cfg
from scripts import make_formal_grpo_dataset as formal
from scripts import make_grpo_r2_dataset as previous
from scripts.build_grpo_states import file_record, lines
from scripts.sample_sft_subset import require, write_json_exclusive

CELLS = formal.CELLS
LAYERS = ('gold_varying','token_only_varying')


def plan(pools, target, seed):
    """Apply sorted-cell quotas and redistribution to each layer in priority order."""
    rng = random.Random(seed)
    chosen = []
    counts = {}
    for layer in LAYERS:
        available = {c:len(pools[layer][c]) for c in CELLS}
        take = formal.quotas(available, target - len(chosen))
        counts[layer] = take
        for cell in CELLS:
            chosen.extend(rng.sample(pools[layer][cell], take[cell]))
    rng.shuffle(chosen)
    require(len(chosen) == len(set(chosen)), 'same state selected twice')
    return chosen, counts


def group_layers(groups_path, signals):
    pools = {layer:{cell:[] for cell in CELLS} for layer in LAYERS}
    for group in lines(groups_path):
        key = formal.state_key(group)
        cell, _, informative = signals[key]
        parts = group['reward_parts']
        require(len(parts) == 8, 'missing per-sample components')
        gold = len({(p['n_lost'],p['invalid']) for p in parts}) > 1
        token = not gold and len({p['used_after'] for p in parts}) > 1
        require((gold, token) == (group['gold_varying'], group['token_only_varying']),
                f'group components and flags disagree for {key}')
        if informative:
            require(gold or token, f'informative group without reward component change: {key}')
            pools[LAYERS[0] if gold else LAYERS[1]][cell].append(key)
    return pools


def build(pool, groups, informative, destination, orm, target=2000):
    require(not destination.exists(), f'existing destination: {destination}')
    greedy = previous.greedy_rewards(groups, informative)
    signals, available = formal.filter_signals(groups)
    pools = group_layers(groups, signals)
    n_informative = sum(available.values())
    keys, counts = plan(pools, target, cfg.DATA_SEED)
    require(set(greedy) == {k for k,v in signals.items() if v[2]}
            and len(keys) == min(n_informative, target), 'informative subset or selected count differs')
    selected = set(keys)
    rows = {}
    for line in formal.candidate_lines(pool, signals):
        row = json.loads(line)
        key = formal.state_key(row)
        if key in selected:
            require(row['split'] == 'train', 'selected a non-training row')
            rows[key] = {'messages':row['messages'],'reward_state_json':row['reward_state_json'],
                         'trajectory_id':key[0],'state_step':key[1],'cell':formal.cell(row)}
    require(selected == rows.keys(), 'missing selected input state')
    with destination.open('x', encoding='utf-8') as stream:
        for key in keys:
            stream.write(json.dumps(rows[key],ensure_ascii=False) + '\n')
    verified = previous.verify_plugin(destination, pool, greedy, orm)
    n = len(keys)
    require(verified == n, 'plugin did not check all selected rows')
    total_gold = sum(counts[LAYERS[0]].values())
    return {'n':n,'n_states':len(signals),'n_informative':n_informative,'target_n':target,
            'minimum_informative':500,'status':'stopped_insufficient_signal' if n_informative < 500 else 'complete',
            'per_cell_available':dict(available),'per_cell_counts':{c:sum(counts[l][c] for l in LAYERS) for c in CELLS},
            'per_layer_available':{l:{c:len(v) for c,v in by_cell.items()} for l,by_cell in pools.items()},
            'per_layer_counts':counts,'gold_varying':{'count':total_gold,'n':n,'rate':total_gold/n if n else None},
            'seed':cfg.DATA_SEED,'rows_checked':verified,'plugin_greedy_checked':verified,'heldout_rows':0,
            'plugin_greedy_mismatches':0,'user_text_mismatches':0,'output':file_record(destination),
            'shortfall_from_target':target-n,'informative_std_threshold':formal.THRESHOLD,'mu':0.5}


def run():
    from scripts import probe_grpo_r2 as filtering
    proof = cfg.read(cfg.FILTER / 'sampling_probe.json')
    require(proof['status'] == 'complete' and proof['split'] == 'train' and proof['mu'] == .5
            and proof['generation']['seed'] == cfg.SEED
            and proof['informative_threshold'] == formal.THRESHOLD, 'filter configuration differs')
    pool = filtering.checked_file(proof['state_file'], cfg.STATES)
    groups = filtering.checked_file(proof['group_file'], cfg.GROUPS)
    subset = filtering.checked_file(proof['informative_file'], cfg.FILTER/'informative.jsonl')
    require(cfg.DATA.parent.resolve() == cfg.DATA.parent, 'redirected dataset directory')
    cfg.DATA.parent.mkdir()
    summary = build(pool, groups, subset, cfg.DATA, previous.make_orm())
    require(summary['n_states'] == proof['n_states'] and
            summary['n_informative'] == proof['pooled']['informative']['count'] == proof['n_informative'],
            'incomplete C/D coverage')
    write_json_exclusive(cfg.DATA.with_name('dataset_stats.json'), {
        **summary, 'source_states':proof['state_file'], 'source_groups':proof['group_file'],
        'source_informative':proof['informative_file']})

