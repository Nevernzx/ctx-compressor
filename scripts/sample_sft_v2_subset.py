#!/usr/bin/env python3
"""Reapply the frozen Task 05 sampling procedure and compare the sampled rows in order."""
import json
import os
from itertools import zip_longest
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import sample_sft_subset as old

ROOT = old.EXTERNAL
RUN = ROOT / 'runs/sft_20k_v2_20261118'
OUT = PROJECT / 'eval/sft_run_v2'
V1 = PROJECT / 'eval/sft_run'


def verify_rows(current, previous, chosen):
    counts = {'rows': 0, 'append': 0, 'reselect_step0': 0}
    with current.open('rb') as new_stream, previous.open('rb') as old_stream:
        for new, original, entry in zip_longest(new_stream, old_stream, chosen):
            old.require(None not in (new, original, entry), 'subset length mismatch')
            key = old.metadata_identity(entry)
            old.require(old.metadata_identity(json.loads(new)) == key == old.metadata_identity(json.loads(original)),
                        f'row/selected-ID mismatch: {key}')
            counts['rows'] += 1
            if key[2] == 'append' or key[1] == 0:
                old.require(new == original, f'fixed row bytes changed: {key}')
                counts['append' if key[2] == 'append' else 'reselect_step0'] += 1
    return counts


def main():
    old.require(PROJECT == Path('/root/workplace/ctx-compressor')
                and Path(sys.prefix) == ROOT / 'envs/train'
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'run on 5880, train Python, CPU only')
    stats_path = OUT / 'data_v2_stats.json'
    v2_stats = json.loads(stats_path.read_bytes())
    previous = json.loads((V1 / 'sample_manifest.json').read_bytes())
    id_record = previous['files']['sample_ids']
    old.require(old.digest(id_record['path']) == id_record['sha256'], 'Task 05 IDs SHA256 mismatch')
    previous_ids = json.loads(Path(id_record['path']).read_bytes())
    outputs, selections, checks, stats, sources = {}, {}, {}, {}, {}
    RUN.mkdir(exist_ok=True, parents=True)
    for split, key, id_key, task_count in [('train', 'train', 'train', 4735),
                                           ('heldout', 'eval_loss', 'heldout_eval_loss', 249)]:
        source = v2_stats['files'][split]
        old.require(Path(source['path']) == ROOT / f'data/sft_v2/{split}.jsonl', 'unexpected v2 data path')
        old.require(source['lines'] == previous['sources'][split]['lines'], 'wrong source count')
        items = old.scan_source(source, split, task_count)
        lines = old.select_training(items) if split == 'train' else old.select_eval_loss(items)
        chosen = [items[index - 1] for index in lines]
        ids = [(row['source_line_1based'], old.metadata_identity(row)) for row in chosen]
        old.require(ids == [(row['source_line_1based'], old.metadata_identity(row))
                            for row in previous_ids[id_key]], f'{split}: selected IDs or ordering differ')
        ref = previous['files'][key]
        original = Path(ref['path'])
        old.require(old.digest(original) == ref['sha256'], f'{split}: old subset changed')
        outputs[key] = old.copy_selected(source, lines, RUN / original.name)
        checks[key] = verify_rows(Path(outputs[key]['path']), original, chosen)
        stats[key] = old.statistics(chosen)
        selections[id_key] = chosen
        sources[split] = {**source, 'full_sha256_verified_twice': True, 'task_count': task_count}
    ids = {'schema_version': 1, 'protocol': previous['protocol'], 'sources': sources, **selections}
    outputs['sample_ids'] = old.write_json_exclusive(RUN / 'sample_ids.json', ids)
    manifest = {'task': 'TASK_06', 'protocol': previous['protocol'], 'sources': sources, 'files': outputs,
                'statistics': stats, 'validation': checks, 'source_stats_sha256': old.digest(stats_path),
                'task05_sample_ids_sha256': id_record['sha256'], 'ordered_id_lists_and_source_lines_identical': True,
                'selection_code_sha256': old.digest(old.__file__), 'seed': old.SEED}
    old.write_json_exclusive(OUT / 'sample_manifest.json', manifest)
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
