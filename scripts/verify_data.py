"""Offline verification of all downloaded bytes, records and original fields."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'

def typed(obj, key, typ):
    assert key in obj, f'missing field {key}'
    assert type(obj[key]) is typ, f'{key}: {type(obj[key]).__name__}, expected {typ.__name__}'

def musique(row, labelled, ans):
    for key in ('id', 'question'):
        typed(row, key, str)
    typed(row, 'paragraphs', list)
    indices = set()
    for p in row['paragraphs']:
        typed(p, 'idx', int)
        assert p['idx'] not in indices, 'duplicate paragraph idx'
        indices.add(p['idx'])
        typed(p, 'title', str)
        typed(p, 'paragraph_text', str)
        if labelled:
            typed(p, 'is_supporting', bool)
    if labelled:
        typed(row, 'answer', str)
        typed(row, 'answer_aliases', list)
        assert all(type(x) is str for x in row['answer_aliases'])
        typed(row, 'answerable', bool)
        typed(row, 'question_decomposition', list)
        for h in row['question_decomposition']:
            typed(h, 'id', int)
            typed(h, 'question', str)
            typed(h, 'answer', str)
            assert 'paragraph_support_idx' in h
            idx = h['paragraph_support_idx']
            assert idx is None or type(idx) is int
            if ans:
                assert type(idx) is int and idx in indices
        if ans:
            assert row['answerable']

def hotpot(row, wiki=False):
    for key in ('_id', 'question', 'answer', 'type'):
        typed(row, key, str)
    typed(row, 'supporting_facts', list)
    typed(row, 'context', list)
    for p in row['context']:
        assert type(p) is list and len(p) == 2
        assert type(p[0]) is str and type(p[1]) is list
        assert all(type(s) is str for s in p[1])
    for p in row['supporting_facts']:
        assert type(p) is list and len(p) == 2
        assert type(p[0]) is str and type(p[1]) is int
    if wiki:
        typed(row, 'evidences', list)
        for e in row['evidences']:
            assert type(e) is list and len(e) == 3 and all(type(v) is str for v in e)
    else:
        typed(row, 'level', str)

def main():
    start = time.monotonic()
    manifest_path = RAW / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    report = {'verified_at': datetime.now(timezone.utc).isoformat(), 'files': [], 'errors': []}
    listed = {f['path'] for f in manifest['files']}
    present = {str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_file()} - {'MANIFEST.json'}
    if listed != present:
        report['errors'].append({'unlisted': sorted(present-listed), 'missing': sorted(listed-present)})
    for entry in manifest['files']:
        path = RAW / entry['path']
        try:
            h = hashlib.sha256()
            with path.open('rb') as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            assert h.hexdigest() == entry['sha256'], 'manifest SHA256 mismatch'
            assert path.stat().st_size == entry['size_bytes'], 'size mismatch'
            if entry.get('source_sha256'):
                assert h.hexdigest() == entry['source_sha256'], 'source SHA256 mismatch'
            count = None
            stats = Counter()
            if path.suffix == '.jsonl':
                count = 0
                labelled = '_test.' not in path.name
                ans = 'musique_ans_' in path.name
                for line in path.open():
                    row = json.loads(line)
                    musique(row, labelled, ans)
                    count += 1
                    if labelled:
                        stats['answerable' if row['answerable'] else 'unanswerable'] += 1
                        stats['hops_' + str(len(row['question_decomposition']))] += 1
                        stats['support_idx_null'] += sum(h['paragraph_support_idx'] is None for h in row['question_decomposition'])
            elif path.suffix == '.json':
                row = json.loads(path.read_text())
                if isinstance(row, list):
                    count = len(row)
                    for ex in row:
                        hotpot(ex, wiki='2wikimultihopqa' in entry['path'])
                else:
                    assert type(row) is dict
            else:
                path.read_text(encoding='utf-8')
            assert count == entry['record_count'], 'record count mismatch'
            if entry.get('expected_count') is not None:
                assert count == entry['expected_count'], 'expected count mismatch'
            report['files'].append({'path': entry['path'], 'size_bytes': entry['size_bytes'], 'sha256': h.hexdigest(), 'record_count': count, 'all_records_schema_valid': True, 'statistics': dict(stats)})
        except Exception as e:
            report['errors'].append({'path': entry['path'], 'error': repr(e)})
    report['passed'] = not report['errors']
    report['elapsed_seconds'] = round(time.monotonic()-start, 3)
    target = ROOT / 'eval/acceptance/data_verification.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)

if __name__ == '__main__':
    main()
