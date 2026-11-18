"""Download only explicit dataset/tokenizer files on 5880, with no proxy."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path('/root/workplace/ctx-compressor')
RAW = ROOT / 'data/raw'

def utc():
    return datetime.now(timezone.utc).isoformat()

def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))

def get_json(url):
    with opener().open(url, timeout=60) as r:
        return json.load(r)

def inspect_file(path):
    count = None
    structure = None
    if path.suffix == '.jsonl':
        count = 0
        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                if count == 0:
                    structure = schema(obj)
                count += 1
    elif path.suffix == '.json':
        with path.open() as f:
            obj = json.load(f)
        if isinstance(obj, list):
            count = len(obj)
            structure = schema(obj[0]) if obj else None
        else:
            structure = {'type': type(obj).__name__}
    else:
        path.read_text(encoding='utf-8')
    return count, structure

def schema(obj):
    if isinstance(obj, dict):
        return {k: schema(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return []
        # Collections of like records need one example, but tuple positions
        # and scalar sequences must retain every element's type.
        if isinstance(obj[0], (dict, list)) and all(type(v) is type(obj[0]) for v in obj):
            return [schema(obj[0])]
        return [schema(v) for v in obj]
    return type(obj).__name__

def download(spec):
    start = time.monotonic()
    url, rel = spec['url'], spec['path']
    path = RAW / rel
    if not path.resolve().is_relative_to(RAW.resolve()):
        raise ValueError('Path escapes raw data root')
    if '.safetensors' in rel or rel.endswith(('.bin', '.pt', '.pth')):
        raise ValueError('Weights are forbidden')
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    digest = hashlib.sha256()
    size = 0
    headers = {'User-Agent': 'ctx-compressor-data-audit/1.0'}
    if spec.get('accept'):
        headers['Accept'] = spec['accept']
    request = urllib.request.Request(url, headers=headers)
    with opener().open(request, timeout=90) as response, part.open('wb') as out:
        expected_size = response.headers.get('Content-Length')
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if expected_size and size != int(expected_size):
        raise ValueError(f'Truncated response {size} != {expected_size}')
    sha = digest.hexdigest()
    if spec.get('sha256') and sha != spec['sha256']:
        raise ValueError('Source SHA256 mismatch')
    if spec.get('size') and size != spec['size']:
        raise ValueError('Source size mismatch')
    part.replace(path)
    count, structure = inspect_file(path)
    if spec.get('expected_count') is not None and count != spec['expected_count']:
        raise ValueError(f'Count mismatch {count} != {spec["expected_count"]}')
    result = {'path': rel, 'source_url': url, 'size_bytes': size,
              'sha256': sha, 'record_count': count, 'expected_count': spec.get('expected_count'),
              'downloaded_at': utc(), 'download_seconds': round(time.monotonic() - start, 3),
              'source_sha256': spec.get('sha256'), 'source_size_bytes': spec.get('size'),
              'validation': '完整解析、大小、来源SHA256' if spec.get('sha256') else '完整解析、大小；数据条数（如适用）；无官方SHA256',
              'sample_field_structure': structure, 'source_kind': spec.get('source_kind', 'mirror')}
    print(json.dumps({'complete': rel, 'bytes': size, 'count': count}, ensure_ascii=False), flush=True)
    return result

def main():
    if not ROOT.is_dir() or Path.cwd().resolve() != ROOT:
        raise SystemExit('Run from /root/workplace/ctx-compressor on 5880 only')
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', action='store_true')
    parser.add_argument('--musique', action='store_true')
    parser.add_argument('--spec')
    args = parser.parse_args()
    specs = []
    if args.tokenizer:
        api = 'https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-1.7B/repo/files?Revision=master&Recursive=true'
        files = get_json(api)['Data']['Files']
        allow = {'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'config.json', 'LICENSE'}
        for f in files:
            if f['Name'] in allow:
                specs.append({'url': 'https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-1.7B/repo?Revision=' + f['Revision'] + '&FilePath=' + urllib.parse.quote(f['Path']), 'path': 'qwen3_tokenizer/' + f['Name'], 'sha256': f['Sha256'], 'size': f['Size'], 'source_kind': 'official'})
    if args.spec:
        specs.extend(json.loads(Path(args.spec).read_text()))
    if args.musique:
        api = 'https://www.modelscope.cn/api/v1/datasets/79389/repo/tree?Revision=aa92b9e2a0bb07120885f620e6f9116ea4f62376&Root=/&Recursive=True&PageNumber=1&PageSize=100'
        files = get_json(api)['Data']['Files']
        expected = {'musique_ans_v1.0_train.jsonl': 19938, 'musique_ans_v1.0_dev.jsonl': 2417, 'musique_ans_v1.0_test.jsonl': 2459,
                    'musique_full_v1.0_train.jsonl': 39876, 'musique_full_v1.0_dev.jsonl': 4834}
        files.sort(key=lambda f: (f['Name'] not in ('musique_ans_v1.0_train.jsonl', 'musique_ans_v1.0_dev.jsonl'), f['Name']))
        for f in files:
            if f['Name'].startswith(('musique_ans_', 'musique_full_')) and f['Name'].endswith('.jsonl'):
                specs.append({'url': 'https://www.modelscope.cn/api/v1/datasets/voidful/MuSiQue/repo?Revision=' + f['Revision'] + '&FilePath=' + f['Path'], 'path': 'musique/' + f['Name'], 'sha256': f['Sha256'], 'size': f['Size'], 'expected_count': expected.get(f['Name']), 'source_kind': 'mirror'})
    manifest_path = RAW / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'created_at': utc(), 'files': [], 'failed_attempts': []}
    failed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(download, spec): spec for spec in specs}
        for future in concurrent.futures.as_completed(jobs):
            try:
                entry = future.result()
                manifest['files'] = [f for f in manifest['files'] if f['path'] != entry['path']] + [entry]
            except Exception as e:
                failed_count += 1
                failure = {'url': jobs[future]['url'], 'path': jobs[future]['path'], 'error': repr(e), 'time': utc()}
                manifest['failed_attempts'].append(failure)
                print(json.dumps(failure), flush=True)
            manifest['updated_at'] = utc()
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    if failed_count:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
