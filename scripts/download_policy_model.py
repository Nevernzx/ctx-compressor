#!/usr/bin/env python3
"""Download a frozen ModelScope snapshot, prioritizing tokenizer files.

Use after sourcing train_environment.sh on 5880. HTTP is direct and proxy-free;
all payloads and resumable partials stay under --model-root. Existing manifests
pin per-file revisions and are never silently refreshed from master.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
import urllib.parse
import urllib.request


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def root_disk_check(preflight):
    st = os.statvfs('/')
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    delta = used - preflight['root_df_used_bytes']
    if delta > preflight['root_final_growth_limit_bytes']:
        raise RuntimeError(f'Root disk growth exceeds preflight limit: {delta} bytes')
    return {'used_bytes': used, 'baseline_bytes': preflight['root_df_used_bytes'], 'delta_bytes': delta}


def discover(args, preflight):
    listing_url = f'https://www.modelscope.cn/api/v1/models/{args.model_id}/repo/files?' + urllib.parse.urlencode(
        {'Revision': args.revision, 'Recursive': 'true'})
    with opener().open(listing_url, timeout=90) as response:
        listing = json.load(response)
    if listing.get('Code') != 200:
        raise RuntimeError(f'ModelScope listing failed: {listing}')
    files = []
    for item in listing['Data']['Files']:
        if item['Type'] != 'blob':
            continue
        path = PurePosixPath(item['Path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Unsafe path: {path}')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', item.get('Sha256', '')):
            raise ValueError(f'Missing file SHA256: {path}')
        url = f'https://www.modelscope.cn/api/v1/models/{args.model_id}/repo?' + urllib.parse.urlencode(
            {'Revision': item['Revision'], 'FilePath': str(path)})
        files.append({'path': str(path), 'size_bytes': item['Size'], 'sha256': item['Sha256'].lower(),
                      'revision': item['Revision'], 'url': url, 'status': 'pending'})
    if not files or len({f['path'] for f in files}) != len(files):
        raise ValueError('Empty or duplicate model file listing')
    return {'model_id': args.model_id, 'requested_revision': args.revision, 'model_root': str(args.model_root),
            'listing_url': listing_url, 'listing_fetched_at_utc': now(), 'source_listing': listing,
            'files': files, 'total_expected_bytes': sum(f['size_bytes'] for f in files),
            'status': 'downloading', 'root_disk_before': root_disk_check(preflight)}


def download_file(item, args, preflight):
    item = dict(item)
    destination = args.model_root / item['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.part')
    started = time.monotonic()
    errors = []
    if destination.exists() and destination.stat().st_size == item['size_bytes'] and sha256(destination) == item['sha256']:
        item.setdefault('downloaded_at_utc', datetime.fromtimestamp(destination.stat().st_mtime, timezone.utc).isoformat())
        item.update(status='verified', verified_at_utc=now(), actual_size_bytes=destination.stat().st_size,
                    actual_sha256=item['sha256'], reused_verified_file=True)
        return item
    if args.verify_only:
        item.update(status='failed', error='Local file missing, size mismatch, or SHA256 mismatch')
        return item
    attempt, failures = 0, 0
    while failures < args.attempts:
        attempt += 1
        offset = temporary.stat().st_size if temporary.exists() else 0
        try:
            root_disk_check(preflight)
            if offset > item['size_bytes']:
                temporary.unlink()
                offset = 0
            if offset < item['size_bytes']:
                headers = {'User-Agent': 'ctx-compressor-task04/1.0'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                request = urllib.request.Request(item['url'], headers=headers)
                with opener().open(request, timeout=90) as response:
                    if offset and response.status == 206:
                        content_range = response.headers.get('Content-Range', '')
                        if not content_range.startswith(f'bytes {offset}-'):
                            raise RuntimeError(f'Unexpected Content-Range: {content_range}')
                        mode = 'ab'
                    else:
                        if response.status != 200:
                            raise RuntimeError(f'Unexpected HTTP status: {response.status}')
                        mode = 'wb'
                    with temporary.open(mode) as output:
                        last_check = time.monotonic()
                        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
                            output.write(chunk)
                            if time.monotonic() - last_check > 30:
                                root_disk_check(preflight)
                                last_check = time.monotonic()
            actual_size = temporary.stat().st_size
            if actual_size != item['size_bytes']:
                if offset < actual_size < item['size_bytes']:
                    print(json.dumps({'path': item['path'], 'status': 'partial_resume',
                                      'downloaded_bytes': actual_size, 'expected_bytes': item['size_bytes']}), flush=True)
                    continue
                raise RuntimeError(f'Size mismatch: {actual_size} != {item["size_bytes"]}')
            actual_hash = sha256(temporary)
            if actual_hash != item['sha256']:
                temporary.unlink()
                raise RuntimeError(f'SHA256 mismatch: {actual_hash} != {item["sha256"]}; discarded partial')
            temporary.replace(destination)
            item.update(status='verified', downloaded_at_utc=now(), verified_at_utc=now(),
                        elapsed_seconds=round(time.monotonic() - started, 3), attempts=attempt,
                        actual_size_bytes=actual_size, actual_sha256=actual_hash, retry_errors=errors)
            print(json.dumps({'path': item['path'], 'status': 'verified', 'bytes': actual_size}), flush=True)
            return item
        except Exception as error:
            errors.append({'attempt': attempt, 'at_utc': now(), 'error': f'{type(error).__name__}: {error}'})
            print(json.dumps({'path': item['path'], **errors[-1]}), flush=True)
            if 'Root disk growth' in str(error):
                break
            current_size = temporary.stat().st_size if temporary.exists() else 0
            # Partial progress resumes without consuming the finite failure budget.
            if current_size <= offset or 'SHA256 mismatch' in str(error):
                failures += 1
            if failures < args.attempts:
                time.sleep(min(30, 3 * max(1, failures)))
    item.update(status='failed', retry_errors=errors, elapsed_seconds=round(time.monotonic() - started, 3))
    return item


def compare_tokenizer(args, manifest):
    model_tokenizer = args.model_root / 'tokenizer.json'
    if not model_tokenizer.exists():
        return
    reference_sha = sha256(args.reference_tokenizer)
    actual_sha = sha256(model_tokenizer)
    manifest['tokenizer_comparison'] = {
        'reference_path': str(args.reference_tokenizer), 'model_path': str(model_tokenizer),
        'reference_sha256': reference_sha, 'model_sha256': actual_sha,
        'identical': reference_sha == actual_sha,
        'ntok_impact': 'none; tokenizer.json is byte-identical' if reference_sha == actual_sha
                       else 'tokenizer.json differs; compare tokenization before trusting prior ntok counts',
        'checked_at_utc': now(),
    }
    print(json.dumps({'tokenizer_comparison': manifest['tokenizer_comparison']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--revision', default='master')
    parser.add_argument('--model-root', type=Path, default=Path(os.environ.get('TRAIN_MODEL', '/root/userfolder/ctx-compressor/models/Qwen3-1.7B')))
    parser.add_argument('--manifest', type=Path, default=Path('eval/sft_prep/model_manifest.json'))
    parser.add_argument('--preflight', type=Path, default=Path('eval/sft_prep/preflight.json'))
    parser.add_argument('--reference-tokenizer', type=Path, default=Path('data/raw/qwen3_tokenizer/tokenizer.json'))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--attempts', type=int, default=6)
    parser.add_argument('--small-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.attempts < 1:
        parser.error('workers must be 1..4 and attempts >= 1')
    preflight = json.loads(args.preflight.read_text())
    root_disk_check(preflight)
    args.model_root.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        if manifest['model_id'] != args.model_id or manifest['model_root'] != str(args.model_root):
            raise ValueError('Existing manifest model or directory differs from requested snapshot')
    elif args.verify_only:
        raise ValueError('--verify-only requires an existing frozen manifest')
    else:
        manifest = discover(args, preflight)
    save_manifest(args.manifest, manifest)
    started = time.monotonic()
    small = [i for i, item in enumerate(manifest['files']) if item['size_bytes'] <= 32 * 1024 * 1024]
    large = [i for i in range(len(manifest['files'])) if i not in small]
    # Complete all tokenizer/config files first, unblocking CPU-only data work.
    for indices in ([small] if args.small_only else [small, large]):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_file, manifest['files'][i], args, preflight): i for i in indices}
            for future in concurrent.futures.as_completed(futures):
                manifest['files'][futures[future]] = future.result()
                save_manifest(args.manifest, manifest)
        if indices == small:
            compare_tokenizer(args, manifest)
            manifest['small_files_ready'] = all(manifest['files'][i]['status'] == 'verified' for i in small)
            manifest['small_files_checked_at_utc'] = now()
            save_manifest(args.manifest, manifest)
            print(json.dumps({'small_files_ready': manifest['small_files_ready']}), flush=True)
    manifest['verified_file_count'] = sum(item['status'] == 'verified' for item in manifest['files'])
    manifest['status'] = ('verified' if manifest['verified_file_count'] == len(manifest['files'])
                          else 'small_files_ready' if args.small_only and manifest['small_files_ready'] else 'failed')
    manifest['last_run_finished_at_utc'] = now()
    manifest['last_run_elapsed_seconds'] = round(time.monotonic() - started, 3)
    if manifest['status'] == 'verified':
        manifest.setdefault('download_finished_at_utc', now())
        manifest['wall_seconds_since_listing'] = round((datetime.now(timezone.utc) - datetime.fromisoformat(manifest['listing_fetched_at_utc'])).total_seconds(), 3)
        expected = {item['path'] for item in manifest['files']}
        actual = {str(p.relative_to(args.model_root)) for p in args.model_root.rglob('*') if p.is_file()}
        manifest['unexpected_files'] = sorted(actual - expected)
        if manifest['unexpected_files']:
            manifest['status'] = 'failed'
    manifest['root_disk_after'] = root_disk_check(preflight)
    save_manifest(args.manifest, manifest)
    print(json.dumps({key: manifest[key] for key in ['status', 'verified_file_count', 'root_disk_after']}), flush=True)
    return 0 if manifest['status'] in ('verified', 'small_files_ready') else 1


if __name__ == '__main__':
    raise SystemExit(main())
