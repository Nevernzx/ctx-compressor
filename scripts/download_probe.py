"""Read-only source discovery; run only on 5880, never uses proxies."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path('/root/workplace/ctx-compressor')
if not ROOT.is_dir():
    raise SystemExit('This helper must run on 5880.')

def probe(url):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(url, headers={'User-Agent': 'ctx-compressor-data-audit/1.0'}), timeout=35) as r:
            data = r.read(8_000_000)
        cache = ROOT / 'logs' / ('probe_' + hashlib.sha256(url.encode()).hexdigest()[:12] + '.txt')
        cache.write_bytes(data)
        try:
            obj = json.loads(data)
            if isinstance(obj, dict) and 'items' in obj:
                result = [{k: x.get(k) for k in ('full_name', 'default_branch', 'size')} for x in obj['items']]
            elif isinstance(obj, dict) and 'tree' in obj:
                result = [x for x in obj['tree'] if any(s in x['path'].lower() for s in ('musique', 'hotpot', '2wiki', 'license', 'readme', 'download', 'dev.json')) and not x['path'].startswith(('base_configs/', 'prompts/', 'experiment_configs/'))]
            elif isinstance(obj, dict) and isinstance(obj.get('Data'), list):
                result = {'TotalCount': obj.get('TotalCount'), 'Data': [{k: x.get(k) for k in ('Id', 'Namespace', 'Name', 'StorageSize', 'License')} for x in obj['Data']]}
            else:
                result = obj
        except ValueError:
            result = data.decode(errors='replace')[:12000]
        return {'url': url, 'cache': str(cache), 'result': result}
    except Exception as e:
        return {'url': url, 'error': repr(e)}

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for result in pool.map(probe, sys.argv[1:]):
        print(json.dumps(result, ensure_ascii=False), flush=True)
