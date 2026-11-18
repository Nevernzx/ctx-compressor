"""Read-only GitHub discovery, to be executed only on 5880 without proxies."""
import concurrent.futures
import hashlib
import json
import pathlib
import sys
import urllib.request

LOGS = pathlib.Path('/root/workplace/ctx-compressor/logs')
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def get(url):
    name = LOGS / ('musique_discovery_' + hashlib.sha256(url.encode()).hexdigest()[:16] + '.json')
    if name.exists():
        return json.loads(name.read_text())
    req = urllib.request.Request(url, headers={'User-Agent': 'ctx-compressor-source-verification', 'Accept': 'application/vnd.github+json'})
    try:
        with opener.open(req, timeout=35) as response:
            body = response.read(12000000)
        result = json.loads(body)
        name.write_text(json.dumps(result))
        return result
    except Exception as exc:
        print(json.dumps({'url': url, 'error': str(exc)}), flush=True)
        return {}

def tree(repo):
    data = get(f'https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1')
    matches = [x for x in data.get('tree', []) if x.get('type') == 'blob' and any(w in x['path'].lower() for w in ['musique', 'download_data'])]
    print(json.dumps({'repo': repo, 'truncated': data.get('truncated'), 'matches': matches}), flush=True)

if __name__ == '__main__':
    command, *args = sys.argv[1:]
    if command == 'search':
        for query in args:
            data = get('https://api.github.com/search/repositories?q=' + urllib.parse.quote(query) + '&per_page=60')
            print(json.dumps({'query': query, 'total_count': data.get('total_count'), 'repos': [{'name': x['full_name'], 'size': x['size'], 'default_branch': x['default_branch']} for x in data.get('items', [])]}), flush=True)
    elif command == 'trees':
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(tree, args))
    else:
        raise SystemExit('use search QUERY or trees REPO...')
