"""Refresh only MANIFEST sample schemas from the first existing data record."""
import json
from pathlib import Path

from download_data import schema

ROOT = Path('/root/workplace/ctx-compressor')


def first_record(path):
    with path.open(encoding='utf-8') as stream:
        if path.suffix == '.jsonl':
            return json.loads(next(stream))
        decoder = json.JSONDecoder()
        buffer = ''
        while chunk := stream.read(8192):
            buffer += chunk
            stripped = buffer.lstrip()
            if not stripped.startswith('['):
                raise ValueError(f'Expected a JSON array in {path}')
            try:
                record, _ = decoder.raw_decode(stripped[1:].lstrip())
                return record
            except json.JSONDecodeError:
                continue
        raise ValueError(f'No complete first record in {path}')


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit('Run only from the project directory on 5880')
    manifest_path = ROOT / 'data/raw/MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    updated = []
    for entry in manifest['files']:
        path = ROOT / 'data/raw' / entry['path']
        if entry['record_count'] is not None:
            entry['sample_field_structure'] = schema(first_record(path))
            updated.append(entry['path'])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'updated_sample_schemas': updated}, ensure_ascii=False))


if __name__ == '__main__':
    main()
