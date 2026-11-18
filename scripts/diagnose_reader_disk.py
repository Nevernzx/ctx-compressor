"""CPU-only inspection of root-disk transients; never loads a model."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import stat
import tempfile
import threading
import time

ROOT = Path('/root/workplace/ctx-compressor')
BASELINE = json.loads((ROOT / 'eval/reader/preflight.json').read_text())['root_df_used_bytes']
TRACE = Path('/root/userfolder/ctx-compressor/cache/root_disk_diagnostic.json')


def writable_root_fds():
    found = []
    root_device = os.stat('/').st_dev
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process / 'fd').iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                info = descriptor.stat()
                if info.st_dev != root_device or not stat.S_ISREG(info.st_mode):
                    continue
                fdinfo = (process / 'fdinfo' / descriptor.name).read_text()
                flags = int(next(line.split()[1] for line in fdinfo.splitlines() if line.startswith('flags:')), 8)
                if flags & 3 not in (1, 2):
                    continue
                found.append({'pid': int(process.name), 'process': (process / 'comm').read_text().strip(),
                              'target': os.readlink(descriptor), 'bytes': info.st_size,
                              'allocated_bytes': info.st_blocks * 512})
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    return found


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert Path(tempfile.gettempdir()).is_relative_to('/root/userfolder/ctx-compressor/cache')
    samples, incidents = [], []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            disk = os.statvfs('/')
            used = (disk.f_blocks - disk.f_bfree) * disk.f_frsize
            sample = {'unix_time': time.time(), 'delta_bytes': used - BASELINE}
            samples.append(sample)
            if sample['delta_bytes'] > 100_000_000:
                incidents.append({**sample, 'writable_root_fds': writable_root_fds()})
            stop.wait(0.1)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        namespace = runpy.run_path(str(ROOT / 'scripts/run_reader.py'), run_name='disk_diagnostic')
        requests, result = namespace['prepare_requests'](Path('/root/userfolder/ctx-compressor/models/Qwen3-8B'))
        assert len(requests) == 12000
    finally:
        stop.set()
        thread.join(timeout=10)
    TRACE.write_text(json.dumps({'samples': samples, 'incidents': incidents}, indent=2) + '\n')
    summary = {'mode': 'CPU-only; CUDA_VISIBLE_DEVICES empty; no vLLM import or model loading',
               'elapsed_seconds': time.monotonic() - started, 'n_samples': len(samples),
               'root_min_delta_bytes': min(row['delta_bytes'] for row in samples),
               'root_max_delta_bytes': max(row['delta_bytes'] for row in samples),
               'samples_over_100MB': len(incidents), 'actual_tempfile_directory': tempfile.gettempdir(),
               'diagnostic_trace_path': str(TRACE), 'diagnostic_trace_sha256': hashlib.sha256(TRACE.read_bytes()).hexdigest(),
               'prompt_precheck': result}
    (ROOT / 'eval/reader/root_disk_cpu_diagnostic.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
