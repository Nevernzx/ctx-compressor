#!/usr/bin/env python3
"""Record final root-disk delta and release of the Task 04 GPU process."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    project = Path(__file__).resolve().parents[1]
    if project != Path('/root/workplace/ctx-compressor') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('run on 5880 in CPU mode')
    out = project / 'eval/sft_prep'
    preflight = json.loads((out / 'preflight.json').read_text())
    identity = json.loads((out / 'gpu_process_check.json').read_text())
    bench = json.loads((out / 'bench_results.json').read_text())
    if bench['optimizer_steps'] != 100 or bench['status'] != 'ok':
        raise ValueError('finish the benchmark before final resource checks')
    pid = identity['benchmark_pid']
    apps = []
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    for line in raw.splitlines():
        fields = line.split(', ')
        if int(fields[1]) == pid:
            apps.append({'gpu_uuid': fields[0], 'pid': pid, 'used_memory_mib': int(fields[2])})
    gpu = subprocess.check_output(['nvidia-smi', '-i', '1',
          '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    used = int(subprocess.check_output(['df', '-B1', '--output=used', '/'], text=True).splitlines()[-1])
    growth = used - preflight['root_df_used_bytes']
    alive = Path(f'/proc/{pid}').exists()
    record = {
        'utc': datetime.now(timezone.utc).isoformat(), 'root_final_used_bytes': used,
        'root_baseline_used_bytes': preflight['root_df_used_bytes'], 'root_delta_bytes': growth,
        'root_limit_bytes': preflight['root_final_growth_limit_bytes'],
        'root_limit_passed': growth <= preflight['root_final_growth_limit_bytes'],
        'gpu1_after': {'index': int(gpu[0]), 'uuid': gpu[1], 'memory_used_mib': int(gpu[2]),
                       'utilization_percent': int(gpu[3])},
        'task04_gpu_processes_remaining': apps, 'benchmark_pid_alive': alive,
        'gpu_released': not apps and not alive,
        'only_gpu1': identity['only_gpu1'] and identity['cuda_visible_devices'] == '1'
                     and all(row['gpu_uuid'] == preflight['gpu']['uuid'] for row in identity['observed_processes']),
        'gpu_identity_evidence': 'gpu_process_check.json', 'benchmark_pid': pid,
        'benchmark_optimizer_steps': bench['optimizer_steps'],
        'longest_probe_extra_optimizer_steps': bench['longest_probe']['optimizer_steps'],
    }
    (out / 'final_resource_check.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    return 0 if record['root_limit_passed'] and record['gpu_released'] and record['only_gpu1'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
