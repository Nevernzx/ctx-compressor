#!/usr/bin/env python3
"""Export the 100 raw benchmark loss values as a small SVG and CSV, CPU only."""
import csv
import json
import math
import os
from pathlib import Path
import statistics


def main():
    project = Path(__file__).resolve().parents[1]
    if project != Path('/root/workplace/ctx-compressor') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('run only on 5880, CPU mode, with the train environment')
    out = project / 'eval/sft_prep'
    bench = json.loads((out / 'bench_results.json').read_text())
    rows = bench['loss_curve']
    if [row['step'] for row in rows] != list(range(1, 101)):
        raise ValueError('require exactly the 100 measured optimizer steps')
    losses = [row['loss'] for row in rows]
    if not all(math.isfinite(value) and value > 0 for value in losses):
        raise ValueError('finite positive loss values are required for the logarithmic axis')
    with (out / 'loss_curve.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['optimizer_step', 'loss', 'elapsed_seconds'])
        for row in rows:
            writer.writerow([row['step'], row['loss'], row['elapsed_seconds']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.2, 3.4), layout='constrained')
    ax.plot(range(1, 101), losses, color='#1f5f99', linewidth=1.2,
            marker='.', markersize=3, label='Raw optimizer-step loss')
    first, last = statistics.mean(losses[:10]), statistics.mean(losses[-10:])
    ax.hlines(first, 1, 10, color='#b85c24', linewidth=2.1,
              label=f'First 10 mean: {first:.4f}')
    ax.hlines(last, 91, 100, color='#25835a', linewidth=2.1,
              label=f'Last 10 mean: {last:.4f}')
    ax.set(xlabel='Optimizer step (16 samples per step)', ylabel='Loss (log scale)',
           yscale='log', xlim=(0, 101), title='Qwen3-1.7B LoRA throughput benchmark; n = 100 steps')
    ax.grid(axis='y', alpha=0.18, which='major')
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / 'loss_curve.svg', metadata={'Date': None,
                 'Description': 'All 100 measured losses, no smoothing; 1600 samples, 800 per mode.'})
    plt.close(fig)
    print(json.dumps({'csv_rows': 100, 'svg': str(out / 'loss_curve.svg'),
                      'first_10_mean': first, 'last_10_mean': last}))


if __name__ == '__main__':
    main()
