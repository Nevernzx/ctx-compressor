"""Pure acceptance rules shared by the Task 05 batch comparison and training."""
import math

SEED = 20261118
MAX_LENGTH = 8392
MEMORY_LIMIT_BYTES = 40_000_000_000


def choose_batch(results):
    eligible = [row for row in results if row.get('status') == 'ok'
                and row.get('optimizer_steps') == 30 and row.get('consumed_samples') == 480
                and row.get('longest_probe', {}).get('oom') is False
                and row.get('longest_probe', {}).get('finite_gradients') is True
                and row.get('longest_probe', {}).get('optimizer_steps') == 0
                and 0 < row.get('selection_peak_bytes', math.inf) <= MEMORY_LIMIT_BYTES
                and math.isfinite(row.get('tokens_per_second', float('nan')))
                and row.get('tokens_per_second', 0) > 0]
    if not eligible:
        raise ValueError('no batch configuration passed the longest-batch and 40 GB gates')
    return max(eligible, key=lambda row: (row['tokens_per_second'], -row['batch_size']))


def validate_training_counts(steps, samples, epochs):
    if (steps, samples) != (2500, 40000) or abs(epochs - 2.0) > 1e-8:
        raise ValueError('formal SFT must consume exactly 40,000 samples in 2 epochs / 2,500 steps')


def checkpoint_steps(paths):
    steps = sorted(int(path.rsplit('-', 1)[-1]) for path in paths)
    if steps != [500, 1000, 1500, 2000, 2500]:
        raise ValueError('expected exactly five checkpoints at 500-step intervals')
    return steps
