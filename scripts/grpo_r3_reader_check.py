"""Accept the round-three reader comparison only for the selected held-out checkpoint."""
import json

from scripts.sample_sft_subset import require


def verify(checkpoint, path):
    strategy = 'grpo_r3:' + checkpoint
    result = json.loads(path.read_bytes())
    pooled = result['strategies'][strategy]['pooled']
    comparisons = result['comparisons']
    require(result['n_predictions'] == pooled['n'] == 5976 and len(comparisons) == 1
            and 0 <= pooled['f1'] <= 1 and 0 <= pooled['em'] <= 1, 'G reader coverage differs')
    pair, = comparisons
    require(pair['left'] == strategy and pair['right'] == 'grpo_r2:checkpoint-125' and pair['n'] == 5976
            and 'f1_difference_pp' in pair and 'em_difference_pp' in pair
            and sum(pair['f1_counts'].values()) == 5976
            and sum(map(sum,pair['em_matrix_correct_then_incorrect'])) == 5976, 'G reader pairing differs')
    require(path.with_name('reader_f1_report.md').is_file(), 'G reader report missing')
    return result

