"""MuSiQue/SQuAD answer EM and token F1, expressed as fractions in [0, 1].

Normalization removes ASCII string.punctuation (not Unicode punctuation).
MuSiQue's official metrics/answer.py returns 1 for F1 when both normalized
token lists are empty, and 0 when exactly one is empty. This follows the
verified official implementation, not the older SQuAD-v1 zero-overlap branch.
Official source blob: 7f8249369b79954733bd53cd93a0edb737c9b1c0.
"""
from collections import Counter
import re
import string


def normalize_answer(text):
    """Lowercase, remove punctuation/articles, then collapse whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction, ground_truth):
    """Normalized exact match against one reference."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    """Official MuSiQue token F1, including its empty-token-list special case."""
    predicted = normalize_answer(prediction).split()
    reference = normalize_answer(ground_truth).split()
    if not predicted or not reference:
        return float(predicted == reference)
    common = sum((Counter(predicted) & Counter(reference)).values())
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(reference)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction, answer, answer_aliases=()):
    """Take separate EM/F1 maxima over the main answer and every answer alias."""
    if answer_aliases is None:
        answer_aliases = ()
    if isinstance(answer_aliases, str):
        raise TypeError("answer_aliases must be an iterable of complete answer strings")
    references = [answer, *answer_aliases]
    return {
        "em": max(exact_match_score(prediction, ref) for ref in references),
        "f1": max(f1_score(prediction, ref) for ref in references),
    }
