"""CPU-only, label-blind baseline policies for task 02.

Latest and BM25 read only compressor-visible state fields; BM25 additionally
uses the annotation-free local-ID to corpus-ID map. The first-fit rule skips an
oversized fragment and continues, as specified in TASK_02 §5 (Claude's reading).
"""
import bm25s
import numpy as np


def score_query(index, query):
    """Score every indexed paragraph with exactly the retrieval tokenizer.

    bm25s 0.3.16's retrieve() converts its Tokenized object back to string tokens
    and calls get_scores(). Using strings here prevents query-local token IDs
    from being confused with corpus-vocabulary token IDs. Empty token lists
    receive zeros, matching retrieve()'s empty-query path.
    """
    tokens = bm25s.tokenize([query], return_ids=False, show_progress=False)[0]
    if not tokens:
        return np.zeros(index.scores["num_docs"], dtype=index.dtype)
    return index.get_scores(tokens)


def _resolve_mode(state, requested_mode):
    mode = state["mode"]
    if mode not in ("reselect", "append"):
        raise ValueError("mode must be reselect or append")
    if requested_mode is not None and requested_mode != mode:
        raise ValueError("policy mode does not match the rollout state")
    return mode


def _pack(state, priority, mode):
    """First-fit by priority, while action execution retains arrival order."""
    kept, new = state["kept_ids"], state["new_ids"]
    budget, lengths = state["budget"], state["ntok"]
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer")
    selected = set(kept) if mode == "append" else set()
    used = sum(lengths[i] for i in selected)
    if used > budget:
        raise ValueError("append state already exceeds budget")
    for fragment_id in priority:
        if used + lengths[fragment_id] <= budget:
            selected.add(fragment_id)
            used += lengths[fragment_id]
    return {"add": [i for i in new if i in selected],
            "drop": [i for i in kept if i not in selected]}


def latest_policy(mode=None):
    """Return a policy prioritizing reverse kept+new arrival/presentation order."""
    def policy(state):
        active_mode = _resolve_mode(state, mode)
        candidates = (state["kept_ids"] + state["new_ids"]
                      if active_mode == "reselect" else state["new_ids"])
        return _pack(state, reversed(candidates), active_mode)

    return policy


def bm25_policy(index, mode=None, scores=None):
    """Return BM25 first-fit using the fixed, full dev corpus index's IDF.

    Query text is the four task questions joined by one space. If ``scores`` is
    supplied, it must be that task query's full-corpus score vector; callers may
    reuse it across modes/budgets and save candidate scores for auditing. With
    no supplied vector, this factory caches only the most recent task query.
    Equal scores prefer the later entry in kept+new, not the larger numeric ID.
    ``index=None`` is allowed only with supplied scores (useful for hand tests).
    """
    fixed_scores = None if scores is None else np.asarray(scores)
    if fixed_scores is not None:
        if fixed_scores.ndim != 1 or not np.all(np.isfinite(fixed_scores)):
            raise ValueError("scores must be a finite one-dimensional corpus vector")
        if index is not None and len(fixed_scores) != index.scores["num_docs"]:
            raise ValueError("scores length must match the full indexed corpus")
    elif index is None:
        raise ValueError("index is required when scores are not supplied")
    cached_query, cached_scores = None, None

    def policy(state):
        nonlocal cached_query, cached_scores
        active_mode = _resolve_mode(state, mode)
        if fixed_scores is None:
            query = " ".join(state["questions"])
            if query != cached_query:
                cached_scores = score_query(index, query)
                cached_query = query
            corpus_scores = cached_scores
        else:
            corpus_scores = fixed_scores
        candidates = (state["kept_ids"] + state["new_ids"]
                      if active_mode == "reselect" else state["new_ids"])
        paragraph_ids = state["pid_by_lid"]
        priority = [fragment_id for _, fragment_id in sorted(
            enumerate(candidates),
            key=lambda pair: (-float(corpus_scores[paragraph_ids[pair[1]]]), -pair[0]),
        )]
        return _pack(state, priority, active_mode)

    return policy


def no_compress_policy(state):
    """Keep everything. The caller must supply a sufficient no-compress budget."""
    return {"add": list(state["new_ids"]), "drop": []}
