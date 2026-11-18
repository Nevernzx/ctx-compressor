"""Input cost in base-input-price × token units (PLAN v2 §5)."""


def trajectory_cost(contexts, ntok, r_read=0.05, r_write=1.25):
    """Return uncached/cached cost, using the preceding step's common prefix."""
    plain = cached = 0.0
    prev = []
    for cur in contexts:
        k = 0
        while k < min(len(prev), len(cur)) and prev[k] == cur[k]:
            k += 1
        hit = sum(ntok[i] for i in cur[:k])
        miss = sum(ntok[i] for i in cur[k:])
        plain += hit + miss
        cached += r_read * hit + r_write * miss
        prev = cur
    return plain, cached
