import math
from typing import Sequence


def recall_at_k(ranked_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    if not expected_ids:
        return 0.0
    top = list(ranked_ids)[:max(0, int(k))]
    hits = 0
    for item in expected_ids:
        if item in top:
            hits += 1
    return hits / float(len(expected_ids))


def reciprocal_rank(ranked_ids: Sequence[str], expected_ids: Sequence[str]) -> float:
    wanted = set(expected_ids or [])
    if not wanted:
        return 0.0
    for position, item in enumerate(ranked_ids, 1):
        if item in wanted:
            return 1.0 / position
    return 0.0


def mean(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def percentile(values, ratio: float) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    if len(items) == 1:
        return items[0]
    position = max(0.0, min(1.0, float(ratio))) * (len(items) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(items) - 1)
    weight = position - lower
    return items[lower] * (1.0 - weight) + items[upper] * weight
