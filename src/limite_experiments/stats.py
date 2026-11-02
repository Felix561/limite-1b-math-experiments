"""Bootstrap intervals and AUROC (numpy only)."""

from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(values, n: int = 10_000, seed: int = 0, stat=np.mean) -> tuple[float, float, float]:
    """(statistic, 2.5 %, 97.5 %) of a percentile bootstrap. Callers pass one value per problem (or per rollout), so
    the resampling unit is explicit at the call site. nan for empty input."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)])
    return float(stat(v)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def auroc(scores, labels) -> float:
    """Mann-Whitney AUROC with ties counted as 1/2; nan if one class is missing."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    ranks = np.empty(len(sorted_v))
    i = 0
    while i < len(sorted_v):  # average ranks over ties
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(sorted_v))
    r[order] = ranks
    rpos = r[: len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def within_problem_auroc(groups: dict[str, list[tuple[float, bool]]]) -> tuple[float, list[float]]:
    """Mean over problems (with both outcomes) of the AUROC among that problem's rollouts: how well a judge ranks the
    candidates it would actually have to choose between. Returns (mean, per-problem values)."""
    per = []
    for rows in groups.values():
        a = auroc([s for s, _ in rows], [y for _, y in rows])
        if not math.isnan(a):
            per.append(a)
    return (float(np.mean(per)) if per else math.nan), per
