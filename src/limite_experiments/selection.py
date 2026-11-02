"""Judges, majority voting and the early-selection policy, simulated exactly on stored rollouts.

A judge scores a prefix (higher = more likely to end correct). Stopping a rollout after c tokens costs c tokens, and
the judge sees exactly the prefix it would see live, so a policy's accuracy and cost can be computed offline.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

#: per-prefix features written by `scripts/value_study.py judge`, in this order
FEATURES = ("v", "vmean", "lp_w", "ent_w", "conf_w", "conf_min", "lp_mean")
#: feature judges, oriented so that higher = more likely correct
JUDGES = {
    "value": ("v", 1.0),
    "value_mean": ("vmean", 1.0),
    "conf_window": ("conf_w", 1.0),
    "conf_lowest": ("conf_min", 1.0),
    "logprob_window": ("lp_w", 1.0),
    "neg_entropy_window": ("ent_w", -1.0),
    "logprob_mean": ("lp_mean", 1.0),
}
#: tie-break when choosing the primary judge: cheaper first
COST_RANK = {"conf_window": 0, "conf_lowest": 0, "logprob_window": 0, "neg_entropy_window": 0, "logprob_mean": 0,
             "probe": 1, "value": 2, "value_mean": 2}


@dataclass
class Rollout:
    key: str
    problem: str
    tokens: int
    correct: bool
    answer: str | None


def vote(rollouts: list[Rollout]) -> Rollout | None:
    """Majority answer (None never wins); ties go to the earliest rollout in the given order."""
    answers = [r.answer for r in rollouts if r.answer is not None]
    if not answers:
        return rollouts[0] if rollouts else None
    counts = Counter(answers)
    best = max(counts.values())
    for r in rollouts:
        if r.answer is not None and counts[r.answer] == best:
            return r
    return None


def orders(n: int, n_orders: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(n_orders)])


def frontier(problems: dict[str, list[Rollout]], ks, n_orders: int = 200, seed: int = 0):
    """Plain sampling (maj@k): per k and problem, the mean voted correctness, mean per-sample correctness and mean
    tokens over random sample orders. Returns {k: {problem: (voted, per_sample, tokens)}}."""
    out = {}
    for k in ks:
        per = {}
        for pi, (p, rs) in enumerate(sorted(problems.items())):
            if k > len(rs):
                continue
            vo, ps, tk = [], [], []
            for o in orders(len(rs), n_orders, seed + pi):
                sel = [rs[i] for i in o[:k]]
                w = vote(sel)
                vo.append(float(w.correct))
                ps.append(float(np.mean([r.correct for r in sel])))
                tk.append(sum(r.tokens for r in sel))
            per[p] = (float(np.mean(vo)), float(np.mean(ps)), float(np.mean(tk)))
        out[k] = per
    return out


def select_policy(problems: dict[str, list[Rollout]], score, c: int, B: int, m: int, n_orders: int = 200,
                  seed: int = 0, judge_cost_per_token: float = 0.0):
    """select(judge, c, B, m): start B samples, read the judge after c tokens (samples that finished earlier are
    judged on their whole response), continue the m best, majority vote over them. score(rollout, c) -> float.
    Cost = sum min(L, c) + sum over kept (L - c)+ + judge_cost_per_token * sum min(L, c).
    Returns {problem: (voted, per_sample_kept, tokens)}."""
    per = {}
    for pi, (p, rs) in enumerate(sorted(problems.items())):
        if B > len(rs):
            continue
        sc = {r.key: score(r, c) for r in rs}
        vo, ps, tk = [], [], []
        for o in orders(len(rs), n_orders, seed + pi):
            cand = [rs[i] for i in o[:B]]
            ranked = sorted(cand, key=lambda r: -sc[r.key])  # stable: ties keep the sample order
            kept = ranked[:m]
            w = vote(kept)
            vo.append(float(w.correct))
            ps.append(float(np.mean([r.correct for r in kept])))
            prefix = sum(min(r.tokens, c) for r in cand)
            tk.append(prefix + sum(max(0, r.tokens - c) for r in kept) + judge_cost_per_token * prefix)
        per[p] = (float(np.mean(vo)), float(np.mean(ps)), float(np.mean(tk)))
    return per


def matched_delta(policy: dict, front: dict) -> tuple[float, list[float], float]:
    """Per-problem voted accuracy of the policy minus plain sampling interpolated (linearly in mean tokens, with one
    global weight) at the policy's mean tokens. Returns (mean delta, per-problem deltas, policy mean tokens)."""
    probs = sorted(policy)
    cost = float(np.mean([policy[p][2] for p in probs]))
    ks = sorted(k for k in front if all(p in front[k] for p in probs))
    costs = [float(np.mean([front[k][p][2] for p in probs])) for k in ks]
    if cost <= costs[0]:
        lo = hi = ks[0]
        w = 0.0
    elif cost >= costs[-1]:
        lo = hi = ks[-1]
        w = 0.0
    else:
        i = next(i for i in range(len(ks) - 1) if costs[i] <= cost <= costs[i + 1])
        lo, hi = ks[i], ks[i + 1]
        w = (cost - costs[i]) / (costs[i + 1] - costs[i])
    deltas = [policy[p][0] - ((1 - w) * front[lo][p][0] + w * front[hi][p][0]) for p in probs]
    return float(np.mean(deltas)), deltas, cost
