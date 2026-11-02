import math

import numpy as np

from limite_experiments.probe import grouped_folds, ridge_fit, ridge_predict
from limite_experiments.selection import Rollout, frontier, matched_delta, select_policy, vote
from limite_experiments.stats import auroc, bootstrap_ci, within_problem_auroc


def test_auroc_basic_and_ties():
    assert auroc([3, 2, 1], [1, 1, 0]) == 1.0
    assert auroc([1, 2, 3], [1, 1, 0]) == 0.0
    assert auroc([1, 1], [1, 0]) == 0.5
    assert math.isnan(auroc([1, 2], [1, 1]))


def test_within_problem_auroc_skips_single_class():
    mean, per = within_problem_auroc({"a": [(1.0, True), (0.0, False)], "b": [(1.0, True), (2.0, True)]})
    assert mean == 1.0 and per == [1.0]


def test_bootstrap_ci_brackets_the_mean():
    x = np.random.default_rng(1).random(30)
    m, lo, hi = bootstrap_ci(x)
    assert lo < m < hi and abs(m - x.mean()) < 1e-12
    assert all(math.isnan(v) for v in bootstrap_ci([]))


def test_vote_majority_and_ties():
    rs = [Rollout("1", "p", 10, False, "a"), Rollout("2", "p", 10, True, "b"), Rollout("3", "p", 10, True, "b")]
    assert vote(rs).answer == "b"
    assert vote(rs[:2]).key == "1"  # tie -> earliest
    assert vote([Rollout("x", "p", 1, False, None)]).key == "x"


def _problems():
    # one problem: correct rollouts are long, wrong ones short; a perfect judge separates them
    rs = [Rollout(f"k{i}", "p", 100 + i, i % 2 == 0, "right" if i % 2 == 0 else f"wrong{i}") for i in range(16)]
    return {"p": rs}


def test_select_with_oracle_beats_random_and_costs_are_exact():
    probs = _problems()
    oracle = select_policy(probs, lambda r, c: float(r.correct), c=10, B=4, m=1, n_orders=50)
    rand = frontier(probs, [1], n_orders=50)
    assert oracle["p"][1] >= rand[1]["p"][1]
    # cost: 4 prefixes of 10 tokens + the kept sample's remainder
    for o in range(5):
        per = select_policy(probs, lambda r, c: 0.0, c=10, B=4, m=1, n_orders=1, seed=o)
        assert 40 + 90 <= per["p"][2] <= 40 + 105


def test_judge_cost_is_charged_on_scored_prefixes():
    probs = _problems()
    free = select_policy(probs, lambda r, c: 0.0, c=10, B=4, m=1, n_orders=3)
    paid = select_policy(probs, lambda r, c: 0.0, c=10, B=4, m=1, n_orders=3, judge_cost_per_token=0.1)
    assert abs((paid["p"][2] - free["p"][2]) - 0.1 * 40) < 1e-9


def test_matched_delta_zero_for_plain_sampling():
    probs = _problems()
    front = frontier(probs, [1, 2, 4], n_orders=30)
    plain = {p: front[2][p] for p in front[2]}
    d, _, _ = matched_delta(plain, front)
    assert abs(d) < 1e-9


def test_ridge_probe_ranks_signal_and_folds_group():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + 0.1 * rng.normal(size=400) > 0).astype(float)
    m = ridge_fit(X, y, 1e-3)
    assert auroc(ridge_predict(m, X), y > 0.5) > 0.95
    groups = [f"g{i // 2}" for i in range(20)]
    folds = grouped_folds(groups, k=5)
    for idx in folds:
        inside = {groups[i] for i in idx}
        outside = {groups[i] for j, other in enumerate(folds) if other is not idx for i in other}
        assert not inside & outside
    assert sorted(np.concatenate(folds).tolist()) == list(range(20))
