"""Dev-pool helpers (no downloads) and the Math-Verify grading of dev rollouts."""

import random
import sys
from pathlib import Path

import pytest

from limite_experiments.answers import answer_key, last_boxed
from limite_experiments.dev_pool import Decontaminator, ngrams, normalise, select

AIME = "Find the number of ordered pairs of positive integers (m, n) such that m^2 n = 20^20 and m divides n."


def test_normalise_and_ngrams():
    assert normalise("  Find $x$, if   x+1=2!") == "find x if x 1 2"
    assert len(ngrams("a b c d e f g h i j k", n=10)) == 2
    assert ngrams("too short", n=10) == set()


def test_last_boxed_and_answer_key():
    assert last_boxed(r"so \boxed{1} then \boxed{\frac{a}{b}}.") == r"\frac{a}{b}"
    assert last_boxed("no box") is None
    assert last_boxed(r"\boxed{unclosed") is None
    assert answer_key(r"\boxed{ 2 \pi }") == r"2\pi"
    assert answer_key(None) is None


def test_decontamination_catches_shared_ngrams_and_containment():
    d = Decontaminator([AIME])
    assert d.contaminated("Problem 3. " + AIME + " Give your answer mod 1000.")  # shares 10-grams
    assert d.contaminated(AIME.upper())  # normalisation
    assert not d.contaminated("Compute the sum of all positive divisors of 2024 that are perfect squares.")
    assert d.contaminated("   ")  # empty after normalisation


def test_select_dedups_skips_contaminated_and_respects_n():
    rows = [{"problem": f"Compute {i} times seven plus a fixed offset", "answer": str(i)} for i in range(20)]
    rows += [{"problem": rows[0]["problem"], "answer": "dup"}, {"problem": AIME, "answer": "x"}]
    picked, stats = select(list(rows), Decontaminator([AIME]), n=50, rng=random.Random(0), seen=set())
    assert stats["duplicate"] == 1 and stats["contaminated"] == 1 and len(picked) == 20
    picked, _ = select(list(rows), Decontaminator([AIME]), n=5, rng=random.Random(0), seen=set())
    assert len(picked) == 5


def test_math_verify_grading():
    pytest.importorskip("math_verify")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from generate_dev_rollouts import grade

    assert grade(r"... so the answer is \boxed{\frac{1}{2}}", "0.5")
    assert grade(r"thus \boxed{42}.", "42")
    assert not grade(r"thus \boxed{41}.", "42")
    assert not grade("no final answer", "42")
