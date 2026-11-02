"""Deduplication and decontamination of the development pool against every evaluation set of the harness."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable

NGRAM = 10


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = normalise(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class Decontaminator:
    """A problem is contaminated if it shares any word 10-gram with an evaluation problem, or contains / is contained
    in one after normalisation."""

    def __init__(self, eval_problems: Iterable[str]) -> None:
        self.texts = [normalise(p) for p in eval_problems if p]
        self.grams: set[tuple[str, ...]] = set()
        for text in self.texts:
            self.grams |= ngrams(text)

    def contaminated(self, problem: str) -> bool:
        norm = normalise(problem)
        if not norm:
            return True
        if ngrams(norm) & self.grams:
            return True
        return any((t in norm or norm in t) for t in self.texts if len(t) > 40 and len(norm) > 40)


def select(rows: list[dict], decon: Decontaminator, n: int, rng: random.Random, seen: set[str]) -> tuple[list, dict]:
    """Shuffle, then take up to n rows that are neither duplicates (of `seen`) nor contaminated."""
    stats = {"candidates": len(rows), "duplicate": 0, "contaminated": 0}
    rng.shuffle(rows)
    picked = []
    for row in rows:
        key = normalise(row["problem"])
        if key in seen:
            stats["duplicate"] += 1
            continue
        if decon.contaminated(row["problem"]):
            stats["contaminated"] += 1
            continue
        seen.add(key)
        picked.append(row)
        if len(picked) >= n:
            break
    stats["selected"] = len(picked)
    return picked, stats
