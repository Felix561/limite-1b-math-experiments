"""Controls for the selection policy on HMMT Feb 2026 (CPU): a random judge and a final-length judge.

    python scripts/controls.py RUN_DIR [RUN_DIR ...] > results/value_model/controls.json

RUN_DIR = the GPU runs that generated the HMMT samples (each contains out/gen/samples.jsonl). A random judge must land
on or below the plain-sampling frontier (otherwise the policy itself is biased). The "shorter is better" judge uses the
final length (not available at the checkpoint) and bounds how much of a gain could come from preferring samples that
finish soon.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_experiments.selection import Rollout, frontier, matched_delta, select_policy
from limite_experiments.stats import bootstrap_ci


def main() -> int:
    probs = defaultdict(list)
    for run in sys.argv[1:]:
        for line in (Path(run) / "out/gen/samples.jsonl").open():
            r = json.loads(line)
            key = f"hmmt26:{r['problem']}:{r['rollout']}"
            probs[str(r["problem"])].append(Rollout(key, str(r["problem"]), r["tokens"], bool(r["exact"]), r["answer"]))
    n_min = min(len(rs) for rs in probs.values())
    front = frontier(probs, list(range(1, n_min + 1)), 200)
    rng = random.Random(0)
    rnd = {r.key: rng.random() for rs in probs.values() for r in rs}
    out = {"controls": []}
    for name, fn in (("random", lambda r, t: rnd[r.key]), ("shorter_is_better_final_length", lambda r, t: -r.tokens)):
        for c, B, m in ((4096, 8, 2), (2048, 16, 4), (4096, 16, 1), (1024, 16, 1)):
            per = select_policy(probs, fn, c, B, m, 200)
            d, deltas, tok = matched_delta(per, front)
            out["controls"].append({"judge": name, "c": c, "B": B, "m": m, "delta_vs_frontier": d,
                                    "delta_ci95": list(bootstrap_ci(deltas)[1:]), "tokens": tok})
    lens = []
    for rs in probs.values():
        ok = [r.tokens for r in rs if r.correct]
        bad = [r.tokens for r in rs if not r.correct]
        if ok and bad:
            lens.append((float(np.mean(ok)), float(np.mean(bad))))
    out["mixed_problems"] = len(lens)
    out["mean_tokens_correct_vs_wrong"] = [float(np.mean([a for a, _ in lens])), float(np.mean([b for _, b in lens]))]
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
