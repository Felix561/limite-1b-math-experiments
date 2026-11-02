"""Build the GPU inputs of the value-model study from data already on disk (CPU, no downloads).

    python scripts/prepare_inputs.py \
        --aime-samples work/limite-violetto/limite-evals/outputs/eval/<run-id>/metrics/samples.jsonl \
        --dev-rollouts work/runs/dev/data/rollouts.jsonl --out work/inputs

aime26.jsonl.gz      the 960 rollouts of the AIME 2026 reproduction, with the harness's `exact` verdicts
aime_smoke.jsonl.gz  4 of them (3 long enough for the 1.5k/4k/8k parity checks, 1 very long one for the memory check)
dev.jsonl.gz         the dev-pool rollouts (16k cap, Math-Verify labels): the only data used to fit the probe and to
                     choose the primary judge (never the evaluated sets)
plan.jsonl.gz        commitment-curve plan: failed and successful AIME26 rollouts x prefix fractions x m continuations
plan_smoke.jsonl.gz  2 tiny branch requests for a smoke run
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_experiments.answers import answer_key

FRACS = (0.05, 0.15, 0.35, 0.6)


def dump(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{path.name}: {len(rows)} rows")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aime-samples", type=Path, required=True, help="metrics/samples.jsonl of the AIME26 harness run")
    ap.add_argument("--dev-rollouts", type=Path, required=True, help="rollouts.jsonl of generate_dev_rollouts.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-fail", type=int, default=24)
    ap.add_argument("--n-succ", type=int, default=24)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    aime = []
    for line in args.aime_samples.open():
        r = json.loads(line)
        aime.append({"key": f"aime26:{r['problem_index']}:{r['rollout_index']}", "taskset": "aime26",
                     "problem": r["problem_index"], "rollout": r["rollout_index"], "completion": r["completion"],
                     "tokens": r["completion_tokens"], "finish_reason": r["finish_reason"],
                     "exact": bool(r["metrics"]["exact"]), "answer": answer_key(r["completion"])})
    aime.sort(key=lambda r: (r["problem"], r["rollout"]))
    assert len(aime) == 960, len(aime)
    dump(args.out / "aime26.jsonl.gz", aime)

    long_ = [r for r in aime if r["tokens"] >= 9000 and r["finish_reason"] == "stop"]
    longest = max((r for r in aime if r["finish_reason"] == "stop"), key=lambda r: r["tokens"])
    dump(args.out / "aime_smoke.jsonl.gz", long_[:3] + [longest])

    dev = []
    for line in args.dev_rollouts.open():
        p = json.loads(line)
        for s, smp in enumerate(p["samples"]):
            dev.append({"key": f"dev:{p['id']}:{s}", "problem_text": p["problem"], "completion": smp["text"],
                        "tokens": int(smp["tokens"]), "finish_reason": smp["finish_reason"],
                        "correct": smp["correct"] in (True, "True"), "answer": answer_key(smp["text"])})
    dump(args.out / "dev.jsonl.gz", dev)

    # commitment-curve plan: rollouts that ended with an answer (finish_reason stop), from problems with failures
    by_problem = defaultdict(list)
    for r in aime:
        by_problem[r["problem"]].append(r)
    rng = random.Random(args.seed)
    mixed = sorted(p for p, rs in by_problem.items() if 0 < sum(r["exact"] for r in rs) < len(rs))
    fails = [r for p in mixed for r in by_problem[p] if not r["exact"] and r["finish_reason"] == "stop"]
    never = sorted(p for p, rs in by_problem.items() if not any(r["exact"] for r in rs))
    extra = [r for p in never for r in by_problem[p] if r["finish_reason"] == "stop"]
    # failures: a random sample from the mixed problems + 4 from the never-solved problem
    fails = rng.sample(fails, min(len(fails), args.n_fail - 4)) + extra[:4]
    succ_pool = {p: [r for r in by_problem[p] if r["exact"] and r["finish_reason"] == "stop"] for p in mixed}
    for rs in succ_pool.values():
        rng.shuffle(rs)
    succ, i = [], 0
    while len(succ) < args.n_succ and any(succ_pool.values()):  # round-robin over the mixed problems
        p = mixed[i % len(mixed)]
        if succ_pool[p]:
            succ.append(succ_pool[p].pop())
        i += 1
    plan = [{"key": r["key"], "taskset": "aime26", "problem": r["problem"], "rollout": r["rollout"],
             "completion": r["completion"], "f": f, "m": args.m, "orig_exact": r["exact"]}
            for r in fails + succ for f in FRACS]
    est = sum((1 - x["f"]) * next(r["tokens"] for r in aime if r["key"] == x["key"]) * x["m"] for x in plan)
    print(f"plan: {len(fails)} failed + {len(succ)} successful rollouts from problems {mixed} + {never[:1]}; "
          f"{len(plan)} prefixes, {sum(x['m'] for x in plan)} continuations, ~{est / 1e6:.1f}M tokens (estimate)")
    dump(args.out / "plan.jsonl.gz", plan)
    smoke = [dict(plan[0], m=1), dict(plan[-1], m=1)]
    dump(args.out / "plan_smoke.jsonl.gz", smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
