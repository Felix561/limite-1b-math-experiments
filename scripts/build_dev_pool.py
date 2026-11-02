"""Development pool: licence-checked public math problems, deduplicated and decontaminated (CPU + downloads).

    python scripts/build_dev_pool.py --out pool.jsonl [--deepscaler 1400 --numina 1000 --math 600 --seed 0]

Runs inside the upstream `limite-evals` environment (it loads the evaluation sets with the harness's own loader).
Sources (licences checked 2026-10-26 on the Hub): agentica-org/DeepScaleR-Preview-Dataset (MIT),
AI-MO/NuminaMath-1.5 (Apache-2.0; non-synthetic competition sources with a verifiable non-proof answer),
EleutherAI/hendrycks_math train (MIT; levels 3-5, answer = last \\boxed{} of the solution).
Decontamination: every taskset of the harness suite `math-extended` (AIME 2024/25/26, HMMT Feb 2025/26, BeyondAIME,
Apex Shortlist, MATH-500, OlympiadBench, GSM8K); see limite_experiments.dev_pool.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_experiments.answers import last_boxed
from limite_experiments.dev_pool import Decontaminator, select

REVISIONS = {  # dataset commits read on 2026-10-26
    "agentica-org/DeepScaleR-Preview-Dataset": "b6ae8c60f5c1",
    "AI-MO/NuminaMath-1.5": "1b05109f9e5c",
    "EleutherAI/hendrycks_math": "21a5633873b6",
}
NUMINA_SOURCES = {"amc_aime", "olympiads", "olympiads_ref", "aops_forum", "cn_contest", "number_theory", "inequalities"}
MATH_CONFIGS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                "prealgebra", "precalculus")


def eval_problems() -> tuple[list[str], dict]:
    from limite_evals import suites, tasksets

    problems, counts = [], {}
    for spec in suites.resolve("math-extended"):
        rows = tasksets.load_rows(spec)
        counts[spec.name] = len(rows)
        problems += [p for p, _ in rows]
    return problems, counts


def load_sources() -> dict[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train",
                      revision=REVISIONS["agentica-org/DeepScaleR-Preview-Dataset"])
    deepscaler = [{"problem": r["problem"], "answer": str(r["answer"]), "source": "deepscaler"} for r in ds
                  if r["problem"] and r["answer"]]
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train", revision=REVISIONS["AI-MO/NuminaMath-1.5"])
    numina = [
        {"problem": r["problem"], "answer": str(r["answer"]).strip(), "source": f"numina/{r['source']}"}
        for r in ds
        if r["source"] in NUMINA_SOURCES and r["question_type"] == "math-word-problem"
        and r["problem_is_valid"] == "Yes" and r["solution_is_valid"] == "Yes" and str(r["synthetic"]) == "False"
        and str(r["answer"]).strip().lower() not in {"", "proof", "notfound", "none", "nan"}
    ]
    math_rows = []
    for config in MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="train",
                          revision=REVISIONS["EleutherAI/hendrycks_math"])
        for r in ds:
            answer = last_boxed(r["solution"])
            if answer and r["level"] in ("Level 3", "Level 4", "Level 5"):
                math_rows.append({"problem": r["problem"], "answer": answer, "source": f"math/{config}/{r['level']}"})
    return {"deepscaler": deepscaler, "numina": numina, "math": math_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--deepscaler", type=int, default=1400)
    ap.add_argument("--numina", type=int, default=1000)
    ap.add_argument("--math", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    evals, eval_counts = eval_problems()
    decon = Decontaminator(evals)
    sources = load_sources()
    seen: set[str] = set()
    pool, report = [], {"eval_sets": eval_counts, "eval_problems": len(evals), "revisions": REVISIONS, "sources": {}}
    for name, n in (("deepscaler", args.deepscaler), ("numina", args.numina), ("math", args.math)):
        picked, stats = select(sources[name], decon, n, rng, seen)
        report["sources"][name] = stats
        pool += picked
    rng.shuffle(pool)
    with args.out.open("w") as f:
        for i, row in enumerate(pool):
            f.write(json.dumps({"id": f"p{i:05d}", **row}) + "\n")
    report["pool_size"] = len(pool)
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
