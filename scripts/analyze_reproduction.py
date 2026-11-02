"""Accuracy and cost of one upstream harness run (CPU), from the harness's own records.

    python scripts/analyze_reproduction.py RUN_DIR [--env ENV_DIR] --out results/reproduction/aime26.json

RUN_DIR  = limite-evals/outputs/eval/<run-id> (contains metrics/samples.jsonl and, for a complete run, summary.json)
ENV_DIR  = optional evidence written by gpu/aime26_reproduction.sh (gpu-monitor.csv, vllm-metrics.jsonl, runtime.txt)

Accuracy is reported per grader view (`exact` = MathArena's published grader = the paper's headline column) as avg@k
with a percentile bootstrap over PROBLEMS (not samples). Cost is reported separately: generated tokens of all
rollouts, engine throughput, peak VRAM. Label: PILOT for --limit runs, FULL otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_experiments.stats import bootstrap_ci

PAPER = {"aime26": 94.01, "hmmt26": 83.62}  # technical report, Table 5, final stage (exact / MathArena grader)


def accuracy(samples: list[dict]) -> dict:
    """avg@k per grader view: mean over problems of the per-problem mean over rollouts."""
    views: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        for name, verdict in (s.get("protocols") or {}).items():
            if verdict.get("correct") is not None:
                views[name][s["problem_index"]].append(1.0 if verdict["correct"] else 0.0)
        for name in ("strict", "lenient", "permissive", "format_ok"):
            if name in s:
                views["ladder:" + name][s["problem_index"]].append(1.0 if s[name] else 0.0)
    out = {}
    for name, per in sorted(views.items()):
        per_problem = [sum(v) / len(v) for _, v in sorted(per.items())]
        mean, low, high = bootstrap_ci(per_problem)
        out[name] = {
            "avg_at_k_pct": round(100 * mean, 2),
            "ci95_pct": [round(100 * low, 2), round(100 * high, 2)],
            "problems": len(per_problem),
            "samples": sum(len(v) for v in per.values()),
            "per_problem_pct": [round(100 * x, 1) for x in per_problem],
        }
    return out


def tokens(samples: list[dict]) -> dict:
    t = [s["completion_tokens"] for s in samples if s.get("completion_tokens") is not None]
    if not t:
        return {"missing": True}
    q = statistics.quantiles(t, n=20)
    return {
        "generated_total": sum(t),
        "mean": round(statistics.mean(t)),
        "median": round(statistics.median(t)),
        "p95": round(q[18]),
        "max": max(t),
        "min": min(t),
        "truncated": sum(1 for s in samples if s.get("truncated")),
        "finish_reason": dict(Counter(str(s.get("finish_reason")) for s in samples)),
    }


def gpu(env: Path) -> dict:
    path = env / "gpu-monitor.csv"
    if not path.exists():
        return {}
    rows = list(csv.reader(path.open(encoding="utf-8")))
    head = [h.strip() for h in rows[0]]
    used = [float(r[head.index("memory.used [MiB]")].split()[0]) for r in rows[1:] if len(r) == len(head)]
    util = [float(r[head.index("utilization.gpu [%]")].split()[0]) for r in rows[1:] if len(r) == len(head)]
    return {"peak_vram_mib": max(used, default=None), "mean_util_pct": round(statistics.mean(util), 1) if util else None}


def engine(env: Path, run_id: str) -> dict:
    """vLLM Prometheus counters scraped every 30 s during the run (gpu/aime26_reproduction.sh)."""
    path = env / "vllm-metrics.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("run") == run_id]
    if not rows:
        return {}

    def series(prefix):  # gauges can carry several label series per scrape: max per scrape, never a sum
        return [max(vals) for r in rows if (vals := [v for k, v in r["m"].items() if k.startswith(prefix)
                                                      and "_bucket" not in k])]

    last = rows[-1]["m"]

    def final(prefix):
        return sum(v for k, v in last.items() if k.startswith(prefix) and "_bucket" not in k)

    span = rows[-1]["t"] - rows[0]["t"]
    gen = final("vllm:generation_tokens_total")
    first_gen = sum(v for k, v in rows[0]["m"].items() if k.startswith("vllm:generation_tokens_total"))
    return {
        "max_running": max(series("vllm:num_requests_running"), default=None),
        "max_kv_cache_usage": max(series("vllm:kv_cache_usage_perc") or series("vllm:gpu_cache_usage_perc"), default=None),
        "preemptions": final("vllm:num_preemptions_total"),
        "generation_tokens": gen,
        "prompt_tokens": final("vllm:prompt_tokens_total"),
        "mean_generation_throughput_tok_s": round((gen - first_gen) / span, 1) if span > 0 else None,
        "wall_minutes": round(span / 60, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="limite-evals/outputs/eval/<run-id>")
    ap.add_argument("--env", type=Path, default=None, help="evidence dir (gpu-monitor.csv, vllm-metrics.jsonl, ...)")
    ap.add_argument("--taskset", default="aime26")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    samples = [json.loads(line) for line in (run / "metrics" / "samples.jsonl").open(encoding="utf-8") if line.strip()]
    report = {
        "taskset": args.taskset,
        "run_id": run.name,
        "label": "PILOT" if run.name.startswith("pilot") else "FULL",
        "complete": (run / "metrics" / "summary.json").exists(),
        "samples": len(samples),
        "paper_exact_pct": PAPER.get(args.taskset),
        "accuracy": accuracy(samples),
        "tokens": tokens(samples),
    }
    if args.env:
        report["engine"] = engine(args.env, run.name)
        report["gpu"] = gpu(args.env)
        runtime = args.env / "runtime.txt"
        if runtime.exists():
            report["runtime"] = runtime.read_text(errors="replace").strip().splitlines()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    ex = report["accuracy"].get("exact", {})
    print(f"{report['label']} {run.name}: exact avg@k {ex.get('avg_at_k_pct')} % (95 % CI {ex.get('ci95_pct')}), "
          f"paper {report['paper_exact_pct']} %, {report['samples']} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
