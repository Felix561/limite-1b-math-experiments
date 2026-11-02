"""Value-model study: every number from the retrieved GPU runs (CPU only). Design: docs/preregistration.md.

    python scripts/analyze_value_study.py --runs RUN_DIR [RUN_DIR ...] --inputs work/inputs --out results/value_model

A run directory is what one GPU job wrote (gpu/value_study.sh): it contains out/gen, out/branch and/or
out/judge-*. Writes analysis.json. The dev pool chooses the primary judge and fits the probe; the evaluated sets
(AIME 2026, HMMT Feb 2026) are only scored.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from limite_experiments.probe import grouped_folds, ridge_fit, ridge_predict
from limite_experiments.selection import COST_RANK, FEATURES, JUDGES, Rollout, frontier, matched_delta, select_policy
from limite_experiments.stats import auroc, bootstrap_ci, within_problem_auroc

GRID = 256
ABS = (512, 1024, 2048, 4096, 8192, 16384)
POLICY_C = (1024, 2048, 4096, 8192)
FRACS = (0.05, 0.15, 0.35, 0.6)
PRIMARY_C, PRIMARY_B, PRIMARY_M = 4096, 8, 2
VALUE_JUDGE_COST = 0.1  # token-equivalents per prefix token scored by the separate value model


def jl(path: Path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        return [json.loads(line) for line in f]


def frac_cut(f: float, n: int) -> int:
    return min(n - 1, max(1, round(f * n)))


class Judged:
    """Judge features (and probe hidden states) of all judged rollouts, by key."""

    def __init__(self, runs: list[Path]):
        self.rows, self.hid = {}, {}
        for run in runs:
            for f in sorted(run.glob("out/judge*/judge-*.jsonl")):
                for r in jl(f):
                    self.rows[r["key"]] = r
            for f in sorted(run.glob("out/judge*/hidden-*.npz")):
                with np.load(f) as z:
                    for k in z.files:
                        self.hid[k] = z[k].astype(np.float32)

    def feature(self, key: str, name: str, t: int) -> float:
        r = self.rows[key]
        j = FEATURES.index(name)
        if t >= r["n"]:
            return r["features"][-1][j]
        i = max(0, t // GRID - 1) if t >= GRID else 0
        return r["features"][i][j]

    def hidden(self, key: str, t: int, layer: int):
        r = self.rows[key]
        keep = r["keep"]
        pos = [i for i, k in enumerate(keep) if k <= t] or [0]
        i = pos[-1] if t not in keep else keep.index(t)
        return self.hid[key][i, layer]


def judge_score(J: Judged, name: str, probes: dict | None = None):
    if name == "probe":
        def s(r, t):
            fitted = sorted(probes)
            c = max([x for x in fitted if x <= t], default=fitted[0])  # the probe of the nearest checkpoint <= t
            model, layer = probes[c]
            return float(ridge_predict(model, J.hidden(r.key, t, layer)[None])[0])
        return s
    feat, sign = JUDGES[name]
    return lambda r, t: sign * J.feature(r.key, feat, t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--orders", type=int, default=200)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    J = Judged(args.runs)
    res: dict = {"judged_rollouts": len(J.rows)}

    # --- rollouts per evaluated set --------------------------------------------------------------------------------------
    sets: dict[str, dict[str, list[Rollout]]] = {"aime26": defaultdict(list), "hmmt26": defaultdict(list)}
    for r in jl(args.inputs / "aime26.jsonl.gz"):
        if r["key"] in J.rows:
            sets["aime26"][str(r["problem"])].append(Rollout(r["key"], str(r["problem"]), r["tokens"], r["exact"],
                                                             r["answer"]))
    hmmt_summaries = []
    for run in args.runs:
        for f in run.glob("out/gen/samples.jsonl"):
            for r in jl(f):
                key = f"{r['taskset']}:{r['problem']}:{r['rollout']}"
                if key in J.rows:
                    sets["hmmt26"][str(r["problem"])].append(Rollout(key, str(r["problem"]), r["tokens"],
                                                                     bool(r.get("exact")), r["answer"]))
        for f in run.glob("out/gen/summary.json"):
            hmmt_summaries.append(json.loads(f.read_text()))
    res["hmmt26_gen"] = hmmt_summaries
    for name, probs in sets.items():
        allr = [r for rs in probs.values() for r in rs]
        if allr:
            per = [np.mean([r.correct for r in rs]) for rs in probs.values()]
            m, lo, hi = bootstrap_ci(per)
            res[f"{name}_baseline"] = {"problems": len(probs), "rollouts": len(allr), "avg_exact_pct": 100 * m,
                                       "ci95": [100 * lo, 100 * hi],
                                       "mean_tokens": float(np.mean([r.tokens for r in allr])),
                                       "per_problem_pct": {p: 100 * float(np.mean([r.correct for r in rs]))
                                                           for p, rs in sorted(probs.items(), key=lambda x: int(x[0]))}}

    # --- dev pool: probe fit and judge choice ---------------------------------------------------------------------------
    dev = [r for r in jl(args.inputs / "dev.jsonl.gz") if r["key"] in J.rows]
    for r in dev:
        r["problem"] = r["key"].split(":")[1]
    probes, dev_report = {}, {}
    for c in ABS:
        use = [r for r in dev if J.rows[r["key"]]["n"] > c and r["finish_reason"] == "stop"]
        if len(use) < 50:
            continue
        y = np.array([r["correct"] for r in use], float)
        groups = [r["problem"] for r in use]
        best = None
        for layer in range(3):
            X = np.stack([J.hidden(r["key"], c, layer) for r in use])
            cv = np.zeros(len(use))
            for idx in grouped_folds(groups):
                tr = np.setdiff1d(np.arange(len(use)), idx)
                cv[idx] = ridge_predict(ridge_fit(X[tr], y[tr], 1e-2), X[idx])
            a = auroc(cv, y > 0.5)
            if best is None or a > best[0]:
                best = (a, layer, cv, X)
        a, layer, cv, X = best
        probes[c] = (ridge_fit(X, y, 1e-2), layer)
        scores = {"probe": dict(zip([r["key"] for r in use], cv, strict=True))}
        for name in JUDGES:
            sc = judge_score(J, name)
            scores[name] = {r["key"]: sc(Rollout(r["key"], r["problem"], 0, r["correct"], None), c) for r in use}
        by_p = defaultdict(list)
        for r in use:
            by_p[r["problem"]].append(r)
        pair, glob = {}, {}
        for name, s in scores.items():
            wins = [(1.0 if s[a_["key"]] > s[b_["key"]] else 0.5 if s[a_["key"]] == s[b_["key"]] else 0.0)
                    for rs in by_p.values() if len(rs) == 2 and rs[0]["correct"] != rs[1]["correct"]
                    for a_, b_ in [(rs[0], rs[1]) if rs[0]["correct"] else (rs[1], rs[0])]]
            pair[name] = (float(np.mean(wins)) if wins else math.nan, len(wins))
            glob[name] = auroc([s[r["key"]] for r in use], [r["correct"] for r in use])
        dev_report[c] = {"rollouts": len(use), "probe_layer": layer, "probe_cv_auroc": a,
                         "pairwise": pair, "global_auroc": glob}
    res["dev"] = dev_report
    if PRIMARY_C in dev_report:
        pc = dev_report[PRIMARY_C]["pairwise"]
        primary = max(pc, key=lambda k: (-1 if math.isnan(pc[k][0]) else round(pc[k][0], 4), -COST_RANK[k]))
    else:  # only without dev data
        primary = "value"
    res["primary_judge"] = primary

    judges = list(JUDGES) + (["probe"] if probes else [])
    score_fns = {n: judge_score(J, n, probes) for n in judges}

    # --- AUROC curves on the evaluated sets: how early can a judge tell? ------------------------------------------------
    curves = {}
    for sname, probs in sets.items():
        if not probs:
            continue
        cs = {}
        for name in judges:
            row = {}
            for c in ABS:
                groups = {p: [(score_fns[name](r, c), r.correct) for r in rs if r.tokens > c] for p, rs in probs.items()}
                m, per = within_problem_auroc(groups)
                row[str(c)] = {"auroc": m, "ci95": list(bootstrap_ci(per)[1:]) if per else None, "problems": len(per)}
            for f in FRACS:
                groups = {p: [(score_fns[name](r, frac_cut(f, J.rows[r.key]["n"])), r.correct) for r in rs]
                          for p, rs in probs.items()}
                m, per = within_problem_auroc(groups)
                row[f"f{f}"] = {"auroc": m, "ci95": list(bootstrap_ci(per)[1:]) if per else None, "problems": len(per)}
            groups = {p: [(score_fns[name](r, 10**9), r.correct) for r in rs] for p, rs in probs.items()}
            m, per = within_problem_auroc(groups)
            row["final"] = {"auroc": m, "ci95": list(bootstrap_ci(per)[1:]) if per else None, "problems": len(per)}
            cs[name] = row
        curves[sname] = cs
    res["auroc_curves"] = curves

    # --- selection policies vs plain sampling at matched compute -------------------------------------------------------
    pol = {}
    for sname, probs in sets.items():
        if not probs:
            continue
        n_min = min(len(rs) for rs in probs.values())
        front = frontier(probs, list(range(1, n_min + 1)), args.orders)
        fr = {k: {"voted": float(np.mean([v[0] for v in d.values()])),
                  "per_sample": float(np.mean([v[1] for v in d.values()])),
                  "tokens": float(np.mean([v[2] for v in d.values()]))} for k, d in front.items()}
        entries = []
        for name in judges + ["oracle"]:
            fn = (lambda r, t: float(r.correct)) if name == "oracle" else score_fns[name]
            cost = VALUE_JUDGE_COST if name.startswith("value") else 0.0
            for c in POLICY_C:
                for B in (4, 8, 16):
                    for m in (1, 2, 4):
                        if m >= B or B > n_min:
                            continue
                        per = select_policy(probs, fn, c, B, m, args.orders, judge_cost_per_token=cost)
                        d, deltas, tok = matched_delta(per, front)
                        e = {"judge": name, "c": c, "B": B, "m": m, "tokens": tok,
                             "voted": float(np.mean([v[0] for v in per.values()])),
                             "per_sample": float(np.mean([v[1] for v in per.values()])), "delta_vs_frontier": d}
                        if (c, B, m) == (PRIMARY_C, PRIMARY_B, PRIMARY_M) or name == "oracle":
                            e["delta_ci95"] = list(bootstrap_ci(deltas)[1:])
                        if m in (1, 2):  # selection lift: kept samples vs random samples, per problem
                            base = front[m]
                            lift = [per[p][1] - base[p][1] for p in per]
                            e["lift_per_sample"] = float(np.mean(lift))
                            e["lift_ci95"] = list(bootstrap_ci(lift)[1:])
                        entries.append(e)
        pol[sname] = {"frontier": fr, "policies": entries}
        # EXPLORATORY (not pre-registered): a late checkpoint, and the value model as a verifier of finished samples
        late = []
        for name in ("value", "value_mean", "oracle"):
            fn = (lambda r, t: float(r.correct)) if name == "oracle" else score_fns[name]
            for B, m in ((4, 1), (8, 1), (8, 2), (16, 2)):
                if B > n_min:
                    continue
                per = select_policy(probs, fn, 16384, B, m, args.orders,
                                    judge_cost_per_token=VALUE_JUDGE_COST if name != "oracle" else 0.0)
                d, deltas, tok = matched_delta(per, front)
                late.append({"judge": name, "c": 16384, "B": B, "m": m, "tokens": tok, "delta_vs_frontier": d,
                             "delta_ci95": list(bootstrap_ci(deltas)[1:]),
                             "voted": float(np.mean([v[0] for v in per.values()]))})
        verifier = {}
        for k in (2, 4, 8, 16):
            if k > n_min:
                continue
            per = select_policy(probs, score_fns["value"], 10**9, k, 1, args.orders,
                                judge_cost_per_token=VALUE_JUDGE_COST)  # best of k by the final value
            maj = front[k]
            diff = [per[p][0] - maj[p][0] for p in per]
            verifier[str(k)] = {"best_of_k_by_value": float(np.mean([v[0] for v in per.values()])),
                                "maj_at_k": float(np.mean([v[0] for v in maj.values()])),
                                "diff_ci95": list(bootstrap_ci(diff))}
        pol[sname]["exploratory"] = {"late_checkpoint": late, "value_as_verifier": verifier}
    res["policies"] = pol

    # --- commitment curve: restart stored AIME26 rollouts from a prefix --------------------------------------------------
    plan = {(r["key"], r["f"]): r for r in jl(args.inputs / "plan.jsonl.gz")}
    cont = defaultdict(list)
    for run in args.runs:
        for f in run.glob("out/branch/continuations.jsonl"):
            for r in jl(f):
                cont[(r["key"], r["f"])].append(bool(r.get("exact")))
    if cont:
        aime = {r.key: r for rs in sets["aime26"].values() for r in rs}
        full = defaultdict(list)  # base rate per problem from all 960 stored rollouts (judged or not)
        for r in jl(args.inputs / "aime26.jsonl.gz"):
            full[str(r["problem"])].append(r["exact"])
        base = {p: float(np.mean(v)) for p, v in full.items()}
        cc, paired = {}, {}
        for cls in (False, True):
            label = "success" if cls else "failure"
            keys = sorted({k for (k, f), it in plan.items() if it["orig_exact"] == cls and cont.get((k, f))})
            pts = {"0": bootstrap_ci([base[k.split(":")[1]] for k in keys])}
            for f in FRACS:
                vals = [float(np.mean(cont[(k, f)])) for k in keys if cont.get((k, f))]
                pts[str(f)] = bootstrap_ci(vals) + (len(vals),)
            pts["1"] = (float(cls), float(cls), float(cls))
            cc[label] = pts
            # paired: each prefix's P(correct | prefix) minus its own problem's base rate, bootstrap over rollouts
            paired[label] = {}
            for f in FRACS:
                d = [float(np.mean(cont[(k, ff)])) - base[k.split(":")[1]]
                     for (k, ff), it in plan.items() if ff == f and it["orig_exact"] == cls and cont.get((k, ff))]
                paired[label][str(f)] = bootstrap_ci(d) + (len(d),)
        corr = {}
        for name in judges:
            row = {}
            for f in FRACS:
                xs, ys = [], []
                for (k, ff), v in cont.items():
                    if ff == f and k in aime and k in J.rows:
                        xs.append(score_fns[name](aime[k], frac_cut(f, J.rows[k]["n"])))
                        ys.append(float(np.mean(v)))
                if len(xs) > 5:
                    rx, ry = np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys))
                    row[str(f)] = float(np.corrcoef(rx, ry)[0, 1])
            corr[name] = row
        res["commitment"] = {"curves": cc, "paired_vs_base_rate": paired, "spearman_judge_vs_mc_value": corr,
                             "continuations": sum(len(v) for v in cont.values())}
    (args.out / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"primary_judge": primary, **{k: {kk: vv for kk, vv in res[k].items() if kk != "per_problem_pct"}
                                                   for k in ("aime26_baseline", "hmmt26_baseline") if k in res}},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
