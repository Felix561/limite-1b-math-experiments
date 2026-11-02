"""Draw the four README figures as small, deterministic SVGs (white background, text kept as text).

    python scripts/make_figures.py          # writes figures/*.svg
    python scripts/make_figures.py --png    # also writes PNG previews outside the repo (<tmp>/figs)

Inputs: results/value_model/analysis.json, results/value_model/controls.json and, if present,
results/reproduction/aime26.json (otherwise its two published numbers below are used).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
PNG_DIR = Path(tempfile.gettempdir()) / "figs"

# Two accent colours (blue, orange) plus greys; text always uses the ink tones, never the accents.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GREY, SOFT, GRID, WASH = "#1f2328", "#5f5f5b", "#9a9a96", "#c4c4c0", "#ececea", "#f3f3f1"
MINUS = "\N{MINUS SIGN}"
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GREY,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "limite-1b-math-experiments",
}
AIME_FALLBACK = (92.40, (84.27, 98.12), 94.01)  # exact avg@32 (%), 95 % CI, paper
HMMT_PAPER = 83.62
POLICY = (4096, 8, 2)  # judge reads after c tokens; B samples started; m finished


# ---------------------------------------------------------------- helpers


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def signed(x: float, nd: int = 0) -> str:
    """Number with an explicit sign and a true minus."""
    return "0" if round(x, nd) == 0 else f"{x:+.{nd}f}".replace("-", MINUS)


def make_axes(width: float, height: float, left: float, right: float, top: float, bottom: float):
    """Figure of `width` x `height` inches with the axes inset by the given margins (inches)."""
    fig = Figure(figsize=(width, height))
    ax = fig.add_axes([left / width, bottom / height, 1 - (left + right) / width, 1 - (top + bottom) / height])
    return fig, ax


def header(fig: Figure, main: str, note: str | None = None) -> None:
    """Short panel label at the top left: one line in ink, an optional second line in muted ink."""
    w, h = fig.get_size_inches()
    fig.text(0.12 / w, 1 - 0.10 / h, main, ha="left", va="top", size=8.5)
    if note:
        fig.text(0.12 / w, 1 - 0.30 / h, note, ha="left", va="top", size=7.5, color=MUTED)


def label(ax, text: str, xy, offset=(0, 0), ha="left", va="center", color=INK, size=8, xycoords="data") -> None:
    """Text anchored to a data point, shifted by `offset` points; may sit outside the axes."""
    ax.annotate(
        text,
        xy,
        xycoords=xycoords,
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        color=color,
        size=size,
        annotation_clip=False,
    )


def row_label(ax, y: float, name: str, note: str = "") -> None:
    """Row name (ink) with an optional muted qualifier beneath it, right-aligned left of the axes."""
    tf = ax.get_yaxis_transform()
    label(ax, name, (0, y), (-9, 4.5 if note else 0), ha="right", xycoords=tf)
    if note:
        label(ax, note, (0, y), (-9, -5.5), ha="right", color=MUTED, size=7.5, xycoords=tf)


def dot(ax, x, y, color, marker="o", size=6.5, hollow=False):
    """Point marker: filled with a white ring, or hollow with an outline in `color`."""
    face, edge, width = ("white", color, 1.3) if hollow else (color, "white", 1.1)
    ax.plot([x], [y], marker, ms=size, mfc=face, mec=edge, mew=width, ls="none", zorder=4, clip_on=False)


def interval(ax, y, lo, hi, color, lw=1.8):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=3, clip_on=False)


def band_and_line(ax, x, mean, lo, hi, color, marker, name, first_marker=0, lw=1.6):
    """Line, markers from index `first_marker` on, and an optional confidence band (`lo`/`hi` may be None)."""
    if lo is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0, zorder=2)
    ax.plot(x, mean, color=color, lw=lw, zorder=3, clip_on=False)
    for xi, yi in list(zip(x, mean, strict=True))[first_marker:]:
        dot(ax, xi, yi, color, marker, size=5.5)
    label(ax, name, (x[-1], mean[-1]), (9, 0))


# ---------------------------------------------------------------- figures


def reproduction_rows(analysis: dict):
    """(name, protocol, ours %, 95 % CI, reported %) per benchmark."""
    aime = AIME_FALLBACK
    path = ROOT / "results/reproduction/aime26.json"
    if path.exists():
        rep = json.loads(path.read_text(encoding="utf-8"))
        if rep.get("label", "FULL") == "FULL" and "exact" in rep.get("accuracy", {}):
            ex = rep["accuracy"]["exact"]
            aime = (ex["avg_at_k_pct"], tuple(ex["ci95_pct"]), rep["paper_exact_pct"])
    hmmt = analysis["hmmt26_baseline"]
    return [
        ("AIME 2026", "avg@32", *aime),
        ("HMMT Feb 2026", "avg@24", hmmt["avg_exact_pct"], hmmt["ci95"], HMMT_PAPER),
    ]


def fig_reproduction(rows) -> Figure:
    fig, ax = make_axes(6.8, 2.1, left=1.45, right=0.4, top=0.1, bottom=0.55)
    ax.set_xlim(63, 100)
    ax.set_ylim(-0.6, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xticks(range(65, 101, 5))
    ax.grid(axis="x")
    ax.set_xlabel("Accuracy (%)", labelpad=5)
    for i, (name, protocol, mean, (lo, hi), paper) in enumerate(rows):
        y = 1 - i
        row_label(ax, y, name, protocol)
        yo, yr = y + 0.17, y - 0.17  # this study above, reported value below
        interval(ax, yo, lo, hi, BLUE)
        dot(ax, mean, yo, BLUE)
        label(ax, f"{mean:.1f}  [{lo:.1f}, {hi:.1f}]", (mean, yo), (0, 7), ha="center", va="bottom")
        label(ax, "this study", (lo, yo), (-8, 0), ha="right", color=MUTED, size=7.5)
        dot(ax, paper, yr, INK, "D", size=5.5, hollow=True)
        label(ax, f"reported  {paper:.1f}", (paper, yr), (0, -7), ha="center", va="top", color=MUTED, size=7.5)
    return fig


def fig_commitment(analysis: dict) -> Figure:
    paired = analysis["commitment"]["paired_vs_base_rate"]
    fig, ax = make_axes(6.8, 3.3, left=0.6, right=1.5, top=0.62, bottom=0.58)
    header(
        fig,
        "Change in P(correct) vs. the problem's base rate (pp)",
        "AIME 2026, 24 failed and 24 successful solutions, 2 fresh continuations per prefix",
    )
    ax.set_xlim(0, 60)
    ax.set_ylim(-46, 24)
    ax.set_xticks([0, 5, 15, 35, 60])
    ax.set_yticks(range(-40, 21, 10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: signed(v)))
    ax.grid(axis="y")
    ax.set_xlabel("Share of the original solution kept as prefix (%)", labelpad=5)
    ax.axvspan(35, 60, color=WASH, lw=0, zorder=0)
    ax.text(47.5, 22.5, "outcome set", ha="center", va="top", size=7.5, color=MUTED)
    ax.axhline(0, color=GREY, lw=0.8, zorder=1)
    ax.plot([0], [0], "o", ms=3.5, color=INK, zorder=5, clip_on=False)  # 0 % kept: change is 0 by definition
    for key, color, marker, name in (
        ("success", BLUE, "o", "successful solutions"),
        ("failure", ORANGE, "s", "failed solutions"),
    ):
        pts = sorted(paired[key].items(), key=lambda kv: float(kv[0]))
        x = [0.0] + [100 * float(k) for k, _ in pts]
        col = [[0.0] + [100 * v[i] for _, v in pts] for i in (0, 1, 2)]  # mean, lo, hi; 0 % kept = no change
        band_and_line(ax, x, col[0], col[1], col[2], color, marker, name, first_marker=1)
        label(ax, f"{signed(col[0][-1])} pp at 60 %", (x[-1], col[0][-1]), (9, -11), color=MUTED, size=7.5)
    return fig


def fig_auroc(analysis: dict) -> Figure:
    curves = analysis["auroc_curves"]
    points = ["f0.05", "f0.15", "f0.35", "f0.6", "final"]
    x = [5, 15, 35, 60, 100]

    def series(ds: str, judge: str):
        rec = [curves[ds][judge][p] for p in points]
        ci = [r["ci95"] for r in rec]
        return [r["auroc"] for r in rec], ([c[0] for c in ci], [c[1] for c in ci]) if all(ci) else (None, None)

    fig, ax = make_axes(6.8, 3.3, left=0.55, right=1.95, top=0.62, bottom=0.58)
    n = {ds: curves[ds]["value"]["final"]["problems"] for ds in ("hmmt26", "aime26")}
    header(
        fig,
        "Within-problem AUROC",
        f"Problems with mixed outcomes: {n['hmmt26']} (HMMT Feb 2026), {n['aime26']} (AIME 2026)",
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks(x)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(axis="y")
    ax.set_xlabel("Share of the solution seen by the judge (%)", labelpad=5)
    ax.axhline(0.5, color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(100, 0.508, "chance", ha="right", va="bottom", size=7.5, color=MUTED)
    for judge, name, marker, color in (
        ("conf_window", "model confidence", "D", SOFT),
        ("probe", "hidden-state probe", "^", GREY),
    ):  # grey context, HMMT only
        mean, _ = series("hmmt26", judge)
        ax.plot(x, mean, color=color, lw=1.1, zorder=2, clip_on=False)
        ax.plot(x, mean, marker, ms=3.5, color=color, ls="none", zorder=2, clip_on=False)
        label(ax, name, (x[-1], mean[-1]), (9, 0), color=MUTED, size=7.5)
    for ds, color, marker, name in (
        ("hmmt26", BLUE, "o", "value model, HMMT Feb 2026"),
        ("aime26", ORANGE, "s", "value model, AIME 2026"),
    ):
        mean, (lo, hi) = series(ds, "value")
        band_and_line(ax, x, mean, lo, hi, color, marker, name)
    return fig


def fig_pruning(analysis: dict, controls: dict) -> Figure:
    def pick(entries, judge):
        return next(e for e in entries if e["judge"] == judge and (e["c"], e["B"], e["m"]) == POLICY)

    judged = analysis["policies"]["hmmt26"]["policies"]
    rows = [  # name, qualifier, entry, marker colour, hollow
        ("Random judge", "control", pick(controls["controls"], "random"), GREY, False),
        ("Model confidence", "", pick(judged, "conf_window"), BLUE, False),
        ("Value model", "", pick(judged, "value"), BLUE, False),
        ("Hidden-state probe", "pre-registered", pick(judged, "probe"), BLUE, False),
        ("Perfect judge", "upper bound", pick(judged, "oracle"), MUTED, True),
    ]
    ys = [4.4, 3.4, 2.4, 1.4, 0.0]  # extra gap before the unattainable bound

    fig, ax = make_axes(6.8, 3.0, left=1.6, right=1.4, top=0.72, bottom=0.55)
    header(
        fig,
        "Policy: start 8 samples, read the judge after 4,096 tokens, finish the best 2, majority vote",
        "HMMT Feb 2026",
    )
    ax.set_xlim(-8, 20)
    ax.set_ylim(-0.6, 5.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xticks(range(-5, 21, 5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: signed(v)))
    ax.grid(axis="x")
    ax.set_xlabel("Accuracy vs. plain sampling at equal compute (pp)", labelpad=5)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    label(
        ax,
        "plain sampling",
        (0, 1),
        (0, 4),
        ha="center",
        va="bottom",
        color=MUTED,
        size=7.5,
        xycoords=ax.get_xaxis_transform(),
    )
    ax.axhline(0.7, color=GRID, lw=0.8, zorder=1)
    for (name, note, e, color, hollow), y in zip(rows, ys, strict=True):
        d = 100 * e["delta_vs_frontier"]
        ci = e.get("delta_ci95")  # absent for some entries: draw the dot only
        row_label(ax, y, name, note)
        if ci:
            interval(ax, y, 100 * ci[0], 100 * ci[1], color)
        dot(ax, d, y, color, hollow=hollow, size=7)
        txt = f"{signed(d, 1)}  [{signed(100 * ci[0], 1)}, {signed(100 * ci[1], 1)}]" if ci else signed(d, 1)
        label(ax, txt, (1, y), (10, 0), color=MUTED, size=7.5, xycoords=ax.get_yaxis_transform())
    return fig


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the README figures to figures/.")
    ap.add_argument("--png", action="store_true", help=f"also write PNG previews to {PNG_DIR}")
    args = ap.parse_args()
    analysis, controls = load("results/value_model/analysis.json"), load("results/value_model/controls.json")
    FIG_DIR.mkdir(exist_ok=True)
    if args.png:
        PNG_DIR.mkdir(parents=True, exist_ok=True)
    with rc_context(STYLE):
        figs = {
            "reproduction": fig_reproduction(reproduction_rows(analysis)),
            "commitment": fig_commitment(analysis),
            "value_auroc": fig_auroc(analysis),
            "early_pruning": fig_pruning(analysis, controls),
        }
        for name, fig in figs.items():
            fig.savefig(FIG_DIR / f"{name}.svg", metadata={"Date": None})
            if args.png:
                fig.savefig(PNG_DIR / f"{name}.png", dpi=220)
            print(f"wrote figures/{name}.svg")


if __name__ == "__main__":
    main()
