"""Render the RM paper's figures from the gold artifacts.

Every figure is drawn from files the evaluation already wrote, so a figure can
never show a number the tables do not:

* ``data/gold/evaluation_youtube.json`` - C-index intervals and the paired
  bootstrap on differences;
* ``data/gold/predictions_youtube.parquet`` - out-of-fold predictions, for the
  calibration plot the plan promises;
* the Cohort A analysis frame, rebuilt from silver, for the Kaplan-Meier curves.

Cohort A only. Nothing here reads Cohort B.

Usage::

    python -m cdc.eval.figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cdc.config import ROOT, path_for

OUT_DIR = ROOT / "paper" / "rm" / "figures"

# Reference palette (dataviz skill, light mode). Slots 1-3 validate all-pairs
# for colour-vision deficiency; slot 3 is below 3:1 contrast on the surface, so
# every line carrying it is also labelled directly.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TIER_COLOUR = {"micro": ORANGE, "mid": AQUA, "large": BLUE}
TIER_LABEL = {"micro": "micro (<10k)", "mid": "mid (10k-500k)", "large": "large (>500k)"}

PRETTY = {
    "random_survival_forest": "Random Survival Forest",
    "weibull_aft": "Weibull AFT",
    "subscriber_only": "Creator size only",
    "peak_velocity": "Peak velocity",
    "km_median": "KM median",
    "constant_48h": "Constant 48h",
    "loglinear_uncensored": "Log-linear (deaths only)",
    "gbm_uncensored": "Gradient boosting (deaths only)",
}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.titlesize": 11, "figure.titleweight": "bold",
        "legend.frameon": False,
    })
    return plt


def _frame() -> pd.DataFrame:
    from cdc.models import dataset
    return dataset.build(platform="youtube", cohort="A").frame


def km_overall(plt, df: pd.DataFrame) -> Path:
    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter().fit(df["t_death"], df["event_observed"].astype(int))
    sf, ci = km.survival_function_, km.confidence_interval_
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post",
                    color=BLUE, alpha=0.15, linewidth=0)
    ax.step(sf.index, sf.iloc[:, 0], where="post", color=BLUE, linewidth=2)
    med = float(km.median_survival_time_)
    ax.axhline(0.5, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.annotate(f"median {med:.0f}h after the landmark", xy=(med, 0.5),
                xytext=(med + 25, 0.62), color=INK_2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_xlim(0, df["t_death"].max())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("hours after the 7-hour landmark")
    ax.set_ylabel("share of posts still receiving attention")
    _title(fig, f"Figure 1. Survival past the landmark (n = {len(df)}, "
                 f"{int(df['event_observed'].sum())} deaths)")
    return _save(fig, "fig1_km_overall.png")


def km_by_tier(plt, df: pd.DataFrame) -> Path:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for tier in ("micro", "mid", "large"):
        g = df[df["stratum_tier"] == tier]
        if g.empty:
            continue
        km = KaplanMeierFitter().fit(g["t_death"], g["event_observed"].astype(int))
        sf = km.survival_function_
        ax.step(sf.index, sf.iloc[:, 0], where="post", color=TIER_COLOUR[tier],
                linewidth=2, label=TIER_LABEL[tier])
        # Direct label at the curve's right end, in ink rather than series colour.
        ax.annotate(f"{TIER_LABEL[tier]}  (n = {len(g)})",
                    xy=(sf.index[-1], float(sf.iloc[-1, 0])),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    lr = multivariate_logrank_test(df["t_death"], df["stratum_tier"],
                                   df["event_observed"].astype(int))
    ax.set_xlim(0, df["t_death"].max() * 1.32)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("hours after the 7-hour landmark")
    ax.set_ylabel("share still receiving attention")
    _title(fig, "Figure 4. Survival by creator tier (exploratory)")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.99, 0.62, f"log-rank p = {lr.p_value:.1e}", transform=ax.transAxes,
            ha="right", fontsize=9, color=INK_2)
    return _save(fig, "fig4_km_by_tier.png")


def cindex_dots(plt, ev: dict) -> Path:
    res = ev["results"]
    # A constant predictor cannot rank, so it scores 0.5 within every fold. The
    # pooled out-of-fold value for km_median (about 0.47) is an artefact of
    # its fold constants differing by minutes; see paper section 4.3.
    constants = {"constant_48h", "km_median"}
    c = {m: (0.5 if m in constants else res[m]["c_index"]) for m in res}
    order = sorted(res, key=lambda m: (c[m], m))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, m in enumerate(order):
        r = res[m]
        col = BLUE if r.get("role") == "primary" else MUTED
        if m in constants:
            ax.plot(0.5, i, "o", markersize=8, markerfacecolor=SURFACE,
                    markeredgecolor=MUTED, markeredgewidth=1.5)
            ax.annotate("0.500  (constant: cannot rank)", xy=(0.5, i), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=INK_2)
            continue
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=col, linewidth=2,
                solid_capstyle="round")
        ax.plot(c[m], i, "o", color=col, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{c[m]:.3f}", xy=(r["ci_high"], i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.axvline(0.5, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PRETTY.get(m, m) for m in order], color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Harrell's C-index, 95% creator-bootstrap interval (0.5 = chance)")
    _title(fig, "Figure 2. Discrimination on Cohort A (primary models in blue)")
    return _save(fig, "fig2_cindex.png")


def h2_deltas(plt, ev: dict) -> Path:
    pb = ev["paired_bootstrap"]
    keys = [k for k in pb if pb[k].get("n_boot") and not k.endswith("km_median")]
    keys.sort(key=lambda k: (k.split(" vs ")[0], pb[k]["delta_c_index"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, k in enumerate(keys):
        d = pb[k]
        h2 = k.endswith("vs subscriber_only")
        col = ORANGE if h2 else MUTED
        ax.plot([d["delta_c_ci_low"], d["delta_c_ci_high"]], [i, i], color=col,
                linewidth=2, solid_capstyle="round")
        ax.plot(d["delta_c_index"], i, "o", color=col, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{d['delta_c_index']:+.3f}", xy=(d["delta_c_ci_high"], i),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    m, b = zip(*(k.split(" vs ") for k in keys))
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{PRETTY[x]} - {PRETTY[y]}" for x, y in zip(m, b)],
                       fontsize=9, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("difference in C-index, paired creator bootstrap 95% interval")
    _title(fig, "Figure 3. Model minus baseline, C-index (H2 tests in orange)")
    return _save(fig, "fig3_paired_deltas.png")


def calibration(plt, preds: pd.DataFrame, n_bins: int = 10) -> Path:
    """Predicted against observed, censoring-aware.

    Posts are grouped by decile of predicted time. Within each group the
    observed value is the Kaplan-Meier median of the actual remaining lifetime,
    so censored posts contribute what is known about them instead of being
    dropped. A perfectly calibrated model sits on the diagonal.
    """
    from lifelines import KaplanMeierFitter
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    lo, hi = np.inf, 0.0
    for m, col in (("random_survival_forest", BLUE), ("weibull_aft", ORANGE)):
        s = preds[preds["model"] == m].copy()
        if s.empty:
            continue
        s["bin"] = pd.qcut(s["predicted"].rank(method="first"), n_bins, labels=False)
        xs, ys = [], []
        for _, g in s.groupby("bin"):
            km = KaplanMeierFitter().fit(g["duration"], g["event"].astype(int))
            med = float(km.median_survival_time_)
            if np.isfinite(med) and med > 0:
                xs.append(float(g["predicted"].median()))
                ys.append(med)
        ax.plot(xs, ys, "-o", color=col, linewidth=2, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=PRETTY[m])
        if xs:
            ax.annotate(PRETTY[m], xy=(xs[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=INK_2)
            lo, hi = min(lo, *xs, *ys), max(hi, *xs, *ys)
    top = 200.0
    ax.plot([0, top], [0, top], color=AXIS, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(top * 0.97, top * 0.9, "perfect calibration", rotation=38, ha="right",
            va="top", fontsize=8, color=MUTED)
    ax.set_xlim(0, top * 1.3)
    ax.set_ylim(0, top)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xlabel("predicted remaining hours (decile median)")
    ax.set_ylabel("observed remaining hours (KM median)")
    _title(fig, "Figure 5. Calibration, Cohort A")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, "fig5_calibration.png")


def _title(fig, text: str) -> None:
    fig.suptitle(text, x=0.015, ha="left", color=INK)


def _save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


def main(argv: list[str] | None = None) -> int:
    gold = path_for("gold_dir")
    ev = json.loads((gold / "evaluation_youtube.json").read_text(encoding="utf-8"))
    if ev.get("cohort") != "A" or ev.get("provisional"):
        raise SystemExit("evaluation_youtube.json is not a matured Cohort A run")
    preds = pd.read_parquet(gold / "predictions_youtube.parquet")
    plt = _style()
    df = _frame()
    for p in (km_overall(plt, df), km_by_tier(plt, df), cindex_dots(plt, ev),
              h2_deltas(plt, ev), calibration(plt, preds)):
        print(f"  wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
