"""The paper's figures render from artifacts shaped like the real ones.

The gold artifacts are not committed, so these tests build small stand-ins with
the same keys. They check that every figure is written, and pin the one rule a
figure could otherwise break silently: a constant baseline is drawn at 0.5, not
at the pooled out-of-fold value the harness records for it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdc.eval import figures


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "OUT_DIR", tmp_path)
    return tmp_path


def _ev():
    res = {
        "random_survival_forest": (0.616, 0.566, 0.672, "primary"),
        "weibull_aft": (0.589, 0.531, 0.643, "primary"),
        "constant_48h": (0.5, 0.5, 0.5, "baseline"),
        "km_median": (0.466, 0.418, 0.514, "baseline"),
        "subscriber_only": (0.594, 0.544, 0.644, "baseline"),
        "peak_velocity": (0.565, 0.519, 0.606, "baseline"),
    }
    results = {m: {"c_index": c, "ci_low": lo, "ci_high": hi, "role": r}
               for m, (c, lo, hi, r) in res.items()}
    pb = {}
    for m in ("random_survival_forest", "weibull_aft"):
        for b in ("constant_48h", "km_median", "subscriber_only", "peak_velocity"):
            pb[f"{m} vs {b}"] = {"n_boot": 10, "delta_c_index": 0.02,
                                 "delta_c_ci_low": -0.01, "delta_c_ci_high": 0.06}
    return {"results": results, "paired_bootstrap": pb}


def test_every_figure_is_written(out):
    plt = figures._style()
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "t_death": rng.exponential(80, n) + 1,
        "event_observed": rng.random(n) < 0.8,
        "stratum_tier": rng.choice(["micro", "mid", "large"], n),
    })
    preds = pd.concat([pd.DataFrame({
        "model": m, "predicted": df["t_death"] * rng.lognormal(0, 0.5, n),
        "duration": df["t_death"], "event": df["event_observed"]})
        for m in ("random_survival_forest", "weibull_aft")])
    paths = [figures.km_overall(plt, df), figures.km_by_tier(plt, df),
             figures.cindex_dots(plt, _ev()), figures.h2_deltas(plt, _ev()),
             figures.calibration(plt, preds)]
    assert all(p.exists() and p.stat().st_size > 5000 for p in paths)
    assert {p.name for p in paths} == {
        "fig1_km_overall.png", "fig2_cindex.png", "fig3_paired_deltas.png",
        "fig4_km_by_tier.png", "fig5_calibration.png"}


def test_constant_baselines_are_drawn_at_one_half(out, monkeypatch):
    plt = figures._style()
    drawn = []
    real = plt.Axes.plot

    def spy(self, *args, **kw):
        if len(args) >= 2 and np.isscalar(args[0]):
            drawn.append(float(args[0]))
        return real(self, *args, **kw)

    monkeypatch.setattr(plt.Axes, "plot", spy)
    figures.cindex_dots(plt, _ev())
    assert 0.466 not in drawn            # the pooled artefact is never plotted
    assert drawn.count(0.5) == 2         # both constants sit at 0.5
