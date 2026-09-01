"""Tests for the saturation robustness label and its reporting.

The frozen plan promises a second outcome definition and a comparison against
it. The label was always computed but never carried into the analysis frame, so
none of this could run. These tests pin the three things that would make the
robustness check quietly wrong rather than loudly broken:

* the saturation columns must be **outcomes, never features**;
* a failed curve fit must be **censored, not dropped**;
* the agreement statistics must **say when they are not interpretable**.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdc.eval.robustness import MIN_COMPARABLE, label_agreement, saturation_frame
from cdc.models.synthetic import FEATURE_COLUMNS

SAT_COLUMNS = ["t_saturation", "t_saturation_from_publish",
               "saturation_beyond_window", "fitted_A", "fitted_k"]


def _frame(n=30, seed=1):
    rng = np.random.default_rng(seed)
    death = rng.uniform(5, 200, size=n)
    return pd.DataFrame({
        "post_id": [f"p{i}" for i in range(n)],
        "creator_id": [f"c{i % 6}" for i in range(n)],
        "t_death": death,
        "event_observed": rng.random(n) < 0.6,
        "t_saturation": death * rng.uniform(0.8, 1.4, size=n),
        "saturation_beyond_window": rng.random(n) < 0.2,
        "last_observed_hours": death + rng.uniform(1, 50, size=n),
    })


# ------------------------------------------------------- outcomes, not features
def test_saturation_columns_are_never_features():
    """Saturation uses the post's whole history — as a predictor it is leakage."""
    for col in SAT_COLUMNS:
        assert col not in FEATURE_COLUMNS, (
            f"{col} is an outcome computed from the full series; "
            "it must never enter the feature whitelist")


# ----------------------------------------------------------------- agreement
def test_agreement_reports_coverage_and_correlation():
    a = label_agreement(_frame())
    assert a["status"] == "ok"
    assert a["saturation_coverage"] == pytest.approx(1.0)
    # Saturation was built as a jittered multiple of death, so ranks must agree.
    assert a["spearman_rho"] > 0.5


def test_agreement_flags_itself_as_underpowered_when_small():
    df = _frame(n=8)
    df["event_observed"] = True          # 8 comparable posts, below the minimum
    a = label_agreement(df)
    assert a["comparison_underpowered"] is True


def test_agreement_not_underpowered_when_large():
    df = _frame(n=MIN_COMPARABLE + 15)
    df["event_observed"] = True
    a = label_agreement(df)
    assert a["comparison_underpowered"] is False


def test_agreement_uses_only_posts_observed_to_die():
    """A censored post's t_death is a censoring time, not a duration."""
    df = _frame(n=20)
    df["event_observed"] = [True] * 5 + [False] * 15
    a = label_agreement(df)
    assert a["n_comparable"] == 5


def test_agreement_survives_a_frame_with_no_saturation_column():
    df = _frame().drop(columns=["t_saturation"])
    a = label_agreement(df)
    assert a["status"] == "no saturation labels in frame"


# ------------------------------------------------------------ outcome swapping
def test_failed_fit_is_censored_not_dropped():
    df = _frame(n=20)
    df.loc[:4, "t_saturation"] = np.nan       # 5 posts whose fit failed
    out = saturation_frame(df)
    assert len(out) == 20, "posts with a failed fit must be kept, as censored"
    assert (~out["event_observed"]).sum() == 5
    # Their duration must be the censoring time, not a fitted value.
    censored = out[~out["event_observed"]]
    assert censored["t_death"].notna().all()


def test_fitted_posts_become_events():
    out = saturation_frame(_frame(n=20))
    assert out["event_observed"].all()


def test_saturation_before_the_landmark_is_dropped_not_clipped():
    """Mirrors the primary label's 'died before the landmark' exclusion."""
    df = _frame(n=10)
    df.loc[:2, "t_saturation"] = -5.0         # saturated before the landmark
    out = saturation_frame(df)
    assert len(out) == 7
    assert (out["t_death"] > 0).all()


def test_swapped_frame_keeps_features_intact():
    df = _frame(n=15)
    df["log_value_at_3h"] = np.arange(15, dtype=float)
    out = saturation_frame(df)
    assert "log_value_at_3h" in out.columns
    assert out["log_value_at_3h"].notna().all()
