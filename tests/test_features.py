"""Feature construction, and the leakage boundary above all.

A feature that peeks past the 6h cutoff produces a model that scores brilliantly
and predicts nothing, with no visible symptom — the numbers just look good. It is
the single most dangerous bug available to this project, so the boundary is
tested from several directions rather than asserted once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdc.transform.features import (
    STANDARD_MARKS,
    early_window,
    features_for_post,
    interpolate_at,
)

CUTOFF = 6.0
META = {
    "post_id": "p1", "platform": "youtube", "creator_id": "c1",
    "published_at": pd.Timestamp("2026-09-01T14:00:00Z"),
    "duration_sec": 600.0, "title_len": 40, "tag_count": 5,
    "follower_count": 100_000, "media_type": "video",
    "stratum_tier": "mid", "stratum_category": "technology",
}


def snaps(pairs):
    return pd.DataFrame({
        "age_hours": [float(a) for a, _ in pairs],
        "primary_value": [float(v) for _, v in pairs],
    })


# ------------------------------------------------------------- THE LEAKAGE LINE
def test_observations_past_the_cutoff_are_excluded():
    """The whole study rests on this line."""
    g = snaps([(1, 100), (3, 300), (6, 600), (24, 5000), (72, 9000)])
    early = early_window(g, CUTOFF)
    assert early["age_hours"].max() <= CUTOFF
    assert 24 not in early["age_hours"].values


def test_late_observations_cannot_change_any_feature():
    """Strongest form of the test: build features for two posts identical up to
    6h but wildly different afterwards. Every feature must match exactly. If a
    late observation can move a feature, this fails."""
    early_part = [(0.5, 50), (1, 100), (3, 300), (6, 600)]
    a = features_for_post(snaps(early_part), META, CUTOFF)
    b = features_for_post(
        snaps(early_part + [(12, 50_000), (24, 900_000), (72, 5_000_000)]),
        META, CUTOFF)
    for k in a:
        if isinstance(a[k], float) and np.isnan(a[k]):
            assert np.isnan(b[k]), f"{k} differs"
        else:
            assert a[k] == b[k], f"feature {k} leaked: {a[k]} != {b[k]}"


def test_a_post_younger_than_a_mark_gets_nan_not_a_guess():
    """A two-hour-old post has no 6h feature. Extrapolating one invents data,
    and inventing data at the top of the funnel poisons everything below."""
    f = features_for_post(snaps([(0.5, 50), (1, 100), (2, 180)]), META, CUTOFF)
    assert np.isfinite(f["value_at_1h"])
    assert np.isnan(f["value_at_3h"])
    assert np.isnan(f["value_at_6h"])


# --------------------------------------------------------------- interpolation
def test_interpolation_hits_a_known_value():
    """Linear growth of 100/h: the value at t=3 must be exactly 300."""
    ages = np.array([0.0, 6.0])
    vals = np.array([0.0, 600.0])
    assert interpolate_at(ages, vals, 3.0) == pytest.approx(300.0)


def test_interpolation_handles_the_drifting_scheduler():
    """Snapshots land at 0.9h and 3.2h, not 1h and 3h. The 1h value must come
    from the bracketing pair, not from the nearest snapshot."""
    g = snaps([(0.9, 90), (3.2, 320), (5.8, 580)])
    f = features_for_post(g, META, CUTOFF)
    assert f["value_at_1h"] == pytest.approx(100.0, rel=0.02)
    assert f["value_at_3h"] == pytest.approx(300.0, rel=0.02)


def test_never_extrapolates_beyond_the_last_observation():
    ages, vals = np.array([1.0, 3.0]), np.array([100.0, 300.0])
    assert np.isnan(interpolate_at(ages, vals, 6.0))
    assert np.isnan(interpolate_at(ages, vals, 0.5))


# -------------------------------------------------------------------- features
def test_velocity_is_per_hour_since_publish():
    f = features_for_post(snaps([(1, 100), (3, 300), (6, 600)]), META, CUTOFF)
    assert f["velocity_to_1h"] == pytest.approx(100.0)
    assert f["velocity_to_6h"] == pytest.approx(100.0)


def test_interval_velocities_capture_a_decaying_curve():
    """Fast then slow: the 3-6h interval velocity must be below the 1-3h one,
    and the decay ratio below 1. This is the shape the model keys on."""
    f = features_for_post(snaps([(1, 1000), (3, 1800), (6, 2000)]), META, CUTOFF)
    assert f["velocity_1_3h"] == pytest.approx(400.0)
    assert f["velocity_3_6h"] == pytest.approx(66.67, rel=0.01)
    assert f["decay_ratio_3_6_over_1_3"] < 1.0
    assert f["acceleration"] < 0


def test_log_growth_is_scale_free():
    """A 300-view post and a 3M-view post with the same SHAPE must produce the
    same growth features. Without this, the model just learns channel size."""
    shape = [(1, 100), (3, 300), (6, 600)]
    small = features_for_post(snaps(shape), META, CUTOFF)
    big = features_for_post(snaps([(t, v * 10_000) for t, v in shape]), META, CUTOFF)
    assert small["log_growth_1_3h"] == pytest.approx(big["log_growth_1_3h"], abs=1e-3)
    assert small["log_growth_3_6h"] == pytest.approx(big["log_growth_3_6h"], abs=1e-3)


def test_missing_metric_values_are_skipped_not_zeroed():
    """Instagram image posts have no view count. A None must not become a 0
    observation, which would look like a real collapse to zero engagement."""
    g = pd.DataFrame({"age_hours": [1.0, 3.0, 6.0],
                      "primary_value": [100.0, np.nan, 600.0]})
    f = features_for_post(g, META, CUTOFF)
    assert f["n_early_observations"] == 2
    assert f["value_at_1h"] == pytest.approx(100.0)


def test_publish_time_features_are_derived_correctly():
    meta = dict(META, published_at=pd.Timestamp("2026-09-05T14:00:00Z"))  # Saturday
    f = features_for_post(snaps([(1, 10), (3, 30)]), meta, CUTOFF)
    assert f["publish_hour_utc"] == 14
    assert f["publish_is_weekend"] == 1


def test_no_observations_yields_nans_without_crashing():
    g = pd.DataFrame({"age_hours": [], "primary_value": []})
    f = features_for_post(g, META, CUTOFF)
    assert f["n_early_observations"] == 0
    assert np.isnan(f["value_at_1h"])


def test_log_growth_of_a_zero_baseline_is_nan_not_a_fabricated_number():
    """A post with no engagement at 1h has no defined growth ratio. Smoothing
    with +1 would manufacture one, and would manufacture a different one
    depending on the post's scale."""
    f = features_for_post(snaps([(1, 0), (3, 300), (6, 600)]), META, CUTOFF)
    assert np.isnan(f["log_growth_1_3h"])
    assert np.isfinite(f["log_growth_3_6h"])


def test_log_growth_is_exactly_scale_free_across_six_orders_of_magnitude():
    """The micro tier makes this matter: 8 -> 24 views must give the same growth
    feature as 8M -> 24M."""
    shape = [(1, 8), (3, 24), (6, 48)]
    ref = features_for_post(snaps(shape), META, CUTOFF)
    for factor in (10, 1_000, 1_000_000):
        scaled = features_for_post(
            snaps([(t, v * factor) for t, v in shape]), META, CUTOFF)
        assert scaled["log_growth_1_3h"] == pytest.approx(ref["log_growth_1_3h"], abs=1e-12)
        assert scaled["log_growth_3_6h"] == pytest.approx(ref["log_growth_3_6h"], abs=1e-12)
