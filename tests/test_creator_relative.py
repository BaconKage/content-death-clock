"""The creator-relative feature must not import the future or the answer.

This is the riskiest feature in the study, because it is the only one computed
from *other posts*. Two distinct ways it could cheat:

* using posts published LATER than the target (importing the future), and
* using prior posts' death times rather than their early velocity
  (importing the answer).

Both would inflate the score and neither would be visible in the output. The
tests below are written so that a feature which cheats fails them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdc.models.dataset import MIN_PRIOR_POSTS, _add_category_dummies, _add_creator_relative


def _frame(vels, creator="c1", start="2026-09-01"):
    t = pd.date_range(start, periods=len(vels), freq="D", tz="UTC")
    return pd.DataFrame({
        "post_id": [f"p{i}" for i in range(len(vels))],
        "creator_id": creator,
        "published_at": t,
        "velocity_to_6h": vels,
        "t_death": np.arange(len(vels), dtype=float) + 10.0,
        "event_observed": True,
    })


def test_uses_only_earlier_posts():
    """Changing a LATER post must not change an earlier post's feature."""
    base = _add_creator_relative(_frame([10.0, 20.0, 30.0, 40.0, 50.0]))
    bumped = _add_creator_relative(_frame([10.0, 20.0, 30.0, 40.0, 9999.0]))
    a = base.set_index("post_id")["log_rel_velocity_6h"]
    b = bumped.set_index("post_id")["log_rel_velocity_6h"]
    for pid in ["p0", "p1", "p2", "p3"]:
        assert (pd.isna(a[pid]) and pd.isna(b[pid])) or a[pid] == pytest.approx(b[pid]), (
            f"{pid} changed when a LATER post changed — the feature sees the future")


def test_first_posts_have_no_baseline():
    out = _add_creator_relative(_frame([10.0, 20.0, 30.0, 40.0]))
    out = out.sort_values("published_at").reset_index(drop=True)
    # fewer than MIN_PRIOR_POSTS priors -> no baseline
    assert out.loc[0, "n_prior_posts"] == 0
    assert pd.isna(out.loc[0, "log_rel_velocity_6h"])
    assert pd.isna(out.loc[1, "log_rel_velocity_6h"])
    assert not pd.isna(out.loc[MIN_PRIOR_POSTS, "log_rel_velocity_6h"])


def test_the_value_is_the_ratio_to_the_prior_median():
    out = _add_creator_relative(_frame([10.0, 20.0, 60.0])).sort_values("published_at")
    # third post: priors are 10 and 20, median 15; 60/15 = 4
    assert out.iloc[2]["log_rel_velocity_6h"] == pytest.approx(np.log10(4.0))


def test_outperforming_and_underperforming_have_opposite_signs():
    hot = _add_creator_relative(_frame([10.0, 10.0, 100.0])).sort_values("published_at")
    cold = _add_creator_relative(_frame([10.0, 10.0, 1.0])).sort_values("published_at")
    assert hot.iloc[2]["log_rel_velocity_6h"] > 0
    assert cold.iloc[2]["log_rel_velocity_6h"] < 0


def test_creators_are_independent():
    """One creator's history must never inform another's."""
    a = _frame([10.0, 10.0, 10.0], creator="c1")
    b = _frame([500.0, 500.0, 500.0], creator="c2")
    out = _add_creator_relative(pd.concat([a, b], ignore_index=True))
    c1 = out[out.creator_id == "c1"].sort_values("published_at")
    alone = _add_creator_relative(a).sort_values("published_at")
    assert c1.iloc[2]["log_rel_velocity_6h"] == pytest.approx(
        alone.iloc[2]["log_rel_velocity_6h"])


def test_death_times_are_not_used():
    """Scrambling the outcome must leave the feature untouched."""
    f = _frame([10.0, 20.0, 30.0, 40.0])
    g = f.copy()
    g["t_death"] = g["t_death"].to_numpy()[::-1]
    g["event_observed"] = [True, False, True, False]
    a = _add_creator_relative(f)["log_rel_velocity_6h"].to_numpy()
    b = _add_creator_relative(g)["log_rel_velocity_6h"].to_numpy()
    assert np.allclose(a, b, equal_nan=True), (
        "the feature moved when only the OUTCOME changed — it is leaking the answer")


def test_zero_or_missing_velocity_gives_nan_not_a_guess():
    out = _add_creator_relative(_frame([0.0, 0.0, 50.0])).sort_values("published_at")
    assert pd.isna(out.iloc[2]["log_rel_velocity_6h"])


def test_empty_frame_is_safe():
    assert _add_creator_relative(pd.DataFrame()).empty


# ------------------------------------------------------------------- category
def test_category_is_one_hot_not_an_integer_code():
    df = pd.DataFrame({"stratum_category": ["science", "gaming", "science", None]})
    out = _add_category_dummies(df)
    assert out["cat_science"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert out["cat_gaming"].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_category_dummies_absent_when_column_missing():
    out = _add_category_dummies(pd.DataFrame({"x": [1]}))
    assert not [c for c in out.columns if c.startswith("cat_")]
