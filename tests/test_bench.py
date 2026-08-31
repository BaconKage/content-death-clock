"""Amplification and the benchmark workload.

Two things must hold or the benchmark measures the wrong thing:

* amplified data must keep the structure the job depends on — per-post ordered
  sequences, growing group cardinality, realistic value spread. Amplification
  that flattens any of those turns a hard workload into an easy one and the
  crossover point becomes fiction.
* the feature job must compute velocity **within** a post. Differencing across a
  post boundary is a silent bug that produces plausible-looking nonsense.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdc.bench.amplify import amplify
from cdc.bench.spark_bench import feature_job_pandas


def base_frame():
    """Two posts from two creators, each with an ordered observation sequence."""
    return pd.DataFrame({
        "post_id": ["a", "a", "a", "b", "b", "b"],
        "creator_id": ["c1", "c1", "c1", "c2", "c2", "c2"],
        "age_hours": [1.0, 3.0, 6.0, 1.0, 3.0, 6.0],
        "primary_value": [100.0, 300.0, 600.0, 50.0, 80.0, 90.0],
        "views": [100.0, 300.0, 600.0, 50.0, 80.0, 90.0],
    })


# ------------------------------------------------------------- amplification
def test_factor_one_returns_the_real_data_untouched():
    """The 1x row of the benchmark must be the genuine dataset, not a
    reconstruction of it."""
    df = base_frame()
    pd.testing.assert_frame_equal(amplify(df, 1), df)


def test_row_count_scales_exactly():
    df = base_frame()
    for f in (2, 5, 20):
        assert len(amplify(df, f)) == len(df) * f


def test_replicated_posts_get_distinct_ids():
    """Colliding ids would merge replicas into giant posts, changing the
    workload from many small groups to a few huge ones."""
    out = amplify(base_frame(), 10)
    assert out["post_id"].nunique() == 2 * 10


def test_group_cardinality_grows_with_the_data():
    """Creators must be replicated too. Holding them fixed at 63 would make the
    shuffle unrealistically cheap as rows grow."""
    out = amplify(base_frame(), 10)
    assert out["creator_id"].nunique() == 2 * 10


def test_within_post_time_ordering_survives():
    out = amplify(base_frame(), 5)
    for _, g in out.groupby("post_id"):
        assert len(g) == 3
        assert g["age_hours"].is_monotonic_increasing or \
            g.sort_values("age_hours")["age_hours"].is_monotonic_increasing


def test_values_are_jittered_not_copied():
    """Thousands of identical rows are compressible and cacheable in ways real
    data is not, which would flatter whichever engine exploits that.

    Checked by counting distinct values: 6 in the base, so 20 replicas of exact
    copies would still be 6. Jitter must multiply that.
    """
    df = base_frame()
    out = amplify(df, 20)
    base_distinct = df["primary_value"].nunique()
    assert out["primary_value"].nunique() > base_distinct * 10, (
        f"only {out['primary_value'].nunique()} distinct values from "
        f"{base_distinct} base values — jitter is not being applied"
    )
    # ages must be jittered too, or every replica sorts identically
    assert out["age_hours"].nunique() > df["age_hours"].nunique() * 10


def test_missing_values_stay_missing():
    """Instagram image posts have no view count. Jitter must not turn a NaN into
    a number, which would silently invent an observation."""
    df = base_frame()
    df.loc[1, "primary_value"] = np.nan
    out = amplify(df, 5)
    assert out["primary_value"].isna().sum() == 5


def test_amplified_values_never_go_negative():
    df = base_frame()
    df["primary_value"] = 1.0        # small values are where jitter could flip sign
    assert (amplify(df, 50)["primary_value"] >= 0).all()


def test_rejects_a_nonsense_factor():
    with pytest.raises(ValueError):
        amplify(base_frame(), 0)


# ---------------------------------------------------------------- the workload
def test_velocity_is_computed_within_a_post_not_across_the_boundary():
    """The silent bug this test exists for.

    Post 'a' ends at 600 and post 'b' begins at 50. If the job differences across
    the boundary it produces a large negative velocity for b's first row, and
    every downstream number is quietly wrong while looking plausible.
    """
    out = feature_job_pandas(base_frame()).set_index("post_id")
    # a: +100/h then +100/h  -> peak 100
    assert out.loc["a", "peak_velocity"] == pytest.approx(100.0)
    # b: +15/h then +3.33/h  -> peak 15, and never negative
    assert out.loc["b", "peak_velocity"] == pytest.approx(15.0)
    assert out.loc["b", "mean_velocity"] > 0


def test_job_reports_one_row_per_post():
    out = feature_job_pandas(amplify(base_frame(), 7))
    assert len(out) == 2 * 7
    assert out["post_id"].is_unique


def test_job_final_value_is_the_last_observation_by_age():
    """Rows arrive unordered from bronze; 'last' must mean latest by age, not
    last by row position."""
    df = base_frame().iloc[::-1].reset_index(drop=True)   # reversed
    out = feature_job_pandas(df).set_index("post_id")
    assert out.loc["a", "final_value"] == pytest.approx(600.0)
    assert out.loc["b", "final_value"] == pytest.approx(90.0)


def test_single_observation_post_yields_no_velocity():
    """One observation cannot produce a velocity. NaN, not zero — zero would
    read as a real measurement of no movement."""
    df = pd.DataFrame({
        "post_id": ["solo"], "creator_id": ["c9"],
        "age_hours": [1.0], "primary_value": [10.0], "views": [10.0],
    })
    out = feature_job_pandas(df)
    assert out.loc[0, "n_obs"] == 1
    assert np.isnan(out.loc[0, "peak_velocity"])
