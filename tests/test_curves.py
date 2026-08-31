"""Curve construction for the demo and the paper figures.

The same code draws both, so a bug here would make the demo an illustration of
something we did not actually do.

The sharpest test is the honesty one: the demo must refuse to display a
countdown for a post whose death has not been observed. A project arguing that
decay should be measured rather than guessed cannot open its demo with a guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdc.app.curves import build_curve, countdown_text, summarise


def frame(rows, platform="youtube", post_id="p1"):
    return pd.DataFrame({
        "post_id": [post_id] * len(rows),
        "platform": [platform] * len(rows),
        "creator_id": ["c1"] * len(rows),
        "creator_handle": ["@someone"] * len(rows),
        "age_hours": [float(a) for a, _ in rows],
        "primary_value": [float(v) for _, v in rows],
    })


DEAD = [(0, 0), (1, 1000), (3, 2400), (6, 3000), (12, 3300),
        (24, 3340), (36, 3350), (48, 3355)]
ALIVE = [(0, 0), (1, 100), (3, 300), (6, 600), (12, 1200), (24, 2400)]


# ------------------------------------------------------------------- honesty
def test_no_countdown_is_invented_for_a_post_that_has_not_died():
    """The demo's central discipline. A confident number with nothing behind it
    is exactly what this project argues against."""
    c = build_curve(frame(ALIVE), "p1")
    text = countdown_text(c)
    assert not c.event_observed
    assert "still alive" in text.lower()
    assert "no prediction available" in text.lower()


def test_a_post_with_too_few_observations_says_so():
    c = build_curve(frame([(0, 0), (1, 100), (3, 150)]), "p1")
    assert c.exclude_reason is not None
    assert countdown_text(c) == "Not enough observations yet"


def test_an_observed_death_is_reported_with_its_time():
    c = build_curve(frame(DEAD), "p1")
    assert c.event_observed
    assert c.t_death == pytest.approx(24.0)
    assert "24" in countdown_text(c)


# -------------------------------------------------------------------- curve
def test_threshold_is_five_percent_of_this_posts_own_peak():
    c = build_curve(frame(DEAD), "p1")
    assert c.peak_velocity == pytest.approx(1000.0)
    assert c.threshold == pytest.approx(50.0)


def test_velocities_match_the_label_definition():
    """The plotted velocity line and the reported death time must come from the
    same arithmetic, or the picture contradicts the number beside it."""
    c = build_curve(frame(DEAD), "p1")
    assert c.velocity_ages[0] == pytest.approx(1.0)
    assert c.velocities[0] == pytest.approx(1000.0)      # 1000 views in 1h
    assert c.velocities[1] == pytest.approx(700.0)       # 1400 views over 2h


def test_instagram_posts_are_measured_on_likes_not_views():
    """Image posts have no view count at all, so the metric label shown to the
    user must reflect what was actually measured."""
    c = build_curve(frame(DEAD, platform="instagram"), "p1")
    assert c.metric == "likes"
    assert build_curve(frame(DEAD, platform="youtube"), "p1").metric == "views"


def test_missing_values_are_dropped_not_plotted_as_zero():
    df = frame(DEAD)
    df.loc[2, "primary_value"] = np.nan
    c = build_curve(df, "p1")
    assert len(c.ages) == len(DEAD) - 1
    assert not np.isnan(c.values).any()


def test_late_first_observation_is_surfaced_to_the_user():
    """A post first seen at 3h has no 1-hour feature. The demo should say so
    rather than quietly showing a curve that starts late."""
    c = build_curve(frame([(3, 500), (6, 800), (12, 900), (24, 950), (36, 960)]), "p1")
    assert any("1-hour feature is unavailable" in n for n in c.notes)


def test_retracted_view_counts_are_flagged():
    c = build_curve(frame([(0, 0), (1, 1000), (3, 900), (6, 1100),
                           (12, 1150), (24, 1160)]), "p1")
    assert any("retracted-count" in n for n in c.notes)


def test_unknown_post_raises():
    with pytest.raises(KeyError):
        build_curve(frame(DEAD), "does-not-exist")


# ------------------------------------------------------------------ summary
def test_summary_reports_one_row_per_post_with_label_state():
    df = pd.concat([frame(DEAD, post_id="dead"), frame(ALIVE, post_id="alive")],
                   ignore_index=True)
    s = summarise(df).set_index("post_id")
    assert len(s) == 2
    assert bool(s.loc["dead", "died"]) is True
    assert bool(s.loc["alive", "died"]) is False
    assert bool(s.loc["alive", "labelable"]) is True
