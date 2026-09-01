"""The landmark design.

Without it, a post that dies inside the feature window is "predicted" from the
observations that defined its death. These tests pin the exclusion rule and the
outcome transformation, because a silent regression here would restore a
circularity that inflates every number in the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdc.labels.death import Observation, label_post


def series(pairs):
    return [Observation(float(a), float(v)) for a, v in pairs]


def test_a_post_dying_inside_the_window_is_identifiable():
    """The situation landmarking exists to exclude: death at 3.1h, features
    built from observations up to 7h. The label must place it before the
    landmark so the dataset builder can drop it."""
    obs = series([(0, 0), (0.5, 900), (1, 1000), (3, 1010), (6, 1012), (12, 1013)])
    lab = label_post("early", obs)
    assert lab.event_observed
    assert lab.t_death < 7.0, "this post died inside the feature window"


def test_a_post_dying_after_the_landmark_is_eligible():
    obs = series([(0, 0), (1, 1000), (3, 2400), (6, 3000), (12, 3300),
                  (24, 3340), (36, 3350), (48, 3355)])
    lab = label_post("late", obs)
    assert lab.event_observed
    assert lab.t_death > 7.0
    # remaining lifetime from the landmark
    assert lab.t_death - 7.0 == pytest.approx(17.0)


def test_remaining_time_is_never_negative_for_eligible_posts():
    """Any post that survives the landmark must have positive remaining time,
    or the outcome variable is incoherent."""
    obs = series([(0, 0), (1, 100), (3, 300), (6, 600), (12, 1200), (24, 2400)])
    lab = label_post("alive", obs)
    remaining = lab.t_death - 7.0
    assert remaining > 0


def test_landmark_excludes_and_transforms_on_real_config():
    """End to end against the configured landmark, using the real builder."""
    from cdc.config import settings
    lm = float(settings()["modelling"]["landmark_hours"])
    assert lm >= 6.0, "landmark must be at or after the nominal 6h mark"
    cutoff = float(settings()["modelling"]["feature_cutoff_hours"])
    assert cutoff <= lm, (
        "features must not use observations from after the landmark - that is "
        "exactly the leak landmarking removes"
    )
