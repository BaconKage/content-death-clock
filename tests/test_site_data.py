"""The public website must never identify a creator or a post.

``site/data.json`` is published on GitHub Pages. It shows real videos, so the
guard against naming one has to be in the builder, not in good intentions:
subscriber counts are shown only as wide bands (an exact count is enough to
name some of these channels), view counts are rounded, and no identifier key
survives into the output.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cdc.site.build_data import pick_pairs, round_views, strip_ids, subscriber_band


@pytest.mark.parametrize("n, band", [
    (1240, "under 10k subscribers"), (10800, "10k–100k subscribers"),
    (373000, "100k–500k subscribers"), (2.1e6, "500k–5M subscribers"),
    (515e6, "over 5M subscribers"), (None, "unknown size"), (float("nan"), "unknown size"),
])
def test_subscriber_counts_become_wide_bands(n, band):
    assert subscriber_band(n) == band


def test_views_are_rounded_to_two_significant_figures():
    assert round_views(1530517.36) == 1500000
    assert round_views(8659.8) == 8700
    assert round_views(42.4) == 42
    assert round_views(float("nan")) is None


def test_identifier_keys_are_stripped_at_any_depth():
    raw = {"post_id": "x", "game": [{"a": {"creator_id": "UCabc", "views_1h": 3}}], "title": "t"}
    out = strip_ids(raw)
    s = json.dumps(out)
    assert "post_id" not in s and "creator_id" not in s and "UCabc" not in s and "title" not in s
    assert out["game"][0]["a"]["views_1h"] == 3


def _frame(n=40):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "post_id": [f"vid{i}" for i in range(n)], "creator_id": [f"UC{i % 7}" for i in range(n)],
        "event_observed": rng.random(n) < 0.8, "t_death_from_publish": rng.choice([12, 24, 48, 96, 168], n).astype(float),
        "stratum_tier": rng.choice(["micro", "mid", "large"], n), "stratum_category": "news",
        "follower_count": rng.choice([5e3, 2e5, 9e6], n),
        "value_at_1h": rng.random(n) * 1e4, "value_at_3h": rng.random(n) * 3e4, "value_at_6h": rng.random(n) * 9e4,
    })


def test_game_pairs_have_a_true_answer_and_no_identifiers():
    f = _frame()
    preds = {"model": pd.Series(np.linspace(10, 200, len(f)), index=f.post_id),
             "size": pd.Series(np.linspace(200, 10, len(f)), index=f.post_id)}
    pairs = pick_pairs(f, preds, n=10, min_gap_h=24)
    assert pairs
    for p in pairs:
        a, b = p["a"]["died_h"], p["b"]["died_h"]
        assert abs(a - b) >= 24                                   # a real, decidable answer
        assert p["first"] == ("a" if a < b else "b")
        assert p["model_pick"] in ("a", "b") and p["size_pick"] in ("a", "b")
    s = json.dumps(pairs)
    assert "vid" not in s and "UC" not in s                       # no ids, no channel ids
    assert all("subscribers" in p["a"]["band"] for p in pairs)


def test_pairs_are_drawn_only_from_observed_deaths():
    f = _frame()
    f.loc[f.index[:20], "event_observed"] = False
    alive = set(f.post_id[~f.event_observed])
    preds = {"model": pd.Series(1.0, index=f.post_id), "size": pd.Series(1.0, index=f.post_id)}
    died = {round(float(x), 1) for x in f.loc[f.event_observed, "t_death_from_publish"]}
    for p in pick_pairs(f, preds, n=10):
        assert p["a"]["died_h"] in died and p["b"]["died_h"] in died
    assert alive  # the fixture really did contain survivors to exclude
