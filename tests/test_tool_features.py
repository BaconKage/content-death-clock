"""The instant model's inputs: training and the live app must compute the same thing.

The parity test pushes three recorded YouTube responses through both paths.
The training path runs the real silver transform (``silver.build_posts`` and
``build_snapshots``) on the recorded bronze records; the app path parses the
raw ``videos.list`` item the way the app does. Both must give identical inputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from cdc.tool import features as F

FIX = json.loads((Path(__file__).parent / "fixtures" / "tool_parity.json").read_text(encoding="utf-8"))


def _same(a: dict, b: dict) -> bool:
    return all((math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in F.FEATURES)


@pytest.fixture()
def silver(monkeypatch):
    from cdc.transform import silver as S

    def fake_iter(kind, platform=None, root=None):
        yield from (FIX["posts"] if kind == "posts" else FIX["snapshots"]) if platform == "youtube" else []

    monkeypatch.setattr(S, "iter_bronze", fake_iter)
    return S.build_posts(("youtube",)).set_index("post_id"), S.build_snapshots(("youtube",))


def test_training_and_app_paths_agree_on_recorded_videos(silver):
    posts, snaps = silver
    assert len(snaps) == 3
    for post in FIX["posts"]:
        s = snaps[snaps["post_id"] == post["post_id"]].iloc[0]
        train = F.from_silver(s["age_hours"], s["views"], s["likes"], s["comments"],
                              posts.loc[post["post_id"]].to_dict())
        now = pd.Timestamp(s["snapshot_ts"]).to_pydatetime()
        app, facts = F.from_api(post["raw"], FIX["subscribers"][post["creator_id"]], now)
        assert _same(train, app), (post["post_id"], train, app)
        assert facts["views"] == s["views"]


def test_the_fixture_covers_hidden_and_zero_likes(silver):
    _, snaps = silver
    likes = dict(zip(snaps["post_id"], snaps["likes"]))
    assert any(pd.isna(v) for v in likes.values())          # likes hidden
    assert 0 in likes.values()                               # likes visible but zero


def test_missing_stays_missing_and_zero_is_floored():
    f = F.instant_features(2.0, 100, None, 0, None, None, 13, 40)
    assert math.isnan(f["log_like_rate"])                    # hidden likes are not zero likes
    assert f["log_comment_rate"] == -6.0                     # zero comments: the floor, not -inf
    assert math.isnan(f["log_subscribers"]) and math.isnan(f["log_views_per_sub"])
    assert math.isnan(f["log_duration_sec"])
    z = F.instant_features(2.0, 0, 0, 0, 1000, 60, 13, 40)
    assert math.isnan(z["log_like_rate"])                    # no views: no rate at all


def test_inputs_are_rounded():
    f = F.instant_features(3.14159, 12345, 678, 9, 54321, 300, 7, 55)
    assert f["age_h"] == 3.14
    assert all(round(f[k], 2) == f[k] for k in F.FEATURES if not math.isnan(f[k]))


@pytest.mark.parametrize("bad", [0, -1, None, float("nan")])
def test_rejects_impossible_ages(bad):
    with pytest.raises(ValueError):
        F.instant_features(bad, 10, 1, 1, 100, 60, 1, 10)


@pytest.mark.parametrize("t, mark", [(0.4, 1), (1.2, 1), (2.4, 3), (3.3, 3), (8.3, 6), (11.9, 12),
                                     (12.6, 12), (23.6, 24), (24.3, 24), (49.0, 48), (168.4, 168),
                                     (240.3, 240)])
def test_fade_times_land_on_the_mark_they_were_read_for(t, mark):
    assert F.fade_mark(t) == mark


def test_age_window_matches_the_checkpoint_windows():
    assert F.MIN_AGE_H == 0.25 and F.MAX_AGE_H == 13.0
    assert F.mark_of(0.3) == 1 and F.mark_of(13.0) == 12 and F.mark_of(9.0) is None
