"""The report: what it says at each age, and when it must stay silent."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cdc.tool import model as M
from cdc.tool import report as R
from tests.tool_synth import as_table, synthetic_rows

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def env():
    rows = synthetic_rows()
    model = M.InstantModel(b=0.6, n_estimators=60).fit(rows)
    card = {"b": 0.6, "thresholds_p48": {"high": 0.45, "low": 0.23}, "cohort_c": None,
            "rehearsal_A_to_B": {"primary": {"n_videos": 900, "n_channels": 51, "flag_precision": 0.53,
                                             "base_rate": 0.31, "brier_skill": 0.08}}}
    tips = {"tips": [], "candidates": {}, "facts": {"share_fastest_in_first_6h": 0.72,
                                                    "median_share_of_10_day_views_by_24h": 0.52}}
    return model, as_table(rows), card, tips


def item(age_h, views=2000, likes="80", comments="6", live="none"):
    pub = (NOW - timedelta(hours=age_h)).strftime("%Y-%m-%dT%H:%M:%SZ")
    st = {"viewCount": str(views), "commentCount": comments}
    if likes is not None:
        st["likeCount"] = likes
    if comments is None:
        st.pop("commentCount")
    return {"id": "abcdefghijk", "statistics": st, "contentDetails": {"duration": "PT8M"},
            "snippet": {"publishedAt": pub, "title": "A video", "channelId": "UCx", "channelTitle": "Chan",
                        "liveBroadcastContent": live}}


def run(env, it, subs=50_000, card=None):
    model, table, c, tips = env
    return R.build_report(it, subs, NOW, model, table, card or c, tips)


def test_fresh_video_gets_an_outlook_that_makes_sense(env):
    rep = run(env, item(5))
    assert rep["status"] == "ok"
    o = rep["outlook"]
    assert 0 <= o["fade_within_2_days"] <= 1 and o["risk"] in ("high", "typical", "low")
    chances = [x["chance"] for x in o["still_going"]]
    assert chances == sorted(chances, reverse=True)
    assert o["fade_within_2_days"] == pytest.approx(1 - chances[1])
    assert len(rep["comparisons"]) == 3 and rep["todo"]


def test_risk_label_follows_the_frozen_thresholds(env):
    model, table, card, tips = env
    rep = run(env, item(5))
    p = rep["outlook"]["fade_within_2_days"]
    for high, low, want in ((p, 0.0, "high"), (1.0, p, "low"), (1.0, 0.0, "typical")):
        c = dict(card, thresholds_p48={"high": high, "low": low})
        assert run(env, item(5), card=c)["outlook"]["risk"] == want


def test_too_new_says_when_to_come_back(env):
    rep = run(env, item(0.1))
    assert rep["status"] == "too_new" and rep["outlook"] is None
    assert 0 < rep["minutes_until_ready"] <= 9


def test_older_videos_get_comparisons_but_no_odds(env):
    rep = run(env, item(30))
    assert rep["status"] == "too_old_for_odds" and rep["outlook"] is None
    assert rep["comparisons"] and rep["comparisons"][0]["at_h"] == (24.0, 36.0)


def test_beyond_the_study_window_says_so(env):
    rep = run(env, item(24 * 30))
    assert rep["status"] == "beyond_window" and not rep["comparisons"] and rep["notes"]


def test_live_and_upcoming_are_not_scored(env):
    for live in ("live", "upcoming"):
        rep = run(env, item(2, live=live))
        assert rep["status"] == "live" and rep["outlook"] is None


def test_hidden_counts_are_flagged_and_still_scored(env):
    rep = run(env, item(5, likes=None, comments=None), subs=None)
    assert rep["status"] == "ok" and rep["outlook"] is not None
    text = " ".join(rep["notes"])
    assert "subscriber" in text and "Likes are hidden" in text and "Comments are turned off" in text
    assert rep["comparisons"][0]["metric"] == "views"          # no subscriber count to divide by
    assert all(c["group"] == "all channel sizes" for c in rep["comparisons"])


def test_a_failed_test_switches_the_odds_off(env):
    model, table, card, tips = env
    failed = dict(card, cohort_c={"passes": False, "unseen_channels": card["rehearsal_A_to_B"]["primary"]})
    rep = run(env, item(5), card=failed)
    assert rep["status"] == "ok" and rep["outlook"] is None
    assert any("did not pass" in n for n in rep["notes"])
    assert rep["accuracy"]["source"] == "test" and rep["accuracy"]["passed"] is False


def test_percentile_interpolates_between_marks(env):
    _, table, _, _ = env
    at6 = R.percentile(table, "log_like_rate", -1.4, 6.0, 4.7)["percentile"]
    at12 = R.percentile(table, "log_like_rate", -1.4, 12.0, 4.7)["percentile"]
    mid = R.percentile(table, "log_like_rate", -1.4, (6.0 * 12.0) ** 0.5, 4.7)["percentile"]
    assert mid == pytest.approx((at6 + at12) / 2)
