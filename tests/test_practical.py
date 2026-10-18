"""The practical early-warning extension (amendment 2026-10-18).

Pinned here: Cohort C cannot be evaluated early or twice by accident; no input
reads anything after the moment of prediction; and the early-warning metric
behaves as declared on cases with known answers.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cdc.eval import practical as P


def _snaps(rows):
    return pd.DataFrame(rows, columns=["post_id", "age_hours", "views", "likes", "comments", "primary_value"])


def test_cohort_boundaries():
    s = pd.Series(["2026-09-15T23:59:59Z", "2026-09-16T00:00:00Z", "2026-10-03T23:59:59Z",
                   "2026-10-04T00:00:00Z", "2026-10-14T23:59:59Z", "2026-10-15T00:00:00Z"])
    assert P.cohort_of(s).tolist() == ["A", "B", "B", "C", "C", "later"]


def test_refuses_before_the_declared_date(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "ledger_path", lambda: tmp_path / "ledger.jsonl")
    with pytest.raises(SystemExit, match="on or after"):
        P.check_allowed(now=P.NOT_BEFORE - pd.Timedelta(minutes=1))
    P.check_allowed(now=P.NOT_BEFORE)                       # allowed from the instant


def test_refuses_a_second_look_unless_asked(tmp_path, monkeypatch):
    led = tmp_path / "ledger.jsonl"
    led.write_text(json.dumps({"evaluated_at": "2026-10-25T03:00:00+00:00"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(P, "ledger_path", lambda: led)
    with pytest.raises(SystemExit, match="already evaluated"):
        P.check_allowed(now=P.NOT_BEFORE + pd.Timedelta(days=1))
    P.check_allowed(now=P.NOT_BEFORE + pd.Timedelta(days=1), again=True)


def test_engagement_reads_nothing_after_the_landmark():
    early = [("p", 1.0, 100, 10, 1, 100), ("p", 6.5, 1000, 50, 5, 1000)]
    late = early + [("p", 7.2, 2000, 1000, 900, 2000)]           # 12 minutes after the landmark
    a = P.engagement_at(_snaps(early), 7.0).loc["p"]
    b = P.engagement_at(_snaps(late), 7.0).loc["p"]
    assert a.equals(b)
    assert a["log_like_rate"] == pytest.approx(np.log10(50 / 1000))


def test_trajectory_speed_share_is_latest_over_best():
    rows = [("p", 0.0, 0, 0, 0, 0), ("p", 1.0, 100, 0, 0, 100),      # 100/h, the best
            ("p", 3.0, 200, 0, 0, 200), ("p", 24.0, 410, 0, 0, 410),  # 10/h latest
            ("p", 26.0, 10_000, 0, 0, 10_000)]                         # after the landmark: ignored
    t = P.trajectory_at(_snaps(rows), 25.0).loc["p"]
    assert t["speed_share_at_landmark"] == pytest.approx(10 / 100)
    assert t["log_views_at_landmark"] == pytest.approx(np.log10(411))


def test_flag_metrics_on_a_perfect_and_a_useless_predictor():
    rng = np.random.default_rng(0)
    n = 500
    fast = rng.random(n) < 0.15
    creators = np.array([f"c{i % 40}" for i in range(n)])
    perfect = np.where(fast, 1.0, 100.0) + rng.random(n)            # fast faders predicted shortest
    m = P.flag_metrics(perfect, fast, creators, n_boot=200)
    # With fewer fast faders than flags, a perfect model catches all of them and
    # its precision is capped at (fast faders / flags), not 1.
    assert m["recall"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(fast.sum() / m["n_flagged"])
    assert m["passes"]
    useless = rng.random(n)
    u = P.flag_metrics(useless, fast, creators, n_boot=200)
    assert not u["passes"]
    assert u["precision_minus_base_ci"][0] < 0 < u["precision_minus_base_ci"][1]


def test_inputs_never_include_outcomes():
    for L in P.LANDMARKS:
        cols = P.inputs_for(L)
        assert not any(c.startswith(("t_death", "t_saturation", "event", "fitted_")) for c in cols)
        assert ("speed_share_at_landmark" in cols) == (L > 7.0)
