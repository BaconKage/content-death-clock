"""A cohort must not be analysed for the record while it is still maturing.

Cohort membership is frozen by publication date. Outcomes are not: a post inside
its ``collection.max_track_hours`` window is still being observed, and one
currently recorded as right-censored may yet be seen to die.

This was found late and by accident. A Cohort A run on 2026-09-20 reported 498
observed deaths while 416 of its 852 posts were still inside the window; a
second run hours later, on identical membership, reported 500. Every figure in
the results section was drifting, and nothing said so.

Two behaviours are pinned here:

* **Cohort A warns.** It is re-runnable, and a provisional look is legitimate,
  so the run proceeds but says loudly that it is provisional and stamps the
  artifact so a reader of the JSON cannot miss it either.
* **Cohort B refuses.** It is evaluated exactly once. Spending that single shot
  on outcomes that are going to change would also confound the temporal
  generalisation being tested with how long each cohort happened to be watched.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cdc.models import dataset


FREEZE = pd.Timestamp("2026-09-16T00:00:00Z")
CLOSE = pd.Timestamp("2026-10-04T00:00:00Z")


def _posts(pubs: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "post_id": [f"p{i}" for i in range(len(pubs))],
        "platform": ["youtube"] * len(pubs),
        "published_at": pd.to_datetime(pubs, utc=True),
    })


@pytest.fixture
def frame(monkeypatch, tmp_path):
    """Point cohort_maturity at a silver directory we control."""
    def _install(pubs):
        d = tmp_path / "silver"
        d.mkdir(exist_ok=True)
        _posts(pubs).to_parquet(d / "posts.parquet", index=False)
        monkeypatch.setattr(dataset, "path_for", lambda k: d)
        monkeypatch.setattr(dataset, "freeze_instant", lambda: FREEZE)
        monkeypatch.setattr(dataset, "_cohort_b_close", lambda: CLOSE)
    return _install


def test_cohort_still_inside_the_window_is_not_mature(frame):
    """One post published 3 days ago; the window is 14."""
    now = pd.Timestamp("2026-09-20T00:00:00Z")
    frame(["2026-09-10T00:00:00Z", "2026-09-17T00:00:00Z"])
    a = dataset.cohort_maturity(cohort="A", now=now)
    assert a["n"] == 1 and a["immature"] == 1 and a["mature"] is False


def test_cohort_past_the_window_is_mature(frame):
    """Everything published more than 336h ago."""
    now = pd.Timestamp("2026-10-15T00:00:00Z")
    frame(["2026-09-10T00:00:00Z", "2026-09-14T00:00:00Z"])
    a = dataset.cohort_maturity(cohort="A", now=now)
    assert a["n"] == 2 and a["immature"] == 0 and a["mature"] is True


def test_maturity_date_is_the_last_publication_plus_the_window(frame):
    now = pd.Timestamp("2026-09-20T00:00:00Z")
    frame(["2026-09-01T00:00:00Z", "2026-09-15T12:00:00Z"])
    a = dataset.cohort_maturity(cohort="A", now=now)
    assert a["matures_at"].startswith("2026-09-29T12:00")


def test_cohort_b_respects_both_boundaries(frame):
    """B is bounded below by the freeze and above by the close date."""
    now = pd.Timestamp("2026-10-20T00:00:00Z")
    frame([
        "2026-09-10T00:00:00Z",   # Cohort A — before the freeze
        "2026-09-17T00:00:00Z",   # Cohort B
        "2026-10-01T00:00:00Z",   # Cohort B
        "2026-10-09T00:00:00Z",   # after the close — in neither
    ])
    b = dataset.cohort_maturity(cohort="B", now=now)
    assert b["n"] == 2, "B must exclude pre-freeze posts and post-close posts"


def test_maturity_check_reads_no_outcomes(frame):
    """It must be safe to call against the sealed holdout.

    The check exists to gate the holdout, so it cannot itself require the
    labels it is gating. It reads publication dates and nothing else — the
    fixture's frame has no snapshot, velocity or event column at all, and the
    call still succeeds.
    """
    now = pd.Timestamp("2026-09-20T00:00:00Z")
    frame(["2026-09-17T00:00:00Z"])
    b = dataset.cohort_maturity(cohort="B", now=now)
    assert b["n"] == 1 and b["mature"] is False


def test_immature_cohort_b_is_refused(monkeypatch):
    """The holdout is evaluated once. Not on outcomes that are still moving."""
    from cdc.eval import report

    monkeypatch.setattr(report.holdout, "check_unlocked", lambda *a, **k: [])
    monkeypatch.setattr(report.dataset, "cohort_maturity",
                        lambda **k: {"cohort": "B", "n": 236, "immature": 236,
                                     "mature": False,
                                     "matures_at": "2026-10-18T00:00:00+00:00"})

    with pytest.raises(SystemExit) as e:
        report.run(platform="youtube", cohort="B", unlock_holdout=True,
                   write=False)
    assert "REFUSED" in str(e.value)
    assert "2026-10-18" in str(e.value), "must name the date it becomes valid"


def test_mature_cohort_b_is_not_blocked_by_the_gate(monkeypatch):
    """The gate must open once the cohort has matured.

    Stops short of a real evaluation: `build` is stubbed, so reaching it is the
    assertion. This test must never cause a genuine holdout run.
    """
    from cdc.eval import report

    monkeypatch.setattr(report.holdout, "check_unlocked", lambda *a, **k: [])
    monkeypatch.setattr(report.dataset, "cohort_maturity",
                        lambda **k: {"cohort": "B", "n": 236, "immature": 0,
                                     "mature": True, "matures_at": "x"})

    reached = {}

    def _stop(**kw):
        reached["yes"] = True
        raise RuntimeError("reached build")

    monkeypatch.setattr(report.dataset, "build", _stop)
    with pytest.raises(RuntimeError, match="reached build"):
        report.run(platform="youtube", cohort="B", unlock_holdout=True,
                   write=False)
    assert reached.get("yes")
