"""Regression tests: the at-discovery snapshot must not swallow the t+1h mark.

The bug this pins was silent and expensive. Discovery writes a snapshot the
instant a post is found — usually within minutes of publication, well before the
t+1h window opens. `due` compared a *count* of snapshots against a *count* of
marks passed, so that early snapshot made the post look as though it had already
satisfied the first mark. Every post was permanently one behind, and the mark it
lost was always t+1h: the earliest, most valuable, least recoverable point on a
decay curve, and the basis of the study's early-velocity features.

Measured on real data before the fix: posts published after the scheduler was
repaired reached the t+1h window only 45% of the time, while t+3h ran at 97%
and t+12h at 92%. The shape of that gap is what gave the bug away.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cdc.collect.panel import Panel, TrackedPost, _first_mark_opens

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def _post(age_hours: float, snaps: int = 0, scheduled: int | None = None):
    return TrackedPost(
        post_id="p1", platform="youtube", creator_id="c1",
        published_at=NOW - timedelta(hours=age_hours),
        snapshot_count=snaps,
        scheduled_snapshot_count=snaps if scheduled is None else scheduled,
    )


def _panel(p: TrackedPost) -> Panel:
    return Panel(posts={p.post_id: p})


def test_discovery_snapshot_does_not_satisfy_the_first_mark():
    """The bug, stated directly."""
    # Found 3 minutes after publication, now one hour old: t+1h is owed.
    p = _post(age_hours=1.0, snaps=1, scheduled=0)
    assert _panel(p).due(NOW), "post at t+1h with only a discovery snapshot must be due"


def test_a_snapshot_inside_the_first_window_does_satisfy_it():
    """Discovery that happens late enough is a legitimate t+1h observation."""
    p = _post(age_hours=1.0, snaps=1, scheduled=1)
    assert not _panel(p).due(NOW)


def test_first_mark_window_boundary():
    """A snapshot counts toward the schedule from when the window opens."""
    opens = _first_mark_opens()
    assert opens == pytest.approx(0.25)          # 1h mark, 0.75h tolerance


def test_post_still_becomes_due_at_each_later_mark():
    # Two scheduled snapshots (t+1, t+3) and now past t+6.
    p = _post(age_hours=6.5, snaps=3, scheduled=2)
    assert _panel(p).due(NOW)


def test_not_due_when_fully_caught_up():
    # Past t+6h: three marks passed, three scheduled snapshots taken.
    p = _post(age_hours=6.5, snaps=4, scheduled=3)
    assert not _panel(p).due(NOW)


def test_self_healing_after_an_outage_is_preserved():
    """A post that slept through marks catches up one snapshot per cycle."""
    # 25h old: marks 1, 3, 6, 12, 24 have passed. Only one taken.
    p = _post(age_hours=25.0, snaps=2, scheduled=1)
    assert _panel(p).due(NOW)


def test_aged_out_posts_are_never_due():
    p = _post(age_hours=400.0, snaps=1, scheduled=0)
    assert not _panel(p).due(NOW)


def test_unpublished_future_post_is_never_due():
    p = _post(age_hours=-2.0, snaps=0, scheduled=0)
    assert not _panel(p).due(NOW)


def test_panel_from_bronze_classifies_snapshots_by_age(tmp_path):
    """End-to-end through the bronze reader, which is where ages come from."""
    import json

    d = tmp_path / "platform=youtube" / "dt=2026-09-01"
    d.mkdir(parents=True)
    (d / "posts-x.jsonl").write_text(json.dumps({
        "post_id": "v1", "creator_id": "c1",
        "published_at": "2026-09-01T10:00:00+00:00"}) + "\n", encoding="utf-8")
    (d / "snapshots-x.jsonl").write_text("\n".join([
        json.dumps({"post_id": "v1", "snapshot_ts": "2026-09-01T10:02:00+00:00",
                    "age_hours": 0.03, "at_discovery": True}),
        json.dumps({"post_id": "v1", "snapshot_ts": "2026-09-01T11:01:00+00:00",
                    "age_hours": 1.02}),
    ]) + "\n", encoding="utf-8")

    panel = Panel.from_bronze(platform="youtube", root=tmp_path)
    p = panel.posts["v1"]
    assert p.snapshot_count == 2, "both snapshots are real data and are kept"
    assert p.scheduled_snapshot_count == 1, "only the t+1h one satisfies a mark"
