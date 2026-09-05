"""What is allowed to turn the build red.

An earlier version failed the build whenever mean completeness fell below 80%.
Completeness naturally sits at 79-82%, so the alarm flapped red/green/red on
noise. These tests pin the corrected behaviour: alarm on outages and on
collapse, stay quiet through normal jitter.

Two layers here, and the split matters. The `base()` tests stub `report()` out
and check only what `main()` decides to do with a report — that is where the
flapping regression lives. But stubbing the report is also how a real defect
hid for days: `report()` could not detect an outage that was still in progress,
and every test passed anyway because none of them ever called it. The
`test_ongoing_*` tests below therefore drive the real `report()` against a real
bronze tree on disk. Anything claiming to detect an outage must be tested
against data, not against a dict we wrote ourselves.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from cdc.collect import monitor


def _run(monkeypatch, report):
    """Invoke the CLI with a stubbed report and capture its exit code."""
    monkeypatch.setattr(monitor, "report", lambda **kw: report)
    return monitor.main(["--fail-on-gap"])


def base(**over):
    r = {
        "now": "2026-08-31T15:00:00+00:00",
        "panel": {"posts_total": 200, "posts_active": 200, "snapshots_total": 900,
                  "snapshots_per_post_mean": 4.5},
        "posts_evaluated": 150, "mean_completeness": 0.81,
        "mean_strict_completeness": 0.74, "posts_below_80pct": 20,
        "mean_lateness_hours": 0.9, "max_lateness_hours": 4.0,
        "hit_counts_by_mark": {}, "strict_counts_by_mark": {},
        "late_counts_by_mark": {}, "cycle_hours_observed": 12, "gaps": [],
        "recent_hours": 12.0,
    }
    r.update(over)
    return r


def test_normal_jitter_does_not_fail_the_build(monkeypatch):
    """The regression. 79.3% used to fail; it must not."""
    for c in (0.75, 0.79, 0.793, 0.80, 0.85):
        assert _run(monkeypatch, base(mean_completeness=c)) == 0,             f"completeness {c:.1%} failed the build - alarm is flapping again"


def test_an_outage_fails_the_build(monkeypatch):
    """The real alarm: data stopped arriving."""
    assert _run(monkeypatch, base(gaps=[("2026-08-31T02", 7)])) == 1


def test_total_collapse_fails_the_build(monkeypatch):
    """Far below the natural band means something is broken, not jittering."""
    assert _run(monkeypatch, base(mean_completeness=0.20)) == 1


def test_collapse_is_ignored_when_too_few_posts_to_judge(monkeypatch):
    """With a handful of posts the mean is noise, not evidence."""
    assert _run(monkeypatch, base(mean_completeness=0.20, posts_evaluated=8)) == 0


def test_no_posts_at_all_fails_the_build(monkeypatch):
    p = dict(base()["panel"], posts_total=0, snapshots_total=0)
    assert _run(monkeypatch, base(panel=p)) == 1


def test_posts_but_no_snapshots_fails_the_build(monkeypatch):
    p = dict(base()["panel"], snapshots_total=0)
    assert _run(monkeypatch, base(panel=p)) == 1


# --------------------------------------------------------------------------- #
# The real report(), against real files. See the module docstring.
# --------------------------------------------------------------------------- #
LAST_DATA = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def bronze(tmp_path):
    """A small YouTube panel whose most recent snapshot is at LAST_DATA.

    Six posts, hourly snapshots over the preceding six hours, so completeness is
    healthy and the only thing a test varies is how long ago that was.
    """
    root = tmp_path / "bronze"
    d = root / "platform=youtube" / "dt=2026-09-05"
    d.mkdir(parents=True)
    posts, snaps = [], []
    for i in range(6):
        pid = f"vid{i}"
        published = LAST_DATA - timedelta(hours=6)
        posts.append({"post_id": pid, "creator_id": f"ch{i}", "_platform": "youtube",
                      "published_at": published.isoformat(), "views": 0})
        for h in range(7):                       # ages 0..6h, one per hour
            ts = published + timedelta(hours=h)
            snaps.append({"post_id": pid, "creator_id": f"ch{i}",
                          "_platform": "youtube", "snapshot_ts": ts.isoformat(),
                          "published_at": published.isoformat(),
                          "age_hours": float(h), "views": 100 * (h + 1)})
    (d / "posts-2026-09-05T10.jsonl").write_text(
        "\n".join(json.dumps(r) for r in posts), encoding="utf-8")
    (d / "snapshots-2026-09-05T10.jsonl").write_text(
        "\n".join(json.dumps(r) for r in snaps), encoding="utf-8")
    return root


@pytest.mark.parametrize("hours", [1, 2, 6, 12, 24, 48, 96])
def test_ongoing_outage_is_measured_at_every_length(bronze, hours):
    """The regression.

    `gaps` walks only between the first and last hours that HAVE data, so an
    outage with no closing edge is invisible to it. On top of that, CI scopes
    the alarm to a trailing 12h window, which empties the cycle-hour set
    entirely once the outage outlasts it. Every one of these lengths previously
    reported `gaps=[]` and STATUS: healthy — the longer the outage, the more
    confident the all-clear.
    """
    r = monitor.report(now=LAST_DATA + timedelta(hours=hours),
                       recent_hours=12.0, root=bronze)
    assert r["hours_since_last_data"] == pytest.approx(hours, abs=0.01), \
        "an outage in progress must be measured regardless of the alarm window"
    assert r["last_data_at"].startswith("2026-09-05T10:00")


def _pin_now(monkeypatch, bronze, hours):
    """Let main() run its real report(), but against `bronze` at a fixed instant."""
    real = monitor.report
    monkeypatch.setattr(monitor, "report", lambda **kw: real(
        now=LAST_DATA + timedelta(hours=hours),
        recent_hours=kw.get("recent_hours"), root=bronze))


def test_ongoing_outage_fails_the_build(monkeypatch, bronze):
    """A stopped collector must turn the Actions tab red while it is still fixable."""
    _pin_now(monkeypatch, bronze, hours=6)
    assert monitor.main(["--fail-on-gap"]) == 1


def test_ongoing_outage_past_the_alarm_window_still_fails(monkeypatch, bronze):
    """The nastiest case: outage longer than the 12h judging window.

    This is the one that reported healthy most confidently, because windowing
    left nothing to find a gap in.
    """
    _pin_now(monkeypatch, bronze, hours=48)
    assert monitor.main(["--fail-on-gap"]) == 1


def test_fresh_data_stays_green(monkeypatch, bronze):
    """Normal operation: the collector runs every 30 min, so ~1h old is fine."""
    _pin_now(monkeypatch, bronze, hours=1)
    assert monitor.main(["--fail-on-gap"]) == 0
