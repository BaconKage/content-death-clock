"""What is allowed to turn the build red.

An earlier version failed the build whenever mean completeness fell below 80%.
Completeness naturally sits at 79-82%, so the alarm flapped red/green/red on
noise. These tests pin the corrected behaviour: alarm on outages and on
collapse, stay quiet through normal jitter.
"""
from __future__ import annotations

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
        "posts_evaluated": 150, "mean_completeness": 0.81, "posts_below_80pct": 20,
        "mean_lateness_hours": 0.9, "max_lateness_hours": 4.0,
        "hit_counts_by_mark": {}, "cycle_hours_observed": 12, "gaps": [],
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
