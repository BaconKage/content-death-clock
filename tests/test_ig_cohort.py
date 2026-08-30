"""Instagram cohort gate.

Every call this gate lets through costs a real, non-renewable credit. There are
99 of them and no way to check the balance from the API response. A bug here
does not produce a wrong number in a table — it silently spends the budget, and
the Instagram half of the project is simply over. So the gate is tested harder
than its size suggests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cdc.collect import ig_runner
from cdc.collect.instagram import CreditLedger, CreditsExhausted

START = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
ACCOUNTS = [{"handle": f"acct{i}", "tier": "large", "category": "test"}
            for i in range(10)]


class FakeIG:
    """Counts calls so a test can assert exactly what was spent."""

    def __init__(self, budget=99, tmp_path=None):
        self.ledger = CreditLedger(budget_total=budget,
                                   path=(tmp_path / "c.json") if tmp_path else None)
        self.calls = []

    def profile(self, handle):
        self.ledger.charge(1)
        self.calls.append(handle)
        return {
            "handle": handle, "creator_id": f"id_{handle}", "username": handle,
            "full_name": handle, "follower_count": 1000, "following_count": 10,
            "is_verified": False, "fetched_at": "2026-09-05T12:00:00+00:00",
            "posts": [{
                "post_id": f"ig_{handle}_p1", "shortcode": f"{handle}p1",
                "creator_id": f"id_{handle}", "creator_handle": handle,
                "published_at": (START - timedelta(hours=2)).isoformat(),
                "is_video": True, "media_type": "video", "views": 5000,
                "likes": 300, "comments": 12, "counts_hidden": False,
                "comments_disabled": False, "caption_len": 40,
                "hashtag_count": 2, "mention_count": 0, "is_pinned": False,
            }],
        }


def cfg_patch(monkeypatch, **over):
    """Point settings() at a cohort config we control."""
    from cdc import config as C
    base = C.settings()
    ig = dict(base["instagram"])
    ig.update({"enabled": True, "mode": "cohort",
               "cohort_start_utc": START.isoformat(),
               "cohort_interval_hours": 3, "cohort_duration_hours": 48,
               "max_accounts": 6, "max_calls_per_cycle": 25})
    ig.update(over)
    merged = dict(base, instagram=ig)
    monkeypatch.setattr(C, "settings", lambda: merged)
    monkeypatch.setattr(ig_runner, "settings", lambda: merged)
    return merged


def run(tmp_path, now, client, monkeypatch, **over):
    cfg_patch(monkeypatch, **over)
    return ig_runner.run_cycle(client=client, accounts=ACCOUNTS,
                               bronze_root=tmp_path / "bronze", now=now)


# ------------------------------------------------------- refusing to spend
def test_unstarted_cohort_spends_nothing(tmp_path, monkeypatch):
    """The default state is `cohort_start_utc: null`. Until a human sets it,
    not one credit may be spent."""
    c = FakeIG(tmp_path=tmp_path)
    r = run(tmp_path, START, c, monkeypatch, cohort_start_utc=None)
    assert "not started" in r["skipped"]
    assert c.calls == []


def test_before_start_spends_nothing(tmp_path, monkeypatch):
    c = FakeIG(tmp_path=tmp_path)
    r = run(tmp_path, START - timedelta(hours=1), c, monkeypatch)
    assert "starts at" in r["skipped"]
    assert c.calls == []


def test_after_duration_spends_nothing(tmp_path, monkeypatch):
    """The window closing must preserve leftover credits, not keep polling."""
    c = FakeIG(tmp_path=tmp_path)
    r = run(tmp_path, START + timedelta(hours=49), c, monkeypatch)
    assert "complete" in r["skipped"]
    assert c.calls == []


def test_hourly_ci_invocation_between_rounds_spends_nothing(tmp_path, monkeypatch):
    """CI calls this every hour but the interval is 3h, so two of every three
    invocations must be free. This is the test that protects the budget."""
    bronze = tmp_path / "bronze"
    c = FakeIG(tmp_path=tmp_path)
    cfg_patch(monkeypatch)

    ig_runner.run_cycle(client=c, accounts=ACCOUNTS, bronze_root=bronze, now=START)
    after_first = len(c.calls)
    assert after_first == 6

    # +1h and +2h must both no-op
    for h in (1, 2):
        r = ig_runner.run_cycle(client=c, accounts=ACCOUNTS, bronze_root=bronze,
                                now=START + timedelta(hours=h))
        assert "since last round" in r["skipped"]
    assert len(c.calls) == after_first, "spent credits between scheduled rounds"

    # +3h is a real round
    ig_runner.run_cycle(client=c, accounts=ACCOUNTS, bronze_root=bronze,
                        now=START + timedelta(hours=3))
    assert len(c.calls) == after_first * 2


# ----------------------------------------------------------- spending caps
def test_never_calls_more_accounts_than_the_cohort_was_sized_for(tmp_path, monkeypatch):
    """10 accounts are configured but the cohort is sized for 6. Calling all 10
    would blow the budget by 67%."""
    c = FakeIG(tmp_path=tmp_path)
    run(tmp_path, START, c, monkeypatch)
    assert len(c.calls) == 6


def test_stops_at_remaining_credits_not_at_configured_accounts(tmp_path, monkeypatch):
    """With 2 credits left, a 6-account round must make 2 calls, not 6."""
    c = FakeIG(budget=2, tmp_path=tmp_path)
    r = run(tmp_path, START, c, monkeypatch)
    assert len(c.calls) == 2
    assert r["credits"]["remaining"] == 0


def test_whole_cohort_stays_within_budget(tmp_path, monkeypatch):
    """End to end: 48h of hourly CI invocations must cost exactly the planned
    96 credits and never exceed the 99 available."""
    bronze = tmp_path / "bronze"
    c = FakeIG(budget=99, tmp_path=tmp_path)
    cfg_patch(monkeypatch)
    for h in range(49):
        ig_runner.run_cycle(client=c, accounts=ACCOUNTS, bronze_root=bronze,
                            now=START + timedelta(hours=h))
    assert len(c.calls) == 96, f"planned 96 credits, spent {len(c.calls)}"
    assert c.ledger.remaining == 3


# ----------------------------------------------------------------- output
def test_snapshots_are_written_for_the_whole_grid(tmp_path, monkeypatch):
    c = FakeIG(tmp_path=tmp_path)
    r = run(tmp_path, START, c, monkeypatch)
    assert r["accounts_called"] == 6
    assert r["snapshots"] == 6          # one post per fake account
    assert r["new_posts"] == 6


def test_post_admitted_once_across_rounds(tmp_path, monkeypatch):
    bronze = tmp_path / "bronze"
    c = FakeIG(tmp_path=tmp_path)
    cfg_patch(monkeypatch)
    r1 = ig_runner.run_cycle(client=c, accounts=ACCOUNTS, bronze_root=bronze, now=START)
    r2 = ig_runner.run_cycle(client=c, accounts=ACCOUNTS, bronze_root=bronze,
                             now=START + timedelta(hours=3))
    assert r1["new_posts"] == 6
    assert r2["new_posts"] == 0, "re-admitted posts already in the panel"
    assert r2["snapshots"] == 6, "but must still snapshot them"
