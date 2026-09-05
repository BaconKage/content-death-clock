"""The screening tool must persist what it charged, even when calls fail.

The first version of this tool did not. It charged ten profile calls in memory
and never wrote them to disk, so the ledger went on claiming a balance that had
already been spent — on a budget that cannot be topped up. The ledger is the
only record of spending that exists, because the API returns no balance.
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import MagicMock

from cdc.collect import ig_screen


def test_ledgers_are_saved_even_when_every_call_fails(monkeypatch, tmp_path):
    client = MagicMock()
    client.profile.side_effect = RuntimeError("HTTP 500")
    client.total_remaining = 42
    monkeypatch.setattr(ig_screen, "InstagramClient", lambda **kw: client)
    monkeypatch.setattr(ig_screen, "PAUSE_BETWEEN_CALLS_S", 0.0)
    monkeypatch.setattr(ig_screen, "secrets",
                        lambda: MagicMock(require_scrapecreators_keys=lambda: ["k"]))

    ig_screen.run(["a", "b", "c"], out_path=tmp_path / "screening.json")
    client.save_ledgers.assert_called_once()


def test_ledgers_are_saved_when_the_loop_raises(monkeypatch, tmp_path):
    """A crash mid-run must not lose the record of what was already spent."""
    client = MagicMock()
    client.total_remaining = 7
    monkeypatch.setattr(ig_screen, "InstagramClient", lambda **kw: client)
    monkeypatch.setattr(ig_screen, "PAUSE_BETWEEN_CALLS_S", 0.0)
    monkeypatch.setattr(ig_screen, "secrets",
                        lambda: MagicMock(require_scrapecreators_keys=lambda: ["k"]))

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(ig_screen, "screen_one", boom)
    try:
        ig_screen.run(["a"], out_path=tmp_path / "screening.json")
    except KeyboardInterrupt:
        pass
    client.save_ledgers.assert_called_once()


def test_dry_run_spends_nothing_and_builds_no_client(monkeypatch):
    def explode(**kw):
        raise AssertionError("dry run must not construct a client")
    monkeypatch.setattr(ig_screen, "InstagramClient", explode)
    out = ig_screen.run(["a", "b"], dry_run=True)
    assert out["would_cost"] == 2


# --------------------------------------------------------------- pinned posts
def _post(iso, pinned=False):
    return {"published_at": iso, "is_pinned": pinned, "likes": 1, "is_video": False}


def test_a_single_pinned_post_must_not_make_a_fast_account_look_slow():
    """The defect that admitted @sarcastic_us to Cohort A.

    Twelve posts published five hours apart is a fast account. One pinned post
    from eight days earlier stretches the naive span to ~197h and makes it read
    as 1.1 posts/day — which is how the worst account in Cohort A passed
    screening.
    """
    from unittest.mock import MagicMock
    recent = [_post(f"2026-08-30T{h:02d}:00:00+00:00") for h in range(10, 22, 1)]
    with_pin = recent + [_post("2026-08-22T11:20:21+00:00", pinned=True)]

    client = MagicMock()
    client.profile.return_value = {"follower_count": 1, "posts": with_pin}
    r = ig_screen.screen_one(client, "sarcastic_us", interval_h=3.0, duration_h=48.0)

    assert r["pinned_excluded"] == 1
    assert r["posts_per_day"] > 20, (
        f"pinned post still dragging the rate down: {r['posts_per_day']}/day")
    assert r["verdict"] == "REJECT"


def test_unpinned_measurement_is_unaffected_when_nothing_is_pinned():
    from unittest.mock import MagicMock
    posts = [_post(f"2026-08-{d:02d}T12:00:00+00:00") for d in range(20, 26)]
    client = MagicMock()
    client.profile.return_value = {"follower_count": 1, "posts": posts}
    r = ig_screen.screen_one(client, "steady", interval_h=3.0, duration_h=48.0)
    assert r["pinned_excluded"] == 0
    assert r["posts_per_day"] == pytest.approx(1.0, abs=0.05)


def test_an_all_pinned_grid_is_refused_rather_than_guessed():
    from unittest.mock import MagicMock
    client = MagicMock()
    client.profile.return_value = {
        "follower_count": 1,
        "posts": [_post("2026-08-20T12:00:00+00:00", pinned=True),
                  _post("2026-08-21T12:00:00+00:00", pinned=True)]}
    r = ig_screen.screen_one(client, "allpinned", interval_h=3.0, duration_h=48.0)
    assert "cannot measure a rate" in r["status"]


def test_screening_keeps_its_own_evidence():
    """A result that discards its inputs makes the next question cost credits."""
    from unittest.mock import MagicMock
    posts = [_post(f"2026-08-{d:02d}T12:00:00+00:00") for d in range(20, 26)]
    client = MagicMock()
    client.profile.return_value = {"follower_count": 1, "posts": posts}
    r = ig_screen.screen_one(client, "steady", interval_h=3.0, duration_h=48.0)
    assert len(r["observations"]) == len(posts)
    assert "is_pinned" in r["observations"][0]


# ------------------------------------------------- the committed evidence
def test_screening_never_overwrites_the_committed_artifact(monkeypatch, tmp_path):
    """The regression that destroyed the Cohort B selection evidence.

    `run()` wrote to `config/ig_screening.json` unconditionally, and the tests
    above call it with fake handles against a mocked client that fails every
    call. So running the suite replaced the real measurements with three
    placeholder HTTP 500s — and that state was then committed, leaving the
    account table in `channels.yaml` with nothing behind it. The real artifact
    had to be recovered from git history.
    """
    client = MagicMock()
    client.profile.side_effect = RuntimeError("HTTP 500")
    client.total_remaining = 9
    monkeypatch.setattr(ig_screen, "InstagramClient", lambda **kw: client)
    monkeypatch.setattr(ig_screen, "PAUSE_BETWEEN_CALLS_S", 0.0)
    monkeypatch.setattr(ig_screen, "secrets",
                        lambda: MagicMock(require_scrapecreators_keys=lambda: ["k"]))

    before = ig_screen.DEFAULT_OUT.read_bytes() if ig_screen.DEFAULT_OUT.exists() else None
    out = tmp_path / "screening.json"
    ig_screen.run(["a", "b", "c"], out_path=out)

    assert out.exists(), "the override path must actually be written"
    after = ig_screen.DEFAULT_OUT.read_bytes() if ig_screen.DEFAULT_OUT.exists() else None
    assert after == before, "screening wrote over the committed artifact"


def test_committed_artifact_backs_the_cohort_b_table():
    """The account table in channels.yaml must be checkable, not just asserted.

    Cohort A failed *because* its screening was wrong, so the numbers that chose
    Cohort B are exactly the ones a reader should be able to verify.
    """
    if not ig_screen.DEFAULT_OUT.exists():
        pytest.skip("no screening artifact committed")
    data = json.loads(ig_screen.DEFAULT_OUT.read_text(encoding="utf-8"))
    by_handle = {r["handle"]: r for r in data["results"]}

    for handle, rate in (("spotify", 1.5), ("bonappetitmag", 1.3), ("9gag", 6.1)):
        assert handle in by_handle, f"{handle} was selected but never screened"
        assert by_handle[handle]["posts_per_day"] == pytest.approx(rate, abs=0.05), \
            f"channels.yaml claims {rate}/day for {handle}"
        assert by_handle[handle]["status"] == "ok"
