"""The screening tool must persist what it charged, even when calls fail.

The first version of this tool did not. It charged ten profile calls in memory
and never wrote them to disk, so the ledger went on claiming a balance that had
already been spent — on a budget that cannot be topped up. The ledger is the
only record of spending that exists, because the API returns no balance.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from cdc.collect import ig_screen


def test_ledgers_are_saved_even_when_every_call_fails(monkeypatch):
    client = MagicMock()
    client.profile.side_effect = RuntimeError("HTTP 500")
    client.total_remaining = 42
    monkeypatch.setattr(ig_screen, "InstagramClient", lambda **kw: client)
    monkeypatch.setattr(ig_screen, "PAUSE_BETWEEN_CALLS_S", 0.0)
    monkeypatch.setattr(ig_screen, "secrets",
                        lambda: MagicMock(require_scrapecreators_keys=lambda: ["k"]))

    ig_screen.run(["a", "b", "c"])
    client.save_ledgers.assert_called_once()


def test_ledgers_are_saved_when_the_loop_raises(monkeypatch):
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
        ig_screen.run(["a"])
    except KeyboardInterrupt:
        pass
    client.save_ledgers.assert_called_once()


def test_dry_run_spends_nothing_and_builds_no_client(monkeypatch):
    def explode(**kw):
        raise AssertionError("dry run must not construct a client")
    monkeypatch.setattr(ig_screen, "InstagramClient", explode)
    out = ig_screen.run(["a", "b"], dry_run=True)
    assert out["would_cost"] == 2
