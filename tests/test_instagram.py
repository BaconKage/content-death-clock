"""Instagram client: the two schema traps, and the credit ledger.

Written against the real response shape observed from
``/v1/instagram/profile?handle=nasa`` on 2026-08-30.
"""
from __future__ import annotations

import pytest

from cdc.collect.instagram import CreditLedger, CreditsExhausted, _normalise_post


def node(shortcode, ts, *, likes=100, comments=5, is_video=False, views=None,
         hidden=False, comments_disabled=False, pinned=None, caption=""):
    return {
        "shortcode": shortcode,
        "taken_at_timestamp": ts,
        "is_video": is_video,
        "video_view_count": views,
        "edge_liked_by": {"count": likes},
        "edge_media_to_comment": {"count": comments},
        "like_and_view_counts_disabled": hidden,
        "comments_disabled": comments_disabled,
        "pinned_for_users": pinned,
        "owner": {"id": "owner1"},
        "edge_media_to_caption": {"edges": [{"node": {"text": caption}}]} if caption else {"edges": []},
    }


USER = {"id": "u1"}


# ------------------------------------------------------------------ the traps
def test_image_posts_have_no_view_count_and_that_is_not_zero():
    """`views` must stay None for an image, never be coerced to 0. A 0 would
    look like a real observation of zero views and read as instant death."""
    p = _normalise_post(node("abc", 1787000000, is_video=False, views=None), "nasa", USER)
    assert p["views"] is None
    assert p["media_type"] == "image"
    assert p["likes"] == 100          # likes is the primary metric on Instagram


def test_video_posts_keep_their_view_count():
    p = _normalise_post(node("vid", 1787000000, is_video=True, views=548147), "nasa", USER)
    assert p["views"] == 548147
    assert p["media_type"] == "video"


def test_hidden_counts_are_flagged_not_recorded_as_zero():
    """A creator hiding likes produces 0 from the API, which is indistinguishable
    from genuine non-engagement. Must surface as None plus a flag."""
    p = _normalise_post(node("hid", 1787000000, likes=0, hidden=True), "nasa", USER)
    assert p["counts_hidden"] is True
    assert p["likes"] is None


def test_disabled_comments_are_none_not_zero():
    p = _normalise_post(node("nc", 1787000000, comments=0, comments_disabled=True),
                        "nasa", USER)
    assert p["comments"] is None
    assert p["comments_disabled"] is True


def test_likes_fall_back_to_preview_edge():
    """Instagram sometimes populates only edge_media_preview_like."""
    n = node("pv", 1787000000)
    n["edge_liked_by"] = {}
    n["edge_media_preview_like"] = {"count": 4242}
    assert _normalise_post(n, "nasa", USER)["likes"] == 4242


def test_pinned_flag_is_preserved():
    p = _normalise_post(node("pin", 1787000000, pinned=["someuser"]), "nasa", USER)
    assert p["is_pinned"] is True


def test_timestamp_converts_to_utc_iso():
    p = _normalise_post(node("ts", 1787148707), "nasa", USER)
    assert p["published_at"].startswith("2026-")
    assert p["published_at"].endswith("+00:00")


def test_caption_features_are_extracted():
    p = _normalise_post(node("cap", 1787000000, caption="Look up! #space #nasa @esa"),
                        "nasa", USER)
    assert p["hashtag_count"] == 2
    assert p["mention_count"] == 1
    assert p["caption_len"] > 0


def test_post_without_shortcode_or_timestamp_is_dropped():
    assert _normalise_post({"shortcode": "x"}, "nasa", USER) is None
    assert _normalise_post({"taken_at_timestamp": 1}, "nasa", USER) is None


# -------------------------------------------------------------- credit ledger
def test_ledger_refuses_to_overspend(tmp_path):
    led = CreditLedger(budget_total=3, path=tmp_path / "c.json")
    for _ in range(3):
        led.charge()
    with pytest.raises(CreditsExhausted):
        led.charge()
    assert led.spent_total == 3


def test_ledger_persists_across_runs(tmp_path):
    """Spending must survive process restarts, or an hourly scheduler would
    reset the count every cycle and drain the account invisibly."""
    p = tmp_path / "c.json"
    led = CreditLedger(budget_total=10, path=p)
    led.charge(4)
    led.save()

    reloaded = CreditLedger.load(budget_total=10, path=p)
    assert reloaded.spent_total == 4
    assert reloaded.remaining == 6
    assert reloaded.spent_this_run == 0


# ------------------------------------------------------- multi-key failover
def test_rolls_over_to_the_second_key_when_the_first_is_spent(tmp_path, monkeypatch):
    """Credits are per-key, so two keys are two wallets, not one bigger one.
    When the first is spent the client must keep going on the second — a cohort
    that stalls mid-window cannot be resumed, because the posts it was tracking
    will have aged out by the time anyone swaps a secret."""
    from cdc.collect.instagram import InstagramClient

    c = InstagramClient(api_keys=["KEY_ONE", "KEY_TWO"])
    c.ledgers[0].budget_total = 2
    c.ledgers[0].path = tmp_path / "k1.json"
    c.ledgers[1].budget_total = 3
    c.ledgers[1].path = tmp_path / "k2.json"
    c.dry_run = True

    for _ in range(5):
        c._get("/v1/instagram/profile", {"handle": "x"})

    assert c.ledgers[0].spent_total == 2
    assert c.ledgers[1].spent_total == 3
    assert c._idx == 1
    assert c.total_remaining == 0


def test_raises_only_when_every_key_is_spent(tmp_path):
    from cdc.collect.instagram import InstagramClient

    c = InstagramClient(api_keys=["A", "B"])
    for i, led in enumerate(c.ledgers):
        led.budget_total = 1
        led.path = tmp_path / f"k{i}.json"
    c.dry_run = True

    c._get("/x", {})
    c._get("/x", {})
    with pytest.raises(CreditsExhausted):
        c._get("/x", {})


def test_each_key_gets_its_own_ledger_file(tmp_path):
    """Key #2 must start from zero spent. Sharing a ledger would make a fresh
    100-credit key look already exhausted and refuse to run."""
    from cdc.collect.instagram import InstagramClient, key_id

    c = InstagramClient(api_keys=["KEY_ONE", "KEY_TWO"])
    assert c.ledgers[0].key_id == key_id("KEY_ONE")
    assert c.ledgers[1].key_id == key_id("KEY_TWO")
    assert c.ledgers[0].key_id != c.ledgers[1].key_id
    assert c.ledgers[1].spent_total == 0


def test_duplicate_keys_are_not_counted_as_extra_budget(monkeypatch):
    """Pasting the same key into both slots must not look like 200 credits."""
    from cdc import config as C
    # Clear every slot first: the developer's real .env.local is loaded at
    # import time, and a leftover key #3 would leak into the assertion.
    for n in range(2, 10):
        monkeypatch.delenv(f"SCRAPECREATORS_API_KEY_{n}", raising=False)
    monkeypatch.setenv("SCRAPECREATORS_API_KEY", "SAME")
    monkeypatch.setenv("SCRAPECREATORS_API_KEY_2", "SAME")
    assert C.secrets().scrapecreators_api_keys == ("SAME",)


def test_key_priority_order_is_preserved(monkeypatch):
    """Spares must be spent in order, so the primary drains first."""
    from cdc import config as C
    for n in range(2, 10):
        monkeypatch.delenv(f"SCRAPECREATORS_API_KEY_{n}", raising=False)
    monkeypatch.setenv("SCRAPECREATORS_API_KEY", "FIRST")
    monkeypatch.setenv("SCRAPECREATORS_API_KEY_2", "SECOND")
    monkeypatch.setenv("SCRAPECREATORS_API_KEY_3", "THIRD")
    assert C.secrets().scrapecreators_api_keys == ("FIRST", "SECOND", "THIRD")
