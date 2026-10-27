"""Link parsing, the daily cap, and the YouTube lookup, with a fake client (no quota spent)."""
from __future__ import annotations

from datetime import date

import pytest

from cdc.collect.youtube import QuotaExceeded, YouTubeAPIError
from cdc.tool import fetch as Fe

ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("text", [
    ID, f" {ID} ", f"https://www.youtube.com/watch?v={ID}", f"http://youtube.com/watch?v={ID}&t=42s",
    f"youtube.com/watch?feature=share&v={ID}", f"https://m.youtube.com/watch?v={ID}",
    f"https://youtu.be/{ID}", f"https://youtu.be/{ID}?si=abc", f"https://www.youtube.com/shorts/{ID}",
    f"https://www.youtube.com/live/{ID}?feature=share", f"https://www.youtube.com/embed/{ID}",
    f"https://www.youtube-nocookie.com/embed/{ID}", f"HTTPS://WWW.YOUTUBE.COM/watch?v={ID}",
])
def test_accepts_every_common_link_shape(text):
    assert Fe.parse_video_id(text) == ID


@pytest.mark.parametrize("text", [
    "", "   ", "hello", f"https://vimeo.com/{ID}", f"https://www.youtube.com.evil.example/watch?v={ID}",
    f"https://evil.example/?u=https://youtu.be/{ID}", "https://www.youtube.com/playlist?list=PL123",
    "https://www.youtube.com/@somechannel", f"https://www.youtube.com/watch?v={ID[:10]}",
    f"https://www.instagram.com/reel/{ID}/", "javascript:alert(1)", "x" * 600, None, 12345,
])
def test_rejects_everything_else(text):
    assert Fe.parse_video_id(text) is None


def test_daily_budget_caps_and_resets_at_midnight():
    b = Fe.DailyBudget(units_per_day=5)
    d1, d2 = date(2026, 10, 27), date(2026, 10, 28)
    assert b.take(2, d1) and b.take(2, d1)
    assert not b.take(2, d1)                     # 4 + 2 > 5
    assert b.take(2, d2)                         # a new UTC day starts from zero


class FakeClient:
    def __init__(self, videos=None, channels=None, raise_on=None):
        self.videos, self.channels, self.raise_on = videos or [], channels or [], raise_on
        self.calls = []

    def video_stats(self, ids):
        self.calls.append(("videos", list(ids)))
        if self.raise_on == "videos":
            raise self.exc
        return self.videos

    def channels_batch(self, ids):
        self.calls.append(("channels", list(ids)))
        return self.channels


ITEM = {"id": ID, "snippet": {"channelId": "UCx", "publishedAt": "2026-10-27T10:00:00Z", "title": "t"},
        "statistics": {"viewCount": "10"}, "contentDetails": {"duration": "PT1M"}}


def test_lookup_returns_the_raw_item_and_subscribers():
    c = FakeClient(videos=[{"raw": ITEM}], channels=[{"subscriber_count": 5000}])
    lk = Fe.lookup(ID, None, Fe.DailyBudget(), client=c)
    assert lk.item is ITEM and lk.subscribers == 5000
    assert c.calls == [("videos", [ID]), ("channels", ["UCx"])]


def test_hidden_subscriber_count_is_none_not_zero():
    c = FakeClient(videos=[{"raw": ITEM}], channels=[{"subscriber_count": None}])
    assert Fe.lookup(ID, None, Fe.DailyBudget(), client=c).subscribers is None


@pytest.mark.parametrize("exc, kind", [(QuotaExceeded("quota ... key=SECRET123"), "youtube_quota"),
                                       (YouTubeAPIError("404 for url ?key=REDACTED"), "youtube_error"),
                                       (KeyError("raw"), "youtube_error"),
                                       (ConnectionError("https://x/?key=SECRET123"), "youtube_error")])
def test_failures_become_plain_messages_without_raw_text(exc, kind):
    c = FakeClient(raise_on="videos")
    c.exc = exc
    with pytest.raises(Fe.ToolError) as e:
        Fe.lookup(ID, None, Fe.DailyBudget(), client=c)
    assert e.value.kind == kind
    assert "SECRET" not in str(e.value) and "key=" not in e.value.message
    assert e.value.__cause__ is None and e.value.__suppress_context__


def test_missing_video_and_missing_key_and_spent_budget():
    with pytest.raises(Fe.ToolError) as e:
        Fe.lookup(ID, None, Fe.DailyBudget(), client=FakeClient(videos=[]))
    assert e.value.kind == "not_found"
    with pytest.raises(Fe.ToolError) as e:
        Fe.lookup(ID, None, Fe.DailyBudget())
    assert e.value.kind == "no_key"
    spent = Fe.DailyBudget(units_per_day=1)
    with pytest.raises(Fe.ToolError) as e:
        Fe.lookup(ID, None, spent, client=FakeClient(videos=[{"raw": ITEM}]))
    assert e.value.kind == "daily_cap"
    with pytest.raises(Fe.ToolError) as e:
        Fe.lookup("not-an-id", None, Fe.DailyBudget(), client=FakeClient())
    assert e.value.kind == "bad_link"
