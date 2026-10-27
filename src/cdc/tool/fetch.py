"""Link parsing, the YouTube lookup, and the tool's daily unit cap.

**Two units per report.** One ``videos.list`` call and one ``channels.list``
call, through the collector's own client, so the key handling (and the
redaction added after the 2026-09-17 leak) is the same code that protects the
collector. Every error that reaches the page is one of the plain ``ToolError``
kinds below; raw exception text, which could carry a URL, never does.

**A cap of its own.** The tool should run on a key in its own Google Cloud
project, so it cannot spend the collector's quota at all. The daily cap is a
second wall: whatever key it is given, it stops at ``DAILY_UNITS``.

**Nothing is stored.** Links and video ids are not logged or written anywhere;
a lookup lives only in the page's short-lived cache.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

UNITS_PER_REPORT = 2
DAILY_UNITS = 1000                     # a tenth of one project's daily quota
MAX_INPUT_CHARS = 500

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
          "youtu.be", "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com"}
_PATH_KINDS = {"shorts", "live", "embed", "v", "e"}


def parse_video_id(text: str) -> str | None:
    """The 11-character video id from a YouTube link (or a bare id); None otherwise."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s or len(s) > MAX_INPUT_CHARS:
        return None
    if _ID.match(s):
        return s
    if not re.match(r"^https?://", s, re.IGNORECASE):
        s = "https://" + s
    try:
        u = urlparse(s)
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    if host not in _HOSTS:
        return None
    parts = [p for p in u.path.split("/") if p]
    if host.endswith("youtu.be"):
        cand = parts[0] if parts else ""
    elif parts[:1] == ["watch"]:
        cand = (parse_qs(u.query).get("v") or [""])[0]
    elif len(parts) >= 2 and parts[0] in _PATH_KINDS:
        cand = parts[1]
    else:
        cand = ""
    return cand if _ID.match(cand) else None


class ToolError(Exception):
    """A failure the page can explain. ``kind`` picks the message; no raw text."""

    MESSAGES = {
        "bad_link": "That doesn't look like a link to a YouTube video. Try one like "
                    "https://www.youtube.com/watch?v=... or https://youtu.be/...",
        "not_found": "YouTube has no public video at that link. It may be private, deleted, or mistyped.",
        "daily_cap": "The tool has reached its daily limit of lookups. Please try again tomorrow (UTC).",
        "youtube_quota": "YouTube's daily limit for this tool has been reached. Please try again tomorrow.",
        "youtube_error": "YouTube didn't answer properly just now. Please try again in a minute.",
        "no_key": "The tool isn't connected to YouTube yet (no API key is configured).",
    }

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind

    @property
    def message(self) -> str:
        return self.MESSAGES.get(self.kind, self.MESSAGES["youtube_error"])


class DailyBudget:
    """Units spent today (UTC), refusing past the cap. Thread-safe; resets at midnight."""

    def __init__(self, units_per_day: int = DAILY_UNITS) -> None:
        self.cap = units_per_day
        self.day: date | None = None
        self.spent = 0
        self._lock = threading.Lock()

    def take(self, units: int, today: date | None = None) -> bool:
        today = today or datetime.now(timezone.utc).date()
        with self._lock:
            if self.day != today:
                self.day, self.spent = today, 0
            if self.spent + units > self.cap:
                return False
            self.spent += units
            return True


@dataclass
class Lookup:
    item: dict[str, Any]               # the raw videos.list item
    subscribers: int | None            # None when the channel hides it
    fetched_at: datetime


def lookup(video_id: str, api_key: str | None, budget: DailyBudget, client=None) -> Lookup:
    """Fetch one video and its channel's subscriber count: two units."""
    from cdc.collect.youtube import QuotaExceeded, QuotaMeter, YouTubeAPIError, YouTubeClient

    if not _ID.match(video_id or ""):
        raise ToolError("bad_link")
    if not api_key and client is None:
        raise ToolError("no_key")
    if not budget.take(UNITS_PER_REPORT):
        raise ToolError("daily_cap")
    client = client or YouTubeClient(api_key, meter=QuotaMeter(budget=UNITS_PER_REPORT))
    try:
        videos = client.video_stats([video_id])
        if not videos:
            raise ToolError("not_found")
        item = videos[0]["raw"]
        channel_id = (item.get("snippet") or {}).get("channelId")
        chans = client.channels_batch([channel_id]) if channel_id else []
    except ToolError:
        raise
    except QuotaExceeded:
        raise ToolError("youtube_quota") from None
    except YouTubeAPIError:
        raise ToolError("youtube_error") from None
    except Exception:
        # Anything else (a malformed response, a network hiccup the client did
        # not wrap): the page gets a plain message and no exception text.
        raise ToolError("youtube_error") from None
    subs = chans[0].get("subscriber_count") if chans else None
    return Lookup(item=item, subscribers=subs, fetched_at=datetime.now(timezone.utc))
