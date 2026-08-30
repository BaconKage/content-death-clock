"""Scrape Creators client for Instagram.

Instagram is the "Variety" half of the BDA story: a genuinely different schema,
a different engagement metric, and a different cost model from YouTube.

**Credit efficiency.** ``/v1/instagram/profile`` returns the ~12 most recent posts
*with* their like and comment counts. One credit therefore snapshots a whole
account's recent grid, the same way one YouTube playlist call discovers a whole
channel. Calling ``/v1/instagram/post`` per post would cost 12x for the same data.

**Two traps this module exists to avoid:**

1. *Grid order is not chronological.* Instagram pins posts to the top of a profile,
   so the newest post can appear last. Anything that assumes newest-first — as the
   YouTube path legitimately does for uploads playlists — will silently skip posts.
   We always sort by ``taken_at_timestamp``.

2. *Caching would fabricate observations.* The endpoint accepts ``cache_max_age``.
   A cached response carries stale counts, which we would then record against a
   fresh snapshot timestamp — inventing a flat interval that never happened and
   corrupting the velocity, hence the death label. We never send that parameter,
   and this comment exists so nobody "optimises" it back in.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from cdc.config import ROOT, settings

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
LEDGER_PATH = ROOT / "data" / "bronze" / "_credits" / "scrapecreators.json"


class CreditsExhausted(RuntimeError):
    """Local credit budget spent. Refuses to spend money we did not agree to."""


@dataclass
class CreditLedger:
    """Persistent count of credits spent.

    The API returns no balance header, so spending is invisible unless we track
    it ourselves. Without this, a misconfigured cadence drains a paid balance
    silently and the first symptom is a 402 three days into collection.
    """

    budget_total: int
    spent_total: int = 0
    spent_this_run: int = 0
    path: Path = LEDGER_PATH
    by_day: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, budget_total: int, path: Path | None = None) -> "CreditLedger":
        path = path or LEDGER_PATH
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls(budget_total=budget_total, spent_total=int(d.get("spent_total", 0)),
                       by_day=dict(d.get("by_day", {})), path=path)
        return cls(budget_total=budget_total, path=path)

    @property
    def remaining(self) -> int:
        return max(0, self.budget_total - self.spent_total)

    def charge(self, n: int = 1) -> None:
        if self.spent_total + n > self.budget_total:
            raise CreditsExhausted(
                f"Scrape Creators budget exhausted: {self.spent_total}/{self.budget_total} "
                f"credits spent. Raise instagram.credit_budget_total in settings.yaml "
                f"only if you have actually topped up the account."
            )
        self.spent_total += n
        self.spent_this_run += n
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.by_day[day] = self.by_day.get(day, 0) + n

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "spent_total": self.spent_total,
            "budget_total": self.budget_total,
            "by_day": self.by_day,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")

    def summary(self) -> dict[str, Any]:
        return {"spent_this_run": self.spent_this_run, "spent_total": self.spent_total,
                "budget_total": self.budget_total, "remaining": self.remaining}


class InstagramClient:
    def __init__(self, api_key: str, ledger: CreditLedger | None = None,
                 dry_run: bool = False) -> None:
        cfg = settings()["instagram"]
        self.api_key = api_key
        self.base = cfg["api_base"].rstrip("/")
        self.dry_run = dry_run
        self.ledger = ledger or CreditLedger.load(int(cfg["credit_budget_total"]))
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "User-Agent": "content-death-clock/0.1 (academic research)",
        })

    def _get(self, path: str, params: dict[str, Any], attempts: int = 3) -> dict:
        self.ledger.charge(1)
        if self.dry_run:
            log.info("[dry-run] GET %s %s", path, params)
            return {}

        url = f"{self.base}/{path.lstrip('/')}"
        backoff = 3.0
        for attempt in range(1, attempts + 1):
            try:
                r = self.session.get(url, params=params, timeout=45)
            except requests.RequestException as exc:
                if attempt == attempts:
                    raise
                log.warning("%s network error (%s), retry %d/%d", path, exc, attempt, attempts)
                time.sleep(backoff); backoff *= 2
                continue

            if r.status_code == 402:
                raise CreditsExhausted(f"{path}: account out of credits (HTTP 402)")
            if r.status_code in RETRY_STATUS:
                if attempt == attempts:
                    r.raise_for_status()
                log.warning("%s HTTP %d, retry %d/%d", path, r.status_code, attempt, attempts)
                time.sleep(backoff); backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ profile
    def profile(self, handle: str) -> dict[str, Any] | None:
        """One credit: account metadata plus its ~12 most recent posts with counts.

        Deliberately does NOT pass ``cache_max_age`` — see the module docstring.
        """
        handle = handle.lstrip("@")
        data = self._get("/v1/instagram/profile", {"handle": handle})
        user = (data or {}).get("data", {}).get("user")
        if not user:
            log.warning("instagram handle not found or empty: %s", handle)
            return None

        edges = (user.get("edge_owner_to_timeline_media") or {}).get("edges") or []
        posts = [_normalise_post(e.get("node") or {}, handle, user) for e in edges]
        posts = [p for p in posts if p is not None]
        # Grid order is not chronological (pinned posts float to the top).
        posts.sort(key=lambda p: p["published_at"], reverse=True)

        return {
            "handle": handle,
            "creator_id": user.get("id"),
            "username": user.get("username"),
            "full_name": user.get("full_name"),
            "follower_count": (user.get("edge_followed_by") or {}).get("count"),
            "following_count": (user.get("edge_follow") or {}).get("count"),
            "is_verified": user.get("is_verified"),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "posts": posts,
        }


# ---------------------------------------------------------------------- helpers
def _normalise_post(node: dict[str, Any], handle: str,
                    user: dict[str, Any]) -> dict[str, Any] | None:
    shortcode = node.get("shortcode")
    ts = node.get("taken_at_timestamp")
    if not shortcode or not ts:
        return None

    # A creator can hide like and view counts. The API then reports 0, which is
    # indistinguishable from a genuinely unengaged post — and would read as
    # instant death. Excluded at the source, with the reason preserved.
    counts_hidden = bool(node.get("like_and_view_counts_disabled"))

    likes = (node.get("edge_liked_by") or {}).get("count")
    if likes is None:
        # Instagram sometimes populates only the preview edge.
        likes = (node.get("edge_media_preview_like") or {}).get("count")

    caption_edges = (node.get("edge_media_to_caption") or {}).get("edges") or []
    caption = (caption_edges[0].get("node", {}).get("text", "")
               if caption_edges else "")

    return {
        "post_id": f"ig_{shortcode}",
        "shortcode": shortcode,
        "creator_id": (node.get("owner") or {}).get("id") or user.get("id"),
        "creator_handle": handle,
        "published_at": datetime.fromtimestamp(int(ts), tz=timezone.utc)
                                .isoformat(timespec="seconds"),
        "is_video": bool(node.get("is_video")),
        "media_type": "video" if node.get("is_video") else "image",
        # Instagram reports no view count for image posts, so `views` is
        # legitimately None there and `likes` is the primary metric instead.
        "views": node.get("video_view_count"),
        "likes": None if counts_hidden else likes,
        "comments": (None if node.get("comments_disabled")
                     else (node.get("edge_media_to_comment") or {}).get("count")),
        "counts_hidden": counts_hidden,
        "comments_disabled": bool(node.get("comments_disabled")),
        "caption_len": len(caption),
        "hashtag_count": caption.count("#"),
        "mention_count": caption.count("@"),
        "is_pinned": bool(node.get("pinned_for_users")),
        "raw": node,
    }
