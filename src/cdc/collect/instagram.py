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

import hashlib
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
CREDITS_DIR = ROOT / "data" / "bronze" / "_credits"


def key_id(api_key: str) -> str:
    """Short, stable, non-reversible identifier for an API key.

    The ledger is per-key because credits are per-key. A second key with its own
    100 credits must start from zero spent, not inherit the first key's count and
    immediately refuse to run. We hash rather than store the key itself so the
    ledger stays committable to a public repo.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


def ledger_path(kid: str) -> Path:
    return CREDITS_DIR / f"scrapecreators-{kid}.json"


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
    path: Path | None = None
    key_id: str | None = None
    by_day: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, budget_total: int, api_key: str | None = None,
             path: Path | None = None) -> "CreditLedger":
        kid = key_id(api_key) if api_key else None
        path = path or (ledger_path(kid) if kid else CREDITS_DIR / "scrapecreators.json")
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return cls(budget_total=budget_total, spent_total=int(d.get("spent_total", 0)),
                       by_day=dict(d.get("by_day", {})), path=path, key_id=kid)
        return cls(budget_total=budget_total, path=path, key_id=kid)

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
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "key_id": self.key_id,
            "spent_total": self.spent_total,
            "budget_total": self.budget_total,
            "by_day": self.by_day,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")

    def summary(self) -> dict[str, Any]:
        return {"key_id": self.key_id, "spent_this_run": self.spent_this_run,
                "spent_total": self.spent_total, "budget_total": self.budget_total,
                "remaining": self.remaining}


class InstagramClient:
    """Scrape Creators client with multi-key failover.

    Credits are per-key and non-transferable, so several keys are several
    wallets rather than one big one. Each gets its own ledger; the client spends
    the first key that still has credit and rolls over to the next the moment it
    does not. Without this, exhausting a key stops collection dead until someone
    notices and swaps a secret — and a cohort that stalls mid-window cannot be
    resumed later, because the posts it was tracking will have aged out.
    """

    def __init__(self, api_key: str | None = None, ledger: CreditLedger | None = None,
                 dry_run: bool = False, api_keys: list[str] | None = None) -> None:
        cfg = settings()["instagram"]
        keys = list(api_keys) if api_keys else ([api_key] if api_key else [])
        if not keys:
            raise ValueError("InstagramClient needs at least one API key")
        self.api_keys = keys
        self.base = cfg["api_base"].rstrip("/")
        self.dry_run = dry_run
        budget = int(cfg["credit_budget_total"])

        if ledger is not None:
            # Explicit ledger (tests) pins a single key.
            self.ledgers = [ledger]
        else:
            self.ledgers = [CreditLedger.load(budget, api_key=k) for k in keys]
        self._idx = 0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "content-death-clock/0.1 (academic research)"

    # --------------------------------------------------------------- key state
    @property
    def ledger(self) -> CreditLedger:
        """Ledger for the key currently being spent."""
        return self.ledgers[min(self._idx, len(self.ledgers) - 1)]

    @property
    def api_key(self) -> str:
        return self.api_keys[min(self._idx, len(self.api_keys) - 1)]

    @property
    def total_remaining(self) -> int:
        return sum(l.remaining for l in self.ledgers)

    def _advance_key(self) -> bool:
        """Move to the next key with credit. False if there is none."""
        while self._idx < len(self.ledgers) - 1:
            self._idx += 1
            if self.ledgers[self._idx].remaining > 0:
                log.warning("switched to Scrape Creators key #%d (%d credits)",
                            self._idx + 1, self.ledgers[self._idx].remaining)
                return True
        return False

    def save_ledgers(self) -> None:
        for l in self.ledgers:
            l.save()

    def credits_summary(self) -> dict[str, Any]:
        return {"active_key_index": self._idx,
                "total_remaining": self.total_remaining,
                "keys": [l.summary() for l in self.ledgers]}

    def _get(self, path: str, params: dict[str, Any], attempts: int = 3) -> dict:
        # Charge the active key, rolling over to a spare if it is spent.
        while True:
            try:
                self.ledger.charge(1)
                break
            except CreditsExhausted:
                if not self._advance_key():
                    raise
        self.session.headers["x-api-key"] = self.api_key
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
                # The account disagrees with our ledger — trust the account.
                # Mark this key spent so we stop guessing, and try a spare.
                log.warning("key #%d returned 402; marking it exhausted", self._idx + 1)
                self.ledger.spent_total = self.ledger.budget_total
                if self._advance_key():
                    self.session.headers["x-api-key"] = self.api_key
                    self.ledger.charge(1)
                    continue
                raise CreditsExhausted(f"{path}: all keys out of credits (HTTP 402)")
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
