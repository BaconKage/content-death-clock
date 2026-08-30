"""YouTube Data API v3 client.

Quota discipline is the whole game here. The daily budget is 10,000 units and
every ``*.list`` call costs 1 unit, but ``search.list`` costs **100**. Discovering
new uploads via search would burn the entire daily budget on ~100 calls, so we
discover through each channel's uploads playlist instead (``playlistItems.list``,
1 unit) and fetch statistics in batches of 50 ids (``videos.list``, 1 unit).

Cost model, per hourly cycle, for C channels and V tracked videos::

    discovery   = C calls
    statistics  = ceil(V / 50) calls

For C=80 and V=1500 that is 80 + 30 = 110 units/cycle, ~2,640 units/day —
comfortably inside 10,000 with room for retries and a second daily backfill.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

import requests

from cdc.config import settings

log = logging.getLogger(__name__)

# Documented base cost of a `list` call.
UNIT_COST_LIST = 1
# search.list is the one expensive endpoint: 100 units per call. It is never
# used in the hourly collection loop. It is used ONCE, offline, to build the
# sampling frame, where spending 2,000 units of a 10,000/day budget is fine.
UNIT_COST_SEARCH = 100
RETRY_STATUS = {429, 500, 502, 503, 504}


class QuotaExceeded(RuntimeError):
    """The API refused us for the rest of the UTC day."""


@dataclass
class QuotaMeter:
    """Tracks units spent and refuses to blow the per-cycle guard rail."""

    budget: int
    spent: int = 0
    calls: int = 0
    by_endpoint: dict[str, int] = field(default_factory=dict)

    def charge(self, endpoint: str, units: int = UNIT_COST_LIST) -> None:
        if self.spent + units > self.budget:
            raise QuotaExceeded(
                f"cycle guard rail hit: {self.spent}+{units} > {self.budget} units. "
                "Reduce the tracked panel or raise youtube.max_quota_per_cycle."
            )
        self.spent += units
        self.calls += 1
        self.by_endpoint[endpoint] = self.by_endpoint.get(endpoint, 0) + units

    def summary(self) -> dict[str, Any]:
        return {
            "units_spent": self.spent,
            "calls": self.calls,
            "budget": self.budget,
            "by_endpoint": dict(self.by_endpoint),
        }


class YouTubeClient:
    def __init__(self, api_key: str, meter: QuotaMeter | None = None,
                 dry_run: bool = False) -> None:
        cfg = settings()["youtube"]
        self.api_key = api_key
        self.base = cfg["api_base"].rstrip("/")
        self.ids_per_call = int(cfg["ids_per_stats_call"])
        self.page_size = int(cfg["playlist_page_size"])
        self.dry_run = dry_run
        self.meter = meter or QuotaMeter(budget=int(cfg["max_quota_per_cycle"]))
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "content-death-clock/0.1 (academic research)"

    # ---------------------------------------------------------------- transport
    def _get(self, endpoint: str, params: dict[str, Any], attempts: int = 4) -> dict:
        self.meter.charge(endpoint)
        if self.dry_run:
            log.info("[dry-run] GET %s %s", endpoint,
                     {k: v for k, v in params.items() if k != "key"})
            return {"items": [], "_dry_run": True}

        url = f"{self.base}/{endpoint}"
        payload = dict(params, key=self.api_key)
        backoff = 2.0
        for attempt in range(1, attempts + 1):
            try:
                r = self.session.get(url, params=payload, timeout=30)
            except requests.RequestException as exc:
                if attempt == attempts:
                    raise
                log.warning("%s network error (%s), retry %d/%d",
                            endpoint, exc, attempt, attempts)
                time.sleep(backoff)
                backoff *= 2
                continue

            if r.status_code == 403 and "quota" in r.text.lower():
                raise QuotaExceeded(f"{endpoint}: daily quota exhausted — {r.text[:200]}")
            if r.status_code in RETRY_STATUS:
                if attempt == attempts:
                    r.raise_for_status()
                log.warning("%s HTTP %d, retry %d/%d",
                            endpoint, r.status_code, attempt, attempts)
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------- channel meta
    def resolve_channel(self, handle: str) -> dict[str, Any] | None:
        """Handle (``@name``) -> channel id, uploads playlist, subscriber count."""
        data = self._get("channels", {
            "part": "snippet,statistics,contentDetails",
            "forHandle": handle if handle.startswith("@") else f"@{handle}",
        })
        items = data.get("items") or []
        if not items:
            log.warning("handle not found: %s", handle)
            return None
        it = items[0]
        stats = it.get("statistics", {})
        return {
            "handle": handle,
            "channel_id": it["id"],
            "title": it["snippet"]["title"],
            "uploads_playlist": it["contentDetails"]["relatedPlaylists"]["uploads"],
            # Hidden subscriber counts exist. None is honest; 0 would be a lie
            # that silently becomes a feature value.
            "subscriber_count": (None if stats.get("hiddenSubscriberCount")
                                 else _int(stats.get("subscriberCount"))),
            "video_count": _int(stats.get("videoCount")),
            "channel_view_count": _int(stats.get("viewCount")),
            "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ----------------------------------------------------------------- discovery
    def recent_uploads(self, uploads_playlist: str, since: datetime,
                       max_pages: int = 2) -> list[dict[str, Any]]:
        """Videos published since `since`, newest first.

        The uploads playlist is ordered newest-first, so we stop paging as soon
        as we see something older than the window instead of walking history.
        """
        out: list[dict[str, Any]] = []
        page_token: str | None = None
        for _ in range(max_pages):
            params = {"part": "contentDetails,snippet",
                      "playlistId": uploads_playlist,
                      "maxResults": self.page_size}
            if page_token:
                params["pageToken"] = page_token
            data = self._get("playlistItems", params)
            items = data.get("items") or []
            stop = False
            for it in items:
                cd = it.get("contentDetails", {})
                vid = cd.get("videoId")
                published = _parse_ts(cd.get("videoPublishedAt"))
                if not vid or published is None:
                    continue
                if published < since:
                    stop = True
                    continue
                out.append({
                    "video_id": vid,
                    "published_at": published.isoformat(),
                    "channel_id": it.get("snippet", {}).get("channelId"),
                    "title": it.get("snippet", {}).get("title"),
                })
            page_token = data.get("nextPageToken")
            if stop or not page_token:
                break
        return out

    # ----------------------------------------------------------- frame building
    def search_recent_videos(self, query: str, published_after: datetime,
                             region_code: str | None = None,
                             max_results: int = 50) -> list[dict[str, Any]]:
        """Recent videos matching a query. 100 units — frame building only.

        We search *videos ordered by date* rather than channels, for two reasons.
        Channel search ranks by relevance, which is a proxy for popularity, so it
        essentially cannot surface a channel under 10k subscribers — exactly the
        stratum we need. Recent-video search surfaces creators of every size, and
        it inherently selects for channels that upload, which is the other thing
        we require: a channel that posts monthly contributes nothing to a
        three-week collection window.
        """
        params = {
            "part": "snippet", "type": "video", "order": "date", "q": query,
            "maxResults": min(max_results, 50),
            "publishedAfter": published_after.astimezone(timezone.utc)
                                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if region_code:
            params["regionCode"] = region_code
        self.meter.charge("search", UNIT_COST_SEARCH)
        if self.dry_run:
            log.info("[dry-run] SEARCH %s", query)
            return []
        # _get charges 1 unit of its own, so refund it: search is billed above.
        self.meter.spent -= UNIT_COST_LIST
        self.meter.calls -= 1
        data = self._get("search", params)
        out = []
        for it in data.get("items") or []:
            sn = it.get("snippet", {})
            if sn.get("channelId"):
                out.append({"channel_id": sn["channelId"],
                            "channel_title": sn.get("channelTitle"),
                            "video_published_at": sn.get("publishedAt")})
        return out

    def channels_batch(self, channel_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Resolve up to 50 channel ids per 1-unit call.

        Batching matters: resolving 250 candidates one at a time costs 250 units,
        batched it costs 5.
        """
        ids = list(dict.fromkeys(channel_ids))
        out = []
        for batch in _chunks(ids, 50):
            data = self._get("channels", {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(batch),
            })
            for it in data.get("items") or []:
                stats = it.get("statistics", {})
                out.append({
                    "channel_id": it["id"],
                    "title": it["snippet"]["title"],
                    "handle": it["snippet"].get("customUrl"),
                    "uploads_playlist": it["contentDetails"]["relatedPlaylists"]["uploads"],
                    "subscriber_count": (None if stats.get("hiddenSubscriberCount")
                                         else _int(stats.get("subscriberCount"))),
                    "video_count": _int(stats.get("videoCount")),
                    "country": it["snippet"].get("country"),
                })
        return out

    # ---------------------------------------------------------------- statistics
    def video_stats(self, video_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Batched statistics pull. 50 ids per 1-unit call."""
        ids = list(dict.fromkeys(video_ids))          # dedupe, preserve order
        out: list[dict[str, Any]] = []
        for batch in _chunks(ids, self.ids_per_call):
            data = self._get("videos", {
                "part": "statistics,snippet,contentDetails,status",
                "id": ",".join(batch),
            })
            for it in data.get("items") or []:
                out.append(_normalise_video(it))
        return out


# ---------------------------------------------------------------------- helpers
def _normalise_video(item: dict[str, Any]) -> dict[str, Any]:
    sn = item.get("snippet", {})
    st = item.get("statistics", {})
    cd = item.get("contentDetails", {})
    return {
        "post_id": item["id"],
        "creator_id": sn.get("channelId"),
        "creator_title": sn.get("channelTitle"),
        "published_at": sn.get("publishedAt"),
        "title": sn.get("title"),
        "description_len": len(sn.get("description") or ""),
        "tag_count": len(sn.get("tags") or []),
        "category_id": sn.get("categoryId"),
        "duration_iso": cd.get("duration"),
        # A creator can disable likes or comments. None means "not reported",
        # which is not the same as zero — conflating them would bias the label.
        "views": _int(st.get("viewCount")),
        "likes": _int(st.get("likeCount")),
        "comments": _int(st.get("commentCount")),
        "raw": item,
    }


def _int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _chunks(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
