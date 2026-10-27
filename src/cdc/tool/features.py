"""The instant model's inputs, from one reading of a video.

**One function, two callers.** ``instant_features`` is the only place the
inputs are computed. Training reaches it from the silver tables
(``from_silver``); the app reaches it from a live ``videos.list`` response
(``from_api``). The usual way a deployed model goes quietly wrong is that the
two paths compute "the same" input slightly differently: a +1 here, a missing
value turned into 0 there. Sharing the function removes that possibility, and
``tests/test_tool_features.py`` pushes a recorded API response through both
paths, including the real silver transform, and requires identical results.

Everything is rounded (logs to 2 decimals, age to 0.01h) before it leaves this
module. That keeps the shipped training table from carrying exact counts, and
because the rounding is applied on both paths, the model sees identical inputs
in training and in use.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

# Readings cluster at the collector's scheduled marks; these four are the ones
# inside the first half-day, which is where the tool gives odds.
CHECKPOINTS = (1.0, 3.0, 6.0, 12.0)
# Every scheduled mark (config: collection.snapshot_schedule_hours, minus the
# 336h reading that was never taken). Fade times are recorded on this grid.
MARKS = (1.0, 3.0, 6.0, 12.0, 24.0, 36.0, 48.0, 72.0, 96.0, 120.0, 168.0, 240.0)
# A reading belongs to mark m if taken within [m - 0.75, m + 1.0] hours.
# 0.75 is the collector's configured tolerance; readings are usually late, not
# early, hence the wider upper side.
EARLY_TOL_H, LATE_TOL_H = 0.75, 1.0
# Odds are given only between these ages: the training data has readings from
# the first checkpoint's window to the last one's.
MIN_AGE_H = CHECKPOINTS[0] - EARLY_TOL_H
MAX_AGE_H = CHECKPOINTS[-1] + LATE_TOL_H

FEATURES = [
    "age_h", "log_views", "log_views_per_h", "log_like_rate", "log_comment_rate",
    "log_views_per_sub", "log_subscribers", "log_duration_sec", "publish_hour_utc",
    "title_len",
]

NAN = float("nan")


def _num(v: Any) -> float:
    """A count as float, with None (not reported) kept as NaN, never 0."""
    if v is None:
        return NAN
    try:
        f = float(v)
    except (TypeError, ValueError):
        return NAN
    return f if math.isfinite(f) else NAN


def _r(x: float, nd: int = 2) -> float:
    return round(x, nd) if math.isfinite(x) else NAN


def instant_features(age_h: float, views: Any, likes: Any, comments: Any,
                     subscribers: Any, duration_sec: Any, publish_hour_utc: Any,
                     title_len: Any) -> dict[str, float]:
    """The model's inputs from one reading. Missing stays missing (NaN)."""
    age = _num(age_h)
    v, lk, cm = _num(views), _num(likes), _num(comments)
    subs, dur = _num(subscribers), _num(duration_sec)
    if not (math.isfinite(age) and age > 0):
        raise ValueError(f"age must be positive, got {age_h!r}")
    if not (math.isfinite(v) and v >= 0):
        raise ValueError(f"views must be a non-negative count, got {views!r}")

    per_view = (lambda n, floor: math.log10(max(n / v, floor))
                if v > 0 and math.isfinite(n) else NAN)
    return {
        "age_h": _r(age),
        "log_views": _r(math.log10(v + 1)),
        "log_views_per_h": _r(math.log10(v / age + 1)),
        # Floors match the research features (cdc.eval.practical.engagement_at):
        # a video with no likes yet gets log10(1e-5), not minus infinity.
        "log_like_rate": _r(per_view(lk, 1e-5)),
        "log_comment_rate": _r(per_view(cm, 1e-6)),
        "log_views_per_sub": _r(math.log10((v + 1) / subs) if subs > 0 else NAN),
        "log_subscribers": _r(math.log10(subs + 1) if subs > 0 else NAN),
        "log_duration_sec": _r(math.log10(dur + 1) if dur > 0 else NAN),
        "publish_hour_utc": _num(publish_hour_utc),
        "title_len": _num(title_len),
    }


# --------------------------------------------------------------- the two paths
def from_silver(age_h: float, views: Any, likes: Any, comments: Any,
                post: dict[str, Any]) -> dict[str, float]:
    """Training path: one silver snapshot reading plus its silver post row."""
    pub = post.get("published_at")
    return instant_features(
        age_h, views, likes, comments,
        subscribers=post.get("follower_count"),
        duration_sec=post.get("duration_sec"),
        publish_hour_utc=pub.hour if pub is not None and not _isnan(pub) else None,
        title_len=post.get("title_len"))


def from_api(item: dict[str, Any], subscribers: Any, now: datetime
             ) -> tuple[dict[str, float], dict[str, Any]]:
    """App path: a raw ``videos.list`` item, the channel's subscriber count, now.

    Parses with the collector's own ``_normalise_video`` and silver's own
    duration parser and title rule, so a field means the same thing here as in
    the data the model was trained on. Returns the inputs and the plain facts
    the report shows.
    """
    from cdc.collect.youtube import _normalise_video, _parse_ts
    from cdc.transform.silver import _iso_duration_sec

    v = _normalise_video(item)
    pub = _parse_ts(v["published_at"])
    if pub is None:
        raise ValueError("the video has no publish time")
    age_h = (now - pub).total_seconds() / 3600.0
    title_len = len(v.get("title") or "") or None          # as silver.build_posts
    dur = _iso_duration_sec(v.get("duration_iso"))
    facts = {"video_id": v["post_id"], "title": v.get("title") or "",
             "channel": v.get("creator_title") or "", "channel_id": v.get("creator_id"),
             "published_at": pub, "age_h": age_h, "views": v["views"],
             "likes": v["likes"], "comments": v["comments"],
             "subscribers": subscribers, "duration_sec": dur}
    if age_h <= 0 or v["views"] is None:
        return {}, facts
    return instant_features(age_h, v["views"], v["likes"], v["comments"], subscribers,
                            dur, pub.hour, title_len), facts


def _isnan(x: Any) -> bool:
    try:
        return x != x
    except Exception:
        return False


def nearest_checkpoint(age_h: float) -> float:
    """The checkpoint closest to an age on a log scale (1h is nearer 2h than 3h is)."""
    return min(CHECKPOINTS, key=lambda c: abs(math.log(age_h / c)))


def mark_of(age_h: float) -> float | None:
    """The scheduled mark a reading at this age belongs to, if any."""
    for m in MARKS:
        if m - EARLY_TOL_H <= age_h <= m + LATE_TOL_H:
            return m
    return None


def fade_mark(t: float) -> float:
    """Put a fade (or censoring) time on the mark grid.

    The death label is always the age of a reading, and readings are taken
    when a video's age crosses a scheduled mark, usually a few minutes late.
    So a death recorded at 24.3h is the 24h mark's reading, and one at 23.6h
    is the same mark read slightly early. The rule: the largest mark at most
    ``EARLY_TOL_H`` past the recorded time, and never below the first mark.
    """
    best = MARKS[0]
    for m in MARKS:
        if m <= t + EARLY_TOL_H:
            best = m
    return best
