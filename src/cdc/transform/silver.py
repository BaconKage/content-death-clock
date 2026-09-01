"""Bronze -> silver: raw API payloads become one clean, typed, cross-platform table.

Silver is where the two platforms stop looking different. YouTube gives views on
every video; Instagram gives views only on Reels and likes on everything. Rather
than pretend these are the same number, silver keeps every raw metric AND adds an
explicit ``primary_metric`` / ``primary_value`` pair resolved per platform from
``settings.labels.primary_metric``. Downstream code reads the primary value and
never has to know which platform a row came from — but the choice stays visible
in the data, so the paper can report it rather than hide it.

Three invariants this module enforces:

* **Deduplication.** Bronze is append-only and its writer already dedupes, but
  silver must not trust that. A duplicated snapshot would corrupt every velocity
  computation downstream, so the guarantee is enforced twice.
* **Monotonic ages.** Ages come from actual observed timestamps, never an assumed
  schedule. A late snapshot is recorded as late.
* **Zero is not missing.** A metric a creator has hidden arrives as None and stays
  None. Coercing it to 0 would read as a real observation of no engagement, which
  the death label would score as instant death.

Usage::

    python -m cdc.transform.silver
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from cdc.config import ROOT, path_for, settings
from cdc.storage.bronze import dedupe, iter_bronze

log = logging.getLogger("cdc.silver")

SNAPSHOT_KEY = ("post_id", "snapshot_ts")


def _parse_ts(v: Any) -> pd.Timestamp | None:
    if v in (None, ""):
        return None
    try:
        ts = pd.Timestamp(str(v).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def build_posts(platforms: tuple[str, ...] = ("youtube", "instagram")) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for plat in platforms:
        recs = dedupe(iter_bronze("posts", platform=plat), ("post_id",))
        for r in recs:
            rows.append({
                "post_id": r.get("post_id"),
                "platform": plat,
                "creator_id": r.get("creator_id"),
                "creator_handle": r.get("creator_handle") or r.get("creator_title"),
                # Instagram only: the grid this post was found on, which is not
                # always its owner (collab posts). Older records predate the
                # field and fall back to what they have.
                "queried_handle": r.get("queried_handle") or r.get("creator_handle"),
                "owned_by_queried_account": r.get("owned_by_queried_account"),
                "published_at": _parse_ts(r.get("published_at")),
                "discovered_at": _parse_ts(r.get("discovered_at")),
                "stratum_tier": r.get("stratum_tier"),
                "stratum_category": r.get("stratum_category"),
                "media_type": r.get("media_type") or "video",
                # --- metadata features, all known at publish time ---
                "title_len": len(r.get("title") or "") or None,
                "description_len": r.get("description_len"),
                "caption_len": r.get("caption_len"),
                "tag_count": r.get("tag_count"),
                "hashtag_count": r.get("hashtag_count"),
                "mention_count": r.get("mention_count"),
                "duration_iso": r.get("duration_iso"),
                "duration_sec": _iso_duration_sec(r.get("duration_iso")),
                "category_id": r.get("category_id"),
                "follower_count": r.get("follower_count"),
                "counts_hidden": bool(r.get("counts_hidden", False)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["post_id", "published_at"])
    df = df.drop_duplicates("post_id").reset_index(drop=True)
    df = _attach_creator_size(df)
    return _mark_creator_cap(df)


def _attach_creator_size(posts: pd.DataFrame) -> pd.DataFrame:
    """Fill follower_count for YouTube from the resolved sampling frame.

    Instagram carries it on every post because the profile call returns it.
    YouTube does not: videos.list says nothing about the channel's subscriber
    count, so it was silently 0% populated — and it is the covariate the H2
    baseline is built on. Without it the "creator size alone" comparison, the
    single most important test in the study, degenerated into a constant.

    Taken from config/channels.resolved.yaml, which records each channel's
    subscriber count at frame-resolution time. Using the frozen frame value
    rather than a live one is deliberate: subscriber counts drift, and the
    stratification the paper reports is the one measured when the frame was
    built.
    """
    import yaml
    path = ROOT / "config" / "channels.resolved.yaml"
    if not path.exists():
        return posts
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    subs = {c["channel_id"]: c.get("subscriber_count")
            for c in (doc.get("channels") or []) if c.get("channel_id")}
    if not subs:
        return posts
    yt = posts["platform"] == "youtube"
    filled = posts.loc[yt, "creator_id"].map(subs)
    posts.loc[yt, "follower_count"] = posts.loc[yt, "follower_count"].fillna(filled)
    return posts


def _mark_creator_cap(posts: pd.DataFrame) -> pd.DataFrame:
    """Flag posts beyond the per-creator daily cap. Plan amendment 2026-08-31.

    Applied here as well as at admission, because posts collected *before* the
    amendment are already in bronze and bronze is append-only evidence that must
    not be rewritten. Marking them at analysis time excludes them consistently
    while leaving the raw record intact and the exclusion countable.

    Deterministic: within a creator-day, the earliest published are kept. For the
    case that motivated this — 100 videos in 25 seconds — any rule is arbitrary,
    so a reproducible one beats a random one.
    """
    caps = settings()["collection"].get("max_posts_per_creator_per_day", 0)
    if isinstance(caps, int):                       # legacy scalar form
        caps = {"youtube": caps, "instagram": caps}
    if posts.empty:
        posts["over_creator_daily_cap"] = False
        return posts

    d = posts.sort_values(["creator_id", "published_at"]).copy()
    d["_day"] = d["published_at"].dt.date
    rank = d.groupby(["creator_id", "_day"]).cumcount()
    limit = d["platform"].map(lambda p: caps.get(p, 0) or 10**9)
    d["over_creator_daily_cap"] = rank >= limit
    return d.drop(columns="_day").reset_index(drop=True)


def build_snapshots(platforms: tuple[str, ...] = ("youtube", "instagram")) -> pd.DataFrame:
    metric_by_platform = settings()["labels"]["primary_metric"]
    rows: list[dict[str, Any]] = []
    for plat in platforms:
        # Dedupe defensively: a duplicated snapshot silently halves a velocity.
        recs = dedupe(iter_bronze("snapshots", platform=plat), SNAPSHOT_KEY)
        metric = (metric_by_platform.get(plat)
                  if isinstance(metric_by_platform, dict) else metric_by_platform)
        for r in recs:
            rows.append({
                "post_id": r.get("post_id"),
                "platform": plat,
                "creator_id": r.get("creator_id"),
                "snapshot_ts": _parse_ts(r.get("snapshot_ts")),
                "published_at": _parse_ts(r.get("published_at")),
                "age_hours": r.get("age_hours"),
                "views": r.get("views"),
                "likes": r.get("likes"),
                "comments": r.get("comments"),
                "primary_metric": metric,
                "primary_value": r.get(metric),
                "at_discovery": bool(r.get("at_discovery", False)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.dropna(subset=["post_id", "snapshot_ts"])

    # Recompute age from timestamps rather than trusting what the collector
    # wrote. If a clock or a publish time was ever corrected, the recomputed
    # value is the honest one, and the two disagreeing is worth knowing about.
    have_both = df["published_at"].notna() & df["snapshot_ts"].notna()
    recomputed = ((df.loc[have_both, "snapshot_ts"] - df.loc[have_both, "published_at"])
                  .dt.total_seconds() / 3600.0)
    df.loc[have_both, "age_hours"] = recomputed

    df = df[df["age_hours"].notna() & (df["age_hours"] >= 0)]
    return df.sort_values(["post_id", "age_hours"]).reset_index(drop=True)


def _iso_duration_sec(iso: str | None) -> float | None:
    """PT1H2M3S -> seconds. YouTube reports duration only in this form."""
    if not iso or not iso.startswith("PT"):
        return None
    import re
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", iso)
    if not m:
        return None
    h, mi, s = (float(x) if x else 0.0 for x in m.groups())
    total = h * 3600 + mi * 60 + s
    return total or None


def build(write: bool = True) -> dict[str, Any]:
    posts = build_posts()
    snaps = build_snapshots()

    out = path_for("silver_dir")
    if write and not posts.empty:
        posts.to_parquet(out / "posts.parquet", index=False)
    if write and not snaps.empty:
        snaps.to_parquet(out / "snapshots.parquet", index=False)

    # Snapshots for posts we never admitted are a real signal, not noise: it
    # means discovery and snapshotting disagree about the panel.
    orphans = (0 if snaps.empty or posts.empty
               else int((~snaps["post_id"].isin(posts["post_id"])).sum()))

    capped = int(posts["over_creator_daily_cap"].sum()) if not posts.empty else 0
    return {
        "posts": len(posts),
        "posts_over_creator_cap": capped,
        "posts_analysable": len(posts) - capped,
        "snapshots": len(snaps),
        "orphan_snapshots": orphans,
        "by_platform": (snaps.groupby("platform").size().to_dict()
                        if not snaps.empty else {}),
        "posts_with_3plus_snapshots": (
            0 if snaps.empty
            else int((snaps.groupby("post_id").size() >= 3).sum())),
        "written_to": str(out) if write else None,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the silver layer from bronze.")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    r = build(write=not args.no_write)
    print("=" * 60)
    print("  SILVER")
    print("=" * 60)
    for k in ("posts", "posts_over_creator_cap", "posts_analysable",
              "snapshots", "posts_with_3plus_snapshots", "orphan_snapshots"):
        print(f"  {k:<28} {r[k]}")
    print(f"  by platform                  {r['by_platform']}")
    print("=" * 60)
    if r["orphan_snapshots"]:
        print(f"  NOTE: {r['orphan_snapshots']} snapshots reference posts not in the "
              f"posts table — discovery and snapshotting disagree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
