"""Build the sampling frame by searching YouTube, instead of by hand.

The frame started as a convenience sample — channels the authors happened to
follow — and came out 30 large / 2 mid / 1 micro. That cannot support the
creator-size comparison in H3, and it makes "large YouTube channels" the only
population the paper can claim.

**Method.** For each query in ``config/discovery_queries.yaml`` we search
*recent videos* (``search.list``, ordered by date), collect the channels behind
them, resolve those channels in batches, and keep the ones that fall in the
under-represented subscriber tiers and upload often enough to contribute posts
during the collection window.

Searching recent *videos* rather than *channels* is deliberate. Channel search
ranks by relevance, which is a proxy for popularity, so it essentially cannot
surface a sub-10k channel — precisely the stratum we are missing. Video search
ordered by date surfaces creators of every size, and it inherently selects for
channels that actually upload.

**This is a sampling decision and must be reported.** The resulting frame is not
a random sample of YouTube: it is the set of channels reachable by a fixed,
recorded list of queries. That is a weaker claim than random sampling but a
stronger and far more reproducible one than "channels we happened to follow" —
another researcher can re-run these exact queries. The query list is committed
alongside the results for that reason.

Cost: 100 units per query (search is the one expensive endpoint), plus ~1 unit
per 50 channels resolved, plus 1 unit per candidate for the upload-frequency
check. Roughly 2,500 units of the 10,000/day budget for a full run.

Usage::

    python -m cdc.collect.discover_channels --dry-run
    python -m cdc.collect.discover_channels
    python -m cdc.collect.discover_channels --apply    # write into channels.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

import yaml

from cdc.config import ROOT, channels, secrets
from cdc.collect.youtube import QuotaExceeded, QuotaMeter, YouTubeClient

log = logging.getLogger("cdc.discover")

QUERIES_PATH = ROOT / "config" / "discovery_queries.yaml"
CANDIDATES_PATH = ROOT / "config" / "channel_candidates.yaml"
RESOLVED_PATH = ROOT / "config" / "channel_candidates_all.yaml"

TIER_BOUNDS = [(10_000, "micro"), (500_000, "mid")]


def tier_for(subs: int | None) -> str:
    if subs is None:
        return "unknown"
    for bound, name in TIER_BOUNDS:
        if subs < bound:
            return name
    return "large"


def _existing_channel_ids() -> set[str]:
    path = ROOT / "config" / "channels.resolved.yaml"
    if not path.exists():
        return set()
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {c["channel_id"] for c in (doc.get("channels") or []) if c.get("channel_id")}


def upload_frequency(client: YouTubeClient, uploads_playlist: str,
                     lookback_days: int = 30) -> tuple[float, int]:
    """(uploads per week, uploads seen) over the recent past. 1 unit."""
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    try:
        uploads = client.recent_uploads(uploads_playlist, since=since, max_pages=1)
    except QuotaExceeded:
        raise
    except Exception as exc:
        log.debug("frequency check failed: %s", exc)
        return 0.0, 0
    if not uploads:
        return 0.0, 0
    return len(uploads) / (lookback_days / 7.0), len(uploads)


def select_candidates(resolved: list[dict[str, Any]], want_tiers: tuple[str, ...],
                      min_subscribers: dict[str, int]) -> list[dict[str, Any]]:
    """Apply a quality floor, then spread the picks across each tier.

    Two failure modes this exists to prevent, both found the hard way:

    * **Floor.** A naive tier filter admits 0- and 2-subscriber channels. Their
      videos get essentially no views, so every one of them would be dropped by
      the `no_positive_velocity` exclusion. They are technically "micro" and
      practically worthless.
    * **Spread.** Sorting by subscriber count and taking the first N clusters
      every pick at the very bottom of the band — 15 "mid" channels that are all
      10k-16k does not represent a 10k-500k stratum. We sort and then sample at
      even intervals so the picks span the range.
    """
    out: list[dict[str, Any]] = []
    for tier in want_tiers:
        floor = min_subscribers.get(tier, 0)
        pool = [c for c in resolved
                if c["tier"] == tier
                and (c["subscriber_count"] or 0) >= floor
                and (c.get("video_count") or 0) >= 10]   # not a brand-new channel
        pool.sort(key=lambda c: c["subscriber_count"] or 0)
        out.extend(pool)
    return out


def _spread(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Evenly spaced picks across a sorted pool, so the span is represented."""
    if len(pool) <= n:
        return pool
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]


def discover(dry_run: bool = False, want_tiers: tuple[str, ...] = ("micro", "mid"),
             min_uploads_per_week: float = 1.0, per_tier_target: int = 15,
             min_subscribers: dict[str, int] | None = None,
             from_cache: bool = False, quota_budget: int = 6000) -> dict[str, Any]:
    # Quality floors. A "micro" channel with 3 subscribers is not a small
    # creator, it is an empty channel.
    min_subscribers = min_subscribers or {"micro": 1_000, "mid": 10_000}
    cfg = yaml.safe_load(QUERIES_PATH.read_text(encoding="utf-8"))
    queries = cfg["queries"]
    lookback_days = int(cfg.get("search_lookback_days", 7))

    api_key = "DRY-RUN-NO-KEY" if dry_run else secrets().require_youtube()
    meter = QuotaMeter(budget=quota_budget)
    client = YouTubeClient(api_key=api_key, meter=meter, dry_run=dry_run)

    known = _existing_channel_ids()
    seen: dict[str, dict[str, Any]] = {}
    published_after = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Re-selecting from cache costs no search units. Search is 100/query and the
    # selection rules get tuned more than once, so never re-search to re-select.
    if from_cache and RESOLVED_PATH.exists():
        cached = yaml.safe_load(RESOLVED_PATH.read_text(encoding="utf-8"))
        resolved = cached["channels"]
        log.info("loaded %d cached candidates (0 search units)", len(resolved))
        return _select_and_check(client, resolved, want_tiers, min_subscribers,
                                 min_uploads_per_week, per_tier_target,
                                 meter, queries, len(resolved))

    # ---------------------------------------------------------- 1. search
    for q in queries:
        query, category = q["q"], q["category"]
        try:
            hits = client.search_recent_videos(
                query, published_after=published_after,
                region_code=q.get("region_code"))
        except QuotaExceeded as exc:
            log.warning("stopping search early: %s", exc)
            break
        except Exception as exc:
            log.warning("search failed for %r: %s", query, exc)
            continue
        new = 0
        for h in hits:
            cid = h["channel_id"]
            if cid in known or cid in seen:
                continue
            seen[cid] = {"channel_id": cid, "category": category,
                         "found_by_query": query}
            new += 1
        log.info("%-45s %3d hits, %2d new channels", query[:45], len(hits), new)

    log.info("%d unique new channels from %d queries (%d units)",
             len(seen), len(queries), meter.spent)

    # ------------------------------------------- 2. resolve, batched 50/unit
    resolved = []
    try:
        for meta in client.channels_batch(list(seen)):
            base = seen.get(meta["channel_id"], {})
            meta["category"] = base.get("category")
            meta["found_by_query"] = base.get("found_by_query")
            meta["tier"] = tier_for(meta["subscriber_count"])
            resolved.append(meta)
    except QuotaExceeded as exc:
        log.warning("resolve stopped early: %s", exc)

    RESOLVED_PATH.write_text(
        yaml.safe_dump({"resolved_at": datetime.now(timezone.utc).isoformat(),
                        "channels": resolved}, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    return _select_and_check(client, resolved, want_tiers, min_subscribers,
                             min_uploads_per_week, per_tier_target, meter,
                             queries, len(seen))


def _select_and_check(client, resolved, want_tiers, min_subscribers,
                      min_uploads_per_week, per_tier_target, meter,
                      queries, n_seen) -> dict[str, Any]:
    """Pick a frame that SPANS each tier rather than clustering at its floor.

    Frequency is checked on a wide spread of candidates first, and only then is
    the passing set thinned to the target. Checking in ascending order and
    stopping at the target would refill the same bottom-of-band cluster the
    quality floor was added to fix - the floor moves the cluster, it does not
    remove it.
    """
    in_tier = select_candidates(resolved, want_tiers, min_subscribers)
    log.info("%d resolved, %d pass the quality floor", len(resolved), len(in_tier))

    kept: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for tier in want_tiers:
        pool = [c for c in in_tier if c["tier"] == tier]
        # Check ~3x the target, spread across the whole band.
        passers = []
        for c in _spread(pool, per_tier_target * 3):
            try:
                per_week, n = upload_frequency(client, c["uploads_playlist"])
            except QuotaExceeded:
                log.warning("frequency checks stopped: quota")
                break
            c["uploads_per_week"] = round(per_week, 2)
            c["uploads_last_30d"] = n
            if per_week >= min_uploads_per_week:
                passers.append(c)
        passers.sort(key=lambda c: c["subscriber_count"] or 0)
        chosen = _spread(passers, per_tier_target)
        for c in chosen:
            log.info("  KEEP %-30s %-6s %9s subs  %.1f uploads/wk",
                     (c.get("handle") or c["title"])[:30], c["tier"],
                     c["subscriber_count"], c["uploads_per_week"])
        kept.extend(chosen)
        counts[tier] = len(chosen)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": ("search.list on recent videos ordered by date, over the "
                   "queries in config/discovery_queries.yaml; then filtered to "
                   "target subscriber tiers with a quality floor and a minimum "
                   "upload rate, then sampled evenly across each tier"),
        "queries_used": [q["q"] for q in queries],
        "min_uploads_per_week": min_uploads_per_week,
        "channels_seen": n_seen,
        "channels_resolved": len(resolved),
        "quota_spent": meter.summary(),
        "by_tier": dict(counts),
        "channels": kept,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Discover micro/mid channels for the frame.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true",
                    help="append the kept channels to config/channels.yaml")
    ap.add_argument("--per-tier", type=int, default=15)
    ap.add_argument("--min-uploads-per-week", type=float, default=1.0)
    ap.add_argument("--from-cache", action="store_true",
                    help="re-select from cached candidates; spends no search units")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    res = discover(dry_run=args.dry_run, per_tier_target=args.per_tier,
                   min_uploads_per_week=args.min_uploads_per_week,
                   from_cache=args.from_cache)
    print("\n" + "=" * 70)
    print(f"  channels seen      : {res['channels_seen']}")
    print(f"  resolved           : {res['channels_resolved']}")
    print(f"  kept               : {len(res['channels'])}  {res['by_tier']}")
    print(f"  quota spent        : {res['quota_spent']['units_spent']} units")
    print("  subscriber range of picks (must SPAN the tier, not cluster):")
    for tier in ("micro", "mid"):
        subs = sorted(c["subscriber_count"] for c in res["channels"]
                      if c["tier"] == tier)
        if subs:
            print(f"    {tier:<6} {subs[0]:>8,} .. {subs[-1]:>8,}  (n={len(subs)})")
    print("=" * 70)

    if args.dry_run:
        print("[dry-run] nothing written")
        return 0

    CANDIDATES_PATH.write_text(
        yaml.safe_dump(res, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {CANDIDATES_PATH.relative_to(ROOT)}")

    if args.apply and res["channels"]:
        _append_to_frame(res["channels"])
        print("appended to config/channels.yaml — now run:")
        print("  python -m cdc.collect.resolve_channels")
    return 0


def _append_to_frame(kept: list[dict[str, Any]]) -> None:
    path = ROOT / "config" / "channels.yaml"
    s = path.read_text(encoding="utf-8")
    existing = {c["handle"].lstrip("@").lower() for c in (channels().get("channels") or [])}

    lines = ["", "  # ---------------- discovered via search (see "
             "config/channel_candidates.yaml) ----------------",
             "  # Selected by: recent-video search on recorded queries, then filtered to",
             "  # micro/mid subscriber tiers AND >= 1 upload/week. Frequency matters more",
             "  # than size here: a channel that posts monthly contributes nothing to a",
             "  # three-week window regardless of how well it fits the stratum."]
    added = 0
    for c in kept:
        h = (c.get("handle") or "").lstrip("@")
        if not h or h.lower() in existing:
            continue
        lines.append(f'  - {{handle: "@{h}", category: {c["category"]}, '
                     f'tier: {c["tier"]}}}   # {c["subscriber_count"]} subs, '
                     f'{c["uploads_per_week"]}/wk')
        existing.add(h.lower())
        added += 1

    anchor = "\ninstagram:"
    assert anchor in s
    path.write_text(s.replace(anchor, "\n".join(lines) + "\n" + anchor), encoding="utf-8")
    log.info("appended %d channels to config/channels.yaml", added)


if __name__ == "__main__":
    sys.exit(main())
