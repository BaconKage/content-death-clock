"""Verify the sample frame against the API and write config/channels.resolved.yaml.

Run this once on Day 1, and again whenever channels are added.

It does three things that matter to the RM paper:

1. **Proves every handle exists.** A typo'd handle silently contributes zero
   posts, quietly shrinking the sample without anyone noticing.
2. **Replaces guessed tiers with measured subscriber counts.** The strata in
   ``channels.yaml`` are the author's guess; the paper must report the real ones.
3. **Records the frame as it was at resolution time.** Subscriber counts move.
   Freezing them with a timestamp is what makes the stratification reproducible.

Cost: 1 quota unit per channel. Resolving 100 channels costs 100 of 10,000 units.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timezone

import yaml

from cdc.config import ROOT, channels, secrets
from cdc.collect.youtube import QuotaMeter, YouTubeClient

log = logging.getLogger("cdc.resolve")

OUT_PATH = ROOT / "config" / "channels.resolved.yaml"

TIER_BOUNDS = [(10_000, "micro"), (500_000, "mid")]   # else "large"


def tier_for(subscriber_count: int | None) -> str:
    if subscriber_count is None:
        return "unknown"          # hidden count — reported, never guessed
    for bound, name in TIER_BOUNDS:
        if subscriber_count < bound:
            return name
    return "large"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resolve channel handles to ids and strata.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    frame = channels().get("channels") or []
    if not frame:
        log.error("config/channels.yaml has no channels")
        return 1

    api_key = "DRY-RUN-NO-KEY" if args.dry_run else secrets().require_youtube()
    # One unit per channel, plus headroom.
    meter = QuotaMeter(budget=len(frame) + 20)
    client = YouTubeClient(api_key=api_key, meter=meter, dry_run=args.dry_run)

    resolved, failed = [], []
    for entry in frame:
        handle = entry["handle"]
        try:
            info = client.resolve_channel(handle)
        except Exception as exc:
            log.error("%s -> ERROR %s", handle, exc)
            failed.append({"handle": handle, "error": str(exc)})
            continue
        if info is None:
            log.error("%s -> NOT FOUND (check the handle spelling)", handle)
            failed.append({"handle": handle, "error": "not found"})
            continue

        measured = tier_for(info["subscriber_count"])
        if measured != entry.get("tier"):
            log.info("%s tier %s -> %s (%s subs)", handle, entry.get("tier"),
                     measured, info["subscriber_count"])
        info["category"] = entry.get("category")
        info["tier"] = measured
        info["tier_declared"] = entry.get("tier")
        resolved.append(info)
        log.info("%-28s %-24s %-6s %s subs", handle, info["channel_id"],
                 measured, info["subscriber_count"])

    doc = {
        "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "config/channels.yaml",
        "tier_bounds": {"micro": "<10000", "mid": "10000-500000", "large": ">500000"},
        "channels": resolved,
        "failed": failed,
    }

    print("\n" + "=" * 60)
    print(f"resolved : {len(resolved)}/{len(frame)}")
    print(f"failed   : {len(failed)}")
    print(f"by tier  : {dict(Counter(c['tier'] for c in resolved))}")
    print(f"category : {dict(Counter(c['category'] for c in resolved))}")
    print(f"quota    : {meter.summary()['units_spent']} units")

    # --- sample-frame health checks. These are research-validity checks, not
    # --- code checks, so they warn loudly rather than failing the run.
    tiers = Counter(c["tier"] for c in resolved)
    for t in ("micro", "mid", "large"):
        if tiers.get(t, 0) < 10:
            print(f"WARNING: tier '{t}' has only {tiers.get(t, 0)} channels. "
                  f"Below ~10 the stratum cannot support a subgroup claim.")
    if len(resolved) < 60:
        print(f"WARNING: {len(resolved)} channels resolved; target is 60-100.")
    print("=" * 60)

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}")
    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(main())
