"""One collection cycle: discover new posts, snapshot the ones that are due.

This is what the hourly scheduler invokes. It is designed to be safe to run
repeatedly and safe to interrupt:

* ``--dry-run`` plans the whole cycle and prints the exact quota cost without
  spending a single unit. Run this before every schedule change.
* Records are buffered and committed atomically at the end, so an interrupted
  cycle leaves no partial file and the next cycle simply redoes it.
* Quota is metered against a per-cycle guard rail; hitting it aborts loudly
  rather than silently eating the daily budget.

Usage::

    python -m cdc.collect.runner --dry-run
    python -m cdc.collect.runner
    python -m cdc.collect.runner --no-discover      # snapshot only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from cdc.config import ROOT, channels, secrets, settings
from cdc.collect.panel import Panel, discovery_window
from cdc.collect.youtube import QuotaExceeded, QuotaMeter, YouTubeClient
from cdc.storage.bronze import BronzeWriter, cycle_id

log = logging.getLogger("cdc.runner")


def _resolved_channels() -> list[dict[str, Any]]:
    """Channels with a resolved uploads playlist, from config/channels.resolved.yaml."""
    import yaml
    path = ROOT / "config" / "channels.resolved.yaml"
    if not path.exists():
        raise SystemExit(
            "config/channels.resolved.yaml not found.\n"
            "Run:  python -m cdc.collect.resolve_channels\n"
            "It verifies every handle against the API and writes the resolved frame."
        )
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [c for c in (data.get("channels") or []) if c.get("uploads_playlist")]


def run_cycle(dry_run: bool = False, discover: bool = True,
              snapshot: bool = True, client=None, chans=None,
              bronze_root=None, now=None) -> dict[str, Any]:
    # client/chans/bronze_root/now are injection points for the offline
    # integration test. Production always leaves them None.
    now = now or datetime.now(timezone.utc)
    cid = cycle_id(now)
    cfg = settings()
    meter = QuotaMeter(budget=int(cfg["youtube"]["max_quota_per_cycle"]))

    log.info("cycle %s starting (dry_run=%s)", cid, dry_run)

    panel = Panel.from_bronze(platform="youtube", root=bronze_root)
    before = panel.stats(now)
    log.info("panel before: %s", before)

    if client is None:
        # A dry run must work with no key at all, so a new team member can
        # inspect the planned cycle before anyone hands them credentials.
        if dry_run:
            api_key = secrets().youtube_api_key or "DRY-RUN-NO-KEY"
        else:
            api_key = secrets().require_youtube()
        client = YouTubeClient(api_key=api_key, meter=meter, dry_run=dry_run)
    else:
        meter = client.meter

    report: dict[str, Any] = {
        "cycle_id": cid,
        "started_at": now.isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "panel_before": before,
        "discovered": 0,
        "snapshotted": 0,
        "errors": [],
    }

    chans = chans if chans is not None else _resolved_channels()
    posts_writer = BronzeWriter("youtube", kind="posts", cid=cid, root=bronze_root)
    snaps_writer = BronzeWriter("youtube", kind="snapshots", cid=cid, root=bronze_root)

    try:
        # ------------------------------------------------------- 1. discovery
        if discover:
            since = discovery_window(now)
            new_ids: list[str] = []
            for ch in chans:
                try:
                    uploads = client.recent_uploads(ch["uploads_playlist"], since=since)
                except QuotaExceeded:
                    raise
                except Exception as exc:                      # one bad channel
                    log.warning("discovery failed for %s: %s", ch.get("handle"), exc)
                    report["errors"].append(f"discovery:{ch.get('handle')}:{exc}")
                    continue
                for u in uploads:
                    if u["video_id"] not in panel.posts:
                        new_ids.append(u["video_id"])

            # Fetch full metadata for genuinely new videos, then admit them.
            if new_ids:
                for v in client.video_stats(new_ids):
                    rec = dict(v)
                    ch_meta = next((c for c in chans
                                    if c.get("channel_id") == v.get("creator_id")), {})
                    rec["stratum_tier"] = ch_meta.get("tier")
                    rec["stratum_category"] = ch_meta.get("category")
                    rec["discovered_at"] = now.isoformat(timespec="seconds")
                    posts_writer.add(rec)
                report["discovered"] = len(posts_writer)

        # ------------------------------------------------------- 2. snapshots
        if snapshot:
            due = panel.due(now)
            if due:
                stats = client.video_stats([p.post_id for p in due])
                for s in stats:
                    p = panel.posts.get(s["post_id"])
                    snaps_writer.add({
                        "post_id": s["post_id"],
                        "creator_id": s.get("creator_id"),
                        "snapshot_ts": now.isoformat(timespec="seconds"),
                        "published_at": s.get("published_at"),
                        "age_hours": round(p.age_hours(now), 4) if p else None,
                        "views": s.get("views"),
                        "likes": s.get("likes"),
                        "comments": s.get("comments"),
                    })
                report["snapshotted"] = len(snaps_writer)
            report["due_count"] = len(due)

    except QuotaExceeded as exc:
        # Commit what we have — partial data beats no data, and the next cycle
        # catches up. Then fail loudly so CI marks the run red.
        log.error("QUOTA: %s", exc)
        report["errors"].append(f"quota:{exc}")
        report["quota_exceeded"] = True

    finally:
        if not dry_run:
            posts_writer.commit()
            snaps_writer.commit()
        report["quota"] = meter.summary()
        report["files"] = [str(p) for p in (posts_writer.committed_path,
                                            snaps_writer.committed_path) if p]

    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one Content Death Clock collection cycle.")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan the cycle and report quota cost without spending it")
    ap.add_argument("--no-discover", action="store_true", help="skip discovery")
    ap.add_argument("--no-snapshot", action="store_true", help="skip snapshotting")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    report = run_cycle(dry_run=args.dry_run,
                       discover=not args.no_discover,
                       snapshot=not args.no_snapshot)
    print(json.dumps(report, indent=2))

    # Write a machine-readable cycle log for the completeness monitor.
    if not args.dry_run:
        logs = ROOT / "data" / "bronze" / "_cycles"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f"{report['cycle_id']}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

    return 1 if report.get("quota_exceeded") else 0


if __name__ == "__main__":
    sys.exit(main())
