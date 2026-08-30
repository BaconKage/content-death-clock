"""One Instagram collection cycle.

Structurally different from the YouTube runner, because the cost model is
different. On YouTube a quota unit is free and the unit of work is the post. Here
every call costs a real credit, and one ``/v1/instagram/profile`` call returns an
account's ~12 most recent posts *with* their engagement counts — so the unit of
work is the **account**, and discovery and snapshotting happen in the same call.

That has a pleasant consequence: snapshotting an account's whole recent grid
costs exactly the same as snapshotting one post of it.

Usage::

    python -m cdc.collect.ig_runner --dry-run
    python -m cdc.collect.ig_runner
    python -m cdc.collect.ig_runner --force     # ignore the cadence gate
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from cdc.config import ROOT, channels, secrets, settings
from cdc.collect.instagram import CreditLedger, CreditsExhausted, InstagramClient
from cdc.storage.bronze import BronzeWriter, cycle_id, iter_bronze

log = logging.getLogger("cdc.ig_runner")


def _accounts() -> list[dict[str, Any]]:
    ig = (channels().get("instagram") or {})
    accts = ig.get("accounts") or []
    return [a if isinstance(a, dict) else {"handle": a} for a in accts]


def last_cycle_at(bronze_root=None) -> datetime | None:
    """Most recent Instagram snapshot time, derived from bronze."""
    latest = None
    for rec in iter_bronze("snapshots", platform="instagram", root=bronze_root):
        ts = rec.get("snapshot_ts")
        if not ts:
            continue
        try:
            d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        if latest is None or d > latest:
            latest = d
    return latest


def run_cycle(dry_run: bool = False, force: bool = False, client=None,
              accounts=None, bronze_root=None, now=None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    cid = cycle_id(now)
    cfg = settings()["instagram"]
    lab = settings()["labels"]
    max_age_h = float(settings()["collection"]["max_track_hours"])

    report: dict[str, Any] = {
        "platform": "instagram", "cycle_id": cid,
        "started_at": now.isoformat(timespec="seconds"), "dry_run": dry_run,
        "accounts_called": 0, "posts_seen": 0, "snapshots": 0,
        "new_posts": 0, "skipped_hidden_counts": 0, "errors": [],
    }

    if not cfg.get("enabled"):
        report["skipped"] = "instagram.enabled is false"
        return report

    accts = accounts if accounts is not None else _accounts()
    if not accts:
        report["skipped"] = ("no accounts configured — add them under "
                             "`instagram.accounts` in config/channels.yaml")
        return report

    # --- cohort gate ---------------------------------------------------
    # Instagram runs as one bounded cohort, not a continuous panel, because
    # every call costs a real credit. Three separate conditions must hold
    # before we are allowed to spend anything.
    start_raw = cfg.get("cohort_start_utc")
    if not start_raw:
        report["skipped"] = ("cohort not started — set instagram.cohort_start_utc "
                             "in settings.yaml. This is a one-shot budget: once "
                             "started, it runs to completion and the credits are gone.")
        return report

    start = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    interval = float(cfg["cohort_interval_hours"])
    duration = float(cfg["cohort_duration_hours"])
    elapsed = (now - start).total_seconds() / 3600.0

    if elapsed < 0:
        report["skipped"] = f"cohort starts at {start.isoformat(timespec='seconds')}"
        return report
    # End-exclusive. Rounds land at 0, 3, ... 45h for a 48h/3h cohort: that is
    # 16 rounds, not 17. Inclusive would add a whole extra round and overspend
    # the budget by one round's worth of credits.
    if elapsed >= duration:
        report["skipped"] = (f"cohort complete ({elapsed:.1f}h elapsed of "
                             f"{duration:.0f}h). Credits preserved.")
        return report

    if not force:
        # Only fire on a scheduled round, and only once per round. The hourly
        # CI job invokes this every hour; most invocations must do nothing.
        last = last_cycle_at(bronze_root)
        if last is not None:
            since_last = (now - last).total_seconds() / 3600.0
            # Tolerance: CI can be delayed, so accept anything close enough to
            # the interval rather than drifting a round later every time.
            if since_last < interval - 0.5:
                report["skipped"] = (f"only {since_last:.2f}h since last round "
                                     f"(interval {interval}h)")
                return report
        report["cohort_round"] = int(elapsed // interval) + 1
        report["cohort_elapsed_hours"] = round(elapsed, 2)

    if client is None:
        api_key = ("DRY-RUN-NO-KEY" if dry_run
                   else secrets().require_scrapecreators())
        client = InstagramClient(api_key=api_key, dry_run=dry_run)

    known = {rec.get("post_id") for rec in
             iter_bronze("posts", platform="instagram", root=bronze_root)}

    posts_writer = BronzeWriter("instagram", kind="posts", cid=cid, root=bronze_root)
    snaps_writer = BronzeWriter("instagram", kind="snapshots", cid=cid, root=bronze_root)

    # Never call more accounts than the cohort was sized for, and never more
    # than the remaining credits can pay for.
    budget = min(int(cfg["max_accounts"]), int(cfg["max_calls_per_cycle"]),
                 client.ledger.remaining)
    if budget <= 0:
        report["skipped"] = "no credits remaining"
        report["credits"] = client.ledger.summary()
        return report
    try:
        for acct in accts[:budget]:
            handle = acct["handle"]
            try:
                prof = client.profile(handle)
            except CreditsExhausted:
                raise
            except Exception as exc:
                log.warning("profile failed for %s: %s", handle, exc)
                report["errors"].append(f"profile:{handle}:{exc}")
                continue
            report["accounts_called"] += 1
            if not prof:
                continue

            for post in prof["posts"]:
                report["posts_seen"] += 1
                pub = datetime.fromisoformat(post["published_at"])
                age_h = (now - pub).total_seconds() / 3600.0
                if age_h > max_age_h:
                    continue                      # outside the tracking window
                if post["counts_hidden"] and lab.get("exclude_hidden_counts", True):
                    report["skipped_hidden_counts"] += 1
                    continue

                if post["post_id"] not in known:
                    rec = dict(post)
                    rec["stratum_tier"] = acct.get("tier")
                    rec["stratum_category"] = acct.get("category")
                    rec["follower_count"] = prof["follower_count"]
                    rec["discovered_at"] = now.isoformat(timespec="seconds")
                    posts_writer.add(rec)
                    known.add(post["post_id"])
                    report["new_posts"] += 1

                # The profile call already returned live counts — snapshotting
                # the whole grid is free once the call is paid for.
                snaps_writer.add({
                    "post_id": post["post_id"],
                    "creator_id": post["creator_id"],
                    "creator_handle": handle,
                    "snapshot_ts": now.isoformat(timespec="seconds"),
                    "published_at": post["published_at"],
                    "age_hours": round(age_h, 4),
                    "views": post["views"],
                    "likes": post["likes"],
                    "comments": post["comments"],
                    "media_type": post["media_type"],
                })
                report["snapshots"] += 1

    except CreditsExhausted as exc:
        log.error("CREDITS: %s", exc)
        report["errors"].append(f"credits:{exc}")
        report["credits_exhausted"] = True

    finally:
        if not dry_run:
            posts_writer.commit()
            snaps_writer.commit()
            client.ledger.save()
        report["credits"] = client.ledger.summary()
        report["files"] = [str(p) for p in (posts_writer.committed_path,
                                            snaps_writer.committed_path) if p]

    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one Instagram collection cycle.")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan the cycle and report credit cost without spending")
    ap.add_argument("--force", action="store_true",
                    help="ignore the cadence gate and run now")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    report = run_cycle(dry_run=args.dry_run, force=args.force)
    print(json.dumps(report, indent=2))

    if not args.dry_run and not report.get("skipped"):
        logs = ROOT / "data" / "bronze" / "_cycles"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f"ig-{report['cycle_id']}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

    return 1 if report.get("credits_exhausted") else 0


if __name__ == "__main__":
    sys.exit(main())
