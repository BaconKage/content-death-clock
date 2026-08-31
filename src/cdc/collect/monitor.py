"""Collection health monitor.

Answers the one question that matters during the collection weeks: *is the data
actually arriving, and is it arriving on time?* Run it daily. It is also wired
into CI so a gap turns the Actions tab red instead of failing silently.

The completeness table it prints goes straight into the BDA report as evidence
that the pipeline ran as designed, and into the RM paper's limitations section
if it did not.

Usage::

    python -m cdc.collect.monitor
    python -m cdc.collect.monitor --fail-on-gap      # non-zero exit for CI
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from cdc.config import settings
from cdc.collect.panel import Panel
from cdc.storage.bronze import iter_bronze

# A cycle is "missed" if no snapshot record exists for an hour in which at
# least one post was due. Two consecutive misses is a real outage.
MAX_TOLERATED_CONSECUTIVE_MISSES = 2


def _parse(ts):
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def report(now: datetime | None = None, recent_hours: float | None = None) -> dict:
    """Collection health.

    `recent_hours` scopes the *alarm* to a trailing window. Without it, a single
    historical outage fails every subsequent CI run forever, and a permanently
    red alarm is worse than no alarm because people stop reading it. The
    full-history figures are still reported for the paper; only the pass/fail
    judgement is windowed.
    """
    now = now or datetime.now(timezone.utc)
    horizon = (now - timedelta(hours=recent_hours)) if recent_hours else None
    cfg = settings()["collection"]
    schedule = [float(h) for h in cfg["snapshot_schedule_hours"]]
    tol = float(cfg["snapshot_tolerance_hours"])

    panel = Panel.from_bronze(platform="youtube")

    # --- per-post: which scheduled marks were actually hit, and how late
    ages_by_post: dict[str, list[float]] = defaultdict(list)
    cycle_hours: set[str] = set()
    for s in iter_bronze("snapshots", platform="youtube"):
        pid = s.get("post_id")
        age = s.get("age_hours")
        if pid is not None and age is not None:
            ages_by_post[pid].append(float(age))
        ts = _parse(s.get("snapshot_ts"))
        if ts:
            cycle_hours.add(ts.strftime("%Y-%m-%dT%H"))

    hit_counts = Counter()
    lateness: list[float] = []
    per_post_completeness: list[float] = []

    for pid, post in panel.posts.items():
        # Posts that lived through an outage carry its damage permanently. Judge
        # current health on posts published inside the window.
        if horizon is not None and post.published_at < horizon:
            continue
        age_now = post.age_hours(now)
        expected = [m for m in schedule if m <= age_now]
        if not expected:
            continue
        observed = sorted(ages_by_post.get(pid, []))
        hits = 0
        for m in expected:
            near = [a for a in observed if abs(a - m) <= tol]
            if near:
                hits += 1
                hit_counts[m] += 1
                lateness.append(min(near, key=lambda a: abs(a - m)) - m)
            else:
                # A mark can still be covered late — count it, but as late.
                late = [a for a in observed if m < a <= m + 6 * tol]
                if late:
                    hits += 1
                    hit_counts[m] += 1
                    lateness.append(min(late) - m)
        per_post_completeness.append(hits / len(expected))

    # --- cycle continuity: consecutive hours with no snapshot written at all
    if horizon is not None:
        cutoff = horizon.strftime("%Y-%m-%dT%H")
        cycle_hours = {h for h in cycle_hours if h >= cutoff}
    hours_sorted = sorted(cycle_hours)
    gaps: list[tuple[str, int]] = []
    if hours_sorted:
        fmt = "%Y-%m-%dT%H"
        cur = datetime.strptime(hours_sorted[0], fmt).replace(tzinfo=timezone.utc)
        end = datetime.strptime(hours_sorted[-1], fmt).replace(tzinfo=timezone.utc)
        run = 0
        while cur <= end:
            if cur.strftime(fmt) in cycle_hours:
                if run > MAX_TOLERATED_CONSECUTIVE_MISSES:
                    gaps.append(((cur - timedelta(hours=run)).strftime(fmt), run))
                run = 0
            else:
                run += 1
            cur += timedelta(hours=1)
        if run > MAX_TOLERATED_CONSECUTIVE_MISSES:
            gaps.append(((cur - timedelta(hours=run)).strftime(fmt), run))

    mean_completeness = (sum(per_post_completeness) / len(per_post_completeness)
                         if per_post_completeness else 0.0)
    below_80 = sum(1 for c in per_post_completeness if c < 0.80)

    return {
        "now": now.isoformat(timespec="seconds"),
        "panel": panel.stats(now),
        "posts_evaluated": len(per_post_completeness),
        "mean_completeness": mean_completeness,
        "posts_below_80pct": below_80,
        "mean_lateness_hours": (sum(lateness) / len(lateness)) if lateness else 0.0,
        "max_lateness_hours": max(lateness) if lateness else 0.0,
        "hit_counts_by_mark": {str(k): v for k, v in sorted(hit_counts.items())},
        "cycle_hours_observed": len(cycle_hours),
        "gaps": gaps,
        "recent_hours": recent_hours,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Report collection completeness.")
    ap.add_argument("--fail-on-gap", action="store_true",
                    help="exit non-zero if a RECENT outage or low completeness is found")
    ap.add_argument("--recent-hours", type=float, default=None,
                    help="scope the pass/fail judgement to a trailing window "
                         "(CI uses this so an old outage does not fail runs forever)")
    args = ap.parse_args(argv)

    # In CI, judge recent health by default; a human running it wants everything.
    recent = args.recent_hours if args.recent_hours is not None else (
        12.0 if args.fail_on_gap else None)
    r = report(recent_hours=recent)
    p = r["panel"]

    print("=" * 66)
    print("  CONTENT DEATH CLOCK - collection health")
    print("=" * 66)
    print(f"  as of                {r['now']}")
    if r.get("recent_hours"):
        print(f"  judging window       last {r['recent_hours']:.0f}h "
              f"(older outages reported, not alarmed on)")
    print(f"  posts tracked        {p['posts_total']}  (active {p['posts_active']})")
    print(f"  snapshots collected  {p['snapshots_total']}"
          f"  (mean {p['snapshots_per_post_mean']:.1f}/post)")
    print(f"  cycle-hours observed {r['cycle_hours_observed']}")
    print("-" * 66)
    print(f"  mean completeness    {r['mean_completeness']:.1%}"
          f"   ({r['posts_evaluated']} posts evaluated)")
    print(f"  posts below 80%      {r['posts_below_80pct']}")
    print(f"  mean lateness        {r['mean_lateness_hours']:+.2f} h"
          f"   (worst {r['max_lateness_hours']:+.2f} h)")
    print("-" * 66)
    print("  coverage by scheduled mark (hours since publish):")
    for mark, n in r["hit_counts_by_mark"].items():
        bar = "#" * min(40, n)
        print(f"    t+{float(mark):>6.1f}h  {n:>5}  {bar}")

    problems = []
    if p["posts_total"] == 0:
        problems.append("no posts tracked at all - discovery is not working")
    elif p["snapshots_total"] == 0:
        problems.append("posts tracked but zero snapshots - snapshotting is not working")
    if r["gaps"]:
        print("-" * 66)
        print("  OUTAGES:")
        for start, length in r["gaps"]:
            print(f"    {length} consecutive hours with no data from {start}")
        problems.append(f"{len(r['gaps'])} outage(s)")
    if r["posts_evaluated"] and r["mean_completeness"] < 0.80:
        problems.append(f"mean completeness {r['mean_completeness']:.1%} < 80%")

    print("=" * 66)
    if problems:
        print("  STATUS: PROBLEMS - " + "; ".join(problems))
    else:
        print("  STATUS: healthy")
    print("=" * 66)

    return 1 if (problems and args.fail_on_gap) else 0


if __name__ == "__main__":
    sys.exit(main())
