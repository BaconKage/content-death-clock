"""Screen candidate Instagram accounts before committing credits to a cohort.

**Why this exists.** Cohort A spent 80 non-renewable credits on five accounts
chosen by eye, and two of them were unusable. Together those two produced 222 of
the 326 posts collected and almost none of the usable curves: each post was seen
roughly once and then vanished. The credits are gone and cannot be re-spent.

**The mechanism that decides everything.** Instagram's profile endpoint returns
only the ~12 most recent posts. A post is therefore observable for as long as it
takes the account to publish 12 more, and no longer:

    hours_in_grid   = GRID_SIZE / posts_per_hour
    snapshots       = hours_in_grid / round_interval_hours

An account posting 40 times a day pushes its own posts off the grid in about
seven hours — one or two snapshots, never a decay curve. An account posting
three times a day keeps them there for four days.

So the only account property that really matters is **posting rate**, and the
useful range is bounded at both ends: too fast and posts fall off before they
decay, too slow and nothing fresh appears inside the cohort window.

**Why screening is cheap.** One profile call costs one credit and returns twelve
posts *with timestamps*. The posting rate is therefore measurable exactly, from a
single call, before any commitment. Ten candidates cost ten credits — about an
eighth of what Cohort A wasted on two bad picks.

Usage::

    python -m cdc.collect.ig_screen --dry-run          # plan, spend nothing
    python -m cdc.collect.ig_screen --handles natgeo nasa wired
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

from cdc.config import ROOT, secrets, settings
from cdc.collect.instagram import InstagramClient

log = logging.getLogger("cdc.ig_screen")

# The profile endpoint's grid size. Measured, not documented: Cohort A never saw
# more than 12 posts from one call.
GRID_SIZE = 12

# Cohort A calibration. A post published late in the window runs out of cohort
# before it runs out of grid, so the raw grid model overestimates. Deflating by
# half reproduces what actually happened on four of the five accounts:
#
#   account            posts/day   observed   predicted
#   instantbollywood        41.9        1.3         1.1
#   nba                     14.8        3.2         3.2
#   fcbarcelona              9.4        4.7         5.1
#   9gag                     6.0        7.5         8.0
#   sarcastic_us             7.8        1.9         6.2   <-- miss
#
# The miss is an account that posts in bursts: its mean rate understates how
# fast a post is actually displaced. The screen is less vulnerable to that than
# this table suggests, because it measures the span of the twelve posts
# currently on the grid — which IS the residence time — rather than a long-run
# average. A bursty account presents a short span and is correctly rejected.
OPTIMISM_FACTOR = 0.5

# A post needs 4 usable intervals to be labelled at all, so 5 snapshots is the
# floor and anything under it is a wasted credit.
MIN_USEFUL_SNAPSHOTS = 5

# Seconds between profile calls. Ten back-to-back requests produced nine HTTP
# 500s from handles that had worked during Cohort A; spacing them costs nothing
# but wall-clock time and credits are the scarce resource here, not seconds.
PAUSE_BETWEEN_CALLS_S = 6.0


def screen_one(client: InstagramClient, handle: str, interval_h: float,
               duration_h: float) -> dict[str, Any]:
    """One profile call. Returns the measured rate and what it implies."""
    out: dict[str, Any] = {"handle": handle.lstrip("@")}
    prof = client.profile(handle)
    if not prof:
        return out | {"status": "not found"}

    posts = prof.get("posts") or []
    out["followers"] = prof.get("follower_count")
    out["posts_returned"] = len(posts)
    if len(posts) < 2:
        return out | {"status": "too few posts to measure a rate"}

    # EXCLUDE PINNED POSTS. A pinned post is held at the top of the grid
    # indefinitely, so it is not evidence about how fast the account publishes —
    # but it stretches the observed span enormously and makes a fast account
    # look slow.
    #
    # This is not hypothetical. Cohort A screened @sarcastic_us at 0.5 posts/day
    # and admitted it. One pinned post from eight days earlier was sitting in
    # its grid: including it the span read 197h (1.1/day), excluding it the span
    # was 5.1h (37.6/day). The account actually turns its grid over twice a day.
    # It went on to produce the worst curves in the cohort — 1.9 snapshots per
    # post across 79 posts — and the credits are unrecoverable.
    pinned = [p for p in posts if p.get("is_pinned")]
    unpinned = [p for p in posts if not p.get("is_pinned")]
    out["pinned_excluded"] = len(pinned)
    if len(unpinned) < 2:
        return out | {"status": f"only {len(unpinned)} unpinned post(s) — "
                                f"cannot measure a rate"}

    times = sorted(datetime.fromisoformat(p["published_at"]) for p in unpinned)
    span_h = (times[-1] - times[0]).total_seconds() / 3600.0
    if span_h <= 0:
        return out | {"status": "all posts share a timestamp — bulk upload"}

    # n posts span n-1 intervals.
    rate_per_h = (len(times) - 1) / span_h
    out["span_hours"] = round(span_h, 1)
    out["posts_per_day"] = round(rate_per_h * 24, 1)

    hours_in_grid = GRID_SIZE / rate_per_h if rate_per_h > 0 else float("inf")
    raw = hours_in_grid / interval_h
    # Cannot exceed the number of rounds in the cohort itself.
    capped = min(raw, duration_h / interval_h)
    out["hours_in_grid"] = round(hours_in_grid, 1)
    out["predicted_snapshots"] = round(capped * OPTIMISM_FACTOR, 1)
    out["expected_new_posts"] = round(rate_per_h * duration_h, 1)

    hidden = sum(1 for p in posts if p.get("counts_hidden"))
    out["counts_hidden"] = hidden
    out["pct_video"] = round(
        100.0 * sum(1 for p in posts if p.get("is_video")) / len(posts))

    # Keep the underlying timestamps. The pinned-post defect above could only
    # be diagnosed by going back to Cohort A's raw records; a screening result
    # that throws away its own evidence makes the next such question cost
    # credits to answer.
    out["observations"] = [
        {"published_at": p["published_at"], "is_pinned": bool(p.get("is_pinned")),
         "likes": p.get("likes"), "is_video": bool(p.get("is_video"))}
        for p in posts
    ]
    out["status"] = "ok"
    out["verdict"], out["why"] = _verdict(out)
    return out


def _verdict(r: dict[str, Any]) -> tuple[str, str]:
    """Plain-language judgement, with the reason attached."""
    if r.get("counts_hidden"):
        return "REJECT", "likes hidden on some posts — reads as instant death"
    snaps = r.get("predicted_snapshots", 0)
    newp = r.get("expected_new_posts", 0)
    if snaps < MIN_USEFUL_SNAPSHOTS:
        return "REJECT", (f"posts too fast ({r['posts_per_day']}/day) — a post "
                          f"falls off the grid in {r['hours_in_grid']}h")
    if newp < 3:
        return "REJECT", (f"posts too slowly ({r['posts_per_day']}/day) — only "
                          f"~{newp:.0f} new posts inside the window")
    if snaps >= 10 and newp >= 6:
        return "STRONG", (f"{snaps:.0f} snapshots/post and ~{newp:.0f} new posts "
                          f"in the window")
    return "OK", f"{snaps:.0f} snapshots/post, ~{newp:.0f} new posts"


def run(handles: list[str], dry_run: bool = False,
        interval_h: float | None = None,
        duration_h: float | None = None,
        key_index: int | None = None) -> dict[str, Any]:
    cfg = settings()["instagram"]
    interval_h = interval_h or float(cfg["cohort_interval_hours"])
    duration_h = duration_h or float(cfg["cohort_duration_hours"])

    handles = [h.lstrip("@") for h in handles]
    print("=" * 78)
    print("  INSTAGRAM ACCOUNT SCREENING")
    print("=" * 78)
    print(f"  candidates          {len(handles)}")
    print(f"  cost                {len(handles)} credit(s), one per account")
    print(f"  cohort assumed      every {interval_h:.0f}h for {duration_h:.0f}h")
    print(f"  grid size           {GRID_SIZE} posts (the binding constraint)")
    print("-" * 78)

    if dry_run:
        print("  DRY RUN — nothing was called and no credit was spent.")
        print("  Candidates that would be screened:")
        for h in handles:
            print(f"    @{h}")
        return {"dry_run": True, "handles": handles, "would_cost": len(handles)}

    keys = list(secrets().require_scrapecreators_keys())
    if key_index is not None:
        # The client spends keys strictly in order and only rolls over when one
        # is empty, so without this a screening run always draws down key 1.
        # Screening is worth isolating on a chosen key: it is exploratory
        # spending, and mixing it into the key a cohort is about to run on
        # makes the remaining budget hard to reason about.
        if not 1 <= key_index <= len(keys):
            raise SystemExit(f"--key {key_index} out of range (1..{len(keys)})")
        keys = [keys[key_index - 1]]
        print(f"  using key #{key_index} only")
    client = InstagramClient(api_keys=keys)
    print(f"  credits on that key  {client.total_remaining}")
    if client.total_remaining < len(handles):
        raise SystemExit(
            f"  REFUSED: {len(handles)} candidates but only "
            f"{client.total_remaining} credits on this key.")
    print("-" * 78)

    rows = []
    try:
        for i, h in enumerate(handles):
            # Pace the calls. The first screening run fired ten profile
            # requests back to back and nine returned HTTP 500 while the same
            # handles had worked fine during Cohort A, which is the signature
            # of upstream rate limiting rather than bad handles.
            if i:
                time.sleep(PAUSE_BETWEEN_CALLS_S)
            try:
                # attempts=1: a retry may cost another credit upstream, and a
                # screening result is not worth paying twice for. A handle that
                # errors is simply re-screened later.
                r = screen_one(client, h, interval_h, duration_h)
            except Exception as exc:                  # one bad handle
                r = {"handle": h, "status": f"error: {exc}"}
            rows.append(r)
            log.info("screened %s: %s", h, r.get("status"))
    finally:
        # ALWAYS persist what was charged, including on an exception. The first
        # version of this tool never called this at all, so ten charges were
        # counted in memory and none reached disk - the ledger claimed a
        # balance that had already been spent.
        client.save_ledgers()
        log.info("credits after screening: %s", client.total_remaining)

    print(f"  {'account':<20} {'/day':>6} {'grid h':>7} {'snaps':>6} "
          f"{'new':>5}  verdict")
    print("-" * 78)
    order = {"STRONG": 0, "OK": 1, "REJECT": 2}
    for r in sorted(rows, key=lambda x: (order.get(x.get("verdict"), 3),
                                         -(x.get("predicted_snapshots") or 0))):
        if r.get("status") != "ok":
            print(f"  @{r['handle']:<19} {'—':>6} {'—':>7} {'—':>6} {'—':>5}  "
                  f"{r.get('status')}")
            continue
        print(f"  @{r['handle']:<19} {r['posts_per_day']:>6.1f} "
              f"{r['hours_in_grid']:>7.1f} {r['predicted_snapshots']:>6.1f} "
              f"{r['expected_new_posts']:>5.0f}  {r['verdict']}: {r['why']}")

    keep = [r for r in rows if r.get("verdict") in ("STRONG", "OK")]
    print("-" * 78)
    print(f"  usable: {len(keep)} of {len(rows)}")
    if keep:
        best = sorted(keep, key=lambda r: -(r["predicted_snapshots"]))
        print("  recommended cohort, best first:")
        for r in best:
            print(f"    @{r['handle']}  ({r['posts_per_day']}/day, "
                  f"~{r['predicted_snapshots']:.0f} snapshots per post)")

    out = {"screened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "interval_hours": interval_h, "duration_hours": duration_h,
           "results": rows}
    path = ROOT / "config" / "ig_screening.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure candidate Instagram accounts' posting rate "
                    "(1 credit each) before committing a cohort to them.")
    ap.add_argument("--handles", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan and the cost without spending anything")
    ap.add_argument("--key", type=int, default=None,
                    help="spend a specific key (1-based) instead of the "
                         "first with credit; keeps exploratory screening off "
                         "the key a cohort will run on")
    ap.add_argument("--interval", type=float, default=None)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
    run(args.handles, dry_run=args.dry_run,
        interval_h=args.interval, duration_h=args.duration,
        key_index=args.key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
