"""End-to-end pipeline test against a simulated YouTube API. No network.

This drives the *real* runner, bronze writer, and panel through a simulated
21-day collection, using synthetic videos whose true decay parameters we choose.
It is the test that would have caught the integration bugs that only appear
after three weeks of live collection — which is exactly when it is too late.

It verifies the four properties the whole project depends on:

* posts are discovered and admitted once, not repeatedly
* snapshots accumulate on roughly the configured schedule
* re-running a cycle is idempotent (no duplicated snapshots)
* the resulting bronze data actually labels, with fast decayers coming out
  with a shorter time-to-death than slow ones
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from cdc.collect.panel import Panel
from cdc.collect.runner import run_cycle
from cdc.collect.youtube import QuotaMeter
from cdc.labels.death import Observation, label_post
from cdc.storage.bronze import iter_bronze

START = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)

# (video_id, channel, publish offset hours, asymptote A, decay rate k)
# Larger k = faster saturation = earlier death.
FIXTURES = [
    ("vid_fast_a", "chan_1", 0, 50_000, 0.60),
    ("vid_fast_b", "chan_1", 2, 20_000, 0.50),
    ("vid_slow_a", "chan_2", 1, 900_000, 0.04),
    ("vid_slow_b", "chan_2", 5, 400_000, 0.03),
    ("vid_mid_a", "chan_3", 3, 120_000, 0.15),
]


def true_views(A: float, k: float, age_h: float) -> int:
    if age_h <= 0:
        return 0
    return int(A * (1.0 - math.exp(-k * age_h)))


class FakeYouTube:
    """Minimal stand-in for YouTubeClient with the same surface the runner uses."""

    def __init__(self, now_provider):
        self.now = now_provider
        self.meter = QuotaMeter(budget=10_000)
        self.stats_calls = 0

    def recent_uploads(self, uploads_playlist, since, max_pages=2):
        self.meter.charge("playlistItems")
        chan = uploads_playlist.replace("UU_", "")
        out = []
        for vid, c, off, _A, _k in FIXTURES:
            if c != chan:
                continue
            published = START + timedelta(hours=off)
            if published >= since and published <= self.now():
                out.append({"video_id": vid, "published_at": published.isoformat(),
                            "channel_id": c, "title": vid})
        return out

    def video_stats(self, video_ids):
        ids = list(dict.fromkeys(video_ids))
        out = []
        for i in range(0, len(ids), 50):
            self.meter.charge("videos")
            self.stats_calls += 1
            for vid in ids[i:i + 50]:
                spec = next((f for f in FIXTURES if f[0] == vid), None)
                if spec is None:
                    continue
                _, chan, off, A, k = spec
                published = START + timedelta(hours=off)
                age = (self.now() - published).total_seconds() / 3600.0
                out.append({
                    "post_id": vid, "creator_id": chan, "creator_title": chan,
                    "published_at": published.isoformat(), "title": vid,
                    "description_len": 100, "tag_count": 5, "category_id": "28",
                    "duration_iso": "PT10M", "views": true_views(A, k, age),
                    "likes": int(true_views(A, k, age) * 0.04),
                    "comments": int(true_views(A, k, age) * 0.002), "raw": {},
                })
        return out


CHANS = [{"handle": f"@{c}", "channel_id": c, "uploads_playlist": f"UU_{c}",
          "tier": "mid", "category": "technology"} for c in ("chan_1", "chan_2", "chan_3")]


@pytest.fixture
def collected(tmp_path):
    """Run 21 simulated days of hourly cycles into an isolated bronze dir."""
    bronze = tmp_path / "bronze"
    clock = {"t": START}
    fake = FakeYouTube(lambda: clock["t"])

    reports = []
    for hour in range(21 * 24):
        clock["t"] = START + timedelta(hours=hour)
        reports.append(run_cycle(client=fake, chans=CHANS, bronze_root=bronze,
                                 now=clock["t"]))
    return bronze, reports, fake, clock


def test_every_fixture_video_is_discovered_exactly_once(collected):
    bronze, _, _, _ = collected
    posts = list(iter_bronze("posts", platform="youtube", root=bronze))
    ids = [p["post_id"] for p in posts]
    assert sorted(ids) == sorted(f[0] for f in FIXTURES)
    assert len(ids) == len(set(ids)), "a post was admitted to the panel twice"


def test_snapshots_follow_the_configured_schedule(collected):
    """Every scheduled mark is hit, and the discovery snapshot is extra.

    This test previously asserted 13 snapshots in total, which quietly encoded
    a bug: the at-discovery snapshot was consuming the t+1h slot, so a post
    reached only 12 of its 13 scheduled marks and the missing one was always
    the earliest and most valuable. The correct expectation is 13 *scheduled*
    observations plus the discovery one that precedes them.
    """
    bronze, _, _, clock = collected
    panel = Panel.from_bronze("youtube", root=bronze)
    for post in panel.posts.values():
        # 13 scheduled marks, all of which fall inside a 21-day run
        assert post.scheduled_snapshot_count == 13, (
            f"{post.post_id} hit {post.scheduled_snapshot_count} scheduled "
            f"marks, expected 13"
        )
        # Plus at most one earlier observation, written free at discovery.
        assert post.snapshot_count in (13, 14), (
            f"{post.post_id} has {post.snapshot_count} snapshots; expected the "
            f"13 scheduled ones plus at most one at-discovery observation"
        )


def test_rerunning_a_cycle_is_idempotent(collected):
    """The core storage guarantee: a retried cycle must not duplicate data."""
    bronze, _, fake, clock = collected
    before = len(list(iter_bronze("snapshots", platform="youtube", root=bronze)))
    # replay the final cycle verbatim, as a CI retry would
    run_cycle(client=fake, chans=CHANS, bronze_root=bronze, now=clock["t"])
    after = len(list(iter_bronze("snapshots", platform="youtube", root=bronze)))
    assert after == before, "re-running a cycle duplicated snapshot rows"


def test_collected_data_labels_and_ranks_decay_correctly(collected):
    """The payoff: bronze written by the real runner must produce labels whose
    ordering matches the decay rates we simulated. Fast decayers die first."""
    bronze, _, _, _ = collected
    by_post: dict[str, list[Observation]] = {}
    for s in iter_bronze("snapshots", platform="youtube", root=bronze):
        by_post.setdefault(s["post_id"], []).append(
            Observation(age_hours=float(s["age_hours"]), value=float(s["views"]))
        )

    labels = {pid: label_post(pid, obs) for pid, obs in by_post.items()}
    assert all(l.usable for l in labels.values()), "some posts failed to label"

    fast = min(labels["vid_fast_a"].t_death, labels["vid_fast_b"].t_death)
    slow = min(labels["vid_slow_a"].t_death, labels["vid_slow_b"].t_death)
    assert labels["vid_fast_a"].event_observed, "a k=0.6 post must be seen to die"
    assert fast < labels["vid_mid_a"].t_death < slow, (
        f"death order wrong: fast={fast}, mid={labels['vid_mid_a'].t_death}, slow={slow}"
    )


def test_quota_cost_matches_the_documented_model(collected):
    """The BDA report states a cost model. It must match what the code spends."""
    _, reports, fake, _ = collected
    # 3 channels discovered every cycle for 504 cycles
    discovery_units = fake.meter.by_endpoint["playlistItems"]
    assert discovery_units == 3 * 21 * 24
    # statistics calls only happen when something is due — far fewer than cycles
    assert fake.meter.by_endpoint["videos"] < 21 * 24
