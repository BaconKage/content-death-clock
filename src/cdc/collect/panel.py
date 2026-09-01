"""Panel state: who is being tracked, and what is due for a snapshot.

State is **derived from the bronze layer**, never stored separately. A separate
state file is one more thing that can drift, corrupt, or disagree with the data;
rebuilding from bronze every cycle is cheap at this scale (tens of thousands of
records) and means a wiped working directory loses nothing that a ``git pull``
cannot restore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from cdc.config import settings
from cdc.storage.bronze import iter_bronze


def _first_mark_opens() -> float:
    """Age at which the earliest scheduled mark's tolerance window opens."""
    cfg = settings()["collection"]
    schedule = cfg["snapshot_schedule_hours"]
    tol = float(cfg.get("snapshot_tolerance_hours", 0.0))
    return float(min(schedule)) - tol if schedule else 0.0


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


@dataclass
class TrackedPost:
    post_id: str
    platform: str
    creator_id: str | None
    published_at: datetime
    snapshot_count: int = 0
    # Snapshots old enough to satisfy a scheduled mark. The at-discovery
    # snapshot is usually taken minutes after publication, before the first
    # mark's window even opens, so it can satisfy nothing - see `due`.
    scheduled_snapshot_count: int = 0
    last_snapshot_at: datetime | None = None

    def age_hours(self, now: datetime) -> float:
        return (now - self.published_at).total_seconds() / 3600.0


@dataclass
class Panel:
    posts: dict[str, TrackedPost] = field(default_factory=dict)

    @classmethod
    def from_bronze(cls, platform: str | None = None, root=None) -> "Panel":
        panel = cls()
        for rec in iter_bronze("posts", platform=platform, root=root):
            pid = rec.get("post_id")
            pub = _parse(rec.get("published_at"))
            if not pid or pub is None:
                continue
            # First sighting wins for immutable metadata; published_at should
            # never change, and if it does we want the original observation.
            if pid not in panel.posts:
                panel.posts[pid] = TrackedPost(
                    post_id=pid,
                    platform=rec.get("_platform", platform or "unknown"),
                    creator_id=rec.get("creator_id"),
                    published_at=pub,
                )
        for rec in iter_bronze("snapshots", platform=platform, root=root):
            pid = rec.get("post_id")
            p = panel.posts.get(pid)
            if p is None:
                continue
            p.snapshot_count += 1
            age = rec.get("age_hours")
            try:
                if age is not None and float(age) >= _first_mark_opens():
                    p.scheduled_snapshot_count += 1
            except (TypeError, ValueError):
                pass
            ts = _parse(rec.get("snapshot_ts"))
            if ts and (p.last_snapshot_at is None or ts > p.last_snapshot_at):
                p.last_snapshot_at = ts
        return panel

    # ------------------------------------------------------------------ queries
    def due(self, now: datetime | None = None) -> list[TrackedPost]:
        """Posts owed a snapshot this cycle.

        Rule: count how many scheduled marks the post's age has passed; if it
        has fewer *scheduled* snapshots than marks passed, it is due. This is
        self-healing — after downtime a post is simply behind and gets caught up
        on the next cycle, rather than the missed marks being lost forever. The
        actual observation timestamp is what gets recorded, so a late snapshot is
        late-but-honest rather than silently mislabelled as on-time.

        **Why "scheduled" and not simply "all".** Discovery writes a snapshot the
        moment a post is found, typically within minutes of publication. Counting
        that against the marks made every post permanently one behind, and the
        mark it swallowed was always the first — t+1h, the most valuable and
        least recoverable point on a decay curve. Measured effect before the fix:
        posts published after the scheduler was repaired still reached the t+1h
        window only 45% of the time, while t+3h and later ran at 92-97%.

        A snapshot therefore counts toward the schedule only if it is old enough
        to fall inside the first mark's tolerance window. One taken before that
        is real data and is kept — it is the earliest observation we will ever
        have — but it satisfies no mark, because it answers a question no mark
        asked.
        """
        now = now or datetime.now(timezone.utc)
        cfg = settings()["collection"]
        schedule = cfg["snapshot_schedule_hours"]
        max_age = float(cfg["max_track_hours"])

        out = []
        for p in self.posts.values():
            age = p.age_hours(now)
            if age < 0 or age > max_age:
                continue
            marks_passed = sum(1 for m in schedule if age >= float(m))
            if p.scheduled_snapshot_count < marks_passed:
                out.append(p)
        # Oldest-first: a post about to age out of a mark window matters more
        # than one that will still be there next cycle.
        out.sort(key=lambda p: p.published_at)
        return out

    def active(self, now: datetime | None = None) -> list[TrackedPost]:
        now = now or datetime.now(timezone.utc)
        max_age = float(settings()["collection"]["max_track_hours"])
        return [p for p in self.posts.values() if 0 <= p.age_hours(now) <= max_age]

    def stats(self, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        act = self.active(now)
        counts = [p.snapshot_count for p in self.posts.values()]
        return {
            "posts_total": len(self.posts),
            "posts_active": len(act),
            "posts_due": len(self.due(now)),
            "snapshots_total": sum(counts),
            "snapshots_per_post_mean": (sum(counts) / len(counts)) if counts else 0.0,
        }


def discovery_window(now: datetime | None = None) -> datetime:
    """Earliest publish time a newly-seen post may have to enter the panel."""
    now = now or datetime.now(timezone.utc)
    hours = float(settings()["collection"]["discovery_lookback_hours"])
    return now - timedelta(hours=hours)
