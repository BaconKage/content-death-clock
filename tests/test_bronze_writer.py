"""Bronze layer guarantees.

The writer must never be able to lose an observation. Snapshots cannot be
re-collected — a post's first six hours happen once — so a bug that quietly
drops rows is unrecoverable, and would not be noticed until modelling, weeks
later. These tests exist because exactly that happened during the first live
run: two cycles in the same clock hour, and the second overwrote the first.
"""
from __future__ import annotations

import json

from cdc.storage.bronze import BronzeWriter, dedupe, iter_bronze


def _rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_second_run_in_same_hour_merges_instead_of_clobbering(tmp_path):
    """The regression. Two runs share a cycle id but carry different records;
    the union must survive, not just the later write."""
    cid = "2026-09-01T15"

    w1 = BronzeWriter("youtube", kind="snapshots", cid=cid, root=tmp_path)
    w1.add({"post_id": "a", "snapshot_ts": "2026-09-01T15:05:00", "views": 100})
    w1.commit()

    w2 = BronzeWriter("youtube", kind="snapshots", cid=cid, root=tmp_path)
    w2.add({"post_id": "b", "snapshot_ts": "2026-09-01T15:45:00", "views": 200})
    path = w2.commit()

    got = {r["post_id"] for r in _rows(path)}
    assert got == {"a", "b"}, f"lost an observation: {got}"


def test_identical_rerun_does_not_duplicate(tmp_path):
    """True idempotency: replaying the same cycle verbatim changes nothing."""
    cid = "2026-09-01T16"
    rec = {"post_id": "a", "snapshot_ts": "2026-09-01T16:05:00", "views": 100}

    for _ in range(3):
        w = BronzeWriter("youtube", kind="snapshots", cid=cid, root=tmp_path)
        w.add(dict(rec))
        path = w.commit()

    assert len(_rows(path)) == 1


def test_fresher_read_of_same_observation_wins(tmp_path):
    """Same post, same snapshot timestamp, corrected value -> keep the later one
    rather than storing both and creating a phantom duplicate observation."""
    cid = "2026-09-01T17"
    ts = "2026-09-01T17:05:00"

    w1 = BronzeWriter("youtube", kind="snapshots", cid=cid, root=tmp_path)
    w1.add({"post_id": "a", "snapshot_ts": ts, "views": 100})
    w1.commit()

    w2 = BronzeWriter("youtube", kind="snapshots", cid=cid, root=tmp_path)
    w2.add({"post_id": "a", "snapshot_ts": ts, "views": 137})
    path = w2.commit()

    rows = _rows(path)
    assert len(rows) == 1
    assert rows[0]["views"] == 137


def test_posts_dedupe_on_id_alone(tmp_path):
    """A post is admitted to the panel once; re-seeing it must not re-admit it."""
    cid = "2026-09-01T18"
    for views in (10, 20, 30):
        w = BronzeWriter("youtube", kind="posts", cid=cid, root=tmp_path)
        w.add({"post_id": "x", "published_at": "2026-09-01T17:00:00Z", "views": views})
        path = w.commit()
    rows = _rows(path)
    assert len(rows) == 1 and rows[0]["views"] == 30


def test_crash_before_commit_leaves_no_partial_file(tmp_path):
    """An interrupted cycle must leave nothing, so the retry is a clean redo."""
    w = BronzeWriter("youtube", kind="snapshots", cid="2026-09-01T19", root=tmp_path)
    w.add({"post_id": "a", "snapshot_ts": "2026-09-01T19:05:00", "views": 1})
    # simulate the process dying before commit()
    del w
    assert list(iter_bronze("snapshots", platform="youtube", root=tmp_path)) == []


def test_empty_buffer_writes_nothing(tmp_path):
    w = BronzeWriter("youtube", kind="snapshots", cid="2026-09-01T20", root=tmp_path)
    assert w.commit() is None


def test_metadata_is_stamped_on_every_record(tmp_path):
    w = BronzeWriter("youtube", kind="snapshots", cid="2026-09-01T21", root=tmp_path)
    w.add({"post_id": "a", "snapshot_ts": "2026-09-01T21:05:00", "views": 1})
    path = w.commit()
    r = _rows(path)[0]
    assert r["_cycle_id"] == "2026-09-01T21"
    assert r["_platform"] == "youtube"
    assert r["_ingested_at"]


def test_dedupe_helper_is_last_write_wins():
    rows = [{"post_id": "a", "snapshot_ts": "t1", "views": 1},
            {"post_id": "a", "snapshot_ts": "t1", "views": 2},
            {"post_id": "a", "snapshot_ts": "t2", "views": 3}]
    out = dedupe(rows, ("post_id", "snapshot_ts"))
    assert len(out) == 2
    assert {r["views"] for r in out} == {2, 3}
