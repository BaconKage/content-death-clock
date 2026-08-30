"""Bronze layer: append-only raw API responses on disk.

Design notes that matter for the BDA report:

* **Partitioned** as ``platform=<p>/dt=<YYYY-MM-DD>/<kind>-<cycle_id>.jsonl`` so a
  date range can be pruned without opening every file.
* **Idempotent by merge, not by overwrite.** A cycle buffers its records and
  commits them in one atomic ``os.replace``. The filename is keyed on ``cycle_id``
  (the UTC hour), so a re-run targets the same file — but it **merges** with what
  is already there, deduplicating on a natural key, rather than replacing it.
  Plain overwrite was wrong: two runs within the same hour are not necessarily
  retries of identical work (the panel moves between them, and a run aborted by
  quota exhaustion holds strictly less data), so the later write would silently
  destroy observations the earlier one had captured. Observations cannot be
  re-collected — the past is gone — so the writer must never be able to lose one.
  A crash mid-cycle leaves no partial file.
* **Raw is preserved.** We store the provider payload untouched under ``raw`` next
  to our normalised fields, so a schema mistake in silver is always recoverable
  without re-scraping. Re-scraping is impossible here: the past is gone.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from cdc.config import path_for

CYCLE_FMT = "%Y-%m-%dT%H"


def cycle_id(now: datetime | None = None) -> str:
    """Identifier for the current hourly collection cycle (UTC)."""
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).strftime(CYCLE_FMT)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BronzeWriter:
    """Buffers a cycle's records and commits them atomically.

    Usage::

        with BronzeWriter("youtube", kind="snapshots") as w:
            w.add({...})
    """

    # Natural key per record kind, used to merge a re-run with what is on disk.
    DEDUPE_KEYS = {
        "snapshots": ("post_id", "snapshot_ts"),
        "posts": ("post_id",),
    }

    def __init__(self, platform: str, kind: str, cid: str | None = None,
                 root: Path | None = None) -> None:
        self.platform = platform
        self.kind = kind
        self.cycle = cid or cycle_id()
        self.root = root or path_for("bronze_dir")
        self._buf: list[dict[str, Any]] = []
        self.committed_path: Path | None = None

    @property
    def target(self) -> Path:
        dt = self.cycle.split("T")[0]
        d = self.root / f"platform={self.platform}" / f"dt={dt}"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.kind}-{self.cycle}.jsonl"

    def add(self, record: dict[str, Any]) -> None:
        record.setdefault("_cycle_id", self.cycle)
        record.setdefault("_ingested_at", utcnow_iso())
        record.setdefault("_platform", self.platform)
        self._buf.append(record)

    def __len__(self) -> int:
        return len(self._buf)

    def _existing(self, target: Path) -> list[dict[str, Any]]:
        """Records a previous run already wrote to this cycle's file."""
        if not target.exists():
            return []
        out = []
        with target.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def commit(self) -> Path | None:
        """Atomically write what is on disk MERGED with the buffer.

        Returns the path, or None if there is nothing to write.
        """
        if not self._buf:
            return None
        target = self.target

        # Merge with whatever a previous run in this same cycle already wrote.
        # On key collision the later record wins — it is the fresher read — but
        # no earlier observation is ever dropped.
        keys = self.DEDUPE_KEYS.get(self.kind)
        if keys:
            acc: dict[tuple, dict[str, Any]] = {}
            for rec in self._existing(target) + self._buf:
                acc[tuple(rec.get(k) for k in keys)] = rec
            merged = list(acc.values())
        else:
            merged = self._existing(target) + self._buf

        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                for rec in merged:
                    fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(tmp, target)          # atomic on both POSIX and Windows
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.committed_path = target
        return target

    def __enter__(self) -> "BronzeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()


def iter_bronze(kind: str, platform: str | None = None,
                root: Path | None = None) -> Iterator[dict[str, Any]]:
    """Stream every bronze record of a kind, oldest partition first."""
    root = root or path_for("bronze_dir")
    pattern = f"platform={platform}" if platform else "platform=*"
    for pdir in sorted(root.glob(pattern)):
        for ddir in sorted(pdir.glob("dt=*")):
            for f in sorted(ddir.glob(f"{kind}-*.jsonl")):
                with f.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            yield json.loads(line)


def dedupe(records: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Last-write-wins dedup on a composite key.

    Belt-and-braces: writes are already idempotent, but silver must never trust
    that. A duplicated snapshot would corrupt every velocity computation
    downstream, so the guarantee is enforced twice.
    """
    out: dict[tuple, dict[str, Any]] = {}
    for r in records:
        out[tuple(r.get(k) for k in keys)] = r
    return list(out.values())
