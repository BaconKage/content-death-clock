"""The instant model's training table: one row per video per scheduled mark.

Each row is one real reading of one video (the reading nearest a scheduled
mark, inside that mark's window), turned into inputs by ``features.from_silver``,
with the video's fade time attached. Rows at the four checkpoints (1, 3, 6 and
12 hours) train the model; rows at later marks are kept only so the report can
compare an older video with videos of the same age.

**The outcome is not conditioned on the video still being alive.** A video
pasted into the app may already have faded, and one reading cannot tell. So
the model predicts the fade time measured from posting, for every video read at
that age, dead or alive. "Fades within its first two days" then includes "has
already faded", which is the honest question for a tool that sees one reading.

**Only the cohorts asked for are labelled.** Cohort C's fade times are computed
by the single Cohort C evaluation and by nothing else.

**What ships.** ``anonymise`` drops video and channel ids and titles, keeps the
rounded inputs, and replaces ids with arbitrary integers. The app trains from
that file at start-up, so the deployed model is built from exactly the rows the
tests used.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from cdc.tool.features import (CHECKPOINTS, EARLY_TOL_H, FEATURES, LATE_TOL_H,
                               MARKS, MAX_AGE_H, MIN_AGE_H, fade_mark, from_silver)

ASSETS = Path(__file__).resolve().parent / "assets"
TABLE_PATH = ASSETS / "training_table.csv"
N_GROUPS = 5


def _h(s: object) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()


def creator_group(creator_id: object) -> int:
    """A fixed, arbitrary split of channels into five groups, for unseen-channel tests."""
    return int(_h(creator_id), 16) % N_GROUPS


def labelled_videos(cohorts: tuple[str, ...]) -> pd.DataFrame:
    """One row per labelled YouTube video in the named cohorts: outcome + metadata."""
    from cdc.eval.practical import cohort_of
    from cdc.models import dataset

    parts = [dataset.build(platform="youtube", cohort=c, landmark=0.0).frame.assign(cohort=c)
             for c in cohorts if c in ("A", "B")]
    if "C" in cohorts:
        allf = dataset.build(platform="youtube", cohort="all", landmark=0.0).frame
        parts.append(allf[cohort_of(allf["published_at"]).to_numpy() == "C"].assign(cohort="C"))
    keep = ["post_id", "creator_id", "cohort", "published_at", "follower_count", "duration_sec",
            "title_len", "t_death_from_publish", "event_observed"]
    return pd.concat([p[keep] for p in parts], ignore_index=True)


def _snapshots(post_ids) -> pd.DataFrame:
    from cdc.config import path_for
    s = pd.read_parquet(path_for("silver_dir") / "snapshots.parquet")
    s = s[(s["platform"] == "youtube") & s["post_id"].isin(set(post_ids)) & s["views"].notna()]
    return s[["post_id", "age_hours", "views", "likes", "comments"]]


def _with_features(readings: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    meta = videos.set_index("post_id")
    feats = [from_silver(r.age_hours, r.views, r.likes, r.comments, meta.loc[r.post_id].to_dict())
             for r in readings.itertuples(index=False)]
    out = pd.concat([readings.reset_index(drop=True), pd.DataFrame(feats, columns=FEATURES)], axis=1)
    v = meta.loc[out["post_id"]]
    out["creator_id"] = v["creator_id"].to_numpy()
    out["cohort"] = v["cohort"].to_numpy()
    out["fade_h"] = [fade_mark(t) for t in v["t_death_from_publish"].to_numpy(float)]
    out["faded"] = v["event_observed"].to_numpy(bool)
    out["video_key"] = [_h(p) for p in out["post_id"]]
    return out


def build_rows(cohorts: tuple[str, ...] = ("A", "B"),
               videos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Readings at the scheduled marks, one per video per mark, with outcomes."""
    videos = labelled_videos(cohorts) if videos is None else videos
    s = _snapshots(videos["post_id"])
    ages = s["age_hours"].to_numpy(float)
    mark = np.full(len(s), np.nan)
    for m in MARKS:
        mark[(ages >= m - EARLY_TOL_H) & (ages <= m + LATE_TOL_H)] = m
    s = s.assign(mark=mark).dropna(subset=["mark"])
    s = (s.assign(gap=(s["age_hours"] - s["mark"]).abs())
          .sort_values(["post_id", "mark", "gap", "age_hours"])
          .drop_duplicates(["post_id", "mark"]).drop(columns="gap"))
    rows = _with_features(s, videos)
    return _order(rows)


def offgrid_rows(cohorts: tuple[str, ...], videos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Real readings inside the odds age range that belong to no mark (late cycles)."""
    videos = labelled_videos(cohorts) if videos is None else videos
    s = _snapshots(videos["post_id"])
    a = s["age_hours"]
    on = np.zeros(len(s), bool)
    for m in MARKS:
        on |= ((a >= m - EARLY_TOL_H) & (a <= m + LATE_TOL_H)).to_numpy()
    s = s[~on & (a >= MIN_AGE_H).to_numpy() & (a <= MAX_AGE_H).to_numpy()]
    return _order(_with_features(s.assign(mark=np.nan), videos)) if len(s) else pd.DataFrame()


def simulated_rows(cohorts: tuple[str, ...], ages=(2.0, 4.5, 9.0),
                   videos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Readings between the checkpoints, interpolated from each video's two real neighbours.

    A check, not training data: the app is used at any age, readings exist
    mostly at the marks, and this shows whether ages in between behave.
    Linear interpolation of cumulative counts between bracketing readings, the
    same rule ``cdc.transform.features.interpolate_at`` uses; never extrapolated.
    """
    videos = labelled_videos(cohorts) if videos is None else videos
    s = _snapshots(videos["post_id"]).sort_values(["post_id", "age_hours"])
    out = []
    for pid, g in s.groupby("post_id", sort=False):
        a = g["age_hours"].to_numpy(float)
        for x in ages:
            i = int(np.searchsorted(a, x))
            if i == 0 or i >= len(a) or a[i] - a[i - 1] > 8:
                continue
            lo, hi = g.iloc[i - 1], g.iloc[i]
            w = (x - a[i - 1]) / (a[i] - a[i - 1])
            lin = (lambda c: lo[c] + w * (hi[c] - lo[c])
                   if pd.notna(lo[c]) and pd.notna(hi[c]) else np.nan)
            out.append({"post_id": pid, "age_hours": x, "views": lin("views"),
                        "likes": lin("likes"), "comments": lin("comments"), "mark": x})
    return _order(_with_features(pd.DataFrame(out), videos)) if out else pd.DataFrame()


def _order(rows: pd.DataFrame) -> pd.DataFrame:
    """A fixed row order. The forest's bootstrap depends on it, so it must not vary."""
    return rows.sort_values(["video_key", "mark", "age_hours"]).reset_index(drop=True)


def model_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["mark"].isin(CHECKPOINTS)]


# ------------------------------------------------------------- the shipped file
def anonymise(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop ids; number videos and channels arbitrarily. Row order is preserved."""
    vid = {k: i for i, k in enumerate(dict.fromkeys(rows["video_key"]))}
    ch_keys = sorted({_h(c) for c in rows["creator_id"]})
    ch = {k: i for i, k in enumerate(ch_keys)}
    return pd.DataFrame({
        "cohort": rows["cohort"].to_numpy(),
        "video": rows["video_key"].map(vid).to_numpy(),
        "channel": [ch[_h(c)] for c in rows["creator_id"]],
        "mark": rows["mark"].to_numpy(float),
        **{f: rows[f].to_numpy(float) for f in FEATURES},
        "fade_h": rows["fade_h"].to_numpy(float),
        "faded": rows["faded"].to_numpy(bool),
    })


def to_csv_text(table: pd.DataFrame) -> str:
    buf = io.StringIO()
    table.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue()


def digest(table: pd.DataFrame) -> str:
    return hashlib.sha256(to_csv_text(table).encode()).hexdigest()


def save_table(table: pd.DataFrame, path: Path = TABLE_PATH) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = to_csv_text(table)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def load_table(path: Path = TABLE_PATH) -> pd.DataFrame:
    t = pd.read_csv(path)
    t["faded"] = t["faded"].astype(bool)
    return t
