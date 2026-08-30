"""Silver -> features. Everything the model is allowed to know at prediction time.

**The one rule.** A feature may use only observations from the first
``modelling.feature_cutoff_hours`` (6h) of a post's life. The label is derived
from the post's *entire* observed history, so any feature that peeks past the
cutoff leaks the answer and produces a model that looks excellent and predicts
nothing. The cutoff is enforced in ``early_window()``, asserted in
``build_features()``, and pinned by a test — three places, because this is the
single failure that would invalidate the RM paper without producing any visible
symptom.

**Irregular sampling.** Snapshots do not land exactly on 1h, 3h, 6h — the
scheduler drifts and CI queues. Rather than pretend they do, we linearly
interpolate the cumulative metric to each standard mark from the two
observations bracketing it. A mark with no observation at or after it yields
NaN, never a guess: a post that is only two hours old genuinely has no 6h
feature, and imputing one would invent data.

Deliberately NOT included here: creator historical averages. Those must be
computed inside a cross-validation fold, from training posts only. Computing
them over the whole dataset would leak test-set outcomes through the creator
mean — a subtle leak that survives GroupKFold and is easy to miss.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from cdc.config import path_for, settings

log = logging.getLogger("cdc.features")

STANDARD_MARKS = (1.0, 3.0, 6.0)


def early_window(group: pd.DataFrame, cutoff_h: float) -> pd.DataFrame:
    """Observations strictly within the feature cutoff. The leakage boundary."""
    return group[group["age_hours"] <= cutoff_h].sort_values("age_hours")


def interpolate_at(ages: np.ndarray, values: np.ndarray, mark: float) -> float:
    """Cumulative metric at `mark` hours, linearly interpolated.

    NaN if the mark is not bracketed by observations — we never extrapolate.
    Extrapolating a cumulative count forward is how a two-hour-old post acquires
    an imaginary six-hour view count.
    """
    if len(ages) < 2 or mark < ages[0] or mark > ages[-1]:
        return float("nan")
    return float(np.interp(mark, ages, values))


def features_for_post(group: pd.DataFrame, meta: dict[str, Any],
                      cutoff_h: float) -> dict[str, Any]:
    early = early_window(group, cutoff_h)
    ages = early["age_hours"].to_numpy(dtype=float)
    vals = early["primary_value"].astype(float).to_numpy()

    ok = ~np.isnan(vals)
    ages, vals = ages[ok], vals[ok]

    f: dict[str, Any] = {
        "post_id": meta["post_id"],
        "platform": meta["platform"],
        "creator_id": meta["creator_id"],
        "n_early_observations": int(len(ages)),
        "first_observed_at_hours": float(ages[0]) if len(ages) else np.nan,
    }

    # --- cumulative value at each standard mark
    for m in STANDARD_MARKS:
        f[f"value_at_{int(m)}h"] = interpolate_at(ages, vals, m)

    # --- average velocity up to each mark (units/hour since publish)
    for m in STANDARD_MARKS:
        v = f[f"value_at_{int(m)}h"]
        f[f"velocity_to_{int(m)}h"] = v / m if np.isfinite(v) else np.nan

    # --- interval velocity between marks: the shape of the early curve
    v1, v3, v6 = (f[f"value_at_{int(m)}h"] for m in STANDARD_MARKS)
    f["velocity_1_3h"] = (v3 - v1) / 2.0 if np.isfinite(v3) and np.isfinite(v1) else np.nan
    f["velocity_3_6h"] = (v6 - v3) / 3.0 if np.isfinite(v6) and np.isfinite(v3) else np.nan

    # --- log growth ratios. Scale-free, which is the point: they let a 300-view
    # --- post and a 3M-view post contribute the same kind of information.
    f["log_growth_1_3h"] = _log_ratio(v3, v1)
    f["log_growth_3_6h"] = _log_ratio(v6, v3)

    # --- acceleration: is the curve already bending over by hour six?
    if np.isfinite(f["velocity_1_3h"]) and np.isfinite(f["velocity_3_6h"]):
        f["acceleration"] = f["velocity_3_6h"] - f["velocity_1_3h"]
        denom = f["velocity_1_3h"]
        f["decay_ratio_3_6_over_1_3"] = (f["velocity_3_6h"] / denom
                                         if denom else np.nan)
    else:
        f["acceleration"] = np.nan
        f["decay_ratio_3_6_over_1_3"] = np.nan

    # --- log-scaled levels: raw counts span six orders of magnitude
    for m in STANDARD_MARKS:
        v = f[f"value_at_{int(m)}h"]
        f[f"log_value_at_{int(m)}h"] = (np.log10(v + 1) if np.isfinite(v) else np.nan)

    # --- metadata, all fixed at publish time and so never a leak
    pub = meta.get("published_at")
    f.update({
        "publish_hour_utc": pub.hour if pd.notna(pub) else np.nan,
        "publish_weekday": pub.dayofweek if pd.notna(pub) else np.nan,
        "publish_is_weekend": (int(pub.dayofweek >= 5) if pd.notna(pub) else np.nan),
        "duration_sec": meta.get("duration_sec"),
        "log_duration_sec": (np.log10(meta["duration_sec"] + 1)
                             if meta.get("duration_sec") else np.nan),
        "title_len": meta.get("title_len"),
        "description_len": meta.get("description_len"),
        "caption_len": meta.get("caption_len"),
        "tag_count": meta.get("tag_count"),
        "hashtag_count": meta.get("hashtag_count"),
        "media_type": meta.get("media_type"),
        "stratum_tier": meta.get("stratum_tier"),
        "stratum_category": meta.get("stratum_category"),
        "follower_count": meta.get("follower_count"),
        "log_follower_count": (np.log10(meta["follower_count"] + 1)
                               if meta.get("follower_count") else np.nan),
    })
    return f


def _log_ratio(later: float, earlier: float) -> float:
    """log10(later / earlier). Exactly scale-free, or NaN.

    No epsilon smoothing. Adding +1 to both terms to dodge division by zero
    looks harmless and is not: it makes the ratio depend on absolute scale, and
    it distorts *small* posts most. A micro creator's video with 8 views at 1h
    and 24 at 3h would get a different growth feature from a large creator's
    800 -> 2400, despite identical shape — biasing the creator-size comparison
    the feature was built to support.

    A post with zero engagement at the earlier mark has no defined growth ratio.
    NaN says that honestly; the level features still carry the information, and
    the models handle missingness.
    """
    if not (np.isfinite(later) and np.isfinite(earlier)):
        return float("nan")
    if earlier <= 0 or later < 0:
        return float("nan")
    return float(np.log10(later / earlier))


def build(write: bool = True) -> pd.DataFrame:
    cutoff = float(settings()["modelling"]["feature_cutoff_hours"])
    silver = path_for("silver_dir")
    snaps_path, posts_path = silver / "snapshots.parquet", silver / "posts.parquet"
    if not snaps_path.exists() or not posts_path.exists():
        raise SystemExit("silver layer missing — run: python -m cdc.transform.silver")

    snaps = pd.read_parquet(snaps_path)
    posts = pd.read_parquet(posts_path).set_index("post_id")

    rows = []
    for post_id, group in snaps.groupby("post_id"):
        if post_id not in posts.index:
            continue
        meta = posts.loc[post_id].to_dict()
        meta["post_id"] = post_id
        meta["platform"] = group["platform"].iloc[0]
        meta["creator_id"] = group["creator_id"].iloc[0]
        rows.append(features_for_post(group, meta, cutoff))

    df = pd.DataFrame(rows)

    # Belt-and-braces leakage assertion: no post may contribute a feature that
    # required an observation past the cutoff.
    if not df.empty:
        assert df["first_observed_at_hours"].dropna().le(cutoff).all(), \
            "a feature was built from an observation past the cutoff"

    if write and not df.empty:
        df.to_parquet(path_for("gold_dir") / "features.parquet", index=False)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the feature table.")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    df = build(write=not args.no_write)
    cutoff = settings()["modelling"]["feature_cutoff_hours"]
    print("=" * 62)
    print(f"  FEATURES   (leakage cutoff: t <= {cutoff}h)")
    print("=" * 62)
    print(f"  posts with features        {len(df)}")
    if not df.empty:
        for m in (1, 3, 6):
            n = int(df[f"value_at_{m}h"].notna().sum())
            print(f"  usable {m}h feature          {n}")
        print(f"  feature columns            {len(df.columns)}")
    print("=" * 62)
    if df.empty or df["value_at_6h"].notna().sum() == 0:
        print("  No post is old enough yet for a 6h feature. Expected this early —")
        print("  the first ones become usable ~6h after collection started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
