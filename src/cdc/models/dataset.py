"""Assemble the analysis frame: one row per post, features + label.

This is where the two halves finally meet. Features come from the first six
hours only; the label comes from the post's entire observed history. Keeping
them in separate code paths until this point is deliberate — it is why the
leakage boundary is testable at all.

Exclusions are applied here, all pre-specified in the frozen plan, and every
one is **counted** rather than silently dropped. The attrition table the paper
reports comes straight out of this function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cdc.config import path_for, settings
from cdc.labels.death import Observation, label_post
from cdc.transform.features import features_for_post


@dataclass
class AnalysisFrame:
    frame: pd.DataFrame
    attrition: dict[str, int] = field(default_factory=dict)
    platform: str = "youtube"

    @property
    def n_deaths(self) -> int:
        return int(self.frame["event_observed"].sum()) if len(self.frame) else 0

    @property
    def n_creators(self) -> int:
        return int(self.frame["creator_id"].nunique()) if len(self.frame) else 0

    @property
    def censoring_rate(self) -> float:
        if not len(self.frame):
            return float("nan")
        return float(1.0 - self.frame["event_observed"].mean())

    def attrition_table(self) -> pd.DataFrame:
        return pd.DataFrame({"stage": list(self.attrition),
                             "posts": list(self.attrition.values())})


def build(platform: str = "youtube", cohort_end: pd.Timestamp | None = None
          ) -> AnalysisFrame:
    """Features joined to labels, with the pre-specified exclusions applied.

    Landmarked: only posts still alive at ``modelling.landmark_hours`` are
    eligible, and the outcome is the time remaining from the landmark. Both the
    landmarked outcome (``t_death``) and the raw one (``t_death_from_publish``)
    are kept, so a sensitivity analysis needs no re-derivation.
    """
    sd = path_for("silver_dir")
    snaps = pd.read_parquet(sd / "snapshots.parquet")
    posts = pd.read_parquet(sd / "posts.parquet").set_index("post_id")

    cfg = settings()["modelling"]
    cutoff = float(cfg["feature_cutoff_hours"])
    landmark = float(cfg.get("landmark_hours", 0) or 0)
    att: dict[str, int] = {}

    snaps = snaps[snaps["platform"] == platform]
    ids = set(snaps["post_id"].unique())
    att["observed on this platform"] = len(ids)

    # --- exclusion: per-creator daily cap (plan amendment 2026-08-31)
    capped = set(posts.index[posts.get("over_creator_daily_cap", False)])
    ids -= capped
    att["after creator_daily_cap"] = len(ids)

    # --- exclusion: Cohort A boundary, if a freeze date is being applied
    if cohort_end is not None:
        keep = {p for p in ids if p in posts.index
                and posts.loc[p, "published_at"] < cohort_end}
        att["after cohort A cutoff"] = len(keep)
        ids = keep

    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for pid, g in snaps[snaps["post_id"].isin(ids)].groupby("post_id"):
        v = g[g["primary_value"].notna()].sort_values("age_hours")
        if v.empty:
            reasons["no usable metric"] = reasons.get("no usable metric", 0) + 1
            continue

        obs = [Observation(a, x) for a, x in
               zip(v["age_hours"].to_numpy(float), v["primary_value"].to_numpy(float))]
        lab = label_post(str(pid), obs)
        if not lab.usable:
            reasons[lab.exclude_reason] = reasons.get(lab.exclude_reason, 0) + 1
            continue

        # --- LANDMARK. A post must still be at risk when the prediction is
        # --- made, or "predicting" its death is circular.
        if landmark:
            if lab.event_observed and lab.t_death <= landmark:
                key = "died before the landmark"
                reasons[key] = reasons.get(key, 0) + 1
                continue
            if float(v["age_hours"].max()) < landmark:
                # Not yet observed at the landmark: we cannot assert it was
                # still alive there, so it is not yet eligible.
                key = "not yet observed at the landmark"
                reasons[key] = reasons.get(key, 0) + 1
                continue

        meta = posts.loc[pid].to_dict() if pid in posts.index else {}
        meta.update({"post_id": pid, "platform": platform,
                     "creator_id": g["creator_id"].iloc[0]})
        feats = features_for_post(v, meta, cutoff)
        # Outcome is time REMAINING from the landmark, not from publication.
        t = (lab.t_death - landmark) if landmark else lab.t_death
        feats.update({
            "t_death": t,
            "t_death_from_publish": lab.t_death,
            "event_observed": bool(lab.event_observed),
            "n_intervals": lab.n_intervals,
            "peak_velocity": lab.peak_velocity,
            "monotonicity_repairs": lab.monotonicity_repairs,
            "stratum_tier": meta.get("stratum_tier"),
        })
        rows.append(feats)

    for r, n in reasons.items():
        att[f"excluded: {r}"] = -n
    df = pd.DataFrame(rows)
    att["FINAL analysis set"] = len(df)

    # A post with no usable early observation cannot contribute a prediction.
    if len(df):
        before = len(df)
        df = df[df["n_early_observations"] > 0].reset_index(drop=True)
        if len(df) != before:
            att["excluded: no early observation"] = -(before - len(df))
            att["FINAL analysis set"] = len(df)

    return AnalysisFrame(frame=df, attrition=att, platform=platform)


def usable_features(df: pd.DataFrame, min_coverage: float = 0.5) -> list[str]:
    """Feature columns present often enough to model with.

    A column that is 90% missing contributes nothing but noise and destabilises
    the fit on a small sample. Which columns survived is reported, so the paper
    states what the model actually used rather than what it was offered.
    """
    from cdc.models.synthetic import FEATURE_COLUMNS
    keep = []
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().mean() >= min_coverage and col.nunique(dropna=True) >= 2:
            keep.append(c)
    return keep
