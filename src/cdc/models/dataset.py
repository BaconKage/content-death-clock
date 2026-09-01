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
    cohort: str = "all"

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


def freeze_instant() -> pd.Timestamp:
    """The pre-specified Cohort A boundary, from settings.

    Fixed on 2026-08-30, before any outcome data was examined. It lives in
    config rather than in code so the frozen plan can quote an exact value and
    a reader can check that the two agree.
    """
    raw = str(settings()["modelling"]["cohort_a_freeze_utc"])
    return pd.Timestamp(raw.replace("Z", "+00:00")).tz_convert("UTC")


def build(platform: str = "youtube", cohort_end: pd.Timestamp | None = None,
          cohort: str = "all") -> AnalysisFrame:
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

    # `cohort` is the ordinary way to select; `cohort_end` stays as a raw
    # override for sensitivity analyses that need a different boundary.
    cohort = (cohort or "all").upper()
    if cohort not in ("A", "B", "ALL"):
        raise ValueError(f"cohort must be A, B or all — got {cohort!r}")
    if cohort in ("A", "B") and cohort_end is None:
        cohort_end = freeze_instant()

    snaps = snaps[snaps["platform"] == platform]
    ids = set(snaps["post_id"].unique())
    att["observed on this platform"] = len(ids)

    # --- exclusion: per-creator daily cap (plan amendment 2026-08-31)
    capped = set(posts.index[posts.get("over_creator_daily_cap", False)])
    ids -= capped
    att["after creator_daily_cap"] = len(ids)

    # --- Cohort split at the pre-specified freeze instant.
    # Cohort A is the analysis set. Cohort B is the temporal holdout: posts
    # published after the boundary, evaluated exactly once, at the end, after
    # all model selection is complete.
    if cohort_end is not None and cohort != "ALL":
        after = cohort == "B"
        keep = set()
        for p in ids:
            if p not in posts.index:
                continue
            pub = posts.loc[p, "published_at"]
            if (pub >= cohort_end) if after else (pub < cohort_end):
                keep.add(p)
        att[f"cohort {cohort} "
            f"({'on/after' if after else 'before'} "
            f"{cohort_end.isoformat()})"] = len(keep)
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
            "last_observed_hours": lab.last_observed_hours,
        })
        feats.update(_saturation_fields(lab, landmark))
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

    return AnalysisFrame(frame=df, attrition=att, platform=platform,
                         cohort=cohort.lower())


def _saturation_fields(lab, landmark: float) -> dict[str, Any]:
    """Carry the robustness label through to the analysis frame.

    The frozen plan specifies a second, mechanically different outcome —
    ``t_saturation``, the time to reach 90% of a fitted asymptote — and requires
    it to be reported alongside the primary one. The label function computed it
    from the beginning; it was simply never carried past this point, so the
    robustness check the plan promises could not be run at all.

    Three things are recorded rather than one, because the raw number alone
    would be misleading:

    ``t_saturation_from_publish``
        The fitted quantity as it comes out of the curve fit, measured from
        publication.
    ``t_saturation``
        The same quantity on the landmark clock, so it is directly comparable
        with the primary outcome. Negative values are kept as-is rather than
        clipped: a post that saturated before the landmark is a real
        disagreement between the two definitions, and hiding it would defeat
        the purpose of having a second definition.
    ``saturation_beyond_window``
        True when the fitted saturation time lies past the post's last actual
        observation. Such a value is an extrapolation of the fitted curve, not
        an observation. The plan pre-specifies this label, so we keep it — but
        the count is reported, because a robustness check resting mostly on
        extrapolated values is weak evidence and the reader should be told.

    **These columns are outcomes, never features.** They are computed from the
    post's whole history, so using one as a predictor would be gross leakage.
    The feature whitelist in ``synthetic.FEATURE_COLUMNS`` is what keeps them
    out, and a test pins that.
    """
    ts = lab.t_saturation
    if ts is None or not np.isfinite(ts):
        return {"t_saturation": float("nan"),
                "t_saturation_from_publish": float("nan"),
                "saturation_beyond_window": False,
                "fitted_A": float("nan"), "fitted_k": float("nan")}
    last = lab.last_observed_hours
    return {
        "t_saturation": float(ts - landmark) if landmark else float(ts),
        "t_saturation_from_publish": float(ts),
        "saturation_beyond_window": bool(last is not None and ts > last),
        "fitted_A": float(lab.fitted_A) if lab.fitted_A is not None else float("nan"),
        "fitted_k": float(lab.fitted_k) if lab.fitted_k is not None else float("nan"),
    }


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
