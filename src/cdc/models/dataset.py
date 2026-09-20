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


def _cohort_b_close() -> pd.Timestamp | None:
    """The pre-specified Cohort B upper boundary, from settings.

    Fixed on 2026-09-20, after the Cohort A analysis was complete and before the
    holdout was opened; the holdout ledger was empty at the time. Returns None
    when unset, which leaves Cohort B open-ended — acceptable only before a
    close date has been chosen.
    """
    raw = settings()["modelling"].get("cohort_b_close_utc")
    if not raw:
        return None
    return pd.Timestamp(str(raw).replace("Z", "+00:00")).tz_convert("UTC")


def cohort_maturity(platform: str = "youtube", cohort: str = "A",
                    now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Has every post in a cohort finished its observation window yet?

    Membership is frozen by publication date; *outcomes* are not. A post inside
    its ``collection.max_track_hours`` window is still being observed, and one
    currently recorded as right-censored may yet be seen to die. Analysing a
    cohort before it has matured therefore understates deaths and overstates
    censoring, and the figures keep moving until the last member ages out.

    Measured 2026-09-20: a Cohort A run made while 416 of 852 posts were still
    inside their window reported 498 deaths, and 500 a few hours later on
    identical membership. That is what this guard exists to catch.

    Reads publication dates only. It computes no labels, so it is safe to call
    against the sealed holdout.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    window = pd.Timedelta(hours=float(settings()["collection"]["max_track_hours"]))
    posts = pd.read_parquet(path_for("silver_dir") / "posts.parquet")
    posts = posts[posts["platform"] == platform]
    pub = pd.to_datetime(posts["published_at"], utc=True, format="mixed")

    cohort = (cohort or "all").upper()
    if cohort == "A":
        sel = pub < freeze_instant()
    elif cohort == "B":
        close = _cohort_b_close()
        sel = pub >= freeze_instant()
        if close is not None:
            sel &= pub < close
    else:
        sel = pd.Series(True, index=pub.index)

    pub = pub[sel]
    if pub.empty:
        return {"cohort": cohort, "n": 0, "immature": 0, "mature": True,
                "matures_at": None}
    matures_at = pub.max() + window
    immature = int(((pub + window) > now).sum())
    return {"cohort": cohort, "n": int(len(pub)), "immature": immature,
            "mature": immature == 0,
            "matures_at": matures_at.isoformat()}


def build(platform: str = "youtube", cohort_end: pd.Timestamp | None = None,
          cohort: str = "all", landmark: float | None = None,
          threshold: float | None = None) -> AnalysisFrame:
    """Features joined to labels, with the pre-specified exclusions applied.

    Landmarked: only posts still alive at ``modelling.landmark_hours`` are
    eligible, and the outcome is the time remaining from the landmark. Both the
    landmarked outcome (``t_death``) and the raw one (``t_death_from_publish``)
    are kept, so a sensitivity analysis needs no re-derivation.

    ``landmark`` and ``threshold`` override the configured values for one run.
    They exist for the pre-registered sensitivity analyses (plan section 6) and
    for nothing else: the committed values in ``settings.yaml`` remain the
    pre-specified ones, and a sensitivity run never overwrites them.
    """
    sd = path_for("silver_dir")
    snaps = pd.read_parquet(sd / "snapshots.parquet")
    posts = pd.read_parquet(sd / "posts.parquet").set_index("post_id")

    cfg = settings()["modelling"]
    landmark = (float(cfg.get("landmark_hours", 0) or 0) if landmark is None
                else float(landmark))

    # Features may never be drawn from after the moment the prediction is made.
    # At the pre-registered 7h landmark the configured 7h cutoff already
    # satisfies that. But a sensitivity run at a 3h landmark must not be allowed
    # to read the 6h observation: that would be leakage, and it would make the
    # earlier landmark look better precisely because it is cheating. Capping at
    # the landmark is what keeps the comparison honest.
    cutoff = min(float(cfg["feature_cutoff_hours"]), landmark or float("inf"))

    label_cfg = None
    if threshold is not None:
        label_cfg = dict(settings()["labels"])
        label_cfg["velocity_frac_of_peak"] = float(threshold)
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
        # Cohort B is bounded at BOTH ends. Its lower edge is the Cohort A
        # freeze; its upper edge is `cohort_b_close_utc`, fixed in settings.yaml
        # on 2026-09-20 while the holdout ledger was still empty. Without an
        # upper edge the holdout would silently grow every time collection ran,
        # so "evaluate it once" would have no fixed referent and a disappointing
        # result could be diluted by waiting.
        close = _cohort_b_close() if after else None
        keep = set()
        for p in ids:
            if p not in posts.index:
                continue
            pub = posts.loc[p, "published_at"]
            ok = (pub >= cohort_end) if after else (pub < cohort_end)
            if ok and close is not None and pub >= close:
                ok = False
            if ok:
                keep.add(p)
        label = (f"cohort {cohort} (on/after {cohort_end.isoformat()}"
                 + (f", before {close.isoformat()}" if close is not None else "")
                 + ")") if after else \
                f"cohort {cohort} (before {cohort_end.isoformat()})"
        att[label] = len(keep)
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
        lab = label_post(str(pid), obs, cfg=label_cfg)
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
