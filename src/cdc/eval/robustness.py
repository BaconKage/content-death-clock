"""Does the conclusion survive a different definition of "attention death"?

The frozen plan commits to two outcomes. ``t_death`` is the primary one:
velocity falls below 5% of the post's own peak and stays there. ``t_saturation``
is the robustness one: time to reach 90% of the asymptote A from a fitted
C(t) = A(1 - e^(-kt)).

The two are worth having precisely because they fail differently. The velocity
label is non-parametric and local — it reads the shape of the curve where it
flattens, and is sensitive to a single noisy interval. The saturation label is
parametric and global — it fits one functional form to the whole series, and is
sensitive to that form being wrong. If both say the same thing, the finding is
not an artefact of either. If they disagree systematically, the plan says that
**is** the finding and gets reported.

This module answers two questions:

1. **Do the labels agree?** Coverage, rank correlation, and the size and
   direction of the typical disagreement.
2. **Does the model's verdict change?** The whole grouped-CV evaluation, re-run
   with saturation as the outcome, so H1 and H2 can be checked against a
   definition that shares none of the primary one's failure modes.

**Censoring under the saturation label, stated explicitly.** A curve fit that
fails is not a missing value to be dropped — it is a post whose asymptote is not
yet identifiable, which is to say a post that has not saturated. It is therefore
treated as **censored at its last observation**, exactly as an un-died post is
under the primary label. Dropping those posts would bias the sample toward
fast-saturating content, which is the same mistake censoring exists to prevent.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Below this many posts with both labels, the agreement statistics are
# arithmetic rather than evidence and are printed with a warning.
MIN_COMPARABLE = 20


def label_agreement(df: pd.DataFrame) -> dict[str, Any]:
    """Compare the two outcome definitions on the posts that have both.

    Restricted to posts where death was actually *observed*: a censored post's
    ``t_death`` is a censoring time, not a duration, and correlating it against
    a fitted saturation time would be comparing two different quantities.
    """
    out: dict[str, Any] = {"n_posts": int(len(df))}
    if df.empty or "t_saturation" not in df.columns:
        return out | {"status": "no saturation labels in frame"}

    sat = pd.to_numeric(df["t_saturation"], errors="coerce")
    out["saturation_fitted"] = int(sat.notna().sum())
    out["saturation_coverage"] = float(sat.notna().mean())
    out["saturation_beyond_window"] = int(
        df.get("saturation_beyond_window", pd.Series(False, index=df.index)).sum())

    # How often the two definitions disagree about whether the post was even
    # still alive at the landmark. This is the headline disagreement, and it is
    # measured over every fitted post rather than the handful that also died.
    from cdc.config import settings
    landmark = float(settings()["modelling"].get("landmark_hours", 0) or 0)
    if landmark and sat.notna().any():
        out["saturated_before_landmark"] = int((sat < 0).sum())
        out["frac_saturated_before_landmark"] = float((sat < 0).mean())

    both = df[sat.notna() & df["event_observed"].astype(bool)].copy()
    out["n_comparable"] = int(len(both))
    if len(both) < 3:
        return out | {"status": "too few posts with both labels to compare"}

    d = pd.to_numeric(both["t_death"], errors="coerce").to_numpy(float)
    s = pd.to_numeric(both["t_saturation"], errors="coerce").to_numpy(float)
    ok = np.isfinite(d) & np.isfinite(s)
    d, s = d[ok], s[ok]

    from scipy.stats import spearmanr
    rho, p = spearmanr(d, s)
    out["spearman_rho"] = float(rho)
    out["spearman_p"] = float(p)

    # Ratio rather than difference: these are durations spanning orders of
    # magnitude, so "saturation lands 6 hours later" means something very
    # different for a 2-hour post than a 200-hour one.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d > 0, s / d, np.nan)
    finite = ratio[np.isfinite(ratio)]
    if finite.size:
        out["median_ratio_sat_over_death"] = float(np.median(finite))
        out["iqr_ratio"] = [float(np.percentile(finite, 25)),
                            float(np.percentile(finite, 75))]
    # A rank correlation on a handful of points is not evidence, and this one
    # will be quoted in a paper, so it says so about itself.
    out["comparison_underpowered"] = bool(len(d) < MIN_COMPARABLE)
    out["status"] = "ok"
    return out


def saturation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Re-express the analysis frame with saturation as the outcome.

    ``t_death`` and ``event_observed`` are replaced so the existing evaluation
    machinery can run unchanged. A fitted saturation time is an event; a failed
    fit is censored at the last observation (see the module docstring).
    """
    out = df.copy()
    sat = pd.to_numeric(out["t_saturation"], errors="coerce")
    fitted = sat.notna() & np.isfinite(sat)

    # Censoring time on the landmark clock, matching how t_death is stored.
    from cdc.config import settings
    landmark = float(settings()["modelling"].get("landmark_hours", 0) or 0)
    last = pd.to_numeric(out.get("last_observed_hours"), errors="coerce") - landmark

    out["t_death"] = np.where(fitted, sat, last)
    out["event_observed"] = fitted.to_numpy(bool)

    # A duration must be positive for every survival model here. A post that
    # saturated at or before the landmark has no remaining lifetime to predict
    # and is dropped, mirroring the primary label's "died before the landmark"
    # exclusion rather than inventing a floor for it.
    keep = pd.to_numeric(out["t_death"], errors="coerce") > 0
    return out[keep.fillna(False)].reset_index(drop=True)


def print_agreement(a: dict[str, Any]) -> None:
    print("\n" + "-" * 68)
    print("  ROBUSTNESS — primary label vs saturation label")
    print("-" * 68)
    if a.get("status") != "ok":
        print(f"  {a.get('status', 'unavailable')}")
        cov = a.get("saturation_coverage")
        if cov is not None:
            print(f"  saturation fitted for {a['saturation_fitted']} of "
                  f"{a['n_posts']} posts ({cov:.0%})")
        return
    print(f"  saturation fitted        {a['saturation_fitted']}/{a['n_posts']} "
          f"({a['saturation_coverage']:.0%})")
    print(f"  fit beyond last obs.     {a['saturation_beyond_window']} "
          f"(extrapolated — weak evidence)")
    if "saturated_before_landmark" in a:
        print(f"  saturated pre-landmark   {a['saturated_before_landmark']} "
              f"({a['frac_saturated_before_landmark']:.0%}) — the two labels "
              f"disagree that these were alive")
    print(f"  comparable (died + fit)  {a['n_comparable']}")
    print(f"  Spearman rho             {a['spearman_rho']:+.3f}  "
          f"(p={a['spearman_p']:.4f})")
    if "median_ratio_sat_over_death" in a:
        lo, hi = a["iqr_ratio"]
        print(f"  median t_sat / t_death   {a['median_ratio_sat_over_death']:.2f}  "
              f"[IQR {lo:.2f}, {hi:.2f}]")
        print("  (>1 = saturation after velocity collapse; <0 = saturation "
              "before the landmark)")
    if a.get("comparison_underpowered"):
        print(f"  *** fewer than {MIN_COMPARABLE} comparable posts — these "
              f"agreement statistics are NOT interpretable yet. ***")
