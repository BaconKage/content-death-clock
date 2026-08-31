"""Turn one post's observations into everything needed to draw its decay curve.

Shared deliberately between the demo app and the papers' figures. If the demo
drew a curve one way and the paper drew it another, the demo would be an
illustration of something we did not do. Same code, same numbers, both places.

Pure functions over a dataframe — no plotting, no Streamlit — so the arithmetic
can be tested without a browser or a display.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cdc.config import settings
from cdc.labels.death import Observation, label_post


@dataclass
class PostCurve:
    post_id: str
    platform: str
    creator: str
    metric: str
    ages: np.ndarray            # hours since publish, at each observation
    values: np.ndarray          # cumulative metric
    velocity_ages: np.ndarray   # right edge of each interval
    velocities: np.ndarray      # units per hour
    peak_velocity: float | None
    peak_at: float | None
    threshold: float | None     # the 5%-of-peak line
    t_death: float | None
    event_observed: bool
    last_observed: float
    exclude_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_dead(self) -> bool:
        return bool(self.event_observed)

    @property
    def status(self) -> str:
        if self.exclude_reason:
            return f"not yet labelable — {self.exclude_reason.replace('_', ' ')}"
        if self.event_observed:
            return f"attention death at {self.t_death:.1f}h after publishing"
        return f"still alive at {self.last_observed:.1f}h — no death observed yet"


def build_curve(snapshots: pd.DataFrame, post_id: str,
                posts: pd.DataFrame | None = None) -> PostCurve:
    """Assemble the curve, velocities and label for one post."""
    g = snapshots[snapshots["post_id"] == post_id].sort_values("age_hours")
    if g.empty:
        raise KeyError(f"no observations for {post_id}")

    platform = str(g["platform"].iloc[0]) if "platform" in g else "unknown"
    creator = str(g["creator_handle"].iloc[0]) if "creator_handle" in g and \
        pd.notna(g["creator_handle"].iloc[0]) else str(g.get("creator_id", pd.Series(["?"])).iloc[0])
    metric_cfg = settings()["labels"]["primary_metric"]
    metric = (metric_cfg.get(platform, "views") if isinstance(metric_cfg, dict)
              else metric_cfg)

    valid = g[g["primary_value"].notna()]
    ages = valid["age_hours"].to_numpy(float)
    values = valid["primary_value"].to_numpy(float)

    label = label_post(post_id, [Observation(a, v) for a, v in zip(ages, values)])

    # Velocities on the same definition the label uses, so the plotted line and
    # the reported death time can never disagree.
    vel_ages, vels = [], []
    for (a0, v0), (a1, v1) in zip(zip(ages, values), zip(ages[1:], values[1:])):
        dt = a1 - a0
        if dt > 0:
            vel_ages.append(a1)
            vels.append((v1 - v0) / dt)

    notes = []
    if label.monotonicity_repairs:
        notes.append(f"{label.monotonicity_repairs} retracted-count correction(s) "
                     "— the platform removed views it judged inauthentic")
    if len(ages) and ages[0] > 1.0:
        notes.append(f"first observed at {ages[0]:.1f}h, so the 1-hour feature "
                     "is unavailable for this post")

    return PostCurve(
        post_id=post_id, platform=platform, creator=creator, metric=metric,
        ages=ages, values=values,
        velocity_ages=np.array(vel_ages), velocities=np.array(vels),
        peak_velocity=label.peak_velocity, peak_at=label.peak_at_hours,
        threshold=(label.peak_velocity * float(settings()["labels"]["velocity_frac_of_peak"])
                   if label.peak_velocity else None),
        t_death=label.t_death, event_observed=bool(label.event_observed),
        last_observed=float(ages[-1]) if len(ages) else 0.0,
        exclude_reason=label.exclude_reason, notes=notes,
    )


def countdown_text(curve: PostCurve, now_hours: float | None = None) -> str:
    """The demo's headline line.

    Deliberately refuses to invent a countdown for a post whose death has not
    been observed and for which no model prediction was supplied. A confident
    number with nothing behind it is the exact failure the whole project is
    arguing against.
    """
    if curve.exclude_reason:
        return "Not enough observations yet"
    if curve.event_observed:
        return f"Died {curve.t_death:.1f}h after publishing"
    age = now_hours if now_hours is not None else curve.last_observed
    return f"Still alive at {age:.1f}h — no prediction available yet"


def summarise(snapshots: pd.DataFrame) -> pd.DataFrame:
    """One row per post: what it is and whether it can be labelled yet."""
    rows = []
    for pid, g in snapshots.groupby("post_id"):
        v = g[g["primary_value"].notna()].sort_values("age_hours")
        if v.empty:
            continue
        obs = [Observation(a, x) for a, x in
               zip(v["age_hours"].to_numpy(float), v["primary_value"].to_numpy(float))]
        lab = label_post(str(pid), obs)
        rows.append({
            "post_id": pid,
            "platform": g["platform"].iloc[0] if "platform" in g else "?",
            "n_obs": len(v),
            "age_hours": float(v["age_hours"].max()),
            "final_value": float(v["primary_value"].iloc[-1]),
            "labelable": lab.exclude_reason is None,
            "died": bool(lab.event_observed),
            "t_death": lab.t_death,
        })
    return pd.DataFrame(rows).sort_values("age_hours", ascending=False)
