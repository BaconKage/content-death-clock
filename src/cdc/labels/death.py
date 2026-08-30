"""Operationalising "attention death".

This module defines the dependent variable of the entire study. Everything the
RM paper claims rests on it being correct, so it is deliberately small, pure
(no I/O), fully configurable from ``config/settings.yaml``, and covered by tests
that check against analytically known answers.

Two labels are produced for every post:

``t_death`` (primary)
    The first time engagement velocity falls below ``velocity_frac_of_peak`` of
    that post's own peak velocity and *stays* below for ``sustain_intervals``
    consecutive intervals. Normalising against the post's own peak rather than
    an absolute threshold is what makes a 300-view video and a 3M-view video
    comparable.

``t_saturation`` (robustness)
    Time to reach ``saturation_target`` of the asymptote A from a fitted
    saturating-growth curve C(t) = A(1 - e^(-kt)). Reported alongside the
    primary label. If the two disagree systematically, that is a finding worth
    reporting, not a bug to hide.

**Censoring.** A post whose velocity never drops below threshold within the
observation window has not been observed to die. Its ``event_observed`` is
False and ``t_death`` holds the *censoring time* (last observation age). These
rows must be kept and handed to a survival model. Dropping them would bias the
sample toward short-lived content and inflate every accuracy number we report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Sequence

from cdc.config import settings

# Reason codes for a post that cannot be labelled. Recorded, counted, and
# reported in the paper's sample-attrition table — never silently dropped.
INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
NO_POSITIVE_VELOCITY = "no_positive_velocity"
NONE = None


@dataclass(frozen=True)
class Observation:
    age_hours: float
    value: float


@dataclass
class DeathLabel:
    post_id: str
    t_death: float | None            # hours since publish; censoring time if not observed
    event_observed: bool             # True = death seen, False = right-censored
    peak_velocity: float | None      # units/hour
    peak_at_hours: float | None
    n_observations: int
    n_intervals: int
    last_observed_hours: float | None
    final_value: float | None
    monotonicity_repairs: int        # counter decreases we clamped — data quality
    t_saturation: float | None = None
    fitted_A: float | None = None
    fitted_k: float | None = None
    exclude_reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.exclude_reason is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
def _prepare(observations: Sequence[Observation]) -> tuple[list[Observation], int]:
    """Sort by age, drop duplicate ages, enforce cumulative monotonicity.

    Cumulative view counts occasionally *decrease* — platforms retroactively
    remove views they judge inauthentic. A raw decrease would produce a negative
    velocity and could masquerade as death. We clamp to the running maximum and
    count how often we had to, so data quality is a reported number rather than
    an invisible assumption.
    """
    clean = sorted((o for o in observations if o.value is not None
                    and o.age_hours is not None), key=lambda o: o.age_hours)
    # Collapse duplicate timestamps, keeping the last (a re-run's fresher read).
    dedup: list[Observation] = []
    for o in clean:
        if dedup and math.isclose(dedup[-1].age_hours, o.age_hours, abs_tol=1e-9):
            dedup[-1] = o
        else:
            dedup.append(o)

    repairs = 0
    out: list[Observation] = []
    running = -math.inf
    for o in dedup:
        if o.value < running:
            repairs += 1
            out.append(Observation(o.age_hours, running))
        else:
            running = o.value
            out.append(o)
    return out, repairs


def velocities(obs: Sequence[Observation]) -> list[tuple[float, float]]:
    """[(age_at_interval_end, units_per_hour), ...] for consecutive intervals."""
    out = []
    for a, b in zip(obs, obs[1:]):
        dt = b.age_hours - a.age_hours
        if dt <= 0:
            continue
        out.append((b.age_hours, (b.value - a.value) / dt))
    return out


def label_post(post_id: str, observations: Sequence[Observation],
               cfg: dict[str, Any] | None = None) -> DeathLabel:
    """Compute the death label for one post's observation series."""
    cfg = cfg or settings()["labels"]
    frac = float(cfg["velocity_frac_of_peak"])
    sustain = int(cfg["sustain_intervals"])
    min_intervals = int(cfg["min_intervals_for_label"])
    skip_first = bool(cfg.get("peak_skip_first_interval", False))

    obs, repairs = _prepare(observations)
    base = dict(post_id=post_id, n_observations=len(obs),
                monotonicity_repairs=repairs,
                last_observed_hours=obs[-1].age_hours if obs else None,
                final_value=obs[-1].value if obs else None)

    vel = velocities(obs)
    if len(vel) < min_intervals:
        return DeathLabel(t_death=None, event_observed=False, peak_velocity=None,
                          peak_at_hours=None, n_intervals=len(vel),
                          exclude_reason=INSUFFICIENT_OBSERVATIONS, **base)

    peak_pool = vel[1:] if (skip_first and len(vel) > 1) else vel
    peak_at, peak_v = max(peak_pool, key=lambda tv: tv[1])
    if peak_v <= 0:
        # Nothing ever grew — a private, blocked, or zero-traffic post. Not a
        # death, just an absence of life; excluded and counted separately.
        return DeathLabel(t_death=None, event_observed=False, peak_velocity=peak_v,
                          peak_at_hours=peak_at, n_intervals=len(vel),
                          exclude_reason=NO_POSITIVE_VELOCITY, **base)

    threshold = frac * peak_v

    t_death, observed = None, False
    for i in range(len(vel)):
        window = vel[i:i + sustain]
        if len(window) < sustain:
            break                      # not enough remaining intervals to confirm
        if all(v < threshold for _, v in window):
            t_death, observed = vel[i][0], True
            break

    if not observed:
        t_death = obs[-1].age_hours    # right-censored at last observation

    sat = _fit_saturation(obs, float(cfg["saturation_target"]))

    return DeathLabel(t_death=t_death, event_observed=observed,
                      peak_velocity=peak_v, peak_at_hours=peak_at,
                      n_intervals=len(vel), t_saturation=sat[0],
                      fitted_A=sat[1], fitted_k=sat[2], **base)


# --------------------------------------------------------------------------- #
def _fit_saturation(obs: Sequence[Observation],
                    target: float) -> tuple[float | None, float | None, float | None]:
    """Fit C(t) = A(1 - e^(-kt)); return (time to `target` of A, A, k).

    Returns (None, None, None) when the fit fails or is not credible. A failed
    fit is not an error — some posts are still climbing and the asymptote is
    genuinely not identifiable yet. It is recorded as missing and the primary
    velocity label carries on regardless.
    """
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError:                          # scipy optional at label time
        return None, None, None

    t = np.array([o.age_hours for o in obs], dtype=float)
    c = np.array([o.value for o in obs], dtype=float)
    if len(t) < 3 or c.max() <= 0:
        return None, None, None

    def model(x, A, k):
        return A * (1.0 - np.exp(-k * x))

    # A starts a little above the last observation, k from a rough half-life
    # guess of the midpoint — bad initial guesses make curve_fit wander.
    p0 = [max(c.max() * 1.1, 1.0), 1.0 / max(t.max() / 3.0, 1e-3)]
    try:
        popt, _ = curve_fit(model, t, c, p0=p0, maxfev=10000,
                            bounds=([c.max() * 0.5, 1e-6], [c.max() * 100, 10.0]))
    except Exception:
        return None, None, None

    A, k = float(popt[0]), float(popt[1])
    if not (math.isfinite(A) and math.isfinite(k)) or k <= 0:
        return None, None, None
    # t such that A(1-e^-kt) = target*A  =>  t = -ln(1-target)/k
    return float(-math.log(1.0 - target) / k), A, k
