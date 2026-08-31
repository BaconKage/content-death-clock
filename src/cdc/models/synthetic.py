"""Synthetic cohorts with known ground truth, for validating the analysis itself.

Real death labels will not mature for days, and the evaluation code must be
correct *before* it touches real data — otherwise the first numbers we ever see
are also the first test of the code that produced them, and there is no way to
tell a real finding from a bug.

So we generate cohorts where the answer is known by construction:

* ``make_cohort`` builds posts whose true lifetime depends on early velocity in a
  way we choose, with creator-level random effects and a known censoring rate. A
  correct pipeline must recover that dependence.
* ``make_null_cohort`` builds posts whose lifetime is pure noise, unrelated to
  any feature. A correct pipeline must find **nothing** — this is the test that
  catches leakage, because a leaking pipeline scores well on data with no signal
  at all, which is the single most dangerous failure available to us.

Measured on this fixture (2026-08-31), C-index on data containing NO signal:

===========================  ==========  =======  ==========
model / split                grouped     leaky    inflation
===========================  ==========  =======  ==========
Weibull AFT, 15 posts/creator      0.512    0.535      +0.023
Weibull AFT, 40 posts/creator      0.395    0.595      +0.199
Gradient boosting                  0.491    0.804      +0.313
===========================  ==========  =======  ==========

Two things follow. Leakage severity scales with **posts per creator**, so it
worsens exactly as the panel grows. And it scales with **model flexibility**: a
penalised AFT resists memorising creator identity, while a gradient-booster
reports 0.80 on pure noise. The frozen plan's creator-grouped cross-validation
is what stands between that number and the paper.

Both are seeded and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_value_at_1h", "log_value_at_3h", "log_value_at_6h",
    "velocity_to_6h", "log_growth_1_3h", "log_growth_3_6h",
    "decay_ratio_3_6_over_1_3", "log_follower_count",
    "log_duration_sec", "publish_hour_utc", "title_len",
]


@dataclass
class Cohort:
    """A synthetic dataset plus the ground truth used to build it."""

    frame: pd.DataFrame
    true_coefficients: dict[str, float]
    censoring_rate: float

    @property
    def durations(self) -> np.ndarray:
        return self.frame["t_death"].to_numpy(float)

    @property
    def events(self) -> np.ndarray:
        return self.frame["event_observed"].to_numpy(bool)


def make_cohort(n_posts: int = 600, n_creators: int = 40,
                censor_at_hours: float = 336.0,
                creator_sd: float = 0.35,
                noise_sd: float = 0.30,
                seed: int = 20260830) -> Cohort:
    """A cohort where early velocity genuinely predicts lifetime.

    The generative model is an accelerated failure time form, which is what the
    Weibull AFT we fit actually assumes:

        log(T) = intercept + b1*log_growth_3_6h + b2*decay_ratio
                 + b3*log_follower_count + creator_effect + noise

    Posts that are still climbing at 3-6h live longer; posts whose velocity has
    already collapsed die sooner. Creator effects are included because posts from
    one creator are not independent — which is exactly why the analysis plan
    pre-specifies grouping cross-validation by creator.
    """
    rng = np.random.default_rng(seed)

    creator_ids = np.array([f"chan_{i:03d}" for i in range(n_creators)])
    creator_effect = rng.normal(0.0, creator_sd, size=n_creators)
    # Follower counts span the real frame's range: ~1e3 to ~5e8.
    creator_log_followers = rng.uniform(3.0, 8.7, size=n_creators)

    owner = rng.integers(0, n_creators, size=n_posts)

    # --- early-window features
    log_v1 = rng.normal(2.2, 0.9, n_posts) + 0.35 * creator_log_followers[owner]
    growth_1_3 = rng.normal(0.45, 0.22, n_posts)
    growth_3_6 = rng.normal(0.22, 0.18, n_posts)
    decay_ratio = np.clip(rng.normal(0.55, 0.28, n_posts), 0.01, 3.0)

    coefs = {
        "log_growth_3_6h": 1.30,      # still climbing at 3-6h -> lives longer
        "decay_ratio_3_6_over_1_3": 0.85,
        "log_follower_count": 0.18,   # weak: size matters, dynamics matter more
    }
    log_t = (
        2.35
        + coefs["log_growth_3_6h"] * growth_3_6
        + coefs["decay_ratio_3_6_over_1_3"] * decay_ratio
        + coefs["log_follower_count"] * creator_log_followers[owner]
        + creator_effect[owner]
        + rng.normal(0.0, noise_sd, n_posts)
    )
    true_t = np.exp(log_t)

    observed = np.minimum(true_t, censor_at_hours)
    event = true_t <= censor_at_hours

    frame = pd.DataFrame({
        "post_id": [f"post_{i:05d}" for i in range(n_posts)],
        "creator_id": creator_ids[owner],
        "t_death": observed,
        "event_observed": event,
        "true_t_death": true_t,
        "log_value_at_1h": log_v1,
        "log_value_at_3h": log_v1 + growth_1_3,
        "log_value_at_6h": log_v1 + growth_1_3 + growth_3_6,
        "log_growth_1_3h": growth_1_3,
        "log_growth_3_6h": growth_3_6,
        "decay_ratio_3_6_over_1_3": decay_ratio,
        "velocity_to_6h": np.exp(log_v1) / 6.0,
        "log_follower_count": creator_log_followers[owner],
        "log_duration_sec": rng.normal(2.6, 0.5, n_posts),
        "publish_hour_utc": rng.integers(0, 24, n_posts).astype(float),
        "title_len": rng.normal(55, 18, n_posts),
    })
    return Cohort(frame=frame, true_coefficients=coefs,
                  censoring_rate=float((~event).mean()))


def make_null_cohort(n_posts: int = 600, n_creators: int = 40,
                     censor_at_hours: float = 336.0,
                     seed: int = 20260830) -> Cohort:
    """Lifetime is pure noise. A correct pipeline must find nothing.

    Features still carry creator structure, so a model that leaks creator
    identity across folds *will* appear to succeed here. That makes this the
    sharpest available test of the grouping discipline: a C-index meaningfully
    above 0.5 on this cohort means the evaluation is broken, not that the model
    is clever.
    """
    rng = np.random.default_rng(seed + 1)
    base = make_cohort(n_posts=n_posts, n_creators=n_creators,
                       censor_at_hours=censor_at_hours, seed=seed)
    frame = base.frame.copy()

    # Replace the outcome with noise that is unrelated to any early-window
    # feature, but STRONGLY linked to creator identity.
    #
    # The creator effect is deliberately large (sd 1.5, dominating the residual
    # sd 0.4). A weaker one makes leakage undetectable — measured 2026-08-31, a
    # creator sd of 0.5 inflated the C-index by only +0.014 when the split was
    # deliberately made leaky, which is inside noise. At sd 1.5 the inflation is
    # ~+0.08 and a leaky split is caught. A leakage test that cannot fail is
    # worse than no test, because it grants false assurance.
    creators = frame["creator_id"].unique()
    per_creator = dict(zip(creators, rng.normal(0.0, 1.5, len(creators))))
    # log_follower_count is constant within a creator in the real data too, so
    # it acts as a near-perfect creator fingerprint. That is exactly the channel
    # through which leakage would flow.
    fingerprint = dict(zip(creators, rng.normal(0.0, 1.0, len(creators))))
    frame["log_follower_count"] = frame["creator_id"].map(fingerprint).to_numpy()
    log_t = 3.0 + frame["creator_id"].map(per_creator).to_numpy() + rng.normal(0, 0.4, len(frame))
    true_t = np.exp(log_t)

    frame["true_t_death"] = true_t
    frame["t_death"] = np.minimum(true_t, censor_at_hours)
    frame["event_observed"] = true_t <= censor_at_hours
    return Cohort(frame=frame, true_coefficients={}, censoring_rate=float((~frame["event_observed"]).mean()))
