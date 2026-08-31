"""The four pre-registered baselines.

Every claim in the RM paper is a claim *relative to these*. They are therefore
built to be as strong as honesty allows, not as weak as convenience allows.

**The important design decision.** Baselines 3 and 4 use a covariate, and they
are fitted with the same censoring-aware machinery as the main model rather than
with ordinary regression on the uncensored subset. A censoring-naive baseline
would lose to a censoring-aware model partly *because* of the censoring
handling, and we would be unable to tell that apart from the model having learnt
anything. Handicapping the comparison would manufacture the result the paper
exists to test.

The four, from the frozen plan:

1. ``ConstantLifetime`` — "all content dies at 48 hours". The straw man, included
   because it is the folk belief the project is arguing with.
2. ``KaplanMeierMedian`` — predicts the cohort's median survival time for every
   post. Uses Kaplan-Meier rather than the raw median of observed deaths, which
   would be biased downward by censoring: posts still alive are exactly the
   long-lived ones, and dropping them shortens the apparent median.
3. ``SubscriberOnly`` — creator size and nothing else. **This is the baseline
   that matters.** Beating it is the claim that early dynamics carry information
   beyond "big channel, big numbers" (H2).
4. ``PeakVelocityHeuristic`` — a single early-velocity covariate. Tests whether
   the full feature set earns its complexity over one obvious signal.

All models share one interface::

    model.fit(X, durations, events)
    model.predict(X) -> predicted survival time in hours
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class SurvivalModel:
    """Common interface. `predict` returns predicted hours to death."""

    name = "base"

    def fit(self, X: pd.DataFrame, durations: np.ndarray,
            events: np.ndarray) -> "SurvivalModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


class ConstantLifetime(SurvivalModel):
    """Predicts a fixed lifetime for everything. The folk belief."""

    name = "constant_48h"

    def __init__(self, hours: float = 48.0) -> None:
        self.hours = float(hours)

    def fit(self, X, durations, events):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.hours, dtype=float)


class KaplanMeierMedian(SurvivalModel):
    """Predicts the cohort median survival time, estimated by Kaplan-Meier.

    Using the raw median of observed death times instead would be biased: the
    posts still alive at the end of observation are precisely the long-lived
    ones, so discarding them pulls the median down. Kaplan-Meier uses them.
    """

    name = "km_median"

    def __init__(self) -> None:
        self.median_: float | None = None

    def fit(self, X, durations, events):
        from lifelines import KaplanMeierFitter
        km = KaplanMeierFitter()
        km.fit(np.asarray(durations, float), np.asarray(events).astype(int))
        med = km.median_survival_time_
        # If over half the cohort is still alive, the median is not reached and
        # KM returns inf. Fall back to the largest observed time — honest, and
        # it keeps the baseline defined rather than propagating inf.
        self.median_ = (float(np.max(durations)) if not np.isfinite(med)
                        else float(med))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.median_ is not None, "fit() first"
        return np.full(len(X), self.median_, dtype=float)


class _SingleCovariateAFT(SurvivalModel):
    """Weibull AFT on exactly one covariate. Shared by baselines 3 and 4.

    Deliberately the same estimator family as the main model, so a difference in
    score reflects the *information in the covariates*, not a difference in how
    censoring was handled.
    """

    name = "single_covariate_aft"
    covariate = ""

    def __init__(self, covariate: str | None = None) -> None:
        if covariate:
            self.covariate = covariate
        self.model_: Any = None
        self.fallback_: float | None = None

    def fit(self, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray):
        from lifelines import WeibullAFTFitter

        col = X[self.covariate] if self.covariate in X.columns else None
        df = pd.DataFrame({
            "T": np.asarray(durations, float),
            "E": np.asarray(events).astype(int),
        })
        if col is None or col.isna().all() or col.nunique(dropna=True) < 2:
            # Nothing to learn from: degrade to a constant, recorded honestly.
            self.model_ = None
            self.fallback_ = float(np.median(durations))
            return self

        df[self.covariate] = col.fillna(col.median()).to_numpy(float)
        try:
            m = WeibullAFTFitter()
            m.fit(df, duration_col="T", event_col="E")
            self.model_ = m
        except Exception:
            self.model_ = None
            self.fallback_ = float(np.median(durations))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            return np.full(len(X), self.fallback_ or 48.0, dtype=float)
        col = X[self.covariate]
        df = pd.DataFrame({self.covariate: col.fillna(col.median()).to_numpy(float)})
        return self.model_.predict_median(df).to_numpy(float)


class SubscriberOnly(_SingleCovariateAFT):
    """Creator size alone. The baseline H2 is about."""

    name = "subscriber_only"
    covariate = "log_follower_count"


class PeakVelocityHeuristic(_SingleCovariateAFT):
    """One early-velocity signal. Tests whether the full feature set earns itself."""

    name = "peak_velocity"
    covariate = "log_value_at_6h"


def all_baselines() -> list[SurvivalModel]:
    """The four, in the order the paper reports them."""
    return [ConstantLifetime(48.0), KaplanMeierMedian(),
            SubscriberOnly(), PeakVelocityHeuristic()]
