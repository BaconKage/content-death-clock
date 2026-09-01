"""The models under test: Weibull AFT and a tree-based comparator.

**Weibull AFT** is the primary model. Accelerated failure time is the right
family here because the quantity we care about is a *time*, and AFT models it
directly: coefficients are multiplicative effects on survival time, so a
coefficient of 1.3 on early growth means "posts still climbing at 3-6h live
e^1.3 times longer, holding the rest fixed." That is a sentence a reader can
check against intuition, which matters more in a paper than a fractional gain in
score.

**Random Survival Forest** is the second primary model, and the frozen plan
names it as such. It is censoring-aware, makes no proportional-hazards or
parametric-shape assumption, and finds interactions the AFT model cannot. Where
Weibull AFT buys interpretable coefficients at the cost of a rigid functional
form, the forest buys flexibility at the cost of interpretability, so the pair
covers the two failure modes between them.

An earlier version of this module substituted a gradient-boosted comparator on
the grounds that ``scikit-survival`` "needs compiled dependencies that are
awkward on Windows". That was simply wrong: a prebuilt ``cp310`` Windows wheel
exists and installs in seconds against the versions already pinned here. The
substitution was an undeclared deviation from the plan — the plan promises a
censoring-aware forest and a non-censoring-aware booster is not one — so the
forest is now implemented as specified.

**Gradient-boosted comparator.** Retained, but only as what the plan calls it: a
comparator on the uncensored subset. It is *not* censoring-aware, it is trained
only on posts observed to die, and its numbers carry that selection bias. The
paper reports it for contrast, never as a headline.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cdc.models.baselines import SurvivalModel


class WeibullAFT(SurvivalModel):
    """Weibull accelerated failure time over the full feature set."""

    name = "weibull_aft"

    def __init__(self, features: list[str], penalizer: float = 0.05,
                 l1_ratio: float = 0.0) -> None:
        self.features = list(features)
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.model_: Any = None
        self.used_features_: list[str] = []
        self.fallback_: float | None = None
        self.medians_: dict[str, float] = {}

    def _design(self, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = {}
        for c in cols:
            v = X[c].astype(float) if c in X.columns else pd.Series(np.nan, index=X.index)
            fill = self.medians_.get(c, 0.0)
            out[c] = v.fillna(fill).to_numpy(float)
        return pd.DataFrame(out, index=X.index)

    def fit(self, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray):
        from lifelines import WeibullAFTFitter

        # Keep only features that exist and actually vary in this fold. A
        # constant column makes the fit singular; dropping it is better than
        # failing, and which columns survived is recorded for the paper.
        usable = []
        for c in self.features:
            if c not in X.columns:
                continue
            col = X[c].astype(float)
            if col.notna().sum() >= 10 and col.nunique(dropna=True) >= 2:
                usable.append(c)
                self.medians_[c] = float(col.median())
        self.used_features_ = usable

        if not usable:
            self.model_ = None
            self.fallback_ = float(np.median(durations))
            return self

        df = self._design(X, usable)
        df["T"] = np.asarray(durations, float)
        df["E"] = np.asarray(events).astype(int)

        # A penalizer is needed: features here are strongly collinear by
        # construction (value_at_3h contains value_at_1h), and an unpenalised
        # fit on a small fold does not converge.
        try:
            m = WeibullAFTFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
            m.fit(df, duration_col="T", event_col="E")
            self.model_ = m
        except Exception:
            self.model_ = None
            self.fallback_ = float(np.median(durations))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            return np.full(len(X), self.fallback_ or 48.0, dtype=float)
        return self.model_.predict_median(self._design(X, self.used_features_)).to_numpy(float)

    def coefficients(self) -> pd.Series | None:
        """Fitted AFT coefficients, for the paper's interpretation section."""
        if self.model_ is None:
            return None
        return self.model_.params_.get("lambda_", None)


class RandomSurvivalForest(SurvivalModel):
    """Random survival forest, censoring-aware, from ``scikit-survival``.

    **What it predicts, and why that needed a decision.** The shared interface
    requires a predicted *time in hours*, higher meaning longer life. A forest
    natively produces a survival curve per post, not a scalar, so one has to be
    summarised — and the obvious summary does not work here.

    The obvious summary is the median: the first time at which the predicted
    survival curve crosses 0.5. With ~88% of our posts censored, most predicted
    curves never reach 0.5 inside the observation window, so the median is
    undefined for most of the sample. Software conventionally reports the
    horizon in that case, which would collapse most posts onto one value and
    destroy exactly the ordering the C-index measures.

    We therefore use **restricted mean survival time**: the area under the
    predicted survival curve, from 0 to the last time observed in the training
    fold. RMST is always defined, is measured in hours, orders posts the same
    way risk does, and is a standard quantity rather than something invented
    here.

    **The cost of that choice, stated plainly.** RMST is *restricted* — it is
    truncated at the training horizon and therefore systematically
    under-estimates the duration of long-lived posts. This does not affect the
    C-index, which depends only on ordering and is the plan's primary metric,
    but it does bias this model's MAE downward relative to Weibull AFT, which
    extrapolates freely. **The two models' MAE figures are not directly
    comparable and the paper must say so.**

    We could remove the truncation by fitting an exponential tail beyond the
    horizon. We deliberately do not. Everywhere else in this codebase we refuse
    to extrapolate past the data — ``interpolate_at`` returns missing rather
    than guessing — and buying a prettier error metric with an assumption we
    forbid elsewhere would be inconsistent.

    The horizon is taken from the **training** fold only, so nothing about the
    test fold leaks into the prediction scale.
    """

    name = "random_survival_forest"

    # Below this many observed deaths a forest cannot split meaningfully and
    # scikit-survival's own fit becomes unstable. Falling back to a constant is
    # honest; a forest fitted on three deaths is not.
    MIN_EVENTS = 5

    def __init__(self, features: list[str], n_estimators: int = 300,
                 min_samples_leaf: int = 5, max_features: Any = "sqrt",
                 random_state: int = 20260830) -> None:
        self.features = list(features)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.model_: Any = None
        self.used_features_: list[str] = []
        self.medians_: dict[str, float] = {}
        self.horizon_: float | None = None
        self.fallback_: float | None = None

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.used_features_:
            v = X[c].astype(float) if c in X.columns else pd.Series(np.nan, index=X.index)
            cols.append(v.fillna(self.medians_.get(c, 0.0)).to_numpy(float))
        return np.column_stack(cols) if cols else np.zeros((len(X), 1))

    def fit(self, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray):
        from sksurv.ensemble import RandomSurvivalForest as _SkRSF
        from sksurv.util import Surv

        durations = np.asarray(durations, float)
        events = np.asarray(events).astype(bool)

        # Same rule as the AFT model: keep features that exist and vary in this
        # fold. A constant column is noise to a tree and a singularity to AFT,
        # and using one rule for both keeps the comparison like-for-like.
        usable = []
        for c in self.features:
            if c not in X.columns:
                continue
            col = X[c].astype(float)
            if col.notna().sum() >= 10 and col.nunique(dropna=True) >= 2:
                usable.append(c)
                self.medians_[c] = float(col.median())
        self.used_features_ = usable

        if not usable or int(events.sum()) < self.MIN_EVENTS:
            self.model_ = None
            self.fallback_ = float(np.median(durations))
            return self

        try:
            m = _SkRSF(n_estimators=self.n_estimators,
                       min_samples_leaf=self.min_samples_leaf,
                       max_features=self.max_features,
                       random_state=self.random_state,
                       n_jobs=-1)
            m.fit(self._design(X), Surv.from_arrays(event=events, time=durations))
            self.model_ = m
            self.horizon_ = float(durations.max())
        except Exception:
            self.model_ = None
            self.fallback_ = float(np.median(durations))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            return np.full(len(X), self.fallback_ or 48.0, dtype=float)

        times = np.asarray(self.model_.unique_times_, float)
        surv = np.asarray(
            self.model_.predict_survival_function(self._design(X), return_array=True),
            dtype=float)
        return _rmst(times, surv)


def _rmst(times: np.ndarray, surv: np.ndarray) -> np.ndarray:
    """Restricted mean survival time: the area under a step survival curve.

    ``times`` is ascending and ``surv`` is (n_samples, n_times). S is a
    right-continuous step function: it equals 1 before the first time, then
    holds ``surv[:, i]`` over ``[times[i], times[i+1])``. The area is therefore
    the first segment at S=1 plus one rectangle per subsequent step. Truncated
    at ``times[-1]`` by construction — that truncation is the "restricted" in
    the name, and is discussed in the caller's docstring.
    """
    if times.size == 0:
        return np.zeros(surv.shape[0], dtype=float)
    area = np.full(surv.shape[0], float(times[0]))       # S = 1 up to first time
    if times.size > 1:
        widths = np.diff(times)
        area = area + (surv[:, :-1] * widths).sum(axis=1)
    return area


class GradientBoostedUncensored(SurvivalModel):
    """Gradient boosting on log-time, trained on observed deaths only.

    **Not censoring-aware.** Reported as a contrast, and the paper states that
    its training set excludes surviving posts and therefore over-represents
    short-lived content.
    """

    name = "gbm_uncensored"

    def __init__(self, features: list[str], random_state: int = 20260830) -> None:
        self.features = list(features)
        self.random_state = random_state
        self.model_: Any = None
        self.medians_: dict[str, float] = {}
        self.fallback_: float | None = None

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.features:
            v = X[c].astype(float) if c in X.columns else pd.Series(np.nan, index=X.index)
            cols.append(v.fillna(self.medians_.get(c, 0.0)).to_numpy(float))
        return np.column_stack(cols) if cols else np.zeros((len(X), 1))

    def fit(self, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray):
        from sklearn.ensemble import HistGradientBoostingRegressor

        for c in self.features:
            if c in X.columns:
                self.medians_[c] = float(X[c].astype(float).median())

        mask = np.asarray(events).astype(bool)
        if mask.sum() < 20:
            self.model_ = None
            self.fallback_ = float(np.median(durations))
            return self

        y = np.log10(np.asarray(durations, float)[mask])
        self.model_ = HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.06,
            random_state=self.random_state)
        self.model_.fit(self._design(X.loc[mask]), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            return np.full(len(X), self.fallback_ or 48.0, dtype=float)
        return np.power(10.0, self.model_.predict(self._design(X)))
