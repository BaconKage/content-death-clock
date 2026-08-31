"""The models under test: Weibull AFT and a tree-based comparator.

**Weibull AFT** is the primary model. Accelerated failure time is the right
family here because the quantity we care about is a *time*, and AFT models it
directly: coefficients are multiplicative effects on survival time, so a
coefficient of 1.3 on early growth means "posts still climbing at 3-6h live
e^1.3 times longer, holding the rest fixed." That is a sentence a reader can
check against intuition, which matters more in a paper than a fractional gain in
score.

**Gradient-boosted comparator.** ``scikit-survival`` provides a proper random
survival forest, but it needs compiled dependencies that are awkward on Windows,
so the plan permits a comparator on the uncensored subset instead. That
comparator is *not* censoring-aware, and the paper must say so: it is trained
only on posts observed to die, and its numbers carry that selection bias. It is
reported for contrast, never as the headline.
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
