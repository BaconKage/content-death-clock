"""Tests for the random survival forest and its RMST summary.

The plan names this model as primary, so it needs more than a smoke test. Three
properties are worth pinning:

1. **RMST is arithmetically right.** It is the one piece of maths written by
   hand rather than taken from a library, so it is tested against a case whose
   area can be computed on paper.
2. **The forest recovers real signal** on a cohort where the answer is known,
   and **finds nothing** on a cohort with no signal. A model that scores well on
   noise is worse than useless.
3. **The fallback fires rather than the fit exploding** when there are too few
   deaths, which is the regime this project is actually in right now.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdc.eval.validate import concordance, grouped_cv
from cdc.models.survival import RandomSurvivalForest, _rmst

sksurv = pytest.importorskip("sksurv", reason="scikit-survival not installed")


# --------------------------------------------------------------------- RMST
def test_rmst_matches_hand_computed_area():
    # surv[i] is S AT times[i], and holds until times[i+1]:
    #   S = 1    on [0, 2)   ->  1.00 * 2 = 2.0
    #   S = 0.5  on [2, 4)   ->  0.50 * 2 = 1.0
    #   S = 0.25 on [4, 6)   ->  0.25 * 2 = 0.5
    # The final value (0.1 at t=6) covers zero width: RMST is restricted AT
    # the last time, never beyond it.
    times = np.array([2.0, 4.0, 6.0])
    surv = np.array([[0.5, 0.25, 0.1]])
    assert _rmst(times, surv) == pytest.approx(3.5)


def test_rmst_is_monotone_in_survival():
    """A uniformly higher survival curve must give a uniformly larger RMST."""
    times = np.array([1.0, 2.0, 3.0])
    low = np.array([[0.5, 0.2, 0.1]])
    high = np.array([[0.9, 0.8, 0.7]])
    assert _rmst(times, high)[0] > _rmst(times, low)[0]


def test_rmst_handles_single_time_point():
    assert _rmst(np.array([5.0]), np.array([[0.3]])) == pytest.approx(5.0)


def test_rmst_empty_grid_is_zero_not_a_crash():
    assert _rmst(np.array([]), np.zeros((3, 0))).tolist() == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------- fitting
def _cohort(n=240, signal=True, seed=7):
    """Posts whose lifetime depends on an early feature — or does not."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    # With signal, lifetime rises with x. Without, x is unrelated to lifetime.
    base = np.exp(3.0 + (0.9 * x if signal else 0.0))
    t_true = base * rng.exponential(size=n)
    c = rng.uniform(20, 400, size=n)             # censoring times
    duration = np.minimum(t_true, c)
    event = t_true <= c
    X = pd.DataFrame({
        "log_value_at_3h": x,
        "velocity_to_6h": noise,
        "post_id": [f"p{i}" for i in range(n)],
        "creator_id": [f"c{i % 24}" for i in range(n)],
        "t_death": duration,
        "event_observed": event,
    })
    return X


def test_forest_recovers_known_signal():
    df = _cohort(signal=True)
    feats = ["log_value_at_3h", "velocity_to_6h"]
    s = grouped_cv(lambda: RandomSurvivalForest(feats, n_estimators=120),
                   df, n_splits=4).stacked()
    c = concordance(s["duration"], s["event"], s["predicted"])
    assert c > 0.62, f"forest failed to find a strong planted signal (C={c:.3f})"


def test_forest_finds_nothing_in_noise():
    """The property that makes the previous test meaningful."""
    df = _cohort(signal=False)
    feats = ["log_value_at_3h", "velocity_to_6h"]
    s = grouped_cv(lambda: RandomSurvivalForest(feats, n_estimators=120),
                   df, n_splits=4).stacked()
    c = concordance(s["duration"], s["event"], s["predicted"])
    assert 0.38 < c < 0.62, f"forest scored {c:.3f} on pure noise — it is leaking"


def test_predictions_are_times_in_hours_not_risk_scores():
    """The shared interface promises hours, higher = longer life.

    A risk score would be both wrong in scale and inverted in sign, and the
    C-index would silently come out below 0.5 rather than erroring.
    """
    df = _cohort(signal=True)
    m = RandomSurvivalForest(["log_value_at_3h", "velocity_to_6h"], n_estimators=80)
    m.fit(df, df["t_death"].to_numpy(float), df["event_observed"].to_numpy(bool))
    p = m.predict(df)
    assert np.all(p > 0), "predicted times must be positive"
    assert np.all(p <= m.horizon_ + 1e-9), "RMST cannot exceed the training horizon"
    # Higher early value was planted to mean longer life; the ordering must agree.
    hi = p[df["log_value_at_3h"] > 1.0].mean()
    lo = p[df["log_value_at_3h"] < -1.0].mean()
    assert hi > lo


def test_falls_back_instead_of_fitting_on_too_few_deaths():
    df = _cohort(signal=True, n=60)
    events = np.zeros(len(df), bool)
    events[:3] = True                     # 3 deaths, below MIN_EVENTS
    m = RandomSurvivalForest(["log_value_at_3h", "velocity_to_6h"])
    m.fit(df, df["t_death"].to_numpy(float), events)
    assert m.model_ is None, "should not fit a forest on 3 deaths"
    p = m.predict(df)
    assert len(set(p.tolist())) == 1, "fallback must be a constant"


def test_horizon_comes_from_training_fold_only():
    """Guards against the test fold leaking into the prediction scale."""
    df = _cohort(signal=True)
    train = df.iloc[:150]
    m = RandomSurvivalForest(["log_value_at_3h", "velocity_to_6h"], n_estimators=80)
    m.fit(train, train["t_death"].to_numpy(float),
          train["event_observed"].to_numpy(bool))
    assert m.horizon_ == pytest.approx(float(train["t_death"].max()))
