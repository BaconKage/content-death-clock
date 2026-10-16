"""The pipeline additions of amendment 2026-10-16.

The plan asks for bootstrap intervals on metric *differences*, RMSE alongside
MAE, and two uncensored comparators. None of them was produced by the
evaluation harness until 2026-10-16, which mattered most for Cohort B: its
out-of-fold predictions were never saved, so any quantity the one-shot run did
not compute could only be had by running the holdout a second time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdc.eval.validate import paired_delta_bootstrap, rmse_log10
from cdc.models import dataset
from cdc.models.survival import LogLinearUncensored


def _stacked(pred, dur, ev, creators):
    return pd.DataFrame({
        "fold": 0, "post_id": [f"p{i}" for i in range(len(dur))],
        "creator_id": creators, "predicted": pred,
        "duration": dur, "event": ev,
    })


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 200
    dur = rng.exponential(50, n) + 1
    ev = rng.random(n) < 0.8
    creators = [f"c{i % 20}" for i in range(n)]
    return dur, ev, creators


def test_identical_models_have_zero_difference_on_every_draw(toy):
    dur, ev, cr = toy
    pred = dur * np.exp(np.random.default_rng(1).normal(0, 1, len(dur)))
    s = {"m": _stacked(pred, dur, ev, cr), "b": _stacked(pred, dur, ev, cr)}
    d = paired_delta_bootstrap(s, [("m", "b")], n_boot=50, seed=1)["m vs b"]
    assert d["delta_c_index"] == 0
    assert d["delta_c_ci_low"] == 0 and d["delta_c_ci_high"] == 0


def test_a_model_that_orders_perfectly_beats_a_constant(toy):
    dur, ev, cr = toy
    s = {"m": _stacked(dur.copy(), dur, ev, cr),
         "b": _stacked(np.full(len(dur), 48.0), dur, ev, cr)}
    d = paired_delta_bootstrap(s, [("m", "b")], n_boot=100, seed=1)["m vs b"]
    assert d["delta_c_index"] == pytest.approx(0.5)
    assert d["delta_c_ci_low"] > 0.4
    assert d["share_draws_model_c_higher"] == 1.0
    assert d["delta_mae_log10"] < 0          # negative favours the model


def test_alignment_is_by_post_id_not_by_row_order(toy):
    """Shuffling one model's rows must not change the paired result."""
    dur, ev, cr = toy
    pred = dur * 1.5
    a = _stacked(pred, dur, ev, cr)
    b = _stacked(np.full(len(dur), 48.0), dur, ev, cr)
    d1 = paired_delta_bootstrap({"m": a, "b": b}, [("m", "b")], n_boot=50, seed=3)
    d2 = paired_delta_bootstrap({"m": a.sample(frac=1, random_state=7), "b": b},
                                [("m", "b")], n_boot=50, seed=3)
    assert d1["m vs b"] == pytest.approx(d2["m vs b"])


def test_rmse_ignores_censored_posts_and_matches_hand_value():
    dur = np.array([10.0, 100.0, 1000.0])
    pred = np.array([100.0, 100.0, 1.0])
    ev = np.array([True, True, False])
    # log10 errors on the two deaths: 1 and 0  ->  sqrt(1/2)
    assert rmse_log10(dur, ev, pred) == pytest.approx(np.sqrt(0.5))


def test_loglinear_comparator_trains_on_deaths_only_and_predicts_hours():
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(0, 1, n)
    dur = 10 ** (1.5 + 0.4 * x)
    ev = np.ones(n, bool)
    ev[:30] = False
    dur[:30] = 1e6                     # absurd censoring times: must be ignored
    X = pd.DataFrame({"f": x})
    m = LogLinearUncensored(["f"]).fit(X, dur, ev)
    p = m.predict(X)
    assert np.all(p > 0)
    assert m.model_.coef_[0] == pytest.approx(0.4, abs=1e-6)


def test_exploratory_feature_set_is_refused_on_the_holdout():
    from cdc.eval import report
    with pytest.raises(SystemExit, match="exploratory"):
        report.run(platform="youtube", cohort="B", unlock_holdout=True,
                   feature_set="exploratory", write=False)


def test_exploratory_columns_extend_the_confirmatory_eleven():
    from cdc.models.synthetic import FEATURE_COLUMNS
    df = pd.DataFrame(columns=["cat_music", "cat_news", "other"])
    cols = dataset.exploratory_columns(df)
    assert cols[:len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    assert "log_rel_velocity_6h" in cols
    assert {"cat_music", "cat_news"} <= set(cols) and "other" not in cols
