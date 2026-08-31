"""Models and evaluation, validated on data whose answer we know.

Real labels are days away. If the first numbers we ever see are also the first
test of the code that produced them, there is no way to distinguish a finding
from a bug. So the whole analysis is exercised here against synthetic cohorts
built with known ground truth — including a NULL cohort where the correct answer
is "no signal", which is the only reliable way to catch leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdc.eval.validate import (
    assert_no_creator_leakage,
    bootstrap_ci_by_creator,
    concordance,
    grouped_cv,
    mae_log10,
    paired_wilcoxon,
    per_post_abs_error,
)
from cdc.models.baselines import (
    ConstantLifetime,
    KaplanMeierMedian,
    SubscriberOnly,
    all_baselines,
)
from cdc.models.survival import GradientBoostedUncensored, WeibullAFT
from cdc.models.synthetic import FEATURE_COLUMNS, make_cohort, make_null_cohort


@pytest.fixture(scope="module")
def cohort():
    return make_cohort(n_posts=600, n_creators=40)


@pytest.fixture(scope="module")
def null_cohort():
    return make_null_cohort(n_posts=600, n_creators=40)


# ------------------------------------------------------- the synthetic ground
def test_cohort_has_censoring_and_creator_structure(cohort):
    """If the fixture had no censoring, none of the survival machinery would be
    exercised and the tests below would prove nothing."""
    assert 0.0 < cohort.censoring_rate < 0.6
    assert cohort.frame["creator_id"].nunique() == 40
    assert cohort.frame["t_death"].gt(0).all()
    # Censored rows must sit AT the censoring horizon, not past it.
    censored = cohort.frame[~cohort.frame["event_observed"]]
    assert np.allclose(censored["t_death"], 336.0)


# --------------------------------------------------------------- THE BIG ONE
def _cv_with(splitter, frame, groups=None):
    """Run a chosen splitter by hand, to compare grouping against leaking."""
    d = frame["t_death"].to_numpy(float)
    e = frame["event_observed"].to_numpy()
    preds = np.empty(len(frame))
    it = (splitter.split(frame, groups=groups) if groups is not None
          else splitter.split(frame))
    for tr, te in it:
        m = WeibullAFT(FEATURE_COLUMNS).fit(frame.iloc[tr], d[tr], e[tr])
        preds[te] = m.predict(frame.iloc[te])
    return concordance(d, e, preds)


def test_null_cohort_yields_no_signal(null_cohort):
    """Lifetime here is pure noise with respect to every early-window feature.
    A correct pipeline finds nothing."""
    res = grouped_cv(lambda: WeibullAFT(FEATURE_COLUMNS), null_cohort.frame)
    c = concordance(*[res.stacked()[k] for k in ("duration", "event", "predicted")])
    assert c < 0.55, (
        f"C-index {c:.3f} on data with NO signal — the evaluation is leaking, "
        "not the model succeeding"
    )


def test_grouping_actually_suppresses_leakage(null_cohort):
    """Proof that the test above has teeth.

    Run the SAME null data through GroupKFold and through a deliberately leaky
    random split. If grouping did nothing, both would score alike and the null
    test would be worthless reassurance. The leaky split must visibly inflate.
    """
    from sklearn.model_selection import GroupKFold, KFold

    frame = null_cohort.frame
    d = frame["t_death"].to_numpy(float)
    e = frame["event_observed"].to_numpy()

    def run(splitter, groups=None):
        preds = np.empty(len(frame))
        it = (splitter.split(frame, groups=groups) if groups is not None
              else splitter.split(frame))
        for tr, te in it:
            m = GradientBoostedUncensored(FEATURE_COLUMNS)
            m.fit(frame.iloc[tr], d[tr], e[tr])
            preds[te] = m.predict(frame.iloc[te])
        return concordance(d, e, preds)

    groups = frame["creator_id"].to_numpy()
    grouped = run(GroupKFold(n_splits=5), groups=groups)
    leaky = run(KFold(n_splits=5, shuffle=True, random_state=0))

    # A flexible learner is used here deliberately. The penalised Weibull AFT
    # inflates only ~+0.02 under a leaky split, because regularisation limits how
    # much creator identity it can memorise. The gradient-booster inflates ~+0.31
    # — measured 2026-08-31: 0.49 grouped, 0.80 leaky, on data containing NO
    # signal whatsoever. That is the failure mode in its true form: a result that
    # looks like a strong discovery and is entirely an artefact of the split.
    assert leaky > grouped + 0.15, (
        f"leaky split scored {leaky:.3f} vs grouped {grouped:.3f} — the fixture "
        "cannot demonstrate leakage, so the null test proves nothing"
    )
    assert grouped < 0.55, f"grouped split still leaking: C = {grouped:.3f}"


def test_leakage_assertion_actually_fires():
    with pytest.raises(AssertionError, match="creator leakage"):
        assert_no_creator_leakage(["a", "b"], ["b", "c"])


def test_grouped_cv_never_splits_a_creator(cohort):
    res = grouped_cv(lambda: ConstantLifetime(48.0), cohort.frame)
    seen_in_test = [set(f.creator_ids) for f in res.folds]
    for i, a in enumerate(seen_in_test):
        for b in seen_in_test[i + 1:]:
            assert not (a & b), "a creator appeared in two different test folds"
    # every post predicted exactly once, out of fold
    assert len(res.stacked()) == len(cohort.frame)


# ------------------------------------------------------------------- metrics
def test_concordance_of_a_perfect_predictor_is_one():
    d = np.array([10.0, 20.0, 30.0, 40.0])
    e = np.array([1, 1, 1, 1])
    assert concordance(d, e, d) == pytest.approx(1.0)


def test_concordance_of_a_constant_predictor_is_half():
    d = np.array([10.0, 20.0, 30.0, 40.0])
    e = np.array([1, 1, 1, 1])
    assert concordance(d, e, np.full(4, 25.0)) == pytest.approx(0.5)


def test_concordance_of_an_inverted_predictor_is_zero():
    d = np.array([10.0, 20.0, 30.0, 40.0])
    e = np.array([1, 1, 1, 1])
    assert concordance(d, e, -d) == pytest.approx(0.0)


def test_mae_uses_only_observed_deaths():
    """A censored post has no known duration; scoring against its censoring time
    would reward systematic under-prediction."""
    d = np.array([10.0, 100.0])
    e = np.array([True, False])
    # second post is censored, so only the first (perfectly predicted) counts
    assert mae_log10(d, e, np.array([10.0, 1.0])) == pytest.approx(0.0)


def test_per_post_error_is_nan_where_censored():
    err = per_post_abs_error(np.array([10.0, 50.0]), np.array([True, False]),
                             np.array([10.0, 10.0]))
    assert err[0] == pytest.approx(0.0)
    assert np.isnan(err[1])


# ----------------------------------------------------------------- baselines
def test_km_median_is_not_dragged_down_by_censoring():
    """The reason we use Kaplan-Meier rather than the raw median of deaths.

    Here the long-lived half of the cohort is censored. The naive median of
    observed deaths is ~20h; KM must give something larger, because it knows the
    censored posts outlived their censoring time.
    """
    durations = np.array([10, 15, 20, 25] + [100] * 6, dtype=float)
    events = np.array([1, 1, 1, 1] + [0] * 6, dtype=bool)
    X = pd.DataFrame(index=range(len(durations)))

    km = KaplanMeierMedian().fit(X, durations, events)
    naive_median = float(np.median(durations[events]))
    assert km.median_ > naive_median


def test_constant_baseline_predicts_the_constant():
    m = ConstantLifetime(48.0).fit(pd.DataFrame(index=range(5)), np.ones(5), np.ones(5))
    assert np.all(m.predict(pd.DataFrame(index=range(5))) == 48.0)


def test_baselines_all_run_and_produce_positive_finite_predictions(cohort):
    for b in all_baselines():
        res = grouped_cv(lambda b=b: b.__class__(), cohort.frame)
        p = res.stacked()["predicted"].to_numpy()
        assert np.isfinite(p).all(), f"{b.name} produced non-finite predictions"
        assert (p > 0).all(), f"{b.name} produced non-positive survival times"


# --------------------------------------------------------------- the model
def test_weibull_recovers_the_planted_signal(cohort):
    """The positive control: on data built so early growth predicts lifetime,
    the model must find it."""
    res = grouped_cv(lambda: WeibullAFT(FEATURE_COLUMNS), cohort.frame)
    s = res.stacked()
    c = concordance(s["duration"], s["event"], s["predicted"])
    assert c > 0.65, f"C-index {c:.3f} — failed to recover a signal we planted"


def test_model_beats_the_constant_baseline(cohort):
    """H1, on synthetic data. If this fails the pipeline is broken, since the
    signal is there by construction."""
    model = grouped_cv(lambda: WeibullAFT(FEATURE_COLUMNS), cohort.frame).stacked()
    base = grouped_cv(lambda: ConstantLifetime(48.0), cohort.frame).stacked()
    assert (concordance(model["duration"], model["event"], model["predicted"])
            > concordance(base["duration"], base["event"], base["predicted"]))


def test_model_beats_subscriber_only(cohort):
    """H2, on synthetic data. The cohort is built with a weak follower
    coefficient and strong dynamics coefficients, so the dynamics must win."""
    model = grouped_cv(lambda: WeibullAFT(FEATURE_COLUMNS), cohort.frame).stacked()
    subs = grouped_cv(lambda: SubscriberOnly(), cohort.frame).stacked()
    c_model = concordance(model["duration"], model["event"], model["predicted"])
    c_subs = concordance(subs["duration"], subs["event"], subs["predicted"])
    assert c_model > c_subs, f"model {c_model:.3f} did not beat subscriber-only {c_subs:.3f}"


# ----------------------------------------------------------------- inference
def test_bootstrap_interval_contains_the_point_estimate(cohort):
    s = grouped_cv(lambda: WeibullAFT(FEATURE_COLUMNS), cohort.frame).stacked()
    point, lo, hi = bootstrap_ci_by_creator(s, n_boot=120, seed=1)
    assert lo <= point <= hi
    assert hi - lo < 0.5, "interval implausibly wide — check creator resampling"


def test_wilcoxon_detects_a_real_difference(cohort):
    model = grouped_cv(lambda: WeibullAFT(FEATURE_COLUMNS), cohort.frame).stacked()
    base = grouped_cv(lambda: ConstantLifetime(48.0), cohort.frame).stacked()
    r = paired_wilcoxon(model, base)
    assert r["n_pairs"] > 50
    assert r["p_value"] < 0.05
    assert r["median_diff"] < 0, "model errors should be smaller than the baseline's"


def test_wilcoxon_finds_nothing_when_models_are_identical(cohort):
    a = grouped_cv(lambda: ConstantLifetime(48.0), cohort.frame).stacked()
    b = grouped_cv(lambda: ConstantLifetime(48.0), cohort.frame).stacked()
    r = paired_wilcoxon(a, b)
    assert np.isnan(r["p_value"]), "identical models must not produce a p-value"


def test_cv_refuses_when_there_are_fewer_creators_than_folds():
    tiny = make_cohort(n_posts=40, n_creators=3).frame
    with pytest.raises(ValueError, match="cannot group-split"):
        grouped_cv(lambda: ConstantLifetime(48.0), tiny, n_splits=5)
