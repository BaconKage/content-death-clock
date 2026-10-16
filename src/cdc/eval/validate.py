"""Evaluation, exactly as pre-registered.

Three things happen here, and the first is the one that decides whether any
number produced downstream means anything.

**Grouping by creator.** Every post from a channel goes into the same fold.
Random splitting would put a creator's videos on both sides, letting the model
recognise the channel rather than learn the dynamics — scores rise and mean
nothing. ``GroupKFold`` is pre-specified in the frozen analysis plan precisely so
it cannot be quietly relaxed when random splitting produces prettier numbers, and
``assert_no_creator_leakage`` re-checks it on every fold at runtime rather than
trusting the splitter.

**Censoring-aware scoring.** Harrell's C-index is the primary metric: it asks
whether the model ranks post A as shorter-lived than post B when A really did die
first, over all comparable pairs. Pairs involving a censored post are only
counted where the ordering is knowable. MAE on log-time is reported too, but only
over posts observed to die — stating that restriction matters, because it is a
different and easier population.

**Inference resampled by creator.** Bootstrap confidence intervals resample
*creators*, not posts. Resampling posts would treat 40 videos from one channel as
40 independent observations and produce intervals far too narrow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from cdc.config import settings


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    predictions: np.ndarray
    durations: np.ndarray
    events: np.ndarray
    post_ids: np.ndarray
    creator_ids: np.ndarray


@dataclass
class ModelResult:
    name: str
    folds: list[FoldResult] = field(default_factory=list)

    def stacked(self) -> pd.DataFrame:
        """Out-of-fold predictions for every post, one row each."""
        return pd.concat([
            pd.DataFrame({
                "fold": f.fold, "post_id": f.post_ids, "creator_id": f.creator_ids,
                "predicted": f.predictions, "duration": f.durations, "event": f.events,
            }) for f in self.folds
        ], ignore_index=True)


# --------------------------------------------------------------------- metrics
def concordance(durations, events, predictions) -> float:
    """Harrell's C-index. 0.5 is coin-flip; 1.0 is perfect ordering.

    lifelines expects a *predicted survival time* here (higher = longer life),
    which is what all our models emit, so no sign flip is needed.
    """
    from lifelines.utils import concordance_index
    return float(concordance_index(np.asarray(durations, float),
                                   np.asarray(predictions, float),
                                   np.asarray(events).astype(int)))


def mae_log10(durations, events, predictions) -> float:
    """Mean absolute error on log10 hours, over observed deaths only.

    Restricted to uncensored posts because a censored post's true duration is
    unknown — scoring against its censoring time would reward under-prediction.
    """
    m = np.asarray(events).astype(bool)
    if m.sum() == 0:
        return float("nan")
    d = np.log10(np.clip(np.asarray(durations, float)[m], 1e-6, None))
    p = np.log10(np.clip(np.asarray(predictions, float)[m], 1e-6, None))
    return float(np.mean(np.abs(d - p)))


def rmse_log10(durations, events, predictions) -> float:
    """Root mean squared error on log10 hours, over observed deaths only.

    Named in the frozen plan alongside MAE. Same restriction and for the same
    reason: a censored post has no true duration to score against.
    """
    m = np.asarray(events).astype(bool)
    if m.sum() == 0:
        return float("nan")
    d = np.log10(np.clip(np.asarray(durations, float)[m], 1e-6, None))
    p = np.log10(np.clip(np.asarray(predictions, float)[m], 1e-6, None))
    return float(np.sqrt(np.mean((d - p) ** 2)))


def per_post_abs_error(durations, events, predictions) -> pd.Series:
    """Per-post |error| on log10 time, NaN where censored. Feeds Wilcoxon."""
    d = np.asarray(durations, float)
    p = np.asarray(predictions, float)
    e = np.asarray(events).astype(bool)
    out = np.full(len(d), np.nan)
    ok = e & (d > 0) & (p > 0)
    out[ok] = np.abs(np.log10(d[ok]) - np.log10(p[ok]))
    return pd.Series(out)


# ------------------------------------------------------------------ leakage
def assert_no_creator_leakage(train_creators: Iterable, test_creators: Iterable) -> None:
    """Fail loudly if any creator appears on both sides of a split.

    Checked at runtime on every fold rather than trusting GroupKFold, because
    this is the failure that inflates every score in the paper while leaving no
    visible symptom.
    """
    overlap = set(train_creators) & set(test_creators)
    if overlap:
        raise AssertionError(
            f"creator leakage: {len(overlap)} creator(s) in both train and test, "
            f"e.g. {sorted(overlap)[:3]}"
        )


# --------------------------------------------------------------- the CV loop
def grouped_cv(model_factory: Callable[[], object], frame: pd.DataFrame,
               n_splits: int | None = None,
               duration_col: str = "t_death",
               event_col: str = "event_observed",
               group_col: str = "creator_id") -> ModelResult:
    """Run GroupKFold by creator and collect out-of-fold predictions."""
    from sklearn.model_selection import GroupKFold

    cfg = settings()["modelling"]
    n_splits = n_splits or int(cfg["cv"]["n_splits"])

    groups = frame[group_col].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(
            f"{n_groups} creators but {n_splits} folds requested — cannot group-split. "
            "Either collect more creators or reduce n_splits."
        )

    durations = frame[duration_col].to_numpy(float)
    events = frame[event_col].to_numpy()
    proto = model_factory()
    result = ModelResult(name=getattr(proto, "name", proto.__class__.__name__))

    gkf = GroupKFold(n_splits=n_splits)
    for i, (tr, te) in enumerate(gkf.split(frame, groups=groups)):
        assert_no_creator_leakage(groups[tr], groups[te])
        model = model_factory()
        model.fit(frame.iloc[tr], durations[tr], events[tr])
        preds = np.asarray(model.predict(frame.iloc[te]), dtype=float)
        # A model that emits nonsense should not silently poison the metrics.
        preds = np.where(np.isfinite(preds) & (preds > 0), preds, np.nan)
        if np.isnan(preds).any():
            preds = np.where(np.isnan(preds), np.nanmedian(durations[tr]), preds)

        result.folds.append(FoldResult(
            fold=i, n_train=len(tr), n_test=len(te), predictions=preds,
            durations=durations[te], events=events[te],
            post_ids=frame["post_id"].to_numpy()[te], creator_ids=groups[te],
        ))
    return result


# ----------------------------------------------------------------- inference
def bootstrap_ci_by_creator(stacked: pd.DataFrame, metric: str = "concordance",
                            n_boot: int | None = None, seed: int | None = None,
                            alpha: float = 0.05) -> tuple[float, float, float]:
    """(point estimate, lower, upper), resampling CREATORS with replacement.

    Resampling posts would treat many videos from one channel as independent
    observations and produce intervals that are far too narrow.
    """
    cfg = settings()["modelling"]
    n_boot = n_boot or int(cfg["bootstrap_iterations"])
    seed = cfg["random_seed"] if seed is None else seed
    rng = np.random.default_rng(seed)

    fn = concordance if metric == "concordance" else mae_log10
    point = fn(stacked["duration"], stacked["event"], stacked["predicted"])

    by_creator = {c: g for c, g in stacked.groupby("creator_id")}
    creators = np.array(list(by_creator))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(creators, size=len(creators), replace=True)
        samp = pd.concat([by_creator[c] for c in pick], ignore_index=True)
        try:
            v = fn(samp["duration"], samp["event"], samp["predicted"])
        except (ZeroDivisionError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_delta_bootstrap(stacked: dict[str, pd.DataFrame],
                           pairs: Iterable[tuple[str, str]],
                           n_boot: int | None = None, seed: int | None = None,
                           alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """Bootstrap intervals on metric DIFFERENCES, model minus baseline.

    The frozen plan asks for "confidence intervals on metric differences",
    resampled by creator. Two separate per-model intervals are not that: they
    can overlap heavily while the paired difference is clearly non-zero, or the
    reverse. The difference must be computed on the **same** creator draw for
    both members of the pair, and every pair here shares one set of draws.

    Returns, per ``"model vs baseline"``: the point difference in C-index and
    in MAE (log10), their percentile intervals, and the share of draws in which
    the model's C-index was higher. Positive delta-C favours the model; negative
    delta-MAE favours the model.
    """
    cfg = settings()["modelling"]
    n_boot = n_boot or int(cfg["bootstrap_iterations"])
    seed = cfg["random_seed"] if seed is None else seed
    pairs = list(pairs)
    names = sorted({n for p in pairs for n in p})

    # Align every model on post_id, so row i is the same post everywhere.
    ref = stacked[names[0]][["post_id", "creator_id", "duration", "event"]]
    ref = ref.reset_index(drop=True)
    order = ref["post_id"].to_numpy()
    preds = {}
    for n in names:
        s = stacked[n].set_index("post_id").loc[order]
        preds[n] = s["predicted"].to_numpy(float)
    dur = ref["duration"].to_numpy(float)
    ev = ref["event"].to_numpy()
    # Sorted, to match the creator order bootstrap_ci_by_creator draws from.
    rows_by_creator = {c: np.flatnonzero(ref["creator_id"].to_numpy() == c)
                       for c in sorted(pd.unique(ref["creator_id"]))}
    creators = np.array(list(rows_by_creator))

    def both(idx):
        return {n: (concordance(dur[idx], ev[idx], preds[n][idx]),
                    mae_log10(dur[idx], ev[idx], preds[n][idx])) for n in names}

    point = both(np.arange(len(dur)))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[tuple[float, float]]] = {f"{m} vs {b}": [] for m, b in pairs}
    for _ in range(n_boot):
        pick = rng.choice(creators, size=len(creators), replace=True)
        idx = np.concatenate([rows_by_creator[c] for c in pick])
        try:
            v = both(idx)
        except (ZeroDivisionError, ValueError):
            continue
        for m, b in pairs:
            draws[f"{m} vs {b}"].append((v[m][0] - v[b][0], v[m][1] - v[b][1]))

    out = {}
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for m, b in pairs:
        key = f"{m} vs {b}"
        arr = np.array(draws[key], dtype=float)
        if not len(arr):
            out[key] = {"n_boot": 0}
            continue
        dc, dm = arr[:, 0], arr[:, 1]
        out[key] = {
            "n_boot": int(len(arr)),
            "delta_c_index": point[m][0] - point[b][0],
            "delta_c_ci_low": float(np.percentile(dc, lo_q)),
            "delta_c_ci_high": float(np.percentile(dc, hi_q)),
            "share_draws_model_c_higher": float((dc > 0).mean()),
            "delta_mae_log10": point[m][1] - point[b][1],
            "delta_mae_ci_low": float(np.nanpercentile(dm, lo_q)),
            "delta_mae_ci_high": float(np.nanpercentile(dm, hi_q)),
        }
    return out


def paired_wilcoxon(model_stacked: pd.DataFrame,
                    baseline_stacked: pd.DataFrame) -> dict[str, float]:
    """Paired Wilcoxon signed-rank on per-post |error|, model vs baseline.

    Pairs are matched on post_id, and only posts observed to die contribute —
    a censored post has no error to compare.
    """
    from scipy.stats import wilcoxon

    a = model_stacked.set_index("post_id")
    b = baseline_stacked.set_index("post_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    ea = per_post_abs_error(a["duration"], a["event"], a["predicted"]).to_numpy()
    eb = per_post_abs_error(b["duration"], b["event"], b["predicted"]).to_numpy()
    ok = np.isfinite(ea) & np.isfinite(eb)
    ea, eb = ea[ok], eb[ok]

    if len(ea) < 10 or np.allclose(ea, eb):
        return {"n_pairs": int(len(ea)), "statistic": float("nan"),
                "p_value": float("nan"), "median_diff": float("nan")}

    stat, p = wilcoxon(ea, eb)
    return {
        "n_pairs": int(len(ea)),
        "statistic": float(stat),
        "p_value": float(p),
        # Negative favours the model: its errors are smaller.
        "median_diff": float(np.median(ea - eb)),
    }
