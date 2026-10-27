"""The instant model: one reading in, a fade-time curve out.

**The forest.** The study's registered random survival forest, with the same
settings (300 trees, leaves of at least 5, square-root feature sampling, the
project seed), trained on the checkpoint rows of ``table``. It outputs, for each
video, the chance it is still getting attention at each scheduled mark.

**Why fade times sit on the mark grid.** A forest from ``scikit-survival``
stores a survival curve over every distinct training time in every node; with
raw times that runs to hundreds of megabytes. Fade times are readings, and
readings are taken at the scheduled marks, so ``features.fade_mark`` puts them
on that grid of twelve times. Nothing is lost (a fade "at 24.3h" is the 24-hour
reading) and the forest stays small enough to rebuild at app start-up.

**The recalibration: one number.** Trained on several readings of the same
video and channel, the forest is overconfident on channels it has not seen:
when it says 5% the truth is nearer 12%, when it says 84%, nearer 70%
(measured out-of-fold on Cohorts A and B, 2026-10-27). The fix pulls each curve
part of the way toward the average curve, on the log scale:

    log S(t) = b * log S_forest(t) + (1 - b) * log S_average(t)

``b`` = 1 keeps the forest, 0 quotes the average. One parameter cannot overfit
much, and a blend of two decreasing curves is still decreasing, so the odds for
day 1, 2, 3 and 7 can never contradict each other. ``b`` is chosen on
predictions for channels held out of training (``out_of_fold``), never on the
data being tested.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from cdc.tool.features import CHECKPOINTS, FEATURES, MARKS
from cdc.tool.table import N_GROUPS, creator_group, model_rows

SEED = 20260830
HORIZONS = (24.0, 48.0, 72.0, 168.0)
FAST_H = 48.0              # "fades fast" = within two days of posting, as in amendment 7
FLAG_SHARE = 0.20          # "high risk" = the 20% of held-out predictions most likely to fade fast
_MIN_S = 1e-6


def _video_ids(rows: pd.DataFrame) -> np.ndarray:
    return (rows["video"] if "video" in rows else rows["video_key"]).to_numpy()


def average_curve(rows: pd.DataFrame) -> np.ndarray:
    """Kaplan-Meier on the mark grid, one entry per video: the "average video"."""
    v = rows.assign(_v=_video_ids(rows)).drop_duplicates("_v")
    t, e = v["fade_h"].to_numpy(float), v["faded"].to_numpy(bool)
    s, out = 1.0, []
    for m in MARKS:
        at_risk = (t >= m).sum()
        d = ((t == m) & e).sum()
        if at_risk:
            s *= 1.0 - d / at_risk
        out.append(s)
    return np.array(out)


def blend(raw: np.ndarray, average: np.ndarray, b: float) -> np.ndarray:
    raw = np.clip(raw, _MIN_S, 1.0)
    avg = np.clip(np.broadcast_to(average, raw.shape), _MIN_S, 1.0)
    return np.exp(b * np.log(raw) + (1.0 - b) * np.log(avg))


def col(h: float) -> int:
    return MARKS.index(float(h))


class InstantModel:
    def __init__(self, b: float = 1.0, n_estimators: int = 300, min_samples_leaf: int = 5,
                 max_features: Any = "sqrt", random_state: int = SEED) -> None:
        self.b = float(b)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, rows: pd.DataFrame) -> "InstantModel":
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv

        r = model_rows(rows)
        self.medians_ = r[FEATURES].astype(float).median()
        self.forest_ = RandomSurvivalForest(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, random_state=self.random_state, n_jobs=-1)
        self.forest_.fit(self._x(r), Surv.from_arrays(event=r["faded"].to_numpy(bool),
                                                      time=r["fade_h"].to_numpy(float)))
        # Trees are built in parallel (deterministic: each has its own seed), but
        # a parallel predict sums the trees in whatever order threads finish,
        # which moves the last digits. One thread makes predictions exact.
        self.forest_.n_jobs = 1
        self.average_ = average_curve(r)
        self.n_rows_ = int(len(r))
        self.n_videos_ = int(len(set(_video_ids(r))))
        return self

    def _x(self, X: pd.DataFrame) -> np.ndarray:
        return X[FEATURES].astype(float).fillna(self.medians_).to_numpy()

    def raw_survival(self, X: pd.DataFrame) -> np.ndarray:
        """Chance still getting attention at each mark, straight from the forest."""
        sf = self.forest_.predict_survival_function(self._x(X), return_array=True)
        idx = np.searchsorted(self.forest_.unique_times_, np.array(MARKS), side="right") - 1
        out = np.ones((len(X), len(MARKS)))
        ok = idx >= 0
        out[:, ok] = sf[:, idx[ok]]
        return out

    def survival(self, X: pd.DataFrame) -> np.ndarray:
        return blend(self.raw_survival(X), self.average_, self.b)

    def fade_by(self, X: pd.DataFrame, h: float = FAST_H) -> np.ndarray:
        """Chance of fading by ``h`` hours after posting (already faded included)."""
        return 1.0 - self.survival(X)[:, col(h)]


# ----------------------------------------------------- held-out-channel predictions
def held_out_predictions(train: pd.DataFrame, test: pd.DataFrame,
                         b: float = 1.0) -> pd.DataFrame:
    """Predict every test row with a model that never saw that row's channel.

    Test channels are split into the five fixed groups of ``creator_group``. For
    each group, a model is trained on the training rows minus every channel in
    that group, and predicts that group. Returned: the test rows plus raw and
    blended fade-by probabilities at each horizon, and the average curve's value
    (from the same training rows), which is the "average rate" comparator.
    """
    test = test.copy()
    test["_group"] = [creator_group(c) for c in test["creator_id"]]
    parts = []
    for g in range(N_GROUPS):
        te = test[test["_group"] == g]
        if te.empty:
            continue
        tr = train[~train["creator_id"].isin(set(te["creator_id"]))]
        m = InstantModel(b=b).fit(tr)
        raw = m.raw_survival(te)
        bl = blend(raw, m.average_, b)
        add = {}
        for h in HORIZONS:
            add[f"raw_{h:g}"] = 1.0 - raw[:, col(h)]
            add[f"p_{h:g}"] = 1.0 - bl[:, col(h)]
            add[f"avg_{h:g}"] = np.full(len(te), 1.0 - m.average_[col(h)])
        parts.append(te.assign(**add))
    return pd.concat(parts).drop(columns="_group").loc[test.index]


def out_of_fold(rows: pd.DataFrame) -> pd.DataFrame:
    """Held-out-channel predictions for the training data itself (b = 1)."""
    r = model_rows(rows)
    return held_out_predictions(r, r, b=1.0)


def known(df: pd.DataFrame, h: float) -> np.ndarray:
    """Rows whose status at h is known: faded by then, or followed past it."""
    return (df["faded"].to_numpy(bool) | (df["fade_h"].to_numpy(float) > h))


def outcome(df: pd.DataFrame, h: float) -> np.ndarray:
    return df["faded"].to_numpy(bool) & (df["fade_h"].to_numpy(float) <= h)


def fit_b(oof: pd.DataFrame) -> float:
    """The blend weight minimising the Brier score, averaged over the four horizons."""
    best, best_loss = 1.0, np.inf
    for b in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        loss = 0.0
        for h in HORIZONS:
            k = known(oof, h)
            raw_s = np.clip(1.0 - oof[f"raw_{h:g}"].to_numpy(), _MIN_S, 1)
            avg_s = np.clip(1.0 - oof[f"avg_{h:g}"].to_numpy(), _MIN_S, 1)
            p = 1.0 - np.exp(b * np.log(raw_s) + (1 - b) * np.log(avg_s))
            loss += np.mean((p[k] - outcome(oof, h)[k]) ** 2)
        if loss < best_loss - 1e-12:
            best, best_loss = float(b), loss
    return best


def thresholds(oof: pd.DataFrame, b: float) -> dict[str, float]:
    """Risk cut-offs on the two-day fade chance, from held-out predictions.

    "High" is the top 20% (the flag that Cohort C tests), "low" the bottom 20%.
    Fixed once here, then used unchanged by the test and by the app.
    """
    raw_s = np.clip(1.0 - oof[f"raw_{FAST_H:g}"].to_numpy(), _MIN_S, 1)
    avg_s = np.clip(1.0 - oof[f"avg_{FAST_H:g}"].to_numpy(), _MIN_S, 1)
    p = 1.0 - np.exp(b * np.log(raw_s) + (1 - b) * np.log(avg_s))
    return {"high": float(np.quantile(p, 1 - FLAG_SHARE)), "low": float(np.quantile(p, FLAG_SHARE))}


# ------------------------------------------------------------------ fingerprint
def probe_inputs() -> pd.DataFrame:
    """A fixed set of made-up videos, for checking that two builds agree."""
    rows = []
    for age in CHECKPOINTS:
        for lv in (1.0, 2.5, 4.0):
            for like in (-2.6, -1.4):
                for subs in (3.5, 5.5):
                    rows.append({"age_h": age, "log_views": lv,
                                 "log_views_per_h": round(np.log10(10 ** lv / age + 1), 2),
                                 "log_like_rate": like, "log_comment_rate": like - 1.3,
                                 "log_views_per_sub": round(lv - subs, 2), "log_subscribers": subs,
                                 "log_duration_sec": 2.6, "publish_hour_utc": 14.0, "title_len": 60.0})
    return pd.DataFrame(rows, columns=FEATURES)


def fingerprint(model: InstantModel) -> str:
    s = model.survival(probe_inputs())
    payload = json.dumps(np.round(s, 4).tolist())
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
