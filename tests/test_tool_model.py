"""The instant model: curves, recalibration, held-out channels, and the shipped build."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cdc.tool import model as M
from cdc.tool import table as T
from cdc.tool.features import FEATURES, MARKS
from tests.tool_synth import as_table, synthetic_rows


@pytest.fixture(scope="module")
def rows():
    return synthetic_rows()


@pytest.fixture(scope="module")
def fitted(rows):
    return M.InstantModel(b=0.6, n_estimators=60).fit(rows)


def test_average_curve_is_kaplan_meier_on_the_marks():
    r = pd.DataFrame({"video": [1, 2, 3, 4], "fade_h": [3.0, 24.0, 24.0, 240.0],
                      "faded": [True, True, False, False]})
    s = dict(zip(MARKS, M.average_curve(r)))
    assert s[1.0] == 1.0 and s[3.0] == 0.75 and s[12.0] == 0.75
    assert s[24.0] == pytest.approx(0.75 * (1 - 1 / 3))      # the censored-at-24 video is still at risk
    assert s[240.0] == pytest.approx(0.5)


def test_blend_ends_and_shape():
    rng = np.random.default_rng(1)
    raw = np.sort(rng.random((50, len(MARKS))), axis=1)[:, ::-1]
    avg = np.linspace(1, 0.2, len(MARKS))
    assert np.allclose(M.blend(raw, avg, 1.0), np.clip(raw, 1e-6, 1))
    assert np.allclose(M.blend(raw, avg, 0.0), np.broadcast_to(avg, raw.shape))
    mid = M.blend(raw, avg, 0.5)
    assert (np.diff(mid, axis=1) <= 1e-12).all()             # still never increases with time


def test_curves_are_probabilities_that_never_rise(fitted, rows):
    s = fitted.survival(rows[FEATURES].head(200))
    assert ((s >= 0) & (s <= 1)).all()
    assert (np.diff(s, axis=1) <= 1e-12).all()
    assert np.allclose(fitted.fade_by(rows[FEATURES].head(200), 48), 1 - s[:, M.col(48)])


def test_the_build_is_deterministic(rows):
    a = M.InstantModel(b=0.5, n_estimators=40).fit(rows)
    b = M.InstantModel(b=0.5, n_estimators=40).fit(rows.sample(frac=1, random_state=3).pipe(T._order))
    assert M.fingerprint(a) == M.fingerprint(b)


def test_held_out_predictions_never_train_on_the_predicted_channel(rows, monkeypatch):
    seen = []
    real_fit = M.InstantModel.fit

    def spy(self, r):
        seen.append(set(r["creator_id"]))
        self.n_estimators = 20
        return real_fit(self, r)

    monkeypatch.setattr(M.InstantModel, "fit", spy)
    test = T.model_rows(rows)
    pred = M.held_out_predictions(rows, test, b=0.7)
    assert len(pred) == len(test) and pred.index.equals(test.index)
    groups = {g: set(test.loc[[T.creator_group(c) == g for c in test["creator_id"]], "creator_id"])
              for g in range(T.N_GROUPS)}
    used = [g for g in range(T.N_GROUPS) if groups[g]]
    for trained_on, g in zip(seen, used):
        assert not (trained_on & groups[g])


def test_fit_b_finds_the_ends():
    n = 2000
    rng = np.random.default_rng(2)
    y = rng.random(n) < 0.3
    df = pd.DataFrame({"faded": y, "fade_h": np.where(y, 24.0, 240.0)})
    for h in M.HORIZONS:
        df[f"avg_{h:g}"] = 0.3
    noise = df.copy()
    for h in M.HORIZONS:
        noise[f"raw_{h:g}"] = rng.random(n)
    assert M.fit_b(noise) <= 0.1                            # useless forest: quote the average
    sharp = df.copy()
    for h in M.HORIZONS:
        sharp[f"raw_{h:g}"] = np.where(y, 0.97, 0.03)
    assert M.fit_b(sharp) >= 0.9                            # sharp, right forest: keep it


def test_shipped_table_rebuilds_the_frozen_model():
    """Laptop or server: the table plus the recipe must reproduce the tested model."""
    card = json.loads((T.ASSETS / "model_card.json").read_text(encoding="utf-8"))
    table = T.load_table()
    assert T.digest(table) == card["trained_on"]["table_sha256"]
    m = M.InstantModel(b=card["b"]).fit(table)
    assert M.fingerprint(m) == card["fingerprint"]


def test_shipped_table_carries_no_identifiers():
    t = T.load_table()
    assert list(t.columns) == ["cohort", "video", "channel", "mark", *FEATURES, "fade_h", "faded"]
    assert set(t["cohort"]) <= {"A", "B", "C"}
    assert all(pd.api.types.is_numeric_dtype(t[c]) or c == "cohort" for c in t.columns)


def test_anonymise_round_trips_through_csv(rows, tmp_path):
    anon = T.anonymise(rows)
    sha = T.save_table(anon, tmp_path / "t.csv")
    back = T.load_table(tmp_path / "t.csv")
    assert T.digest(back) == sha
    a = M.InstantModel(n_estimators=30).fit(rows)
    b = M.InstantModel(n_estimators=30).fit(back)
    assert M.fingerprint(a) == M.fingerprint(b)
    assert as_table(rows).shape[0] == len(back)
