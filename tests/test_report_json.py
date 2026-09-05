"""The gold evaluation artifact must be JSON that is not Python-specific.

`data/gold/evaluation_*.json` is the machine-readable result of the study — the
file a reader parses instead of taking the paper's word for a number. It was
being written with `json.dumps(..., default=str)`, whose `allow_nan` defaults to
True, so a Wilcoxon test with too few pairs put bare `NaN` literals in it.
Python's own loader accepts those, which is exactly why it went unnoticed; every
strict parser (JSON.parse, R's jsonlite, jq) rejects the file.

NaN itself is a legitimate result here and must survive as *something* — these
tests pin it to null rather than to a crash or a silently invalid file.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from cdc.eval.report import dumps_strict


def strict_loads(s: str):
    """Parse the way a non-Python consumer would: no NaN/Infinity extension."""
    def reject(x):
        raise ValueError(f"non-JSON constant in output: {x}")
    return json.loads(s, parse_constant=reject)


def test_nan_becomes_null_not_a_bare_nan_literal():
    out = dumps_strict({"wilcoxon": {"p_value": float("nan"), "statistic": float("nan")}})
    assert "NaN" not in out
    assert strict_loads(out)["wilcoxon"]["p_value"] is None


def test_infinities_survive_as_null():
    out = dumps_strict({"a": float("inf"), "b": float("-inf")})
    assert strict_loads(out) == {"a": None, "b": None}


def test_finite_numbers_are_untouched():
    payload = {"c_index": 0.7130177514792899, "n": 60, "ci": [0.44, 0.95]}
    assert strict_loads(dumps_strict(payload)) == payload


def test_numpy_scalars_are_serialised():
    """Metrics come back from numpy, not from Python floats."""
    out = strict_loads(dumps_strict({"m": np.float64(0.5), "n": np.int64(7),
                                     "bad": np.float64("nan")}))
    assert out == {"m": 0.5, "n": 7, "bad": None}


def test_nested_structures_are_cleaned_throughout():
    payload = {"results": {"aft": {"mae": float("nan")}},
               "rows": [{"p": float("nan")}, {"p": 0.04}]}
    got = strict_loads(dumps_strict(payload))
    assert got["results"]["aft"]["mae"] is None
    assert got["rows"] == [{"p": None}, {"p": 0.04}]


def test_the_real_shape_round_trips():
    """A cut-down copy of a real evaluation payload, NaNs and all."""
    payload = {
        "generated_at": "2026-09-01T20:09:50+00:00", "platform": "youtube",
        "underpowered": True, "n_posts": 60, "n_deaths": 7,
        "censoring_rate": 0.8833333333333333,
        "attrition": {"observed on this platform": 215, "FINAL analysis set": 60},
        "results": {"weibull_aft": {"c_index": 0.713, "mae_log10": 1.68}},
        "wilcoxon": {"weibull_aft vs constant_48h": {
            "n_pairs": 7, "statistic": float("nan"),
            "p_value": float("nan"), "median_diff": float("nan")}},
    }
    got = strict_loads(dumps_strict(payload))
    assert got["n_deaths"] == 7
    assert got["wilcoxon"]["weibull_aft vs constant_48h"]["p_value"] is None


def test_the_old_writer_produced_a_file_strict_parsers_reject():
    """Pins the defect itself, so the regression is unambiguous.

    This is exactly what `json.dumps(out, indent=2, default=str)` did, and why
    the committed artifact could not be read outside Python.
    """
    old = json.dumps({"p_value": float("nan")}, indent=2, default=str)
    assert "NaN" in old
    with pytest.raises(ValueError):
        strict_loads(old)
