"""Cohort A/B split at the pre-specified freeze instant, and the holdout guard.

The freeze date sat in `settings.yaml` from 2026-08-30 and no code read it. A
temporal holdout that nothing implements is not a holdout, and "evaluated
exactly once" is not a property you get by writing it in a plan — something has
to make the second look inconvenient and visible.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from cdc.eval import holdout
from cdc.models.dataset import freeze_instant


# ------------------------------------------------------------------ the split
def test_freeze_instant_is_read_from_settings_and_is_utc():
    f = freeze_instant()
    assert f == pd.Timestamp("2026-09-16T00:00:00+00:00")
    assert f.tzinfo is not None, "a naive freeze instant would silently shift"


def test_cohort_a_and_b_partition_the_posts():
    """Every post belongs to exactly one cohort — none lost, none double-counted."""
    from cdc.models import dataset
    a = dataset.build(cohort="A").frame
    b = dataset.build(cohort="B").frame
    both = dataset.build(cohort="all").frame
    assert len(a) + len(b) == len(both)
    if len(a) and len(b):
        assert not set(a["post_id"]) & set(b["post_id"])


def test_cohort_argument_is_validated():
    from cdc.models import dataset
    with pytest.raises(ValueError, match="cohort must be"):
        dataset.build(cohort="C")


def test_boundary_is_exclusive_for_a_inclusive_for_b():
    """A post published exactly at the instant belongs to B, not A.

    Stated as a test because 'before the freeze' and 'on or after' must not
    both claim the boundary, and an off-by-one here would move a post between
    the analysis set and the holdout.
    """
    from cdc.models import dataset
    import inspect
    src = inspect.getsource(dataset.build)
    assert "(pub >= cohort_end) if after else (pub < cohort_end)" in src


# ------------------------------------------------------------ the holdout lock
def test_cohort_b_is_refused_without_the_unlock_flag():
    with pytest.raises(SystemExit) as e:
        holdout.check_unlocked(False, "youtube")
    msg = str(e.value)
    assert "REFUSED" in msg and "--unlock-holdout" in msg


def test_unlocking_returns_prior_evaluations(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "ledger_path", lambda: tmp_path / "h.jsonl")
    assert holdout.check_unlocked(True, "youtube") == []


def test_ledger_records_and_reads_back(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "ledger_path", lambda: tmp_path / "h.jsonl")
    results = {"weibull_aft": {"c_index": 0.71}, "km_median": {"c_index": 0.55}}
    entry = holdout.record("youtube", 120, 60, 30, results)
    assert entry["n_deaths"] == 60
    assert entry["c_index"]["weibull_aft"] == 0.71

    prior = holdout.prior_evaluations()
    assert len(prior) == 1
    assert prior[0]["platform"] == "youtube"


def test_a_second_evaluation_is_visible_in_the_ledger(tmp_path, monkeypatch):
    """The point of the ledger: a repeat look cannot be a private decision."""
    monkeypatch.setattr(holdout, "ledger_path", lambda: tmp_path / "h.jsonl")
    holdout.record("youtube", 120, 60, 30, {"m": {"c_index": 0.7}})
    holdout.record("youtube", 120, 60, 30, {"m": {"c_index": 0.7}})
    prior = holdout.check_unlocked(True, "youtube")
    assert len(prior) == 2


def test_digest_changes_when_results_change():
    a = holdout.results_digest({"m": {"c_index": 0.70}})
    b = holdout.results_digest({"m": {"c_index": 0.71}})
    assert a != b
    assert a == holdout.results_digest({"m": {"c_index": 0.70}})


def test_ledger_survives_a_corrupt_line(tmp_path, monkeypatch):
    p = tmp_path / "h.jsonl"
    p.write_text(json.dumps({"platform": "youtube"}) + "\nnot json\n",
                 encoding="utf-8")
    monkeypatch.setattr(holdout, "ledger_path", lambda: p)
    assert len(holdout.prior_evaluations()) == 1


def test_prior_evaluations_is_empty_when_no_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "ledger_path", lambda: tmp_path / "absent.jsonl")
    assert holdout.prior_evaluations() == []
