"""The Cohort C guard rails and the primary metrics, and the wall between tool and research."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cdc.tool import evaluate as E

SRC = Path(__file__).resolve().parents[1] / "src" / "cdc"


def _pred(p, y, n_creators=40):
    n = len(p)
    return pd.DataFrame({"post_id": [f"v{i}" for i in range(n)],
                         "creator_id": [f"c{i % n_creators}" for i in range(n)],
                         "faded": y, "fade_h": np.where(y, 24.0, 240.0),
                         "p_48": p, "avg_48": np.full(n, y.mean())})


def test_primary_passes_a_good_predictor_and_fails_a_useless_one():
    rng = np.random.default_rng(0)
    y = rng.random(1500) < 0.3
    good = E.primary(_pred(np.clip(np.where(y, 0.7, 0.15) + rng.normal(0, 0.1, 1500), 0, 1), y),
                     high=0.45, n_boot=300)
    assert good["passes"] and good["flag_precision"] > good["base_rate"]
    bad = E.primary(_pred(rng.random(1500), y), high=0.8, n_boot=300)
    assert not bad["odds_pass"] and bad["brier_skill"] < 0


def test_unknown_status_rows_are_left_out():
    y = np.array([True] * 50 + [False] * 150)
    d = _pred(np.where(y, 0.8, 0.1), y)
    d.loc[:9, ["faded", "fade_h"]] = [False, 24.0]          # followed only a day, not faded
    r = E.primary(d, high=0.5, n_boot=50)
    assert r["n_unknown_status_left_out"] == 10 and r["n_rows"] == 190


def test_refuses_without_a_frozen_card(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "CARD_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(E, "ledger_path", lambda: tmp_path / "ledger.jsonl")
    with pytest.raises(SystemExit, match="no frozen model card"):
        E.check_allowed()


def test_refuses_a_second_look_unless_asked(tmp_path, monkeypatch):
    card = tmp_path / "card.json"
    card.write_text("{}", encoding="utf-8")
    led = tmp_path / "ledger.jsonl"
    led.write_text(json.dumps({"evaluated_at": "2026-10-27T20:00:00+00:00"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(E, "CARD_PATH", card)
    monkeypatch.setattr(E, "ledger_path", lambda: led)
    with pytest.raises(SystemExit, match="already tested"):
        E.check_allowed()
    E.check_allowed(again=True)


def test_research_code_never_imports_the_tool():
    """The wall: nothing the paper's results depend on may reach into cdc.tool."""
    pat = re.compile(r"^\s*(from|import)\s+cdc\.tool\b", re.MULTILINE)
    for pkg in ("eval", "models", "labels", "collect", "transform", "storage", "bench", "site", "app"):
        for f in (SRC / pkg).rglob("*.py"):
            assert not pat.search(f.read_text(encoding="utf-8")), f"{f} imports cdc.tool"


def test_the_tool_does_not_write_into_research_outputs():
    """Every file the tool writes is in its own assets folder or a named gold file of its own."""
    text = "\n".join(f.read_text(encoding="utf-8") for f in (SRC / "tool").glob("*.py"))
    names = set(re.findall(r'_gold\("([^"]+)"\)', text)) | {E.LEDGER_NAME}
    assert names <= {"instant_rehearsal_B.json", "instant_cohortC.json", "instant_evaluations.jsonl"}
    allowed = ("CARD_PATH.write_text", "TIPS_PATH.write_text", "path.write_text",
               '_gold("instant_', "ledger_path().open(")
    writes = [l for l in text.splitlines() if "write_text(" in l or ".open(" in l or "to_parquet(" in l]
    assert writes and all(any(a in l for a in allowed) for l in writes), writes
