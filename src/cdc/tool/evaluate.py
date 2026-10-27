"""The instant model's checks: the A-to-B rehearsal, the freeze, and one Cohort C test.

**Exploratory extension, declared in the analysis plan's amendment log on
2026-10-27** (amendment 8), after the main study and both earlier tests were
designed, and before any Cohort C outcome had been computed. Everything that
decides the result is fixed here and in that amendment. Like the Cohort B
holdout and the amendment 7 test, the Cohort C run refuses a second look unless
asked, and records every run in a committed ledger.

The question, fixed in advance. A creator pastes a link to a video between 1
and 13 hours old; the tool has one reading of it and, usually, has never seen
the channel. **Do its odds of the video fading within two days of posting beat
quoting the average rate, and does its "high risk" label pick out videos that
really fade that fast?**

Design:

* **Channels the model has not seen.** Each Cohort C video is predicted by a
  model trained on Cohorts A and B *without that video's channel*: channels
  are split into five fixed groups (``table.creator_group``) and each group is
  predicted by a model trained without it. Nearly every Cohort C channel also
  appears in A or B, and a channel's subscriber count is distinctive enough to
  identify it, so testing on seen channels would measure memory of the channel,
  not what a new creator gets. The seen-channel result is reported as secondary.
* **Fixed in advance:** the recalibration weight ``b`` and the risk thresholds,
  computed from Cohorts A and B (``freeze``) and stored in
  ``assets/model_card.json``. The test reads them; it does not refit them.
* **Rows:** each video's reading at the 1, 3, 6 and 12-hour checkpoints.
* **Primary, both must pass (95% intervals, 2,000 creator-bootstrap draws):**
  1. *Odds beat the average rate.* Brier skill score of the two-day fade
     chance, against the average-curve rate from the same training rows: lower
     bound above 0.
  2. *The high-risk label works.* Among rows labelled high risk (two-day fade
     chance at or above the frozen threshold), the share that faded within two
     days, minus the share among all rows: lower bound above 0.
* Rows whose two-day status is unknown (followed for less than two days and
  not yet faded) are left out of both.

Usage::

    python -m cdc.tool.evaluate --rehearsal   # train A, test B (already examined): the dress rehearsal
    python -m cdc.tool.evaluate --freeze      # fit b and thresholds on A+B, write the shipped assets
    python -m cdc.tool.evaluate --status      # ledger and freeze state, no labels
    python -m cdc.tool.evaluate --cohort-c    # the single Cohort C test
    python -m cdc.tool.evaluate --finalize    # record it; on a pass, retrain on A+B+C
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform as _platform
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from cdc.tool import table as T
from cdc.tool.features import CHECKPOINTS, MARKS
from cdc.tool.model import (FAST_H, HORIZONS, SEED, InstantModel, col, fingerprint,
                            fit_b, held_out_predictions, known, out_of_fold, outcome,
                            thresholds)

CARD_PATH = T.ASSETS / "model_card.json"
LEDGER_NAME = "instant_evaluations.jsonl"
N_BOOT = 2000


# ------------------------------------------------------------------- metrics
def _creator_bootstrap(df: pd.DataFrame, sums: dict[str, np.ndarray], stat, n_boot: int,
                       seed: int = SEED) -> np.ndarray:
    """Resample channels with replacement; ``stat`` maps summed columns to numbers."""
    creators = df["creator_id"].to_numpy()
    names = np.array(sorted(set(creators)))
    pos = {c: i for i, c in enumerate(names)}
    idx = np.array([pos[c] for c in creators])
    per = {k: np.bincount(idx, weights=v.astype(float), minlength=len(names)) for k, v in sums.items()}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
        draws.append(stat({k: float(w @ v) for k, v in per.items()}))
    return np.array(draws, dtype=float)


def primary(pred: pd.DataFrame, high: float, n_boot: int = N_BOOT) -> dict[str, Any]:
    k = known(pred, FAST_H)
    d = pred[k]
    y = outcome(d, FAST_H).astype(float)
    p = d[f"p_{FAST_H:g}"].to_numpy()
    avg = d[f"avg_{FAST_H:g}"].to_numpy()
    flag = (p >= high).astype(float)
    sums = {"n": np.ones(len(d)), "y": y, "f": flag, "fy": flag * y,
            "se_m": (p - y) ** 2, "se_a": (avg - y) ** 2}

    def stat(s):
        prec = s["fy"] / s["f"] if s["f"] else np.nan
        base = s["y"] / s["n"]
        return [1 - s["se_m"] / s["se_a"], prec - base, prec, base]

    point = stat({k_: float(v.sum()) for k_, v in sums.items()})
    draws = _creator_bootstrap(d, sums, stat, n_boot)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    names = ("brier_skill", "flag_precision_minus_base", "flag_precision", "base_rate")
    out = {n: float(point[i]) for i, n in enumerate(names)}
    out.update({f"{n}_ci": [float(lo[i]), float(hi[i])] for i, n in enumerate(names)})
    out["odds_pass"] = bool(lo[0] > 0)
    out["flag_pass"] = bool(lo[1] > 0)
    out["passes"] = out["odds_pass"] and out["flag_pass"]
    out["n_rows"], out["n_videos"] = int(len(d)), int(d["post_id"].nunique())
    out["n_channels"] = int(d["creator_id"].nunique())
    out["n_flagged"], out["n_fast"] = int(flag.sum()), int(y.sum())
    out["share_flagged"] = float(flag.mean())
    out["lift"] = out["flag_precision"] / out["base_rate"] if out["base_rate"] else float("nan")
    out["n_unknown_status_left_out"] = int((~k).sum())
    return out


def secondary(pred: pd.DataFrame, card_thresholds: dict[str, float]) -> dict[str, Any]:
    """Everything reported but not tested: other horizons, calibration, ranking, by age."""
    from cdc.eval.validate import concordance
    out: dict[str, Any] = {}
    for h in HORIZONS:
        k = known(pred, h)
        y = outcome(pred, h)[k].astype(float)
        p, a, r = (pred[f"{s}_{h:g}"].to_numpy()[k] for s in ("p", "avg", "raw"))
        out[f"{h:g}h"] = {"observed": float(y.mean()), "predicted": float(p.mean()),
                          "brier_skill": float(1 - np.mean((p - y) ** 2) / np.mean((a - y) ** 2)),
                          "brier_skill_before_recalibration":
                              float(1 - np.mean((r - y) ** 2) / np.mean((a - y) ** 2))}
    out["ranking_c_index"] = float(concordance(pred["fade_h"].to_numpy(float), pred["faded"].to_numpy(bool),
                                               -pred[f"p_{FAST_H:g}"].to_numpy()))
    k = known(pred, FAST_H)
    cal = pd.DataFrame({"p": pred[f"p_{FAST_H:g}"].to_numpy()[k], "raw": pred[f"raw_{FAST_H:g}"].to_numpy()[k],
                        "y": outcome(pred, FAST_H)[k].astype(float)})
    cal["bin"] = pd.qcut(cal["p"], 10, labels=False, duplicates="drop")
    out["calibration_deciles"] = [
        {"predicted": float(g["p"].mean()), "observed": float(g["y"].mean()), "n": int(len(g)),
         "predicted_before_recalibration": float(g["raw"].mean())}
        for _, g in cal.groupby("bin")]
    low = cal["p"] <= card_thresholds["low"]
    out["low_risk_label"] = {"share_labelled": float(low.mean()),
                             "faded_within_2_days": float(cal.loc[low, "y"].mean()) if low.any() else None,
                             "base_rate": float(cal["y"].mean())}
    by = {}
    for c in sorted(pred["mark"].dropna().unique()):
        sub = pred[pred["mark"] == c]
        kk = known(sub, FAST_H)
        y = outcome(sub, FAST_H)[kk].astype(float)
        p = sub[f"p_{FAST_H:g}"].to_numpy()[kk]
        a = sub[f"avg_{FAST_H:g}"].to_numpy()[kk]
        f = p >= card_thresholds["high"]
        by[f"{c:g}h"] = {"rows": int(kk.sum()), "brier_skill": float(1 - np.mean((p - y) ** 2) / np.mean((a - y) ** 2)),
                         "flag_precision": float(y[f].mean()) if f.any() else None,
                         "base_rate": float(y.mean()), "share_flagged": float(f.mean())}
    out["by_age"] = by
    return out


def _predict_extra(train: pd.DataFrame, extra: pd.DataFrame, b: float) -> pd.DataFrame | None:
    if extra is None or extra.empty:
        return None
    return held_out_predictions(train, extra, b=b)


def assess(train: pd.DataFrame, test: pd.DataFrame, b: float, th: dict[str, float],
           offgrid: pd.DataFrame | None = None, simulated: pd.DataFrame | None = None,
           n_boot: int = N_BOOT) -> dict[str, Any]:
    pred = held_out_predictions(train, T.model_rows(test), b=b)
    res = {"unseen_channels": {"primary": primary(pred, th["high"], n_boot),
                               "secondary": secondary(pred, th)}}
    # Seen channels: one model on all training rows, as the deployed app would be.
    full = InstantModel(b=b).fit(train)
    seen = T.model_rows(test).copy()
    raw, s = full.raw_survival(seen), full.survival(seen)
    for h in HORIZONS:
        seen[f"raw_{h:g}"], seen[f"p_{h:g}"] = 1 - raw[:, col(h)], 1 - s[:, col(h)]
        seen[f"avg_{h:g}"] = 1 - full.average_[col(h)]
    res["seen_channels"] = {"primary_metrics_descriptive": primary(seen, th["high"], n_boot)}
    for name, extra in (("real_readings_between_checkpoints", offgrid),
                        ("simulated_readings_at_2_4.5_9h", simulated)):
        p = _predict_extra(train, extra, b)
        if p is not None and known(p, FAST_H).sum() >= 30:
            res[name] = {"primary_metrics_descriptive": primary(p, th["high"], n_boot)}
        else:
            res[name] = {"rows": 0 if p is None else int(len(p)), "note": "too few rows to score"}
    return res


# -------------------------------------------------------------- the three runs
def rehearsal(n_boot: int = N_BOOT, write: bool = True) -> dict:
    """Train A, test B, exactly as Cohort C will be run. B was examined long ago."""
    vids = T.labelled_videos(("A", "B"))
    rows = T.build_rows(videos=vids)
    a, bb = rows[rows["cohort"] == "A"], rows[rows["cohort"] == "B"]
    oof = out_of_fold(a)
    b = fit_b(oof)
    th = thresholds(oof, b)
    vb = vids[vids["cohort"] == "B"]
    out = {"evaluated_at": _now(), "kind": "rehearsal: train A (b and thresholds from A), test B",
           "b": b, "thresholds": th,
           "results": assess(a, bb, b, th, T.offgrid_rows(("B",), videos=vb),
                             T.simulated_rows(("B",), videos=vb), n_boot)}
    _print(out)
    if write:
        _gold("instant_rehearsal_B.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out


def freeze(n_boot: int = N_BOOT) -> dict:
    """Fix b and the thresholds on A+B, and write the table and card the app ships with."""
    rows = T.build_rows(("A", "B"))
    oof = out_of_fold(rows)
    b = fit_b(oof)
    th = thresholds(oof, b)
    anon = T.anonymise(rows)
    table_sha = T.save_table(anon)
    # The app trains from the saved file; prove that gives the same model as the rows.
    m_rows = InstantModel(b=b).fit(rows)
    m_file = InstantModel(b=b).fit(T.load_table())
    fp_rows, fp_file = fingerprint(m_rows), fingerprint(m_file)
    if fp_rows != fp_file:
        raise SystemExit(f"model from the saved table differs from the model from the rows "
                         f"({fp_file} vs {fp_rows}); not freezing")
    oof_b = oof.copy()
    for h in HORIZONS:
        raw_s = np.clip(1 - oof_b[f"raw_{h:g}"].to_numpy(), 1e-6, 1)
        avg_s = np.clip(1 - oof_b[f"avg_{h:g}"].to_numpy(), 1e-6, 1)
        oof_b[f"p_{h:g}"] = 1 - np.exp(b * np.log(raw_s) + (1 - b) * np.log(avg_s))
    reh = json.loads(_gold("instant_rehearsal_B.json").read_text(encoding="utf-8")) \
        if _gold("instant_rehearsal_B.json").exists() else None
    card = {
        "frozen_at": _now(),
        "status": "frozen; awaiting the single Cohort C test",
        "trained_on": {"cohorts": ["A", "B"], "rows": int(len(T.model_rows(rows))),
                       "videos": int(rows["video_key"].nunique()),
                       "channels": int(rows["creator_id"].nunique()),
                       "table_rows_all_marks": int(len(anon)), "table_sha256": table_sha},
        "forest": {"n_estimators": m_rows.n_estimators, "min_samples_leaf": m_rows.min_samples_leaf,
                   "max_features": m_rows.max_features, "random_state": m_rows.random_state},
        "b": b, "thresholds_p48": th, "fast_h": FAST_H, "horizons_h": list(HORIZONS),
        "checkpoints_h": list(CHECKPOINTS), "marks_h": list(MARKS),
        "average_curve": dict(zip([f"{m:g}" for m in MARKS], m_rows.average_.round(4).tolist())),
        "fingerprint": fp_rows,
        "versions": _versions(),
        "held_out_channels_A_plus_B": {"primary": primary(oof_b, th["high"], n_boot),
                                       "secondary": secondary(oof_b, th)},
        "rehearsal_A_to_B": None if reh is None else {
            "b": reh["b"], "thresholds": reh["thresholds"],
            "primary": reh["results"]["unseen_channels"]["primary"],
            "ranking_c_index": reh["results"]["unseen_channels"]["secondary"]["ranking_c_index"]},
        "cohort_c": None,
    }
    CARD_PATH.write_text(json.dumps(card, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"froze b={b}, high>={th['high']:.4f}, low<={th['low']:.4f}, fingerprint {fp_rows}")
    print(f"  wrote {T.TABLE_PATH.relative_to(T.ASSETS.parents[3])} and {CARD_PATH.name}")
    return card


def ledger_path():
    from cdc.config import path_for
    return path_for("gold_dir") / LEDGER_NAME


def prior_runs() -> list[dict]:
    p = ledger_path()
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_allowed(again: bool = False) -> None:
    from cdc.eval.practical import NOT_BEFORE
    if pd.Timestamp.now(tz="UTC") < NOT_BEFORE:
        raise SystemExit(f"REFUSED - Cohort C is not mature before {NOT_BEFORE.isoformat()}.")
    if not CARD_PATH.exists():
        raise SystemExit("REFUSED - no frozen model card. Run --freeze and commit it first.")
    if prior_runs() and not again:
        raise SystemExit(f"\nREFUSED - the instant model was already tested on Cohort C at "
                         f"{prior_runs()[0].get('evaluated_at')}.\n  A second run is a second look "
                         "and must be reported as one. Pass --again only if you mean that.\n")


def cohort_c(again: bool = False, n_boot: int = N_BOOT, write: bool = True) -> dict:
    check_allowed(again)
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    vids = T.labelled_videos(("A", "B", "C"))
    rows = T.build_rows(videos=vids)
    train, test = rows[rows["cohort"].isin(["A", "B"])], rows[rows["cohort"] == "C"]
    # The training data must be exactly what was frozen.
    sha = T.digest(T.anonymise(train))
    if sha != card["trained_on"]["table_sha256"]:
        raise SystemExit(f"REFUSED - Cohorts A+B no longer match the frozen table "
                         f"({sha[:12]} vs {card['trained_on']['table_sha256'][:12]}).")
    b, th = float(card["b"]), card["thresholds_p48"]
    vc = vids[vids["cohort"] == "C"]
    out = {"evaluated_at": _now(), "kind": "Cohort C, single evaluation",
           "git_commit": _git_commit(), "b": b, "thresholds": th,
           "results": assess(train, test, b, th, T.offgrid_rows(("C",), videos=vc),
                             T.simulated_rows(("C",), videos=vc), n_boot)}
    _print(out)
    if write:
        _gold("instant_cohortC.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        head = out["results"]["unseen_channels"]["primary"]
        digest = hashlib.sha256(json.dumps(out["results"], sort_keys=True, default=str).encode()).hexdigest()[:12]
        entry = {"evaluated_at": out["evaluated_at"], "git_commit": out["git_commit"],
                 "n_rows": head["n_rows"], "n_videos": head["n_videos"], "results_digest": digest,
                 "odds_pass": head["odds_pass"], "flag_pass": head["flag_pass"], "passes": head["passes"]}
        with ledger_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(f"\n  recorded in data/gold/{LEDGER_NAME} (digest {digest}). Commit it.")
    return out


def finalize(n_boot: int = N_BOOT) -> dict:
    """After the Cohort C test: record it in the card; on a pass, retrain on A+B+C.

    Declared in amendment 8. A pass means the same recipe (forest, held-out-
    channel recalibration, thresholds) is refitted on every labelled video,
    which is what the app then serves; the tested numbers stay those of the
    A+B model, and the card keeps both. A fail leaves the A+B model in place
    with the odds switched off.
    """
    if not prior_runs():
        raise SystemExit("REFUSED - there is no Cohort C run in the ledger to record.")
    res = json.loads(_gold("instant_cohortC.json").read_text(encoding="utf-8"))
    tested = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    u = res["results"]["unseen_channels"]
    summary = {"evaluated_at": res["evaluated_at"], "git_commit": res["git_commit"],
               "passes": bool(u["primary"]["passes"]), "unseen_channels": u["primary"],
               "ranking_c_index": u["secondary"]["ranking_c_index"],
               "seen_channels": res["results"]["seen_channels"]["primary_metrics_descriptive"],
               "tested_model": {k: tested[k] for k in ("b", "thresholds_p48", "fingerprint", "trained_on")}}
    if not summary["passes"]:
        tested.update(status="failed its single Cohort C test; the app shows no odds", cohort_c=summary)
        CARD_PATH.write_text(json.dumps(tested, indent=2, default=float) + "\n", encoding="utf-8")
        print("recorded a failed Cohort C test; odds switched off")
        return tested
    rows = T.build_rows(("A", "B", "C"))
    oof = out_of_fold(rows)
    b = fit_b(oof)
    th = thresholds(oof, b)
    sha = T.save_table(T.anonymise(rows))
    model = InstantModel(b=b).fit(T.load_table())
    card = dict(tested)
    card.update({
        "finalized_at": _now(),
        "status": "passed its single Cohort C test; retrained on Cohorts A+B+C with the same recipe",
        "trained_on": {"cohorts": ["A", "B", "C"], "rows": int(len(T.model_rows(rows))),
                       "videos": int(rows["video_key"].nunique()),
                       "channels": int(rows["creator_id"].nunique()),
                       "table_rows_all_marks": int(len(rows)), "table_sha256": sha},
        "b": b, "thresholds_p48": th,
        "average_curve": dict(zip([f"{m:g}" for m in MARKS], model.average_.round(4).tolist())),
        "fingerprint": fingerprint(model), "versions": _versions(), "cohort_c": summary,
    })
    CARD_PATH.write_text(json.dumps(card, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"retrained on A+B+C: b={b}, high>={th['high']:.4f}, fingerprint {card['fingerprint']}")
    return card


def status() -> dict:
    card = json.loads(CARD_PATH.read_text(encoding="utf-8")) if CARD_PATH.exists() else None
    return {"frozen": card is not None, "frozen_at": card and card["frozen_at"],
            "b": card and card["b"], "fingerprint": card and card["fingerprint"],
            "cohort_c_runs": len(prior_runs())}


# ------------------------------------------------------------------- helpers
def _gold(name: str):
    from cdc.config import path_for
    return path_for("gold_dir") / name


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str | None:
    from cdc.eval.practical import _git_commit as g
    return g()


def _versions() -> dict[str, str]:
    import numpy, pandas, scipy, sklearn, sksurv
    return {"python": _platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "scikit-survival": sksurv.__version__}


def _print(out: dict) -> None:
    print("=" * 74)
    print(f"  INSTANT MODEL - {out['kind']}")
    print("=" * 74)
    print(f"  recalibration b = {out['b']}, high risk = two-day chance >= {out['thresholds']['high']:.3f}")
    r = out["results"]
    p = r["unseen_channels"]["primary"]
    print(f"\n  channels the model never saw: {p['n_rows']} readings, {p['n_videos']} videos, "
          f"{p['n_channels']} channels")
    print(f"    odds vs average rate   Brier skill {p['brier_skill']:+.3f} "
          f"[{p['brier_skill_ci'][0]:+.3f}, {p['brier_skill_ci'][1]:+.3f}]   {'PASS' if p['odds_pass'] else 'no'}")
    print(f"    high-risk label        {p['flag_precision']:.1%} faded within 2 days vs {p['base_rate']:.1%} "
          f"overall ({p['lift']:.2f}x), gap [{p['flag_precision_minus_base_ci'][0]:+.3f}, "
          f"{p['flag_precision_minus_base_ci'][1]:+.3f}]   {'PASS' if p['flag_pass'] else 'no'}")
    print(f"    labelled high risk     {p['share_flagged']:.1%} of readings")
    s = r["unseen_channels"]["secondary"]
    print(f"    ranking C-index        {s['ranking_c_index']:.3f}")
    for h in HORIZONS:
        x = s[f"{h:g}h"]
        print(f"    by {h:>3g}h  observed {x['observed']:.3f}  predicted {x['predicted']:.3f}  "
              f"Brier skill {x['brier_skill']:+.3f} (before recalibration {x['brier_skill_before_recalibration']:+.3f})")
    for name in ("seen_channels", "real_readings_between_checkpoints", "simulated_readings_at_2_4.5_9h"):
        q = r[name].get("primary_metrics_descriptive")
        if q:
            print(f"  {name.replace('_', ' ')}: Brier skill {q['brier_skill']:+.3f}, high-risk "
                  f"{q['flag_precision']:.1%} vs {q['base_rate']:.1%} ({q['n_rows']} readings)")
        else:
            print(f"  {name.replace('_', ' ')}: {r[name]['note']} ({r[name]['rows']})")
    print(f"\n  PRIMARY: {'PASS' if p['passes'] else 'DID NOT PASS'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Instant model checks (exploratory, amendment 8).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rehearsal", action="store_true")
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--cohort-c", action="store_true")
    g.add_argument("--finalize", action="store_true")
    ap.add_argument("--again", action="store_true")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    if a.status:
        print(json.dumps(status(), indent=2))
    elif a.rehearsal:
        rehearsal(a.boot)
    elif a.freeze:
        freeze(a.boot)
    elif a.finalize:
        finalize(a.boot)
    else:
        cohort_c(again=a.again, n_boot=a.boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
