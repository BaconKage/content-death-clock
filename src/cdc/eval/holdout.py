"""Guard rails for Cohort B, the temporal holdout.

The frozen plan says Cohort B is "evaluated **exactly once**, at the end, after
all model selection is complete." That sentence is worth nothing if the command
that evaluates it can be run casually, repeatedly, while the model is still
being tuned. A holdout that can be peeked at is not a holdout — each look leaks
a little of it into the choices made next, and by the fifth look it has quietly
become a second validation set.

So the promise is enforced rather than trusted:

* **Cohort B requires an explicit unlock flag.** Running the evaluation the
  ordinary way can never touch it by accident.
* **Every evaluation is appended to a committed ledger**, with the timestamp,
  the git commit, the sample size and a digest of the results. The ledger is
  data, not a lock — anyone determined can run it twice — but the second run is
  then a matter of public record rather than a private decision.
* **A second run warns loudly**, names the date of the first, and requires the
  unlock flag again.

This is the same reasoning as the pre-registration itself. The point is not to
make dishonesty impossible; it is to make it visible, including to ourselves.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cdc.config import path_for

LEDGER_NAME = "holdout_evaluations.jsonl"


def ledger_path() -> Path:
    return path_for("gold_dir") / LEDGER_NAME


def prior_evaluations() -> list[dict[str, Any]]:
    """Every previous Cohort B evaluation, oldest first."""
    p = ledger_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def results_digest(results: dict[str, Any]) -> str:
    """Short stable hash of the metric table, so two runs can be compared.

    If the digest differs between runs, the holdout was evaluated against
    different code or different data — which is exactly the thing a reader
    would want to know about.
    """
    blob = json.dumps(results, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def check_unlocked(unlocked: bool, platform: str) -> list[dict[str, Any]]:
    """Refuse to proceed unless the holdout was unlocked deliberately.

    Returns the prior evaluations so the caller can warn about them.
    """
    prior = [e for e in prior_evaluations() if e.get("platform") == platform]
    if not unlocked:
        raise SystemExit(
            "\n"
            "  REFUSED: Cohort B is the temporal holdout.\n"
            "\n"
            "  The frozen analysis plan commits to evaluating it exactly once,\n"
            "  after all model selection is complete. If you are still choosing\n"
            "  models, features, or hyper-parameters, this is not that moment.\n"
            "\n"
            "  If it genuinely is, pass --unlock-holdout. The run will be\n"
            f"  recorded in data/gold/{LEDGER_NAME}, which is committed.\n"
            + (f"\n  NOTE: this holdout has already been evaluated "
               f"{len(prior)} time(s).\n" if prior else "")
        )
    return prior


def record(platform: str, n_posts: int, n_deaths: int, n_creators: int,
           results: dict[str, Any]) -> dict[str, Any]:
    """Append this evaluation to the ledger. Called after a Cohort B run."""
    entry = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform,
        "git_commit": _git_commit(),
        "n_posts": n_posts,
        "n_deaths": n_deaths,
        "n_creators": n_creators,
        "results_digest": results_digest(results),
        "c_index": {k: v.get("c_index") for k, v in results.items()},
    }
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def warn_if_repeated(prior: list[dict[str, Any]]) -> None:
    if not prior:
        return
    first = prior[0]
    print()
    print("  " + "!" * 64)
    print("  !! THIS HOLDOUT HAS ALREADY BEEN EVALUATED.")
    print(f"  !! First evaluated: {first.get('evaluated_at')} "
          f"(commit {str(first.get('git_commit'))[:8]})")
    print(f"  !! Previous runs: {len(prior)}")
    print("  !! The plan commits to evaluating Cohort B exactly once. A second")
    print("  !! look is no longer an untouched holdout, and the paper must say")
    print("  !! so — report every evaluation, not just the one you prefer.")
    print("  " + "!" * 64)
