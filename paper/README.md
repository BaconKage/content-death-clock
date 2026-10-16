# Papers

Two deliverables, one dataset, one repo.

| | File | Course | Status |
|---|---|---|---|
| Empirical paper | [`rm/paper.md`](rm/paper.md) | Research Methodology | Written in full; **Cohort A results final** (matured, 2026-10-16); figures in `rm/figures/`; Cohort B sealed, evaluation declared, runs on/after 2026-10-18 |
| Technical report | [`bda/report.md`](bda/report.md) | Big Data Analytics | Method + benchmark final; scale-up numbers final |

## How to read the drafts

Everything marked **`[PENDING]`** is a number that does not exist yet and must not be
invented. One remains, in §4.7: the Cohort B holdout, which is sealed.

Cohort A closed **2026-09-16T00:00:00Z** (pre-specified in `settings.yaml` on
2026-08-30). Every Cohort A post has completed observation, and the results in the RM paper
are the **final, matured** ones (686 posts, 563 deaths), regenerated on 2026-10-16. The
provisional 2026-09-20 figures (498 deaths) are superseded. `cdc.eval.report` stamps
`"provisional": true` into the JSON whenever a cohort has not matured; the current artifact
says `false`.

Cohort B closed to admissions on **2026-10-04**. Its evaluation was declared in full on
2026-10-16 (plan amendment 6), and it is evaluated exactly once, on or after **2026-10-18**.

The figures are regenerated from the gold artifacts with `python -m cdc.eval.figures`.

`RESULTS-EXPLAINED.md` is still written from the provisional run and is marked as such.

Everything **not** marked pending is real, measured, and traceable to a file in this repo.
Where a figure comes from a command, the command is named next to it so a reader can
re-derive it.

## The rule we are holding ourselves to

The Method sections are written **before** the results, and they are written from
`ANALYSIS_PLAN.md` (frozen 2026-08-30) plus its dated amendments. If the results
disappoint, the Method does not change. That is the whole point of having frozen it.

The dress-rehearsal output in Appendix B of the RM paper is included **as evidence that
the pipeline runs**, labelled as such, and is not a finding. It has 7 observed deaths.
