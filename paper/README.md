# Papers

Two deliverables, one dataset, one repo.

| | File | Course | Status |
|---|---|---|---|
| Empirical paper | [`rm/paper.md`](rm/paper.md) | Research Methodology | Method, Results, Discussion and Conclusion written; **Cohort A figures provisional until 2026-09-30**; Cohort B sealed |
| Technical report | [`bda/report.md`](bda/report.md) | Big Data Analytics | Method + benchmark final; scale-up numbers final |

## How to read the drafts

Everything marked **`[PENDING]`** is a number that does not exist yet and must not be
invented. One remains, in §4.7: the Cohort B holdout, which is sealed.

Cohort A closed **2026-09-16T00:00:00Z** (pre-specified in `settings.yaml` on
2026-08-30) and its results are written — but they are **provisional**, because
membership froze on that date while outcomes did not. Posts published shortly before it
are still inside their 336-hour observation window, so deaths will rise and the censoring
rate will fall until the last member ages out on **2026-09-29**. The final run happens on
or after 2026-09-30. `cdc.eval.report` prints a banner and stamps
`"provisional": true` into the JSON whenever a cohort has not matured.

Cohort B closes to admissions on **2026-10-04**, matures on or after **2026-10-18**, and
is evaluated exactly once after that.

Everything **not** marked pending is real, measured, and traceable to a file in this repo.
Where a figure comes from a command, the command is named next to it so a reader can
re-derive it.

## The rule we are holding ourselves to

The Method sections are written **before** the results, and they are written from
`ANALYSIS_PLAN.md` (frozen 2026-08-30) plus its two dated amendments. If the results
disappoint, the Method does not change. That is the whole point of having frozen it.

The dress-rehearsal output in Appendix B of the RM paper is included **as evidence that
the pipeline runs**, labelled as such, and is not a finding. It has 7 observed deaths.
