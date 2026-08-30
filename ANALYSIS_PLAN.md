# Pre-registered Analysis Plan

**Content Death Clock** — RV University, 7th Semester
Research Methodology + Big Data Analytics

> **Commit this file, unchanged, before examining any outcome data.** Its value comes
> entirely from being timestamped in git *before* we know what the results look like. It
> is the cheapest credibility we can buy, and it is the difference between "we tested a
> hypothesis" and "we found something after trying several things".
>
> Amendments are allowed and expected — but they must be made as *new commits with stated
> reasons*, never by editing history. The paper reports the amendments.

| | |
|---|---|
| **Status** | ⬜ DRAFT — not yet frozen |
| **Frozen at commit** | _(fill in: `git rev-parse HEAD` on the freezing commit)_ |
| **Frozen on** | _(date)_ |
| **Authors** | _(names)_ |

---

## 1. Research question

Can the time at which a social media post stops receiving meaningful attention be
predicted from signals observable within the first six hours after publication?

## 2. Hypotheses

**H1 (primary).** A model using early-engagement and metadata features observed at
t ≤ 6h predicts time-to-attention-death more accurately than a naive constant-lifetime
baseline.

- *H1₀*: model performance ≤ best naive baseline performance.
- *H1₁*: model performance > best naive baseline performance.

**H2 (secondary).** Early engagement *velocity* carries predictive information beyond
creator size alone — i.e. the model beats a follower-count-only baseline.

> H2 is the hypothesis that actually matters. Beating "everything dies at 48 hours" is
> trivially easy and proves almost nothing; beating creator size proves the *dynamics*
> carry signal.

**H3 (exploratory, not confirmatory).** Time-to-death differs across creator-size strata
and content categories. Reported as exploratory; not used to support the main claim.

## 3. Design

Observational, prospective panel. Non-experimental — we predict decay, we do not
manipulate anything, and no causal claim is made.

## 4. Sample

- **Frame**: public YouTube channels listed in `config/channels.resolved.yaml`, stratified
  by subscriber tier (micro <10k / mid 10k–500k / large >500k) and content category;
  plus public Instagram accounts if Scrape Creators credits permit.
- **Target**: ~800–1500 YouTube posts across 60–100 channels; ~200 Instagram posts.
- **Admission**: any post published by a frame channel during the collection window and
  discovered within `discovery_lookback_hours` of publication.
- **Observation**: snapshots at t+1, 3, 6, 12, 24, 36, 48, 72, 96, 120, 168, 240, 336h.

**Exclusions**, all pre-specified, all counted and reported in a sample-attrition table:

| Rule | Reason code |
|---|---|
| Fewer than 4 usable inter-snapshot intervals | `insufficient_observations` |
| Never achieved positive view velocity | `no_positive_velocity` |
| Post deleted or made private during observation | recorded as censored at last observation |

## 5. Variables

**Dependent.** `t_death` — first time engagement velocity falls below 5% of that post's
own peak velocity and remains below for 2 consecutive intervals. Right-censored at last
observation if never reached. Velocity uses **views** as the primary metric (monotonic,
available on both platforms, cannot be disabled by the creator) and is computed on
**actual observed timestamps**.

**Robustness DV.** `t_saturation` — time to reach 90% of the asymptote A from a fitted
C(t) = A(1 − e^(−kt)). Reported alongside. Systematic disagreement between the two is a
finding to report, not a problem to hide.

**Independent — hard constraint: nothing observed after t = 6h.** Early velocity at 1h,
3h, 6h; log growth ratios between consecutive snapshots; acceleration; video duration;
title length; tag count; publish hour and weekday; category; creator subscriber count;
creator historical median velocity.

## 6. Analysis

**Models.** Weibull AFT and Random Survival Forest (primary, censoring-aware);
log-time linear regression and gradient boosting (comparators, uncensored subset).

**Baselines**, all four reported:
1. Constant 48h ("all content dies at 48 hours")
2. Training-set median lifetime
3. Subscriber-count-only model
4. Peak-velocity heuristic

**Validation.** `GroupKFold(n_splits=5)` **grouped by creator**. Random k-fold would place
the same creator in train and test, leaking creator identity and inflating scores; this is
pre-specified precisely so it cannot be quietly relaxed later.

**Temporal holdout.** Posts collected after the Cohort A freeze date form Cohort B and are
evaluated **exactly once**, at the end, after all model selection is complete.

**Metrics.** Harrell's C-index (primary, censoring-aware); MAE and RMSE on log₁₀
time-to-death (uncensored subset); calibration plot of predicted vs observed.

**Inference.** Paired Wilcoxon signed-rank test on per-post absolute errors, model vs each
baseline. Bootstrap (2,000 resamples, resampled by creator) for confidence intervals on
metric differences. α = 0.05. Effect sizes reported alongside p-values; a p-value alone
will not be treated as a result.

**Multiplicity.** H1 and H2 are the only confirmatory tests. Everything else is labelled
exploratory in the paper.

## 7. Pre-specified stopping rule

Collection ends at the Cohort A freeze date, set in advance in `settings.yaml`
(`modelling.cohort_a_freeze_utc`) — **not** when results look good.

## 8. What we will report regardless of outcome

- If the model does **not** beat the baselines, that is the reported result. A negative
  finding, honestly reported with an accompanying power discussion, is a legitimate RM
  paper. Searching for a specification that wins is not.
- Censoring rate, sample attrition by reason code, collection completeness and outages,
  and the count of monotonicity repairs (retracted view counts) — all reported.
- Any deviation from this plan, with its reason and its date.

## 9. Known limitations, stated in advance

Observational design; no causal claim. Convenience sample of channels — not a random
sample of YouTube, so generalisation beyond the frame is not claimed. Views are a
platform-reported metric we cannot independently audit. Scrape Creators is an unofficial
third-party API. The observation window truncates genuinely long-lived content, which is
what censoring-aware modelling is there to handle and what the retrospective cohort is
there to bound.

---

## Amendment log

| Date | Commit | Change | Reason |
|---|---|---|---|
| | | | |
