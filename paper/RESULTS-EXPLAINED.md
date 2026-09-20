# Content Death Clock — What We Found

**RV University · 7th Semester · Research Methodology**
Amogha V Prasad · S Anannya · Sanidhya Tiwari · Shubhang Srinivas Varda

Cohort A analysis, run 20 September 2026. Every number here comes from
`data/gold/evaluation_youtube.json`, reproducible with:

```bash
python -m cdc.eval.report --platform youtube
```

---

## The short version

We asked whether you can watch a post's first six hours and predict when it will
stop getting attention.

**The answer is: a little, but not much — and most of what you can predict, you
could already have guessed from the size of the channel.**

That is a real answer, and we are reporting it as we promised to on 30 August,
before we had seen any results.

---

## 1. What we were testing

Two questions, written down and frozen in `ANALYSIS_PLAN.md` on 30 August 2026,
before a single death label had been computed.

**H1 — Can we beat "every post dies at about the same time"?**
The dumbest possible prediction is to ignore the post entirely and say every
post lives roughly 48 hours. Can a real model do better?

**H2 — Do the first six hours tell us anything that subscriber count doesn't?**
This is the one that matters. Beating "everything dies at 48 hours" is easy and
proves almost nothing. But if watching how a post actually *behaves* early on
tells us nothing beyond "this is a big channel," then there is no point watching.

---

## 2. What we had to work with

| | |
|---|---|
| Posts analysed | **686** |
| Posts we actually saw die | **498** |
| Posts still alive when we stopped watching | 188 (27%) |
| Channels | 47 |
| Cut-off date (fixed in advance) | 16 September 2026 |

We needed at least 50 observed deaths for the results to mean anything. We got
498, so the sample is comfortably large enough.

### Where the other posts went

Nothing was quietly dropped — every exclusion was a rule written down in
advance, and here is the full accounting:

```
posts collected on YouTube                1086
  minus: more than 5 posts from one         -99   (one channel uploading in bulk
         channel in one day                        would otherwise dominate)
  minus: published after the cut-off       -234   (these are held back as the
                                                   sealed holdout set)
  minus: already dead before the 7h         -58   (you cannot predict something
         prediction point                          that has already happened)
  minus: too few observations                -7
  minus: never gained any views              -2
= posts used in the analysis                686
```

---

## 3. The main result, in plain terms

The clearest way to read our headline score: **show the model two posts and ask
which one will die first.** How often does it get that right?

| Method | Gets it right |
|---|---|
| Flipping a coin | 50% |
| Just knowing the channel's subscriber count | **60%** |
| Weibull AFT model (our first model) | **60%** |
| **Random Survival Forest (our best model)** | **62%** |

That is the whole finding in one table.

Watching a post's real early behaviour — how fast views arrived, whether it was
speeding up or slowing down — got us from 60% to 62%. The statistics say that
2-point gain is probably real and not luck. But it is small.

**Subscriber count alone does almost all of the work.**

---

## 4. The part we did not expect

Ranking posts is one thing. Actually saying *when* a post will die is another,
and here the result is worse.

When our models predict a time, they are typically **off by a factor of about
two**. A post that really dies at 20 hours might be predicted at 9 hours, or at
43.

And here is the uncomfortable bit:

| Method | How far off, typically |
|---|---|
| **Just guessing the typical lifetime** (a single number, same for every post) | **factor of 2.07** |
| Random Survival Forest | factor of 2.13 |
| "Everything dies at 48 hours" | factor of 2.17 |
| Weibull AFT | factor of 2.23 |

**The simplest possible method — one number, the same for every post — is the
most accurate.** Our models are significantly *worse* than it (p = 0.0006 for
the Weibull, p = 0.006 for the forest).

This is not a contradiction. The models are better at putting posts *in order*
but worse at naming a *number*, because a model that spreads its guesses around
takes on extra error whenever the signal is weak. When there is not much to
predict, betting on the average is hard to beat.

It is an honest and slightly humbling result, and it goes in the paper.

---

## 5. So what happened to each hypothesis?

### H1 — partly supported

| Judged on | Verdict |
|---|---|
| Ranking which post dies first | ✅ **Supported.** Both models beat a constant, and the confidence intervals exclude a coin flip. |
| Predicting the actual time | ❌ **Not supported.** No better than "everything dies at 48 hours" (p = 0.40 and p = 0.55), and significantly worse than guessing the typical lifetime. |

### H2 — weakly supported, and only for one model

| Model | vs subscriber count | Verdict |
|---|---|---|
| Random Survival Forest | p = 0.019 | ✅ Beats it — the one test that passes |
| Weibull AFT | p = 0.76 | ❌ No difference at all |

The forest scores 62% against subscriber count's 60%. The test says that gap is
statistically real. But the confidence intervals around those two numbers
overlap across most of their range, so we describe this as **weak support, not
a clear win.**

**Plain summary:** early engagement dynamics carry a small, detectable amount of
extra information — but creator size accounts for most of what is predictable,
and the effect is small enough that a different modelling choice makes it
disappear.

---

## 6. A bonus finding from our cross-check

We measured "death" a second way as a safety check — instead of watching the
speed of views collapse, we fitted a growth curve and asked when the post
reached 90% of its final total.

The two measures agree on the ordering of posts (correlation **+0.72**, very
strongly significant). But they disagree systematically on timing: the
growth-curve method says posts finish at about **63%** of the time the
velocity method gives.

That is not a bug. It is a genuine observation: **a post stops meaningfully
growing well before it stops getting attention entirely.** Those are two
different ideas of "finished," and we can now show the gap between them with
data.

Our frozen plan said in advance that disagreement between the two measures would
be a finding to report, not a problem to hide. So here it is.

---

## 7. Why this is a good result, not a failed project

It is tempting to read "our model only just beats subscriber count" as failure.
It is not, for three reasons.

**We committed to this answer in advance.** On 30 August, before any death label
existed, we wrote: *"If the model does not beat the baselines, that is the
reported result. Searching for a specification that wins is not."* We could have
quietly tried twenty variations until one looked impressive. We did not, and the
git history proves we did not.

**A negative-leaning result is still a finding about the world.** We can now say
something specific and evidence-backed: *how long a post holds attention is
mostly determined by who posted it, not by how it started.* That is more useful
to a creator than a countdown clock that is wrong by a factor of two.

**We found the limit honestly.** Every method we tried — including the cleverest
and the dumbest — lands within a whisker of the same absolute accuracy. That is
strong evidence that the problem itself is hard, not that we modelled it badly.

---

## 8. What we can and cannot claim

**We can say:**
- Time-to-attention-death can be ranked slightly better than chance from signals
  in a post's first six hours.
- Creator size accounts for most of that predictive power.
- Absolute predictions remain imprecise (typically wrong by ~2×).
- A post stops growing before it stops receiving attention.

**We cannot say:**
- That early engagement *causes* anything. This is an observational study; we
  changed nothing and manipulated nothing.
- That these findings generalise to all of YouTube. Our 47 channels are not a
  random sample — they are the channels reachable by a fixed, committed list of
  search queries.
- Anything about Instagram. It was a feasibility demonstration only; no
  hypothesis was tested on it, and it uses likes rather than views, so the two
  platforms are never pooled.
- Anything yet about the holdout set. It stays sealed.

---

## 9. What happens next

| Step | Status |
|---|---|
| Cohort A analysis | ✅ Done — this document |
| Write up results in the RM paper | Next |
| Cohort B holdout — 234 posts, sealed | Opened **once**, at the very end |
| Collection | Still running |

The holdout is the last piece. We look at it exactly one time, after every
modelling decision is final, and whatever it says is what we report.

---

## Appendix — full numbers for the record

**Sample:** 686 posts · 498 deaths · 27.4% censored · 47 creators · 5-fold
GroupKFold, grouped by creator · freeze 2026-09-16T00:00:00Z

### Model scores

| Model | C-index | 95% CI | MAE (log₁₀) |
|---|---|---|---|
| random_survival_forest | 0.6207 | [0.5692, 0.6781] | 0.3288 |
| weibull_aft | 0.5995 | [0.5398, 0.6566] | 0.3491 |
| subscriber_only | 0.5987 | [0.5439, 0.6461] | 0.3436 |
| peak_velocity | 0.5632 | [0.5162, 0.6077] | 0.3377 |
| constant_48h | 0.5000 | [0.5000, 0.5000] | 0.3368 |
| km_median | 0.4608 | [0.4181, 0.5077] | 0.3150 |

*Note on `km_median` scoring below 0.5:* it predicts one constant per fold, and
pooling five slightly different constants creates a spurious ordering. It is an
artefact of the scoring, not evidence that the method is worse than chance.

### Paired Wilcoxon tests (n = 498; negative median favours the model)

| Comparison | Median diff | p |
|---|---|---|
| weibull_aft vs constant_48h | +0.0098 | 0.547 |
| weibull_aft vs km_median | +0.0269 | **0.0006** |
| weibull_aft vs subscriber_only | −0.0007 | 0.760 |
| weibull_aft vs peak_velocity | +0.0163 | 0.095 |
| random_survival_forest vs constant_48h | −0.0346 | 0.399 |
| random_survival_forest vs km_median | +0.0155 | **0.0062** |
| random_survival_forest vs subscriber_only | −0.0101 | **0.0192** |
| random_survival_forest vs peak_velocity | +0.0003 | 0.801 |

### Features used (11, all observed at t ≤ 7h)

`log_value_at_1h` · `log_value_at_3h` · `log_value_at_6h` · `velocity_to_6h` ·
`log_growth_1_3h` · `log_growth_3_6h` · `decay_ratio_3_6_over_1_3` ·
`log_follower_count` · `log_duration_sec` · `publish_hour_utc` · `title_len`

### Label agreement (robustness check)

| | |
|---|---|
| Saturation curve fitted | 686/686 (100%) |
| Fitted beyond the observation window (weak evidence) | 71 |
| Saturated before the landmark | 19 (2.8%) |
| Comparable (died + fitted) | 498 |
| Spearman ρ | **+0.718** (p ≈ 4×10⁻⁸⁰) |
| Median `t_saturation / t_death` | **0.633** [IQR 0.500, 0.857] |
