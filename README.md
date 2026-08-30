# Content Death Clock

Predicting **when** social media content stops getting attention, rather than whether it
goes viral. RV University, 7th Semester — Research Methodology + Big Data Analytics.

Most content-performance research predicts *growth*. We predict *decay*: given a post,
how long until engagement velocity collapses? The deliverable is a decay curve and a
countdown to "attention death", trained on repeated real observations of real posts.

---

## The one thing that matters right now

**Decay labels are wall-clock bound.** A post's first six hours can be observed once and
never again. Every hour the collector is not running is an hour of data that cannot be
recovered by working harder later. Get the collector live first; build everything else
while data accrues.

---

## Setup (15 minutes)

```bash
git clone <this-repo>
cd "RM BDA Project"
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Get a **YouTube Data API v3** key: [console.cloud.google.com](https://console.cloud.google.com)
→ new project → *APIs & Services* → enable **YouTube Data API v3** → *Credentials* →
create API key. It is free and takes about three minutes.

```bash
cp .env.example .env
```

Put the key in `.env`. It is gitignored — **never commit it**. For the scheduled
collector, the same key goes in the GitHub repo under
*Settings → Secrets and variables → Actions* as `YOUTUBE_API_KEY`.

Verify everything works without spending any quota:

```bash
python -m pytest
```

---

## Day 1–3 checklist (the critical path)

- [ ] **Expand `config/channels.yaml` to 60–100 channels.** The `micro` tier (<10k subs)
      is currently empty, so the sample cannot yet claim to span creator size — and small
      channels are where decay is fastest and most variable. Everyone adds 10 channels
      they actually follow.
- [ ] **Resolve and verify the frame.** Costs 1 quota unit per channel:
      ```bash
      python -m cdc.collect.resolve_channels
      ```
      This proves every handle exists, replaces guessed tiers with measured subscriber
      counts, and writes `config/channels.resolved.yaml`. A typo'd handle silently
      contributes zero posts, which is exactly the kind of thing nobody notices until
      week four.
- [ ] **Plan a cycle without spending quota:**
      ```bash
      python -m cdc.collect.runner --dry-run
      ```
- [ ] **Run one real cycle**, confirm files appear under `data/bronze/`.
- [ ] **Push to GitHub, add the secret, trigger `collect` manually** from the Actions tab.
- [ ] **Gate: three consecutive successful hourly runs.** Do not start Week 1 work until
      this passes.
- [ ] **Commit `ANALYSIS_PLAN.md` before looking at any outcome data.**

---

## Daily habit during collection

```bash
python -m cdc.collect.monitor
```

Prints completeness, lateness, and any outage. **Check the Actions tab every day.** A
silent three-day outage is the single most expensive failure available to this project.

---

## How it works

| Stage | Module | What it does |
|---|---|---|
| Discover | `cdc.collect.youtube` | Finds new uploads via each channel's uploads playlist (1 quota unit) — never `search.list`, which costs 100 |
| Snapshot | `cdc.collect.runner` | Records views/likes/comments for every post owed an observation, batched 50 ids per call |
| Store | `cdc.storage.bronze` | Append-only JSONL, partitioned `platform=/dt=`, committed atomically so a retried cycle cannot duplicate rows |
| Track | `cdc.collect.panel` | Rebuilds panel state *from bronze* — no separate state file to drift or corrupt |
| Label | `cdc.labels.death` | Velocity relative to each post's own peak → time-to-death, with right-censoring |
| Monitor | `cdc.collect.monitor` | Completeness, lateness, outages |

Snapshots are dense early (1, 3, 6, 12, 24, 36, 48, 72h) then thin to daily out to 14 days,
because that is where the signal is.

### Quota cost model

Every `list` call costs 1 unit against a 10,000/day budget. For **C** channels and
**V** tracked posts, one hourly cycle costs `C + ceil(V/50)` units. At C=80, V=1500:
**110 units/cycle ≈ 2,640/day** — comfortably inside budget with room for retries.

---

## Definitions that the papers rest on

All configured in [`config/settings.yaml`](config/settings.yaml), so the analysis plan can
cite exact values.

**Attention death** — the first time engagement velocity falls below **5% of that post's
own peak velocity** and stays below for **2 consecutive intervals**. Normalising against
the post's own peak is what makes a 300-view post and a 3M-view post comparable.

**Right-censoring** — a post still alive when observation ends has *not* died. It is kept,
flagged `event_observed=False`, and handed to a survival model. Dropping these would bias
the sample toward short-lived content and inflate every number we report.

**Velocity** is computed on **actual observed timestamps**, never an assumed grid, because
schedulers drift and a late snapshot must be late-but-honest.

---

## Layout

```
config/          settings.yaml (all experimental parameters) + the sample frame
src/cdc/
  collect/       youtube client, panel state, cycle runner, monitor, resolver
  storage/       bronze layer
  transform/     silver/gold (week 1)
  labels/        death.py — the dependent variable
  models/        baselines + survival (week 2)
  eval/          GroupKFold, bootstrap, Wilcoxon (week 2)
  bench/         pandas vs Spark scalability (week 2)
  app/           Streamlit demo (week 3)
tests/           label correctness + offline end-to-end pipeline test
paper/rm/        Research Methodology paper
paper/bda/       Big Data Analytics report
```

---

## Testing

```bash
python -m pytest
```

`tests/test_death_labels.py` checks the label function against analytically known answers.
`tests/test_pipeline_offline.py` drives the real runner through a simulated 21-day
collection against a fake API — no network — and asserts that fast decayers come out with
shorter lifetimes than slow ones, that a retried cycle does not duplicate data, and that
the quota cost matches the model documented above.

---

## Ethics

Public content and public accounts only. No personal data is collected. Results are
reported in aggregate and creators are not individually identified. Scrape Creators is an
unofficial third-party API; its use is disclosed as a stated limitation rather than
buried. The design is observational: it predicts decay, it does not establish its causes.
