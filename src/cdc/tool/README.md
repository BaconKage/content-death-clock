# Check a video: the paste-a-link tool

A creator pastes a YouTube link and gets a one-page report: the video's numbers,
the odds it stops getting attention within two days of posting (and after 1, 3
and 7 days), how it compares with videos of the same age from channels of a
similar size, and what that suggests doing now.

It is an **exploratory extension** of the study, declared in `ANALYSIS_PLAN.md`
(amendment of 2026-10-27) and tested once on Cohort C. It is kept apart from the
research code: nothing in `src/cdc/eval`, `src/cdc/models`, `src/cdc/labels` or
`config/settings.yaml` is changed by it or imports it, and a test pins that.

## What it can and cannot say

- **Odds, not dates.** The study found exact timing can't be predicted from public
  numbers. The tool gives probabilities and says how often its "high risk" label
  was right.
- **Videos 15 minutes to 13 hours old** get the outlook. Older videos (up to 10
  days) get comparisons only; the outlook was built and tested on the first
  half-day.
- **Channels it has never seen.** It was tested the way a new creator uses it:
  every test video was predicted by a model that had not seen its channel.
- **No "do this and it will last" tips.** Five within-channel patterns were
  checked; none held in both cohorts, so none is shown (`assets/tips.json` has
  every number).

## Files

| | |
|---|---|
| `features.py` | the model's inputs from one reading, shared by training and the app |
| `table.py` | the training table, one row per video per scheduled reading, anonymised |
| `model.py` | random survival forest plus one-number recalibration |
| `evaluate.py` | rehearsal (A to B), freeze, the single Cohort C test, finalize |
| `tips.py` | the within-channel patterns and the rule for keeping one |
| `fetch.py` | link parsing, the YouTube lookup (2 units), the daily cap |
| `report.py` | builds the report; no I/O |
| `app.py` | the Streamlit page |
| `assets/` | the shipped training table, model card and tips |

## Rebuild

```
python -m cdc.transform.silver
python -m cdc.tool.evaluate --rehearsal
python -m cdc.tool.tips
python -m cdc.tool.evaluate --freeze
```

After the Cohort C test (`--cohort-c`, once), `--finalize` records the result in
the card and, on a pass, retrains on Cohorts A+B+C.

## Run it

Locally, with the tool's own key in `YOUTUBE_TOOL_API_KEY`:

```
streamlit run src/cdc/tool/app.py
```

On **Streamlit Community Cloud** (share.streamlit.io, sign in with GitHub):

1. *Create app*, then choose the `BaconKage/content-death-clock` repository, branch `main`.
2. Main file path: `src/cdc/tool/app.py`.
3. *Advanced settings*: Python 3.12. Under *Secrets*, paste
   `YOUTUBE_TOOL_API_KEY = "..."` with the tool's key. That box is the only place the
   key goes.
4. Deploy. The first start builds the model (under a minute). The footer shows the
   model fingerprint; if it reports "built differently on this server", a library
   version differs from `requirements.txt`.

**Use a key from its own Google Cloud project.** YouTube's daily quota belongs to
the project, not the key, so a second key in the collector's project would share
the collector's 10,000 units. The tool also stops itself at 1,000 units a day (500
reports) and 30 lookups per visitor per hour.
