"""The paste-a-link report tool: one reading of a YouTube video, honest odds.

**Kept apart from the research on purpose** (amendment 2026-10-27). This
package reads the collected data and imports a few helpers (the YouTube client,
the silver parsers, the death label through ``cdc.models.dataset``), but it
changes nothing under ``cdc.eval``, ``cdc.models``, ``cdc.labels`` or
``config/settings.yaml``, and nothing there imports it. The research results
and the two frozen tests therefore cannot be moved by anything done here.

Modules:

``features``  the model's inputs from one reading; the *same* function serves
              training and the live app, so the two cannot drift apart
``table``     the training table: one row per video per checkpoint, anonymised
``model``     the instant model: a random survival forest plus a one-number
              recalibration toward the average curve
``tips``      "what tended to make videos last", kept only if it held in both
              Cohort A and Cohort B
``evaluate``  the A-to-B rehearsal, the freeze, and the single Cohort C test
``fetch``     link parsing, the YouTube lookup, the daily unit cap
``report``    turns a lookup into the report the app shows (no I/O)
``app``       the Streamlit page
"""
