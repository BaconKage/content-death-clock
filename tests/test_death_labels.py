"""Tests for the death label.

These check against *analytically known* answers, not against whatever the code
happens to produce today. This is the one module where a silent bug invalidates
every result in both papers, so the tests are written to be readable by a marker
who does not know Python well.
"""
from __future__ import annotations

import math

import pytest

from cdc.labels.death import (
    INSUFFICIENT_OBSERVATIONS,
    NO_POSITIVE_VELOCITY,
    Observation,
    label_post,
    velocities,
)

CFG = {
    "velocity_frac_of_peak": 0.05,
    "sustain_intervals": 2,
    "min_intervals_for_label": 4,
    "peak_skip_first_interval": False,
    "saturation_target": 0.90,
    "primary_metric": "views",
}


def series(pairs):
    return [Observation(age_hours=float(t), value=float(v)) for t, v in pairs]


# --------------------------------------------------------------- basic mechanics
def test_velocity_is_per_hour_not_per_interval():
    """A 600-view gain over 6 hours is 100/hour, not 600."""
    v = velocities(series([(0, 0), (6, 600)]))
    assert v == [(6.0, 100.0)]


def test_unevenly_spaced_observations_use_real_elapsed_time():
    """Schedulers drift. Velocity must use observed gaps, never assumed ones."""
    v = velocities(series([(0, 0), (1, 100), (3.5, 350)]))
    assert v[0] == (1.0, 100.0)
    assert v[1][0] == 3.5
    assert v[1][1] == pytest.approx(100.0)      # 250 views / 2.5 h


# ------------------------------------------------------------------- the label
def test_death_detected_at_first_sustained_drop():
    """Peak velocity 1000/h; threshold 50/h. Velocity crosses at t=24 and stays
    down, so death is 24 — not 36, and not the first momentary dip."""
    obs = series([
        (0, 0),          # ---
        (1, 1000),       # v=1000  <- peak
        (3, 2400),       # v=700
        (6, 3000),       # v=200
        (12, 3300),      # v=50    (exactly at threshold, NOT below -> not death)
        (24, 3340),      # v=3.33  <- first below
        (36, 3350),      # v=0.83  <- sustained
        (48, 3355),      # v=0.42
    ])
    lab = label_post("p1", obs, CFG)
    assert lab.usable
    assert lab.event_observed is True
    assert lab.peak_velocity == pytest.approx(1000.0)
    assert lab.t_death == pytest.approx(24.0)


def test_momentary_dip_does_not_count_as_death():
    """A single quiet interval followed by recovery is not death — that is
    exactly what `sustain_intervals` exists to prevent."""
    obs = series([
        (0, 0),
        (1, 1000),       # v=1000 peak
        (2, 1010),       # v=10   below threshold(50)... but
        (3, 1600),       # v=600  ...it recovers
        (6, 1900),       # v=100
        (12, 1930),      # v=5    below
        (24, 1940),      # v=0.83 below -> sustained, death here
        (36, 1945),
    ])
    lab = label_post("p2", obs, CFG)
    assert lab.event_observed is True
    assert lab.t_death == pytest.approx(12.0), "death must not be the t=2 blip"


def test_still_alive_post_is_right_censored_not_dropped():
    """A post growing steadily to the end of the window has NOT died. It must
    come back censored, carrying its last observation time — dropping it would
    bias the sample toward short-lived content."""
    obs = series([(0, 0), (1, 100), (3, 300), (6, 600), (12, 1200), (24, 2400)])
    lab = label_post("p3", obs, CFG)
    assert lab.usable
    assert lab.event_observed is False
    assert lab.t_death == pytest.approx(24.0)     # censoring time
    assert lab.last_observed_hours == pytest.approx(24.0)


def test_threshold_is_relative_so_scale_does_not_matter():
    """The same shape at 1000x the scale must give the same death time. This is
    what lets a micro creator and MrBeast sit in one model."""
    shape = [(0, 0), (1, 1000), (3, 1400), (6, 1500), (12, 1520), (24, 1525), (36, 1527)]
    small = label_post("s", series(shape), CFG)
    big = label_post("b", series([(t, v * 1000) for t, v in shape]), CFG)
    assert small.t_death == big.t_death
    assert small.event_observed == big.event_observed


# ------------------------------------------------------------- data robustness
def test_view_count_decrease_is_clamped_and_counted():
    """Platforms retract inauthentic views, so cumulative counts can fall. That
    must not read as negative velocity, and it must be *counted* so data quality
    is a reported number rather than an invisible assumption."""
    obs = series([(0, 0), (1, 1000), (3, 900), (6, 1100), (12, 1150), (24, 1160)])
    lab = label_post("p4", obs, CFG)
    assert lab.monotonicity_repairs == 1
    assert lab.peak_velocity == pytest.approx(1000.0)
    assert lab.peak_velocity > 0


def test_too_few_observations_is_excluded_with_a_reason():
    lab = label_post("p5", series([(0, 0), (1, 100), (3, 150)]), CFG)
    assert not lab.usable
    assert lab.exclude_reason == INSUFFICIENT_OBSERVATIONS
    assert lab.t_death is None


def test_post_that_never_grew_is_excluded_not_called_dead():
    """Zero traffic throughout is an absence of life, not a death. Labelling it
    as instant death would poison the survival model with fake events."""
    obs = series([(0, 0), (1, 0), (3, 0), (6, 0), (12, 0), (24, 0)])
    lab = label_post("p6", obs, CFG)
    assert not lab.usable
    assert lab.exclude_reason == NO_POSITIVE_VELOCITY


def test_out_of_order_and_duplicate_observations_are_handled():
    """Bronze is append-only from an unreliable scheduler; order is not promised."""
    unordered = series([(6, 3000), (0, 0), (24, 3340), (1, 1000),
                        (36, 3350), (3, 2400), (12, 3300), (12, 3300)])
    ordered = series([(0, 0), (1, 1000), (3, 2400), (6, 3000),
                      (12, 3300), (24, 3340), (36, 3350)])
    assert label_post("u", unordered, CFG).t_death == label_post("o", ordered, CFG).t_death


# ------------------------------------------------ saturation (robustness label)
def test_saturation_fit_recovers_known_parameters():
    """Generate data from C(t) = A(1 - e^-kt) with A and k known, then check the
    fit recovers the analytic t90 = ln(10)/k."""
    A, k = 50_000.0, 0.20
    obs = series([(t, A * (1 - math.exp(-k * t)))
                  for t in (0, 1, 3, 6, 12, 24, 36, 48, 72, 96, 120)])
    lab = label_post("sat", obs, CFG)
    expected_t90 = math.log(10) / k        # ~11.51 hours
    assert lab.fitted_A == pytest.approx(A, rel=0.02)
    assert lab.fitted_k == pytest.approx(k, rel=0.05)
    assert lab.t_saturation == pytest.approx(expected_t90, rel=0.05)


def test_saturation_failure_does_not_break_the_primary_label():
    """A still-climbing post has no identifiable asymptote. The velocity label
    must survive that."""
    obs = series([(0, 0), (1, 100), (3, 300), (6, 600), (12, 1200), (24, 2400)])
    lab = label_post("lin", obs, CFG)
    assert lab.usable                       # primary label still produced
    assert lab.event_observed is False
