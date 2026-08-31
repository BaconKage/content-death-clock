"""Content Death Clock — the demo.

Pick a tracked post and see its decay curve, the velocity that defines death,
and where the threshold sits.

**What it deliberately does not do.** It does not show a countdown for a post
whose death has not been observed and for which no model prediction exists. The
tempting demo is a confident number over every post; that number would be
invented, and a project arguing that decay should be measured rather than
guessed cannot open with a guess. Where there is nothing to say, it says so.

Run::

    streamlit run src/cdc/app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file as a script, so the package root needs to be on
# the path before cdc imports resolve.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import plotly.graph_objects as go                             # noqa: E402
import streamlit as st                                        # noqa: E402

from cdc.app.curves import build_curve, countdown_text, summarise   # noqa: E402
from cdc.config import path_for, settings                     # noqa: E402

st.set_page_config(page_title="Content Death Clock", page_icon="⏳",
                   layout="wide")

ACCENT = "#B4762A"
DEAD = "#8A3324"
ALIVE = "#1F6A6B"


@st.cache_data(ttl=300)
def load():
    d = path_for("silver_dir")
    snaps, posts = d / "snapshots.parquet", d / "posts.parquet"
    if not snaps.exists():
        return None, None
    return pd.read_parquet(snaps), (pd.read_parquet(posts) if posts.exists() else None)


def curve_figure(curve) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve.ages, y=curve.values, mode="lines+markers", name=curve.metric,
        line=dict(color=ACCENT, width=2.5), marker=dict(size=7),
        hovertemplate="%{x:.2f}h<br>%{y:,.0f} " + curve.metric + "<extra></extra>",
    ))
    if curve.event_observed and curve.t_death is not None:
        fig.add_vline(x=curve.t_death, line=dict(color=DEAD, width=2, dash="dash"),
                      annotation_text=f"attention death · {curve.t_death:.1f}h",
                      annotation_position="top right")
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="hours since publishing", yaxis_title=curve.metric,
        showlegend=False, hovermode="x unified",
    )
    return fig


def velocity_figure(curve) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve.velocity_ages, y=curve.velocities, mode="lines+markers",
        name="velocity", line=dict(color=ALIVE, width=2.5), marker=dict(size=7),
        hovertemplate="%{x:.2f}h<br>%{y:,.1f} per hour<extra></extra>",
    ))
    if curve.threshold is not None:
        fig.add_hline(
            y=curve.threshold, line=dict(color=DEAD, width=1.5, dash="dot"),
            annotation_text=f"death threshold · 5% of peak · {curve.threshold:,.1f}/h",
            annotation_position="top right")
    if curve.peak_at is not None and curve.peak_velocity is not None:
        fig.add_trace(go.Scatter(
            x=[curve.peak_at], y=[curve.peak_velocity], mode="markers",
            marker=dict(size=13, color=ACCENT, symbol="star"),
            hovertemplate="peak %{y:,.1f}/h at %{x:.2f}h<extra></extra>"))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="hours since publishing",
        yaxis_title=f"{curve.metric} per hour", showlegend=False,
        hovermode="x unified")
    return fig


def main() -> None:
    st.title("⏳ Content Death Clock")
    st.caption("When does a post stop getting attention? "
               "RV University · Research Methodology + Big Data Analytics")

    snaps, posts = load()
    if snaps is None or snaps.empty:
        st.warning("No data yet. Run `python -m cdc.transform.silver` after collection.")
        return

    summary = summarise(snaps)
    labelable = int(summary["labelable"].sum())
    died = int(summary["died"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("posts tracked", f"{len(summary):,}")
    c2.metric("observations", f"{len(snaps):,}")
    c3.metric("labelable", f"{labelable:,}")
    c4.metric("observed deaths", f"{died:,}")

    if died == 0:
        st.info(
            "**No post has died yet.** That is expected this early rather than a "
            "fault: establishing death requires watching engagement rise, peak, "
            "and stay low across several observations, which takes about a day "
            "per post. The curves below are real and already show the shape."
        )

    st.divider()

    with st.sidebar:
        st.header("Pick a post")
        plats = ["all"] + sorted(summary["platform"].dropna().unique().tolist())
        plat = st.selectbox("Platform", plats)
        view = summary if plat == "all" else summary[summary.platform == plat]
        only_lab = st.checkbox("Only posts with enough observations", value=True)
        if only_lab:
            view = view[view.labelable]
        if view.empty:
            st.warning("Nothing matches. Loosen the filters.")
            st.stop()
        view = view.sort_values("n_obs", ascending=False)
        options = view["post_id"].tolist()
        chosen = st.selectbox(
            "Post", options,
            format_func=lambda p: (
                f"{p[:26]}  ·  {int(view.loc[view.post_id == p, 'n_obs'].iloc[0])} obs"))
        st.caption(f"{len(view)} post(s) available")

    curve = build_curve(snaps, chosen, posts)

    left, right = st.columns([3, 2])
    with left:
        st.subheader(countdown_text(curve))
        st.caption(f"{curve.platform} · {curve.creator} · measured on **{curve.metric}**")
    with right:
        st.metric("observations", len(curve.ages))
        if curve.peak_velocity:
            st.metric("peak velocity", f"{curve.peak_velocity:,.0f} {curve.metric}/h",
                      help=f"reached at {curve.peak_at:.1f}h")

    st.plotly_chart(curve_figure(curve), use_container_width=True)
    st.plotly_chart(velocity_figure(curve), use_container_width=True)

    st.caption(
        "**Attention death** is the first moment velocity falls below 5% of this "
        "post's *own* peak and stays there for two consecutive intervals. "
        "Measuring against the post's own peak rather than a fixed number is what "
        "lets a small creator and a large one be compared at all."
    )

    for n in curve.notes:
        st.warning(n)

    with st.expander("Observations behind this curve"):
        st.dataframe(pd.DataFrame({
            "hours since publish": np.round(curve.ages, 2),
            curve.metric: curve.values.astype("int64"),
        }), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
