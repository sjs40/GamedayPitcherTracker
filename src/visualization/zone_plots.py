"""
Strike-zone spatial visualization.

Two-mode design
---------------
prep_*  functions: return chart-ready dicts/DataFrames (no matplotlib)
plot_*  functions: accept prepped data, render matplotlib Figure objects

This separation means a future Streamlit app calls:
    prepped = prep_cluster_map(...)
    fig = plot_cluster_map(prepped, title="...")
    st.pyplot(fig)

without touching any analytical code.

Public API
----------
draw_strike_zone(ax)
prep_cluster_map(game_with_clusters, cluster_centers) -> dict
plot_cluster_map(prepped, title) -> Figure
prep_miss_vectors(game_with_clusters, cluster_centers, max_pitches) -> dict
plot_miss_vectors(prepped, title) -> Figure
prep_cluster_drift(drift_report) -> dict
plot_cluster_drift(prepped, title) -> Figure
prep_cluster_usage(game_with_clusters) -> dict
plot_cluster_usage(prepped, title) -> Figure
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src import config

if TYPE_CHECKING:
    from src.features.drift import GameDriftReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utility: draw strike zone rectangle
# ---------------------------------------------------------------------------

def draw_strike_zone(ax: plt.Axes) -> None:
    """
    Draw an approximate MLB strike zone rectangle on the given axes.

    Uses config.SZ_* bounds. This is an approximate display aid only —
    NOT used as intent proxy in any analytics.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    """
    rect = mpatches.Rectangle(
        (config.SZ_LEFT, config.SZ_BOT),
        config.SZ_RIGHT - config.SZ_LEFT,
        config.SZ_TOP - config.SZ_BOT,
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(rect)


def _set_zone_axes(ax: plt.Axes) -> None:
    """Apply consistent axis limits and labels for zone plots."""
    ax.set_xlim(config.PLOT_X_MIN, config.PLOT_X_MAX)
    ax.set_ylim(config.PLOT_Z_MIN, config.PLOT_Z_MAX)
    ax.set_xlabel("Horizontal (ft) — catcher's view: left = arm side RHP", fontsize=9)
    ax.set_ylabel("Vertical (ft)", fontsize=9)
    ax.set_aspect("equal")
    draw_strike_zone(ax)


# ---------------------------------------------------------------------------
# Cluster map
# ---------------------------------------------------------------------------

def prep_cluster_map(
    game_with_clusters: pd.DataFrame,
    cluster_centers: np.ndarray,
) -> dict:
    """
    Prepare data for the cluster map scatter plot.

    Parameters
    ----------
    game_with_clusters : pd.DataFrame
        Must contain plate_x, plate_z, cluster_label columns.
    cluster_centers : np.ndarray
        Shape (k, 2). Cluster center coordinates.

    Returns
    -------
    dict with keys:
        'pitches' : pd.DataFrame  (plate_x, plate_z, cluster_label)
        'centers' : np.ndarray    shape (k, 2)
        'k'       : int
        'color_map' : dict[int, str]
    """
    from src.features.clustering import CLUSTER_LABEL_COL

    k = len(cluster_centers)
    color_map = {
        i: config.CLUSTER_COLORS[i % len(config.CLUSTER_COLORS)]
        for i in range(k)
    }

    pitches = game_with_clusters[
        [config.PLATE_X_COL, config.PLATE_Z_COL, CLUSTER_LABEL_COL]
    ].dropna().copy()

    return {
        "pitches": pitches,
        "centers": cluster_centers,
        "k": k,
        "color_map": color_map,
    }


def plot_cluster_map(prepped: dict, title: str = "") -> Figure:
    """
    Strike-zone scatter plot showing pitch locations and cluster centers.

    All pitch locations are shown as small semi-transparent dots, colored
    by cluster assignment. Cluster centers (location hubs) are shown as
    large markers.

    Parameters
    ----------
    prepped : dict
        Output of prep_cluster_map().
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    from src.features.clustering import CLUSTER_LABEL_COL

    fig, ax = plt.subplots(figsize=(6, 7), dpi=config.FIGURE_DPI)

    pitches = prepped["pitches"]
    centers = prepped["centers"]
    color_map = prepped["color_map"]

    # Plot pitch points by cluster
    for cluster_id, color in color_map.items():
        mask = pitches[CLUSTER_LABEL_COL] == cluster_id
        subset = pitches[mask]
        ax.scatter(
            subset[config.PLATE_X_COL],
            subset[config.PLATE_Z_COL],
            c=color,
            alpha=0.35,
            s=18,
            label=f"Cluster {cluster_id} (n={len(subset)})",
            zorder=3,
        )

    # Plot cluster centers (location hubs)
    for i, center in enumerate(centers):
        ax.scatter(
            center[0],
            center[1],
            c=color_map[i],
            marker="*",
            s=280,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=f"Hub {i}",
        )

    _set_zone_axes(ax)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title or "Location Hubs (Empirical Clusters)", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cluster miss vectors
# ---------------------------------------------------------------------------

def prep_miss_vectors(
    game_with_clusters: pd.DataFrame,
    cluster_centers: np.ndarray,
    max_pitches: int | None = 80,
) -> dict:
    """
    Prepare data for miss-vector arrows: lines from hub center to actual pitch.

    Parameters
    ----------
    game_with_clusters : pd.DataFrame
        Must contain plate_x, plate_z, cluster_label.
    cluster_centers : np.ndarray  shape (k, 2)
    max_pitches : int or None
        If set, subsample to at most max_pitches pitches (evenly spaced by
        pitch order) to avoid an over-cluttered plot.

    Returns
    -------
    dict with keys:
        'pitches'     : pd.DataFrame  (plate_x, plate_z, cluster_label)
        'centers'     : np.ndarray
        'color_map'   : dict[int, str]
        'max_pitches' : int or None
    """
    from src.features.clustering import CLUSTER_LABEL_COL

    k = len(cluster_centers)
    color_map = {
        i: config.CLUSTER_COLORS[i % len(config.CLUSTER_COLORS)]
        for i in range(k)
    }

    cols = [config.PLATE_X_COL, config.PLATE_Z_COL, CLUSTER_LABEL_COL]
    pitches = game_with_clusters[cols].dropna().copy()

    if max_pitches is not None and len(pitches) > max_pitches:
        idx = np.linspace(0, len(pitches) - 1, max_pitches, dtype=int)
        pitches = pitches.iloc[idx].reset_index(drop=True)

    return {
        "pitches": pitches,
        "centers": cluster_centers,
        "color_map": color_map,
        "max_pitches": max_pitches,
    }


def plot_miss_vectors(prepped: dict, title: str = "") -> Figure:
    """
    Strike-zone plot with arrows from each cluster center to actual pitch locations.

    Shows directional miss patterns — whether a pitcher's misses are
    consistently arm-side, glove-side, up, or down relative to each hub.

    Parameters
    ----------
    prepped : dict
        Output of prep_miss_vectors().
    title : str

    Returns
    -------
    Figure
    """
    from src.features.clustering import CLUSTER_LABEL_COL

    fig, ax = plt.subplots(figsize=(6, 7), dpi=config.FIGURE_DPI)

    pitches = prepped["pitches"]
    centers = prepped["centers"]
    color_map = prepped["color_map"]

    for _, row in pitches.iterrows():
        cluster_id = int(row[CLUSTER_LABEL_COL])
        color = color_map.get(cluster_id, "gray")
        cx, cz = centers[cluster_id]
        px, pz = row[config.PLATE_X_COL], row[config.PLATE_Z_COL]
        dx, dz = px - cx, pz - cz

        ax.annotate(
            "",
            xy=(px, pz),
            xytext=(cx, cz),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                alpha=0.4,
                lw=0.8,
            ),
            zorder=3,
        )

    # Plot cluster centers
    for i, center in enumerate(centers):
        ax.scatter(
            center[0],
            center[1],
            c=color_map[i],
            marker="*",
            s=280,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=f"Hub {i}",
        )

    _set_zone_axes(ax)
    ax.legend(fontsize=8, loc="upper right")
    n = len(pitches)
    ax.set_title(title or f"Miss Vectors from Location Hubs (n={n})", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cluster drift over time
# ---------------------------------------------------------------------------

def prep_cluster_drift(drift_report: "GameDriftReport") -> dict:
    """
    Prepare data for the cluster drift visualization.

    Parameters
    ----------
    drift_report : GameDriftReport

    Returns
    -------
    dict with keys:
        'summaries'     : list[ClusterDriftSummary]
        'bucket_labels' : list[str]  e.g. ['early', 'mid', 'late']
        'color_map'     : dict[int, str]
    """
    k = len(drift_report.cluster_summaries)
    color_map = {
        i: config.CLUSTER_COLORS[i % len(config.CLUSTER_COLORS)]
        for i in range(k)
    }

    return {
        "summaries": drift_report.cluster_summaries,
        "bucket_labels": config.BUCKET_LABELS,
        "color_map": color_map,
    }


def plot_cluster_drift(prepped: dict, title: str = "") -> Figure:
    """
    Strike-zone plot showing how each location hub's actual mean location
    shifts across early/mid/late game buckets.

    For each cluster:
    - baseline center is plotted as a large star marker
    - early/mid/late mean locations are plotted as smaller markers
    - a line + arrows connect them in temporal order

    Purpose: show whether a pitcher's location hubs drift arm-side,
    glove-side, up, or down over the course of the game.

    Parameters
    ----------
    prepped : dict
        Output of prep_cluster_drift().
    title : str

    Returns
    -------
    Figure
    """
    summaries = prepped["summaries"]
    bucket_labels = prepped["bucket_labels"]
    color_map = prepped["color_map"]

    fig, ax = plt.subplots(figsize=(6, 7), dpi=config.FIGURE_DPI)

    bucket_markers = {"early": "o", "mid": "s", "late": "^"}
    bucket_alpha = {"early": 0.6, "mid": 0.8, "late": 1.0}

    for s in summaries:
        color = color_map.get(s.cluster_id, "gray")
        cx, cz = s.baseline_center

        # Baseline center (location hub)
        ax.scatter(cx, cz, c=color, marker="*", s=280,
                   edgecolors="black", linewidths=0.8, zorder=5)

        # Bucket means with connecting arrows
        prev_point = None
        for bucket in bucket_labels:
            mean = getattr(s, f"{bucket}_mean")
            n = getattr(s, f"{bucket}_n")
            if mean is None:
                continue

            mx, mz = mean
            marker = bucket_markers.get(bucket, "o")
            alpha = bucket_alpha.get(bucket, 0.8)
            ax.scatter(mx, mz, c=color, marker=marker, s=80,
                       alpha=alpha, edgecolors="black", linewidths=0.5,
                       zorder=4, label=f"C{s.cluster_id} {bucket} (n={n})")

            # Arrow from previous bucket mean (or baseline center) to this mean
            if prev_point is not None:
                ax.annotate(
                    "",
                    xy=(mx, mz),
                    xytext=prev_point,
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        alpha=0.7,
                        lw=1.2,
                    ),
                    zorder=4,
                )
            prev_point = (mx, mz)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Add marker legend for bucket shapes
    for bucket, marker in bucket_markers.items():
        by_label[bucket.capitalize()] = plt.Line2D(
            [], [], marker=marker, color="gray", linestyle="none",
            markersize=7, label=bucket.capitalize()
        )

    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right",
              ncol=2)

    _set_zone_axes(ax)
    ax.set_title(title or "Location Hub Drift: Early → Mid → Late", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cluster usage over time
# ---------------------------------------------------------------------------

def prep_cluster_usage(game_with_clusters: pd.DataFrame) -> dict:
    """
    Prepare cluster assignment data for the usage-over-time chart.

    Returns a DataFrame indexed by game_pitch_index with cluster_label,
    suitable for a scatter or step chart showing how cluster usage evolves.

    Parameters
    ----------
    game_with_clusters : pd.DataFrame
        Must contain game_pitch_index and cluster_label.

    Returns
    -------
    dict with keys:
        'usage_df'  : pd.DataFrame  (game_pitch_index, cluster_label)
        'k'         : int
        'color_map' : dict[int, str]
    """
    from src.features.clustering import CLUSTER_LABEL_COL
    from src.features.pitch_index import GAME_PITCH_INDEX_COL

    cols = [c for c in [GAME_PITCH_INDEX_COL, CLUSTER_LABEL_COL] if c in game_with_clusters.columns]
    usage_df = game_with_clusters[cols].dropna().copy()

    k = int(usage_df[CLUSTER_LABEL_COL].max()) + 1 if not usage_df.empty else 0
    color_map = {
        i: config.CLUSTER_COLORS[i % len(config.CLUSTER_COLORS)]
        for i in range(k)
    }

    return {"usage_df": usage_df, "k": k, "color_map": color_map}


def plot_cluster_usage(prepped: dict, title: str = "") -> Figure:
    """
    Chart showing which location hub each pitch was assigned to, by pitch number.

    This is a simplified first-pass implementation: a scatter plot with
    cluster_id on the y-axis and pitch number on the x-axis, colored by
    cluster. This avoids the complexity of rolling stacked area charts
    while still showing usage patterns over the game.

    Parameters
    ----------
    prepped : dict
        Output of prep_cluster_usage().
    title : str

    Returns
    -------
    Figure
    """
    from src.features.clustering import CLUSTER_LABEL_COL
    from src.features.pitch_index import GAME_PITCH_INDEX_COL

    usage_df = prepped["usage_df"]
    color_map = prepped["color_map"]

    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=config.FIGURE_DPI)

    if usage_df.empty:
        ax.set_title("No cluster usage data available")
        return fig

    for cluster_id, color in color_map.items():
        mask = usage_df[CLUSTER_LABEL_COL] == cluster_id
        subset = usage_df[mask]
        ax.scatter(
            subset[GAME_PITCH_INDEX_COL],
            subset[CLUSTER_LABEL_COL],
            c=color,
            s=30,
            alpha=0.7,
            label=f"Cluster {cluster_id}",
            zorder=3,
        )

    ax.set_xlabel("Pitch # within game", fontsize=9)
    ax.set_ylabel("Location hub", fontsize=9)
    ax.set_yticks(list(color_map.keys()))
    ax.set_yticklabels([f"Hub {i}" for i in color_map])
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title or "Location Hub Assignment by Pitch", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
