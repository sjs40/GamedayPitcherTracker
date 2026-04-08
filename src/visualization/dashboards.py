"""
Multi-panel game dashboard combining all visualization types.

Public API
----------
plot_game_dashboard(game_report, save_path) -> Figure
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src import config
from src.services.pitcher_service import GameReport
from src.visualization.rolling_plots import (
    plot_rolling_miss,
    plot_rolling_spreads,
    prep_rolling_miss,
    prep_rolling_spreads,
)
from src.visualization.zone_plots import (
    plot_cluster_drift,
    plot_cluster_map,
    plot_cluster_usage,
    prep_cluster_drift,
    prep_cluster_map,
    prep_cluster_usage,
)

logger = logging.getLogger(__name__)


def plot_game_dashboard(
    game_report: GameReport,
    save_path: Path | str | None = None,
) -> Figure:
    """
    Build a multi-panel summary dashboard for a single game + pitch type.

    Layout (2 x 3 grid):
        [0,0] Rolling horizontal + vertical spread
        [0,1] Rolling miss distance from location hub
        [1,0] Location hub cluster map (strike zone scatter)
        [1,1] Cluster drift over time (early/mid/late)
        [1,2] Cluster usage by pitch number

    Parameters
    ----------
    game_report : GameReport
        Output of build_game_report().
    save_path : Path, str, or None
        If provided, saves the figure to this path (PNG).
        Creates parent directories as needed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(18, 12), dpi=config.FIGURE_DPI)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    pitcher_id = game_report.pitcher_id
    game_date = game_report.game_date
    pitch_type = game_report.pitch_type
    window = game_report.params.get("window", config.ROLLING_WINDOW_DEFAULT)
    k = game_report.cluster_model.k
    n_fit = game_report.cluster_model.n_pitches_fit

    suptitle = (
        f"Pitcher {pitcher_id} | {pitch_type} | Game {game_report.game_pk} "
        f"({game_date}) | k={k} hubs, baseline n={n_fit}"
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)

    # ---- Panel 0,0: Rolling spread ----------------------------------------
    ax_spread = fig.add_subplot(gs[0, 0:2])
    _render_rolling_spreads_on_ax(
        ax_spread,
        game_report,
        window=window,
        pitch_type=pitch_type,
        game_date=game_date,
    )

    # ---- Panel 0,2: Rolling miss distance ------------------------------------
    ax_miss = fig.add_subplot(gs[0, 2])
    _render_rolling_miss_on_ax(ax_miss, game_report, window=window)

    # ---- Panel 1,0: Cluster map ----------------------------------------------
    ax_map = fig.add_subplot(gs[1, 0])
    _render_cluster_map_on_ax(ax_map, game_report, pitch_type=pitch_type)

    # ---- Panel 1,1: Cluster drift -------------------------------------------
    ax_drift = fig.add_subplot(gs[1, 1])
    _render_cluster_drift_on_ax(ax_drift, game_report)

    # ---- Panel 1,2: Cluster usage --------------------------------------------
    ax_usage = fig.add_subplot(gs[1, 2])
    _render_cluster_usage_on_ax(ax_usage, game_report)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Dashboard saved to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Private panel renderers (render on a given Axes, not their own Figure)
# ---------------------------------------------------------------------------

def _render_rolling_spreads_on_ax(
    ax: plt.Axes,
    game_report: GameReport,
    window: int,
    pitch_type: str,
    game_date: str,
) -> None:
    """Render the rolling spread lines on a pre-created Axes."""
    from src.features.rolling import ROLLING_H_STD_COL, ROLLING_V_STD_COL
    from src.features.pitch_index import GAME_PITCH_INDEX_COL

    df = game_report.rolling_spreads
    if df.empty or GAME_PITCH_INDEX_COL not in df.columns:
        ax.set_title("No rolling spread data")
        return

    df = df.sort_values(GAME_PITCH_INDEX_COL)
    idx = df[GAME_PITCH_INDEX_COL]

    if ROLLING_H_STD_COL in df.columns:
        ax.plot(idx, df[ROLLING_H_STD_COL], color="#2166ac", linewidth=1.8,
                label=f"H spread (w={window})")
    if ROLLING_V_STD_COL in df.columns:
        ax.plot(idx, df[ROLLING_V_STD_COL], color="#d6604d", linewidth=1.8,
                linestyle="--", label=f"V spread (w={window})")

    ax.set_xlabel("Pitch # within game", fontsize=8)
    ax.set_ylabel("Spread (ft)", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    ax.set_title(f"Rolling Spread — {pitch_type} ({game_date})", fontsize=9)


def _render_rolling_miss_on_ax(
    ax: plt.Axes,
    game_report: GameReport,
    window: int,
) -> None:
    """Render rolling miss distance on a pre-created Axes."""
    from src.features.clustering import MISS_DISTANCE_COL
    from src.features.rolling import ROLLING_MISS_COL
    from src.features.pitch_index import GAME_PITCH_INDEX_COL

    df = game_report.rolling_miss
    if df.empty or GAME_PITCH_INDEX_COL not in df.columns:
        ax.set_title("No miss distance data")
        return

    df = df.sort_values(GAME_PITCH_INDEX_COL)
    idx = df[GAME_PITCH_INDEX_COL]
    color = "#4dac26"

    if MISS_DISTANCE_COL in df.columns:
        ax.scatter(idx, df[MISS_DISTANCE_COL], color=color, alpha=0.2, s=10, zorder=2)
    if ROLLING_MISS_COL in df.columns:
        ax.plot(idx, df[ROLLING_MISS_COL], color=color, linewidth=2.0, zorder=3,
                label=f"Rolling mean (w={window})")

    ax.set_xlabel("Pitch #", fontsize=8)
    ax.set_ylabel("Miss dist (ft)", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    ax.set_title("Rolling Miss Distance", fontsize=9)


def _render_cluster_map_on_ax(
    ax: plt.Axes,
    game_report: GameReport,
    pitch_type: str,
) -> None:
    """Render cluster map (location hubs) on a pre-created Axes."""
    from src.visualization.zone_plots import draw_strike_zone, _set_zone_axes
    from src.features.clustering import CLUSTER_LABEL_COL

    df = game_report.game_with_clusters
    centers = game_report.cluster_model.centers
    color_map = game_report.cluster_model.color_map

    for cluster_id, color in color_map.items():
        mask = df[CLUSTER_LABEL_COL] == cluster_id
        subset = df[mask]
        ax.scatter(
            subset[config.PLATE_X_COL],
            subset[config.PLATE_Z_COL],
            c=color, alpha=0.3, s=14,
            label=f"C{cluster_id} (n={len(subset)})", zorder=3,
        )

    for i, center in enumerate(centers):
        ax.scatter(center[0], center[1], c=color_map[i], marker="*",
                   s=220, edgecolors="black", linewidths=0.7, zorder=5)

    _set_zone_axes(ax)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Location Hubs — {pitch_type}", fontsize=9)


def _render_cluster_drift_on_ax(ax: plt.Axes, game_report: GameReport) -> None:
    """Render cluster drift arrows on a pre-created Axes."""
    from src.visualization.zone_plots import _set_zone_axes

    summaries = game_report.drift_report.cluster_summaries
    color_map = game_report.cluster_model.color_map
    bucket_markers = {"early": "o", "mid": "s", "late": "^"}

    for s in summaries:
        color = color_map.get(s.cluster_id, "gray")
        cx, cz = s.baseline_center
        ax.scatter(cx, cz, c=color, marker="*", s=220,
                   edgecolors="black", linewidths=0.7, zorder=5)

        prev = None
        for bucket in config.BUCKET_LABELS:
            mean = getattr(s, f"{bucket}_mean")
            if mean is None:
                continue
            mx, mz = mean
            marker = bucket_markers.get(bucket, "o")
            ax.scatter(mx, mz, c=color, marker=marker, s=60, alpha=0.9,
                       edgecolors="black", linewidths=0.4, zorder=4)
            if prev is not None:
                ax.annotate(
                    "", xy=(mx, mz), xytext=prev,
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.7, lw=1.0),
                    zorder=4,
                )
            prev = (mx, mz)

    _set_zone_axes(ax)
    ax.set_title("Hub Drift: Early→Mid→Late", fontsize=9)


def _render_cluster_usage_on_ax(ax: plt.Axes, game_report: GameReport) -> None:
    """Render cluster usage scatter on a pre-created Axes."""
    from src.features.clustering import CLUSTER_LABEL_COL
    from src.features.pitch_index import GAME_PITCH_INDEX_COL

    df = game_report.game_with_clusters
    color_map = game_report.cluster_model.color_map

    for cluster_id, color in color_map.items():
        mask = df[CLUSTER_LABEL_COL] == cluster_id
        subset = df[mask]
        ax.scatter(
            subset[GAME_PITCH_INDEX_COL] if GAME_PITCH_INDEX_COL in subset.columns else subset.index,
            [cluster_id] * len(subset),
            c=color, s=22, alpha=0.7,
            label=f"Hub {cluster_id}", zorder=3,
        )

    ax.set_xlabel("Pitch #", fontsize=8)
    ax.set_ylabel("Hub", fontsize=8)
    ax.set_yticks(list(color_map.keys()))
    ax.legend(fontsize=7)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_title("Hub Assignment by Pitch", fontsize=9)
