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


# ---------------------------------------------------------------------------
# Arsenal dashboard (pitch-type-centric, multi-pitch-type)
# ---------------------------------------------------------------------------

def plot_arsenal_dashboard(
    df,
    game_pk: int,
    cluster_models: dict | None = None,
    window: int = config.ROLLING_WINDOW_DEFAULT,
    max_pitch_types: int = 4,
    save_path=None,
) -> Figure:
    """
    Game summary dashboard organized by pitch type.

    Layout
    ------
    Top row (full width, 4 columns):
        [col 0:2] Pitch mix trend (rolling % of each type over game)
        [col 2:4] Overall rolling H/V spread (all pitches combined)

    Per-pitch-type rows (one row per type, up to max_pitch_types):
        [col 0] Zone scatter  (colored by pitch type, hub markers if available)
        [col 1] Rolling velocity trend
        [col 2] Rolling performance panel (strike%, zone%, whiff%)
        [col 3] Rolling miss distance from location hub

    Pitch types sorted by frequency (most-used first).
    If no cluster_models provided, miss-distance panels are blank.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with pitch outcomes already encoded
        (encode_pitch_outcomes called upstream).
    game_pk : int
    cluster_models : dict[str, PitchTypeClusterModel] | None
    window : int
    max_pitch_types : int
    save_path : Path, str, or None

    Returns
    -------
    Figure
    """
    import math
    import pandas as _pd
    from src.features.performance import (
        compute_all_performance_metrics,
        compute_pitch_mix,
    )
    from src.features.rolling import compute_game_rolling_spreads, compute_game_rolling_miss_distance
    from src.visualization.rolling_plots import (
        prep_pitch_mix_trend, plot_pitch_mix_trend,
        prep_rolling_spreads, plot_rolling_spreads,
        prep_velocity_trend, plot_velocity_trend,
        prep_performance_panel, plot_performance_panel,
        prep_rolling_miss, plot_rolling_miss,
    )
    from src.visualization.zone_plots import (
        prep_pitch_type_zone_map, _set_zone_axes,
    )

    # Determine pitch types to display (most frequent first)
    game_mask = df["game_pk"] == game_pk
    game_df = df[game_mask]
    freq = game_df["pitch_type"].value_counts()
    pitch_types = list(freq.index[:max_pitch_types])

    n_pt = len(pitch_types)
    n_rows = 1 + max(1, n_pt)  # header row + one row per type

    NCOLS = 4
    fig = plt.figure(
        figsize=(6 * NCOLS, 5 * n_rows),
        dpi=config.FIGURE_DPI,
    )
    gs = gridspec.GridSpec(
        n_rows, NCOLS, figure=fig,
        hspace=0.45, wspace=0.35,
    )

    # ---- Top row: pitch mix + overall spread --------------------------------
    ax_mix = fig.add_subplot(gs[0, 0:2])
    mix_df = compute_pitch_mix(df, game_pk, window=config.PITCH_MIX_WINDOW_DEFAULT)
    prepped_mix = prep_pitch_mix_trend(mix_df)
    _render_on_ax(ax_mix, prepped_mix, "Rolling Pitch Mix", _plot_mix_on_ax)

    ax_spread_all = fig.add_subplot(gs[0, 2:4])
    spreads_all = compute_game_rolling_spreads(df, game_pk, pitch_type=None, window=window)
    prepped_spread = prep_rolling_spreads(spreads_all)
    _render_on_ax(ax_spread_all, prepped_spread, "Overall Rolling Spread (all types)",
                  _plot_spreads_on_ax, window=window)

    # ---- Per-pitch-type rows ------------------------------------------------
    # Precompute zone data once (for all types combined)
    zone_prepped = prep_pitch_type_zone_map(df, game_pk, cluster_models)

    for row_idx, pt in enumerate(pitch_types, start=1):
        color = config.PITCH_TYPE_COLORS.get(pt, config.PITCH_TYPE_COLORS["_default"])
        cluster_model = (cluster_models or {}).get(pt)

        # [col 0] Zone scatter
        ax_zone = fig.add_subplot(gs[row_idx, 0])
        _render_pitch_type_zone_on_ax(ax_zone, zone_prepped, pt, color)

        # [col 1] Velocity
        ax_velo = fig.add_subplot(gs[row_idx, 1])
        perf = compute_all_performance_metrics(df, game_pk, pt, window)
        velo_prepped = prep_velocity_trend(perf["velocity"])
        _render_on_ax(ax_velo, velo_prepped, f"{pt} — Velocity",
                      _plot_velo_on_ax, window=window)

        # [col 2] Performance panel
        ax_perf = fig.add_subplot(gs[row_idx, 2])
        perf_prepped = prep_performance_panel(perf)
        _render_on_ax(ax_perf, perf_prepped, f"{pt} — Strike/Zone/Whiff%",
                      _plot_perf_on_ax, window=window)

        # [col 3] Miss distance
        ax_miss_pt = fig.add_subplot(gs[row_idx, 3])
        if cluster_model is not None:
            miss_df = compute_game_rolling_miss_distance(
                df, game_pk, pt, window, cluster_model
            )
            miss_prepped = prep_rolling_miss(miss_df)
        else:
            miss_prepped = {
                "pitch_index": _pd.Series([], dtype=float),
                "miss_distance": _pd.Series([], dtype=float),
                "rolling_miss_dist": _pd.Series([], dtype=float),
                "n_pitches": 0,
                "has_data": False,
            }
        _render_on_ax(ax_miss_pt, miss_prepped, f"{pt} — Miss Distance",
                      _plot_miss_on_ax, window=window)

    # Suptitle
    pitcher_label = str(game_df["player_name"].iloc[0]) if "player_name" in game_df.columns and not game_df.empty else "Pitcher"
    game_date_label = str(game_df["game_date"].iloc[0]) if "game_date" in game_df.columns and not game_df.empty else ""
    fig.suptitle(
        f"{pitcher_label} | Game {game_pk} ({game_date_label}) — Arsenal Overview",
        fontsize=13, fontweight="bold", y=1.005,
    )

    if save_path is not None:
        from pathlib import Path as _Path
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Arsenal dashboard saved to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Arsenal dashboard — private helpers
# ---------------------------------------------------------------------------

def _render_on_ax(ax, prepped, title, render_fn, **kwargs):
    """Call render_fn(ax, prepped, **kwargs) and set a fallback title if empty."""
    render_fn(ax, prepped, **kwargs)
    has = prepped.get("has_data", True)
    if not has:
        ax.set_title(f"{title}\n(no data)", fontsize=8, color="gray")
    else:
        ax.set_title(title, fontsize=9)


def _plot_mix_on_ax(ax, prepped):
    if not prepped["has_data"]:
        return
    idx = prepped["pitch_index"]
    mix_df = prepped["mix_df"]
    for col in prepped["mix_cols"]:
        pt = col[:-4]
        color = prepped["color_map"].get(col, "#888888")
        ax.plot(idx, mix_df[col] * 100, color=color, linewidth=1.5, label=pt, zorder=3)
    ax.set_xlabel("Pitch #", fontsize=8)
    ax.set_ylabel("Usage (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, title="Type", loc="upper right")
    ax.grid(linestyle="--", alpha=0.3)


def _plot_spreads_on_ax(ax, prepped, window=7):
    if not prepped["has_data"]:
        return
    idx = prepped["pitch_index"]
    if not prepped["rolling_h_std"].empty:
        ax.plot(idx, prepped["rolling_h_std"], color="#2166ac", lw=1.5, label=f"H (w={window})")
    if not prepped["rolling_v_std"].empty:
        ax.plot(idx, prepped["rolling_v_std"], color="#d6604d", lw=1.5, ls="--", label=f"V (w={window})")
    ax.set_xlabel("Pitch #", fontsize=8)
    ax.set_ylabel("Spread (ft)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(linestyle="--", alpha=0.3)


def _render_pitch_type_zone_on_ax(ax, zone_prepped, pt, color):
    from src.visualization.zone_plots import _set_zone_axes
    pitches = zone_prepped["pitches"]
    hub_centers = zone_prepped.get("hub_centers") or {}
    subset = pitches[pitches[config.PITCH_TYPE_COL] == pt]
    ax.scatter(subset[config.PLATE_X_COL], subset[config.PLATE_Z_COL],
               c=color, alpha=0.45, s=16, zorder=3)
    if pt in hub_centers:
        for center in hub_centers[pt]:
            ax.scatter(center[0], center[1], c=color, marker="*", s=180,
                       edgecolors="black", linewidths=0.7, zorder=5)
    _set_zone_axes(ax)
    ax.set_title(f"{pt} — Locations (n={len(subset)})", fontsize=9, color=color)


def _plot_velo_on_ax(ax, prepped, window=7):
    from src.features.performance import ROLLING_VELO_MEAN_COL, ROLLING_VELO_STD_COL
    if not prepped["has_data"]:
        return
    VELO_COLOR = "#1b7837"
    idx = prepped["pitch_index"]
    mean = prepped["rolling_velo_mean"]
    std = prepped["rolling_velo_std"]
    if not prepped["release_speed"].empty:
        ax.scatter(idx, prepped["release_speed"], c=VELO_COLOR, alpha=0.2, s=10, zorder=2)
    if not mean.empty:
        ax.plot(idx, mean, color=VELO_COLOR, lw=1.8, zorder=3, label=f"Mean (w={window})")
    if not mean.empty and not std.empty:
        ax.fill_between(idx, mean - std, mean + std, alpha=0.12, color=VELO_COLOR)
    drop = prepped.get("velo_drop")
    if drop is not None:
        sign = "+" if drop < 0 else "-"
        ax.annotate(f"Δvelo: {sign}{abs(drop):.1f} mph", xy=(0.02, 0.05),
                    xycoords="axes fraction", fontsize=7, color="gray")
    ax.set_xlabel("Pitch #", fontsize=8)
    ax.set_ylabel("MPH", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(linestyle="--", alpha=0.3)


def _plot_perf_on_ax(ax, prepped, window=7):
    if not prepped["has_data"]:
        return
    panel_specs = [
        ("strike_pct", "Strike%", "#2166ac", "-"),
        ("zone_pct", "Zone%", "#4dac26", "--"),
        ("whiff_rate", "Whiff%", "#e08214", ":"),
    ]
    for key, label, color, ls in panel_specs:
        entry = prepped.get(key)
        if entry is None:
            continue
        ax.plot(entry["pitch_index"], entry["values"] * 100, color=color,
                lw=1.5, ls=ls, label=label, zorder=3)
    ax.set_xlabel("Pitch #", fontsize=8)
    ax.set_ylabel("Rate (%)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(linestyle="--", alpha=0.3)


def _plot_miss_on_ax(ax, prepped, window=7):
    from src.features.clustering import MISS_DISTANCE_COL
    from src.features.rolling import ROLLING_MISS_COL
    if not prepped["has_data"]:
        return
    idx = prepped["pitch_index"]
    color = "#4dac26"
    if not prepped["miss_distance"].empty:
        ax.scatter(idx, prepped["miss_distance"], c=color, alpha=0.2, s=10, zorder=2)
    if not prepped["rolling_miss_dist"].empty:
        ax.plot(idx, prepped["rolling_miss_dist"], color=color, lw=1.8, zorder=3,
                label=f"Mean (w={window})")
    ax.set_xlabel("Pitch #", fontsize=8)
    ax.set_ylabel("Miss dist (ft)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(linestyle="--", alpha=0.3)
