"""
Rolling command stability visualizations.

Charts rolling horizontal/vertical spread and rolling miss distance
over the course of a game.

Two-mode design (same as zone_plots.py):
    prep_* returns chart-ready dicts
    plot_* renders matplotlib Figures

Public API
----------
prep_rolling_spreads(rolling_spreads_df) -> dict
plot_rolling_spreads(prepped, title, window) -> Figure
prep_rolling_miss(rolling_miss_df) -> dict
plot_rolling_miss(prepped, title, window) -> Figure
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src import config
from src.features.rolling import ROLLING_H_STD_COL, ROLLING_MISS_COL, ROLLING_V_STD_COL
from src.features.pitch_index import GAME_PITCH_INDEX_COL

logger = logging.getLogger(__name__)

# Colors for the two spread lines
H_COLOR = "#2166ac"  # blue — horizontal
V_COLOR = "#d6604d"  # red-orange — vertical
MISS_COLOR = "#4dac26"  # green — miss distance


# ---------------------------------------------------------------------------
# Rolling spreads
# ---------------------------------------------------------------------------

def prep_rolling_spreads(rolling_spreads_df: pd.DataFrame) -> dict:
    """
    Prepare rolling spread data for the command stability chart.

    Parameters
    ----------
    rolling_spreads_df : pd.DataFrame
        Output of compute_game_rolling_spreads(). Must contain
        game_pitch_index, rolling_h_std, rolling_v_std.

    Returns
    -------
    dict with keys:
        'pitch_index'      : pd.Series
        'rolling_h_std'    : pd.Series
        'rolling_v_std'    : pd.Series
        'n_pitches'        : int
        'has_data'         : bool
    """
    if rolling_spreads_df.empty or GAME_PITCH_INDEX_COL not in rolling_spreads_df.columns:
        return {
            "pitch_index": pd.Series([], dtype=float),
            "rolling_h_std": pd.Series([], dtype=float),
            "rolling_v_std": pd.Series([], dtype=float),
            "n_pitches": 0,
            "has_data": False,
        }

    df = rolling_spreads_df.sort_values(GAME_PITCH_INDEX_COL)
    return {
        "pitch_index": df[GAME_PITCH_INDEX_COL],
        "rolling_h_std": df[ROLLING_H_STD_COL] if ROLLING_H_STD_COL in df.columns else pd.Series(),
        "rolling_v_std": df[ROLLING_V_STD_COL] if ROLLING_V_STD_COL in df.columns else pd.Series(),
        "n_pitches": len(df),
        "has_data": True,
    }


def plot_rolling_spreads(
    prepped: dict,
    title: str = "",
    window: int = config.ROLLING_WINDOW_DEFAULT,
) -> Figure:
    """
    Line chart showing rolling horizontal and vertical spread over pitch number.

    X-axis = pitch number within game
    Y-axis = rolling standard deviation of plate_x (horizontal) and plate_z (vertical)
    Two lines on one panel; they are on the same scale (feet) so combining is readable.

    Note: early values may be computed from fewer pitches than the full window
    (min_periods = window // 2). This is annotated on the chart.

    Parameters
    ----------
    prepped : dict
        Output of prep_rolling_spreads().
    title : str
    window : int
        Window size, used for annotation only.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(9, 4), dpi=config.FIGURE_DPI)

    if not prepped["has_data"]:
        ax.set_title("No data available")
        return fig

    idx = prepped["pitch_index"]

    if not prepped["rolling_h_std"].empty:
        ax.plot(
            idx,
            prepped["rolling_h_std"],
            color=H_COLOR,
            linewidth=1.8,
            label=f"Horizontal spread (rolling std, w={window})",
        )

    if not prepped["rolling_v_std"].empty:
        ax.plot(
            idx,
            prepped["rolling_v_std"],
            color=V_COLOR,
            linewidth=1.8,
            linestyle="--",
            label=f"Vertical spread (rolling std, w={window})",
        )

    ax.set_xlabel("Pitch # within game", fontsize=9)
    ax.set_ylabel("Spread (ft)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

    # Annotation: early values may have fewer than window observations
    ax.annotate(
        f"* Early values use ≥{max(2, window // 2)} pitches (min_periods=window//2)",
        xy=(0.01, 0.02),
        xycoords="axes fraction",
        fontsize=7,
        color="gray",
    )

    ax.set_title(title or "Rolling Command Spread", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Rolling miss distance
# ---------------------------------------------------------------------------

def prep_rolling_miss(rolling_miss_df: pd.DataFrame) -> dict:
    """
    Prepare rolling miss distance data for charting.

    Parameters
    ----------
    rolling_miss_df : pd.DataFrame
        Output of compute_game_rolling_miss_distance(). Must contain
        game_pitch_index, miss_distance, rolling_miss_dist.

    Returns
    -------
    dict with keys:
        'pitch_index'       : pd.Series
        'miss_distance'     : pd.Series   (raw per-pitch miss)
        'rolling_miss_dist' : pd.Series   (rolling mean)
        'n_pitches'         : int
        'has_data'          : bool
    """
    from src.features.clustering import MISS_DISTANCE_COL

    if rolling_miss_df.empty or GAME_PITCH_INDEX_COL not in rolling_miss_df.columns:
        return {
            "pitch_index": pd.Series([], dtype=float),
            "miss_distance": pd.Series([], dtype=float),
            "rolling_miss_dist": pd.Series([], dtype=float),
            "n_pitches": 0,
            "has_data": False,
        }

    df = rolling_miss_df.sort_values(GAME_PITCH_INDEX_COL)
    return {
        "pitch_index": df[GAME_PITCH_INDEX_COL],
        "miss_distance": df[MISS_DISTANCE_COL] if MISS_DISTANCE_COL in df.columns else pd.Series(),
        "rolling_miss_dist": df[ROLLING_MISS_COL] if ROLLING_MISS_COL in df.columns else pd.Series(),
        "n_pitches": len(df),
        "has_data": True,
    }


def plot_rolling_miss(
    prepped: dict,
    title: str = "",
    window: int = config.ROLLING_WINDOW_DEFAULT,
) -> Figure:
    """
    Line chart showing rolling mean miss distance from location hub over pitch number.

    Per-pitch miss distances are shown as light scatter points; the rolling
    mean is shown as a solid line.

    Parameters
    ----------
    prepped : dict
        Output of prep_rolling_miss().
    title : str
    window : int

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(9, 4), dpi=config.FIGURE_DPI)

    if not prepped["has_data"]:
        ax.set_title("No miss distance data available — cluster model required")
        return fig

    idx = prepped["pitch_index"]

    # Raw per-pitch miss distances as faint scatter
    if not prepped["miss_distance"].empty:
        ax.scatter(
            idx,
            prepped["miss_distance"],
            color=MISS_COLOR,
            alpha=0.25,
            s=12,
            zorder=2,
            label="Per-pitch miss distance",
        )

    # Rolling mean as solid line
    if not prepped["rolling_miss_dist"].empty:
        ax.plot(
            idx,
            prepped["rolling_miss_dist"],
            color=MISS_COLOR,
            linewidth=2.0,
            zorder=3,
            label=f"Rolling mean miss dist (w={window})",
        )

    ax.set_xlabel("Pitch # within game", fontsize=9)
    ax.set_ylabel("Miss distance from hub (ft)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_title(title or "Rolling Miss Distance from Location Hub", fontsize=11)
    fig.tight_layout()
    return fig
