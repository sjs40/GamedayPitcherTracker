"""
Rolling command metrics computed within game boundaries.

All functions that operate per-game enforce that rolling windows do NOT
cross game boundaries. Each function accepts the full cleaned DataFrame
and filters to the target game internally.

Public API
----------
rolling_std(series, window, min_periods)
    -> pd.Series

rolling_mean(series, window, min_periods)
    -> pd.Series

compute_game_rolling_spreads(df, game_pk, pitch_type, window)
    -> pd.DataFrame

compute_game_rolling_miss_distance(df, game_pk, pitch_type, window, cluster_model)
    -> pd.DataFrame
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.features.clustering import (
    MISS_DISTANCE_COL,
    PitchTypeClusterModel,
    assign_clusters_to_game,
)
from src.features.pitch_index import GAME_PITCH_INDEX_COL

logger = logging.getLogger(__name__)

# Output column names
ROLLING_H_STD_COL = "rolling_h_std"
ROLLING_V_STD_COL = "rolling_v_std"
ROLLING_MISS_COL = "rolling_miss_dist"


# ---------------------------------------------------------------------------
# Core rolling primitives
# ---------------------------------------------------------------------------

def rolling_std(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Compute rolling standard deviation with a trailing (right-aligned) window.

    Parameters
    ----------
    series : pd.Series
        Ordered numeric series (e.g., plate_x values within a game).
    window : int
        Rolling window size in pitches.
    min_periods : int or None
        Minimum observations required to produce a value.
        Defaults to ``window // 2``, allowing early-game values to appear
        without requiring a full window. These early values have higher
        variance due to smaller samples — this is documented in chart
        annotations rather than hidden with NaN.

    Returns
    -------
    pd.Series
        Rolling std values, same index as input.
    """
    if min_periods is None:
        min_periods = max(2, window // 2)
    return series.rolling(window=window, min_periods=min_periods).std()


def rolling_mean(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Compute rolling mean with trailing window.

    Same signature and min_periods convention as rolling_std.

    Parameters
    ----------
    series : pd.Series
    window : int
    min_periods : int or None

    Returns
    -------
    pd.Series
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    return series.rolling(window=window, min_periods=min_periods).mean()


# ---------------------------------------------------------------------------
# Game-level rolling spread
# ---------------------------------------------------------------------------

def compute_game_rolling_spreads(
    df: pd.DataFrame,
    game_pk: int,
    pitch_type: str | None,
    window: int = config.ROLLING_WINDOW_DEFAULT,
) -> pd.DataFrame:
    """
    Compute rolling horizontal and vertical spread for a single game.

    Filters to game_pk (and optionally pitch_type), sorts by
    game_pitch_index, then computes rolling_std on plate_x and plate_z
    separately. Rolling is always done within this single game — no
    cross-game contamination is possible because we filter first.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, feature-engineered DataFrame.
    game_pk : int
    pitch_type : str or None
        If provided, filter to this pitch type before rolling.
        If None, compute over all pitch types together.
    window : int
        Rolling window in pitches.

    Returns
    -------
    pd.DataFrame
        Columns: game_pitch_index, plate_x, plate_z,
                 rolling_h_std, rolling_v_std.
        Sorted by game_pitch_index. May be shorter than the input if
        min_periods NaN rows are present (these are kept, not dropped,
        so the chart x-axis aligns with pitch number).
    """
    # Filter to game
    if config.GAME_PK_COL in df.columns:
        game_df = df[df[config.GAME_PK_COL] == game_pk].copy()
    else:
        logger.warning("game_pk column not found — using entire DataFrame")
        game_df = df.copy()

    # Optionally filter to pitch type
    if pitch_type is not None:
        game_df = game_df[game_df[config.PITCH_TYPE_COL] == pitch_type]

    if game_df.empty:
        logger.warning(
            "No pitches found for game_pk=%s pitch_type=%s", game_pk, pitch_type
        )
        return pd.DataFrame(
            columns=[
                GAME_PITCH_INDEX_COL,
                config.PLATE_X_COL,
                config.PLATE_Z_COL,
                ROLLING_H_STD_COL,
                ROLLING_V_STD_COL,
            ]
        )

    # Sort by within-game index
    if GAME_PITCH_INDEX_COL in game_df.columns:
        game_df = game_df.sort_values(GAME_PITCH_INDEX_COL).reset_index(drop=True)

    game_df[ROLLING_H_STD_COL] = rolling_std(
        game_df[config.PLATE_X_COL], window=window
    )
    game_df[ROLLING_V_STD_COL] = rolling_std(
        game_df[config.PLATE_Z_COL], window=window
    )

    keep_cols = [
        c for c in [
            GAME_PITCH_INDEX_COL,
            config.GAME_PK_COL,
            config.PITCH_TYPE_COL,
            config.PLATE_X_COL,
            config.PLATE_Z_COL,
            ROLLING_H_STD_COL,
            ROLLING_V_STD_COL,
        ] if c in game_df.columns
    ]
    return game_df[keep_cols]


# ---------------------------------------------------------------------------
# Game-level rolling miss distance
# ---------------------------------------------------------------------------

def compute_game_rolling_miss_distance(
    df: pd.DataFrame,
    game_pk: int,
    pitch_type: str,
    window: int = config.ROLLING_WINDOW_DEFAULT,
    cluster_model: PitchTypeClusterModel | None = None,
) -> pd.DataFrame:
    """
    Compute rolling mean Euclidean miss distance for a single game+pitch_type.

    For each pitch, miss distance = Euclidean distance from (plate_x, plate_z)
    to the centroid of its assigned cluster (location hub). Then compute
    a rolling mean of miss distances ordered by game_pitch_index.

    If cluster_model is None, this function returns an empty DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, feature-engineered DataFrame. Must have cluster_label and
        miss_distance columns (i.e., assign_clusters_to_game must have
        been called, or we call it here if cluster_model is provided).
    game_pk : int
    pitch_type : str
    window : int
    cluster_model : PitchTypeClusterModel or None
        If provided, cluster assignment is performed internally.
        If None, the caller is responsible for adding MISS_DISTANCE_COL
        to df before calling.

    Returns
    -------
    pd.DataFrame
        Columns: game_pitch_index, plate_x, plate_z,
                 miss_distance, rolling_miss_dist, cluster_label.
        Sorted by game_pitch_index.
    """
    if cluster_model is None:
        if MISS_DISTANCE_COL not in df.columns:
            logger.warning(
                "cluster_model is None and miss_distance not in df — "
                "cannot compute rolling miss distance"
            )
            return pd.DataFrame(
                columns=[
                    GAME_PITCH_INDEX_COL,
                    config.PLATE_X_COL,
                    config.PLATE_Z_COL,
                    MISS_DISTANCE_COL,
                    ROLLING_MISS_COL,
                ]
            )
        # Use pre-existing miss_distance column
        if config.GAME_PK_COL in df.columns:
            game_df = df[
                (df[config.GAME_PK_COL] == game_pk)
                & (df[config.PITCH_TYPE_COL] == pitch_type)
            ].copy()
        else:
            game_df = df[df[config.PITCH_TYPE_COL] == pitch_type].copy()
    else:
        # Assign clusters and compute miss_distance internally
        game_df = assign_clusters_to_game(df, game_pk, pitch_type, cluster_model)

    if game_df.empty:
        return pd.DataFrame(
            columns=[
                GAME_PITCH_INDEX_COL,
                config.PLATE_X_COL,
                config.PLATE_Z_COL,
                MISS_DISTANCE_COL,
                ROLLING_MISS_COL,
            ]
        )

    if GAME_PITCH_INDEX_COL in game_df.columns:
        game_df = game_df.sort_values(GAME_PITCH_INDEX_COL).reset_index(drop=True)

    game_df[ROLLING_MISS_COL] = rolling_mean(
        game_df[MISS_DISTANCE_COL], window=window
    )

    keep_cols = [
        c for c in [
            GAME_PITCH_INDEX_COL,
            config.GAME_PK_COL,
            config.PITCH_TYPE_COL,
            config.PLATE_X_COL,
            config.PLATE_Z_COL,
            "cluster_label",
            MISS_DISTANCE_COL,
            ROLLING_MISS_COL,
        ] if c in game_df.columns
    ]
    return game_df[keep_cols]
