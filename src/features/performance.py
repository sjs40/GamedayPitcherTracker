"""
Rolling performance metrics for in-game pitcher analysis.

All functions follow the same pattern:
1. Filter to the requested game_pk (and optionally pitch_type)
2. Sort by game_pitch_index
3. Compute rolling metric(s) within that game only
4. Return a DataFrame with game_pitch_index as the first column

encode_pitch_outcomes() should be called once on the full dataset before
calling any of the rolling metric functions — it adds the boolean indicator
columns that the rolling functions depend on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
IS_STRIKE_COL = "is_strike"
IS_WHIFF_COL = "is_whiff"
IS_SWING_COL = "is_swing"
IS_IN_ZONE_COL = "is_in_zone"

ROLLING_VELO_MEAN_COL = "rolling_velo_mean"
ROLLING_VELO_STD_COL = "rolling_velo_std"
ROLLING_STRIKE_PCT_COL = "rolling_strike_pct"
ROLLING_WHIFF_RATE_COL = "rolling_whiff_rate"
ROLLING_ZONE_PCT_COL = "rolling_zone_pct"
ROLLING_PFX_X_STD_COL = "rolling_pfx_x_std"
ROLLING_PFX_Z_STD_COL = "rolling_pfx_z_std"
ROLLING_REL_X_STD_COL = "rolling_rel_x_std"
ROLLING_REL_Z_STD_COL = "rolling_rel_z_std"


# ---------------------------------------------------------------------------
# Outcome encoding (called once, adds boolean columns to df)
# ---------------------------------------------------------------------------

def encode_pitch_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean indicator columns derived from the 'description' and 'zone' fields.

    Columns added:
        is_strike  (bool): any pitch counting as a strike
        is_whiff   (bool): swinging miss only
        is_swing   (bool): any swing (contact or miss)
        is_in_zone (bool): pitch landed in Statcast zones 1-9

    Gracefully handles missing 'description' or 'zone' columns by filling
    the corresponding columns with False.

    Returns a copy of df with the new columns appended.
    """
    df = df.copy()

    if "description" in df.columns:
        desc = df["description"].fillna("").str.strip()
        df[IS_STRIKE_COL] = desc.isin(config.STRIKE_DESCRIPTIONS)
        df[IS_WHIFF_COL] = desc.isin(config.WHIFF_DESCRIPTIONS)
        df[IS_SWING_COL] = desc.isin(config.SWING_DESCRIPTIONS)
    else:
        df[IS_STRIKE_COL] = False
        df[IS_WHIFF_COL] = False
        df[IS_SWING_COL] = False

    if "zone" in df.columns:
        zone = pd.to_numeric(df["zone"], errors="coerce")
        df[IS_IN_ZONE_COL] = zone.isin(config.IN_ZONE_VALUES)
    else:
        df[IS_IN_ZONE_COL] = False

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_game(
    df: pd.DataFrame, game_pk: int, pitch_type: str | None
) -> pd.DataFrame:
    """Filter df to the requested game (and optionally pitch_type), sorted by index."""
    mask = df["game_pk"] == game_pk
    if pitch_type is not None:
        mask &= df["pitch_type"] == pitch_type
    return df.loc[mask].sort_values("game_pitch_index").reset_index(drop=True)


def _min_periods(window: int) -> int:
    return max(1, window // 2)


# ---------------------------------------------------------------------------
# Rolling velocity
# ---------------------------------------------------------------------------

def compute_rolling_velocity(
    df: pd.DataFrame, game_pk: int, pitch_type: str, window: int
) -> pd.DataFrame:
    """
    Rolling mean and std of release_speed within a single game.

    Returns columns: game_pitch_index, release_speed,
                     rolling_velo_mean, rolling_velo_std

    Returns an empty DataFrame if 'release_speed' is not present.
    """
    game = _filter_game(df, game_pk, pitch_type)
    if "release_speed" not in game.columns or game.empty:
        return pd.DataFrame(columns=["game_pitch_index", "release_speed",
                                     ROLLING_VELO_MEAN_COL, ROLLING_VELO_STD_COL])

    mp = _min_periods(window)
    speed = game["release_speed"]
    game = game[["game_pitch_index", "release_speed"]].copy()
    game[ROLLING_VELO_MEAN_COL] = speed.rolling(window, min_periods=mp).mean().values
    game[ROLLING_VELO_STD_COL] = speed.rolling(window, min_periods=mp).std().values
    return game.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rolling strike %
# ---------------------------------------------------------------------------

def compute_rolling_strike_pct(
    df: pd.DataFrame, game_pk: int, pitch_type: str, window: int
) -> pd.DataFrame:
    """
    Rolling percentage of pitches that are strikes (called, swinging, foul).

    Requires is_strike column (call encode_pitch_outcomes first).

    Returns columns: game_pitch_index, is_strike, rolling_strike_pct
    """
    game = _filter_game(df, game_pk, pitch_type)
    if game.empty or IS_STRIKE_COL not in game.columns:
        return pd.DataFrame(columns=["game_pitch_index", IS_STRIKE_COL,
                                     ROLLING_STRIKE_PCT_COL])

    mp = _min_periods(window)
    is_strike = game[IS_STRIKE_COL].astype(float)
    result = game[["game_pitch_index", IS_STRIKE_COL]].copy()
    result[ROLLING_STRIKE_PCT_COL] = is_strike.rolling(window, min_periods=mp).mean().values
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rolling whiff rate
# ---------------------------------------------------------------------------

def compute_rolling_whiff_rate(
    df: pd.DataFrame, game_pk: int, pitch_type: str, window: int
) -> pd.DataFrame:
    """
    Rolling whiff rate = whiffs / swings within the window.

    Windows with zero swings return NaN (no divide-by-zero).
    Requires is_whiff and is_swing columns (call encode_pitch_outcomes first).

    Returns columns: game_pitch_index, is_whiff, is_swing, rolling_whiff_rate
    """
    game = _filter_game(df, game_pk, pitch_type)
    required = {IS_WHIFF_COL, IS_SWING_COL}
    if game.empty or not required.issubset(game.columns):
        return pd.DataFrame(columns=["game_pitch_index", IS_WHIFF_COL,
                                     IS_SWING_COL, ROLLING_WHIFF_RATE_COL])

    mp = _min_periods(window)
    whiffs = game[IS_WHIFF_COL].astype(float)
    swings = game[IS_SWING_COL].astype(float)

    rolling_whiffs = whiffs.rolling(window, min_periods=mp).sum()
    rolling_swings = swings.rolling(window, min_periods=mp).sum()

    whiff_rate = rolling_whiffs / rolling_swings.replace(0, np.nan)

    result = game[["game_pitch_index", IS_WHIFF_COL, IS_SWING_COL]].copy()
    result[ROLLING_WHIFF_RATE_COL] = whiff_rate.values
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rolling zone %
# ---------------------------------------------------------------------------

def compute_rolling_zone_pct(
    df: pd.DataFrame, game_pk: int, pitch_type: str, window: int
) -> pd.DataFrame:
    """
    Rolling percentage of pitches landing in the strike zone (Statcast zones 1-9).

    Requires is_in_zone column (call encode_pitch_outcomes first).

    Returns columns: game_pitch_index, is_in_zone, rolling_zone_pct
    """
    game = _filter_game(df, game_pk, pitch_type)
    if game.empty or IS_IN_ZONE_COL not in game.columns:
        return pd.DataFrame(columns=["game_pitch_index", IS_IN_ZONE_COL,
                                     ROLLING_ZONE_PCT_COL])

    mp = _min_periods(window)
    in_zone = game[IS_IN_ZONE_COL].astype(float)
    result = game[["game_pitch_index", IS_IN_ZONE_COL]].copy()
    result[ROLLING_ZONE_PCT_COL] = in_zone.rolling(window, min_periods=mp).mean().values
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rolling movement consistency
# ---------------------------------------------------------------------------

def compute_rolling_movement_consistency(
    df: pd.DataFrame, game_pk: int, pitch_type: str, window: int
) -> pd.DataFrame:
    """
    Rolling std of pfx_x (horizontal break) and pfx_z (vertical break).

    Returns an empty DataFrame if pfx_x / pfx_z are not present.

    Returns columns: game_pitch_index, pfx_x, pfx_z,
                     rolling_pfx_x_std, rolling_pfx_z_std
    """
    game = _filter_game(df, game_pk, pitch_type)
    required = {"pfx_x", "pfx_z"}
    if game.empty or not required.issubset(game.columns):
        return pd.DataFrame(columns=["game_pitch_index", "pfx_x", "pfx_z",
                                     ROLLING_PFX_X_STD_COL, ROLLING_PFX_Z_STD_COL])

    mp = _min_periods(window)
    result = game[["game_pitch_index", "pfx_x", "pfx_z"]].copy()
    result[ROLLING_PFX_X_STD_COL] = (
        game["pfx_x"].rolling(window, min_periods=mp).std().values
    )
    result[ROLLING_PFX_Z_STD_COL] = (
        game["pfx_z"].rolling(window, min_periods=mp).std().values
    )
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rolling release consistency
# ---------------------------------------------------------------------------

def compute_rolling_release_consistency(
    df: pd.DataFrame, game_pk: int, pitch_type: str, window: int
) -> pd.DataFrame:
    """
    Rolling std of release_pos_x and release_pos_z.

    Returns an empty DataFrame if those columns are not present.

    Returns columns: game_pitch_index, release_pos_x, release_pos_z,
                     rolling_rel_x_std, rolling_rel_z_std
    """
    game = _filter_game(df, game_pk, pitch_type)
    required = {"release_pos_x", "release_pos_z"}
    if game.empty or not required.issubset(game.columns):
        return pd.DataFrame(columns=["game_pitch_index", "release_pos_x",
                                     "release_pos_z", ROLLING_REL_X_STD_COL,
                                     ROLLING_REL_Z_STD_COL])

    mp = _min_periods(window)
    result = game[["game_pitch_index", "release_pos_x", "release_pos_z"]].copy()
    result[ROLLING_REL_X_STD_COL] = (
        game["release_pos_x"].rolling(window, min_periods=mp).std().values
    )
    result[ROLLING_REL_Z_STD_COL] = (
        game["release_pos_z"].rolling(window, min_periods=mp).std().values
    )
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pitch mix (cross-type rolling)
# ---------------------------------------------------------------------------

def compute_pitch_mix(
    df: pd.DataFrame,
    game_pk: int,
    window: int = config.PITCH_MIX_WINDOW_DEFAULT,
) -> pd.DataFrame:
    """
    Rolling pitch type usage % across ALL pitch types simultaneously.

    For each pitch, computes the fraction of each pitch_type in the preceding
    `window` pitches.  Uses a wider default window (15) because individual
    type occurrences in a narrow window are too sparse.

    Returns columns: game_pitch_index, {pitch_type}_pct for each type present.

    Empty DataFrame returned if game has no rows.
    """
    game = _filter_game(df, game_pk, pitch_type=None)
    if game.empty:
        return pd.DataFrame(columns=["game_pitch_index"])

    mp = _min_periods(window)
    pitch_types = sorted(game["pitch_type"].dropna().unique())
    result = game[["game_pitch_index"]].copy()

    for pt in pitch_types:
        indicator = (game["pitch_type"] == pt).astype(float)
        col_name = f"{pt}_pct"
        result[col_name] = indicator.rolling(window, min_periods=mp).mean().values

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------

def compute_all_performance_metrics(
    df: pd.DataFrame,
    game_pk: int,
    pitch_type: str,
    window: int,
) -> dict[str, pd.DataFrame]:
    """
    Run all per-pitch-type rolling metrics and return them as a dict.

    Keys:
        'velocity'   → rolling_velo_mean, rolling_velo_std
        'strike_pct' → rolling_strike_pct
        'whiff_rate' → rolling_whiff_rate
        'zone_pct'   → rolling_zone_pct
        'movement'   → rolling_pfx_x_std, rolling_pfx_z_std
        'release'    → rolling_rel_x_std, rolling_rel_z_std

    Each value is a DataFrame indexed by game_pitch_index.
    Empty DataFrames are returned for metrics with missing source columns.
    """
    return {
        "velocity": compute_rolling_velocity(df, game_pk, pitch_type, window),
        "strike_pct": compute_rolling_strike_pct(df, game_pk, pitch_type, window),
        "whiff_rate": compute_rolling_whiff_rate(df, game_pk, pitch_type, window),
        "zone_pct": compute_rolling_zone_pct(df, game_pk, pitch_type, window),
        "movement": compute_rolling_movement_consistency(df, game_pk, pitch_type, window),
        "release": compute_rolling_release_consistency(df, game_pk, pitch_type, window),
    }
