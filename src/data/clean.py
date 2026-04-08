"""
Statcast data cleaning and normalization.

Public API
----------
select_required_columns(df)     -> pd.DataFrame
drop_missing_locations(df)      -> pd.DataFrame
normalize_pitch_types(df)       -> pd.DataFrame
sort_within_game(df)            -> pd.DataFrame
clean_pitcher_data(df)          -> pd.DataFrame   (main entrypoint)
"""

from __future__ import annotations

import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only the columns listed in config.STATCAST_DESIRED_COLS.

    Columns absent from the DataFrame are silently skipped so this is
    robust to pybaseball schema changes. The returned DataFrame always
    contains at least the LOCATION_REQUIRED_COLS that survive cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Statcast DataFrame.

    Returns
    -------
    pd.DataFrame
        Subset of df with only the desired columns that actually exist.
    """
    keep = [c for c in config.STATCAST_DESIRED_COLS if c in df.columns]
    dropped = set(config.STATCAST_DESIRED_COLS) - set(keep)
    if dropped:
        logger.debug("Columns not present in raw data (skipped): %s", sorted(dropped))
    return df[keep].copy()


def drop_missing_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows missing any of the location-critical columns.

    Required columns: plate_x, plate_z, pitch_type.
    Rows with null values in any of these fields cannot be used for
    location analysis and are removed entirely.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with no nulls in required location columns.
    """
    present = [c for c in config.LOCATION_REQUIRED_COLS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=present)
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows with missing location data", dropped)
    return df.reset_index(drop=True)


def normalize_pitch_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize pitch_type values to uppercase with no leading/trailing whitespace.

    Also replaces empty strings and 'UN' (unknown) with NaN so they can be
    caught by drop_missing_locations().

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if config.PITCH_TYPE_COL not in df.columns:
        return df

    df = df.copy()
    df[config.PITCH_TYPE_COL] = (
        df[config.PITCH_TYPE_COL]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "UN": pd.NA})
    )
    return df


def sort_within_game(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort DataFrame to establish correct within-game pitch ordering.

    Sort order: game_date → game_pk → at_bat_number → pitch_number.
    Columns that don't exist are skipped. This ordering must be done before
    add_game_pitch_index() is called.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Sorted copy with reset index.
    """
    sort_cols = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    available = [c for c in sort_cols if c in df.columns]
    if not available:
        logger.warning("No sort columns found — returning df as-is")
        return df

    return df.sort_values(available).reset_index(drop=True)


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce plate_x, plate_z, and release_speed to float if they aren't already.

    pybaseball sometimes returns these as object dtype if the column
    contained mixed types.
    """
    numeric_cols = ["plate_x", "plate_z", "release_speed"]
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_game_date_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure game_date is a consistent string 'YYYY-MM-DD'.

    pybaseball may return it as a datetime or object; normalize to string.
    """
    if "game_date" not in df.columns:
        return df
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    return df


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def clean_pitcher_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for a raw Statcast pitcher DataFrame.

    Steps (in order):
        1. select_required_columns      — trim to desired fields
        2. normalize_pitch_types        — uppercase, remove unknowns
        3. _coerce_numeric_columns      — ensure plate_x/z are float
        4. _add_game_date_str           — normalize game_date to string
        5. drop_missing_locations       — remove rows with null plate_x/z/pitch_type
        6. sort_within_game             — establish correct pitch ordering

    Parameters
    ----------
    df : pd.DataFrame
        Raw Statcast DataFrame (from fetch.py).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.

    Raises
    ------
    ValueError
        If the DataFrame is empty after cleaning (all rows were invalid).
    """
    logger.info("Starting clean_pitcher_data on %d rows", len(df))

    df = select_required_columns(df)
    df = normalize_pitch_types(df)
    df = _coerce_numeric_columns(df)
    df = _add_game_date_str(df)
    df = drop_missing_locations(df)
    df = sort_within_game(df)

    if df.empty:
        raise ValueError(
            "DataFrame is empty after cleaning. "
            "Check that the raw data contains valid plate_x, plate_z, and pitch_type."
        )

    logger.info("Cleaning complete: %d usable rows", len(df))
    return df
