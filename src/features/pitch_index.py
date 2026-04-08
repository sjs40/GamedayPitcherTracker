"""
Within-game pitch indexing and count bucketing.

These features must be added after sort_within_game() and before any
rolling or clustering computations.

Public API
----------
add_game_pitch_index(df)                    -> pd.DataFrame
add_pitch_count_buckets(df, n_buckets)      -> pd.DataFrame
add_count_string(df)                        -> pd.DataFrame
add_all_pitch_features(df)                  -> pd.DataFrame   (convenience)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Column names written by this module
GAME_PITCH_INDEX_COL = "game_pitch_index"
GAME_BUCKET_COL = "game_bucket"
COUNT_STRING_COL = "count_string"


def add_game_pitch_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 0-based sequential pitch index within each game.

    The index runs from 0 to N-1 where N is the number of pitches in that
    game. It is assigned in the order the DataFrame is sorted (so
    sort_within_game() must be called first).

    Column added: ``game_pitch_index`` (int)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, sorted DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy with 'game_pitch_index' column appended.
    """
    if config.GAME_PK_COL not in df.columns:
        logger.warning(
            "'game_pk' not found — assigning a single global pitch index"
        )
        df = df.copy()
        df[GAME_PITCH_INDEX_COL] = range(len(df))
        return df

    df = df.copy()
    df[GAME_PITCH_INDEX_COL] = df.groupby(config.GAME_PK_COL).cumcount()
    return df


def add_pitch_count_buckets(
    df: pd.DataFrame,
    n_buckets: int = config.N_BUCKETS_DEFAULT,
) -> pd.DataFrame:
    """
    Assign each pitch to an early/mid/late game bucket.

    Buckets are equal thirds (or n equal divisions) of the within-game
    pitch sequence. The assignment is per game, so a 70-pitch game yields
    roughly 23-24 pitches per bucket regardless of the total.

    Requires ``game_pitch_index`` to be present (call add_game_pitch_index first).

    Column added: ``game_bucket`` (str: one of config.BUCKET_LABELS[:n_buckets])

    Parameters
    ----------
    df : pd.DataFrame
    n_buckets : int, default 3
        Number of equal buckets. Use 3 for early/mid/late.

    Returns
    -------
    pd.DataFrame
        Copy with 'game_bucket' column appended.
    """
    if GAME_PITCH_INDEX_COL not in df.columns:
        raise ValueError(
            "Column 'game_pitch_index' not found. "
            "Call add_game_pitch_index() before add_pitch_count_buckets()."
        )

    if n_buckets > len(config.BUCKET_LABELS):
        raise ValueError(
            f"n_buckets={n_buckets} exceeds available label count "
            f"({len(config.BUCKET_LABELS)}). Add more labels to config.BUCKET_LABELS."
        )

    labels = config.BUCKET_LABELS[:n_buckets]

    def _assign_buckets(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        bins = np.linspace(0, n, n_buckets + 1, dtype=int)
        # pd.cut on the row position within this game's group
        positions = np.arange(n)
        # Use pandas cut with explicit bin edges; right=False means left-closed
        cut = pd.cut(
            positions,
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )
        return pd.Series(cut.astype(str), index=group.index)

    if config.GAME_PK_COL in df.columns:
        bucket_series = df.groupby(config.GAME_PK_COL, group_keys=False).apply(
            _assign_buckets
        )
    else:
        n = len(df)
        bins = np.linspace(0, n, n_buckets + 1, dtype=int)
        positions = np.arange(n)
        cut = pd.cut(
            positions,
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )
        bucket_series = pd.Series(cut.astype(str), index=df.index)

    df = df.copy()
    df[GAME_BUCKET_COL] = bucket_series
    return df


def add_count_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a human-readable count string like '1-2' (balls-strikes).

    Requires 'balls' and 'strikes' columns. If missing, the column is
    added as NaN without raising.

    Column added: ``count_string`` (str, e.g. '0-0', '3-2')

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Copy with 'count_string' column appended.
    """
    df = df.copy()

    if "balls" not in df.columns or "strikes" not in df.columns:
        logger.debug("'balls' or 'strikes' missing — count_string will be NaN")
        df[COUNT_STRING_COL] = pd.NA
        return df

    df[COUNT_STRING_COL] = (
        df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    )
    return df


def add_all_pitch_features(
    df: pd.DataFrame,
    n_buckets: int = config.N_BUCKETS_DEFAULT,
) -> pd.DataFrame:
    """
    Convenience function: apply all pitch-level feature additions in one call.

    Order: add_game_pitch_index → add_pitch_count_buckets → add_count_string

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, sorted DataFrame.
    n_buckets : int

    Returns
    -------
    pd.DataFrame
    """
    df = add_game_pitch_index(df)
    df = add_pitch_count_buckets(df, n_buckets=n_buckets)
    df = add_count_string(df)
    return df
