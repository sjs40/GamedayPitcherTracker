"""
Input validation utilities.

These are called at the boundaries of the service layer to catch bad inputs
early with clear error messages. Internal analytics functions can trust that
their inputs have been validated and skip redundant checks.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def validate_date_string(date_str: str) -> str:
    """
    Validate and normalize a date string to 'YYYY-MM-DD' format.

    Parameters
    ----------
    date_str : str
        Date string in any common format (e.g., '2024-04-01', '04/01/2024').

    Returns
    -------
    str
        Normalized 'YYYY-MM-DD' string.

    Raises
    ------
    ValueError
        If the string cannot be parsed as a date.
    TypeError
        If date_str is not a string.
    """
    if not isinstance(date_str, str):
        raise TypeError(f"Expected str for date, got {type(date_str).__name__!r}")

    formats = ["%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%m-%d-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError(
        f"Cannot parse {date_str!r} as a date. Expected format: 'YYYY-MM-DD'."
    )


def validate_pitcher_id(pitcher_id: int | str) -> int:
    """
    Validate that pitcher_id is a positive integer MLBAM ID.

    Parameters
    ----------
    pitcher_id : int or str
        Raw pitcher identifier.

    Returns
    -------
    int
        Confirmed integer MLBAM ID.

    Raises
    ------
    ValueError
        If pitcher_id is not a valid positive integer.
    TypeError
        If pitcher_id cannot be coerced to int.
    """
    try:
        pid = int(pitcher_id)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"pitcher_id must be coercible to int, got {pitcher_id!r}"
        ) from exc

    if pid <= 0:
        raise ValueError(f"pitcher_id must be a positive integer, got {pid}")

    return pid


def validate_pitch_type(pitch_type: str, available: list[str]) -> str:
    """
    Validate pitch_type is uppercase and present in the dataset.

    Parameters
    ----------
    pitch_type : str
        Requested pitch type code (e.g., 'FF', 'SL').
    available : list[str]
        Pitch types present in the loaded DataFrame.

    Returns
    -------
    str
        Uppercase, validated pitch type string.

    Raises
    ------
    ValueError
        If pitch_type (after uppercasing) is not in available pitch types.
    TypeError
        If pitch_type is not a string.
    """
    if not isinstance(pitch_type, str):
        raise TypeError(f"pitch_type must be str, got {type(pitch_type).__name__!r}")

    normalized = pitch_type.strip().upper()

    upper_available = [p.upper() for p in available]
    if normalized not in upper_available:
        raise ValueError(
            f"pitch_type {normalized!r} not found in dataset. "
            f"Available types: {sorted(available)}"
        )

    return normalized


def validate_dataframe_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Assert that all required columns are present in df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required : list[str]
        Column names that must be present.

    Raises
    ------
    ValueError
        With a message listing any missing columns.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Present columns: {list(df.columns)}"
        )


def validate_positive_int(value: int, name: str) -> int:
    """
    Validate that value is a positive integer.

    Parameters
    ----------
    value : int
        Value to check.
    name : str
        Parameter name for error messages.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If value <= 0.
    TypeError
        If value is not int-like.
    """
    try:
        v = int(value)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"{name} must be int, got {type(value).__name__!r}") from exc
    if v <= 0:
        raise ValueError(f"{name} must be a positive integer, got {v}")
    return v


def validate_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    """
    Validate that start_date <= end_date after normalizing both.

    Parameters
    ----------
    start_date : str
    end_date : str

    Returns
    -------
    tuple[str, str]
        (normalized_start, normalized_end)

    Raises
    ------
    ValueError
        If start_date > end_date.
    """
    s = validate_date_string(start_date)
    e = validate_date_string(end_date)
    if s > e:
        raise ValueError(
            f"start_date {s!r} must be <= end_date {e!r}"
        )
    return s, e
