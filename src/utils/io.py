"""
File I/O utilities for saving and loading DataFrames.

Parquet is the preferred format for cached data (efficient, typed).
CSV is provided for human-readable exports and notebook inspection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """
    Save DataFrame to parquet, creating parent directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
    path : Path
        Destination file path (should end in .parquet).

    Returns
    -------
    Path
        The path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    """
    Load a parquet file into a DataFrame.

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet cache not found: {path}")
    return pd.read_parquet(path)


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Save DataFrame to CSV (index=False), creating parent directories.

    Parameters
    ----------
    df : pd.DataFrame
    path : Path

    Returns
    -------
    Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def cache_path_for_pitcher(
    pitcher_id: int,
    start_date: str,
    end_date: str,
    base_dir: Path,
) -> Path:
    """
    Compute a deterministic cache file path for a pitcher + date range.

    File is named: {pitcher_id}_{start_date}_{end_date}.parquet
    Date dashes are stripped to avoid any path-separator ambiguity.

    Parameters
    ----------
    pitcher_id : int
    start_date : str  ('YYYY-MM-DD')
    end_date : str    ('YYYY-MM-DD')
    base_dir : Path

    Returns
    -------
    Path
    """
    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    filename = f"{pitcher_id}_{s}_{e}.parquet"
    return Path(base_dir) / filename


def cache_exists(
    pitcher_id: int,
    start_date: str,
    end_date: str,
    base_dir: Path,
) -> bool:
    """Return True if a cache file exists for the given pitcher + date range."""
    return cache_path_for_pitcher(pitcher_id, start_date, end_date, base_dir).exists()
