"""
Statcast data fetching and local caching.

Public API
----------
resolve_pitcher_id(name_or_id)          -> int
fetch_statcast_pitcher(...)             -> pd.DataFrame   (no caching)
load_or_fetch_pitcher(...)              -> pd.DataFrame   (cache-aware)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config
from src.utils.io import (
    cache_path_for_pitcher,
    load_parquet,
    save_parquet,
)
from src.utils.validation import validate_date_range, validate_pitcher_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pitcher ID resolution
# ---------------------------------------------------------------------------

def resolve_pitcher_id(name_or_id: str | int) -> int:
    """
    Resolve a pitcher name string or raw integer to an MLBAM integer ID.

    For integer inputs, validates and passes through.
    For string inputs, calls pybaseball.playerid_lookup() and returns the
    first pitcher result. Raises if no match found.

    Parameters
    ----------
    name_or_id : str or int
        MLBAM integer ID, or a player name like 'Gerrit Cole' or 'Cole, Gerrit'.

    Returns
    -------
    int
        MLBAM pitcher ID.

    Raises
    ------
    ValueError
        If a name string yields no results from the lookup.
    TypeError
        If name_or_id cannot be handled.
    """
    # Integer path — just validate and return
    try:
        return validate_pitcher_id(name_or_id)
    except (TypeError, ValueError):
        pass

    if not isinstance(name_or_id, str):
        raise TypeError(
            f"name_or_id must be an integer MLBAM ID or a player name string, "
            f"got {type(name_or_id).__name__!r}"
        )

    # String path — use pybaseball lookup
    try:
        from pybaseball import playerid_lookup  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for name-based pitcher lookup. "
            "Install it with: pip install pybaseball"
        ) from exc

    name = name_or_id.strip()
    # Support "First Last" or "Last, First"
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        last, first = parts[0], parts[1]
    else:
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            first, last = parts[0], parts[1]
        else:
            last, first = parts[0], ""

    results = playerid_lookup(last, first)

    if results.empty:
        raise ValueError(
            f"No player found matching {name_or_id!r}. "
            "Check spelling or use the MLBAM integer ID directly."
        )

    # playerid_lookup returns a 'key_mlbam' column
    if "key_mlbam" not in results.columns:
        raise ValueError(
            "pybaseball playerid_lookup did not return 'key_mlbam' column. "
            "Check pybaseball version."
        )

    mlbam_id = int(results.iloc[0]["key_mlbam"])
    logger.info("Resolved %r -> MLBAM ID %d", name_or_id, mlbam_id)
    return mlbam_id


# ---------------------------------------------------------------------------
# Raw Statcast fetch (no caching)
# ---------------------------------------------------------------------------

def fetch_statcast_pitcher(
    pitcher_id: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch Statcast pitch-level data for a single pitcher over a date range.

    Uses pybaseball.statcast_pitcher(). Does NOT cache — see
    load_or_fetch_pitcher() for the cache-aware version.

    Parameters
    ----------
    pitcher_id : int
        MLBAM pitcher ID.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Raw Statcast DataFrame as returned by pybaseball. Returns an
        empty DataFrame (not None) if pybaseball returns nothing.

    Raises
    ------
    ImportError
        If pybaseball is not installed.
    """
    try:
        from pybaseball import statcast_pitcher  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required. Install with: pip install pybaseball"
        ) from exc

    logger.info(
        "Fetching Statcast data: pitcher_id=%d, %s to %s",
        pitcher_id, start_date, end_date,
    )

    df = statcast_pitcher(start_date, end_date, pitcher_id)

    if df is None or df.empty:
        logger.warning(
            "pybaseball returned no data for pitcher_id=%d (%s to %s)",
            pitcher_id, start_date, end_date,
        )
        return pd.DataFrame()

    logger.info("Fetched %d rows for pitcher_id=%d", len(df), pitcher_id)
    return df


# ---------------------------------------------------------------------------
# Cache-aware loader
# ---------------------------------------------------------------------------

def load_or_fetch_pitcher(
    pitcher_id: int,
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Return cached data if it exists, otherwise fetch and cache it.

    Cache is stored as parquet at the path determined by
    cache_path_for_pitcher(). Passing use_cache=False always re-fetches
    and overwrites any existing cache file.

    Parameters
    ----------
    pitcher_id : int
    start_date : str  ('YYYY-MM-DD')
    end_date : str    ('YYYY-MM-DD')
    cache_dir : Path or None
        Override default cache directory. Uses config.CACHE_DIR if None.
    use_cache : bool, default True

    Returns
    -------
    pd.DataFrame
        Raw Statcast data (may be many columns wide).

    Raises
    ------
    ValueError
        If no data is found (empty DataFrame from pybaseball).
    """
    start_date, end_date = validate_date_range(start_date, end_date)
    pitcher_id = validate_pitcher_id(pitcher_id)

    base_dir = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
    cache_path = cache_path_for_pitcher(pitcher_id, start_date, end_date, base_dir)

    if use_cache and cache_path.exists():
        logger.info("Loading from cache: %s", cache_path)
        return load_parquet(cache_path)

    df = fetch_statcast_pitcher(pitcher_id, start_date, end_date)

    if df.empty:
        raise ValueError(
            f"No Statcast data found for pitcher_id={pitcher_id} "
            f"between {start_date} and {end_date}."
        )

    save_parquet(df, cache_path)
    logger.info("Cached %d rows to %s", len(df), cache_path)
    return df
