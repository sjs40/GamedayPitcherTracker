"""
Service / orchestration layer for pitcher command drift analysis.

This is the primary public API surface. Both notebooks and any future
GUI/web app should call only these functions — not the lower-level
analytics modules directly.

Each function:
  - takes explicit, human-scale parameters
  - validates inputs before doing work
  - returns clean, documented outputs
  - fails gracefully with informative errors

Public API
----------
get_pitcher_dataset(...)           -> pd.DataFrame
get_available_games(df)            -> list[GameMetadata]
get_available_pitch_types(df)      -> list[str]
fit_pitch_type_clusters(...)       -> PitchTypeClusterModel | None
compute_game_command_metrics(...)  -> dict
compute_cluster_drift(...)         -> GameDriftReport
build_game_report(...)             -> GameReport
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config
from src.data.clean import clean_pitcher_data
from src.data.fetch import load_or_fetch_pitcher, resolve_pitcher_id
from src.features.clustering import (
    PitchTypeClusterModel,
    assign_clusters_to_game,
    fit_cluster_model,
)
from src.features.drift import GameDriftReport
from src.features.drift import compute_cluster_drift as _compute_cluster_drift
from src.features.pitch_index import add_all_pitch_features
from src.features.rolling import (
    compute_game_rolling_miss_distance,
    compute_game_rolling_spreads,
)
from src.utils.validation import (
    validate_date_range,
    validate_pitcher_id,
    validate_positive_int,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data classes
# ---------------------------------------------------------------------------

@dataclass
class GameMetadata:
    """
    Lightweight game descriptor for navigation and selection UI.

    Attributes
    ----------
    game_pk : int
    game_date : str  ('YYYY-MM-DD')
    opponent : str or None
        Home or away team if available in the data.
    pitch_count : int
        Total pitches in this game for the selected pitcher.
    pitch_types_available : list[str]
        Pitch types with enough pitches to support analysis.
    """
    game_pk: int
    game_date: str
    opponent: str | None
    pitch_count: int
    pitch_types_available: list[str]

    def __str__(self) -> str:
        return (
            f"{self.game_date} (game_pk={self.game_pk}, "
            f"{self.pitch_count} pitches, "
            f"types={self.pitch_types_available})"
        )


@dataclass
class GameReport:
    """
    Full analysis report for a single pitcher + game + pitch type.

    This is the primary output contract of the service layer. It contains
    everything a notebook or GUI needs to render charts and tables.

    Attributes
    ----------
    pitcher_id : int
    game_pk : int
    game_date : str
    pitch_type : str
    cluster_model : PitchTypeClusterModel
        The baseline cluster model (location hubs) used for this analysis.
    rolling_spreads : pd.DataFrame
        Output of compute_game_rolling_spreads(). Ready for rolling_plots.
        Columns: game_pitch_index, plate_x, plate_z, rolling_h_std, rolling_v_std.
    rolling_miss : pd.DataFrame
        Output of compute_game_rolling_miss_distance(). Ready for rolling_plots.
        Columns: game_pitch_index, plate_x, plate_z, miss_distance, rolling_miss_dist.
    game_with_clusters : pd.DataFrame
        Game-filtered df with cluster_label and miss_distance columns.
        Ready for zone_plots.
    drift_report : GameDriftReport
        Early/mid/late bucket drift vs. baseline centers. Ready for dashboards.
    params : dict
        Echo of parameters used: window, k, min_pitches, start_date, end_date.
    warnings : list[str]
        Non-fatal issues (e.g., "k reduced from 4 to 3 due to small sample").
    """
    pitcher_id: int
    game_pk: int
    game_date: str
    pitch_type: str
    cluster_model: PitchTypeClusterModel
    rolling_spreads: pd.DataFrame
    rolling_miss: pd.DataFrame
    game_with_clusters: pd.DataFrame
    drift_report: GameDriftReport
    params: dict
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------

def get_pitcher_dataset(
    pitcher_id: int | str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch, cache, and clean Statcast data for a pitcher over a date range.

    This is the canonical entry point for loading pitcher data. It handles
    name-to-ID resolution, caching, and cleaning in one call.

    Parameters
    ----------
    pitcher_id : int or str
        MLBAM integer ID or player name (e.g., 'Gerrit Cole').
    start_date : str  ('YYYY-MM-DD')
    end_date : str    ('YYYY-MM-DD')
    use_cache : bool, default True
        If True, serve from parquet cache when available.
    cache_dir : Path or None
        Override default cache directory (config.CACHE_DIR).

    Returns
    -------
    pd.DataFrame
        Cleaned, sorted DataFrame with game_pitch_index and game_bucket
        columns added (ready for feature engineering).

    Raises
    ------
    ValueError
        If dates are invalid, or if no data is returned.
    """
    # Resolve name to int ID if needed
    resolved_id = resolve_pitcher_id(pitcher_id)
    start, end = validate_date_range(start_date, end_date)

    raw_df = load_or_fetch_pitcher(
        pitcher_id=resolved_id,
        start_date=start,
        end_date=end,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    cleaned = clean_pitcher_data(raw_df)
    featurized = add_all_pitch_features(cleaned)

    logger.info(
        "Dataset ready: pitcher_id=%d, %d pitches across %d games",
        resolved_id,
        len(featurized),
        featurized[config.GAME_PK_COL].nunique()
        if config.GAME_PK_COL in featurized.columns
        else 0,
    )
    return featurized


def get_available_games(df: pd.DataFrame) -> list[GameMetadata]:
    """
    Extract metadata for every game present in the DataFrame.

    Returns one GameMetadata per unique game_pk, sorted by game_date ascending.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, feature-engineered DataFrame.

    Returns
    -------
    list[GameMetadata]
        Sorted list of game descriptors.
    """
    if config.GAME_PK_COL not in df.columns:
        # Fall back to treating the whole dataset as one game
        return [
            GameMetadata(
                game_pk=0,
                game_date=str(df["game_date"].iloc[0]) if "game_date" in df.columns else "unknown",
                opponent=None,
                pitch_count=len(df),
                pitch_types_available=sorted(df[config.PITCH_TYPE_COL].dropna().unique().tolist()),
            )
        ]

    games = []
    for game_pk, group in df.groupby(config.GAME_PK_COL):
        game_date = (
            str(group["game_date"].iloc[0]) if "game_date" in group.columns else "unknown"
        )

        # Try to derive opponent from home_team / away_team if present
        opponent = None
        if "home_team" in group.columns and "away_team" in group.columns:
            home = group["home_team"].iloc[0]
            away = group["away_team"].iloc[0]
            opponent = f"{away} @ {home}"

        pitch_types = sorted(
            group[config.PITCH_TYPE_COL].dropna().unique().tolist()
        )

        games.append(
            GameMetadata(
                game_pk=int(game_pk),
                game_date=game_date,
                opponent=opponent,
                pitch_count=len(group),
                pitch_types_available=pitch_types,
            )
        )

    games.sort(key=lambda g: g.game_date)
    return games


def get_available_pitch_types(
    df: pd.DataFrame,
    min_pitches: int = config.MIN_PITCHES_DEFAULT,
    game_pk: int | None = None,
) -> list[str]:
    """
    Return pitch types with at least min_pitches in the dataset (or game).

    Parameters
    ----------
    df : pd.DataFrame
    min_pitches : int
        Minimum pitch count for a type to be included.
    game_pk : int or None
        If provided, filter to this game before counting.

    Returns
    -------
    list[str]
        Sorted list of qualifying pitch type codes.
    """
    subset = df
    if game_pk is not None and config.GAME_PK_COL in df.columns:
        subset = df[df[config.GAME_PK_COL] == game_pk]

    counts = subset[config.PITCH_TYPE_COL].value_counts()
    return sorted(counts[counts >= min_pitches].index.tolist())


def fit_pitch_type_clusters(
    df: pd.DataFrame,
    pitcher_id: int,
    pitch_type: str,
    k: int = config.K_DEFAULT,
    min_pitches: int = config.MIN_PITCHES_DEFAULT,
) -> PitchTypeClusterModel | None:
    """
    Fit baseline KMeans location-hub clusters for one pitcher+pitch_type.

    The baseline is all pitches in df for this pitcher/pitch_type (typically
    spanning multiple games in the selected date range). The returned model
    is then used to assign clusters to individual games — never refitted
    per game.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (multi-game). More data = more stable cluster centers.
    pitcher_id : int
    pitch_type : str
    k : int, default 4
    min_pitches : int, default 15

    Returns
    -------
    PitchTypeClusterModel or None
        None if this pitch type has fewer than min_pitches in the dataset.
    """
    pitcher_id = validate_pitcher_id(pitcher_id)
    k = validate_positive_int(k, "k")
    min_pitches = validate_positive_int(min_pitches, "min_pitches")

    return fit_cluster_model(
        df=df,
        pitcher_id=pitcher_id,
        pitch_type=pitch_type,
        k=k,
        min_pitches=min_pitches,
    )


def compute_game_command_metrics(
    df: pd.DataFrame,
    game_pk: int,
    pitch_type: str | None,
    window: int = config.ROLLING_WINDOW_DEFAULT,
    cluster_model: PitchTypeClusterModel | None = None,
) -> dict:
    """
    Compute rolling command metrics for a single game.

    Returns a dict with rolling spread and (optionally) rolling miss distance
    DataFrames. Both are indexed by game_pitch_index and ready for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (must have game_pitch_index, game_bucket).
    game_pk : int
    pitch_type : str or None
        None computes spreads over all pitch types together. Miss distance
        requires a specific pitch type.
    window : int, default 7
    cluster_model : PitchTypeClusterModel or None
        Required for rolling miss distance. If None, rolling_miss is empty.

    Returns
    -------
    dict with keys:
        'rolling_spreads' : pd.DataFrame
        'rolling_miss'    : pd.DataFrame (empty if cluster_model is None)
    """
    rolling_spreads = compute_game_rolling_spreads(
        df=df,
        game_pk=game_pk,
        pitch_type=pitch_type,
        window=window,
    )

    rolling_miss = (
        compute_game_rolling_miss_distance(
            df=df,
            game_pk=game_pk,
            pitch_type=pitch_type,
            window=window,
            cluster_model=cluster_model,
        )
        if cluster_model is not None and pitch_type is not None
        else pd.DataFrame()
    )

    return {"rolling_spreads": rolling_spreads, "rolling_miss": rolling_miss}


def compute_cluster_drift(
    df: pd.DataFrame,
    game_pk: int,
    pitch_type: str,
    cluster_model: PitchTypeClusterModel,
) -> GameDriftReport:
    """
    Compute cluster center drift for a single game + pitch type.

    Delegates to features.drift.compute_cluster_drift. The df must
    already have cluster_label and game_bucket columns.

    Parameters
    ----------
    df : pd.DataFrame
    game_pk : int
    pitch_type : str
    cluster_model : PitchTypeClusterModel

    Returns
    -------
    GameDriftReport
    """
    return _compute_cluster_drift(
        df=df,
        game_pk=game_pk,
        pitch_type=pitch_type,
        cluster_model=cluster_model,
    )


def build_game_report(
    pitcher_id: int | str,
    game_pk: int,
    pitch_type: str,
    start_date: str,
    end_date: str,
    window: int = config.ROLLING_WINDOW_DEFAULT,
    k: int = config.K_DEFAULT,
    min_pitches: int = config.MIN_PITCHES_DEFAULT,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> GameReport:
    """
    Build a complete analysis report for one pitcher, game, and pitch type.

    This is the main entry point for end-to-end analysis. It:
        1. Resolves pitcher ID (handles name strings)
        2. Fetches and cleans data (cache-aware)
        3. Adds pitch features (game index, buckets)
        4. Fits baseline cluster model for the pitch type
        5. Assigns clusters to the target game
        6. Computes rolling spreads and miss distance
        7. Computes cluster drift (early/mid/late)
        8. Assembles and returns a GameReport

    Parameters
    ----------
    pitcher_id : int or str
        MLBAM integer ID or player name (e.g., 'Gerrit Cole').
    game_pk : int
        Target game. Must be present in the loaded dataset.
    pitch_type : str
        Pitch type to analyze (e.g., 'FF' for four-seam fastball).
    start_date : str  ('YYYY-MM-DD')
        Start of the data range used for baseline clustering.
    end_date : str    ('YYYY-MM-DD')
    window : int, default 7
        Rolling window size in pitches.
    k : int, default 4
        Number of KMeans clusters.
    min_pitches : int, default 15
        Minimum pitches per pitch type for cluster fitting.
    use_cache : bool, default True
    cache_dir : Path or None

    Returns
    -------
    GameReport
        Contains all DataFrames and summaries needed for visualization.

    Raises
    ------
    ValueError
        If pitcher has no data in the date range, game_pk not found,
        or pitch_type has too few pitches to cluster.
    """
    warnings: list[str] = []

    # Step 1: Load dataset
    resolved_id = resolve_pitcher_id(pitcher_id)
    df = get_pitcher_dataset(
        pitcher_id=resolved_id,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )

    # Step 2: Validate game_pk
    if config.GAME_PK_COL in df.columns:
        available_games = df[config.GAME_PK_COL].unique().tolist()
        if game_pk not in available_games:
            raise ValueError(
                f"game_pk={game_pk} not found in dataset. "
                f"Available game_pks: {sorted(available_games)}"
            )

    # Step 3: Validate pitch type and normalize
    from src.utils.validation import validate_pitch_type
    available_types = df[config.PITCH_TYPE_COL].dropna().unique().tolist()
    pitch_type = validate_pitch_type(pitch_type, available_types)

    # Step 4: Fit baseline cluster model
    cluster_model = fit_pitch_type_clusters(
        df=df,
        pitcher_id=resolved_id,
        pitch_type=pitch_type,
        k=k,
        min_pitches=min_pitches,
    )

    if cluster_model is None:
        raise ValueError(
            f"Insufficient data to fit cluster model for pitch_type={pitch_type!r}. "
            f"Found fewer than {min_pitches} pitches. "
            f"Try a wider date range or lower min_pitches."
        )

    if cluster_model.k < k:
        warnings.append(
            f"k reduced from {k} to {cluster_model.k} due to small sample size "
            f"({cluster_model.n_pitches_fit} pitches for {pitch_type})."
        )

    # Step 5: Assign clusters to the target game
    game_with_clusters = assign_clusters_to_game(
        df=df,
        game_pk=game_pk,
        pitch_type=pitch_type,
        cluster_model=cluster_model,
    )

    if game_with_clusters.empty:
        raise ValueError(
            f"No {pitch_type} pitches found for game_pk={game_pk}."
        )

    # We need game_bucket on the game-level data for drift
    # It should already be there from add_all_pitch_features
    from src.features.pitch_index import GAME_BUCKET_COL
    if GAME_BUCKET_COL not in game_with_clusters.columns:
        # Merge bucket from df
        bucket_map = df.set_index(df.index)[GAME_BUCKET_COL] if GAME_BUCKET_COL in df.columns else None
        if bucket_map is not None:
            game_with_clusters = game_with_clusters.merge(
                df[[config.GAME_PK_COL, "at_bat_number", "pitch_number", GAME_BUCKET_COL]]
                if all(c in df.columns for c in ["at_bat_number", "pitch_number"])
                else df[[GAME_BUCKET_COL]],
                left_index=True,
                right_index=True,
                how="left",
            )

    # Step 6: Rolling metrics
    metrics = compute_game_command_metrics(
        df=df,
        game_pk=game_pk,
        pitch_type=pitch_type,
        window=window,
        cluster_model=cluster_model,
    )

    # Step 7: Drift
    # Merge cluster assignments back into main df for drift computation
    # (drift needs game_bucket which is on df)
    if config.GAME_PK_COL in df.columns:
        game_mask = (df[config.GAME_PK_COL] == game_pk) & (df[config.PITCH_TYPE_COL] == pitch_type)
        df_for_drift = df[game_mask].copy()
        # Add cluster labels
        from src.features.clustering import CLUSTER_LABEL_COL, MISS_DISTANCE_COL
        if CLUSTER_LABEL_COL not in df_for_drift.columns:
            df_for_drift = df_for_drift.join(
                game_with_clusters[[CLUSTER_LABEL_COL, MISS_DISTANCE_COL]],
                how="left",
                rsuffix="_new",
            )
            # Use the new columns if they exist
            if f"{CLUSTER_LABEL_COL}_new" in df_for_drift.columns:
                df_for_drift[CLUSTER_LABEL_COL] = df_for_drift[f"{CLUSTER_LABEL_COL}_new"]
                df_for_drift[MISS_DISTANCE_COL] = df_for_drift[f"{MISS_DISTANCE_COL}_new"]
    else:
        df_for_drift = game_with_clusters

    drift_report = compute_cluster_drift(
        df=df_for_drift,
        game_pk=game_pk,
        pitch_type=pitch_type,
        cluster_model=cluster_model,
    )

    # Step 8: Get game date
    game_date = "unknown"
    if "game_date" in df.columns and config.GAME_PK_COL in df.columns:
        date_series = df.loc[df[config.GAME_PK_COL] == game_pk, "game_date"]
        if not date_series.empty:
            game_date = str(date_series.iloc[0])

    return GameReport(
        pitcher_id=resolved_id,
        game_pk=game_pk,
        game_date=game_date,
        pitch_type=pitch_type,
        cluster_model=cluster_model,
        rolling_spreads=metrics["rolling_spreads"],
        rolling_miss=metrics["rolling_miss"],
        game_with_clusters=game_with_clusters,
        drift_report=drift_report,
        params={
            "window": window,
            "k": k,
            "min_pitches": min_pitches,
            "start_date": start_date,
            "end_date": end_date,
            "use_cache": use_cache,
        },
        warnings=warnings,
    )
