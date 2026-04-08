"""
KMeans-based location hub clustering per pitcher + pitch type.

Design notes
------------
- Clusters are fit on a baseline sample (typically all loaded games for
  a pitcher/date range) and then kept FIXED for in-game analysis.
  This is intentional: we never refit on a single game because that
  produces noisy, incomparable centers.
- Clustering is always performed per pitcher + pitch type.
  Do NOT pass a mixed-type DataFrame to fit_cluster_model().
- cluster centers are called "location hubs" in comments and docs to
  avoid implying we know the catcher target or true pitcher intent.

Public API
----------
fit_cluster_model(df, pitcher_id, pitch_type, k, min_pitches)
    -> PitchTypeClusterModel | None

assign_clusters_to_game(df, game_pk, pitch_type, cluster_model)
    -> pd.DataFrame

select_k_elbow(df, pitcher_id, pitch_type, k_range, min_pitches)
    -> dict[int, float]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans  # type: ignore[import]

from src import config

logger = logging.getLogger(__name__)

# Column names written by this module
CLUSTER_LABEL_COL = "cluster_label"
MISS_DISTANCE_COL = "miss_distance"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PitchTypeClusterModel:
    """
    Container for a fitted KMeans model and associated metadata.

    The model represents empirical location hubs — where this pitcher
    typically throws a given pitch type — learned from a baseline sample.

    Attributes
    ----------
    pitcher_id : int
    pitch_type : str
        Pitch type code (uppercase, e.g. 'FF', 'SL').
    k : int
        Number of clusters that were fitted (may be < requested k
        if data was limited).
    model : KMeans
        The fitted sklearn KMeans instance.
    centers : np.ndarray
        Shape (k, 2). Cluster centers in [plate_x, plate_z] space.
        Columns 0 = horizontal, 1 = vertical.
    n_pitches_fit : int
        Number of pitches in the baseline sample used to fit this model.
    inertia : float
        KMeans within-cluster sum of squares for the fit.
    """
    pitcher_id: int
    pitch_type: str
    k: int
    model: KMeans
    centers: np.ndarray
    n_pitches_fit: int
    inertia: float
    # Cluster-to-color mapping for consistent visualization
    color_map: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.color_map:
            self.color_map = {
                i: config.CLUSTER_COLORS[i % len(config.CLUSTER_COLORS)]
                for i in range(self.k)
            }


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_cluster_model(
    df: pd.DataFrame,
    pitcher_id: int,
    pitch_type: str,
    k: int = config.K_DEFAULT,
    min_pitches: int = config.MIN_PITCHES_DEFAULT,
    random_state: int = 42,
) -> PitchTypeClusterModel | None:
    """
    Fit a KMeans location-hub model on plate_x/plate_z for one pitcher+pitch_type.

    Filters df to the given pitcher and pitch type, then fits KMeans(k) on
    the [plate_x, plate_z] coordinates. Returns None gracefully if fewer
    than min_pitches rows remain after filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame — may span multiple games (this is the baseline).
        Must NOT be pre-filtered to a single game; the more data the better
        for stable cluster centers.
    pitcher_id : int
        MLBAM pitcher ID (used to filter df[config.PITCHER_COL]).
    pitch_type : str
        Uppercase pitch type code (e.g., 'FF').
    k : int, default 4
        Requested number of clusters.
    min_pitches : int, default 15
        Minimum pitches required. Returns None if fewer are available.
    random_state : int, default 42
        For KMeans reproducibility.

    Returns
    -------
    PitchTypeClusterModel or None
        None indicates insufficient data for this pitch type.

    Notes
    -----
    n_init is set to 10 (sklearn default as of 1.2+). This gives good
    cluster stability without excessive compute time.
    """
    # Filter to pitcher
    if config.PITCHER_COL in df.columns:
        subset = df[df[config.PITCHER_COL] == pitcher_id].copy()
    else:
        logger.warning(
            "Column '%s' not in DataFrame — assuming entire df belongs to pitcher %d",
            config.PITCHER_COL, pitcher_id,
        )
        subset = df.copy()

    # Filter to pitch type
    subset = subset[subset[config.PITCH_TYPE_COL] == pitch_type]

    # Drop any remaining nulls in location cols
    subset = subset.dropna(subset=[config.PLATE_X_COL, config.PLATE_Z_COL])

    n = len(subset)
    if n < min_pitches:
        logger.info(
            "Skipping cluster fit for pitcher=%d pitch_type=%s: "
            "only %d pitches (min_pitches=%d)",
            pitcher_id, pitch_type, n, min_pitches,
        )
        return None

    X = subset[[config.PLATE_X_COL, config.PLATE_Z_COL]].to_numpy(dtype=float)

    # Cap k at n // 3 to avoid degenerate clusters
    effective_k = min(k, max(1, n // 3))
    if effective_k < k:
        logger.warning(
            "Reducing k from %d to %d for pitcher=%d pitch_type=%s (only %d pitches)",
            k, effective_k, pitcher_id, pitch_type, n,
        )

    km = KMeans(n_clusters=effective_k, random_state=random_state, n_init=10)
    km.fit(X)

    return PitchTypeClusterModel(
        pitcher_id=pitcher_id,
        pitch_type=pitch_type,
        k=effective_k,
        model=km,
        centers=km.cluster_centers_,
        n_pitches_fit=n,
        inertia=float(km.inertia_),
    )


# ---------------------------------------------------------------------------
# Assign clusters to a game
# ---------------------------------------------------------------------------

def assign_clusters_to_game(
    df: pd.DataFrame,
    game_pk: int,
    pitch_type: str,
    cluster_model: PitchTypeClusterModel,
) -> pd.DataFrame:
    """
    Assign cluster labels and miss distances to pitches in a specific game.

    Uses the pre-fitted baseline model — never re-fits. The returned
    DataFrame is a filtered subset of df for (game_pk, pitch_type).

    Columns added: ``cluster_label`` (int), ``miss_distance`` (float)

    miss_distance = Euclidean distance from (plate_x, plate_z) to the
    centroid of the assigned cluster. This is the raw spatial miss for
    each pitch relative to its location hub.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned full DataFrame (covering multiple games).
    game_pk : int
    pitch_type : str
    cluster_model : PitchTypeClusterModel

    Returns
    -------
    pd.DataFrame
        Subset for game_pk + pitch_type, sorted by game_pitch_index,
        with cluster_label and miss_distance appended.
        Returns empty DataFrame with correct schema if no pitches found.
    """
    # Filter
    mask = (df[config.PITCH_TYPE_COL] == pitch_type)
    if config.GAME_PK_COL in df.columns:
        mask = mask & (df[config.GAME_PK_COL] == game_pk)

    subset = df[mask].copy()

    if subset.empty:
        logger.warning(
            "No pitches found for game_pk=%s pitch_type=%s", game_pk, pitch_type
        )
        subset[CLUSTER_LABEL_COL] = pd.array([], dtype=int)
        subset[MISS_DISTANCE_COL] = pd.array([], dtype=float)
        return subset

    X = subset[[config.PLATE_X_COL, config.PLATE_Z_COL]].to_numpy(dtype=float)

    labels = cluster_model.model.predict(X)
    subset[CLUSTER_LABEL_COL] = labels

    # Compute miss distance for each pitch
    centers = cluster_model.centers  # shape (k, 2)
    assigned_centers = centers[labels]  # shape (n, 2)
    diffs = X - assigned_centers
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    subset[MISS_DISTANCE_COL] = distances

    # Sort by within-game index if available
    from src.features.pitch_index import GAME_PITCH_INDEX_COL
    if GAME_PITCH_INDEX_COL in subset.columns:
        subset = subset.sort_values(GAME_PITCH_INDEX_COL).reset_index(drop=True)

    return subset


# ---------------------------------------------------------------------------
# Elbow method helper (notebook exploration only)
# ---------------------------------------------------------------------------

def select_k_elbow(
    df: pd.DataFrame,
    pitcher_id: int,
    pitch_type: str,
    k_range: range = range(2, 7),
    min_pitches: int = config.MIN_PITCHES_DEFAULT,
) -> dict[int, float]:
    """
    Compute KMeans inertia for each k in k_range to aid elbow selection.

    Intended for use in notebook 03 (cluster modeling exploration).
    Not called by the service layer — k is treated as a parameter there.

    Parameters
    ----------
    df : pd.DataFrame
    pitcher_id : int
    pitch_type : str
    k_range : range
        K values to evaluate. Default range(2, 7).
    min_pitches : int

    Returns
    -------
    dict[int, float]
        Mapping of k -> inertia. Empty dict if insufficient data.
    """
    if config.PITCHER_COL in df.columns:
        subset = df[df[config.PITCHER_COL] == pitcher_id]
    else:
        subset = df

    subset = subset[subset[config.PITCH_TYPE_COL] == pitch_type].dropna(
        subset=[config.PLATE_X_COL, config.PLATE_Z_COL]
    )

    if len(subset) < min_pitches:
        return {}

    X = subset[[config.PLATE_X_COL, config.PLATE_Z_COL]].to_numpy(dtype=float)
    results: dict[int, float] = {}
    for k in k_range:
        if k > len(X):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        results[k] = float(km.inertia_)

    return results
