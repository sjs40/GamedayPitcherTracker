"""
Cluster center drift analysis — comparing baseline location hubs to actual
pitch locations in early/mid/late game buckets.

Drift is defined as: for a given cluster, how far does the mean actual
pitch location in each game bucket stray from the baseline cluster center?

Design notes
------------
- We do NOT recompute clusters per window. The baseline centers are fixed.
- Drift is directional: we report delta_x and delta_z (not just magnitude)
  so visualizations can show arm-side vs. glove-side, up vs. down.
- Buckets with fewer than MIN_PITCHES_PER_BUCKET pitches yield None
  rather than noisy estimates.

Public API
----------
compute_bucket_means(df)                    -> dict[str, np.ndarray | None]
compute_cluster_drift(df, game_pk, ...)     -> GameDriftReport
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.features.clustering import (
    CLUSTER_LABEL_COL,
    PitchTypeClusterModel,
    assign_clusters_to_game,
)
from src.features.pitch_index import GAME_BUCKET_COL, GAME_PITCH_INDEX_COL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClusterDriftSummary:
    """
    Drift statistics for a single cluster in a single game.

    Drift is measured as the displacement of the mean actual pitch
    location in each bucket from the baseline cluster center (location hub).

    Attributes
    ----------
    cluster_id : int
    baseline_center : np.ndarray
        Shape (2,): [plate_x, plate_z] of the fitted cluster center.
    early_mean : np.ndarray or None
        Mean [plate_x, plate_z] in the early bucket. None if < MIN_PITCHES_PER_BUCKET.
    mid_mean : np.ndarray or None
    late_mean : np.ndarray or None
    early_delta : np.ndarray or None
        Vector from baseline_center to early_mean: [delta_x, delta_z].
        Positive delta_x = arm-side (for RHP) or left on the chart.
    mid_delta : np.ndarray or None
    late_delta : np.ndarray or None
    early_drift_magnitude : float or None
        Euclidean magnitude of early_delta.
    mid_drift_magnitude : float or None
    late_drift_magnitude : float or None
    early_n : int    Pitch count in early bucket for this cluster.
    mid_n : int
    late_n : int
    """
    cluster_id: int
    baseline_center: np.ndarray

    early_mean: np.ndarray | None = None
    mid_mean: np.ndarray | None = None
    late_mean: np.ndarray | None = None

    early_delta: np.ndarray | None = None
    mid_delta: np.ndarray | None = None
    late_delta: np.ndarray | None = None

    early_drift_magnitude: float | None = None
    mid_drift_magnitude: float | None = None
    late_drift_magnitude: float | None = None

    early_n: int = 0
    mid_n: int = 0
    late_n: int = 0


@dataclass
class GameDriftReport:
    """
    Full drift report for one pitcher, one game, one pitch type.

    Attributes
    ----------
    pitcher_id : int
    game_pk : int
    pitch_type : str
    cluster_summaries : list[ClusterDriftSummary]
        One entry per cluster. Empty list if no data.
    overall_drift_magnitude : float
        Mean of all non-None drift magnitudes across clusters and buckets.
        NaN if no valid drift magnitudes were computed.
    n_pitches : int
        Total pitches in this game/pitch_type used for drift analysis.
    """
    pitcher_id: int
    game_pk: int
    pitch_type: str
    cluster_summaries: list[ClusterDriftSummary] = field(default_factory=list)
    overall_drift_magnitude: float = float("nan")
    n_pitches: int = 0


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def compute_bucket_means(
    df: pd.DataFrame,
    min_pitches: int = config.MIN_PITCHES_PER_BUCKET,
) -> dict[str, np.ndarray | None]:
    """
    Compute mean [plate_x, plate_z] for each game bucket.

    Buckets with fewer than min_pitches pitches return None rather than
    a potentially noisy estimate.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'game_bucket', 'plate_x', 'plate_z' columns.
        Typically a cluster-filtered subset.
    min_pitches : int
        Minimum pitches to compute a bucket mean. Default: 3.

    Returns
    -------
    dict with keys from config.BUCKET_LABELS (e.g., 'early', 'mid', 'late').
    Each value is np.ndarray shape (2,) [plate_x, plate_z], or None.
    """
    result: dict[str, np.ndarray | None] = {}

    for bucket in config.BUCKET_LABELS:
        bucket_df = df[df[GAME_BUCKET_COL] == bucket]
        n = len(bucket_df)
        if n < min_pitches:
            result[bucket] = None
        else:
            x_mean = bucket_df[config.PLATE_X_COL].mean()
            z_mean = bucket_df[config.PLATE_Z_COL].mean()
            result[bucket] = np.array([x_mean, z_mean])

    return result


# ---------------------------------------------------------------------------
# Main drift computation
# ---------------------------------------------------------------------------

def compute_cluster_drift(
    df: pd.DataFrame,
    game_pk: int,
    pitch_type: str,
    cluster_model: PitchTypeClusterModel,
) -> GameDriftReport:
    """
    Compute per-cluster drift (baseline center vs. bucket means) for one game.

    The input df should already have 'cluster_label' and 'game_bucket'
    columns (i.e., assign_clusters_to_game and add_pitch_count_buckets
    have been called). If not, this function calls assign_clusters_to_game
    internally and warns.

    Steps:
        1. Filter df to game_pk + pitch_type.
        2. Check for required columns; call assign_clusters_to_game if needed.
        3. For each cluster, compute bucket means and drift vs. baseline center.
        4. Assemble and return GameDriftReport.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, feature-engineered DataFrame. Should have cluster_label
        and game_bucket columns.
    game_pk : int
    pitch_type : str
    cluster_model : PitchTypeClusterModel
        Provides the baseline cluster centers to compare against.

    Returns
    -------
    GameDriftReport
    """
    # Filter to game + pitch type
    mask = df[config.PITCH_TYPE_COL] == pitch_type
    if config.GAME_PK_COL in df.columns:
        mask = mask & (df[config.GAME_PK_COL] == game_pk)
    game_df = df[mask].copy()

    if game_df.empty:
        logger.warning(
            "No pitches for game_pk=%s pitch_type=%s — returning empty drift report",
            game_pk, pitch_type,
        )
        return GameDriftReport(
            pitcher_id=cluster_model.pitcher_id,
            game_pk=game_pk,
            pitch_type=pitch_type,
        )

    # Ensure cluster_label exists
    if CLUSTER_LABEL_COL not in game_df.columns:
        logger.warning(
            "cluster_label not found in df — calling assign_clusters_to_game"
        )
        game_df = assign_clusters_to_game(df, game_pk, pitch_type, cluster_model)

    if GAME_BUCKET_COL not in game_df.columns:
        raise ValueError(
            "game_bucket column not found. "
            "Call add_pitch_count_buckets() before compute_cluster_drift()."
        )

    summaries: list[ClusterDriftSummary] = []

    for cluster_id in range(cluster_model.k):
        cluster_df = game_df[game_df[CLUSTER_LABEL_COL] == cluster_id]
        baseline_center = cluster_model.centers[cluster_id]

        n_counts = {
            b: int((cluster_df[GAME_BUCKET_COL] == b).sum())
            for b in config.BUCKET_LABELS
        }

        bucket_means = compute_bucket_means(cluster_df)

        def _delta(mean: np.ndarray | None) -> np.ndarray | None:
            if mean is None:
                return None
            return mean - baseline_center

        def _magnitude(delta: np.ndarray | None) -> float | None:
            if delta is None:
                return None
            return float(np.sqrt((delta ** 2).sum()))

        early_delta = _delta(bucket_means.get("early"))
        mid_delta = _delta(bucket_means.get("mid"))
        late_delta = _delta(bucket_means.get("late"))

        summaries.append(
            ClusterDriftSummary(
                cluster_id=cluster_id,
                baseline_center=baseline_center,
                early_mean=bucket_means.get("early"),
                mid_mean=bucket_means.get("mid"),
                late_mean=bucket_means.get("late"),
                early_delta=early_delta,
                mid_delta=mid_delta,
                late_delta=late_delta,
                early_drift_magnitude=_magnitude(early_delta),
                mid_drift_magnitude=_magnitude(mid_delta),
                late_drift_magnitude=_magnitude(late_delta),
                early_n=n_counts.get("early", 0),
                mid_n=n_counts.get("mid", 0),
                late_n=n_counts.get("late", 0),
            )
        )

    # Compute overall drift magnitude (mean of all valid bucket magnitudes)
    magnitudes = []
    for s in summaries:
        for mag in [s.early_drift_magnitude, s.mid_drift_magnitude, s.late_drift_magnitude]:
            if mag is not None:
                magnitudes.append(mag)

    overall = float(np.mean(magnitudes)) if magnitudes else float("nan")

    return GameDriftReport(
        pitcher_id=cluster_model.pitcher_id,
        game_pk=game_pk,
        pitch_type=pitch_type,
        cluster_summaries=summaries,
        overall_drift_magnitude=overall,
        n_pitches=len(game_df),
    )
