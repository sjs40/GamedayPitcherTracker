"""
Central configuration constants for GamedayPitcherTracker.

All tuneable defaults live here so callers can import a single module
and override at runtime without touching source files.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — used to resolve data paths relative to repo root
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
CACHE_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "data" / "outputs"

# ---------------------------------------------------------------------------
# Default analytical parameters
# ---------------------------------------------------------------------------

# Rolling window size (number of pitches).
# 7 is enough to smooth single-pitch noise while keeping early-game values
# visible. Can be overridden per function call.
ROLLING_WINDOW_DEFAULT: int = 7

# Number of KMeans clusters per pitcher+pitch_type.
# 4 covers the natural up/down × arm-side/glove-side quadrants of the zone.
K_DEFAULT: int = 4

# Minimum pitches required to fit a cluster model for a pitch type.
# Below this threshold fit_cluster_model() returns None.
MIN_PITCHES_DEFAULT: int = 15

# Minimum pitches per cluster per bucket for drift calculation.
# Below this, a bucket mean is reported as None rather than a noisy estimate.
MIN_PITCHES_PER_BUCKET: int = 3

# Number of equal game-time buckets (early / mid / late).
N_BUCKETS_DEFAULT: int = 3

# ---------------------------------------------------------------------------
# Approximate MLB average strike zone bounds (feet from home plate center).
# Used for consistent axis limits and zone rectangle across all plots.
# These are display defaults — do NOT use as intent proxies in analytics.
# ---------------------------------------------------------------------------
SZ_TOP: float = 3.5
SZ_BOT: float = 1.5
SZ_LEFT: float = -0.83
SZ_RIGHT: float = 0.83

# Axis limits for zone plots — slightly wider than zone for context
PLOT_X_MIN: float = -2.5
PLOT_X_MAX: float = 2.5
PLOT_Z_MIN: float = 0.5
PLOT_Z_MAX: float = 5.0

# ---------------------------------------------------------------------------
# Statcast column names
# ---------------------------------------------------------------------------
PLATE_X_COL: str = "plate_x"
PLATE_Z_COL: str = "plate_z"
PITCH_TYPE_COL: str = "pitch_type"
GAME_PK_COL: str = "game_pk"
PITCHER_COL: str = "pitcher"  # MLBAM ID column in pybaseball output

# Columns we want to retain from raw Statcast data.
# Any column not present in a given pull is silently skipped (see clean.py).
STATCAST_DESIRED_COLS: list[str] = [
    "pitcher",
    "player_name",
    "game_date",
    "game_pk",
    "inning",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "pitch_name",
    "batter",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "outs_when_up",
    "release_speed",
    "plate_x",
    "plate_z",
    "description",
    "events",
    "zone",
]

# Columns that must be non-null for a pitch to be usable in location analysis
LOCATION_REQUIRED_COLS: list[str] = ["plate_x", "plate_z", "pitch_type"]

# ---------------------------------------------------------------------------
# Game bucket labels (must match N_BUCKETS_DEFAULT)
# ---------------------------------------------------------------------------
BUCKET_LABELS: list[str] = ["early", "mid", "late"]

# ---------------------------------------------------------------------------
# Visualization defaults
# ---------------------------------------------------------------------------
CLUSTER_COLORS: list[str] = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
FIGURE_DPI: int = 120
