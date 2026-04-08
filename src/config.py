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
    # Movement (inches, pitcher's perspective)
    "pfx_x",
    "pfx_z",
    # Release point (feet)
    "release_pos_x",
    "release_pos_z",
    "release_pos_y",
]

# Columns that must be non-null for a pitch to be usable in location analysis
LOCATION_REQUIRED_COLS: list[str] = ["plate_x", "plate_z", "pitch_type"]

# ---------------------------------------------------------------------------
# Game bucket labels (must match N_BUCKETS_DEFAULT)
# ---------------------------------------------------------------------------
BUCKET_LABELS: list[str] = ["early", "mid", "late"]

# ---------------------------------------------------------------------------
# Pitch outcome classification (derived from Statcast 'description' column)
# Used by features/performance.py to encode boolean indicators.
# ---------------------------------------------------------------------------
STRIKE_DESCRIPTIONS: frozenset = frozenset([
    "called_strike", "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
])
WHIFF_DESCRIPTIONS: frozenset = frozenset([
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
])
SWING_DESCRIPTIONS: frozenset = frozenset([
    "swinging_strike", "swinging_strike_blocked",
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
])
# Statcast zones 1-9 are inside the strike zone; 11+ are outside
IN_ZONE_VALUES: frozenset = frozenset(range(1, 10))

# ---------------------------------------------------------------------------
# Visualization defaults
# ---------------------------------------------------------------------------
CLUSTER_COLORS: list[str] = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
FIGURE_DPI: int = 120

# Pitch-type color palette — consistent across all charts
# Fallback to _default for any pitch type not listed here
PITCH_TYPE_COLORS: dict[str, str] = {
    "FF": "#e41a1c",   # four-seam fastball — red
    "SI": "#ff7f00",   # sinker — orange
    "FC": "#f781bf",   # cutter — pink
    "SL": "#377eb8",   # slider — blue
    "CU": "#984ea3",   # curveball — purple
    "KC": "#a65628",   # knuckle-curve — brown
    "CH": "#4daf4a",   # changeup — green
    "FS": "#999999",   # splitter — gray
    "ST": "#a6cee3",   # sweeper — light blue
    "SV": "#b2df8a",   # slurve — light green
    "CS": "#cab2d6",   # slow curve — lavender
    "_default": "#888888",
}

# Wider rolling window for pitch mix (individual type frequencies are sparse)
PITCH_MIX_WINDOW_DEFAULT: int = 15
