# GamedayPitcherTracker

MLB pitcher command drift analysis using Statcast pitch-level data.

## What it does

Measures how a pitcher's command changes over the course of a game by:
1. Learning empirical location hubs (KMeans clusters) per pitch type from historical data
2. Tracking rolling horizontal/vertical spread within each game
3. Measuring rolling miss distance from each pitch's assigned location hub
4. Computing whether hub centers drift (arm-side, glove-side, up, or down) from early to late game

**Important**: Location hubs are empirical cluster centers — where a pitcher *typically* throws a given pitch type. They are not direct observations of catcher targets or true intent.

## Architecture

Three-layer design, cleanly separated:

```
CLI / notebooks / future Streamlit app
           |
src/services/pitcher_service.py     ← service layer (single entry point)
           |
src/data/  src/features/            ← core analytics (pure Python)
src/visualization/                  ← visualization (prep_* / plot_* pattern)
```

## Quick start

```bash
pip install -r requirements.txt

# Run full analysis from CLI:
python scripts/run_pipeline.py \
    --pitcher "Gerrit Cole" \
    --start-date 2024-04-01 \
    --end-date 2024-09-30 \
    --game 746175 \
    --pitch-type FF

# List available games only:
python scripts/run_pipeline.py \
    --pitcher "Gerrit Cole" \
    --start-date 2024-04-01 \
    --end-date 2024-09-30 \
    --list-games
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_pull_and_inspection.ipynb` | Pull Statcast data, inspect raw structure |
| `02_clean_and_feature_engineering.ipynb` | Walk through cleaning pipeline and engineered features |
| `03_cluster_modeling.ipynb` | Explore KMeans clustering, elbow method, location hub maps |
| `04_command_metrics.ipynb` | Compute rolling spreads, miss distance, and cluster drift |
| `05_visualizations.ipynb` | Full end-to-end visualization for one pitcher/game/pitch type |

## Project structure

```
src/
  config.py                    — constants and defaults
  data/
    fetch.py                   — Statcast fetching + caching
    clean.py                   — cleaning pipeline
  features/
    pitch_index.py             — within-game index, early/mid/late buckets
    clustering.py              — KMeans location hubs (PitchTypeClusterModel)
    rolling.py                 — rolling std, rolling miss distance
    drift.py                   — cluster drift by game bucket (GameDriftReport)
  services/
    pitcher_service.py         — orchestration, GameReport output contract
  visualization/
    rolling_plots.py           — rolling spread + miss charts
    zone_plots.py              — strike zone scatter, miss vectors, drift
    dashboards.py              — multi-panel dashboard
  utils/
    io.py                      — parquet/CSV I/O, cache path
    validation.py              — input validators
scripts/
  run_pipeline.py              — CLI entry point
```

## Default parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Rolling window | 7 pitches | Trailing window for std/mean |
| Cluster count k | 4 | KMeans clusters per pitcher+pitch_type |
| Min pitches | 15 | Minimum pitches required to fit a cluster model |
| Game buckets | 3 (early/mid/late) | Equal thirds of game pitch sequence |

## Future app

The service layer (`src/services/pitcher_service.py`) is designed to be called
directly from a future Streamlit app with no refactoring:

```python
from src.services.pitcher_service import build_game_report
from src.visualization.dashboards import plot_game_dashboard

report = build_game_report(pitcher_id="Gerrit Cole", game_pk=..., pitch_type="FF", ...)
fig = plot_game_dashboard(report)
st.pyplot(fig)
```
