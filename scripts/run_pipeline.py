#!/usr/bin/env python3
"""
run_pipeline.py — CLI entry point for pitcher command drift analysis.

Usage examples
--------------
# Full analysis by name:
python scripts/run_pipeline.py \\
    --pitcher "Gerrit Cole" \\
    --start-date 2024-04-01 \\
    --end-date 2024-09-30 \\
    --game 746175 \\
    --pitch-type FF

# Full analysis by MLBAM ID:
python scripts/run_pipeline.py \\
    --pitcher 543037 \\
    --start-date 2024-04-01 \\
    --end-date 2024-09-30 \\
    --game 746175 \\
    --pitch-type FF \\
    --window 7 \\
    --k 4 \\
    --output-dir data/outputs/cole_2024

# List available games only (no game analysis):
python scripts/run_pipeline.py \\
    --pitcher "Gerrit Cole" \\
    --start-date 2024-04-01 \\
    --end-date 2024-09-30 \\
    --list-games

Outputs (saved to --output-dir):
    dashboard.png       — multi-panel summary chart
    rolling_spread.png  — rolling spread chart
    rolling_miss.png    — rolling miss distance chart
    cluster_map.png     — location hubs scatter
    miss_vectors.png    — miss direction arrows
    cluster_drift.png   — early/mid/late drift chart
    cluster_usage.png   — hub assignment by pitch
    rolling_metrics.csv — rolling spreads + miss distance table
    cluster_assignments.csv — per-pitch cluster and miss data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use

import matplotlib.pyplot as plt

from src import config
from src.services.pitcher_service import (
    build_game_report,
    get_available_games,
    get_available_pitch_types,
    get_pitcher_dataset,
)
from src.utils.io import save_csv
from src.visualization.dashboards import plot_game_dashboard
from src.visualization.rolling_plots import (
    plot_rolling_miss,
    plot_rolling_spreads,
    prep_rolling_miss,
    prep_rolling_spreads,
)
from src.visualization.zone_plots import (
    plot_cluster_drift,
    plot_cluster_map,
    plot_cluster_usage,
    plot_miss_vectors,
    prep_cluster_drift,
    prep_cluster_map,
    prep_cluster_usage,
    prep_miss_vectors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="MLB Pitcher Command Drift Analysis — Statcast-based",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--pitcher",
        type=str,
        help="Player name (e.g., 'Gerrit Cole') or MLBAM integer ID",
    )
    p.add_argument(
        "--start-date",
        type=str,
        dest="start_date",
        help="Data range start date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        type=str,
        dest="end_date",
        help="Data range end date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--game",
        type=int,
        dest="game_pk",
        help="game_pk (integer) for the target game",
    )
    p.add_argument(
        "--pitch-type",
        type=str,
        dest="pitch_type",
        help="Pitch type code to analyze (e.g., FF, SL, CH)",
    )
    p.add_argument(
        "--window",
        type=int,
        default=config.ROLLING_WINDOW_DEFAULT,
        help=f"Rolling window size in pitches (default: {config.ROLLING_WINDOW_DEFAULT})",
    )
    p.add_argument(
        "--k",
        type=int,
        default=config.K_DEFAULT,
        help=f"Number of KMeans location-hub clusters (default: {config.K_DEFAULT})",
    )
    p.add_argument(
        "--min-pitches",
        type=int,
        default=config.MIN_PITCHES_DEFAULT,
        dest="min_pitches",
        help=f"Min pitches per pitch type for clustering (default: {config.MIN_PITCHES_DEFAULT})",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(config.OUTPUTS_DIR),
        dest="output_dir",
        help="Directory for output files (default: data/outputs)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        dest="no_cache",
        help="Force re-fetch even if cached data exists",
    )
    p.add_argument(
        "--list-games",
        action="store_true",
        dest="list_games",
        help="Print available games and exit (no game analysis)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return p


def _prompt(prompt_text: str, default: str | None = None) -> str:
    """Interactive prompt with optional default."""
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt_text}{suffix}: ").strip()
    return value if value else (default or "")


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Interactively fill in any missing required arguments."""
    if not args.pitcher:
        args.pitcher = _prompt("Pitcher (name or MLBAM ID)")
    if not args.start_date:
        args.start_date = _prompt("Start date", default="2024-04-01")
    if not args.end_date:
        args.end_date = _prompt("End date", default="2024-09-30")
    return args


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------------------
    logger.info("Loading dataset for pitcher=%r (%s to %s)", args.pitcher, args.start_date, args.end_date)
    df = get_pitcher_dataset(
        pitcher_id=args.pitcher,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=not args.no_cache,
    )
    logger.info("Loaded %d pitches", len(df))

    # ------------------------------------------------------------------
    # Step 2: List games (if requested)
    # ------------------------------------------------------------------
    games = get_available_games(df)
    print(f"\nAvailable games ({len(games)}):")
    for g in games:
        print(f"  {g}")

    if args.list_games:
        return

    # ------------------------------------------------------------------
    # Step 3: Prompt for game and pitch type if missing
    # ------------------------------------------------------------------
    if not args.game_pk:
        game_pk_str = _prompt("game_pk (integer)")
        try:
            args.game_pk = int(game_pk_str)
        except ValueError:
            print("Invalid game_pk. Please enter an integer.")
            sys.exit(1)

    available_types = get_available_pitch_types(df, game_pk=args.game_pk)
    print(f"\nAvailable pitch types for game {args.game_pk}: {available_types}")

    if not args.pitch_type:
        args.pitch_type = _prompt("Pitch type", default=available_types[0] if available_types else "FF")

    # ------------------------------------------------------------------
    # Step 4: Build game report
    # ------------------------------------------------------------------
    logger.info(
        "Building report: game_pk=%d pitch_type=%s window=%d k=%d",
        args.game_pk, args.pitch_type, args.window, args.k,
    )

    report = build_game_report(
        pitcher_id=args.pitcher,
        game_pk=args.game_pk,
        pitch_type=args.pitch_type,
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        k=args.k,
        min_pitches=args.min_pitches,
        use_cache=not args.no_cache,
    )

    if report.warnings:
        for w in report.warnings:
            logger.warning("Report warning: %s", w)

    # ------------------------------------------------------------------
    # Step 5: Save data outputs
    # ------------------------------------------------------------------
    if not report.rolling_spreads.empty:
        save_csv(report.rolling_spreads, output_dir / "rolling_metrics.csv")
        logger.info("Saved rolling_metrics.csv")

    if not report.game_with_clusters.empty:
        save_csv(report.game_with_clusters, output_dir / "cluster_assignments.csv")
        logger.info("Saved cluster_assignments.csv")

    # ------------------------------------------------------------------
    # Step 6: Generate and save visualizations
    # ------------------------------------------------------------------
    title_prefix = (
        f"Pitcher {report.pitcher_id} | {report.pitch_type} | "
        f"Game {report.game_pk} ({report.game_date})"
    )

    # Dashboard
    dash_fig = plot_game_dashboard(report)
    _save_fig(dash_fig, output_dir, "dashboard.png")

    # Rolling spread
    rs_prepped = prep_rolling_spreads(report.rolling_spreads)
    rs_fig = plot_rolling_spreads(
        rs_prepped,
        title=f"{title_prefix} — Rolling Spread",
        window=args.window,
    )
    _save_fig(rs_fig, output_dir, "rolling_spread.png")

    # Rolling miss distance
    if not report.rolling_miss.empty:
        rm_prepped = prep_rolling_miss(report.rolling_miss)
        rm_fig = plot_rolling_miss(
            rm_prepped,
            title=f"{title_prefix} — Rolling Miss Distance",
            window=args.window,
        )
        _save_fig(rm_fig, output_dir, "rolling_miss.png")

    # Cluster map
    cm_prepped = prep_cluster_map(
        report.game_with_clusters, report.cluster_model.centers
    )
    cm_fig = plot_cluster_map(cm_prepped, title=f"{title_prefix} — Location Hubs")
    _save_fig(cm_fig, output_dir, "cluster_map.png")

    # Miss vectors
    mv_prepped = prep_miss_vectors(
        report.game_with_clusters, report.cluster_model.centers, max_pitches=80
    )
    mv_fig = plot_miss_vectors(mv_prepped, title=f"{title_prefix} — Miss Vectors")
    _save_fig(mv_fig, output_dir, "miss_vectors.png")

    # Cluster drift
    cd_prepped = prep_cluster_drift(report.drift_report)
    cd_fig = plot_cluster_drift(cd_prepped, title=f"{title_prefix} — Hub Drift")
    _save_fig(cd_fig, output_dir, "cluster_drift.png")

    # Cluster usage
    cu_prepped = prep_cluster_usage(report.game_with_clusters)
    cu_fig = plot_cluster_usage(cu_prepped, title=f"{title_prefix} — Hub Usage")
    _save_fig(cu_fig, output_dir, "cluster_usage.png")

    # ------------------------------------------------------------------
    # Step 7: Print summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Analysis complete: {title_prefix}")
    print(f"  Cluster model: {report.cluster_model.k} hubs fitted on "
          f"{report.cluster_model.n_pitches_fit} pitches")
    print(f"  Game pitches analyzed: {report.drift_report.n_pitches}")
    print(f"  Overall drift magnitude: {report.drift_report.overall_drift_magnitude:.3f} ft")
    print(f"  Outputs saved to: {output_dir.resolve()}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    args = _resolve_args(args)

    try:
        run(args)
    except ValueError as exc:
        logger.error("Analysis failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
