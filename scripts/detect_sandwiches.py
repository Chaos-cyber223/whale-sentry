#!/usr/bin/env python
"""Detect sandwich attacks in Uniswap V3 swap data.

This script performs sandwich attack detection on cleaned swap data with the
following features:
- Configurable time window for detection
- Minimum USD value filtering
- Confidence-based filtering
- Comprehensive reporting with statistics
- Results export to parquet format

Example:
    # Basic usage
    $ python scripts/detect_sandwiches.py --input data/processed/swaps_clean.parquet

    # With custom parameters
    $ python scripts/detect_sandwiches.py \
        --input data/processed/swaps_clean.parquet \
        --output data/results/sandwich_candidates.parquet \
        --time-window 30 \
        --min-usd 500 \
        --min-confidence 0.8

    # Export only high-confidence results
    $ python scripts/detect_sandwiches.py \
        --input data/processed/swaps_clean.parquet \
        --min-confidence 0.9 \
        --report data/results/detection_report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

from whalesentry.detection import candidates_to_dataframe, detect_sandwich_attacks

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.detect_sandwiches")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with comprehensive options."""
    parser = argparse.ArgumentParser(
        prog="detect_sandwiches",
        description="Detect sandwich attacks in Uniswap V3 swap data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/processed/swaps_clean.parquet
  %(prog)s -i swaps_clean.parquet -o sandwich_candidates.parquet
  %(prog)s --input swaps.parquet --time-window 30 --min-usd 1000
  %(prog)s -i swaps.parquet --min-confidence 0.9 --verbose
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        type=str,
        default="data/processed/swaps_clean.parquet",
        help="Input parquet file path (default: data/processed/swaps_clean.parquet)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        "-o",
        dest="output_path",
        type=str,
        default="data/results/sandwich_candidates.parquet",
        help="Output parquet file path (default: data/results/sandwich_candidates.parquet)",
    )
    output_group.add_argument(
        "--report",
        "-r",
        dest="report_path",
        type=str,
        default="data/results/DETECTION_REPORT.md",
        help="Detection report markdown file path",
    )

    # Detection options
    detection_group = parser.add_argument_group("Detection Options")
    detection_group.add_argument(
        "--time-window",
        "-t",
        type=int,
        default=60,
        help="Time window in seconds for sandwich detection (default: 60)",
    )
    detection_group.add_argument(
        "--min-usd",
        type=float,
        default=100.0,
        help="Minimum USD value for transactions to consider (default: 100.0)",
    )
    detection_group.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence score for results (0.0-1.0, default: 0.0)",
    )
    detection_group.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Minimum amount similarity ratio (0.0-1.0, default: 0.5)",
    )

    # Logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    logging_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    return parser


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level based on arguments."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    elif quiet:
        logging.getLogger().setLevel(logging.WARNING)


def validate_input_file(path: Path) -> None:
    """Validate that input file exists and is readable."""
    if not path.exists():
        logger.error("Input file not found: %s", path)
        raise SystemExit(1)

    if not path.is_file():
        logger.error("Input path is not a file: %s", path)
        raise SystemExit(1)

    logger.info("Input file validated: %s", path)


def load_data(path: Path) -> pd.DataFrame:
    """Load parquet file with error handling."""
    try:
        logger.info("Loading data from %s", path)
        df = pd.read_parquet(path)
        logger.info("Loaded %d rows with %d columns", len(df), len(df.columns))
        return df
    except Exception as e:
        logger.error("Failed to load parquet file: %s", e)
        raise SystemExit(1) from e


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet with directory creation."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving %d rows to %s", len(df), path)
        df.to_parquet(path, index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error("Failed to save parquet file: %s", e)
        raise SystemExit(1) from e


def save_report(report: str, path: Path) -> None:
    """Save detection report to markdown file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        logger.info("Report saved to %s", path)
    except Exception as e:
        logger.error("Failed to save report: %s", e)


def generate_detection_report(
    result,
    min_confidence: float,
    elapsed_time: float,
) -> str:
    """Generate markdown report from detection results.

    Args:
        result: DetectionResult from detect_sandwich_attacks.
        min_confidence: Minimum confidence filter applied.
        elapsed_time: Detection elapsed time in seconds.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# Sandwich Attack Detection Report",
        "",
        "## Detection Configuration",
        "",
        f"- **Time Window**: {result.time_window_seconds} seconds",
        f"- **Minimum Confidence Filter**: {min_confidence:.2f}",
        f"- **Processing Time**: {elapsed_time:.2f} seconds",
        "",
        "## Summary Statistics",
        "",
        f"- **Total Swaps Analyzed**: {result.total_swaps_analyzed:,}",
        f"- **Pools Analyzed**: {result.pools_analyzed:,}",
        f"- **Total Candidates Detected**: {result.total_candidates:,}",
        f"- **Unique Attackers**: {len(result.unique_attackers):,}",
        f"- **High Confidence Candidates (≥0.8)**: {len(result.high_confidence_candidates):,}",
        "",
    ]

    if result.detection_errors:
        lines.extend([
            "## Errors",
            "",
        ])
        for error in result.detection_errors:
            lines.append(f"- {error}")
        lines.append("")

    if result.candidates:
        lines.extend([
            "## Top Candidates by Confidence",
            "",
            "| Rank | Attacker | Pool | Confidence | Profit (USD) | Victim Tx |",
            "|------|----------|------|------------|--------------|-----------|",
        ])

        # Sort by confidence
        sorted_candidates = sorted(
            result.candidates,
            key=lambda c: c.confidence_score,
            reverse=True,
        )[:20]  # Top 20

        for rank, candidate in enumerate(sorted_candidates, 1):
            attacker_short = f"{candidate.attacker[:10]}...{candidate.attacker[-8:]}"
            pool_short = f"{candidate.pool[:10]}...{candidate.pool[-8:]}"
            victim_short = f"{candidate.victim_tx[:10]}...{candidate.victim_tx[-8:]}"
            lines.append(
                f"| {rank} | {attacker_short} | {pool_short} | "
                f"{candidate.confidence_score:.2f} | "
                f"{candidate.profit_estimate_usd} | {victim_short} |"
            )
        lines.append("")

        # Statistics by pool
        lines.extend([
            "## Detection by Pool",
            "",
            "| Pool | Candidates | Avg Confidence | Total Profit (USD) |",
            "|------|------------|----------------|-------------------|",
        ])

        pool_stats: dict[str, dict] = {}
        for c in result.candidates:
            pool = c.pool
            if pool not in pool_stats:
                pool_stats[pool] = {"count": 0, "confidence_sum": 0.0, "profit_sum": Decimal("0")}
            pool_stats[pool]["count"] += 1
            pool_stats[pool]["confidence_sum"] += c.confidence_score
            try:
                pool_stats[pool]["profit_sum"] += Decimal(c.profit_estimate_usd)
            except Exception:
                pass

        for pool, stats in sorted(pool_stats.items(), key=lambda x: x[1]["count"], reverse=True):
            pool_short = f"{pool[:10]}...{pool[-8:]}"
            avg_conf = stats["confidence_sum"] / stats["count"] if stats["count"] > 0 else 0
            lines.append(
                f"| {pool_short} | {stats['count']} | {avg_conf:.2f} | {stats['profit_sum']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def print_summary(result, output_path: Path, report_path: Path, elapsed_time: float) -> None:
    """Print detection summary to console."""
    print("\n" + "=" * 60)
    print("Sandwich Detection Summary")
    print("=" * 60)
    print(f"Swaps analyzed:     {result.total_swaps_analyzed:,}")
    print(f"Pools analyzed:     {result.pools_analyzed:,}")
    print(f"Candidates found:   {result.total_candidates:,}")
    print(f"Unique attackers:   {len(result.unique_attackers):,}")
    print(f"High confidence:    {len(result.high_confidence_candidates):,}")
    print(f"Processing time:    {elapsed_time:.2f}s")
    print("-" * 60)
    print(f"Results file:       {output_path}")
    print(f"Report file:        {report_path}")
    print("=" * 60 + "\n")


def main() -> int:
    """Main entry point for the detection script."""
    import time

    parser = create_parser()
    parsed = parser.parse_args()

    # Setup logging
    setup_logging(parsed.verbose, parsed.quiet)

    # Resolve paths
    input_path = Path(parsed.input_path)
    output_path = Path(parsed.output_path)
    report_path = Path(parsed.report_path)

    logger.info("=" * 60)
    logger.info("Whale Sentry - Sandwich Attack Detection")
    logger.info("=" * 60)

    # Validate input
    validate_input_file(input_path)

    # Load data
    df = load_data(input_path)

    if df.empty:
        logger.warning("Input file is empty, no data to process")
        return 0

    # Run detection
    logger.info("Running sandwich detection...")
    logger.info("  Time window: %d seconds", parsed.time_window)
    logger.info("  Min USD: $%.2f", parsed.min_usd)
    logger.info("  Similarity threshold: %.2f", parsed.similarity_threshold)

    start_time = time.time()

    result = detect_sandwich_attacks(
        df,
        time_window_seconds=parsed.time_window,
        min_usd_value=Decimal(str(parsed.min_usd)),
        amount_similarity_threshold=parsed.similarity_threshold,
    )

    elapsed_time = time.time() - start_time

    logger.info("Detection complete in %.2f seconds", elapsed_time)
    logger.info("Found %d candidates", result.total_candidates)

    # Filter by confidence if specified
    candidates = result.candidates
    if parsed.min_confidence > 0:
        candidates = tuple(c for c in candidates if c.confidence_score >= parsed.min_confidence)
        logger.info(
            "After confidence filter (≥%.2f): %d candidates",
            parsed.min_confidence,
            len(candidates),
        )

    # Convert to DataFrame and save
    if candidates:
        candidates_df = candidates_to_dataframe(list(candidates))
        save_data(candidates_df, output_path)
    else:
        # Create empty DataFrame with correct schema
        empty_df = candidates_to_dataframe([])
        save_data(empty_df, output_path)
        logger.info("No candidates found, saved empty results file")

    # Generate and save report
    report = generate_detection_report(result, parsed.min_confidence, elapsed_time)
    save_report(report, report_path)

    # Print summary
    if not parsed.quiet:
        print_summary(result, output_path, report_path, elapsed_time)

    return 0


if __name__ == "__main__":
    sys.exit(main())
