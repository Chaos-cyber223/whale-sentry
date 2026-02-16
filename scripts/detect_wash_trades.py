#!/usr/bin/env python
"""Detect wash trading in Uniswap V3 swap data.

This script performs wash trading detection on cleaned swap data with the
following features:
- Configurable time window for roundtrip detection (default: 300s)
- Minimum USD value filtering (default: $1000)
- Minimum roundtrip count threshold (default: 3)
- Confidence-based filtering
- Comprehensive reporting with statistics
- Results export to parquet format

Example:
    # Basic usage
    $ python scripts/detect_wash_trades.py --input data/processed/swaps_clean.parquet

    # With custom parameters
    $ python scripts/detect_wash_trades.py \
        --input data/processed/swaps_clean.parquet \
        --output data/results/wash_trade_candidates.parquet \
        --time-window 300 \
        --min-usd 1000 \
        --min-roundtrips 3

    # Export only high-confidence results
    $ python scripts/detect_wash_trades.py \
        --input data/processed/swaps_clean.parquet \
        --min-confidence 0.8 \
        --report data/results/wash_detection_report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from decimal import Decimal
from pathlib import Path

import pandas as pd

from whalesentry.detection.wash_trade import (
    DetectionResult,
    WashTradeCandidate,
    candidates_to_dataframe,
    detect_wash_trades,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.detect_wash_trades")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with comprehensive options."""
    parser = argparse.ArgumentParser(
        prog="detect_wash_trades",
        description="Detect wash trading in Uniswap V3 swap data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/processed/swaps_clean.parquet
  %(prog)s -i swaps_clean.parquet -o wash_trade_candidates.parquet
  %(prog)s --input swaps.parquet --time-window 300 --min-usd 1000
  %(prog)s -i swaps.parquet --min-confidence 0.8 --verbose
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        type=str,
        default="data/processed/swaps_clean.parquet",
        help="Input parquet file path (default: data/processed/swaps_clean.parquet)",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        "-o",
        dest="output_path",
        type=str,
        default="data/results/wash_trade_candidates.parquet",
        help="Output parquet file path (default: data/results/wash_trade_candidates.parquet)",
    )
    output_group.add_argument(
        "--report",
        "-r",
        dest="report_path",
        type=str,
        default="data/results/wash_detection_report.md",
        help="Detection report output path (default: data/results/wash_detection_report.md)",
    )

    detection_group = parser.add_argument_group("Detection Parameters")
    detection_group.add_argument(
        "--time-window",
        "-t",
        dest="time_window",
        type=int,
        default=300,
        help="Maximum time window for roundtrip pattern in seconds (default: 300)",
    )
    detection_group.add_argument(
        "--min-usd",
        "-u",
        dest="min_usd",
        type=float,
        default=1000.0,
        help="Minimum USD value per trade (default: 1000.0)",
    )
    detection_group.add_argument(
        "--min-roundtrips",
        "-n",
        dest="min_roundtrips",
        type=int,
        default=3,
        help="Minimum number of roundtrips to flag (default: 3)",
    )
    detection_group.add_argument(
        "--min-confidence",
        "-c",
        dest="min_confidence",
        type=float,
        default=0.0,
        help="Minimum confidence score to include in results (0.0-1.0, default: 0.0)",
    )

    display_group = parser.add_argument_group("Display Options")
    display_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output",
    )
    display_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress summary output (only errors)",
    )

    return parser


def validate_args(parsed: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if parsed.time_window <= 0:
        msg = f"Time window must be positive, got: {parsed.time_window}"
        raise ValueError(msg)

    if parsed.min_usd < 0:
        msg = f"Minimum USD must be non-negative, got: {parsed.min_usd}"
        raise ValueError(msg)

    if parsed.min_roundtrips < 1:
        msg = f"Minimum roundtrips must be at least 1, got: {parsed.min_roundtrips}"
        raise ValueError(msg)

    if not 0.0 <= parsed.min_confidence <= 1.0:
        msg = f"Confidence score must be between 0.0 and 1.0, got: {parsed.min_confidence}"
        raise ValueError(msg)

    input_path = Path(parsed.input_path)
    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        raise FileNotFoundError(msg)

    if not input_path.suffix == ".parquet":
        msg = f"Input file must be .parquet format, got: {input_path.suffix}"
        raise ValueError(msg)


def load_data(input_path: Path) -> pd.DataFrame:
    """Load swap data from parquet file."""
    logger.info("Loading swap data from: %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d swaps from %d pools", len(df), df["pool"].nunique())
    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save results to parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d candidates to: %s", len(df), output_path)


def save_report(report: str, report_path: Path) -> None:
    """Save detection report to markdown file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info("Saved detection report to: %s", report_path)


def generate_detection_report(
    result: DetectionResult,
    min_confidence: float,
    elapsed_time: float,
) -> str:
    """Generate markdown detection report."""
    filtered_candidates = [c for c in result.candidates if c.confidence_score >= min_confidence]

    report_lines = [
        "# Wash Trading Detection Report",
        "",
        "## Detection Summary",
        "",
        f"- **Total Swaps Analyzed**: {result.total_swaps_analyzed:,}",
        f"- **Pools Analyzed**: {result.pools_analyzed:,}",
        f"- **Time Window**: {result.time_window_seconds}s",
        f"- **Min USD Threshold**: ${result.min_usd_threshold:,.2f}",
        f"- **Min Roundtrips**: {result.min_roundtrips}",
        f"- **Total Candidates Found**: {result.total_candidates}",
        f"- **High-Confidence Candidates** (≥{min_confidence:.2f}): {len(filtered_candidates)}",
        f"- **Detection Time**: {elapsed_time:.2f}s",
        "",
        "## Detection Parameters",
        "",
        f"- **Time Window**: {result.time_window_seconds} seconds",
        f"- **Minimum USD per Trade**: ${result.min_usd_threshold:,.2f}",
        f"- **Minimum Roundtrips**: {result.min_roundtrips}",
        f"- **Confidence Filter**: ≥{min_confidence:.2f}",
        "",
    ]

    if filtered_candidates:
        report_lines.extend([
            "## Top Candidates",
            "",
            "| Trader | Pool | Roundtrips | Volume (USD) | Avg Time (s) | Confidence |",
            "|--------|------|------------|--------------|--------------|------------|",
        ])

        for candidate in sorted(filtered_candidates, key=lambda c: c.confidence_score, reverse=True)[:10]:
            trader_short = f"{candidate.trader[:6]}...{candidate.trader[-4:]}"
            pool_short = f"{candidate.pool[:6]}...{candidate.pool[-4:]}"
            volume = Decimal(candidate.total_volume_usd)
            report_lines.append(
                f"| {trader_short} | {pool_short} | {candidate.roundtrip_count} | "
                f"${volume:,.2f} | {candidate.avg_roundtrip_time:.1f} | {candidate.confidence_score:.3f} |"
            )

        report_lines.extend(["", ""])

    report_lines.extend([
        "## Statistics",
        "",
    ])

    if filtered_candidates:
        total_volume = sum(Decimal(c.total_volume_usd) for c in filtered_candidates)
        avg_confidence = sum(c.confidence_score for c in filtered_candidates) / len(filtered_candidates)
        avg_roundtrips = sum(c.roundtrip_count for c in filtered_candidates) / len(filtered_candidates)

        report_lines.extend([
            f"- **Total Suspicious Volume**: ${total_volume:,.2f}",
            f"- **Average Confidence**: {avg_confidence:.3f}",
            f"- **Average Roundtrips**: {avg_roundtrips:.1f}",
        ])
    else:
        report_lines.append("No candidates found matching the criteria.")

    return "\n".join(report_lines)


def print_summary(
    result: DetectionResult,
    output_path: Path,
    report_path: Path,
    elapsed_time: float,
) -> None:
    """Print detection summary to console."""
    print("\n" + "=" * 70)
    print("WASH TRADING DETECTION SUMMARY")
    print("=" * 70)
    print(f"\nAnalyzed {result.total_swaps_analyzed:,} swaps from {result.pools_analyzed:,} pools")
    print(f"Found {result.total_candidates} wash trading candidates")
    print(f"Detection completed in {elapsed_time:.2f}s")
    print(f"\nResults saved to:")
    print(f"  - Candidates: {output_path}")
    print(f"  - Report: {report_path}")
    print("=" * 70 + "\n")


def main() -> int:
    """Main entry point for wash trading detection script."""
    parser = create_parser()
    parsed = parser.parse_args()

    if parsed.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        validate_args(parsed)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Argument validation failed: %s", e)
        return 1

    input_path = Path(parsed.input_path)
    output_path = Path(parsed.output_path)
    report_path = Path(parsed.report_path)

    try:
        swaps_df = load_data(input_path)
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        return 1

    logger.info("Starting wash trading detection...")
    logger.info("  Time window: %ds", parsed.time_window)
    logger.info("  Min USD threshold: $%.2f", parsed.min_usd)
    logger.info("  Min roundtrips: %d", parsed.min_roundtrips)

    start_time = time.time()
    try:
        result = detect_wash_trades(
            swaps_df,
            time_window_seconds=parsed.time_window,
            min_usd_threshold=Decimal(str(parsed.min_usd)),
            min_roundtrips=parsed.min_roundtrips,
        )
    except Exception as e:
        logger.error("Detection failed: %s", e)
        return 1

    elapsed_time = time.time() - start_time
    logger.info("Detection completed in %.2fs", elapsed_time)
    logger.info("Found %d candidates", result.total_candidates)

    candidates = result.candidates
    if parsed.min_confidence > 0:
        candidates = tuple(c for c in candidates if c.confidence_score >= parsed.min_confidence)
        logger.info(
            "After confidence filter (≥%.2f): %d candidates",
            parsed.min_confidence,
            len(candidates),
        )

    if candidates:
        candidates_df = candidates_to_dataframe(list(candidates))
        save_data(candidates_df, output_path)
    else:
        empty_df = candidates_to_dataframe([])
        save_data(empty_df, output_path)
        logger.info("No candidates found, saved empty results file")

    report = generate_detection_report(result, parsed.min_confidence, elapsed_time)
    save_report(report, report_path)

    if not parsed.quiet:
        print_summary(result, output_path, report_path, elapsed_time)

    return 0


if __name__ == "__main__":
    sys.exit(main())
