#!/usr/bin/env python
"""Clean raw swap parquet files with comprehensive validation and reporting.

This script performs data cleaning on Uniswap V3 swap data with the following features:
- Pydantic model validation for type safety and data integrity
- Duplicate detection and removal
- Comprehensive cleaning report generation
- Configurable output paths and validation strictness

Example:
    # Basic usage
    $ python scripts/clean_swaps.py --input data/raw/swaps.parquet

    # With custom output and validation
    $ python scripts/clean_swaps.py \\
        --input data/raw/swaps.parquet \\
        --output data/processed/swaps_clean.parquet \\
        --report data/processed/cleaning_report.md \\
        --strict

    # Filter by minimum USD value
    $ python scripts/clean_swaps.py --min-usd 100.0 --output data/processed/large_swaps.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from whalesentry.processing.clean_swaps import clean_swap_frame
from whalesentry.validation import CleaningReport, SwapValidator

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.clean_swaps")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser with comprehensive options."""
    parser = argparse.ArgumentParser(
        prog="clean_swaps",
        description="Clean and validate Uniswap V3 swap data with comprehensive reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/raw/swaps.parquet
  %(prog)s -i data/raw/swaps.parquet -o data/processed/clean.parquet
  %(prog)s --input swaps.parquet --min-usd 1000 --strict
  %(prog)s -i swaps.parquet --report cleaning_report.md --verbose
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        type=str,
        default="data/raw/swaps.parquet",
        help="Input parquet file path (default: data/raw/swaps.parquet)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        "-o",
        dest="output_path",
        type=str,
        default="data/processed/swaps_clean.parquet",
        help="Output parquet file path (default: data/processed/swaps_clean.parquet)",
    )
    output_group.add_argument(
        "--report",
        "-r",
        dest="report_path",
        type=str,
        default="data/processed/CLEANING_REPORT.md",
        help="Cleaning report markdown file path",
    )

    # Validation options
    validation_group = parser.add_argument_group("Validation Options")
    validation_group.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if any validation fails",
    )
    validation_group.add_argument(
        "--min-usd",
        type=float,
        default=0.0,
        help="Minimum USD value filter (default: 0.0)",
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


def save_report(report: CleaningReport, path: Path) -> None:
    """Save cleaning report to markdown file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        markdown = report.generate_markdown()
        path.write_text(markdown, encoding="utf-8")
        logger.info("Report saved to %s", path)
    except Exception as e:
        logger.error("Failed to save report: %s", e)


def filter_by_usd_value(df: pd.DataFrame, min_usd: float) -> pd.DataFrame:
    """Filter DataFrame by minimum USD value."""
    if min_usd <= 0:
        return df

    if "amount_usd" not in df.columns:
        logger.warning("amount_usd column not found, skipping USD filter")
        return df

    initial_count = len(df)
    df_filtered = df[df["amount_usd"].astype(float) >= min_usd]
    removed = initial_count - len(df_filtered)

    logger.info("USD filtering: removed %d rows below $%.2f", removed, min_usd)
    return df_filtered


def main() -> int:
    """Main entry point for the cleaning script."""
    parser = create_parser()
    parsed = parser.parse_args()

    # Setup logging
    setup_logging(parsed.verbose, parsed.quiet)

    # Resolve paths
    input_path = Path(parsed.input_path)
    output_path = Path(parsed.output_path)
    report_path = Path(parsed.report_path)

    logger.info("=" * 60)
    logger.info("Whale Sentry - Data Cleaning Pipeline")
    logger.info("=" * 60)

    # Validate input
    validate_input_file(input_path)

    # Load data
    df = load_data(input_path)
    before_count = len(df)

    # Initialize report
    report = CleaningReport()
    report.add_statistics(df)

    # Run Pydantic validation
    logger.info("Running Pydantic validation...")
    validator = SwapValidator()
    validation_result = validator.validate_dataframe(df)
    report.add_validation_result(validation_result)

    logger.info(
        "Validation complete: %d/%d valid (%.1f%%)",
        validation_result.valid_records,
        validation_result.total_records,
        validation_result.success_rate,
    )

    if validation_result.invalid_records > 0:
        logger.warning("Found %d invalid records", validation_result.invalid_records)
        for error in validation_result.validation_errors[:3]:
            logger.warning("  - %s", error)

    # Check for missing columns
    missing = validator.check_schema(df)
    if missing:
        logger.warning("Missing columns: %s", ", ".join(missing))

    # Run cleaning pipeline
    logger.info("Running cleaning pipeline...")
    cleaned = clean_swap_frame(df)

    # Apply USD value filter if specified
    if parsed.min_usd > 0:
        cleaned = filter_by_usd_value(cleaned, parsed.min_usd)

    after_count = len(cleaned)
    removed = before_count - after_count

    # Save cleaned data
    save_data(cleaned, output_path)

    # Save report
    save_report(report, report_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Cleaning Summary:")
    logger.info("  Input rows:     %d", before_count)
    logger.info("  Output rows:    %d", after_count)
    pct_removed = (removed / before_count * 100) if before_count > 0 else 0
    logger.info("  Removed:        %d (%.1f%%)", removed, pct_removed)
    logger.info("  Validation:     %.1f%% passed", validation_result.success_rate)
    logger.info("  Output file:    %s", output_path)
    logger.info("  Report file:    %s", report_path)
    logger.info("=" * 60)

    # Exit with error if strict mode and validation failed
    if parsed.strict and validation_result.invalid_records > 0:
        logger.error("Strict mode: exiting due to validation failures")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
