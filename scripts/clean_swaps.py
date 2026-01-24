#!/usr/bin/env python
"""Clean raw swap parquet files and write normalized outputs."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from whalesentry.processing.clean_swaps import clean_swap_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("scripts.clean_swaps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw Uniswap swaps data")
    parser.add_argument("--in", dest="input", default="data/raw/swaps.parquet", help="Input parquet path")
    parser.add_argument(
        "--out",
        dest="output",
        default="data/clean/swaps_clean.parquet",
        help="Output parquet path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    before = len(df)
    cleaned = clean_swap_frame(df)
    after = len(cleaned)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)

    logger.info(
        "Cleaned swaps -> rows before=%s after=%s output=%s",
        before,
        after,
        output_path,
    )


if __name__ == "__main__":
    main()
