#!/usr/bin/env python
"""Fetch Uniswap V3 swaps for a pool and persist to parquet."""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from whalesentry.ingest.subgraph_client import SubgraphClient
from whalesentry.ingest.uniswap_v3_swaps import fetch_swaps_for_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("scripts.fetch_swaps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Uniswap V3 swaps for a pool")
    parser.add_argument("--pool", required=True, help="Pool address (checksum or lowercase)")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours")
    parser.add_argument(
        "--out",
        default="data/raw/swaps.parquet",
        help="Output parquet path",
    )
    return parser.parse_args()


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = os.getenv("SUBGRAPH_API_KEY")
    if api_key:
        key = api_key.strip()
        if key.lower().startswith("bearer "):
            headers["Authorization"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"
    return headers


def main() -> None:
    load_dotenv()
    args = parse_args()

    endpoint = os.getenv("SUBGRAPH_ENDPOINT")
    if not endpoint:
        raise SystemExit("SUBGRAPH_ENDPOINT env var required. See configs/.env.example")

    end_ts = int(datetime.now(tz=timezone.utc).timestamp())
    start_ts = end_ts - int(timedelta(hours=args.hours).total_seconds())

    client = SubgraphClient(endpoint, headers=_build_headers())
    df = fetch_swaps_for_pool(client, args.pool, start_ts, end_ts)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(
        "Fetched %s swaps for pool %s (%s -> %s) -> %s",
        len(df),
        args.pool,
        start_ts,
        end_ts,
        output_path,
    )


if __name__ == "__main__":
    main()
