"""Utilities for fetching Uniswap V3 swap events from The Graph subgraphs."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from .subgraph_client import SubgraphClient

logger = logging.getLogger(__name__)

# Graph schemas sometimes rename nested fields (pool.id vs poolAddress);
# the normaliser below handles those differences best-effort.
SWAPS_QUERY = """
query ($pool: String!, $start: Int!, $end: Int!, $first: Int!, $skip: Int!) {
  swaps(
    first: $first
    skip: $skip
    where: { pool: $pool, timestamp_gte: $start, timestamp_lte: $end }
    orderBy: timestamp
    orderDirection: asc
  ) {
    id
    timestamp
    sender
    recipient
    amount0
    amount1
    amountUSD
    tick
    sqrtPriceX96
    pool {
      id
    }
    transaction {
      id
      blockNumber
    }
  }
}
"""


def fetch_swaps_for_pool(
    client: SubgraphClient,
    pool_address: str,
    start_ts: int,
    end_ts: int,
    page_size: int = 1000,
) -> pd.DataFrame:
    """Return swaps for a pool between two timestamps (inclusive)."""
    if start_ts > end_ts:
        raise ValueError("start_ts must be <= end_ts")

    normalized_pool = pool_address.lower()
    records: List[Dict[str, Any]] = []
    skip = 0

    while True:
        variables = {
            "pool": normalized_pool,
            "start": start_ts,
            "end": end_ts,
            "first": page_size,
            "skip": skip,
        }
        logger.info(
            "Fetching swaps for pool %s (%s -> %s) skip=%s",
            normalized_pool,
            start_ts,
            end_ts,
            skip,
        )
        data = client.query(SWAPS_QUERY, variables)
        swaps = data.get("swaps", [])
        if not swaps:
            break

        records.extend(_normalise_swap(swap) for swap in swaps)

        if len(swaps) < page_size:
            break
        skip += page_size

    columns = [
        "timestamp",
        "block_number",
        "tx_hash",
        "pool",
        "sender",
        "recipient",
        "amount0",
        "amount1",
        "amount_usd",
        "tick",
        "sqrtPriceX96",
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame.from_records(records)
    missing_cols = set(columns) - set(df.columns)
    for col in missing_cols:
        df[col] = None
    return df[columns]


def _normalise_swap(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a raw swap payload into a tabular dict."""

    def _safe_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_str(value: Any) -> str | None:
        return str(value) if value not in (None, "") else None

    pool_info = raw.get("pool")
    if isinstance(pool_info, dict):
        pool_id = pool_info.get("id") or pool_info.get("address")
    else:
        pool_id = pool_info

    tx_info = raw.get("transaction") or {}
    block_number = tx_info.get("blockNumber") or raw.get("blockNumber")

    return {
        "timestamp": _safe_int(raw.get("timestamp")),
        "block_number": _safe_int(block_number),
        "tx_hash": _safe_str(tx_info.get("id") or raw.get("transactionHash") or raw.get("id")),
        "pool": _safe_str(pool_id),
        "sender": _safe_str(raw.get("sender")),
        "recipient": _safe_str(raw.get("recipient")),
        "amount0": _safe_str(raw.get("amount0")),
        "amount1": _safe_str(raw.get("amount1")),
        "amount_usd": _safe_str(raw.get("amountUSD")),
        "tick": _safe_int(raw.get("tick")),
        "sqrtPriceX96": _safe_str(raw.get("sqrtPriceX96")),
    }
