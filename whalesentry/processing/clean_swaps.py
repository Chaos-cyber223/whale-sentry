"""Cleaning helpers for swap data."""
from __future__ import annotations

from typing import Iterable, List

import pandas as pd

ADDRESS_COLUMNS = ("pool", "sender", "recipient")
DEDUP_COLUMNS = ["tx_hash", "timestamp", "sender", "recipient", "amount0", "amount1"]


def clean_swap_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned dataframe ready for downstream detectors."""
    if "timestamp" not in frame.columns:
        raise ValueError("Input frame must include a 'timestamp' column")

    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_numeric(cleaned["timestamp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp"])
    cleaned["timestamp"] = cleaned["timestamp"].astype("int64")

    for column in ADDRESS_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = None
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.lower()
            .where(lambda s: s.notna(), None)
        )

    for column in ("amount0", "amount1"):
        if column not in cleaned.columns:
            cleaned[column] = None
        else:
            cleaned[column] = cleaned[column].astype("string")

    if "tx_hash" not in cleaned.columns:
        cleaned["tx_hash"] = None
    cleaned["tx_hash"] = cleaned["tx_hash"].astype("string")

    subset = _existing_columns(cleaned, DEDUP_COLUMNS)
    if subset:
        cleaned = cleaned.drop_duplicates(subset=subset)

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def _existing_columns(frame: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [col for col in columns if col in frame.columns]
