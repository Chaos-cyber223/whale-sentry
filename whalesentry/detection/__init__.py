"""Detection algorithms for MEV and suspicious trading patterns.

This module provides detection algorithms for identifying various types of
on-chain trading abuse, including sandwich attacks and wash trading.
"""

from whalesentry.detection.sandwich import (
    DetectionResult,
    SandwichCandidate,
    candidates_to_dataframe,
    detect_sandwich_attacks,
    detect_sandwich_attacks_optimized,
)
from whalesentry.detection.wash_trade import (
    WashTradeCandidate,
    candidates_to_dataframe as wash_candidates_to_dataframe,
    detect_wash_trades,
)

__all__ = [
    "SandwichCandidate",
    "DetectionResult",
    "detect_sandwich_attacks",
    "detect_sandwich_attacks_optimized",
    "candidates_to_dataframe",
    "WashTradeCandidate",
    "detect_wash_trades",
    "wash_candidates_to_dataframe",
]
