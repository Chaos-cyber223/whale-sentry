"""Detection algorithms for MEV and suspicious trading patterns.

This module provides detection algorithms for identifying various types of
on-chain trading abuse, including sandwich attacks and wash trading.
"""

from whalesentry.detection.sandwich import (
    DetectionResult,
    SandwichCandidate,
    candidates_to_dataframe,
    detect_sandwich_attacks,
)

__all__ = [
    "SandwichCandidate",
    "DetectionResult",
    "detect_sandwich_attacks",
    "candidates_to_dataframe",
]
