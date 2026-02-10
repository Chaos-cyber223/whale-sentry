"""Sandwich attack detection for Uniswap V3 swaps.

This module provides detection logic for identifying sandwich attacks in
on-chain swap data. A sandwich attack occurs when an attacker executes
two transactions around a victim's trade to profit from price manipulation.

Pattern:
1. Attacker front-runs victim with a buy (pushing price up)
2. Victim buys at inflated price
3. Attacker back-runs victim with a sell (profiting from the higher price)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from whalesentry.models.swap import ETH_ADDRESS_PATTERN


class SandwichCandidate(BaseModel):
    """Represents a detected sandwich attack candidate.

    A sandwich attack involves an attacker placing transactions before (front-run)
    and after (back-run) a victim's transaction to profit from price manipulation.

    Attributes:
        attacker: Ethereum address of the suspected attacker.
        victim_tx: Transaction hash of the victim's trade.
        front_run_tx: Transaction hash of the attacker's front-running trade.
        back_run_tx: Transaction hash of the attacker's back-running trade.
        pool: Liquidity pool address where the attack occurred.
        front_run_timestamp: Unix timestamp of the front-run transaction.
        victim_timestamp: Unix timestamp of the victim transaction.
        back_run_timestamp: Unix timestamp of the back-run transaction.
        front_run_amount_usd: USD value of the front-run transaction.
        victim_amount_usd: USD value of the victim transaction.
        back_run_amount_usd: USD value of the back-run transaction.
        profit_estimate_usd: Estimated profit from the attack.
        confidence_score: Detection confidence (0.0 to 1.0).

    Example:
        >>> candidate = SandwichCandidate(
        ...     attacker="0x1234...",
        ...     victim_tx="0xabcd...",
        ...     front_run_tx="0xef01...",
        ...     back_run_tx="0x2345...",
        ...     pool="0x88e6...",
        ...     front_run_timestamp=1769263223,
        ...     victim_timestamp=1769263224,
        ...     back_run_timestamp=1769263225,
        ...     front_run_amount_usd="1000.00",
        ...     victim_amount_usd="5000.00",
        ...     back_run_amount_usd="1100.00",
        ...     profit_estimate_usd="100.00",
        ...     confidence_score=0.85,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    attacker: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Ethereum address of the suspected attacker",
        examples=["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"],
    )
    victim_tx: str = Field(
        ...,
        min_length=66,
        max_length=66,
        description="Transaction hash of the victim's trade",
        examples=["0x0c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e832"],
    )
    front_run_tx: str = Field(
        ...,
        min_length=66,
        max_length=66,
        description="Transaction hash of the front-running trade",
        examples=["0x1c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e833"],
    )
    back_run_tx: str = Field(
        ...,
        min_length=66,
        max_length=66,
        description="Transaction hash of the back-running trade",
        examples=["0x2c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e834"],
    )
    pool: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Liquidity pool address",
        examples=["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"],
    )
    front_run_timestamp: int = Field(
        ...,
        ge=1438214400,
        le=2000000000,
        description="Unix timestamp of front-run transaction",
        examples=[1769263223],
    )
    victim_timestamp: int = Field(
        ...,
        ge=1438214400,
        le=2000000000,
        description="Unix timestamp of victim transaction",
        examples=[1769263224],
    )
    back_run_timestamp: int = Field(
        ...,
        ge=1438214400,
        le=2000000000,
        description="Unix timestamp of back-run transaction",
        examples=[1769263225],
    )
    front_run_amount_usd: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="USD value of front-run transaction",
        examples=["1000.00"],
    )
    victim_amount_usd: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="USD value of victim transaction",
        examples=["5000.00"],
    )
    back_run_amount_usd: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="USD value of back-run transaction",
        examples=["1100.00"],
    )
    profit_estimate_usd: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Estimated profit in USD",
        examples=["100.00"],
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0.0 to 1.0)",
        examples=[0.85],
    )

    @field_validator("attacker", "pool")
    @classmethod
    def validate_eth_address(cls, v: str) -> str:
        """Validate Ethereum address format and normalize to lowercase."""
        if not v:
            raise ValueError("Address cannot be empty")
        v = v.lower()
        if not ETH_ADDRESS_PATTERN.match(v):
            raise ValueError(
                f"Invalid Ethereum address: {v[:20]}... "
                "Expected format: 0x followed by 40 hex characters"
            )
        return v

    @field_validator("victim_tx", "front_run_tx", "back_run_tx")
    @classmethod
    def validate_tx_hash(cls, v: str) -> str:
        """Validate transaction hash format and normalize to lowercase."""
        if not v:
            raise ValueError("Transaction hash cannot be empty")
        v = v.lower()
        if not v.startswith("0x") or len(v) != 66:
            raise ValueError(
                f"Invalid transaction hash: {v[:20]}... "
                "Expected format: 0x followed by 64 hex characters"
            )
        return v

    @field_validator(
        "front_run_amount_usd",
        "victim_amount_usd",
        "back_run_amount_usd",
        "profit_estimate_usd",
    )
    @classmethod
    def validate_amount(cls, v: str) -> str:
        """Validate that amount is a valid decimal string."""
        if not v:
            raise ValueError("Amount cannot be empty")
        try:
            Decimal(v)
        except Exception as e:
            raise ValueError(f"Invalid decimal amount: {v}") from e
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert candidate to dictionary format."""
        return {
            "attacker": self.attacker,
            "victim_tx": self.victim_tx,
            "front_run_tx": self.front_run_tx,
            "back_run_tx": self.back_run_tx,
            "pool": self.pool,
            "front_run_timestamp": self.front_run_timestamp,
            "victim_timestamp": self.victim_timestamp,
            "back_run_timestamp": self.back_run_timestamp,
            "front_run_amount_usd": self.front_run_amount_usd,
            "victim_amount_usd": self.victim_amount_usd,
            "back_run_amount_usd": self.back_run_amount_usd,
            "profit_estimate_usd": self.profit_estimate_usd,
            "confidence_score": self.confidence_score,
        }


@dataclass(frozen=True)
class DetectionResult:
    """Result of sandwich attack detection.

    Attributes:
        candidates: List of detected sandwich attack candidates.
        total_swaps_analyzed: Total number of swaps processed.
        pools_analyzed: Number of unique pools analyzed.
        time_window_seconds: Time window used for detection.
        detection_errors: List of any errors encountered during detection.
    """

    candidates: tuple[SandwichCandidate, ...]
    total_swaps_analyzed: int
    pools_analyzed: int
    time_window_seconds: int
    detection_errors: list[str] = field(default_factory=list)

    @property
    def total_candidates(self) -> int:
        """Return total number of detected candidates."""
        return len(self.candidates)

    @property
    def unique_attackers(self) -> set[str]:
        """Return set of unique attacker addresses."""
        return {c.attacker for c in self.candidates}

    @property
    def high_confidence_candidates(self) -> list[SandwichCandidate]:
        """Return candidates with confidence >= 0.8."""
        return [c for c in self.candidates if c.confidence_score >= 0.8]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for reporting."""
        return {
            "total_candidates": self.total_candidates,
            "total_swaps_analyzed": self.total_swaps_analyzed,
            "pools_analyzed": self.pools_analyzed,
            "time_window_seconds": self.time_window_seconds,
            "unique_attackers": len(self.unique_attackers),
            "high_confidence_count": len(self.high_confidence_candidates),
            "detection_errors": self.detection_errors,
        }


def detect_sandwich_attacks(
    df: pd.DataFrame,
    time_window_seconds: int = 60,
    min_usd_value: Decimal = Decimal("100"),
    amount_similarity_threshold: float = 0.5,
) -> DetectionResult:
    """Detect sandwich attacks in swap data.

    This function analyzes swap transactions to identify potential sandwich
    attacks where an attacker places trades before and after a victim's trade
    to profit from price manipulation.

    Detection Logic:
    1. Group transactions by pool
    2. Sort by timestamp within each pool
    3. Look for patterns where the same address appears before AND after
       another address within the time window
    4. Check for directional reversal (buy then sell, or sell then buy)
    5. Validate that front-run and back-run amounts are similar in magnitude

    Args:
        df: DataFrame containing swap data with columns:
            - timestamp: Unix timestamp
            - pool: Pool address
            - sender: Transaction sender address
            - tx_hash: Transaction hash
            - amount0, amount1: Token amounts (one positive, one negative)
            - amount_usd: USD value of the swap
        time_window_seconds: Maximum time between front-run and back-run
            transactions to consider as a sandwich attack.
        min_usd_value: Minimum USD value for transactions to be considered.
        amount_similarity_threshold: Minimum ratio between smaller and larger
            transaction amounts to consider them similar (0.0 to 1.0).

    Returns:
        DetectionResult containing all detected sandwich candidates and
        detection statistics.

    Example:
        >>> df = pd.read_parquet("swaps_clean.parquet")
        >>> result = detect_sandwich_attacks(df, time_window_seconds=60)
        >>> print(f"Found {result.total_candidates} potential attacks")
    """
    errors: list[str] = []

    # Validate input
    required_columns = {
        "timestamp", "pool", "sender", "tx_hash",
        "amount0", "amount1", "amount_usd",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return DetectionResult(
            candidates=tuple(),
            total_swaps_analyzed=0,
            pools_analyzed=0,
            time_window_seconds=time_window_seconds,
            detection_errors=errors,
        )

    if df.empty:
        return DetectionResult(
            candidates=tuple(),
            total_swaps_analyzed=0,
            pools_analyzed=0,
            time_window_seconds=time_window_seconds,
            detection_errors=["Empty DataFrame provided"],
        )

    # Make a copy to avoid modifying original
    data = df.copy()

    # Convert amount_usd to numeric for filtering
    data["amount_usd_numeric"] = pd.to_numeric(data["amount_usd"], errors="coerce")
    data["amount0_numeric"] = pd.to_numeric(data["amount0"], errors="coerce")
    data["amount1_numeric"] = pd.to_numeric(data["amount1"], errors="coerce")

    # Filter by minimum USD value
    data = data[data["amount_usd_numeric"] >= float(min_usd_value)]

    if data.empty:
        return DetectionResult(
            candidates=tuple(),
            total_swaps_analyzed=len(df),
            pools_analyzed=df["pool"].nunique(),
            time_window_seconds=time_window_seconds,
            detection_errors=["No transactions above minimum USD threshold"],
        )

    # Determine trade direction for each transaction
    data["is_buy"] = (data["amount0_numeric"] < 0) & (data["amount1_numeric"] > 0)
    data["is_sell"] = (data["amount0_numeric"] > 0) & (data["amount1_numeric"] < 0)

    # Group by pool and detect sandwiches
    candidates: list[SandwichCandidate] = []

    for _pool, group in data.groupby("pool"):
        # Sort by timestamp
        group = group.sort_values("timestamp").reset_index(drop=True)

        # Look for sandwich patterns
        pool_candidates = _detect_sandwiches_in_pool(
            group,
            time_window_seconds,
            amount_similarity_threshold,
        )
        candidates.extend(pool_candidates)

    return DetectionResult(
        candidates=tuple(candidates),
        total_swaps_analyzed=len(df),
        pools_analyzed=df["pool"].nunique(),
        time_window_seconds=time_window_seconds,
        detection_errors=errors,
    )


def _detect_sandwiches_in_pool(
    pool_data: pd.DataFrame,
    time_window_seconds: int,
    amount_similarity_threshold: float,
) -> list[SandwichCandidate]:
    """Detect sandwich attacks within a single pool.

    Args:
        pool_data: DataFrame containing swaps for a single pool,
            sorted by timestamp.
        time_window_seconds: Maximum time between front-run and back-run.
        amount_similarity_threshold: Minimum ratio between smaller and larger
            transaction amounts.

    Returns:
        List of detected sandwich candidates in this pool.
    """
    candidates: list[SandwichCandidate] = []
    n = len(pool_data)

    if n < 3:
        return candidates

    # For each potential victim transaction (middle transaction)
    for victim_idx in range(1, n - 1):
        victim = pool_data.iloc[victim_idx]

        # Look backwards for front-run by same attacker
        for front_idx in range(victim_idx - 1, -1, -1):
            front = pool_data.iloc[front_idx]

            # Check time window
            time_diff = victim["timestamp"] - front["timestamp"]
            if time_diff > time_window_seconds:
                break

            # Look forwards for back-run by same attacker
            for back_idx in range(victim_idx + 1, n):
                back = pool_data.iloc[back_idx]

                # Check time window from front to back
                total_time = back["timestamp"] - front["timestamp"]
                if total_time > time_window_seconds:
                    break

                # Check if front and back are from same sender (attacker)
                if front["sender"] != back["sender"]:
                    continue

                # Skip if attacker is also the victim
                if front["sender"] == victim["sender"]:
                    continue

                # Check directional reversal (buy -> victim -> sell OR sell -> victim -> buy)
                front_is_buy = front["is_buy"]
                front_is_sell = front["is_sell"]
                back_is_buy = back["is_buy"]
                back_is_sell = back["is_sell"]

                # Must have clear direction on both sides
                if not (front_is_buy or front_is_sell) or not (back_is_buy or back_is_sell):
                    continue

                # Check for reversal: front buy + back sell, or front sell + back buy
                is_reversal = (front_is_buy and back_is_sell) or (front_is_sell and back_is_buy)
                if not is_reversal:
                    continue

                # Check amount similarity
                front_usd = float(front["amount_usd_numeric"])
                back_usd = float(back["amount_usd_numeric"])

                if front_usd <= 0 or back_usd <= 0:
                    continue

                min_amount = min(front_usd, back_usd)
                max_amount = max(front_usd, back_usd)
                similarity_ratio = min_amount / max_amount if max_amount > 0 else 0

                if similarity_ratio < amount_similarity_threshold:
                    continue

                # Calculate confidence score
                confidence = _calculate_confidence(
                    time_diff,
                    total_time,
                    similarity_ratio,
                    float(victim["amount_usd_numeric"]),
                    front_usd,
                )

                # Estimate profit (simplified: back - front amounts)
                profit_estimate = back_usd - front_usd if back_usd > front_usd else Decimal("0")

                candidate = SandwichCandidate(
                    attacker=front["sender"],
                    victim_tx=victim["tx_hash"],
                    front_run_tx=front["tx_hash"],
                    back_run_tx=back["tx_hash"],
                    pool=front["pool"],
                    front_run_timestamp=int(front["timestamp"]),
                    victim_timestamp=int(victim["timestamp"]),
                    back_run_timestamp=int(back["timestamp"]),
                    front_run_amount_usd=str(front_usd),
                    victim_amount_usd=str(victim["amount_usd_numeric"]),
                    back_run_amount_usd=str(back_usd),
                    profit_estimate_usd=str(profit_estimate),
                    confidence_score=confidence,
                )
                candidates.append(candidate)

    return candidates


def _calculate_confidence(
    victim_front_time: int,
    total_time: int,
    amount_similarity: float,
    victim_amount: float,
    front_amount: float,
) -> float:
    """Calculate confidence score for a sandwich detection.

    Higher scores indicate stronger evidence of an actual sandwich attack.

    Factors:
    - Time proximity (shorter = higher confidence)
    - Amount similarity (more similar = higher confidence)
    - Victim transaction size (larger = more attractive target)
    - Amount ratio (larger front-run relative to victim = higher confidence)

    Args:
        victim_front_time: Seconds between front-run and victim.
        total_time: Seconds between front-run and back-run.
        amount_similarity: Ratio of smaller to larger amount (0.0 to 1.0).
        victim_amount: USD value of victim transaction.
        front_amount: USD value of front-run transaction.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    # Time factor: shorter time = higher confidence
    # Optimal is within 5 seconds, degrades after that
    if total_time <= 5:
        time_factor = 1.0
    elif total_time <= 15:
        time_factor = 0.9
    elif total_time <= 30:
        time_factor = 0.8
    else:
        time_factor = max(0.5, 1.0 - (total_time - 30) / 60)

    # Amount similarity factor
    similarity_factor = amount_similarity

    # Victim size factor: transactions between $1000-$10000 are optimal targets
    if victim_amount >= 10000:
        victim_factor = 1.0
    elif victim_amount >= 5000:
        victim_factor = 0.9
    elif victim_amount >= 1000:
        victim_factor = 0.8
    else:
        victim_factor = 0.6

    # Front-run ratio factor: attacker should put up meaningful capital
    ratio = front_amount / victim_amount if victim_amount > 0 else 0
    if ratio >= 0.5:
        ratio_factor = 1.0
    elif ratio >= 0.3:
        ratio_factor = 0.9
    elif ratio >= 0.1:
        ratio_factor = 0.7
    else:
        ratio_factor = 0.5

    # Weighted combination
    confidence = (
        time_factor * 0.35 + similarity_factor * 0.25 + victim_factor * 0.2 + ratio_factor * 0.2
    )

    return round(min(1.0, max(0.0, confidence)), 4)


def candidates_to_dataframe(candidates: list[SandwichCandidate]) -> pd.DataFrame:
    """Convert list of candidates to DataFrame.

    Args:
        candidates: List of SandwichCandidate objects.

    Returns:
        DataFrame with one row per candidate.
    """
    if not candidates:
        return pd.DataFrame(
            columns=[
                "attacker",
                "victim_tx",
                "front_run_tx",
                "back_run_tx",
                "pool",
                "front_run_timestamp",
                "victim_timestamp",
                "back_run_timestamp",
                "front_run_amount_usd",
                "victim_amount_usd",
                "back_run_amount_usd",
                "profit_estimate_usd",
                "confidence_score",
            ]
        )

    records = [c.to_dict() for c in candidates]
    return pd.DataFrame(records)


__all__ = [
    "SandwichCandidate",
    "DetectionResult",
    "detect_sandwich_attacks",
    "candidates_to_dataframe",
]
