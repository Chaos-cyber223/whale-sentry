"""Wash trading detection for Uniswap V3 swaps.

This module provides detection logic for identifying wash trading patterns in
on-chain swap data. Wash trading occurs when a trader executes multiple
roundtrip trades (buy-sell cycles) to artificially inflate trading volume
or manipulate price perception.

Pattern (ROUNDTRIP):
1. Trader buys token A with token B
2. Within a short time window, trader sells token A back for token B
3. This pattern repeats multiple times (minimum 3 roundtrips)
4. Minimal price difference between buy and sell (not normal arbitrage)

Detection focuses on same-address roundtrip patterns within 300-second windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from whalesentry.models.swap import ETH_ADDRESS_PATTERN


class WashTradeCandidate(BaseModel):
    """Represents a detected wash trading candidate.

    A wash trade involves a trader executing multiple roundtrip trades
    (buy-sell cycles) within a short time window to artificially inflate
    trading volume or manipulate market perception.

    Attributes:
        trader: Ethereum address of the suspected wash trader.
        pool: Liquidity pool address where the wash trading occurred.
        roundtrip_count: Number of detected buy-sell roundtrips.
        first_tx: Transaction hash of the first trade in the pattern.
        last_tx: Transaction hash of the last trade in the pattern.
        first_timestamp: Unix timestamp of the first transaction.
        last_timestamp: Unix timestamp of the last transaction.
        total_volume_usd: Total USD volume across all roundtrip trades.
        avg_roundtrip_time: Average time (seconds) between buy and sell.
        price_variance: Price variance across roundtrips (lower = more suspicious).
        confidence_score: Detection confidence (0.0 to 1.0).

    Example:
        >>> candidate = WashTradeCandidate(
        ...     trader="0x1234...",
        ...     pool="0x88e6...",
        ...     roundtrip_count=5,
        ...     first_tx="0xabcd...",
        ...     last_tx="0xef01...",
        ...     first_timestamp=1769263223,
        ...     last_timestamp=1769263523,
        ...     total_volume_usd="15000.00",
        ...     avg_roundtrip_time=60.0,
        ...     price_variance="0.02",
        ...     confidence_score=0.85,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    trader: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Ethereum address of the suspected wash trader",
        examples=["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"],
    )
    pool: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Liquidity pool address",
        examples=["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"],
    )
    roundtrip_count: int = Field(
        ...,
        ge=3,
        description="Number of detected buy-sell roundtrips (minimum 3)",
        examples=[5],
    )
    first_tx: str = Field(
        ...,
        min_length=66,
        max_length=66,
        description="Transaction hash of the first trade",
        examples=["0x0c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e832"],
    )
    last_tx: str = Field(
        ...,
        min_length=66,
        max_length=66,
        description="Transaction hash of the last trade",
        examples=["0x1c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e833"],
    )
    first_timestamp: int = Field(
        ...,
        ge=1438214400,
        le=2000000000,
        description="Unix timestamp of the first transaction",
        examples=[1769263223],
    )
    last_timestamp: int = Field(
        ...,
        ge=1438214400,
        le=2000000000,
        description="Unix timestamp of the last transaction",
        examples=[1769263523],
    )
    total_volume_usd: str = Field(
        ...,
        description="Total USD volume across all roundtrip trades",
        examples=["15000.00"],
    )
    avg_roundtrip_time: float = Field(
        ...,
        ge=0.0,
        description="Average time (seconds) between buy and sell in each roundtrip",
        examples=[60.0],
    )
    price_variance: str = Field(
        ...,
        description="Price variance across roundtrips (Decimal as string)",
        examples=["0.02"],
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0.0 to 1.0)",
        examples=[0.85],
    )

    @field_validator("trader", "pool")
    @classmethod
    def validate_address(cls, v: str) -> str:
        """Validate and normalize Ethereum address."""
        v_lower = v.lower()
        if not ETH_ADDRESS_PATTERN.match(v_lower):
            msg = f"Invalid Ethereum address format: {v}"
            raise ValueError(msg)
        return v_lower

    @field_validator("first_tx", "last_tx")
    @classmethod
    def validate_tx_hash(cls, v: str) -> str:
        """Validate transaction hash format."""
        v_lower = v.lower()
        if not v_lower.startswith("0x") or len(v_lower) != 66:
            msg = f"Invalid transaction hash format: {v}"
            raise ValueError(msg)
        return v_lower

    @field_validator("total_volume_usd", "price_variance")
    @classmethod
    def validate_decimal_string(cls, v: str) -> str:
        """Validate that string can be converted to Decimal."""
        try:
            Decimal(v)
        except Exception as e:
            msg = f"Invalid decimal string: {v}"
            raise ValueError(msg) from e
        return v

    @property
    def volume_usd(self) -> Decimal:
        """Return total volume as Decimal."""
        return Decimal(self.total_volume_usd)

    @property
    def variance(self) -> Decimal:
        """Return price variance as Decimal."""
        return Decimal(self.price_variance)

    @property
    def time_span_seconds(self) -> int:
        """Return total time span of the wash trading pattern."""
        return self.last_timestamp - self.first_timestamp


@dataclass(frozen=True)
class DetectionResult:
    """Result of wash trading detection analysis.

    Attributes:
        candidates: Tuple of detected wash trade candidates.
        total_swaps_analyzed: Total number of swap transactions analyzed.
        pools_analyzed: Number of unique pools analyzed.
        total_candidates: Number of wash trade candidates detected.
        time_window_seconds: Time window used for roundtrip detection.
        min_usd_threshold: Minimum USD value threshold applied.
        min_roundtrips: Minimum number of roundtrips required.
    """

    candidates: tuple[WashTradeCandidate, ...] = field(default_factory=tuple)
    total_swaps_analyzed: int = 0
    pools_analyzed: int = 0
    time_window_seconds: int = 300
    min_usd_threshold: Decimal = Decimal("1000")
    min_roundtrips: int = 3

    @property
    def total_candidates(self) -> int:
        """Return total number of candidates."""
        return len(self.candidates)


def _calculate_confidence(
    roundtrip_count: int,
    price_variance: Decimal,
    avg_roundtrip_time: float,
    time_window: int,
) -> float:
    """Calculate confidence score for wash trade detection.

    Higher confidence when:
    - More roundtrips detected
    - Lower price variance (not normal arbitrage)
    - Shorter roundtrip times (rapid trading)
    - Pattern fits within time window

    Args:
        roundtrip_count: Number of detected roundtrips.
        price_variance: Price variance across roundtrips.
        avg_roundtrip_time: Average time between buy and sell.
        time_window: Maximum time window for detection.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    # Base score from roundtrip count (3 roundtrips = 0.5, 10+ = 1.0)
    roundtrip_score = min(1.0, (roundtrip_count - 3) / 7.0 + 0.5)

    # Price variance penalty (lower variance = higher confidence)
    # Variance < 1% = 1.0, variance > 10% = 0.0
    variance_score = max(0.0, 1.0 - float(price_variance) * 10.0)

    # Time score (faster roundtrips = higher confidence)
    # < 30s = 1.0, > 150s = 0.5
    time_score = max(0.5, 1.0 - (avg_roundtrip_time / time_window) * 0.5)

    # Weighted average
    confidence = (roundtrip_score * 0.4) + (variance_score * 0.4) + (time_score * 0.2)

    return round(min(1.0, max(0.0, confidence)), 4)


def _detect_wash_trades_in_pool(
    pool_swaps: pd.DataFrame,
    time_window_seconds: int,
    min_usd_threshold: Decimal,
    min_roundtrips: int,
) -> list[WashTradeCandidate]:
    """Detect wash trading patterns within a single pool.

    Algorithm:
    1. Group swaps by trader (sender address)
    2. For each trader, identify buy-sell roundtrips
    3. Count roundtrips within time window
    4. Calculate confidence based on pattern characteristics

    Args:
        pool_swaps: DataFrame of swaps in a single pool.
        time_window_seconds: Maximum time between first and last trade.
        min_usd_threshold: Minimum USD value per trade.
        min_roundtrips: Minimum number of roundtrips to flag.

    Returns:
        List of detected wash trade candidates.
    """
    candidates: list[WashTradeCandidate] = []

    # Group by trader
    for trader, trader_swaps in pool_swaps.groupby("sender"):
        # Skip if not enough trades for minimum roundtrips (need at least 2 * min_roundtrips)
        if len(trader_swaps) < min_roundtrips * 2:
            continue

        # Sort by timestamp
        trader_swaps = trader_swaps.sort_values("timestamp")

        # Identify direction of each trade based on amount0 (negative = sell token0, positive = buy token0)
        trader_swaps = trader_swaps.copy()
        trader_swaps["direction"] = trader_swaps["amount0"].apply(
            lambda x: "buy" if float(x) < 0 else "sell"
        )

        # Find roundtrips (buy followed by sell, or sell followed by buy)
        roundtrips: list[tuple[pd.Series, pd.Series]] = []
        trades = list(trader_swaps.iterrows())

        i = 0
        while i < len(trades) - 1:
            idx1, trade1 = trades[i]
            idx2, trade2 = trades[i + 1]

            # Check if it's a roundtrip (opposite directions)
            if trade1["direction"] != trade2["direction"]:
                # Check time window
                time_diff = trade2["timestamp"] - trade1["timestamp"]
                if time_diff <= time_window_seconds:
                    # Check minimum USD threshold
                    if (
                        Decimal(str(trade1["amount_usd"])) >= min_usd_threshold
                        and Decimal(str(trade2["amount_usd"])) >= min_usd_threshold
                    ):
                        roundtrips.append((trade1, trade2))
                        i += 2  # Skip both trades
                        continue

            i += 1

        # Check if we have enough roundtrips
        if len(roundtrips) < min_roundtrips:
            continue

        # Calculate pattern statistics
        first_trade = roundtrips[0][0]
        last_trade = roundtrips[-1][1]

        # Check if entire pattern fits within time window
        pattern_time_span = last_trade["timestamp"] - first_trade["timestamp"]
        if pattern_time_span > time_window_seconds:
            continue

        total_volume = sum(
            Decimal(str(t1["amount_usd"])) + Decimal(str(t2["amount_usd"]))
            for t1, t2 in roundtrips
        )

        # Calculate average roundtrip time
        roundtrip_times = [t2["timestamp"] - t1["timestamp"] for t1, t2 in roundtrips]
        avg_roundtrip_time = sum(roundtrip_times) / len(roundtrip_times)

        # Calculate price variance across roundtrips
        prices = []
        for t1, t2 in roundtrips:
            # Estimate price from sqrtPriceX96 if available
            if "sqrtPriceX96" in t1 and pd.notna(t1["sqrtPriceX96"]):
                sqrt_price = int(t1["sqrtPriceX96"])
                # Convert sqrtPriceX96 to price (simplified)
                price = (sqrt_price / (2**96)) ** 2
                prices.append(price)

        if prices:
            price_variance = Decimal(str(pd.Series(prices).std() / pd.Series(prices).mean()))
        else:
            # Fallback: use amount ratio variance
            ratios = [
                abs(float(t1["amount0"]) / float(t2["amount0"]))
                for t1, t2 in roundtrips
                if float(t2["amount0"]) != 0
            ]
            if ratios:
                price_variance = Decimal(str(pd.Series(ratios).std() / pd.Series(ratios).mean()))
            else:
                price_variance = Decimal("0.1")  # Default moderate variance

        # Calculate confidence score
        confidence = _calculate_confidence(
            roundtrip_count=len(roundtrips),
            price_variance=price_variance,
            avg_roundtrip_time=avg_roundtrip_time,
            time_window=time_window_seconds,
        )

        # Create candidate
        candidate = WashTradeCandidate(
            trader=str(trader),
            pool=str(first_trade["pool"]),
            roundtrip_count=len(roundtrips),
            first_tx=str(first_trade["tx_hash"]),
            last_tx=str(last_trade["tx_hash"]),
            first_timestamp=int(first_trade["timestamp"]),
            last_timestamp=int(last_trade["timestamp"]),
            total_volume_usd=str(total_volume),
            avg_roundtrip_time=float(avg_roundtrip_time),
            price_variance=str(price_variance),
            confidence_score=confidence,
        )

        candidates.append(candidate)

    return candidates


def detect_wash_trades(
    swaps_df: pd.DataFrame,
    time_window_seconds: int = 300,
    min_usd_threshold: Decimal = Decimal("1000"),
    min_roundtrips: int = 3,
) -> DetectionResult:
    """Detect wash trading patterns in swap data.

    This function analyzes swap transactions to identify potential wash trading
    patterns where traders execute multiple roundtrip trades (buy-sell cycles)
    within a short time window.

    Args:
        swaps_df: DataFrame with columns: timestamp, tx_hash, pool, sender,
                  amount0, amount1, amount_usd, sqrtPriceX96.
        time_window_seconds: Maximum time window for roundtrip pattern (default: 300s).
        min_usd_threshold: Minimum USD value per trade (default: $1000).
        min_roundtrips: Minimum number of roundtrips to flag (default: 3).

    Returns:
        DetectionResult containing all detected wash trade candidates.

    Raises:
        ValueError: If required columns are missing from swaps_df.

    Example:
        >>> df = pd.DataFrame({
        ...     "timestamp": [1769263223, 1769263283, 1769263343],
        ...     "tx_hash": ["0xabc...", "0xdef...", "0x123..."],
        ...     "pool": ["0x88e6..."] * 3,
        ...     "sender": ["0x1234..."] * 3,
        ...     "amount0": ["-1000", "1000", "-1000"],
        ...     "amount1": ["1.0", "-1.0", "1.0"],
        ...     "amount_usd": ["1000", "1000", "1000"],
        ...     "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
        ... })
        >>> result = detect_wash_trades(df)
        >>> print(f"Found {result.total_candidates} candidates")
    """
    # Validate required columns
    required_cols = {"timestamp", "tx_hash", "pool", "sender", "amount0", "amount_usd"}
    missing_cols = required_cols - set(swaps_df.columns)
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}"
        raise ValueError(msg)

    # Handle empty DataFrame
    if swaps_df.empty:
        return DetectionResult(
            candidates=tuple(),
            total_swaps_analyzed=0,
            pools_analyzed=0,
            time_window_seconds=time_window_seconds,
            min_usd_threshold=min_usd_threshold,
            min_roundtrips=min_roundtrips,
        )

    # Collect all candidates across pools
    all_candidates: list[WashTradeCandidate] = []

    # Process each pool independently
    for pool, pool_swaps in swaps_df.groupby("pool"):
        pool_candidates = _detect_wash_trades_in_pool(
            pool_swaps=pool_swaps,
            time_window_seconds=time_window_seconds,
            min_usd_threshold=min_usd_threshold,
            min_roundtrips=min_roundtrips,
        )
        all_candidates.extend(pool_candidates)

    # Sort candidates by confidence score (descending)
    all_candidates.sort(key=lambda c: c.confidence_score, reverse=True)

    return DetectionResult(
        candidates=tuple(all_candidates),
        total_swaps_analyzed=len(swaps_df),
        pools_analyzed=swaps_df["pool"].nunique(),
        time_window_seconds=time_window_seconds,
        min_usd_threshold=min_usd_threshold,
        min_roundtrips=min_roundtrips,
    )


def candidates_to_dataframe(candidates: list[WashTradeCandidate]) -> pd.DataFrame:
    """Convert list of wash trade candidates to pandas DataFrame.

    Args:
        candidates: List of WashTradeCandidate objects.

    Returns:
        DataFrame with all candidate fields as columns.

    Example:
        >>> candidates = [candidate1, candidate2]
        >>> df = candidates_to_dataframe(candidates)
        >>> print(df[["trader", "roundtrip_count", "confidence_score"]])
    """
    if not candidates:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=[
                "trader",
                "pool",
                "roundtrip_count",
                "first_tx",
                "last_tx",
                "first_timestamp",
                "last_timestamp",
                "total_volume_usd",
                "avg_roundtrip_time",
                "price_variance",
                "confidence_score",
                "time_span_seconds",
            ]
        )

    # Convert to list of dicts
    data = [
        {
            "trader": c.trader,
            "pool": c.pool,
            "roundtrip_count": c.roundtrip_count,
            "first_tx": c.first_tx,
            "last_tx": c.last_tx,
            "first_timestamp": c.first_timestamp,
            "last_timestamp": c.last_timestamp,
            "total_volume_usd": c.total_volume_usd,
            "avg_roundtrip_time": c.avg_roundtrip_time,
            "price_variance": c.price_variance,
            "confidence_score": c.confidence_score,
            "time_span_seconds": c.time_span_seconds,
        }
        for c in candidates
    ]

    return pd.DataFrame(data)


__all__ = [
    "WashTradeCandidate",
    "DetectionResult",
    "detect_wash_trades",
    "candidates_to_dataframe",
]
