"""Pydantic models for wash trading detection.

This module provides type-safe data models for representing wash trading
detections, including round-trip trading patterns and coordinated trading
behavior among multiple addresses.

Wash trading involves artificial trading activity where the same entity
controls multiple addresses to create fake volume without actual value transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from whalesentry.models.swap import ETH_ADDRESS_PATTERN


class WashTradeType(StrEnum):
    """Type of wash trading pattern detected.

    Attributes:
        ROUNDTRIP: Single address buying and selling in quick succession.
        CLOSED_LOOP: Multiple addresses forming a circular transfer pattern.
        COORDINATED: Multiple addresses trading at similar amounts/times.
    """

    ROUNDTRIP = "roundtrip"
    CLOSED_LOOP = "closed_loop"
    COORDINATED = "coordinated"


class WashTradeMetrics(BaseModel):
    """Quantitative metrics for wash trading detection.

    These metrics provide objective measurements of suspicious trading
    patterns, used to calculate confidence scores.

    Attributes:
        trade_count: Total number of trades involved in the pattern.
        unique_addresses: Number of unique addresses participating.
        time_window_seconds: Duration of the suspicious activity window.
        round_trip_count: Number of complete buy-sell cycles (for roundtrip type).
        avg_trade_interval_seconds: Average time between consecutive trades.
        amount_similarity_score: Similarity ratio of trade amounts (0.0 to 1.0).
        volume_ratio: Ratio of wash volume to legitimate volume in the window.
        counterparty_diversity: Ratio of unique counterparties to total trades.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trade_count: int = Field(
        ...,
        ge=2,
        description="Total number of trades involved in the pattern",
        examples=[5],
    )
    unique_addresses: int = Field(
        ...,
        ge=1,
        description="Number of unique addresses participating",
        examples=[2],
    )
    time_window_seconds: int = Field(
        ...,
        ge=0,
        le=86400,  # Max 24 hours
        description="Duration of suspicious activity in seconds",
        examples=[300],
    )
    round_trip_count: int | None = Field(
        default=None,
        ge=1,
        description="Number of complete buy-sell cycles (roundtrip type only)",
        examples=[3],
    )
    avg_trade_interval_seconds: float = Field(
        ...,
        ge=0.0,
        description="Average seconds between consecutive trades",
        examples=[60.5],
    )
    amount_similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score of trade amounts (1.0 = identical)",
        examples=[0.95],
    )
    volume_usd_total: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Total USD volume involved in the pattern",
        examples=["10000.00"],
    )
    counterparty_diversity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ratio of unique counterparties to total trades",
        examples=[0.2],
    )

    @field_validator("volume_usd_total")
    @classmethod
    def validate_volume(cls, v: str) -> str:
        """Validate that volume is a valid non-negative decimal string."""
        if not v:
            raise ValueError("Volume cannot be empty")
        try:
            val = Decimal(v)
            if val < 0:
                raise ValueError("Volume must be non-negative")
        except Exception as e:
            raise ValueError(f"Invalid decimal volume: {v}") from e
        return v


class WashTradeCandidate(BaseModel):
    """Represents a detected wash trading pattern candidate.

    Wash trading involves artificial transaction volume created by related
    addresses without genuine value transfer. This model captures various
    patterns including:

    1. Round-trip: Single address rapidly buying then selling the same asset
    2. Closed-loop: Multiple addresses transferring assets in a circular pattern
    3. Coordinated: Multiple addresses trading at suspiciously similar times/amounts

    Attributes:
        detection_type: Classification of the wash trading pattern.
        primary_address: Main address initiating the suspicious pattern.
        involved_addresses: All addresses participating in the pattern.
        pool: Liquidity pool address where the activity occurred.
        token_pair: Human-readable token pair identifier (e.g., "WETH/USDC").
        related_tx_hashes: Ordered list of transaction hashes in the pattern.
        start_timestamp: Unix timestamp of the first suspicious transaction.
        end_timestamp: Unix timestamp of the last suspicious transaction.
        metrics: Quantitative measurements of the suspicious pattern.
        confidence_score: Overall confidence in the detection (0.0 to 1.0).
        description: Human-readable explanation of the detected pattern.

    Example:
        >>> candidate = WashTradeCandidate(
        ...     detection_type=WashTradeType.ROUNDTRIP,
        ...     primary_address="0x1234...",
        ...     involved_addresses=["0x1234..."],
        ...     pool="0x88e6...",
        ...     token_pair="WETH/USDC",
        ...     related_tx_hashes=["0xabc...", "0xdef...", "0xghi..."],
        ...     start_timestamp=1769263200,
        ...     end_timestamp=1769263500,
        ...     metrics=WashTradeMetrics(
        ...         trade_count=6,
        ...         unique_addresses=1,
        ...         time_window_seconds=300,
        ...         round_trip_count=3,
        ...         avg_trade_interval_seconds=50.0,
        ...         amount_similarity_score=0.98,
        ...         volume_usd_total="15000.00",
        ...         counterparty_diversity=None,
        ...     ),
        ...     confidence_score=0.92,
        ...     description="Address executed 3 complete round-trip trades within 5 minutes "
        ...                "with highly consistent amounts ($5000 each), suggesting "
        ...                "artificial volume generation.",
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    detection_type: WashTradeType = Field(
        ...,
        description="Classification of the detected wash trading pattern",
        examples=[WashTradeType.ROUNDTRIP],
    )
    primary_address: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Primary address initiating the suspicious pattern",
        examples=["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"],
    )
    involved_addresses: list[str] = Field(
        ...,
        min_length=1,
        description="All addresses participating in the pattern (ordered by significance)",
        examples=[["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"]],
    )
    pool: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Liquidity pool address where activity occurred",
        examples=["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"],
    )
    token_pair: str | None = Field(
        default=None,
        max_length=50,
        description="Human-readable token pair (e.g., 'WETH/USDC')",
        examples=["WETH/USDC"],
    )
    related_tx_hashes: list[str] = Field(
        ...,
        min_length=2,
        description="Ordered list of transaction hashes comprising the pattern",
        examples=[
            [
                "0x0c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e832",
                "0x1c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e833",
            ]
        ],
    )
    start_timestamp: int = Field(
        ...,
        ge=1438214400,  # Ethereum genesis: July 30, 2015
        le=2000000000,  # Reasonable upper bound (year 2033)
        description="Unix timestamp of the first suspicious transaction",
        examples=[1769263200],
    )
    end_timestamp: int = Field(
        ...,
        ge=1438214400,
        le=2000000000,
        description="Unix timestamp of the last suspicious transaction",
        examples=[1769263500],
    )
    metrics: WashTradeMetrics = Field(
        ...,
        description="Quantitative measurements of the suspicious pattern",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the detection (0.0 to 1.0)",
        examples=[0.92],
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Human-readable explanation of the detected pattern",
        examples=[
            "Address executed 3 complete round-trip trades within 5 minutes "
            "with highly consistent amounts ($5000 each)"
        ],
    )

    @field_validator("primary_address", "pool")
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

    @field_validator("involved_addresses")
    @classmethod
    def validate_involved_addresses(cls, v: list[str]) -> list[str]:
        """Validate all involved addresses and normalize to lowercase."""
        if not v:
            raise ValueError("At least one involved address is required")
        validated = []
        for addr in v:
            if not addr:
                raise ValueError("Address cannot be empty")
            addr = addr.lower()
            if not ETH_ADDRESS_PATTERN.match(addr):
                raise ValueError(
                    f"Invalid Ethereum address in involved_addresses: {addr[:20]}... "
                    "Expected format: 0x followed by 40 hex characters"
                )
            validated.append(addr)
        return validated

    @field_validator("related_tx_hashes")
    @classmethod
    def validate_tx_hashes(cls, v: list[str]) -> list[str]:
        """Validate all transaction hashes and normalize to lowercase."""
        if len(v) < 2:
            raise ValueError("At least 2 transactions are required for wash trading pattern")
        validated = []
        for tx_hash in v:
            if not tx_hash:
                raise ValueError("Transaction hash cannot be empty")
            tx_hash = tx_hash.lower()
            if not tx_hash.startswith("0x") or len(tx_hash) != 66:
                raise ValueError(
                    f"Invalid transaction hash: {tx_hash[:20]}... "
                    "Expected format: 0x followed by 64 hex characters"
                )
            validated.append(tx_hash)
        return validated

    @field_validator("end_timestamp")
    @classmethod
    def validate_time_order(cls, v: int, info) -> int:
        """Validate that end_timestamp is not before start_timestamp."""
        # Note: Pydantic v2 passes data as info.data
        start = info.data.get("start_timestamp")
        if start is not None and v < start:
            raise ValueError(f"end_timestamp ({v}) must not be before start_timestamp ({start})")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert wash trade candidate to dictionary format.

        Returns:
            Dictionary representation suitable for DataFrame serialization.
        """
        return {
            "detection_type": self.detection_type.value,
            "primary_address": self.primary_address,
            "involved_addresses": ",".join(self.involved_addresses),
            "pool": self.pool,
            "token_pair": self.token_pair,
            "related_tx_hashes": ",".join(self.related_tx_hashes),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "trade_count": self.metrics.trade_count,
            "unique_addresses": self.metrics.unique_addresses,
            "time_window_seconds": self.metrics.time_window_seconds,
            "round_trip_count": self.metrics.round_trip_count,
            "avg_trade_interval_seconds": self.metrics.avg_trade_interval_seconds,
            "amount_similarity_score": self.metrics.amount_similarity_score,
            "volume_usd_total": self.metrics.volume_usd_total,
            "counterparty_diversity": self.metrics.counterparty_diversity,
            "confidence_score": self.confidence_score,
            "description": self.description,
        }

    @property
    def duration_seconds(self) -> int:
        """Calculate the duration of the suspicious activity."""
        return self.end_timestamp - self.start_timestamp

    @property
    def avg_trade_size_usd(self) -> Decimal:
        """Calculate average trade size in USD."""
        try:
            total = Decimal(self.metrics.volume_usd_total)
            return total / Decimal(self.metrics.trade_count)
        except (ValueError, ZeroDivisionError):
            return Decimal("0")

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence detection."""
        return self.confidence_score >= 0.8

    @property
    def is_single_address(self) -> bool:
        """Check if pattern involves only one address (roundtrip)."""
        return self.metrics.unique_addresses == 1


@dataclass(frozen=True)
class WashTradeDetectionResult:
    """Result of wash trading detection analysis.

    Attributes:
        candidates: List of detected wash trading candidates.
        total_swaps_analyzed: Total number of swaps processed.
        pools_analyzed: Number of unique pools analyzed.
        time_window_seconds: Time window used for detection.
        detection_errors: List of any errors encountered during detection.
        detection_type_filter: Type of detection performed (if filtered).
    """

    candidates: tuple[WashTradeCandidate, ...]
    total_swaps_analyzed: int
    pools_analyzed: int
    time_window_seconds: int
    detection_errors: list[str] = field(default_factory=list)
    detection_type_filter: WashTradeType | None = None

    @property
    def total_candidates(self) -> int:
        """Return total number of detected candidates."""
        return len(self.candidates)

    @property
    def high_confidence_candidates(self) -> list[WashTradeCandidate]:
        """Return candidates with confidence >= 0.8."""
        return [c for c in self.candidates if c.confidence_score >= 0.8]

    @property
    def unique_primary_addresses(self) -> set[str]:
        """Return set of unique primary addresses across all candidates."""
        return {c.primary_address for c in self.candidates}

    @property
    def unique_pools(self) -> set[str]:
        """Return set of unique pools where wash trading was detected."""
        return {c.pool for c in self.candidates}

    @property
    def total_volume_usd(self) -> Decimal:
        """Calculate total USD volume of all detected wash trades."""
        total = Decimal("0")
        for candidate in self.candidates:
            try:
                total += Decimal(candidate.metrics.volume_usd_total)
            except Exception:
                continue
        return total

    @property
    def roundtrip_count(self) -> int:
        """Count candidates that are roundtrip patterns."""
        return sum(1 for c in self.candidates if c.detection_type == WashTradeType.ROUNDTRIP)

    @property
    def closed_loop_count(self) -> int:
        """Count candidates that are closed-loop patterns."""
        return sum(1 for c in self.candidates if c.detection_type == WashTradeType.CLOSED_LOOP)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for reporting."""
        return {
            "total_candidates": self.total_candidates,
            "high_confidence_count": len(self.high_confidence_candidates),
            "total_swaps_analyzed": self.total_swaps_analyzed,
            "pools_analyzed": self.pools_analyzed,
            "pools_with_wash_trading": len(self.unique_pools),
            "time_window_seconds": self.time_window_seconds,
            "unique_addresses_involved": len(self.unique_primary_addresses),
            "total_volume_usd": str(self.total_volume_usd),
            "roundtrip_count": self.roundtrip_count,
            "closed_loop_count": self.closed_loop_count,
            "detection_errors": self.detection_errors,
            "detection_type_filter": (
                self.detection_type_filter.value if self.detection_type_filter else None
            ),
        }


__all__ = [
    "WashTradeType",
    "WashTradeMetrics",
    "WashTradeCandidate",
    "WashTradeDetectionResult",
]
