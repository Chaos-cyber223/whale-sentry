"""Pydantic models for Uniswap V3 swap data.

This module provides type-safe data models for representing swap events
from Uniswap V3 pools, with built-in validation for Ethereum addresses
and numeric fields.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Ethereum address regex pattern
ETH_ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
# Transaction hash regex pattern
TX_HASH_PATTERN = re.compile(r"^0x[a-fA-F0-9]{64}$")


class SwapEvent(BaseModel):
    """Represents a single Uniswap V3 swap event.

    Attributes:
        timestamp: Unix timestamp of the block containing the swap (seconds).
        block_number: Ethereum block number where the swap occurred.
        tx_hash: Transaction hash (66-character hex string with 0x prefix).
        pool: Liquidity pool contract address (42-character hex string with 0x prefix).
        sender: Address that initiated the swap.
        recipient: Address that received the swap output.
        amount0: Token0 amount as string (can be negative for outgoing).
        amount1: Token1 amount as string (can be negative for outgoing).
        amount_usd: USD value of the swap as string for precision.
        tick: Current tick of the pool after the swap.
        sqrt_price_x96: Square root price scaled by 2^96 as string.

    Example:
        >>> swap = SwapEvent(
        ...     timestamp=1769263223,
        ...     block_number=24305113,
        ...     tx_hash="0x0c8d...4e832",
        ...     pool="0x88e6...f5640",
        ...     sender="0x66a9...ba8af",
        ...     recipient="0xf99e...00101",
        ...     amount0="-423.467818",
        ...     amount1="0.143747594817866756",
        ...     amount_usd="424.108780290392901693",
        ...     tick=196433,
        ...     sqrt_price_x96="1459360160181669463152813566785694",
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    timestamp: int = Field(
        ...,
        ge=1438214400,  # Ethereum genesis: July 30, 2015
        le=2000000000,  # Reasonable upper bound (year 2033)
        description="Unix timestamp of the block containing the swap",
        examples=[1769263223],
    )
    block_number: int = Field(
        ...,
        ge=0,
        le=100_000_000,  # Reasonable upper bound
        description="Ethereum block number",
        examples=[24305113],
    )
    tx_hash: str = Field(
        ...,
        min_length=66,
        max_length=66,
        description="Transaction hash (0x + 64 hex characters)",
        examples=["0x0c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e832"],
    )
    pool: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Liquidity pool contract address",
        examples=["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"],
    )
    sender: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Address that initiated the swap",
        examples=["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"],
    )
    recipient: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Address that received the swap output",
        examples=["0xf99eb958f923b2c74127af0f72a75f458f500101"],
    )
    amount0: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Token0 amount (can be negative for outgoing tokens)",
        examples=["-423.467818", "1000.000000"],
    )
    amount1: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Token1 amount (can be negative for outgoing tokens)",
        examples=["0.143747594817866756", "-50.123456"],
    )
    amount_usd: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="USD value of the swap as string for precision",
        examples=["424.108780290392901693"],
    )
    tick: int = Field(
        ...,
        ge=-887272,  # Uniswap V3 min tick
        le=887272,   # Uniswap V3 max tick
        description="Current tick of the pool after the swap",
        examples=[196433],
    )
    sqrt_price_x96: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Square root price scaled by 2^96",
        examples=["1459360160181669463152813566785694"],
    )

    @field_validator("tx_hash")
    @classmethod
    def validate_tx_hash(cls, v: str) -> str:
        """Validate transaction hash format and normalize to lowercase."""
        if not v:
            raise ValueError("Transaction hash cannot be empty")
        v = v.lower()
        if not TX_HASH_PATTERN.match(v):
            raise ValueError(
                f"Invalid transaction hash: {v[:20]}... "
                "Expected format: 0x followed by 64 hex characters"
            )
        return v

    @field_validator("pool", "sender", "recipient")
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

    @field_validator("amount0", "amount1", "amount_usd")
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

    @field_validator("sqrt_price_x96")
    @classmethod
    def validate_sqrt_price(cls, v: str) -> str:
        """Validate sqrtPriceX96 is a positive integer string."""
        if not v:
            raise ValueError("sqrtPriceX96 cannot be empty")
        try:
            val = int(v)
            if val < 0:
                raise ValueError("sqrtPriceX96 must be non-negative")
        except ValueError as e:
            msg = f"Invalid sqrtPriceX96: {v}. Must be a non-negative integer"
            raise ValueError(msg) from e
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert swap event to dictionary format suitable for DataFrames."""
        return {
            "timestamp": self.timestamp,
            "block_number": self.block_number,
            "tx_hash": self.tx_hash,
            "pool": self.pool,
            "sender": self.sender,
            "recipient": self.recipient,
            "amount0": self.amount0,
            "amount1": self.amount1,
            "amount_usd": self.amount_usd,
            "tick": self.tick,
            "sqrtPriceX96": self.sqrt_price_x96,
        }

    @property
    def is_buy(self) -> bool:
        """Determine if this swap is a buy (positive amount1, negative amount0)."""
        try:
            return Decimal(self.amount0) < 0 and Decimal(self.amount1) > 0
        except Exception:
            return False

    @property
    def is_sell(self) -> bool:
        """Determine if this swap is a sell (positive amount0, negative amount1)."""
        try:
            return Decimal(self.amount0) > 0 and Decimal(self.amount1) < 0
        except Exception:
            return False

    @property
    def usd_value(self) -> Decimal:
        """Get the USD value as a Decimal."""
        return Decimal(self.amount_usd)

    @property
    def abs_amount0(self) -> Decimal:
        """Get the absolute value of amount0."""
        return abs(Decimal(self.amount0))

    @property
    def abs_amount1(self) -> Decimal:
        """Get the absolute value of amount1."""
        return abs(Decimal(self.amount1))


class SwapDataFrame(BaseModel):
    """Container for multiple swap events with validation statistics.

    Attributes:
        swaps: List of validated SwapEvent objects.
        total_count: Total number of swaps in the collection.
        unique_pools: Set of unique pool addresses.
        unique_senders: Set of unique sender addresses.

    Example:
        >>> df = SwapDataFrame.from_records([{"timestamp": 123, ...}, ...])
        >>> print(f"Total swaps: {df.total_count}")
    """

    model_config = ConfigDict(frozen=True)

    swaps: tuple[SwapEvent, ...] = Field(
        ...,
        description="Tuple of validated swap events",
    )

    @property
    def total_count(self) -> int:
        """Return the total number of swaps."""
        return len(self.swaps)

    @property
    def unique_pools(self) -> set[str]:
        """Return set of unique pool addresses."""
        return {swap.pool for swap in self.swaps}

    @property
    def unique_senders(self) -> set[str]:
        """Return set of unique sender addresses."""
        return {swap.sender for swap in self.swaps}

    @property
    def unique_recipients(self) -> set[str]:
        """Return set of unique recipient addresses."""
        return {swap.recipient for swap in self.swaps}

    @property
    def time_range(self) -> tuple[int, int] | None:
        """Return (min_timestamp, max_timestamp) or None if empty."""
        if not self.swaps:
            return None
        timestamps = [swap.timestamp for swap in self.swaps]
        return (min(timestamps), max(timestamps))

    @property
    def block_range(self) -> tuple[int, int] | None:
        """Return (min_block, max_block) or None if empty."""
        if not self.swaps:
            return None
        blocks = [swap.block_number for swap in self.swaps]
        return (min(blocks), max(blocks))

    def to_records(self) -> list[dict[str, Any]]:
        """Convert all swaps to list of dictionaries."""
        return [swap.to_dict() for swap in self.swaps]

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> SwapDataFrame:
        """Create SwapDataFrame from list of record dictionaries.

        Args:
            records: List of dictionaries containing swap data.

        Returns:
            SwapDataFrame containing validated SwapEvent objects.

        Raises:
            ValidationError: If any record fails validation.
        """
        swaps = tuple(SwapEvent(**record) for record in records)
        return cls(swaps=swaps)

    def filter_by_pool(self, pool_address: str) -> SwapDataFrame:
        """Filter swaps by pool address.

        Args:
            pool_address: The pool address to filter by.

        Returns:
            New SwapDataFrame containing only swaps from the specified pool.
        """
        filtered = tuple(swap for swap in self.swaps if swap.pool == pool_address.lower())
        return SwapDataFrame(swaps=filtered)

    def filter_by_sender(self, sender_address: str) -> SwapDataFrame:
        """Filter swaps by sender address.

        Args:
            sender_address: The sender address to filter by.

        Returns:
            New SwapDataFrame containing only swaps from the specified sender.
        """
        filtered = tuple(swap for swap in self.swaps if swap.sender == sender_address.lower())
        return SwapDataFrame(swaps=filtered)

    def filter_by_value_range(
        self, min_usd: Decimal | None = None, max_usd: Decimal | None = None
    ) -> SwapDataFrame:
        """Filter swaps by USD value range.

        Args:
            min_usd: Minimum USD value (inclusive).
            max_usd: Maximum USD value (inclusive).

        Returns:
            New SwapDataFrame containing only swaps within the value range.
        """
        filtered = self.swaps
        if min_usd is not None:
            filtered = tuple(s for s in filtered if s.usd_value >= min_usd)
        if max_usd is not None:
            filtered = tuple(s for s in filtered if s.usd_value <= max_usd)
        return SwapDataFrame(swaps=filtered)


__all__ = [
    "SwapEvent",
    "SwapDataFrame",
    "ETH_ADDRESS_PATTERN",
    "TX_HASH_PATTERN",
]
