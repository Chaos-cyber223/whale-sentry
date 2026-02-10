"""Performance tests for optimized sandwich attack detection.

This module benchmarks the performance difference between the original
O(n³) algorithm and the optimized O(n log n) algorithm.
"""

from __future__ import annotations

import time
from decimal import Decimal

import pandas as pd
import pytest

from whalesentry.detection.sandwich import (
    detect_sandwich_attacks,
    detect_sandwich_attacks_optimized,
)


def generate_synthetic_swaps(
    n_transactions: int,
    n_attackers: int = 5,
    n_victims: int = 20,
    n_pools: int = 3,
    sandwich_probability: float = 0.1,
) -> pd.DataFrame:
    """Generate synthetic swap data for performance testing.

    Args:
        n_transactions: Total number of transactions to generate.
        n_attackers: Number of unique attacker addresses.
        n_victims: Number of unique victim addresses.
        n_pools: Number of unique pools.
        sandwich_probability: Probability of generating a sandwich pattern.

    Returns:
        DataFrame with synthetic swap data.
    """
    import random

    random.seed(42)  # For reproducibility

    attackers = [f"0x{str(i).zfill(40)}" for i in range(n_attackers)]
    victims = [f"0x{str(i + 100).zfill(40)}" for i in range(n_victims)]
    pools = [f"0x{str(i + 1000).zfill(40)}" for i in range(n_pools)]

    data = []
    base_time = 1769263223

    for i in range(n_transactions):
        pool = random.choice(pools)
        timestamp = base_time + i

        # Decide if this is part of a sandwich pattern
        if random.random() < sandwich_probability and i < n_transactions - 2:
            # Generate a sandwich: attacker -> victim -> attacker
            attacker = random.choice(attackers)
            victim = random.choice(victims)

            # Front-run: attacker buys
            data.append({
                "timestamp": timestamp,
                "block_number": 24305113 + i,
                "tx_hash": f"0x{str(i).zfill(64)}",
                "pool": pool,
                "sender": attacker,
                "recipient": attacker,
                "amount0": "-1000",
                "amount1": "1.0",
                "amount_usd": "1000.00",
                "tick": 196433,
                "sqrtPriceX96": "1459360160181669463152813566785694",
            })

            # Victim transaction
            data.append({
                "timestamp": timestamp + 1,
                "block_number": 24305113 + i + 1,
                "tx_hash": f"0x{str(i + 1).zfill(64)}",
                "pool": pool,
                "sender": victim,
                "recipient": victim,
                "amount0": "5000",
                "amount1": "-5.0",
                "amount_usd": "5000.00",
                "tick": 196434,
                "sqrtPriceX96": "1459360160181669463152813566785694",
            })

            # Back-run: attacker sells
            data.append({
                "timestamp": timestamp + 2,
                "block_number": 24305113 + i + 2,
                "tx_hash": f"0x{str(i + 2).zfill(64)}",
                "pool": pool,
                "sender": attacker,
                "recipient": attacker,
                "amount0": "1100",
                "amount1": "-1.1",
                "amount_usd": "1100.00",
                "tick": 196435,
                "sqrtPriceX96": "1459360160181669463152813566785694",
            })

            i += 2  # Skip the next 2 transactions
        else:
            # Normal transaction
            sender = random.choice(victims + attackers)
            is_buy = random.choice([True, False])
            amount0 = "-1000" if is_buy else "1000"
            amount1 = "1.0" if is_buy else "-1.0"

            data.append({
                "timestamp": timestamp,
                "block_number": 24305113 + i,
                "tx_hash": f"0x{str(i).zfill(64)}",
                "pool": pool,
                "sender": sender,
                "recipient": sender,
                "amount0": amount0,
                "amount1": amount1,
                "amount_usd": "1000.00",
                "tick": 196433,
                "sqrtPriceX96": "1459360160181669463152813566785694",
            })

    return pd.DataFrame(data)


class TestPerformanceComparison:
    """Performance comparison tests between original and optimized algorithms."""

    @pytest.mark.parametrize("n_transactions", [100, 500, 1000])
    def test_performance_comparison(self, n_transactions: int) -> None:
        """Compare performance between original and optimized algorithms."""
        df = generate_synthetic_swaps(n_transactions)

        # Test original algorithm
        start_time = time.time()
        result_original = detect_sandwich_attacks(
            df,
            time_window_seconds=60,
            min_usd_value=Decimal("100"),
            amount_similarity_threshold=0.5,
        )
        original_time = time.time() - start_time

        # Test optimized algorithm
        start_time = time.time()
        result_optimized = detect_sandwich_attacks_optimized(
            df,
            time_window_seconds=60,
            min_usd_value=Decimal("100"),
            amount_similarity_threshold=0.5,
        )
        optimized_time = time.time() - start_time

        # Print results
        print(f"\n{n_transactions} transactions:")
        print(f"  Original:   {original_time:.4f}s, found {result_original.total_candidates} candidates")
        print(f"  Optimized:  {optimized_time:.4f}s, found {result_optimized.total_candidates} candidates")
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        print(f"  Speedup:    {speedup:.2f}x")

        # Both should find similar number of candidates
        # (may differ slightly due to algorithm differences)
        assert abs(result_original.total_candidates - result_optimized.total_candidates) <= n_transactions // 10

        # Optimized should be faster for larger datasets
        if n_transactions >= 500:
            assert optimized_time < original_time, (
                f"Optimized algorithm should be faster for {n_transactions} transactions"
            )

    def test_optimized_correctness(self) -> None:
        """Verify optimized algorithm produces correct results."""
        attacker = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        victim = "0x51c72848c68a965f66fa7a88855f9f7784502a7f"
        pool = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"

        df = pd.DataFrame({
            "timestamp": [1769263223, 1769263224, 1769263225],
            "block_number": [24305113, 24305114, 24305115],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}", f"0x{'c' * 64}"],
            "pool": [pool] * 3,
            "sender": [attacker, victim, attacker],
            "recipient": [attacker, victim, attacker],
            "amount0": ["-1000", "5000", "1100"],
            "amount1": ["1.0", "-5.0", "-1.1"],
            "amount_usd": ["1000.00", "5000.00", "1100.00"],
            "tick": [196433, 196434, 196435],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
        })

        result = detect_sandwich_attacks_optimized(df, time_window_seconds=60)

        assert result.total_candidates >= 1
        assert result.total_swaps_analyzed == 3
        assert result.pools_analyzed == 1

        if result.total_candidates > 0:
            candidate = result.candidates[0]
            assert candidate.attacker == attacker
            assert candidate.victim_tx == f"0x{'b' * 64}"

    def test_large_dataset_performance(self) -> None:
        """Test optimized algorithm with large dataset."""
        df = generate_synthetic_swaps(2000, n_pools=5)

        start_time = time.time()
        result = detect_sandwich_attacks_optimized(
            df,
            time_window_seconds=60,
            min_usd_value=Decimal("100"),
            amount_similarity_threshold=0.5,
        )
        elapsed_time = time.time() - start_time

        print(f"\n2000 transactions processed in {elapsed_time:.4f}s")
        print(f"Found {result.total_candidates} candidates")

        # Should complete in reasonable time (< 30 seconds)
        assert elapsed_time < 30.0, "Optimized algorithm should handle 2000 transactions quickly"

    def test_empty_dataframe_optimized(self) -> None:
        """Test optimized algorithm with empty DataFrame."""
        df = pd.DataFrame(
            columns=["timestamp", "pool", "sender", "tx_hash", "amount0", "amount1", "amount_usd"]
        )
        result = detect_sandwich_attacks_optimized(df, time_window_seconds=60)
        assert result.total_candidates == 0
        assert len(result.detection_errors) > 0

    def test_missing_columns_optimized(self) -> None:
        """Test optimized algorithm with missing columns."""
        df = pd.DataFrame({"timestamp": [1769263223]})
        result = detect_sandwich_attacks_optimized(df, time_window_seconds=60)
        assert result.total_candidates == 0
        assert len(result.detection_errors) > 0
