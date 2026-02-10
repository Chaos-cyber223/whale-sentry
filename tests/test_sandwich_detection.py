"""Unit tests for sandwich attack detection.

This module contains comprehensive tests for:
- SandwichCandidate Pydantic model validation
- DetectionResult dataclass functionality
- detect_sandwich_attacks function with various scenarios
- Edge cases and false positive prevention
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from pydantic import ValidationError

from whalesentry.detection.sandwich import (
    DetectionResult,
    SandwichCandidate,
    _calculate_confidence,
    _detect_sandwiches_in_pool,
    candidates_to_dataframe,
    detect_sandwich_attacks,
)


class TestSandwichCandidate:
    """Test suite for SandwichCandidate Pydantic model."""

    @pytest.fixture
    def valid_candidate_dict(self) -> dict:
        """Return a valid sandwich candidate dictionary."""
        return {
            "attacker": "0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
            "victim_tx": "0x0c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e832",
            "front_run_tx": "0x1c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e833",
            "back_run_tx": "0x2c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e834",
            "pool": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
            "front_run_timestamp": 1769263223,
            "victim_timestamp": 1769263224,
            "back_run_timestamp": 1769263225,
            "front_run_amount_usd": "1000.00",
            "victim_amount_usd": "5000.00",
            "back_run_amount_usd": "1100.00",
            "profit_estimate_usd": "100.00",
            "confidence_score": 0.85,
        }

    def test_valid_candidate_creation(self, valid_candidate_dict: dict) -> None:
        """Test creating a valid SandwichCandidate."""
        candidate = SandwichCandidate(**valid_candidate_dict)
        assert candidate.attacker == "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        assert candidate.confidence_score == 0.85

    def test_address_normalization(self, valid_candidate_dict: dict) -> None:
        """Test that addresses are normalized to lowercase."""
        valid_candidate_dict["attacker"] = valid_candidate_dict["attacker"].upper()
        valid_candidate_dict["pool"] = valid_candidate_dict["pool"].upper()

        candidate = SandwichCandidate(**valid_candidate_dict)
        assert candidate.attacker == valid_candidate_dict["attacker"].lower()
        assert candidate.pool == valid_candidate_dict["pool"].lower()

    def test_invalid_attacker_address(self, valid_candidate_dict: dict) -> None:
        """Test validation of invalid attacker address."""
        valid_candidate_dict["attacker"] = "0xinvalid"
        with pytest.raises(ValidationError):
            SandwichCandidate(**valid_candidate_dict)

    def test_invalid_tx_hash(self, valid_candidate_dict: dict) -> None:
        """Test validation of invalid transaction hash."""
        valid_candidate_dict["victim_tx"] = "invalid_hash"
        with pytest.raises(ValidationError):
            SandwichCandidate(**valid_candidate_dict)

    def test_invalid_confidence_score(self, valid_candidate_dict: dict) -> None:
        """Test validation of confidence score out of range."""
        valid_candidate_dict["confidence_score"] = 1.5
        with pytest.raises(ValidationError):
            SandwichCandidate(**valid_candidate_dict)

    def test_negative_confidence_score(self, valid_candidate_dict: dict) -> None:
        """Test validation of negative confidence score."""
        valid_candidate_dict["confidence_score"] = -0.1
        with pytest.raises(ValidationError):
            SandwichCandidate(**valid_candidate_dict)

    def test_invalid_amount(self, valid_candidate_dict: dict) -> None:
        """Test validation of non-numeric amount."""
        valid_candidate_dict["profit_estimate_usd"] = "not_a_number"
        with pytest.raises(ValidationError):
            SandwichCandidate(**valid_candidate_dict)

    def test_to_dict_method(self, valid_candidate_dict: dict) -> None:
        """Test conversion to dictionary."""
        candidate = SandwichCandidate(**valid_candidate_dict)
        result = candidate.to_dict()
        assert result["attacker"] == valid_candidate_dict["attacker"].lower()
        assert result["confidence_score"] == 0.85


class TestDetectionResult:
    """Test suite for DetectionResult dataclass."""

    @pytest.fixture
    def sample_candidates(self) -> list[SandwichCandidate]:
        """Return list of sample candidates."""
        return [
            SandwichCandidate(
                attacker="0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
                victim_tx=f"0x{'a' * 64}",
                front_run_tx=f"0x{'b' * 64}",
                back_run_tx=f"0x{'c' * 64}",
                pool="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                front_run_timestamp=1769263223,
                victim_timestamp=1769263224,
                back_run_timestamp=1769263225,
                front_run_amount_usd="1000.00",
                victim_amount_usd="5000.00",
                back_run_amount_usd="1100.00",
                profit_estimate_usd="100.00",
                confidence_score=0.85,
            ),
            SandwichCandidate(
                attacker="0x51c72848c68a965f66fa7a88855f9f7784502a7f",
                victim_tx=f"0x{'d' * 64}",
                front_run_tx=f"0x{'e' * 64}",
                back_run_tx=f"0x{'f' * 64}",
                pool="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                front_run_timestamp=1769263230,
                victim_timestamp=1769263231,
                back_run_timestamp=1769263232,
                front_run_amount_usd="2000.00",
                victim_amount_usd="8000.00",
                back_run_amount_usd="2200.00",
                profit_estimate_usd="200.00",
                confidence_score=0.90,
            ),
        ]

    def test_total_candidates(self, sample_candidates: list[SandwichCandidate]) -> None:
        """Test total_candidates property."""
        result = DetectionResult(
            candidates=tuple(sample_candidates),
            total_swaps_analyzed=100,
            pools_analyzed=5,
            time_window_seconds=60,
        )
        assert result.total_candidates == 2

    def test_unique_attackers(self, sample_candidates: list[SandwichCandidate]) -> None:
        """Test unique_attackers property."""
        result = DetectionResult(
            candidates=tuple(sample_candidates),
            total_swaps_analyzed=100,
            pools_analyzed=5,
            time_window_seconds=60,
        )
        assert len(result.unique_attackers) == 2

    def test_high_confidence_candidates(self, sample_candidates: list[SandwichCandidate]) -> None:
        """Test high_confidence_candidates property."""
        result = DetectionResult(
            candidates=tuple(sample_candidates),
            total_swaps_analyzed=100,
            pools_analyzed=5,
            time_window_seconds=60,
        )
        # Both candidates have confidence >= 0.8
        assert len(result.high_confidence_candidates) == 2

    def test_to_dict(self, sample_candidates: list[SandwichCandidate]) -> None:
        """Test to_dict method."""
        result = DetectionResult(
            candidates=tuple(sample_candidates),
            total_swaps_analyzed=100,
            pools_analyzed=5,
            time_window_seconds=60,
        )
        d = result.to_dict()
        assert d["total_candidates"] == 2
        assert d["total_swaps_analyzed"] == 100
        assert d["unique_attackers"] == 2


class TestDetectSandwichAttacks:
    """Test suite for detect_sandwich_attacks function."""

    @pytest.fixture
    def create_sandwich_df(self) -> callable:
        """Factory fixture to create DataFrames with sandwich patterns."""
        def _create(
            attacker: str, victim: str, pool: str, base_time: int = 1769263223
        ) -> pd.DataFrame:
            return pd.DataFrame({
                "timestamp": [base_time, base_time + 1, base_time + 2],
                "block_number": [24305113, 24305114, 24305115],
                "tx_hash": [
                    f"0x{'a' * 64}",
                    f"0x{'b' * 64}",
                    f"0x{'c' * 64}",
                ],
                "pool": [pool] * 3,
                "sender": [attacker, victim, attacker],
                "recipient": [attacker, victim, attacker],
                "amount0": ["-1000", "5000", "1100"],  # Buy, victim buy, sell
                "amount1": ["1.0", "-5.0", "-1.1"],
                "amount_usd": ["1000.00", "5000.00", "1100.00"],
                "tick": [196433, 196434, 196435],
                "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
            })
        return _create

    def test_detect_synthetic_sandwich(self, create_sandwich_df: callable) -> None:
        """Test detection of a clear sandwich attack pattern."""
        attacker = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        victim = "0x51c72848c68a965f66fa7a88855f9f7784502a7f"
        pool = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"

        df = create_sandwich_df(attacker, victim, pool)
        result = detect_sandwich_attacks(df, time_window_seconds=60)

        assert result.total_candidates >= 1
        assert result.total_swaps_analyzed == 3
        assert result.pools_analyzed == 1

    def test_no_false_positives_normal_trading(self) -> None:
        """Test that normal trading doesn't trigger false positives."""
        # Different senders, no sandwich pattern
        df = pd.DataFrame({
            "timestamp": [1769263223, 1769263224, 1769263225],
            "block_number": [24305113, 24305114, 24305115],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}", f"0x{'c' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 3,
            "sender": [
                "0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
                "0x51c72848c68a965f66fa7a88855f9f7784502a7f",
                "0x71c72848c68a965f66fa7a88855f9f7784502a8a",
            ],
            "recipient": [
                "0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
                "0x51c72848c68a965f66fa7a88855f9f7784502a7f",
                "0x71c72848c68a965f66fa7a88855f9f7784502a8a",
            ],
            "amount0": ["-1000", "2000", "-500"],
            "amount1": ["1.0", "-2.0", "0.5"],
            "amount_usd": ["1000.00", "2000.00", "500.00"],
            "tick": [196433, 196434, 196435],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
        })

        result = detect_sandwich_attacks(df, time_window_seconds=60)
        assert result.total_candidates == 0

    def test_same_address_legitimate_trading(self) -> None:
        """Test that same address trading doesn't trigger false positives."""
        # Same sender for all transactions - no victim
        df = pd.DataFrame({
            "timestamp": [1769263223, 1769263224, 1769263225],
            "block_number": [24305113, 24305114, 24305115],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}", f"0x{'c' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 3,
            "sender": ["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"] * 3,
            "recipient": ["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"] * 3,
            "amount0": ["-1000", "-500", "2000"],
            "amount1": ["1.0", "0.5", "-2.0"],
            "amount_usd": ["1000.00", "500.00", "2000.00"],
            "tick": [196433, 196434, 196435],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
        })

        result = detect_sandwich_attacks(df, time_window_seconds=60)
        # Should not detect sandwiches when attacker and victim are the same
        assert result.total_candidates == 0

    def test_no_reversal_no_sandwich(self) -> None:
        """Test that buy-buy-sell pattern doesn't trigger sandwich detection."""
        attacker = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        victim = "0x51c72848c68a965f66fa7a88855f9f7784502a7f"

        # Buy, buy, buy - no reversal
        df = pd.DataFrame({
            "timestamp": [1769263223, 1769263224, 1769263225],
            "block_number": [24305113, 24305114, 24305115],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}", f"0x{'c' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 3,
            "sender": [attacker, victim, attacker],
            "recipient": [attacker, victim, attacker],
            "amount0": ["-1000", "-500", "-200"],  # All buys (negative amount0)
            "amount1": ["1.0", "0.5", "0.2"],
            "amount_usd": ["1000.00", "500.00", "200.00"],
            "tick": [196433, 196434, 196435],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
        })

        result = detect_sandwich_attacks(df, time_window_seconds=60)
        assert result.total_candidates == 0

    def test_time_window_exceeded(self, create_sandwich_df: callable) -> None:
        """Test that transactions outside time window are not detected."""
        attacker = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        victim = "0x51c72848c68a965f66fa7a88855f9f7784502a7f"
        pool = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"

        df = create_sandwich_df(attacker, victim, pool, base_time=1769263223)
        # Use a very small time window
        result = detect_sandwich_attacks(df, time_window_seconds=1)
        assert result.total_candidates == 0

    def test_amount_similarity_threshold(self, create_sandwich_df: callable) -> None:
        """Test that dissimilar amounts don't trigger detection."""
        attacker = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        victim = "0x51c72848c68a965f66fa7a88855f9f7784502a7f"

        # Front-run $100, back-run $1000 - too different
        df = pd.DataFrame({
            "timestamp": [1769263223, 1769263224, 1769263225],
            "block_number": [24305113, 24305114, 24305115],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}", f"0x{'c' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 3,
            "sender": [attacker, victim, attacker],
            "recipient": [attacker, victim, attacker],
            "amount0": ["-100", "5000", "1000"],  # Buy, victim, sell
            "amount1": ["0.1", "-5.0", "-1.0"],
            "amount_usd": ["100.00", "5000.00", "1000.00"],
            "tick": [196433, 196434, 196435],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
        })

        # With high similarity threshold, should not detect
        result = detect_sandwich_attacks(
            df, time_window_seconds=60, amount_similarity_threshold=0.8
        )
        assert result.total_candidates == 0

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(
            columns=["timestamp", "pool", "sender", "tx_hash", "amount0", "amount1", "amount_usd"]
        )
        result = detect_sandwich_attacks(df, time_window_seconds=60)
        assert result.total_candidates == 0
        assert len(result.detection_errors) > 0

    def test_missing_columns(self) -> None:
        """Test handling of DataFrame with missing columns."""
        df = pd.DataFrame({"timestamp": [1769263223]})
        result = detect_sandwich_attacks(df, time_window_seconds=60)
        assert result.total_candidates == 0
        assert len(result.detection_errors) > 0
        assert "Missing required columns" in result.detection_errors[0]

    def test_min_usd_filter(self) -> None:
        """Test that min_usd_value filter works correctly."""
        attacker = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        victim = "0x51c72848c68a965f66fa7a88855f9f7784502a7f"

        # Small transactions
        df = pd.DataFrame({
            "timestamp": [1769263223, 1769263224, 1769263225],
            "block_number": [24305113, 24305114, 24305115],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}", f"0x{'c' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 3,
            "sender": [attacker, victim, attacker],
            "recipient": [attacker, victim, attacker],
            "amount0": ["-10", "50", "11"],
            "amount1": ["0.01", "-0.05", "-0.011"],
            "amount_usd": ["10.00", "50.00", "11.00"],
            "tick": [196433, 196434, 196435],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
        })

        # With high min_usd, should not detect
        result = detect_sandwich_attacks(df, time_window_seconds=60, min_usd_value=Decimal("100"))
        assert result.total_candidates == 0


class TestCalculateConfidence:
    """Test suite for confidence calculation."""

    def test_high_confidence_short_time(self) -> None:
        """Test high confidence for short time window."""
        confidence = _calculate_confidence(
            victim_front_time=1,
            total_time=2,
            amount_similarity=0.9,
            victim_amount=5000,
            front_amount=2500,
        )
        assert confidence >= 0.8

    def test_low_confidence_long_time(self) -> None:
        """Test lower confidence for long time window."""
        confidence = _calculate_confidence(
            victim_front_time=50,
            total_time=60,
            amount_similarity=0.5,
            victim_amount=500,
            front_amount=50,
        )
        assert confidence < 0.7

    def test_confidence_bounds(self) -> None:
        """Test that confidence is always between 0 and 1."""
        confidence = _calculate_confidence(
            victim_front_time=100,
            total_time=200,
            amount_similarity=0.1,
            victim_amount=100,
            front_amount=1,
        )
        assert 0.0 <= confidence <= 1.0


class TestCandidatesToDataFrame:
    """Test suite for candidates_to_dataframe function."""

    def test_empty_list(self) -> None:
        """Test conversion of empty candidate list."""
        df = candidates_to_dataframe([])
        assert len(df) == 0
        assert "attacker" in df.columns

    def test_valid_candidates(self) -> None:
        """Test conversion of valid candidates."""
        candidates = [
            SandwichCandidate(
                attacker="0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
                victim_tx=f"0x{'a' * 64}",
                front_run_tx=f"0x{'b' * 64}",
                back_run_tx=f"0x{'c' * 64}",
                pool="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                front_run_timestamp=1769263223,
                victim_timestamp=1769263224,
                back_run_timestamp=1769263225,
                front_run_amount_usd="1000.00",
                victim_amount_usd="5000.00",
                back_run_amount_usd="1100.00",
                profit_estimate_usd="100.00",
                confidence_score=0.85,
            )
        ]
        df = candidates_to_dataframe(candidates)
        assert len(df) == 1
        assert df["confidence_score"].iloc[0] == 0.85


class TestPoolDetection:
    """Test suite for _detect_sandwiches_in_pool function."""

    def test_insufficient_transactions(self) -> None:
        """Test that pools with < 3 transactions return empty."""
        df = pd.DataFrame({
            "timestamp": [1769263223, 1769263224],
            "sender": ["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"] * 2,
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 2,
            "amount_usd_numeric": [1000.0, 2000.0],
            "amount0_numeric": [-1000.0, 2000.0],
            "amount1_numeric": [1.0, -2.0],
            "is_buy": [True, False],
            "is_sell": [False, True],
        })

        candidates = _detect_sandwiches_in_pool(df, 60, 0.5)
        assert len(candidates) == 0

    def test_multiple_victims_same_attacker(self) -> None:
        """Test detection of multiple victims by same attacker."""
        attacker = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        victim1 = "0x51c72848c68a965f66fa7a88855f9f7784502a7f"
        victim2 = "0x71c72848c68a965f66fa7a88855f9f7784502a8a"

        # Two separate sandwich patterns:
        # Pattern 1: T0-T1-T2 (attacker sandwiches victim1)
        # Pattern 2: T3-T4-T5 (attacker sandwiches victim2)
        df = pd.DataFrame({
            "timestamp": [1769263220, 1769263221, 1769263222, 1769263225, 1769263226, 1769263227],
            "block_number": [24305113] * 6,
            "tx_hash": [f"0x{chr(97+i)*64}" for i in range(6)],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 6,
            "sender": [attacker, victim1, attacker, attacker, victim2, attacker],
            "recipient": [attacker, victim1, attacker, attacker, victim2, attacker],
            "amount0": ["-1000", "5000", "1100", "-800", "4000", "900"],
            "amount1": ["1.0", "-5.0", "-1.1", "0.8", "-4.0", "-0.9"],
            "amount_usd": ["1000.00", "5000.00", "1100.00", "800.00", "4000.00", "900.00"],
            "tick": [196433] * 6,
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 6,
        })

        result = detect_sandwich_attacks(df, time_window_seconds=10)
        # Should detect both sandwiches
        assert result.total_candidates >= 2


class TestIntegration:
    """Integration tests for the complete detection pipeline."""

    def test_end_to_end_detection(self) -> None:
        """Test the complete detection pipeline."""
        attacker = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        victim = "0x51c72848c68a965f66fa7a88855f9f7784502a7f"

        df = pd.DataFrame({
            "timestamp": [1769263223, 1769263224, 1769263225],
            "block_number": [24305113, 24305114, 24305115],
            "tx_hash": [
                f"0x{'a' * 64}",
                f"0x{'b' * 64}",
                f"0x{'c' * 64}",
            ],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 3,
            "sender": [attacker, victim, attacker],
            "recipient": [attacker, victim, attacker],
            "amount0": ["-1000", "5000", "1100"],
            "amount1": ["1.0", "-5.0", "-1.1"],
            "amount_usd": ["1000.00", "5000.00", "1100.00"],
            "tick": [196433, 196434, 196435],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
        })

        # Run detection
        result = detect_sandwich_attacks(df, time_window_seconds=60)

        # Verify results
        assert result.total_swaps_analyzed == 3
        assert result.pools_analyzed == 1
        assert result.total_candidates >= 1
        assert result.time_window_seconds == 60

        if result.total_candidates > 0:
            candidate = result.candidates[0]
            assert candidate.attacker == attacker
            assert candidate.victim_tx == f"0x{'b' * 64}"

        # Convert to DataFrame
        candidates_df = candidates_to_dataframe(list(result.candidates))
        assert len(candidates_df) == result.total_candidates

        if not candidates_df.empty:
            assert "confidence_score" in candidates_df.columns
            assert "profit_estimate_usd" in candidates_df.columns
