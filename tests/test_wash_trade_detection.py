"""Unit tests for wash trading detection.

This module contains comprehensive tests for:
- WashTradeCandidate Pydantic model validation
- DetectionResult dataclass functionality
- detect_wash_trades function with various scenarios
- Edge cases and false positive prevention
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from pydantic import ValidationError

from whalesentry.detection.wash_trade import (
    DetectionResult,
    WashTradeCandidate,
    _calculate_confidence,
    _detect_wash_trades_in_pool,
    candidates_to_dataframe,
    detect_wash_trades,
)


class TestWashTradeCandidate:
    """Test suite for WashTradeCandidate Pydantic model."""

    @pytest.fixture
    def valid_candidate_dict(self) -> dict:
        """Return a valid wash trade candidate dictionary."""
        return {
            "trader": "0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
            "pool": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
            "roundtrip_count": 5,
            "first_tx": "0x0c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e832",
            "last_tx": "0x1c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e833",
            "first_timestamp": 1769263223,
            "last_timestamp": 1769263523,
            "total_volume_usd": "15000.00",
            "avg_roundtrip_time": 60.0,
            "price_variance": "0.02",
            "confidence_score": 0.85,
        }

    def test_valid_candidate_creation(self, valid_candidate_dict: dict) -> None:
        """Test creating a valid WashTradeCandidate."""
        candidate = WashTradeCandidate(**valid_candidate_dict)
        assert candidate.trader == "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        assert candidate.roundtrip_count == 5
        assert candidate.confidence_score == 0.85

    def test_address_normalization(self, valid_candidate_dict: dict) -> None:
        """Test that addresses are normalized to lowercase."""
        valid_candidate_dict["trader"] = valid_candidate_dict["trader"].upper()
        valid_candidate_dict["pool"] = valid_candidate_dict["pool"].upper()

        candidate = WashTradeCandidate(**valid_candidate_dict)
        assert candidate.trader == valid_candidate_dict["trader"].lower()
        assert candidate.pool == valid_candidate_dict["pool"].lower()

    def test_invalid_trader_address(self, valid_candidate_dict: dict) -> None:
        """Test validation of invalid trader address."""
        valid_candidate_dict["trader"] = "0xinvalid"
        with pytest.raises(ValidationError):
            WashTradeCandidate(**valid_candidate_dict)

    def test_invalid_tx_hash(self, valid_candidate_dict: dict) -> None:
        """Test validation of invalid transaction hash."""
        valid_candidate_dict["first_tx"] = "invalid_hash"
        with pytest.raises(ValidationError):
            WashTradeCandidate(**valid_candidate_dict)

    def test_invalid_confidence_score(self, valid_candidate_dict: dict) -> None:
        """Test validation of confidence score out of range."""
        valid_candidate_dict["confidence_score"] = 1.5
        with pytest.raises(ValidationError):
            WashTradeCandidate(**valid_candidate_dict)

    def test_invalid_roundtrip_count(self, valid_candidate_dict: dict) -> None:
        """Test validation of roundtrip count below minimum."""
        valid_candidate_dict["roundtrip_count"] = 2
        with pytest.raises(ValidationError):
            WashTradeCandidate(**valid_candidate_dict)

    def test_invalid_decimal_string(self, valid_candidate_dict: dict) -> None:
        """Test validation of invalid decimal string."""
        valid_candidate_dict["total_volume_usd"] = "not_a_number"
        with pytest.raises(ValidationError):
            WashTradeCandidate(**valid_candidate_dict)

    def test_volume_usd_property(self, valid_candidate_dict: dict) -> None:
        """Test volume_usd property returns Decimal."""
        candidate = WashTradeCandidate(**valid_candidate_dict)
        assert isinstance(candidate.volume_usd, Decimal)
        assert candidate.volume_usd == Decimal("15000.00")

    def test_variance_property(self, valid_candidate_dict: dict) -> None:
        """Test variance property returns Decimal."""
        candidate = WashTradeCandidate(**valid_candidate_dict)
        assert isinstance(candidate.variance, Decimal)
        assert candidate.variance == Decimal("0.02")

    def test_immutability(self, valid_candidate_dict: dict) -> None:
        """Test that WashTradeCandidate is immutable."""
        candidate = WashTradeCandidate(**valid_candidate_dict)
        with pytest.raises(ValidationError):
            candidate.confidence_score = 0.9

    def test_extra_fields_forbidden(self, valid_candidate_dict: dict) -> None:
        """Test that extra fields are not allowed."""
        valid_candidate_dict["extra_field"] = "not_allowed"
        with pytest.raises(ValidationError):
            WashTradeCandidate(**valid_candidate_dict)


class TestDetectionResult:
    """Test suite for DetectionResult dataclass."""

    def test_empty_result(self) -> None:
        """Test creating an empty detection result."""
        result = DetectionResult(
            candidates=tuple(),
            total_swaps_analyzed=0,
            pools_analyzed=0,
            time_window_seconds=300,
            min_usd_threshold=Decimal("1000"),
            min_roundtrips=3,
        )
        assert result.total_candidates == 0
        assert len(result.candidates) == 0

    def test_result_with_candidates(self) -> None:
        """Test detection result with candidates."""
        candidate = WashTradeCandidate(
            trader="0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
            pool="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
            roundtrip_count=5,
            first_tx="0x0c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e832",
            last_tx="0x1c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e833",
            first_timestamp=1769263223,
            last_timestamp=1769263523,
            total_volume_usd="15000.00",
            avg_roundtrip_time=60.0,
            price_variance="0.02",
            confidence_score=0.85,
        )

        result = DetectionResult(
            candidates=(candidate,),
            total_swaps_analyzed=100,
            pools_analyzed=5,
            time_window_seconds=300,
            min_usd_threshold=Decimal("1000"),
            min_roundtrips=3,
        )

        assert result.total_candidates == 1
        assert len(result.candidates) == 1
        assert result.candidates[0] == candidate


class TestCalculateConfidence:
    """Test suite for confidence score calculation."""

    def test_high_confidence_pattern(self) -> None:
        """Test high confidence score for suspicious pattern."""
        score = _calculate_confidence(
            roundtrip_count=10,
            price_variance=Decimal("0.005"),
            avg_roundtrip_time=20.0,
            time_window=300,
        )
        assert score >= 0.8

    def test_low_confidence_pattern(self) -> None:
        """Test low confidence score for less suspicious pattern."""
        score = _calculate_confidence(
            roundtrip_count=3,
            price_variance=Decimal("0.15"),
            avg_roundtrip_time=200.0,
            time_window=300,
        )
        assert score <= 0.6

    def test_minimum_roundtrips(self) -> None:
        """Test confidence with minimum roundtrips."""
        score = _calculate_confidence(
            roundtrip_count=3,
            price_variance=Decimal("0.01"),
            avg_roundtrip_time=60.0,
            time_window=300,
        )
        assert 0.0 <= score <= 1.0

    def test_high_variance_penalty(self) -> None:
        """Test that high price variance reduces confidence."""
        low_variance_score = _calculate_confidence(
            roundtrip_count=5,
            price_variance=Decimal("0.01"),
            avg_roundtrip_time=60.0,
            time_window=300,
        )
        high_variance_score = _calculate_confidence(
            roundtrip_count=5,
            price_variance=Decimal("0.20"),
            avg_roundtrip_time=60.0,
            time_window=300,
        )
        assert low_variance_score > high_variance_score


class TestDetectWashTradesInPool:
    """Test suite for pool-level wash trade detection."""

    def test_no_wash_trades(self) -> None:
        """Test detection with no wash trading pattern."""
        df = pd.DataFrame({
            "timestamp": [1769263223, 1769263283, 1769263343],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}", f"0x{'c' * 64}"],
            "sender": [
                "0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
                "0x77b9893cc07d91d95644aedd05d03f95e1dba8af",
                "0x88c9893cc07d91d95644aedd05d03f95e1dba8af",
            ],
            "amount0": ["-1000", "2000", "-1500"],
            "amount1": ["1.0", "-2.0", "1.5"],
            "amount_usd": ["1000", "2000", "1500"],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
        })

        candidates = _detect_wash_trades_in_pool(
            df,
            time_window_seconds=300,
            min_usd_threshold=Decimal("1000"),
            min_roundtrips=3,
        )

        assert len(candidates) == 0

    def test_single_roundtrip_insufficient(self) -> None:
        """Test that single roundtrip is not flagged."""
        trader = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        df = pd.DataFrame({
            "timestamp": [1769263223, 1769263283],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}"],
            "sender": [trader, trader],
            "amount0": ["-1000", "1000"],
            "amount1": ["1.0", "-1.0"],
            "amount_usd": ["1000", "1000"],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 2,
        })

        candidates = _detect_wash_trades_in_pool(
            df,
            time_window_seconds=300,
            min_usd_threshold=Decimal("1000"),
            min_roundtrips=3,
        )

        assert len(candidates) == 0

    def test_multiple_roundtrips_detected(self) -> None:
        """Test detection of multiple roundtrips."""
        trader = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        pool = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        # Create 4 roundtrips within 300 seconds (each roundtrip takes 60s, total 240s)
        timestamps = [1769263223 + i * 30 for i in range(8)]
        tx_hashes = [f"0x{chr(97 + i) * 64}" for i in range(8)]
        amounts0 = ["-1000", "1000", "-1000", "1000", "-1000", "1000", "-1000", "1000"]
        amounts1 = ["1.0", "-1.0", "1.0", "-1.0", "1.0", "-1.0", "1.0", "-1.0"]

        df = pd.DataFrame({
            "timestamp": timestamps,
            "tx_hash": tx_hashes,
            "pool": [pool] * 8,
            "sender": [trader] * 8,
            "amount0": amounts0,
            "amount1": amounts1,
            "amount_usd": ["1000"] * 8,
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 8,
        })

        candidates = _detect_wash_trades_in_pool(
            df,
            time_window_seconds=300,
            min_usd_threshold=Decimal("1000"),
            min_roundtrips=3,
        )

        assert len(candidates) >= 1
        if candidates:
            candidate = candidates[0]
            assert candidate.trader == trader
            assert candidate.roundtrip_count >= 3

    def test_time_window_enforcement(self) -> None:
        """Test that time window is enforced."""
        trader = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        pool = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        timestamps = [1769263223, 1769263283, 1769263643, 1769263703, 1769264063, 1769264123]
        tx_hashes = [f"0x{chr(97 + i) * 64}" for i in range(6)]
        amounts0 = ["-1000", "1000", "-1000", "1000", "-1000", "1000"]
        amounts1 = ["1.0", "-1.0", "1.0", "-1.0", "1.0", "-1.0"]

        df = pd.DataFrame({
            "timestamp": timestamps,
            "tx_hash": tx_hashes,
            "pool": [pool] * 6,
            "sender": [trader] * 6,
            "amount0": amounts0,
            "amount1": amounts1,
            "amount_usd": ["1000"] * 6,
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 6,
        })

        candidates = _detect_wash_trades_in_pool(
            df,
            time_window_seconds=300,
            min_usd_threshold=Decimal("1000"),
            min_roundtrips=3,
        )

        assert len(candidates) == 0

    def test_min_usd_threshold(self) -> None:
        """Test that minimum USD threshold is enforced."""
        trader = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        timestamps = [1769263223 + i * 60 for i in range(8)]
        tx_hashes = [f"0x{chr(97 + i) * 64}" for i in range(8)]
        amounts0 = ["-100", "100", "-100", "100", "-100", "100", "-100", "100"]
        amounts1 = ["0.1", "-0.1", "0.1", "-0.1", "0.1", "-0.1", "0.1", "-0.1"]

        df = pd.DataFrame({
            "timestamp": timestamps,
            "tx_hash": tx_hashes,
            "sender": [trader] * 8,
            "amount0": amounts0,
            "amount1": amounts1,
            "amount_usd": ["100"] * 8,
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 8,
        })

        candidates = _detect_wash_trades_in_pool(
            df,
            time_window_seconds=300,
            min_usd_threshold=Decimal("1000"),
            min_roundtrips=3,
        )

        assert len(candidates) == 0


class TestDetectWashTrades:
    """Test suite for main wash trade detection function."""

    def test_empty_dataframe(self) -> None:
        """Test detection with empty DataFrame."""
        df = pd.DataFrame({
            "timestamp": [],
            "tx_hash": [],
            "pool": [],
            "sender": [],
            "amount0": [],
            "amount1": [],
            "amount_usd": [],
            "sqrtPriceX96": [],
        })

        result = detect_wash_trades(df)
        assert result.total_candidates == 0
        assert result.total_swaps_analyzed == 0

    def test_missing_required_columns(self) -> None:
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({
            "timestamp": [1769263223],
            "tx_hash": [f"0x{'a' * 64}"],
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            detect_wash_trades(df)

    def test_multiple_pools(self) -> None:
        """Test detection across multiple pools."""
        trader = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        pool1 = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        pool2 = "0x99f7a0c2ddd26feeb64f039a2c41296fcb3f5640"

        timestamps = [1769263223 + i * 60 for i in range(8)]
        tx_hashes = [f"0x{chr(97 + i) * 64}" for i in range(8)]
        amounts0 = ["-1000", "1000", "-1000", "1000"] * 2
        amounts1 = ["1.0", "-1.0", "1.0", "-1.0"] * 2

        df = pd.DataFrame({
            "timestamp": timestamps,
            "tx_hash": tx_hashes,
            "pool": [pool1] * 4 + [pool2] * 4,
            "sender": [trader] * 8,
            "amount0": amounts0,
            "amount1": amounts1,
            "amount_usd": ["1000"] * 8,
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 8,
        })

        result = detect_wash_trades(df)
        assert result.pools_analyzed == 2
        assert result.total_swaps_analyzed == 8

    def test_integration_with_real_pattern(self) -> None:
        """Test end-to-end detection with realistic wash trading pattern."""
        trader = "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        victim = "0x77b9893cc07d91d95644aedd05d03f95e1dba8af"

        # Create pattern within 300 seconds: trader does 4 roundtrips, victim does 2 trades
        timestamps = [1769263223 + i * 30 for i in range(10)]
        tx_hashes = [f"0x{chr(97 + i) * 64}" for i in range(10)]
        senders = [trader, trader, victim, trader, trader, trader, trader, victim, trader, trader]
        amounts0 = ["-1000", "1000", "500", "-1000", "1000", "-1000", "1000", "-500", "-1000", "1000"]
        amounts1 = ["1.0", "-1.0", "-0.5", "1.0", "-1.0", "1.0", "-1.0", "0.5", "1.0", "-1.0"]

        df = pd.DataFrame({
            "timestamp": timestamps,
            "tx_hash": tx_hashes,
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 10,
            "sender": senders,
            "amount0": amounts0,
            "amount1": amounts1,
            "amount_usd": ["1000", "1000", "500", "1000", "1000", "1000", "1000", "500", "1000", "1000"],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 10,
        })

        result = detect_wash_trades(df, time_window_seconds=300, min_usd_threshold=Decimal("1000"), min_roundtrips=3)

        assert result.total_swaps_analyzed == 10
        assert result.pools_analyzed == 1
        assert result.total_candidates >= 1

        if result.total_candidates > 0:
            candidate = result.candidates[0]
            assert candidate.trader == trader
            assert candidate.roundtrip_count >= 3


class TestCandidatesToDataFrame:
    """Test suite for candidates_to_dataframe conversion."""

    def test_empty_candidates(self) -> None:
        """Test conversion of empty candidate list."""
        df = candidates_to_dataframe([])
        assert len(df) == 0
        assert "trader" in df.columns
        assert "confidence_score" in df.columns

    def test_single_candidate(self) -> None:
        """Test conversion of single candidate."""
        candidate = WashTradeCandidate(
            trader="0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
            pool="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
            roundtrip_count=5,
            first_tx="0x0c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e832",
            last_tx="0x1c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e833",
            first_timestamp=1769263223,
            last_timestamp=1769263523,
            total_volume_usd="15000.00",
            avg_roundtrip_time=60.0,
            price_variance="0.02",
            confidence_score=0.85,
        )

        df = candidates_to_dataframe([candidate])
        assert len(df) == 1
        assert df.iloc[0]["trader"] == "0x66a9893cc07d91d95644aedd05d03f95e1dba8af"
        assert df.iloc[0]["roundtrip_count"] == 5
        assert df.iloc[0]["confidence_score"] == 0.85

    def test_multiple_candidates(self) -> None:
        """Test conversion of multiple candidates."""
        candidates = [
            WashTradeCandidate(
                trader=f"0x{i}6a9893cc07d91d95644aedd05d03f95e1dba8af",
                pool="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                roundtrip_count=3 + i,
                first_tx="0x0c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e832",
                last_tx="0x1c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e833",
                first_timestamp=1769263223,
                last_timestamp=1769263523,
                total_volume_usd=f"{10000 + i * 1000}.00",
                avg_roundtrip_time=60.0,
                price_variance="0.02",
                confidence_score=0.7 + i * 0.05,
            )
            for i in range(3)
        ]

        df = candidates_to_dataframe(candidates)
        assert len(df) == 3
        assert all(col in df.columns for col in ["trader", "roundtrip_count", "confidence_score"])
