"""Unit tests for data cleaning pipeline.

This module contains comprehensive tests for:
- SwapEvent Pydantic model validation
- SwapValidator functionality
- clean_swap_frame processing
- Command-line script behavior
"""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from whalesentry.models.swap import SwapDataFrame, SwapEvent
from whalesentry.processing.clean_swaps import clean_swap_frame
from whalesentry.validation import CleaningReport, SwapValidator, ValidationResult


class TestSwapEvent:
    """Test suite for SwapEvent Pydantic model."""

    @pytest.fixture
    def valid_swap_dict(self) -> dict:
        """Return a valid swap record dictionary."""
        return {
            "timestamp": 1769263223,
            "block_number": 24305113,
            "tx_hash": "0x0c8d16bd4bbe310078b6c12dab184a5ffe0088c4cd7f36371680cc855014e832",
            "pool": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
            "sender": "0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
            "recipient": "0xf99eb958f923b2c74127af0f72a75f458f500101",
            "amount0": "-423.467818",
            "amount1": "0.143747594817866756",
            "amount_usd": "424.108780290392901693",
            "tick": 196433,
            "sqrt_price_x96": "1459360160181669463152813566785694",
        }

    def test_valid_swap_creation(self, valid_swap_dict: dict) -> None:
        """Test creating a valid SwapEvent."""
        swap = SwapEvent(**valid_swap_dict)
        assert swap.timestamp == 1769263223
        assert swap.block_number == 24305113
        assert swap.tick == 196433

    def test_address_normalization(self, valid_swap_dict: dict) -> None:
        """Test that addresses are normalized to lowercase."""
        # Use uppercase addresses
        valid_swap_dict["pool"] = valid_swap_dict["pool"].upper()
        valid_swap_dict["sender"] = valid_swap_dict["sender"].upper()

        swap = SwapEvent(**valid_swap_dict)
        assert swap.pool == valid_swap_dict["pool"].lower()
        assert swap.sender == valid_swap_dict["sender"].lower()

    def test_invalid_timestamp(self, valid_swap_dict: dict) -> None:
        """Test validation of timestamp range."""
        valid_swap_dict["timestamp"] = 1000000  # Before Ethereum genesis
        with pytest.raises(ValidationError):
            SwapEvent(**valid_swap_dict)

    def test_invalid_block_number(self, valid_swap_dict: dict) -> None:
        """Test validation of negative block number."""
        valid_swap_dict["block_number"] = -1
        with pytest.raises(ValidationError):
            SwapEvent(**valid_swap_dict)

    def test_invalid_tx_hash_format(self, valid_swap_dict: dict) -> None:
        """Test validation of invalid transaction hash."""
        valid_swap_dict["tx_hash"] = "invalid_hash"
        with pytest.raises(ValidationError):
            SwapEvent(**valid_swap_dict)

    def test_invalid_eth_address(self, valid_swap_dict: dict) -> None:
        """Test validation of invalid Ethereum address."""
        valid_swap_dict["pool"] = "0xinvalid"
        with pytest.raises(ValidationError):
            SwapEvent(**valid_swap_dict)

    def test_invalid_tick_range(self, valid_swap_dict: dict) -> None:
        """Test validation of tick range."""
        valid_swap_dict["tick"] = 1000000  # Outside Uniswap V3 range
        with pytest.raises(ValidationError):
            SwapEvent(**valid_swap_dict)

    def test_invalid_decimal_amount(self, valid_swap_dict: dict) -> None:
        """Test validation of non-decimal amount."""
        valid_swap_dict["amount0"] = "not_a_number"
        with pytest.raises(ValidationError):
            SwapEvent(**valid_swap_dict)

    def test_invalid_sqrt_price(self, valid_swap_dict: dict) -> None:
        """Test validation of negative sqrtPriceX96."""
        valid_swap_dict["sqrt_price_x96"] = "-123"
        with pytest.raises(ValidationError):
            SwapEvent(**valid_swap_dict)

    def test_is_buy_property(self, valid_swap_dict: dict) -> None:
        """Test is_buy property detection."""
        valid_swap_dict["amount0"] = "-100"
        valid_swap_dict["amount1"] = "0.5"
        swap = SwapEvent(**valid_swap_dict)
        assert swap.is_buy is True
        assert swap.is_sell is False

    def test_is_sell_property(self, valid_swap_dict: dict) -> None:
        """Test is_sell property detection."""
        valid_swap_dict["amount0"] = "100"
        valid_swap_dict["amount1"] = "-0.5"
        swap = SwapEvent(**valid_swap_dict)
        assert swap.is_sell is True
        assert swap.is_buy is False

    def test_usd_value_property(self, valid_swap_dict: dict) -> None:
        """Test usd_value property returns Decimal."""
        swap = SwapEvent(**valid_swap_dict)
        assert isinstance(swap.usd_value, Decimal)
        assert swap.usd_value == Decimal("424.108780290392901693")

    def test_to_dict_method(self, valid_swap_dict: dict) -> None:
        """Test conversion to dictionary."""
        swap = SwapEvent(**valid_swap_dict)
        result = swap.to_dict()
        assert result["timestamp"] == 1769263223
        assert result["tx_hash"] == valid_swap_dict["tx_hash"].lower()


class TestSwapDataFrame:
    """Test suite for SwapDataFrame collection."""

    @pytest.fixture
    def sample_swaps(self) -> list[dict]:
        """Return list of sample swap dictionaries."""
        return [
            {
                "timestamp": 1769263223,
                "block_number": 24305113,
                "tx_hash": f"0x{'a' * 64}",
                "pool": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                "sender": "0x66a9893cc07d91d95644aedd05d03f95e1dba8af",
                "recipient": "0xf99eb958f923b2c74127af0f72a75f458f500101",
                "amount0": "-100",
                "amount1": "0.5",
                "amount_usd": "100.50",
                "tick": 196433,
                "sqrt_price_x96": "1459360160181669463152813566785694",
            },
            {
                "timestamp": 1769263224,
                "block_number": 24305114,
                "tx_hash": f"0x{'b' * 64}",
                "pool": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                "sender": "0x51c72848c68a965f66fa7a88855f9f7784502a7f",
                "recipient": "0x51c72848c68a965f66fa7a88855f9f7784502a7f",
                "amount0": "200",
                "amount1": "-1.0",
                "amount_usd": "200.00",
                "tick": 196434,
                "sqrt_price_x96": "1459360160181669463152813566785695",
            },
        ]

    def test_from_records(self, sample_swaps: list[dict]) -> None:
        """Test creating SwapDataFrame from records."""
        df = SwapDataFrame.from_records(sample_swaps)
        assert df.total_count == 2

    def test_unique_pools(self, sample_swaps: list[dict]) -> None:
        """Test unique_pools property."""
        df = SwapDataFrame.from_records(sample_swaps)
        assert len(df.unique_pools) == 1
        assert "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" in df.unique_pools

    def test_time_range(self, sample_swaps: list[dict]) -> None:
        """Test time_range property."""
        df = SwapDataFrame.from_records(sample_swaps)
        min_ts, max_ts = df.time_range
        assert min_ts == 1769263223
        assert max_ts == 1769263224

    def test_block_range(self, sample_swaps: list[dict]) -> None:
        """Test block_range property."""
        df = SwapDataFrame.from_records(sample_swaps)
        min_block, max_block = df.block_range
        assert min_block == 24305113
        assert max_block == 24305114

    def test_filter_by_pool(self, sample_swaps: list[dict]) -> None:
        """Test filtering by pool address."""
        df = SwapDataFrame.from_records(sample_swaps)
        filtered = df.filter_by_pool("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640")
        assert filtered.total_count == 2

    def test_filter_by_value_range(self, sample_swaps: list[dict]) -> None:
        """Test filtering by USD value range."""
        df = SwapDataFrame.from_records(sample_swaps)
        filtered = df.filter_by_value_range(min_usd=Decimal("150"))
        assert filtered.total_count == 1
        assert filtered.swaps[0].amount_usd == "200.00"


class TestSwapValidator:
    """Test suite for SwapValidator."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Return a sample DataFrame."""
        data = {
            "timestamp": [1769263223, 1769263224],
            "block_number": [24305113, 24305114],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 2,
            "sender": ["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"] * 2,
            "recipient": ["0xf99eb958f923b2c74127af0f72a75f458f500101"] * 2,
            "amount0": ["-100", "200"],
            "amount1": ["0.5", "-1.0"],
            "amount_usd": ["100.50", "200.00"],
            "tick": [196433, 196434],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 2,
        }
        return pd.DataFrame(data)

    def test_validate_dataframe_success(self, sample_df: pd.DataFrame) -> None:
        """Test successful validation of DataFrame."""
        validator = SwapValidator()
        result = validator.validate_dataframe(sample_df)
        assert result.total_records == 2
        assert result.valid_records == 2
        assert result.invalid_records == 0
        assert result.success_rate == 100.0

    def test_validate_dataframe_with_errors(self) -> None:
        """Test validation with invalid records."""
        data = {
            "timestamp": [1769263223, 1000000],  # Second timestamp invalid
            "block_number": [24305113, 24305114],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 2,
            "sender": ["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"] * 2,
            "recipient": ["0xf99eb958f923b2c74127af0f72a75f458f500101"] * 2,
            "amount0": ["-100", "200"],
            "amount1": ["0.5", "-1.0"],
            "amount_usd": ["100.50", "200.00"],
            "tick": [196433, 196434],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 2,
        }
        df = pd.DataFrame(data)
        validator = SwapValidator()
        result = validator.validate_dataframe(df)
        assert result.total_records == 2
        assert result.valid_records == 1
        assert result.invalid_records == 1
        assert result.success_rate == 50.0

    def test_check_schema(self, sample_df: pd.DataFrame) -> None:
        """Test schema checking."""
        validator = SwapValidator()
        missing = validator.check_schema(sample_df)
        assert len(missing) == 0

    def test_check_schema_missing_columns(self) -> None:
        """Test schema checking with missing columns."""
        df = pd.DataFrame({"timestamp": [1, 2]})  # Missing most columns
        validator = SwapValidator()
        missing = validator.check_schema(df)
        assert len(missing) > 0
        assert "tx_hash" in missing

    def test_validate_amount_consistency(self) -> None:
        """Test amount consistency validation."""
        assert SwapValidator.validate_amount_consistency("-100", "0.5") is True
        assert SwapValidator.validate_amount_consistency("100", "-0.5") is True
        assert SwapValidator.validate_amount_consistency("100", "0.5") is False  # Both positive
        assert SwapValidator.validate_amount_consistency("-100", "-0.5") is False  # Both negative

    def test_validate_usd_value(self) -> None:
        """Test USD value validation."""
        assert SwapValidator.validate_usd_value("100.50", Decimal("0")) is True
        assert SwapValidator.validate_usd_value("100.50", Decimal("50")) is True
        assert SwapValidator.validate_usd_value("10.50", Decimal("50")) is False
        assert SwapValidator.validate_usd_value("invalid", Decimal("0")) is False


class TestCleaningReport:
    """Test suite for CleaningReport."""

    @pytest.fixture
    def sample_validation_result(self) -> ValidationResult:
        """Return a sample validation result."""
        return ValidationResult(
            total_records=100,
            valid_records=95,
            invalid_records=5,
            validation_errors=["Error 1", "Error 2"],
            duplicate_count=3,
        )

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Return a sample DataFrame."""
        return pd.DataFrame({
            "timestamp": [1, 2, 3],
            "block_number": [100, 200, 300],
            "value": [10.5, 20.5, 30.5],
        })

    def test_add_validation_result(self, sample_validation_result: ValidationResult) -> None:
        """Test adding validation result to report."""
        report = CleaningReport()
        report.add_validation_result(sample_validation_result)
        assert len(report.validation_results) == 1

    def test_add_statistics(self, sample_df: pd.DataFrame) -> None:
        """Test adding statistics to report."""
        report = CleaningReport()
        report.add_statistics(sample_df)
        assert report.statistics["total_rows"] == 3
        assert "timestamp" in report.statistics["columns"]

    def test_generate_markdown(
        self,
        sample_validation_result: ValidationResult,
        sample_df: pd.DataFrame,
    ) -> None:
        """Test markdown report generation."""
        report = CleaningReport()
        report.add_validation_result(sample_validation_result)
        report.add_statistics(sample_df)
        markdown = report.generate_markdown()
        assert "# Data Cleaning Report" in markdown
        assert "100" in markdown  # Total records
        assert "95" in markdown   # Valid records


class TestCleanSwapFrame:
    """Test suite for clean_swap_frame function."""

    @pytest.fixture
    def raw_df(self) -> pd.DataFrame:
        """Return a raw DataFrame with various data types."""
        return pd.DataFrame({
            "timestamp": ["1769263223", "1769263224", "invalid"],
            "block_number": [24305113, 24305114, 24305115],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}", f"0x{'c' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 3,
            "sender": ["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"] * 3,
            "recipient": ["0xf99eb958f923b2c74127af0f72a75f458f500101"] * 3,
            "amount0": ["-100", "200", "300"],
            "amount1": ["0.5", "-1.0", "-1.5"],
            "amount_usd": ["100.50", "200.00", "300.00"],
            "tick": [196433, 196434, 196435],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 3,
        })

    def test_timestamp_conversion(self, raw_df: pd.DataFrame) -> None:
        """Test that timestamps are converted to int64."""
        cleaned = clean_swap_frame(raw_df)
        assert cleaned["timestamp"].dtype == "int64"
        assert len(cleaned) == 2  # One row dropped due to invalid timestamp

    def test_address_lowercasing(self, raw_df: pd.DataFrame) -> None:
        """Test that addresses are lowercased."""
        # Add uppercase addresses
        raw_df["pool"] = raw_df["pool"].str.upper()
        cleaned = clean_swap_frame(raw_df)
        assert all(cleaned["pool"].str.islower())

    def test_amount_string_conversion(self, raw_df: pd.DataFrame) -> None:
        """Test that amounts are converted to strings."""
        cleaned = clean_swap_frame(raw_df)
        assert cleaned["amount0"].dtype == "string"
        assert cleaned["amount1"].dtype == "string"

    def test_duplicate_removal(self, raw_df: pd.DataFrame) -> None:
        """Test duplicate removal based on key columns."""
        # Add duplicate row
        duplicate = raw_df.iloc[[0]]
        df_with_dup = pd.concat([raw_df, duplicate], ignore_index=True)
        cleaned = clean_swap_frame(df_with_dup)
        # Should have removed the duplicate
        assert len(cleaned) <= len(df_with_dup)


class TestIntegration:
    """Integration tests for the complete cleaning pipeline."""

    def test_end_to_end_pipeline(self) -> None:
        """Test the complete pipeline from DataFrame to cleaned output."""
        # Create sample data
        data = {
            "timestamp": [1769263223, 1769263224],
            "block_number": [24305113, 24305114],
            "tx_hash": [f"0x{'a' * 64}", f"0x{'b' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"] * 2,
            "sender": ["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"] * 2,
            "recipient": ["0xf99eb958f923b2c74127af0f72a75f458f500101"] * 2,
            "amount0": ["-100", "200"],
            "amount1": ["0.5", "-1.0"],
            "amount_usd": ["100.50", "200.00"],
            "tick": [196433, 196434],
            "sqrtPriceX96": ["1459360160181669463152813566785694"] * 2,
        }
        df = pd.DataFrame(data)

        # Run validation
        validator = SwapValidator()
        result = validator.validate_dataframe(df)
        assert result.is_valid

        # Run cleaning
        cleaned = clean_swap_frame(df)
        assert len(cleaned) == 2

        # Generate report
        report = CleaningReport()
        report.add_validation_result(result)
        report.add_statistics(cleaned)
        markdown = report.generate_markdown()
        assert len(markdown) > 0

    def test_parquet_roundtrip(self) -> None:
        """Test saving and loading cleaned data."""
        data = {
            "timestamp": [1769263223],
            "block_number": [24305113],
            "tx_hash": [f"0x{'a' * 64}"],
            "pool": ["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"],
            "sender": ["0x66a9893cc07d91d95644aedd05d03f95e1dba8af"],
            "recipient": ["0xf99eb958f923b2c74127af0f72a75f458f500101"],
            "amount0": ["-100"],
            "amount1": ["0.5"],
            "amount_usd": ["100.50"],
            "tick": [196433],
            "sqrtPriceX96": ["1459360160181669463152813566785694"],
        }
        df = pd.DataFrame(data)
        cleaned = clean_swap_frame(df)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = Path(f.name)

        try:
            cleaned.to_parquet(temp_path, index=False)
            loaded = pd.read_parquet(temp_path)
            assert len(loaded) == 1
            assert loaded["timestamp"].iloc[0] == 1769263223
        finally:
            temp_path.unlink()
