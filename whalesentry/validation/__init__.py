"""Validation utilities for swap data cleaning."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd
from pydantic import ValidationError

from whalesentry.models.swap import SwapEvent


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a batch of swap records.

    Attributes:
        total_records: Total number of records processed.
        valid_records: Number of records that passed validation.
        invalid_records: Number of records that failed validation.
        validation_errors: List of error messages for invalid records.
        duplicate_count: Number of duplicate records removed.
    """

    total_records: int
    valid_records: int
    invalid_records: int
    validation_errors: list[str] = field(default_factory=list)
    duplicate_count: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate validation success rate as percentage."""
        if self.total_records == 0:
            return 100.0
        return (self.valid_records / self.total_records) * 100

    @property
    def is_valid(self) -> bool:
        """Check if all records passed validation."""
        return self.invalid_records == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for reporting."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "success_rate": f"{self.success_rate:.2f}%",
            "duplicate_count": self.duplicate_count,
            "validation_errors": self.validation_errors[:10],  # Limit errors in report
        }


class SwapValidator:
    """Validator for swap data with detailed error reporting.

    This class provides methods to validate DataFrame records against
    the SwapEvent Pydantic model, with comprehensive error tracking.

    Example:
        >>> validator = SwapValidator()
        >>> result = validator.validate_dataframe(df)
        >>> print(f"Success rate: {result.success_rate:.1f}%")
    """

    REQUIRED_COLUMNS: set[str] = {
        "timestamp",
        "block_number",
        "tx_hash",
        "pool",
        "sender",
        "recipient",
        "amount0",
        "amount1",
        "amount_usd",
        "tick",
        "sqrtPriceX96",
    }

    def __init__(self) -> None:
        """Initialize the validator with empty error tracking."""
        self._errors: list[tuple[int, str, str]] = []  # (index, field, error)

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Validate all records in a DataFrame.

        Args:
            df: DataFrame containing swap records.

        Returns:
            ValidationResult with detailed statistics.
        """
        self._errors = []
        total = len(df)
        valid_count = 0
        records = df.to_dict("records")

        for idx, record in enumerate(records):
            try:
                self._validate_single_record(record, idx)
                valid_count += 1
            except ValidationError as e:
                error_msg = f"Row {idx}: {e.errors()[0]['msg']}"
                self._errors.append((idx, "validation", error_msg))
            except Exception as e:
                error_msg = f"Row {idx}: Unexpected error - {e}"
                self._errors.append((idx, "exception", error_msg))

        # Check for duplicates
        duplicate_count = self._count_duplicates(df)

        return ValidationResult(
            total_records=total,
            valid_records=valid_count,
            invalid_records=total - valid_count,
            validation_errors=[err[2] for err in self._errors],
            duplicate_count=duplicate_count,
        )

    def _validate_single_record(self, record: dict[str, Any], index: int) -> SwapEvent:
        """Validate a single record and return SwapEvent.

        Args:
            record: Dictionary containing swap data.
            index: Row index for error reporting.

        Returns:
            Validated SwapEvent instance.

        Raises:
            ValidationError: If record fails Pydantic validation.
        """
        # Normalize column names (handle sqrtPriceX96 vs sqrt_price_x96)
        normalized = self._normalize_record(record)
        return SwapEvent(**normalized)

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Normalize record keys to match Pydantic model.

        Args:
            record: Raw record from DataFrame.

        Returns:
            Normalized record with correct key names.
        """
        normalized = {}
        for key, value in record.items():
            # Handle column name differences
            if key == "sqrtPriceX96":
                normalized["sqrt_price_x96"] = str(value) if value is not None else "0"
            elif key in self.REQUIRED_COLUMNS:
                normalized[key] = value
            else:
                normalized[key] = value
        return normalized

    def _count_duplicates(self, df: pd.DataFrame) -> int:
        """Count duplicate records based on key fields.

        Args:
            df: DataFrame to check for duplicates.

        Returns:
            Number of duplicate rows that would be removed.
        """
        dup_columns = ["tx_hash", "timestamp", "sender", "recipient"]
        available_cols = [c for c in dup_columns if c in df.columns]
        if not available_cols:
            return 0
        return int(df.duplicated(subset=available_cols).sum())

    def check_schema(self, df: pd.DataFrame) -> list[str]:
        """Check DataFrame schema against required columns.

        Args:
            df: DataFrame to check.

        Returns:
            List of missing required columns.
        """
        df_columns = set(df.columns)
        missing = self.REQUIRED_COLUMNS - df_columns
        return sorted(missing)

    @staticmethod
    def validate_amount_consistency(amount0: str, amount1: str) -> bool:
        """Check if amounts have opposite signs (one in, one out).

        In Uniswap V3 swaps, one token amount is always negative (outgoing)
        and the other is positive (incoming) for a normal swap.

        Args:
            amount0: Token0 amount as string.
            amount1: Token1 amount as string.

        Returns:
            True if amounts have opposite signs.
        """
        try:
            a0 = Decimal(amount0)
            a1 = Decimal(amount1)
            # For a valid swap, one should be positive and one negative
            return (a0 > 0 and a1 < 0) or (a0 < 0 and a1 > 0)
        except Exception:
            return False

    @staticmethod
    def validate_usd_value(amount_usd: str, min_usd: Decimal = Decimal("0")) -> bool:
        """Validate that USD amount is positive and above minimum.

        Args:
            amount_usd: USD value as string.
            min_usd: Minimum acceptable USD value.

        Returns:
            True if USD value is valid.
        """
        try:
            usd = Decimal(amount_usd)
            return usd > min_usd
        except Exception:
            return False


class CleaningReport:
    """Generate cleaning and validation reports.

    Example:
        >>> report = CleaningReport()
        >>> report.add_validation_result(result)
        >>> report.add_statistics(df)
        >>> print(report.generate_markdown())
    """

    def __init__(self) -> None:
        """Initialize empty report."""
        self.validation_results: list[ValidationResult] = []
        self.statistics: dict[str, Any] = {}
        self.timestamps: dict[str, str] = {}

    def add_validation_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.validation_results.append(result)

    def add_statistics(self, df: pd.DataFrame) -> None:
        """Calculate and store DataFrame statistics."""
        self.statistics = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "numeric_summary": {},
        }

        # Add numeric column summaries
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            self.statistics["numeric_summary"][col] = {
                "min": float(df[col].min()) if not df[col].empty else None,
                "max": float(df[col].max()) if not df[col].empty else None,
                "mean": float(df[col].mean()) if not df[col].empty else None,
            }

    def generate_markdown(self) -> str:
        """Generate markdown report.

        Returns:
            Formatted markdown string.
        """
        lines = [
            "# Data Cleaning Report",
            "",
            "## Validation Summary",
            "",
        ]

        for i, result in enumerate(self.validation_results, 1):
            lines.extend([
                f"### Validation Run {i}",
                "",
                f"- **Total Records**: {result.total_records:,}",
                f"- **Valid Records**: {result.valid_records:,}",
                f"- **Invalid Records**: {result.invalid_records:,}",
                f"- **Success Rate**: {result.success_rate:.2f}%",
                f"- **Duplicates Found**: {result.duplicate_count:,}",
                "",
            ])

            if result.validation_errors:
                lines.extend([
                    "#### Sample Errors",
                    "",
                ])
                for error in result.validation_errors[:5]:
                    lines.append(f"- {error}")
                if len(result.validation_errors) > 5:
                    lines.append(f"- ... and {len(result.validation_errors) - 5} more errors")
                lines.append("")

        if self.statistics:
            lines.extend([
                "## Data Statistics",
                "",
                f"- **Total Rows**: {self.statistics['total_rows']:,}",
                f"- **Columns**: {', '.join(self.statistics['columns'])}",
                f"- **Memory Usage**: {self.statistics['memory_usage_mb']:.2f} MB",
                "",
            ])

            if self.statistics["numeric_summary"]:
                lines.extend([
                    "### Numeric Columns Summary",
                    "",
                    "| Column | Min | Max | Mean |",
                    "|--------|-----|-----|------|",
                ])
                for col, stats in self.statistics["numeric_summary"].items():
                    min_val = f"{stats['min']:.2f}" if stats['min'] is not None else "N/A"
                    max_val = f"{stats['max']:.2f}" if stats['max'] is not None else "N/A"
                    mean_val = f"{stats['mean']:.2f}" if stats['mean'] is not None else "N/A"
                    lines.append(f"| {col} | {min_val} | {max_val} | {mean_val} |")
                lines.append("")

        return "\n".join(lines)


__all__ = [
    "ValidationResult",
    "SwapValidator",
    "CleaningReport",
]
