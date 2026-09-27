"""Aggregated data validation using the explicit Pandera schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from customer_sales_analytics.data.schema import RAW_SCHEMA, REQUIRED_COLUMNS


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable validation problem."""

    code: str
    message: str
    column: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    """Structured result of validating one DataFrame."""

    passed: bool
    issues: list[ValidationIssue]
    row_count: int
    column_count: int


class DataValidationError(ValueError):
    """Raised after all validation issues have been collected."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        message = "; ".join(issue.message for issue in report.issues)
        super().__init__(f"Data validation failed with {len(report.issues)} issue(s): {message}")


def validate_data_frame(
    data_frame: pd.DataFrame,
    max_null_rate: float = 0.20,
    allowed_categories: Mapping[str, list[Any]] | None = None,
    allow_duplicate_rows: bool = False,
    nullable_columns: list[str] | None = None,
) -> ValidationReport:
    """Validate a DataFrame and aggregate all discovered issues.

    Args:
        data_frame: DataFrame at the ingestion boundary.
        max_null_rate: Maximum permitted null fraction for any column.
        allowed_categories: Optional configured categorical value allowlists.
        allow_duplicate_rows: Whether exact duplicate records are permitted.
        nullable_columns: Columns exempt from the general null-rate threshold.

    Returns:
        A report containing every issue and an overall pass flag.

    Raises:
        DataValidationError: If validation fails overall.
    """
    issues: list[ValidationIssue] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in data_frame.columns]
    if missing:
        issues.append(ValidationIssue("missing_columns", f"Missing required columns: {missing}"))
    unexpected = sorted(set(data_frame.columns) - set(REQUIRED_COLUMNS))
    if unexpected:
        issues.append(ValidationIssue("unexpected_columns", f"Unexpected columns: {unexpected}"))
    if not missing:
        try:
            RAW_SCHEMA.validate(data_frame, lazy=True)
        except Exception as error:
            failure_cases = getattr(error, "failure_cases", None)
            details = {"failure_cases": failure_cases.to_dict("records")} if failure_cases is not None else {}
            issues.append(ValidationIssue("schema", str(error), details=details))
    if not allow_duplicate_rows:
        duplicate_count = int(data_frame.duplicated().sum())
        if duplicate_count:
            issues.append(ValidationIssue("duplicate_rows", f"Found {duplicate_count} duplicate row(s)."))
    exempt_columns = set(nullable_columns or [])
    for column, rate in data_frame.isna().mean().items():
        if column in exempt_columns:
            continue
        if rate > max_null_rate:
            issues.append(ValidationIssue("null_rate", f"{column} null rate {rate:.2%} exceeds {max_null_rate:.2%}.", column))
    for column, values in (allowed_categories or {}).items():
        if column in data_frame:
            invalid = sorted(set(data_frame[column].dropna().unique()) - set(values), key=str)
            if invalid:
                issues.append(ValidationIssue("category", f"Invalid values in {column}: {invalid}", column))
    report = ValidationReport(not issues, issues, len(data_frame), len(data_frame.columns))
    if not report.passed:
        raise DataValidationError(report)
    return report
