"""Deterministic train/validation/test splitting utilities."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def split_customer_features(
    data_frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly split customer rows into train, validation, and test sets.

    Each row represents one customer, so random row-level splitting is
    appropriate for the independent customer segmentation objective.

    Args:
        data_frame: Customer-level feature table.
        train_fraction: Fraction assigned to training.
        validation_fraction: Fraction assigned to validation.
        seed: Random seed for reproducible row shuffling.

    Returns:
        Train, validation, and test DataFrames in that order.

    Raises:
        ValueError: If fractions are invalid or the input is empty.
    """
    _validate_split_fractions(train_fraction, validation_fraction)
    if data_frame.empty:
        raise ValueError("Customer feature data must not be empty")

    shuffled_df = data_frame.sample(frac=1.0, random_state=seed)
    train_count = int(len(data_frame) * train_fraction)
    validation_count = int(len(data_frame) * validation_fraction)
    train_df = shuffled_df.iloc[:train_count].copy()
    validation_end = train_count + validation_count
    validation_df = shuffled_df.iloc[train_count:validation_end].copy()
    test_df = shuffled_df.iloc[validation_end:].copy()

    _log_split_counts("customer", train_df, validation_df, test_df)

    return train_df, validation_df, test_df


def split_sales_features(
    data_frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split daily sales features into three partitions.

    A chronological split prevents future sales information from entering the
    training data, which would be leakage for the time-based sales objective.
    The rows are sorted by ``Order_Date`` before contiguous partitioning.

    Args:
        data_frame: Daily sales feature table.
        train_fraction: Fraction assigned to training.
        validation_fraction: Fraction assigned to validation.

    Returns:
        Chronological train, validation, and test DataFrames in that order.

    Raises:
        ValueError: If fractions are invalid, dates are missing, or input is
            empty.
    """
    _validate_split_fractions(train_fraction, validation_fraction)
    if data_frame.empty:
        raise ValueError("Sales feature data must not be empty")

    sorted_df = data_frame.copy()
    sorted_df["Order_Date"] = pd.to_datetime(
        sorted_df["Order_Date"],
        errors="raise",
    )
    sorted_df = sorted_df.sort_values("Order_Date").reset_index(drop=True)
    train_count = int(len(sorted_df) * train_fraction)
    validation_count = int(len(sorted_df) * validation_fraction)
    validation_end = train_count + validation_count
    train_df = sorted_df.iloc[:train_count].copy()
    validation_df = sorted_df.iloc[train_count:validation_end].copy()
    test_df = sorted_df.iloc[validation_end:].copy()

    _log_split_counts("sales", train_df, validation_df, test_df)

    return train_df, validation_df, test_df


def _validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
) -> None:
    """Validate two configured fractions and their implied test remainder."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1.0")


def _log_split_counts(
    objective: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Log row counts for one objective's three splits."""
    LOGGER.info(
        "%s split rows: train=%d, validation=%d, test=%d",
        objective,
        len(train_df),
        len(validation_df),
        len(test_df),
    )


def split_data(
    data_frame: pd.DataFrame,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split rows deterministically into train, validation, and test sets.

    Args:
        data_frame: Validated input rows.
        train_size: Fraction assigned to training.
        validation_size: Fraction assigned to validation.
        test_size: Fraction assigned to test.
        seed: Random seed passed to the splitter.

    Returns:
        Mapping with ``train``, ``validation``, and ``test`` copies.

    Raises:
        ValueError: If sizes are invalid or the input is too small.
    """
    sizes = (train_size, validation_size, test_size)
    if any(size <= 0 for size in sizes) or not abs(sum(sizes) - 1.0) < 1e-9:
        raise ValueError("train_size, validation_size, and test_size must be positive and sum to 1.")
    train_df, remainder_df = train_test_split(
        data_frame,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
    )
    relative_validation = validation_size / (validation_size + test_size)
    validation_df, test_df = train_test_split(remainder_df, train_size=relative_validation, random_state=seed, shuffle=True)
    return {"train": train_df.copy(), "validation": validation_df.copy(), "test": test_df.copy()}
