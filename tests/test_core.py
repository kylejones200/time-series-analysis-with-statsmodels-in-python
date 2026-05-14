"""Tests for core time series analysis functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.core import (
# import logging

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(message)s')

    generate_synthetic_data,
    split_data,
    test_stationarity,
    calculate_mape
)

def test_generate_synthetic_data():
    """Test synthetic data generation."""
    df = generate_synthetic_data(n_samples=100, seed=42)
    if not len(df) == 100:
        raise AssertionError()
    if not 'value' in df.columns:
        raise AssertionError()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AssertionError()
def test_split_data():
    """Test data splitting."""
    df = generate_synthetic_data(n_samples=100, seed=42)
    train, hold_out = split_data(df, hold_out_days=20)
    if not len(train) == 80:
        raise AssertionError()
    if not len(hold_out) == 20:
        raise AssertionError()
    if not len(train) + len(hold_out) == len(df):
        raise AssertionError()
def test_stationarity():
    """Test stationarity test."""
    df = generate_synthetic_data(n_samples=100, seed=42)
    result = test_stationarity(df["value"])
    if not 'adf_statistic' in result:
        raise AssertionError()
    if not 'p_value' in result:
        raise AssertionError()
    if not 'is_stationary' in result:
        raise AssertionError()
    if not isinstance(result['is_stationary'], bool):
        raise AssertionError()
def test_calculate_mape():
    """Test MAPE calculation."""
    actual = pd.Series([10, 20, 30, 40, 50])
    predicted = pd.Series([11, 19, 31, 39, 51])
    mape = calculate_mape(actual, predicted)
    if not mape >= 0:
        raise AssertionError()
    if not isinstance(mape, float):
        raise AssertionError()
if __name__ == "__main__":
    pytest.main([__file__])

