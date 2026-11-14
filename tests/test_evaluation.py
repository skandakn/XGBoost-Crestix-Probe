"""Unit tests for the evaluation suite and flood metrics."""

import numpy as np
import pandas as pd
import pytest

from crestix_xgb.evaluate import (
    compute_flood_metrics,
    compute_peak_depth_errors,
    evaluate_forecast_horizons,
    evaluate_risk_classification,
)
from crestix_xgb.data import discretize_flood_risk


def test_compute_flood_metrics():
    y_true = np.array([0.1, 0.2, 0.5, 1.0])
    y_pred = np.array([0.12, 0.18, 0.55, 0.95])

    metrics = compute_flood_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "bias" in metrics
    assert "max_error" in metrics

    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["r2"] > 0.9


def test_compute_peak_depth_errors():
    df = pd.DataFrame({
        "event_id": ["event_A", "event_A", "event_B", "event_B"],
        "water_depth_m": [0.4, 0.8, 0.2, 0.5],
        "pred_depth_m": [0.38, 0.76, 0.22, 0.52],
    })

    peak_errs = compute_peak_depth_errors(df)
    assert "event_A" in peak_errs
    assert "event_B" in peak_errs

    # Event A peak is 0.8 vs 0.76 -> abs error 0.04
    assert peak_errs["event_A"]["true_peak_m"] == 0.8
    assert peak_errs["event_A"]["pred_peak_m"] == 0.76
    assert abs(peak_errs["event_A"]["peak_abs_error_m"] - 0.04) < 1e-4


def test_evaluate_forecast_horizons():
    df = pd.DataFrame({
        "horizon_min": [15, 15, 30, 30, 60, 60],
        "water_depth_m": [0.1, 0.2, 0.15, 0.25, 0.2, 0.3],
        "pred_depth_m": [0.11, 0.19, 0.16, 0.24, 0.22, 0.28],
    })

    res = evaluate_forecast_horizons(df, horizons=[15, 30, 60])
    assert "15min" in res
    assert "30min" in res
    assert "60min" in res
    assert res["15min"]["mae"] >= 0.0


def test_discretize_flood_risk_thresholds():
    depths = np.array([0.02, 0.05, 0.149, 0.15, 0.299, 0.30, 0.85])
    classes = discretize_flood_risk(depths)
    expected = np.array([0, 1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(classes, expected)


def test_evaluate_risk_classification():
    y_true = np.array([0.02, 0.10, 0.20, 0.40])
    y_pred = np.array([0.03, 0.12, 0.22, 0.38])

    res = evaluate_risk_classification(y_true, y_pred)
    assert "regressor_derived" in res
    assert res["regressor_derived"]["accuracy"] == 1.0

