"""
test_feature_engineering.py
===========================
Unit tests for topographic, meteorological, hydraulic, and spatio-temporal features.
"""

import numpy as np
import pytest

from data_pipeline.feature_engineering import (
    build_event_tabular_dataframe,
    compute_hydraulic_lag_features,
    compute_rainfall_features,
    compute_spatial_temporal_features,
    compute_topographic_features,
    get_feature_column_names,
)
from data_pipeline.load_data import generate_synthetic_larno_event


def test_compute_topographic_features():
    # Simple inclined plane: z = 2*x + 3*y
    y, x = np.meshgrid(np.arange(32, dtype=np.float32), np.arange(32, dtype=np.float32), indexing="ij")
    dem = 2.0 * x + 3.0 * y

    feats = compute_topographic_features(dem, cell_size_m=8.0)

    # Required keys exist
    expected_keys = [
        "elevation",
        "slope_magnitude",
        "slope_x",
        "slope_y",
        "aspect_sin",
        "aspect_cos",
        "laplacian_elevation",
        "profile_curvature",
        "plan_curvature",
        "local_elevation_std_3x3",
        "local_elevation_std_5x5",
        "tpi_3x3",
        "tpi_5x5",
        "neighbour_elevation_diff_mean",
        "neighbour_elevation_diff_min",
        "neighbour_elevation_diff_max",
        "flow_accumulation_proxy",
        "is_building_mask",
    ]
    for k in expected_keys:
        assert k in feats, f"Missing topographic feature key: {k}"
        assert feats[k].shape == (32, 32), f"Shape mismatch for {k}"
        assert not np.isnan(feats[k]).any(), f"NaNs found in feature: {k}"

    # Slope on uniform plane should be strictly positive
    assert np.all(feats["slope_magnitude"] > 0.0)


def test_topographic_nan_handling():
    dem = np.ones((32, 32), dtype=np.float32) * 10.0
    dem[5, 5] = np.nan  # Building / masked cell

    feats = compute_topographic_features(dem, cell_size_m=8.0)
    assert not np.isnan(feats["elevation"]).any(), "NaNs must be cleanly imputed"
    assert feats["is_building_mask"][5, 5] == 1.0
    assert feats["is_building_mask"][0, 0] == 0.0


def test_compute_rainfall_features():
    T, H, W = 20, 32, 32
    # Constant 4.0 mm/5min rain
    rainfall = np.ones((T, H, W), dtype=np.float32) * 4.0

    feats = compute_rainfall_features(rainfall, dt_min=5.0, window_steps=(3, 6, 12))

    assert "rainfall_intensity_current" in feats
    assert "rainfall_cumulative" in feats
    assert "rainfall_window_15m" in feats  # 3 * 5m
    assert "rainfall_window_30m" in feats  # 6 * 5m
    assert "rainfall_window_60m" in feats  # 12 * 5m
    assert "rainfall_max_so_far" in feats
    assert "rainfall_rate_of_change" in feats

    # Test cumulative at last step
    assert np.isclose(feats["rainfall_cumulative"][-1, 0, 0], 20 * 4.0)

    # Test 15-minute rolling window (3 steps * 4.0 = 12.0 mm)
    assert np.isclose(feats["rainfall_window_15m"][5, 0, 0], 12.0)


def test_compute_hydraulic_lag_features():
    T, H, W = 15, 32, 32
    # Linearly rising water depth
    t_steps = np.arange(T, dtype=np.float32)[:, np.newaxis, np.newaxis]
    h = np.broadcast_to(t_steps * 0.05, (T, H, W)).copy()  # Rises 0.05m per timestep
    dem = np.ones((H, W), dtype=np.float32) * 100.0

    feats = compute_hydraulic_lag_features(h, dem, dt_min=5.0, lag_steps=(1, 2, 3))

    assert "current_water_depth" in feats
    assert "water_depth_lag_5m" in feats
    assert "water_depth_lag_10m" in feats
    assert "water_depth_lag_15m" in feats
    assert "water_depth_rate_of_rise" in feats
    assert "hydraulic_head" in feats

    # Check lag causality at t=4: lag 1 (t=3) should be 3 * 0.05 = 0.15m
    assert np.isclose(feats["water_depth_lag_5m"][4, 0, 0], 0.15)
    # At t=0, lag 1 must be 0.0 (antecedent dry condition)
    assert np.isclose(feats["water_depth_lag_5m"][0, 0, 0], 0.0)

    # Rate of rise: 0.05 m per 5 min = 0.01 m/min
    assert np.isclose(feats["water_depth_rate_of_rise"][2, 0, 0], 0.01)


def test_build_event_tabular_dataframe():
    event = generate_synthetic_larno_event("tab_001", T=10, seed=42)
    df = build_event_tabular_dataframe(event)

    expected_rows = 10 * 32 * 32  # 10,240 rows
    assert len(df) == expected_rows

    feature_cols = get_feature_column_names(df)
    assert len(feature_cols) >= 30, f"Expected >= 30 features, got {len(feature_cols)}"

    # Ensure metadata columns exist
    assert "event_id" in df.columns
    assert "timestep_idx" in df.columns
    assert "row_idx" in df.columns
    assert "col_idx" in df.columns

    # Check for no infinite values
    assert not np.isinf(df[feature_cols].to_numpy()).any()

