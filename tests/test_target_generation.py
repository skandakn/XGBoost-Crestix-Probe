"""
test_target_generation.py
=========================
Unit tests for multi-horizon target generation and hazard classification.
"""

import numpy as np
import pytest

from data_pipeline.feature_engineering import build_event_tabular_dataframe
from data_pipeline.load_data import generate_synthetic_larno_event
from data_pipeline.target_generation import (
    TARGET_DEFINITIONS,
    attach_targets_to_dataframe,
    generate_event_targets,
    get_target_metadata,
)


def test_target_registry_definitions():
    assert "target_depth_lead_30m" in TARGET_DEFINITIONS
    meta = get_target_metadata("target_depth_lead_30m")
    assert meta.target_type == "regression"
    assert meta.units == "meters [m]"
    assert meta.forecast_horizon_min == 30.0
    assert meta.lead_steps == 6


def test_generate_event_targets_multi_horizon():
    T, H, W = 20, 32, 32
    # Known depth sequence: h(t) = t * 0.1
    t_arr = np.arange(T, dtype=np.float32)[:, np.newaxis, np.newaxis]
    h = np.broadcast_to(t_arr * 0.1, (T, H, W)).copy()

    targets = generate_event_targets(h, dt_min=5.0, lead_steps=(1, 3, 6))

    # Lead 1 (+5m): at t=0, target is h(1) = 0.1m
    assert np.isclose(targets["target_depth_lead_5m"][0, 0, 0], 0.1)
    # Lead 6 (+30m): at t=0, target is h(6) = 0.6m
    assert np.isclose(targets["target_depth_lead_30m"][0, 0, 0], 0.6)

    # Lead 6 (+30m): at t=15, target is h(21) which is beyond T=20 -> must be NaN
    assert np.isnan(targets["target_depth_lead_30m"][15, 0, 0])

    # Event peak depth: across all t, max is h(19) = 1.9m
    assert np.allclose(targets["target_event_peak_depth"], 1.9)

    # Peak remaining depth: at t=10, peak from 10 to 19 is h(19) = 1.9m
    assert np.isclose(targets["target_peak_remaining_depth"][10, 0, 0], 1.9)


def test_hazard_classification_thresholds():
    T, H, W = 10, 32, 32
    h = np.zeros((T, H, W), dtype=np.float32)

    # Set specific depths at t=6 so lead_30m (from t=0) has exact test values
    h[6, 0, 0] = 0.01  # Dry (< 0.03m) -> class 0
    h[6, 1, 1] = 0.08  # Shallow (0.03 - 0.15m) -> class 1
    h[6, 2, 2] = 0.22  # Moderate (0.15 - 0.30m) -> class 2
    h[6, 3, 3] = 0.45  # Critical (>= 0.30m) -> class 3

    targets = generate_event_targets(h, dt_min=5.0, lead_steps=(6,))

    hazard = targets["target_hazard_class_lead_30m"]
    assert hazard[0, 0, 0] == 0.0
    assert hazard[0, 1, 1] == 1.0
    assert hazard[0, 2, 2] == 2.0
    assert hazard[0, 3, 3] == 3.0

    flooded = targets["target_is_flooded_lead_30m"]
    assert flooded[0, 0, 0] == 0.0
    assert flooded[0, 1, 1] == 1.0
    assert flooded[0, 2, 2] == 1.0
    assert flooded[0, 3, 3] == 1.0


def test_attach_targets_to_dataframe():
    event = generate_synthetic_larno_event("targ_001", T=20, seed=99)
    df = build_event_tabular_dataframe(event)

    # Attach targets with NaN trimming
    df_with_targets = attach_targets_to_dataframe(
        df, event, drop_nan_targets=True, primary_target_col="target_depth_lead_30m"
    )

    # Lead 30m is 6 steps, so out of 20 timesteps, timesteps 0..13 (14 steps) are kept
    expected_rows = 14 * 32 * 32  # 14,336
    assert len(df_with_targets) == expected_rows
    assert "target_depth_lead_30m" in df_with_targets.columns
    assert not df_with_targets["target_depth_lead_30m"].isna().any()

