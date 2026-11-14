"""
test_load_data.py
=================
Unit tests for data loading, shape verification, and synthetic generator.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.load_data import (
    FloodEventData,
    create_sample_dataset,
    generate_synthetic_larno_event,
    load_all_events,
    load_event_from_directory,
    save_events_to_disk,
)


def test_flood_event_data_creation_and_properties():
    H, W, T = 32, 32, 24
    dem = np.zeros((H, W), dtype=np.float32)
    rainfall = np.ones((T, H, W), dtype=np.float32) * 2.5
    h = np.ones((T, H, W), dtype=np.float32) * 0.12

    event = FloodEventData(
        event_id="test_01",
        dem=dem,
        rainfall=rainfall,
        h=h,
        timesteps=T,
    )

    assert event.event_id == "test_01"
    assert event.timesteps == 24
    assert event.grid_shape == (32, 32)
    assert np.isclose(event.peak_depth, 0.12)
    assert np.isclose(event.total_rainfall_volume_mm, 2.5 * 24)


def test_flood_event_data_1d_rainfall_broadcasting():
    H, W, T = 32, 32, 10
    dem = np.zeros((H, W), dtype=np.float32)
    rainfall_1d = np.linspace(1.0, 5.0, T, dtype=np.float32)
    h = np.zeros((T, H, W), dtype=np.float32)

    event = FloodEventData(
        event_id="test_broadcast",
        dem=dem,
        rainfall=rainfall_1d,
        h=h,
        timesteps=T,
    )

    assert event.rainfall.shape == (T, H, W)
    assert np.isclose(event.rainfall[0, 5, 5], 1.0)
    assert np.isclose(event.rainfall[-1, 10, 10], 5.0)


def test_flood_event_data_shape_mismatch_raises():
    H, W, T = 32, 32, 10
    dem_wrong = np.zeros((16, 16), dtype=np.float32)
    rainfall = np.zeros((T, H, W), dtype=np.float32)
    h = np.zeros((T, H, W), dtype=np.float32)

    with pytest.raises(ValueError, match="DEM shape mismatch"):
        FloodEventData(event_id="err", dem=dem_wrong, rainfall=rainfall, h=h, timesteps=T)


def test_synthetic_larno_event_generation():
    event = generate_synthetic_larno_event("syn_001", T=36, severity="heavy", seed=123)

    assert event.event_id == "syn_001"
    assert event.timesteps == 36
    assert event.dem.shape == (32, 32)
    assert event.rainfall.shape == (36, 32, 32)
    assert event.h.shape == (36, 32, 32)

    # Physical bounds checks
    assert np.all(event.h >= 0.0), "Water depth must be strictly non-negative"
    assert np.all(event.rainfall >= 0.0), "Rainfall must be non-negative"
    assert not np.isnan(event.dem).any(), "Synthetic DEM should be finite"
    assert event.peak_depth > 0.05, "Heavy storm should induce measurable flood depth"


def test_save_and_reload_events():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        events = [
            generate_synthetic_larno_event("ev_A", T=12, seed=1),
            generate_synthetic_larno_event("ev_B", T=12, seed=2),
        ]

        save_events_to_disk(events, output_dir=tmp_path)

        # Check directory structure
        assert (tmp_path / "dem.npy").exists()
        assert (tmp_path / "ev_A" / "h.npy").exists()
        assert (tmp_path / "ev_A" / "rainfall.npy").exists()
        assert (tmp_path / "ev_B" / "metadata.json").exists()

        # Reload single event
        ev_a_reloaded = load_event_from_directory(tmp_path / "ev_A")
        assert ev_a_reloaded.event_id == "ev_A"
        assert ev_a_reloaded.timesteps == 12

        # Reload all events
        all_reloaded = load_all_events(tmp_path)
        assert len(all_reloaded) == 2
        assert {e.event_id for e in all_reloaded} == {"ev_A", "ev_B"}

