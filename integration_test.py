"""Integration readiness tests for Crestix Probe ML models (Person 9).

Verifies:
1. Standardized input contract validation and unit conversion (cm <-> m).
2. Exact 18-feature ordering match against XGBoost-Crestix-Probe metadata.
3. Spatial output preservation of `event_id`, `(grid_row, grid_col)`, `forecast_horizon_min`,
   `model_name`, and `model_version` using an explicitly marked test double.
4. Honest error handling when weights (`xgb_regressor.json`) or `xgboost` package are missing.
5. Live `FloodInferenceEngine` execution when both `xgboost` and `xgb_regressor.json` are present
   (skipped cleanly otherwise so we never fake live inference).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from integration.common_contract import (
    CANONICAL_XGB_FEATURE_ORDER,
    ArtifactNotInstalledError,
    DependencyNotInstalledError,
    StandardizedCellInput,
    XGBoostContractAdapter,
    load_xgb_feature_order,
)


REPO_ROOT = Path(__file__).resolve().parent
HAS_XGBOOST_PKG = importlib.util.find_spec("xgboost") is not None
XGB_WEIGHTS_PATH = REPO_ROOT / "local_artifacts" / "models" / "xgb_regressor.json"


class _MockEngine:
    """Explicit test double used only when verifying contract output preservation."""

    def __init__(self) -> None:
        self.received_features: list[dict[str, float]] = []

    def predict(self, features: list[dict[str, float]]) -> np.ndarray:
        self.received_features = features
        return np.array(
            [max(0.0, f["depth_lag_15m"] + 0.002 * f["rain_intensity_mmh"]) for f in features],
            dtype=float,
        )


def test_common_input_contract_validation_and_unit_normalization() -> None:
    cell = StandardizedCellInput.from_dict(
        {
            "event_id": "event_09_test_heldout_20yr_benchmark",
            "grid_row": 12,
            "grid_col": 19,
            "forecast_horizon_min": 30,
            "elevation_m": 24.5,
            "slope_pct": 3.2,
            "rain_intensity_mmh": 38.0,
            "previous_water_depth_cm": 15.0,  # converted to 0.15 m
        }
    )
    assert cell.previous_water_depth_m == pytest.approx(0.15)
    assert cell.timestamp_utc != ""

    ordered = cell.to_ordered_feature_vector()
    assert len(ordered) == len(CANONICAL_XGB_FEATURE_ORDER) == 18

    with pytest.raises(ValueError, match="event_id"):
        StandardizedCellInput(
            event_id="",
            grid_row=0,
            grid_col=0,
            forecast_horizon_min=30,
            elevation_m=20.0,
            slope_pct=1.0,
            rain_intensity_mmh=10.0,
            previous_water_depth_m=0.1,
        )

    with pytest.raises(ValueError, match="non-negative"):
        StandardizedCellInput(
            event_id="EV-1",
            grid_row=-1,
            grid_col=0,
            forecast_horizon_min=30,
            elevation_m=20.0,
            slope_pct=1.0,
            rain_intensity_mmh=10.0,
            previous_water_depth_m=0.1,
        )


def test_xgboost_feature_names_match_canonical_order_if_present() -> None:
    feature_meta = REPO_ROOT / "artifacts" / "metadata" / "feature_names.json"
    if not feature_meta.is_file():
        pytest.skip("artifacts/metadata/feature_names.json not present in this repository.")
    loaded_order = load_xgb_feature_order(REPO_ROOT)
    assert tuple(loaded_order) == CANONICAL_XGB_FEATURE_ORDER


def test_spatial_predictions_retain_event_id_and_grid_coordinates() -> None:
    mock_engine = _MockEngine()
    adapter = XGBoostContractAdapter(repo_root=REPO_ROOT, engine=mock_engine)

    grid_inputs = [
        StandardizedCellInput(
            event_id="event_10_test_heldout_100yr_design_flood",
            grid_row=r,
            grid_col=c,
            forecast_horizon_min=60,
            elevation_m=22.0 + r * 0.5,
            slope_pct=2.8,
            rain_intensity_mmh=65.0,
            previous_water_depth_m=0.12,
        )
        for r in range(2)
        for c in range(3)
    ]

    outputs = adapter.predict_batch(grid_inputs)
    assert len(outputs) == 6
    for inp, out in zip(grid_inputs, outputs):
        assert out.event_id == inp.event_id
        assert (out.grid_row, out.grid_col) == (inp.grid_row, inp.grid_col)
        assert out.forecast_horizon_min == 60
        assert out.model_name == "xgboost"
        assert out.model_version == "1.0.0"
        assert out.predicted_water_depth_cm == pytest.approx(out.predicted_water_depth_m * 100.0)
    assert list(mock_engine.received_features[0].keys()) == list(CANONICAL_XGB_FEATURE_ORDER)


def test_missing_dependency_or_weights_raises_clear_error(tmp_path: Path) -> None:
    adapter = XGBoostContractAdapter(
        repo_root=REPO_ROOT,
        model_path=tmp_path / "nonexistent_xgb_regressor.json",
    )
    sample = [
        StandardizedCellInput(
            event_id="EV-ERR",
            grid_row=0,
            grid_col=0,
            forecast_horizon_min=30,
            elevation_m=25.0,
            slope_pct=2.0,
            rain_intensity_mmh=25.0,
            previous_water_depth_m=0.05,
        )
    ]
    expected_exc = ArtifactNotInstalledError if HAS_XGBOOST_PKG else DependencyNotInstalledError
    with pytest.raises(expected_exc):
        adapter.predict_batch(sample)


def test_sample_input_output_fixture_validates() -> None:
    sample_file = REPO_ROOT / "samples" / "sample_input_output.json"
    if not sample_file.is_file():
        pytest.skip("samples/sample_input_output.json not found.")
    payload = json.loads(sample_file.read_text(encoding="utf-8"))
    for row in payload["sample_inputs"]:
        parsed = StandardizedCellInput.from_dict(row)
        assert len(parsed.to_ordered_feature_vector()) == 18


@pytest.mark.skipif(
    not (HAS_XGBOOST_PKG and XGB_WEIGHTS_PATH.is_file()),
    reason="Requires xgboost package and downloaded local_artifacts/models/xgb_regressor.json",
)
def test_live_xgboost_engine_load_and_predict() -> None:
    adapter = XGBoostContractAdapter(repo_root=REPO_ROOT, model_path=XGB_WEIGHTS_PATH)
    sample = [
        StandardizedCellInput(
            event_id="event_001",
            grid_row=10,
            grid_col=14,
            forecast_horizon_min=30,
            elevation_m=28.0,
            slope_pct=3.5,
            rain_intensity_mmh=40.0,
            previous_water_depth_m=0.08,
        )
    ]
    out = adapter.predict_batch(sample)
    assert len(out) == 1
    assert out[0].predicted_water_depth_m >= 0.0
