"""Common Prediction Contract & Non-Invasive Model Integration Adapters (Person 9).

Standardizes inputs, units (meters & centimeters), spatial grid coordinates,
event IDs, forecast horizons, and model metadata across:
- XGBoost (`skandakn/XGBoost-Crestix-Probe`: 18-feature `src/crestix_xgb/` & 43-feature Person 1 common schema)
- Linear Regression (`skandakn/Linear-Regression---Crestix-Probe`, pending model training/inference code)
- Random Forest (`skandakn/Random-Forest---Crestix-Probe`: 11-feature `rf_feature_builder.py`, pending trained artifact & inference wrapper)

Does NOT alter internal model implementations.
"""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

# Exact 18-feature canonical order from XGBoost-Crestix-Probe/artifacts/metadata/feature_names.json
CANONICAL_XGB_FEATURE_ORDER: tuple[str, ...] = (
    "elevation_m",
    "slope_pct",
    "aspect_sin",
    "aspect_cos",
    "twi",
    "dist_to_drain_m",
    "impervious_fraction",
    "manning_n",
    "rain_intensity_mmh",
    "rain_acc_15m",
    "rain_acc_30m",
    "rain_acc_1h",
    "rain_acc_3h",
    "rain_acc_6h",
    "depth_lag_15m",
    "depth_lag_30m",
    "depth_rate_of_rise",
    "horizon_min",
)

# Exact 43-feature canonical order from Person 1 common pipeline (`artifacts/metadata/common_model_metadata.json`)
CANONICAL_PERSON1_COMMON_43_FEATURES: tuple[str, ...] = (
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
    "rainfall_intensity_current",
    "rainfall_cumulative",
    "rainfall_max_so_far",
    "rainfall_rate_of_change",
    "rainfall_window_15m",
    "rainfall_mean_15m",
    "rainfall_window_30m",
    "rainfall_mean_30m",
    "rainfall_window_60m",
    "rainfall_mean_60m",
    "current_water_depth",
    "water_depth_rate_of_rise",
    "water_depth_acceleration",
    "neighbour_water_depth_mean_3x3",
    "neighbour_water_depth_max_3x3",
    "hydraulic_head",
    "hydraulic_head_grad_mag",
    "water_depth_lag_5m",
    "water_depth_lag_10m",
    "water_depth_lag_15m",
    "norm_coord_x",
    "norm_coord_y",
    "dist_to_center",
    "time_since_onset_min",
    "time_fraction",
)

# Exact 11-feature canonical order from Random-Forest---Crestix-Probe/rf_feature_builder.py
CANONICAL_RF_11_FEATURES: tuple[str, ...] = (
    "dem",
    "slope_x",
    "slope_y",
    "slope_mag",
    "elev_above_local_min",
    "t_norm",
    "h_init",
    "rain_t",
    "rain_prev1",
    "rain_prev2",
    "rain_cum",
)


class IntegrationContractError(RuntimeError):
    """Base error raised for contract validation or missing integration prerequisites."""


class DependencyNotInstalledError(IntegrationContractError):
    """Raised when a model runtime dependency (e.g., xgboost, sklearn) is missing."""


class ArtifactNotInstalledError(IntegrationContractError):
    """Raised when a required trained model weight file is not downloaded."""


@dataclass(frozen=True)
class StandardizedCellInput:
    """Agreed Common Prediction Contract input for a single grid cell / station & horizon."""

    event_id: str
    grid_row: int
    grid_col: int
    forecast_horizon_min: int
    elevation_m: float
    slope_pct: float
    rain_intensity_mmh: float
    previous_water_depth_m: float
    timestamp_utc: str = ""
    aspect_sin: float = 0.0
    aspect_cos: float = 1.0
    twi: float = 9.5
    dist_to_drain_m: float = 150.0
    impervious_fraction: float = 0.65
    manning_n: float = 0.025
    rain_acc_15m: float = 0.0
    rain_acc_30m: float = 0.0
    rain_acc_1h: float = 0.0
    rain_acc_3h: float = 0.0
    rain_acc_6h: float = 0.0
    depth_lag_15m: Optional[float] = None
    depth_lag_30m: Optional[float] = None
    depth_rate_of_rise: float = 0.0

    def __post_init__(self) -> None:
        if not self.event_id or not str(self.event_id).strip():
            raise ValueError("event_id must be a non-empty string.")
        if self.grid_row < 0 or self.grid_col < 0:
            raise ValueError("grid_row and grid_col must be non-negative integers.")
        if self.forecast_horizon_min <= 0:
            raise ValueError("forecast_horizon_min must be positive.")
        if self.previous_water_depth_m < 0.0:
            raise ValueError("previous_water_depth_m must be non-negative.")

        numeric_fields = (
            self.elevation_m,
            self.slope_pct,
            self.rain_intensity_mmh,
            self.previous_water_depth_m,
            self.aspect_sin,
            self.aspect_cos,
            self.twi,
            self.dist_to_drain_m,
            self.impervious_fraction,
            self.manning_n,
            self.rain_acc_15m,
            self.rain_acc_30m,
            self.rain_acc_1h,
            self.rain_acc_3h,
            self.rain_acc_6h,
            self.depth_rate_of_rise,
        )
        if any(not math.isfinite(float(v)) for v in numeric_fields):
            raise ValueError("All numeric input features must be finite.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StandardizedCellInput":
        """Normalize incoming dictionary (supports depth in cm or m)."""
        data = dict(payload)
        if "previous_water_depth_m" not in data and "previous_water_depth_cm" in data:
            data["previous_water_depth_m"] = float(data.pop("previous_water_depth_cm")) / 100.0
        if "timestamp_utc" not in data or not data["timestamp_utc"]:
            data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        return cls(**data)

    def to_xgb_feature_dict(self) -> dict[str, float]:
        """Map standardized input into the exact 18 features expected by XGBoost-Crestix-Probe."""
        lag_15 = (
            float(self.depth_lag_15m)
            if self.depth_lag_15m is not None
            else float(self.previous_water_depth_m)
        )
        lag_30 = (
            float(self.depth_lag_30m)
            if self.depth_lag_30m is not None
            else lag_15
        )
        return {
            "elevation_m": float(self.elevation_m),
            "slope_pct": float(self.slope_pct),
            "aspect_sin": float(self.aspect_sin),
            "aspect_cos": float(self.aspect_cos),
            "twi": float(self.twi),
            "dist_to_drain_m": float(self.dist_to_drain_m),
            "impervious_fraction": float(self.impervious_fraction),
            "manning_n": float(self.manning_n),
            "rain_intensity_mmh": float(self.rain_intensity_mmh),
            "rain_acc_15m": float(self.rain_acc_15m),
            "rain_acc_30m": float(self.rain_acc_30m),
            "rain_acc_1h": float(self.rain_acc_1h),
            "rain_acc_3h": float(self.rain_acc_3h),
            "rain_acc_6h": float(self.rain_acc_6h),
            "depth_lag_15m": lag_15,
            "depth_lag_30m": lag_30,
            "depth_rate_of_rise": float(self.depth_rate_of_rise),
            "horizon_min": float(self.forecast_horizon_min),
        }

    def to_ordered_feature_vector(
        self, feature_order: Sequence[str] = CANONICAL_XGB_FEATURE_ORDER
    ) -> list[float]:
        """Return feature values strictly aligned to `feature_order`."""
        fmap = self.to_xgb_feature_dict()
        return [fmap[name] for name in feature_order]

    def to_rf_feature_dict(self, time_norm: float = 0.5) -> dict[str, float]:
        """Map standardized input into the 11 features expected by Random-Forest---Crestix-Probe."""
        slope_mag = float(self.slope_pct) / 100.0
        slope_x = slope_mag * float(self.aspect_cos)
        slope_y = slope_mag * float(self.aspect_sin)
        return {
            "dem": float(self.elevation_m),
            "slope_x": slope_x,
            "slope_y": slope_y,
            "slope_mag": slope_mag,
            "elev_above_local_min": 0.0,
            "t_norm": float(time_norm),
            "h_init": float(self.previous_water_depth_m),
            "rain_t": float(self.rain_intensity_mmh),
            "rain_prev1": float(self.rain_intensity_mmh),
            "rain_prev2": float(self.rain_intensity_mmh),
            "rain_cum": float(self.rain_acc_1h),
        }

    def to_rf_ordered_vector(
        self, feature_order: Sequence[str] = CANONICAL_RF_11_FEATURES
    ) -> list[float]:
        """Return Random Forest features strictly aligned to CANONICAL_RF_11_FEATURES."""
        fmap = self.to_rf_feature_dict()
        return [fmap[name] for name in feature_order]


@dataclass(frozen=True)
class StandardizedPredictionOutput:
    """Agreed Common Prediction Contract output retaining event_id, coordinates, and units."""

    event_id: str
    grid_row: int
    grid_col: int
    forecast_horizon_min: int
    predicted_water_depth_m: float
    predicted_water_depth_cm: float
    model_name: str
    model_version: str
    units: str = "m (and cm)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_xgb_feature_order(repo_root: str | Path) -> list[str]:
    """Load `feature_names.json` from XGBoost-Crestix-Probe to verify schema alignment."""
    path = Path(repo_root) / "artifacts" / "metadata" / "feature_names.json"
    if not path.is_file():
        raise FileNotFoundError(f"Expected feature_names.json at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"Invalid feature_names.json format in {path}")
    return [str(f) for f in features]


class XGBoostContractAdapter:
    """Non-invasive integration wrapper around `crestix_xgb.predict.FloodInferenceEngine`."""

    model_name = "xgboost"
    model_version = "1.0.0"

    def __init__(
        self,
        repo_root: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
        engine: Optional[Any] = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
        self.model_path = (
            Path(model_path)
            if model_path
            else self.repo_root / "local_artifacts" / "models" / "xgb_regressor.json"
        )
        self._engine = engine

    def load(self) -> Any:
        """Load the real FloodInferenceEngine if dependencies and checkpoint exist."""
        if self._engine is not None:
            return self._engine

        try:
            importlib.import_module("xgboost")
        except ImportError as exc:
            raise DependencyNotInstalledError(
                "Python package 'xgboost' is not installed; install requirements.txt first."
            ) from exc

        if not self.model_path.is_file():
            raise ArtifactNotInstalledError(
                f"Trained XGBoost weights not found at '{self.model_path}'. "
                "Download xgb_regressor.json per docs/ARTIFACT_MANIFEST.md."
            )

        from crestix_xgb.predict import FloodInferenceEngine  # type: ignore[import-not-found]

        meta_path = self.repo_root / "artifacts" / "metadata" / "model_metadata.json"
        self._engine = FloodInferenceEngine(
            model_path=self.model_path,
            metadata_path=meta_path if meta_path.is_file() else None,
        )
        return self._engine

    def predict_batch(
        self, inputs: Sequence[StandardizedCellInput]
    ) -> list[StandardizedPredictionOutput]:
        """Run standardized batch inference while preserving event_id and grid coordinates."""
        if not inputs:
            raise ValueError("inputs must contain at least one StandardizedCellInput.")

        engine = self.load()
        feature_dicts = [item.to_xgb_feature_dict() for item in inputs]
        raw_depths_m = np.asarray(engine.predict(feature_dicts), dtype=float).ravel()

        return _format_prediction_outputs(
            inputs=inputs,
            raw_depths_m=raw_depths_m,
            model_name=self.model_name,
            model_version=self.model_version,
        )


class RandomForestContractAdapter:
    """Non-invasive integration wrapper for scikit-learn RandomForestRegressor models."""

    model_name = "random_forest"
    model_version = "1.0.0"

    def __init__(
        self,
        repo_root: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
        estimator: Optional[Any] = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
        self.model_path = (
            Path(model_path)
            if model_path
            else self.repo_root / "artifacts" / "random_forest_model.joblib"
        )
        self._estimator = estimator

    def load(self) -> Any:
        """Load trained scikit-learn RandomForestRegressor via joblib when installed."""
        if self._estimator is not None:
            return self._estimator

        try:
            importlib.import_module("sklearn")
            joblib = importlib.import_module("joblib")
        except ImportError as exc:
            raise DependencyNotInstalledError(
                "Python packages 'scikit-learn' and 'joblib' are required for Random Forest."
            ) from exc

        if not self.model_path.is_file():
            raise ArtifactNotInstalledError(
                f"Trained Random Forest artifact not found at '{self.model_path}'. "
                "Provide artifacts/random_forest_model.joblib per docs/ARTIFACT_MANIFEST.md."
            )

        loaded = joblib.load(self.model_path)
        self._estimator = loaded["model"] if isinstance(loaded, Mapping) else loaded
        return self._estimator

    def predict_batch(
        self, inputs: Sequence[StandardizedCellInput]
    ) -> list[StandardizedPredictionOutput]:
        """Run standardized batch prediction while preserving event_id and grid coordinates."""
        if not inputs:
            raise ValueError("inputs must contain at least one StandardizedCellInput.")

        estimator = self.load()
        matrix = np.asarray([item.to_rf_ordered_vector() for item in inputs], dtype=np.float32)
        raw_depths_m = np.asarray(estimator.predict(matrix), dtype=float).ravel()

        return _format_prediction_outputs(
            inputs=inputs,
            raw_depths_m=raw_depths_m,
            model_name=self.model_name,
            model_version=self.model_version,
        )


def _format_prediction_outputs(
    inputs: Sequence[StandardizedCellInput],
    raw_depths_m: np.ndarray,
    model_name: str,
    model_version: str,
) -> list[StandardizedPredictionOutput]:
    """Validate prediction length and attach event_id, grid coordinates, and depth units."""
    if raw_depths_m.shape[0] != len(inputs):
        raise IntegrationContractError(
            f"Prediction length mismatch: expected {len(inputs)}, got {raw_depths_m.shape[0]}."
        )

    outputs: list[StandardizedPredictionOutput] = []
    for item, depth_m in zip(inputs, raw_depths_m):
        clamped_m = round(max(0.0, float(depth_m)), 6)
        outputs.append(
            StandardizedPredictionOutput(
                event_id=item.event_id,
                grid_row=item.grid_row,
                grid_col=item.grid_col,
                forecast_horizon_min=item.forecast_horizon_min,
                predicted_water_depth_m=clamped_m,
                predicted_water_depth_cm=round(clamped_m * 100.0, 4),
                model_name=model_name,
                model_version=model_version,
            )
        )
    return outputs
