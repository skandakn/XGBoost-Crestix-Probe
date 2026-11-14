"""Inference interface for Crestix Probe XGBoost flood-depth forecasting.

Provides:
- predict_depth(features) -> predictions: Clean, canonical top-level function.
- FloodInferenceEngine: High-performance inference class with schema validation,
  exact feature reordering, physical constraints, and risk categorisation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from crestix_xgb.data import (
    FEATURE_NAMES,
    discretize_flood_risk,
    CLASS_LABELS,
    RISK_THRESHOLDS,
)
from crestix_xgb.model import CrestixXGBRegressor, CrestixXGBClassifier

logger = logging.getLogger(__name__)

_DEFAULT_ENGINE: Optional["FloodInferenceEngine"] = None


class FloodInferenceEngine:
    """Production-grade inference engine guaranteeing strict feature alignment and validation."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        metadata_path: Optional[str | Path] = None,
        classifier_path: Optional[str | Path] = None,
    ):
        self.feature_names = list(FEATURE_NAMES)
        self.preprocessor_stats: Dict[str, Dict[str, float]] = {}
        self.regressor = CrestixXGBRegressor(version="1.0.0")
        self.classifier: Optional[CrestixXGBClassifier] = None

        # Resolve default paths if none provided
        base_dir = Path(__file__).resolve().parent.parent.parent
        default_meta = base_dir / "artifacts" / "metadata" / "model_metadata.json"
        default_reg = base_dir / "local_artifacts" / "models" / "xgb_regressor.json"
        default_clf = base_dir / "local_artifacts" / "models" / "xgb_classifier.json"

        meta_file = Path(metadata_path) if metadata_path else default_meta
        reg_file = Path(model_path) if model_path else default_reg
        clf_file = Path(classifier_path) if classifier_path else default_clf

        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
                if "feature_order" in meta:
                    self.feature_names = meta["feature_order"]
                if "preprocessor_stats" in meta:
                    self.preprocessor_stats = meta["preprocessor_stats"]

        if reg_file.exists():
            self.regressor.load(reg_file, feature_names=self.feature_names)
        else:
            logger.warning(
                f"Regressor model file not found at {reg_file}. "
                "Inference requires fitting or loading a trained checkpoint."
            )

        if clf_file.exists():
            self.classifier = CrestixXGBClassifier(version="1.0.0")
            self.classifier.load(clf_file, feature_names=self.feature_names)

    def validate_and_align_features(
        self, features: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], np.ndarray, List[float]]
    ) -> pd.DataFrame:
        """Validates input schema and strictly re-orders columns to match training order."""
        if isinstance(features, dict):
            # Single observation dictionary
            df = pd.DataFrame([features])
        elif isinstance(features, list):
            if len(features) > 0 and isinstance(features[0], dict):
                # List of dictionaries
                df = pd.DataFrame(features)
            else:
                # 1D list of floats for a single observation
                arr = np.asarray(features, dtype=float)
                if arr.ndim == 1:
                    if len(arr) != len(self.feature_names):
                        raise ValueError(
                            f"Input feature vector has {len(arr)} values, "
                            f"but exactly {len(self.feature_names)} features are expected: {self.feature_names}"
                        )
                    df = pd.DataFrame([arr], columns=self.feature_names)
                else:
                    df = pd.DataFrame(arr, columns=self.feature_names)
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                if len(features) != len(self.feature_names):
                    raise ValueError(
                        f"Input 1D numpy array has shape {features.shape}, "
                        f"expected ({len(self.feature_names)},)"
                    )
                df = pd.DataFrame([features], columns=self.feature_names)
            elif features.ndim == 2:
                if features.shape[1] != len(self.feature_names):
                    raise ValueError(
                        f"Input 2D numpy array has shape {features.shape}, "
                        f"expected (N, {len(self.feature_names)})"
                    )
                df = pd.DataFrame(features, columns=self.feature_names)
            else:
                raise ValueError(f"Unsupported numpy array dimensions: {features.ndim}")
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise TypeError(f"Unsupported input type for features: {type(features)}")

        # Check for missing required features when DataFrame / dict was provided
        missing_cols = [col for col in self.feature_names if col not in df.columns]
        if missing_cols:
            # If preprocessor stats have fallback medians, impute missing columns with a warning
            if self.preprocessor_stats:
                for col in missing_cols:
                    default_val = self.preprocessor_stats.get(col, {}).get("median", 0.0)
                    df[col] = default_val
            else:
                raise ValueError(
                    f"Missing required features for flood-depth inference: {missing_cols}. "
                    f"Expected features: {self.feature_names}"
                )

        # Enforce exact column order as during training
        aligned_df = df[self.feature_names].astype(float)

        # Check for NaNs and impute with training medians
        if aligned_df.isna().any().any():
            for col in self.feature_names:
                if aligned_df[col].isna().any():
                    median_val = self.preprocessor_stats.get(col, {}).get("median", 0.0)
                    aligned_df[col] = aligned_df[col].fillna(median_val)

        return aligned_df

    def predict(
        self, features: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], np.ndarray, List[float]]
    ) -> np.ndarray:
        """Predicts water depth in meters. Enforces non-negativity."""
        X_aligned = self.validate_and_align_features(features)
        preds = self.regressor.predict(X_aligned)
        return np.maximum(0.0, preds)

    def predict_risk(
        self, features: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], np.ndarray, List[float]]
    ) -> Dict[str, Any]:
        """Provides full multi-task prediction: continuous depth, discrete category, and hazard labels."""
        X_aligned = self.validate_and_align_features(features)
        predicted_depths = self.predict(X_aligned)
        derived_classes = discretize_flood_risk(predicted_depths)
        derived_labels = [CLASS_LABELS[c] for c in derived_classes]

        output: Dict[str, Any] = {
            "predicted_depth_m": predicted_depths,
            "risk_class": derived_classes,
            "risk_label": derived_labels,
        }

        if self.classifier is not None:
            clf_classes = self.classifier.predict(X_aligned)
            clf_probs = self.classifier.predict_proba(X_aligned)
            output["classifier_class"] = clf_classes
            output["classifier_probabilities"] = clf_probs

        return output


def get_default_engine() -> FloodInferenceEngine:
    """Returns or instantiates the singleton FloodInferenceEngine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = FloodInferenceEngine()
    return _DEFAULT_ENGINE


def set_default_engine(engine: FloodInferenceEngine) -> None:
    """Sets the global default engine (e.g. after training completes in memory)."""
    global _DEFAULT_ENGINE
    _DEFAULT_ENGINE = engine


def predict_depth(
    features: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], np.ndarray, List[float]],
    engine: Optional[FloodInferenceEngine] = None,
) -> np.ndarray:
    """Clean canonical inference interface for Crestix Probe flood-depth forecasting.

    Usage:
      predictions = predict_depth(features)

    Args:
      features: Single dictionary of feature values, list of dicts,
                DataFrame, or numpy feature matrix.
      engine: Optional preloaded FloodInferenceEngine.

    Returns:
      1D numpy array of predicted water depths in meters (guaranteed >= 0.0).
    """
    eng = engine or get_default_engine()
    return eng.predict(features)

