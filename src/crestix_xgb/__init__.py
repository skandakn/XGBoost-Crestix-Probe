"""Crestix Probe - XGBoost Flood-Depth Forecasting Engine

LarNO UKEA Tabular Baseline for Urban Pluvial Flood Prediction.
"""

from crestix_xgb.predict import predict_depth, FloodInferenceEngine
from crestix_xgb.model import CrestixXGBRegressor, CrestixXGBClassifier
from crestix_xgb.evaluate import evaluate_heldout_events, compute_flood_metrics
from crestix_xgb.data import LarNODataPipeline

__version__ = "1.0.0"

__all__ = [
    "predict_depth",
    "FloodInferenceEngine",
    "CrestixXGBRegressor",
    "CrestixXGBClassifier",
    "evaluate_heldout_events",
    "compute_flood_metrics",
    "LarNODataPipeline",
    "__version__",
]

