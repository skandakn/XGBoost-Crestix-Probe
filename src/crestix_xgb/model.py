"""Model definitions and wrappers for Crestix Probe XGBoost forecasting.

Provides:
- CrestixXGBRegressor: Continuous water-depth regression model.
- CrestixXGBClassifier: Multi-class flood hazard risk classification model.
- Model serialization, schema management, and versioning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


DEFAULT_REGRESSOR_PARAMS: Dict[str, Any] = {
    "n_estimators": 450,
    "max_depth": 6,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 4,
    "gamma": 0.05,
    "reg_alpha": 0.15,
    "reg_lambda": 1.2,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_CLASSIFIER_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 4,
    "random_state": 42,
    "n_jobs": -1,
}


class CrestixXGBRegressor:
    """XGBRegressor wrapper for continuous LarNO urban flood-depth forecasting."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 35,
        version: str = "1.0.0",
    ):
        self.params = {**DEFAULT_REGRESSOR_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.version = version
        self.feature_names_: Optional[List[str]] = None
        self.model: Optional[xgb.XGBRegressor] = None
        self.best_iteration_: Optional[int] = None
        self.evals_result_: Dict[str, Dict[str, List[float]]] = {}

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> "CrestixXGBRegressor":
        """Fits the XGBoost regressor using early stopping on the validation set."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
            X_tr = X_train.values
        else:
            self.feature_names_ = feature_names
            X_tr = np.asarray(X_train)

        y_tr = np.asarray(y_train, dtype=float)

        fit_params = dict(self.params)
        if X_val is not None and y_val is not None:
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBRegressor(**fit_params)

        eval_set = []
        eval_names = []
        eval_set.append((X_tr, y_tr))
        eval_names.append("train")

        if X_val is not None and y_val is not None:
            X_v = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)
            y_v = np.asarray(y_val, dtype=float)
            eval_set.append((X_v, y_v))
            eval_names.append("validation")

        self.model.fit(
            X_tr,
            y_tr,
            eval_set=eval_set,
            verbose=verbose,
        )

        self.evals_result_ = self.model.evals_result()
        if hasattr(self.model, "best_iteration"):
            self.best_iteration_ = self.model.best_iteration

        logger.info(
            f"Fitted CrestixXGBRegressor v{self.version}. "
            f"Best iteration: {self.best_iteration_}"
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts continuous flood water depth (meters), clipping at zero."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")

        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None:
                # Guarantee exact feature ordering
                X_arr = X[self.feature_names_].values
            else:
                X_arr = X.values
        else:
            X_arr = np.asarray(X)

        raw_preds = self.model.predict(X_arr)
        # Physical constraint: water depth in meters is strictly non-negative
        return np.maximum(0.0, raw_preds)

    def get_booster(self) -> xgb.Booster:
        """Returns the underlying native XGBoost Booster."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        return self.model.get_booster()

    def save(self, filepath: str | Path) -> None:
        """Saves model in native XGBoost JSON format."""
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"Saved CrestixXGBRegressor model to {path}")

    def load(self, filepath: str | Path, feature_names: Optional[List[str]] = None) -> None:
        """Loads model from native XGBoost JSON or UBJ file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path))
        if feature_names is not None:
            self.feature_names_ = feature_names
        logger.info(f"Loaded CrestixXGBRegressor model from {path}")


class CrestixXGBClassifier:
    """XGBClassifier wrapper for multi-class flood hazard risk categorisation."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 35,
        version: str = "1.0.0",
    ):
        self.params = {**DEFAULT_CLASSIFIER_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.version = version
        self.feature_names_: Optional[List[str]] = None
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_iteration_: Optional[int] = None

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> "CrestixXGBClassifier":
        """Fits the XGBoost multi-class classifier with early stopping."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
            X_tr = X_train.values
        else:
            self.feature_names_ = feature_names
            X_tr = np.asarray(X_train)

        y_tr = np.asarray(y_train, dtype=int)

        fit_params = dict(self.params)
        if X_val is not None and y_val is not None:
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBClassifier(**fit_params)

        eval_set = [(X_tr, y_tr)]
        if X_val is not None and y_val is not None:
            X_v = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)
            y_v = np.asarray(y_val, dtype=int)
            eval_set.append((X_v, y_v))

        self.model.fit(
            X_tr,
            y_tr,
            eval_set=eval_set,
            verbose=verbose,
        )

        if hasattr(self.model, "best_iteration"):
            self.best_iteration_ = self.model.best_iteration
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts discrete risk categories (0 to 3)."""
        if self.model is None:
            raise RuntimeError("Classifier has not been trained or loaded.")
        if isinstance(X, pd.DataFrame) and self.feature_names_ is not None:
            X_arr = X[self.feature_names_].values
        else:
            X_arr = np.asarray(X)
        return self.model.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts class probabilities across [None, Low, Moderate, High]."""
        if self.model is None:
            raise RuntimeError("Classifier has not been trained or loaded.")
        if isinstance(X, pd.DataFrame) and self.feature_names_ is not None:
            X_arr = X[self.feature_names_].values
        else:
            X_arr = np.asarray(X)
        return self.model.predict_proba(X_arr)

    def save(self, filepath: str | Path) -> None:
        """Saves classifier in native XGBoost format."""
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, filepath: str | Path, feature_names: Optional[List[str]] = None) -> None:
        """Loads classifier from native XGBoost file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        if feature_names is not None:
            self.feature_names_ = feature_names

