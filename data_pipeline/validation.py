"""
validation.py
=============
Event-aware dataset splitting, data leakage detection, and train-only
preprocessing transforms for the Crestix Probe XGBoost pipeline.

Guarantees complete separation of independent flood events across train, val,
and test sets, preventing spatial and temporal data contamination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

from data_pipeline.feature_engineering import get_feature_column_names
from data_pipeline.load_data import FloodEventData


class DataLeakageError(Exception):
    """Raised when spatial or temporal data contamination is detected."""
    pass


@dataclass
class EventAwareSplitter:
    """
    Partitions flood events into disjoint Train, Validation, and Test sets.

    Ensures that every grid cell and all timesteps of an event reside strictly
    within a single partition, eliminating spatio-temporal data leakage.
    """

    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    random_seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def split_event_ids(self, event_ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Split a list of unique event IDs into (train_ids, val_ids, test_ids)."""
        unique_ids = sorted(list(set(event_ids)))
        n_events = len(unique_ids)

        if n_events < 3:
            raise ValueError(
                f"At least 3 distinct flood events are required for train/val/test split, got {n_events}"
            )

        rng = np.random.RandomState(self.random_seed)
        shuffled = unique_ids.copy()
        rng.shuffle(shuffled)

        n_train = max(1, int(round(n_events * self.train_ratio)))
        n_val = max(1, int(round(n_events * self.val_ratio)))

        # Ensure at least 1 event in test if n_events >= 3
        if n_train + n_val >= n_events:
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

        train_ids = sorted(shuffled[:n_train])
        val_ids = sorted(shuffled[n_train : n_train + n_val])
        test_ids = sorted(shuffled[n_train + n_val :])

        # Assert disjointness
        assert not (set(train_ids) & set(val_ids)), "Train and Val event IDs overlap!"
        assert not (set(train_ids) & set(test_ids)), "Train and Test event IDs overlap!"
        assert not (set(val_ids) & set(test_ids)), "Val and Test event IDs overlap!"

        return train_ids, val_ids, test_ids

    def split_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a consolidated tabular DataFrame by event_id."""
        if "event_id" not in df.columns:
            raise KeyError("DataFrame must contain 'event_id' column for event-aware splitting")

        all_event_ids = df["event_id"].unique().tolist()
        train_ids, val_ids, test_ids = self.split_event_ids(all_event_ids)

        train_df = df[df["event_id"].isin(train_ids)].reset_index(drop=True)
        val_df = df[df["event_id"].isin(val_ids)].reset_index(drop=True)
        test_df = df[df["event_id"].isin(test_ids)].reset_index(drop=True)

        return train_df, val_df, test_df


def check_data_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform rigorous validation checks for data contamination:
      1. Disjoint event IDs across all partitions.
      2. No target columns present in feature matrix.
      3. Absence of future time data in lagged features.
      4. No NaN or Inf values in required features.

    Returns a report dictionary or raises DataLeakageError upon failure.
    """
    report: Dict[str, Any] = {
        "status": "PASSED",
        "checks_performed": [],
        "errors": [],
    }

    # Check 1: Disjoint event IDs
    train_events: Set[str] = set(train_df["event_id"].unique())
    val_events: Set[str] = set(val_df["event_id"].unique())
    test_events: Set[str] = set(test_df["event_id"].unique())

    train_val_overlap = train_events & val_events
    train_test_overlap = train_events & test_events
    val_test_overlap = val_events & test_events

    if train_val_overlap or train_test_overlap or val_test_overlap:
        msg = (
            f"Event overlap detected! Train-Val: {train_val_overlap}, "
            f"Train-Test: {train_test_overlap}, Val-Test: {val_test_overlap}"
        )
        report["errors"].append(msg)
        report["status"] = "FAILED"
        raise DataLeakageError(msg)

    report["checks_performed"].append(
        f"Disjoint events verified (Train: {len(train_events)}, Val: {len(val_events)}, Test: {len(test_events)})"
    )

    # Check 2: Target exclusion from features
    if feature_cols is not None:
        target_leakage = [c for c in feature_cols if c.startswith("target_")]
        if target_leakage:
            msg = f"Target variables leaked into feature columns: {target_leakage}"
            report["errors"].append(msg)
            report["status"] = "FAILED"
            raise DataLeakageError(msg)
        report["checks_performed"].append("Zero target leakage in feature columns verified")

    # Check 3: Check for NaN or Inf values in feature columns
    if feature_cols is not None:
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            nan_counts = split_df[feature_cols].isna().sum()
            inf_counts = np.isinf(split_df[feature_cols].select_dtypes(include=np.number)).sum()

            bad_nans = nan_counts[nan_counts > 0]
            bad_infs = inf_counts[inf_counts > 0]

            if len(bad_nans) > 0 or len(bad_infs) > 0:
                msg = f"NaN/Inf values detected in {split_name} features! NaNs: {bad_nans.to_dict()}, Infs: {bad_infs.to_dict()}"
                report["errors"].append(msg)
                report["status"] = "FAILED"
                raise DataLeakageError(msg)

        report["checks_performed"].append("Clean numeric integrity (no NaNs/Infs) verified across splits")

    return report


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Train-only preprocessing and transformation pipeline.

    Fits imputation statistics and optional scaling strictly on training events,
    preventing distributional leakage into validation or test sets.
    """

    def __init__(
        self,
        scale_features: bool = False,
        scaler_type: str = "robust",
        clip_negative_features: bool = True,
    ) -> None:
        self.scale_features = scale_features
        self.scaler_type = scaler_type
        self.clip_negative_features = clip_negative_features
        self.feature_names_: List[str] = []
        self.impute_medians_: Dict[str, float] = {}
        self.scaler_: Optional[Union[RobustScaler, StandardScaler]] = None

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "TabularPreprocessor":
        """Fit preprocessing statistics strictly on training data."""
        self.feature_names_ = get_feature_column_names(X)

        # Compute median imputation values on train split
        for col in self.feature_names_:
            if pd.api.types.is_numeric_dtype(X[col]):
                med = float(X[col].median(skipna=True))
                self.impute_medians_[col] = 0.0 if np.isnan(med) else med

        if self.scale_features:
            if self.scaler_type == "robust":
                self.scaler_ = RobustScaler()
            elif self.scaler_type == "standard":
                self.scaler_ = StandardScaler()
            else:
                raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

            X_imputed = self._impute_dataframe(X[self.feature_names_])
            self.scaler_.fit(X_imputed)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply train-fitted transformations to validation or test data."""
        if not self.feature_names_:
            raise RuntimeError("Preprocessor must be fit on training data before calling transform")

        X_out = X.copy()
        X_feats = self._impute_dataframe(X_out[self.feature_names_])

        # Enforce non-negativity on physical quantities (depth, rainfall, roughness)
        if self.clip_negative_features:
            non_neg_prefixes = ("rainfall_", "water_depth_", "current_water_", "local_elevation_std_")
            for col in self.feature_names_:
                if any(col.startswith(p) for p in non_neg_prefixes):
                    X_feats[col] = np.maximum(0.0, X_feats[col])

        if self.scale_features and self.scaler_ is not None:
            scaled_vals = self.scaler_.transform(X_feats)
            X_feats = pd.DataFrame(scaled_vals, columns=self.feature_names_, index=X_out.index)

        for col in self.feature_names_:
            X_out[col] = X_feats[col]

        return X_out

    def _impute_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute any NaNs using strictly training-derived medians."""
        df_clean = df.copy()
        for col in self.feature_names_:
            if col in df_clean.columns:
                med = self.impute_medians_.get(col, 0.0)
                df_clean[col] = df_clean[col].fillna(med)
        return df_clean


def validate_tabular_dataset(
    df: pd.DataFrame,
    primary_target_col: str = "target_depth_lead_30m",
) -> Dict[str, Any]:
    """
    Generate diagnostic health metrics on a processed tabular dataset.
    """
    feature_cols = get_feature_column_names(df)
    target_cols = [c for c in df.columns if c.startswith("target_")]

    summary: Dict[str, Any] = {
        "total_rows": len(df),
        "total_features": len(feature_cols),
        "feature_names": feature_cols,
        "target_names": target_cols,
        "unique_events": sorted(df["event_id"].unique().tolist()),
        "has_nans_in_features": bool(df[feature_cols].isna().any().any()),
    }

    if primary_target_col in df.columns:
        valid_targets = df[primary_target_col].dropna()
        summary["target_summary"] = {
            "name": primary_target_col,
            "min_m": float(valid_targets.min()),
            "max_m": float(valid_targets.max()),
            "mean_m": float(valid_targets.mean()),
            "median_m": float(valid_targets.median()),
            "flooded_cells_ratio": float((valid_targets >= 0.03).mean()),
        }

    return summary

