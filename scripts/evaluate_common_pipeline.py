"""Reproducible evaluation script for XGBoost on Person 1's common data pipeline.

Aligns XGBoost with the common dataset (`data/sample` or `--data-dir`),
43-feature specification, primary target (`target_depth_lead_30m`),
event-aware split (`EventAwareSplitter` in `configs/baseline.yaml`),
and threshold-based binary flood labels (`target_is_flooded_lead_30m`, >= 0.03 m).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from crestix_xgb import build_training_ready_splits
from crestix_xgb.model import CrestixXGBRegressor
from data_pipeline.validation import check_data_leakage


def compute_common_metrics(
    y_true_depth: np.ndarray,
    y_pred_depth: np.ndarray,
    y_true_flood: np.ndarray,
    flood_threshold_m: float = 0.03,
) -> Dict[str, Any]:
    """Compute regression metrics and threshold-based classification metrics."""
    y_true_d = np.asarray(y_true_depth, dtype=float)
    y_pred_d = np.asarray(y_pred_depth, dtype=float)
    y_true_f = np.asarray(y_true_flood, dtype=int)
    y_pred_f = (y_pred_d >= flood_threshold_m).astype(int)

    mae = float(mean_absolute_error(y_true_d, y_pred_d))
    rmse = float(np.sqrt(mean_squared_error(y_true_d, y_pred_d)))
    r2 = float(r2_score(y_true_d, y_pred_d))
    bias = float(np.mean(y_pred_d - y_true_d))

    has_both_classes = len(np.unique(y_true_f)) == 2
    auroc = float(roc_auc_score(y_true_f, y_pred_d)) if has_both_classes else None
    pr_auc = float(average_precision_score(y_true_f, y_pred_d)) if has_both_classes else None

    f1 = float(f1_score(y_true_f, y_pred_f, zero_division=0))
    precision = float(precision_score(y_true_f, y_pred_f, zero_division=0))
    recall = float(recall_score(y_true_f, y_pred_f, zero_division=0))

    return {
        "regression": {
            "mae_m": round(mae, 6),
            "rmse_m": round(rmse, 6),
            "r2": round(r2, 6),
            "bias_m": round(bias, 6),
        },
        "classification_threshold_0_03m": {
            "flood_threshold_m": flood_threshold_m,
            "positive_rate_true": round(float(np.mean(y_true_f)), 6),
            "positive_rate_pred": round(float(np.mean(y_pred_f)), 6),
            "auroc": round(auroc, 6) if auroc is not None else None,
            "pr_auc": round(pr_auc, 6) if pr_auc is not None else None,
            "f1_score": round(f1, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate XGBoost on Person 1's common dataset, features, target, and split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(repo_root / "configs" / "common_xgb_config.yaml"),
        help="Path to common XGBoost config YAML.",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset directory.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    out_cfg = cfg["outputs"]

    data_dir = args.data_dir or str(repo_root / data_cfg["data_dir"])
    baseline_cfg = str(repo_root / data_cfg["baseline_config"])
    target_col = data_cfg["primary_target_col"]
    flood_col = data_cfg["flood_label_col"]
    threshold_m = float(data_cfg["flood_threshold_m"])

    train_df, val_df, test_df, feature_names = build_training_ready_splits(
        data_dir=data_dir,
        config_path=baseline_cfg,
        primary_target_col=target_col,
    )

    leakage_report = check_data_leakage(
        train_df,
        val_df,
        test_df,
        feature_cols=feature_names,
        target_cols=[target_col, flood_col],
    )

    X_train = train_df[feature_names]
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_val = val_df[feature_names]
    y_val = val_df[target_col].to_numpy(dtype=float)
    X_test = test_df[feature_names]
    y_test = test_df[target_col].to_numpy(dtype=float)
    y_test_flood = test_df[flood_col].to_numpy(dtype=int)
    y_val_flood = val_df[flood_col].to_numpy(dtype=int)

    regressor = CrestixXGBRegressor(
        params=model_cfg["regressor_params"],
        early_stopping_rounds=model_cfg["early_stopping_rounds"],
        version=model_cfg["version"],
    )
    regressor.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        verbose=False,
    )

    val_preds = regressor.predict(X_val)
    test_preds = regressor.predict(X_test)

    val_metrics = compute_common_metrics(y_val, val_preds, y_val_flood, threshold_m)
    test_metrics = compute_common_metrics(y_test, test_preds, y_test_flood, threshold_m)

    summary = {
        "pipeline": "common_data_pipeline_v1",
        "model_version": model_cfg["version"],
        "target_column": target_col,
        "flood_label_column": flood_col,
        "flood_threshold_m": threshold_m,
        "feature_count": len(feature_names),
        "best_iteration": int(regressor.best_iteration_) if regressor.best_iteration_ is not None else None,
        "leakage_check": leakage_report["status"],
        "splits": {
            "train_events": sorted(train_df["event_id"].unique().tolist()),
            "train_rows": int(len(train_df)),
            "val_events": sorted(val_df["event_id"].unique().tolist()),
            "val_rows": int(len(val_df)),
            "test_events": sorted(test_df["event_id"].unique().tolist()),
            "test_rows": int(len(test_df)),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path = repo_root / out_cfg["metrics_json"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    preds_df = pd.DataFrame(
        {
            "event_id": test_df["event_id"].values,
            "timestep": test_df["timestep"].values if "timestep" in test_df.columns else np.arange(len(test_df)),
            "y_true_depth_m": y_test,
            "y_pred_depth_m": test_preds,
            "y_true_flooded": y_test_flood,
            "y_pred_flooded": (test_preds >= threshold_m).astype(int),
        }
    )
    preds_path = repo_root / out_cfg["predictions_csv"]
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(preds_path, index=False)

    metadata = {
        "model_version": model_cfg["version"],
        "model_type": model_cfg["model_type"],
        "feature_order": feature_names,
        "feature_count": len(feature_names),
        "target_column": target_col,
        "flood_label_column": flood_col,
        "flood_threshold_m": threshold_m,
        "regressor_hyperparameters": model_cfg["regressor_params"],
        "early_stopping_rounds": model_cfg["early_stopping_rounds"],
        "best_iteration": summary["best_iteration"],
        "splits": summary["splits"],
        "test_metrics": test_metrics,
    }
    meta_path = repo_root / out_cfg["metadata_json"]
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
