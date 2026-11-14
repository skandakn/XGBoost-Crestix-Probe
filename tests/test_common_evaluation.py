"""Unit tests for the common-pipeline XGBoost evaluation metrics."""

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.evaluate_common_pipeline import compute_common_metrics


def test_compute_common_metrics_perfect_and_thresholded():
    y_true_depth = np.array([0.0, 0.01, 0.03, 0.10, 0.50])
    y_pred_depth = np.array([0.0, 0.01, 0.03, 0.10, 0.50])
    y_true_flood = (y_true_depth >= 0.03).astype(int)

    res = compute_common_metrics(y_true_depth, y_pred_depth, y_true_flood, flood_threshold_m=0.03)
    assert res["regression"]["mae_m"] == 0.0
    assert res["regression"]["rmse_m"] == 0.0
    assert res["regression"]["r2"] == 1.0
    assert res["classification_threshold_0_03m"]["auroc"] == 1.0
    assert res["classification_threshold_0_03m"]["pr_auc"] == 1.0
    assert res["classification_threshold_0_03m"]["f1_score"] == 1.0
    assert res["classification_threshold_0_03m"]["precision"] == 1.0
    assert res["classification_threshold_0_03m"]["recall"] == 1.0
