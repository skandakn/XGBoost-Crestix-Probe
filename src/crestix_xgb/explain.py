"""Explainability and feature importance analysis for Crestix Probe XGBoost.

Computes:
- Gain-based, weight-based, and cover-based feature importances from XGBoost.
- SHAP (SHapley Additive exPlanations) values using TreeExplainer.
- Feature ranking, exportable summary tables, and plots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap

from crestix_xgb.data import FEATURE_NAMES

logger = logging.getLogger(__name__)


def compute_xgboost_importance(
    model,
    feature_names: List[str] = FEATURE_NAMES,
) -> pd.DataFrame:
    """Extracts gain, weight, and cover importances from the trained XGBoost booster."""
    booster = model.get_booster()

    gain_dict = booster.get_score(importance_type="gain")
    weight_dict = booster.get_score(importance_type="weight")
    cover_dict = booster.get_score(importance_type="cover")

    # XGBoost boosters sometimes index features as f0, f1... if feature_names not assigned
    records = []
    for idx, fname in enumerate(feature_names):
        f_key = f"f{idx}"
        gain = gain_dict.get(fname, gain_dict.get(f_key, 0.0))
        weight = weight_dict.get(fname, weight_dict.get(f_key, 0.0))
        cover = cover_dict.get(fname, cover_dict.get(f_key, 0.0))
        records.append({
            "feature": fname,
            "gain": float(gain),
            "weight": float(weight),
            "cover": float(cover),
        })

    df_imp = pd.DataFrame(records)
    # Normalize gain to percentage
    total_gain = df_imp["gain"].sum()
    df_imp["gain_pct"] = (df_imp["gain"] / total_gain * 100.0) if total_gain > 0 else 0.0
    df_imp = df_imp.sort_values(by="gain", ascending=False).reset_index(drop=True)
    return df_imp


def compute_shap_explanations(
    model,
    X_sample: pd.DataFrame | np.ndarray,
    feature_names: List[str] = FEATURE_NAMES,
    max_samples: int = 500,
    random_state: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame, Any]:
    """Computes TreeExplainer SHAP values and mean absolute importance."""
    if isinstance(X_sample, pd.DataFrame):
        X_df = X_sample[feature_names].copy()
    else:
        X_df = pd.DataFrame(X_sample, columns=feature_names)

    if len(X_df) > max_samples:
        X_sub = X_df.sample(n=max_samples, random_state=random_state)
    else:
        X_sub = X_df

    underlying_xgb = model.model
    explainer = shap.TreeExplainer(underlying_xgb)
    shap_values = explainer.shap_values(X_sub)

    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    records = []
    for idx, fname in enumerate(feature_names):
        records.append({
            "feature": fname,
            "mean_abs_shap": float(mean_abs_shap[idx]),
        })

    df_shap = pd.DataFrame(records)
    total_shap = df_shap["mean_abs_shap"].sum()
    df_shap["shap_pct"] = (df_shap["mean_abs_shap"] / total_shap * 100.0) if total_shap > 0 else 0.0
    df_shap = df_shap.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    return shap_values, df_shap, explainer


def save_importance_artifacts(
    df_xgb_imp: pd.DataFrame,
    df_shap_imp: pd.DataFrame,
    shap_values: np.ndarray,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Saves explainability tables and arrays."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xgb_csv = out_dir / "feature_importance_gain.csv"
    shap_csv = out_dir / "feature_importance_shap.csv"
    shap_npy = out_dir / "shap_values_sample.npy"
    summary_json = out_dir / "importance_summary.json"

    df_xgb_imp.to_csv(xgb_csv, index=False)
    df_shap_imp.to_csv(shap_csv, index=False)
    np.save(shap_npy, shap_values)

    summary = {
        "top_features_by_gain": df_xgb_imp["feature"].head(5).tolist(),
        "top_features_by_shap": df_shap_imp["feature"].head(5).tolist(),
        "total_features": len(df_xgb_imp),
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved explainability artifacts to {out_dir}")
    return {
        "xgb_csv": str(xgb_csv),
        "shap_csv": str(shap_csv),
        "shap_npy": str(shap_npy),
        "summary_json": str(summary_json),
    }

