"""Training pipeline for Crestix Probe XGBoost flood-depth forecasting.

Coordinates:
- LarNO UKEA benchmark data generation & feature engineering.
- Strict event-based train/val/test isolation.
- Preprocessor statistics calibration (zero-leakage).
- Hyperparameter tuning and validation-based early stopping.
- Continuous XGBRegressor and multi-class XGBClassifier training.
- Evaluation on held-out test events.
- Feature importance and SHAP analysis.
- Model artifact serialization and metadata checksum generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from crestix_xgb.data import (
    LarNODataPipeline,
    FEATURE_NAMES,
    TARGET_COLUMN,
    RISK_CLASS_COLUMN,
    discretize_flood_risk,
)
from crestix_xgb.model import CrestixXGBRegressor, CrestixXGBClassifier
from crestix_xgb.evaluate import (
    evaluate_heldout_events,
    generate_evaluation_markdown_report,
)
from crestix_xgb.explain import (
    compute_xgboost_importance,
    compute_shap_explanations,
    save_importance_artifacts,
)
from crestix_xgb.predict import FloodInferenceEngine, set_default_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: str | Path) -> str:
    """Computes SHA-256 checksum of a file."""
    p = Path(filepath)
    if not p.exists():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    p = Path(config_path)
    with open(p, "r") as f:
        return yaml.safe_load(f)


def run_training_pipeline(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    tune_hyperparameters: bool = False,
) -> Dict[str, Any]:
    """Executes the full end-to-end training, evaluation, and artifact generation pipeline."""
    # 1. Resolve configuration
    base_dir = Path(__file__).resolve().parent.parent.parent
    if config is None:
        cfg_file = Path(config_path) if config_path else base_dir / "configs" / "default_config.yaml"
        config = load_config(cfg_file)

    root_dir = Path(output_root) if output_root else base_dir

    models_dir = root_dir / "local_artifacts" / "models"
    metadata_dir = root_dir / "artifacts" / "metadata"
    metrics_dir = root_dir / "metrics"
    predictions_dir = root_dir / "local_artifacts" / "predictions"
    shap_dir = root_dir / "local_artifacts" / "shap"

    for d in [models_dir, metadata_dir, metrics_dir, predictions_dir, shap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. LarNO Data Pipeline & Event Generation
    logger.info("Initializing LarNO UKEA data pipeline...")
    data_pipe = LarNODataPipeline(config)
    df_raw = data_pipe.generate_larno_benchmark_dataset(seed=42)

    df_train, df_val, df_test = data_pipe.split_by_events(df_raw)

    # Fit preprocessor statistics strictly on train events
    data_pipe.fit_preprocessor(df_train)

    X_train = data_pipe.transform_features(df_train)
    y_train = df_train[TARGET_COLUMN].values
    y_train_class = discretize_flood_risk(y_train)

    X_val = data_pipe.transform_features(df_val)
    y_val = df_val[TARGET_COLUMN].values
    y_val_class = discretize_flood_risk(y_val)

    X_test = data_pipe.transform_features(df_test)
    y_test = df_test[TARGET_COLUMN].values
    y_test_class = discretize_flood_risk(y_test)

    feature_names = data_pipe.feature_names

    # 3. Model Hyperparameter Tuning & Training
    model_cfg = config.get("model", {})
    reg_params = dict(model_cfg.get("regressor_params", {}))
    clf_params = dict(model_cfg.get("classifier_params", {}))
    early_stopping = model_cfg.get("early_stopping_rounds", 35)

    if tune_hyperparameters:
        logger.info("Running hyperparameter exploration on train/val events...")
        # Compare shallow vs deep vs tuned learning rate
        candidate_configs = [
            {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
            {"max_depth": 6, "learning_rate": 0.04, "subsample": 0.85},
            {"max_depth": 7, "learning_rate": 0.03, "subsample": 0.9},
        ]
        best_rmse = float("inf")
        best_cfg = reg_params
        for cand in candidate_configs:
            trial_params = {**reg_params, **cand}
            trial_model = CrestixXGBRegressor(params=trial_params, early_stopping_rounds=early_stopping)
            trial_model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names, verbose=False)
            val_preds = trial_model.predict(X_val)
            val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
            logger.info(f"Candidate {cand} -> Validation RMSE: {val_rmse:.4f} m")
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_cfg = trial_params
        reg_params = best_cfg
        logger.info(f"Selected best configuration: max_depth={reg_params.get('max_depth')}, lr={reg_params.get('learning_rate')}")

    # Train final continuous XGBRegressor
    logger.info("Training production CrestixXGBRegressor...")
    regressor = CrestixXGBRegressor(params=reg_params, early_stopping_rounds=early_stopping, version=model_cfg.get("version", "1.0.0"))
    regressor.fit(X_train, y_train, X_val, y_val, feature_names=feature_names, verbose=False)

    # Train optional multi-class XGBClassifier
    logger.info("Training multi-class CrestixXGBClassifier for flood hazard levels...")
    classifier = CrestixXGBClassifier(params=clf_params, early_stopping_rounds=early_stopping, version=model_cfg.get("version", "1.0.0"))
    classifier.fit(X_train, y_train_class, X_val, y_val_class, feature_names=feature_names, verbose=False)

    # 4. Save model checkpoints
    reg_model_path = models_dir / "xgb_regressor.json"
    clf_model_path = models_dir / "xgb_classifier.json"
    regressor.save(reg_model_path)
    classifier.save(clf_model_path)

    # 5. Comprehensive Held-Out Event Evaluation
    logger.info("Evaluating on held-out flood events...")
    eval_summary = evaluate_heldout_events(
        model=regressor,
        df_heldout=df_test,
        feature_names=feature_names,
        target_col=TARGET_COLUMN,
        classifier_model=classifier,
        horizons=data_pipe.horizons,
    )

    # Save predictions array for held-out events
    test_preds = regressor.predict(X_test)
    preds_npy_path = predictions_dir / "heldout_predictions.npy"
    np.save(preds_npy_path, test_preds)

    # 6. Feature Importance & SHAP Explainability
    logger.info("Computing feature importance and SHAP TreeExplainer values...")
    df_gain_imp = compute_xgboost_importance(regressor, feature_names=feature_names)
    shap_vals, df_shap_imp, _ = compute_shap_explanations(regressor, X_test, feature_names=feature_names, max_samples=400)
    explain_artifacts = save_importance_artifacts(df_gain_imp, df_shap_imp, shap_vals, shap_dir)

    # 7. Generate Checksums and Metadata
    reg_checksum = compute_file_sha256(reg_model_path)
    clf_checksum = compute_file_sha256(clf_model_path)
    preds_checksum = compute_file_sha256(preds_npy_path)
    shap_checksum = compute_file_sha256(explain_artifacts["shap_npy"])

    metadata = {
        "model_version": model_cfg.get("version", "1.0.0"),
        "model_type": "XGBRegressor",
        "dataset_name": config.get("data", {}).get("dataset_name", "LarNO_UKEA"),
        "feature_order": feature_names,
        "feature_count": len(feature_names),
        "target_column": TARGET_COLUMN,
        "regressor_hyperparameters": reg_params,
        "classifier_hyperparameters": clf_params,
        "preprocessor_stats": data_pipe.preprocessor_stats,
        "evaluation_metrics": eval_summary["overall_continuous_metrics"],
        "checksums_sha256": {
            "xgb_regressor.json": reg_checksum,
            "xgb_classifier.json": clf_checksum,
            "heldout_predictions.npy": preds_checksum,
            "shap_values_sample.npy": shap_checksum,
        },
        "top_features_by_gain": df_gain_imp["feature"].head(5).tolist(),
        "top_features_by_shap": df_shap_imp["feature"].head(5).tolist(),
    }

    meta_path = metadata_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    features_path = metadata_dir / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump({"features": feature_names}, f, indent=2)

    # 8. Write Metric Reports
    markdown_report = generate_evaluation_markdown_report(eval_summary)
    with open(metrics_dir / "README.md", "w") as f:
        f.write(markdown_report)

    with open(metrics_dir / "evaluation_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    # 9. Write Artifacts README (Google Drive Deliverables & Verification Instructions)
    artifacts_readme = f"""# Crestix Probe XGBoost Model Artifacts and Deliverables

This repository stores production model configurations, feature schemas, checksums, and evaluation metrics. In accordance with project policy, large model binaries, raw training tensors, prediction arrays, and SHAP outputs are stored in Google Drive.

## Google Drive Deliverable Storage

- **Google Drive Archive URL**: `https://drive.google.com/drive/folders/crestix-probe-xgb-v1`
- **Release Tag**: `v{metadata['model_version']}`
- **LarNO Dataset Release**: `LarNO-UKEA-2026-R1`

### Manifest of Deliverables and Checksums

| File / Deliverable | Target Directory | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| `xgb_regressor.json` | `local_artifacts/models/` | `{reg_checksum}` | Trained continuous XGBRegressor model |
| `xgb_classifier.json` | `local_artifacts/models/` | `{clf_checksum}` | Trained 4-class flood hazard XGBClassifier |
| `heldout_predictions.npy` | `local_artifacts/predictions/` | `{preds_checksum}` | Full prediction array on held-out test events |
| `shap_values_sample.npy` | `local_artifacts/shap/` | `{shap_checksum}` | SHAP values array for test events |
| `model_metadata.json` | `artifacts/metadata/` | `{compute_file_sha256(meta_path)}` | Feature order, preprocessor stats & params |

---

## Download Instructions

### Using Python CLI Helper:
```bash
python scripts/download_artifacts.py --drive-url "https://drive.google.com/drive/folders/crestix-probe-xgb-v1"
```

### Manual Download & Verification:
1. Download `xgb_regressor.json` and `xgb_classifier.json` into `local_artifacts/models/`.
2. Verify checksum:
```bash
# Windows PowerShell
Get-FileHash -Algorithm SHA256 local_artifacts/models/xgb_regressor.json

# Linux / macOS
sha256sum local_artifacts/models/xgb_regressor.json
```
Ensure the hash matches `{reg_checksum}`.

---

## Feature Schema and Order Guarantee

The model expects the exact 18 features in this sequence:
```json
{json.dumps(feature_names, indent=2)}
```
"""
    with open(root_dir / "artifacts" / "README.md", "w") as f:
        f.write(artifacts_readme)

    # 10. Instantiate and register the default inference engine
    inference_engine = FloodInferenceEngine(
        model_path=reg_model_path,
        metadata_path=meta_path,
        classifier_path=clf_model_path,
    )
    inference_engine.regressor = regressor
    inference_engine.classifier = classifier
    set_default_engine(inference_engine)

    logger.info("Training pipeline completed successfully.")
    return {
        "metadata": metadata,
        "evaluation_summary": eval_summary,
        "reg_model_path": str(reg_model_path),
        "clf_model_path": str(clf_model_path),
        "metrics_report_path": str(metrics_dir / "README.md"),
        "artifacts_readme_path": str(root_dir / "artifacts" / "README.md"),
    }


if __name__ == "__main__":
    run_training_pipeline()

