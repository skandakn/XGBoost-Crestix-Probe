# XGBoost-Crestix-Probe

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Urban Pluvial Flood-Depth Forecasting Tabular Baseline & Feature Engineering Pipeline**  
> Calibrated for the **LarNO UKEA (UK Environment Agency)** Benchmark Catchment Region.  
> Core machine learning component of **Crestix Probe**.

---

## Overview

`XGBoost-Crestix-Probe` delivers both the **LarNO UKEA 32×32 feature engineering / dataset pipeline** (`data_pipeline/`) and the **tabular gradient-boosted training, evaluation, and inference engine** (`src/crestix_xgb/`).

### Key Highlights
- **Strict Event-Based Isolation**: 100% leak-free partitioning across discrete storm events. Held-out flood events are never touched during training or hyperparameter tuning.
- **Physics-Enforced Non-Negativity**: Water depth in meters is strictly constrained to $\ge 0.0\text{ m}$.
- **Multi-Task Support**: Continuous water-depth regression (`XGBRegressor`) combined with 4-class flood hazard risk categorisation (`XGBClassifier`).
- **Explainability**: Gain-based, weight-based, and cover-based feature importances alongside `shap.TreeExplainer` game-theoretic attributions.
- **Enterprise Inference Interface**: Sub-millisecond inference API (`predict_depth(features)`) enforcing strict feature schema validation and automatic column re-ordering.

---

## Working Python Inference Example

```python
import pandas as pd
from crestix_xgb import predict_depth, FloodInferenceEngine

observation = {
    "elevation_m": 24.5,
    "slope_pct": 2.1,
    "aspect_sin": -0.7071,
    "aspect_cos": 0.7071,
    "twi": 11.2,
    "dist_to_drain_m": 45.0,
    "impervious_fraction": 0.85,
    "manning_n": 0.022,
    "rain_intensity_mmh": 42.5,
    "rain_acc_15m": 8.5,
    "rain_acc_30m": 16.2,
    "rain_acc_1h": 28.4,
    "rain_acc_3h": 45.0,
    "rain_acc_6h": 52.0,
    "depth_lag_15m": 0.08,
    "depth_lag_30m": 0.04,
    "depth_rate_of_rise": 0.16,
    "horizon_min": 30,
}

predicted_depth = predict_depth(observation)
print(f"Predicted Flood Depth at t+30m: {predicted_depth[0]:.4f} meters")
```

---

## Pipeline Integration (`build_training_ready_splits`)

```python
from crestix_xgb import build_training_ready_splits
from xgboost import XGBRegressor

train_df, val_df, test_df, feature_names = build_training_ready_splits(
    data_dir="data/sample",
    config_path="configs/baseline.yaml",
    primary_target_col="target_depth_lead_30m",
)
```

---

## Repository Structure

```text
XGBoost-Crestix-Probe/
├── configs/
│   ├── baseline.yaml
│   ├── default_config.yaml
│   └── hyperparameter_tuning.yaml
├── data/
│   └── sample/                    # 5-event benchmark sample (dem, rainfall, h, metadata)
├── data_pipeline/
│   ├── __init__.py
│   ├── load_data.py
│   ├── feature_engineering.py
│   ├── target_generation.py
│   └── validation.py
├── src/crestix_xgb/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── model.py
│   ├── explain.py
│   └── data.py
├── scripts/
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── download_artifacts.py
├── tests/
├── DATA_FORMAT.md
├── pyproject.toml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Quickstart & Testing

```bash
pip install -r requirements.txt
python scripts/run_training.py --config configs/default_config.yaml
python scripts/run_evaluation.py
python -m pytest -v
```
