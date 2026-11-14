# XGBoost-Crestix-Probe — Feature Engineering & Dataset Pipeline
> **Predicting the Flow of Tomorrow: Tabular Hydrodynamic Forecasting for Crestix Probe**

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.0%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

This repository houses the **XGBoost Data Pipeline and Feature Engineering Engine** for **Crestix Probe**. It transforms spatio-temporal hydrodynamic simulation arrays from the **LarNO UKEA 32 × 32 benchmark catchment** into high-performance, scientifically rigorous tabular representations for gradient boosted tree modeling.

### Key Objectives
- **Catchment**: UK Environment Agency (UKEA) urban benchmark ($32 \times 32$ grid cells, $8\text{ m}$ spatial resolution, $5\text{ min}$ temporal cadence).
- **Primary Task**: Predict continuous future floodwater depth $h(t + 30\text{min}, i, j)$ across all grid cells.
- **Secondary Tasks**: Multi-horizon forecasting ($+5\text{m}, +15\text{m}, +60\text{m}$), event peak depth prediction, and 4-tier flood hazard classification.
- **Zero Data Leakage**: Guaranteed event-level isolation across training, validation, and testing partitions.

---

## 2. Repository Structure

```text
XGBoost-Crestix-Probe/
├── configs/
│   └── baseline.yaml              # Master configuration for features, targets, splits, and paths
├── data/
│   └── sample/                    # Lightweight 5-event benchmark sample for reproducible testing
│       ├── dem.npy                # Shared master elevation model (32, 32)
│       ├── event_001/             # Individual storm events with dem, rainfall, h, metadata
│       ├── event_002/
│       ├── event_003/
│       ├── event_004/
│       └── event_005/
├── data_pipeline/
│   ├── __init__.py                # Package exports
│   ├── load_data.py               # Data loading, validation, and synthetic LarNO generator
│   ├── feature_engineering.py     # Topographic, rainfall, hydraulic lag, and spatio-temporal features
│   ├── target_generation.py       # Multi-horizon leads, peak depth, and hazard classes
│   └── validation.py              # Event-aware splitting, leakage checks, and train-only preprocessor
├── src/
│   └── crestix_xgb/
│       └── __init__.py            # Clean public API for Prompt 3 training and inference integration
├── tests/
│   ├── test_load_data.py          # Data ingestion and shape verification tests
│   ├── test_feature_engineering.py# Scientific feature calculation tests
│   ├── test_target_generation.py  # Forecast horizons and classification threshold tests
│   ├── test_validation_and_leakage.py # Event isolation and leakage detection tests
│   └── test_end_to_end_pipeline.py# Full pipeline integration with actual XGBoost training
├── requirements.txt               # Pinned Python package dependencies
├── pyproject.toml                 # Package definition and pytest configuration
├── pytest.ini                     # Pytest runner settings
├── DATA_FORMAT.md                 # Complete data schema and feature definitions specification
└── README.md                      # Pipeline documentation and usage instructions
```

---

## 3. Quickstart & Installation

### Step 1: Install Dependencies
```bash
git clone https://github.com/skandakn/XGBoost-Crestix-Probe.git
cd XGBoost-Crestix-Probe
git checkout feature/xgb-data-pipeline
pip install -r requirements.txt
```

### Step 2: Generate Sample LarNO UKEA Benchmark Dataset
```bash
python data_pipeline/load_data.py
```
This generates 5 physically realistic storm events (`event_001` through `event_005`) spanning Light, Moderate, Heavy, Extreme, and Cloudburst severity in `data/sample/`.

### Step 3: Run the Complete Test Suite
```bash
python -m pytest tests -v
```
All 20 unit and integration tests will execute and verify data integrity, feature math, target horizons, event-aware isolation, and XGBoost training.

---

## 4. Integration Contract with Prompt 3 (Training Lead)

Prompt 3 can consume the data pipeline using **a single function call** that handles loading, feature extraction, target attachment, event-aware splitting, and leakage verification:

```python
from crestix_xgb import build_training_ready_splits
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate clean, leak-free train/val/test splits
train_df, val_df, test_df, feature_names = build_training_ready_splits(
    data_dir="data/sample",
    config_path="configs/baseline.yaml",
    primary_target_col="target_depth_lead_30m",
)

target_col = "target_depth_lead_30m"
X_train, y_train = train_df[feature_names], train_df[target_col]
X_val, y_val = val_df[feature_names], val_df[target_col]
X_test, y_test = test_df[feature_names], test_df[target_col]

# 2. Train continuous water-depth regressor
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# 3. Evaluate on completely held-out test events
y_pred = model.predict(X_test)
print(f"Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f} m")
print(f"Test R²:   {r2_score(y_test, y_pred):.4f}")

# 4. Extract feature importances
importance_dict = dict(zip(feature_names, model.feature_importances_))
```

---

## 5. Engineered Features Summary

Over 35 scientifically meaningful features are extracted across 4 domains:

1. **Topographic Features (DEM)**:
   - Elevation $z$, slope magnitude $\|\nabla z\|$, directional gradients $(\partial z/\partial x, \partial z/\partial y)$, aspect components $(\sin\theta, \cos\theta)$, Laplacian $\nabla^2 z$, profile and planform curvatures, 8-neighbor elevation differences (mean, min, max), local terrain roughness $\sigma(z)$ in $3\times 3$ and $5\times 5$ windows, Topographic Position Index (TPI), flow accumulation proxy, and building footprint mask.
2. **Meteorological Forcing (Rainfall)**:
   - Instantaneous precipitation rate $R(t)$, cumulative storm depth $\sum R$, historical rolling window accumulations and means ($15\text{m}, 30\text{m}, 60\text{m}$), peak intensity experienced so far, and rainfall rate of change (cloudburst acceleration).
3. **Hydraulic State & Antecedent Lags**:
   - Current water depth $h(t)$, antecedent observation lags at $-5\text{m}, -10\text{m}, -15\text{m}$, rate of depth rise $dh/dt$, wave acceleration $d^2h/dt^2$, 8-neighbor mean and max depths (wavefront detection), piezometric hydraulic head $H = z + h$, and driving hydraulic gradient magnitude $\|\nabla(z + h)\|$.
4. **Spatio-Temporal Coordinates**:
   - Normalized spatial indices $x/(W-1), y/(H-1)$, Euclidean distance to catchment center, elapsed minutes since rainfall onset, and event progress fraction $t/(T-1)$.

For complete mathematical formulations and physical units, consult [`DATA_FORMAT.md`](DATA_FORMAT.md).

---

## 6. Data Leakage Prevention Guarantee

1. **Strict Event Disjointness**: Complete flood events are held out atomically. No cell or timestep from a test event is ever seen in training or validation.
2. **Temporal Causality**: All dynamic features (cumulative rain, rolling windows, lag depths, rate of rise) strictly compute over past and present observations ($\tau \le t$).
3. **Target Isolation**: Target columns are prefixed with `target_` and explicitly quarantined from the feature matrix $X$.
4. **Train-Only Preprocessing**: Missing value imputation and optional scalers are fitted exclusively on the training event distribution.

---

## 7. Storage Protocol for Large Files

- **GitHub Repository**: Tracks code, unit tests, configurations, metadata, and the small sample dataset in `data/sample/`.
- **Google Drive / Shared Storage**: Stores large raw simulation arrays and heavy `.parquet` tables. Processed dataset arrays must be uploaded to the project Drive folder and referenced via checksums.

