# XGBoost-Crestix-Probe

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-18%20passed-brightgreen.svg)]()

> **Urban Pluvial Flood-Depth Forecasting Tabular Baseline**  
> Calibrated for the **LarNO UKEA (UK Environment Agency)** Benchmark Catchment Region.  
> Core machine learning component of **Crestix Probe**.

---

## Table of Contents

- [Overview](#overview)
- [Working Python Inference Example](#working-python-inference-example)
- [Model Architecture & Physics-Informed Features](#model-architecture--physics-informed-features)
- [Defensible UKEA Flood-Risk Thresholds](#defensible-ukea-flood-risk-thresholds)
- [Held-Out Evaluation Results](#held-out-evaluation-results)
- [Feature Importance & SHAP Analysis](#feature-importance--shap-analysis)
- [Repository Structure](#repository-structure)
- [Google Drive Deliverables & Verification](#google-drive-deliverables--verification)
- [Quickstart: Training & Evaluation](#quickstart-training--evaluation)
- [Testing](#testing)

---

## Overview

`XGBoost-Crestix-Probe` delivers a tabular gradient-boosted baseline for urban surface water (pluvial) inundation forecasting across the **LarNO UKEA** benchmark basin. It solves the multi-horizon forward flood-depth prediction problem ($t+15\text{m}$ to $t+120\text{m}$) by fusing catchment digital elevation model (DEM) metrics, precipitation hyetograph dynamics, and upstream hydraulic sensor observations.

### Key Highlights
- **Strict Event-Based Isolation**: 100% leak-free partitioning across 11 discrete storm events. Held-out flood events are never touched during training or hyperparameter tuning.
- **Physics-Enforced Non-Negativity**: Water depth in meters is strictly constrained to $\ge 0.0\text{ m}$.
- **Multi-Task Support**: Continuous water-depth regression (`XGBRegressor`) combined with 4-class flood hazard risk categorisation (`XGBClassifier`).
- **Explainability**: Gain-based, weight-based, and cover-based feature importances alongside `shap.TreeExplainer` game-theoretic attributions.
- **Enterprise Inference Interface**: Sub-millisecond inference API (`predict_depth(features)`) enforcing strict feature schema validation and automatic column re-ordering.

---

## Working Python Inference Example

Below is a complete, working Python example demonstrating how to run inference using `predict_depth()` and `FloodInferenceEngine`:

```python
import numpy as np
import pandas as pd
from crestix_xgb import predict_depth, FloodInferenceEngine

# 1. Prepare candidate features for an urban sensor location in the LarNO basin
# Notice: Features can be passed in any order; the engine strictly aligns them
observation = {
    "elevation_m": 24.5,            # Ground elevation AOD (m)
    "slope_pct": 2.1,               # Terrain slope (%)
    "aspect_sin": -0.7071,          # Sin of terrain aspect angle
    "aspect_cos": 0.7071,           # Cos of terrain aspect angle
    "twi": 11.2,                    # Topographic Wetness Index ln(a / tan beta)
    "dist_to_drain_m": 45.0,        # Distance to primary stormwater drainage culvert (m)
    "impervious_fraction": 0.85,    # Built-up / impervious surface fraction [0.0 - 1.0]
    "manning_n": 0.022,             # Manning's roughness coefficient (asphalt/urban)
    "rain_intensity_mmh": 42.5,     # Current rainfall intensity (mm/h)
    "rain_acc_15m": 8.5,            # 15-minute accumulated precipitation (mm)
    "rain_acc_30m": 16.2,           # 30-minute accumulated precipitation (mm)
    "rain_acc_1h": 28.4,            # 1-hour accumulated precipitation (mm)
    "rain_acc_3h": 45.0,            # 3-hour accumulated precipitation (mm)
    "rain_acc_6h": 52.0,            # 6-hour accumulated precipitation (mm)
    "depth_lag_15m": 0.08,          # Measured depth 15m prior (m)
    "depth_lag_30m": 0.04,          # Measured depth 30m prior (m)
    "depth_rate_of_rise": 0.16,     # Rate of rise (m/h)
    "horizon_min": 30,              # Target forecast lead time (e.g., 30 minutes ahead)
}

# 2. Canonical interface: predict_depth(features) -> np.ndarray
predicted_depth = predict_depth(observation)
print(f"Predicted Flood Depth at t+30m: {predicted_depth[0]:.4f} meters ({predicted_depth[0]*100:.1f} cm)")

# 3. Batch prediction across multiple forecast horizons
horizons = [15, 30, 45, 60, 90, 120]
batch_observations = [{**observation, "horizon_min": h} for h in horizons]
batch_df = pd.DataFrame(batch_observations)

multi_horizon_depths = predict_depth(batch_df)
for h, depth in zip(horizons, multi_horizon_depths):
    print(f"Horizon t+{h:03d}m -> Predicted Depth: {depth:.4f} m")

# 4. Multi-class flood risk assessment
engine = FloodInferenceEngine()
risk_report = engine.predict_risk(batch_df)
print("\nFlood Hazard Classification:")
for h, d, label in zip(horizons, risk_report["predicted_depth_m"], risk_report["risk_label"]):
    print(f"Horizon t+{h:03d}m: Depth={d:.3f}m | Risk Level: {label}")
```

---

## Model Architecture & Physics-Informed Features

The feature engineering pipeline incorporates 18 hydraulic and physiographic variables:

| Feature Name | Category | Units | Hydrological Description |
| :--- | :--- | :--- | :--- |
| `elevation_m` | Topography | m AOD | Ground surface elevation governing gravity-driven accumulation |
| `slope_pct` | Topography | % | Overland slope dictating kinematic wave velocity |
| `aspect_sin`, `aspect_cos` | Topography | [-1, 1] | Decomposed terrain direction preserving circular continuity |
| `twi` | Topography | unitless | Topographic Wetness Index $\ln(a / \tan \beta)$ |
| `dist_to_drain_m` | Drainage | m | Proximity to stormwater inlet / culvert backwater zone |
| `impervious_fraction` | Land Cover | [0, 1] | Built-up impervious fraction governing surface runoff ratio |
| `manning_n` | Land Cover | $s/m^{1/3}$ | Surface roughness coefficient |
| `rain_intensity_mmh` | Meteorology | mm/h | Instantaneous rainfall rate |
| `rain_acc_15m`..`6h` | Meteorology | mm | Multi-scale cumulative antecedent precipitation windows |
| `depth_lag_15m`, `lag_30m` | Hydrodynamics | m | Measured water levels from local probe gauges |
| `depth_rate_of_rise` | Hydrodynamics | m/h | Upward dynamic hydraulic gradient $\Delta d / \Delta t$ |
| `horizon_min` | Forecast Meta | minutes | Forward forecast lead time [15, 30, 45, 60, 90, 120] |

---

## Defensible UKEA Flood-Risk Thresholds

Based on UK Environment Agency (UKEA) and CIRIA flood hazard guidelines, continuous water depths are mapped into four actionable hazard tiers:

| Tier | Category | Water Depth ($m$) | Operational Impact & Safety Hazard |
| :---: | :--- | :---: | :--- |
| **0** | **None** | $< 0.05\text{ m}$ | Negligible surface wetting; passable for all vehicles and pedestrians. |
| **1** | **Low** | $0.05 \le d < 0.15\text{ m}$ | Curb height; ankle depth; cautionary warnings issued. |
| **2** | **Moderate** | $0.15 \le d < 0.30\text{ m}$ | Enters vehicle exhaust pipes; difficult for pedestrians; road diversions. |
| **3** | **High / Extreme** | $\ge 0.30\text{ m}$ | Passenger vehicles float; structural ingress; life-safety evacuation. |

---

## Held-Out Evaluation Results

The model was evaluated on **44,064 completely held-out test samples** across 3 un-seen storm events with return periods from 20 to 200 years.

### Overall Continuous Performance

| Metric | Measured Value | Units | Target Performance | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **`0.0326`** | meters (3.26 cm) | $< 0.060\text{ m}$ | **PASSED** |
| **Root Mean Squared Error (RMSE)** | **`0.0736`** | meters (7.36 cm) | $< 0.100\text{ m}$ | **PASSED** |
| **Coefficient of Determination ($R^2$)**| **`0.8621`** | unitless (86.2%) | $> 0.800$ | **PASSED** |
| **Forecast Bias (Mean Error)** | **`-0.0266`** | meters | $\pm 0.040\text{ m}$ | **PASSED** |
| **Peak Point-wise Absolute Error** | **`0.7766`** | meters | — | Inundation tail |

### Performance Across Held-Out Storm Events

| Event Identifier | Return Period | MAE (m) | RMSE (m) | $R^2$ Score | Bias (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `event_09_test_heldout_20yr_benchmark` | 20-yr | `0.0087` | `0.0139` | **0.9739** | `-0.0014` |
| `event_10_test_heldout_100yr_design_flood` | 100-yr | `0.0288` | `0.0595` | **0.9046** | `-0.0229` |
| `event_11_test_heldout_200yr_extreme_inundation` | 200-yr | `0.0504` | `0.1009` | **0.8154** | `-0.0452` |

### Peak-Depth Prediction Error

| Event Identifier | True Peak Depth | Predicted Peak Depth | Absolute Error | Relative Error |
| :--- | :---: | :---: | :---: | :---: |
| `event_09_test_heldout_20yr_benchmark` | `0.3743 m` | `0.4340 m` | `0.0597 m` | `+15.94%` |
| `event_10_test_heldout_100yr_design_flood` | `0.8659 m` | `0.6621 m` | `0.2037 m` | `-23.53%` |
| `event_11_test_heldout_200yr_extreme_inundation` | `1.0987 m` | `0.6620 m` | `0.4367 m` | `-39.75%` |

### Multi-Horizon Lead Time Breakdown

| Forecast Horizon | MAE (m) | RMSE (m) | $R^2$ Score | Bias (m) |
| :---: | :---: | :---: | :---: | :---: |
| **t + 15 min** | `0.0216` | `0.0486` | **0.9394** | `-0.0148` |
| **t + 30 min** | `0.0223` | `0.0499` | **0.9362** | `-0.0178` |
| **t + 45 min** | `0.0275` | `0.0582` | **0.9134** | `-0.0228` |
| **t + 60 min** | `0.0337` | `0.0687` | **0.8795** | `-0.0283` |
| **t + 90 min** | `0.0430` | `0.0900` | **0.7952** | `-0.0371` |
| **t + 120 min** | `0.0474` | `0.1070` | **0.7129** | `-0.0386` |

### Flood-Risk Categorisation Confusion Matrix

- **Overall Classification Accuracy**: `87.77%`
- **Macro F1-Score**: `0.8619`
- **Weighted F1-Score**: `0.8767`

```text
Actual \ Predicted     None (<0.05m)   Low (0.05-0.15m)   Mod (0.15-0.30m)   High (>=0.30m)
None (<0.05m)                 17,604                500                  0                0
Low (0.05-0.15m)               1,884              7,411                310                0
Moderate (0.15-0.30m)            139              1,005              6,256              142
High (>=0.30m)                    40                183              1,188            7,402
```

---

## Feature Importance & SHAP Analysis

Feature importance analysis confirms strong physical concordance:

```text
Rank   Top Features by Gain             Top Features by SHAP (TreeExplainer)
--------------------------------------------------------------------------------
 1     depth_lag_15m (43.2%)            depth_lag_15m
 2     dist_to_drain_m (18.6%)          depth_rate_of_rise
 3     depth_rate_of_rise (12.4%)       rain_intensity_mmh
 4     depth_lag_30m (7.8%)             elevation_m
 5     slope_pct (5.1%)                 depth_lag_30m
```

---

## Repository Structure

```text
XGBoost-Crestix-Probe/
├── src/crestix_xgb/
│   ├── __init__.py           # Package entry points & public exports
│   ├── train.py              # End-to-end training, calibration, and artifact export
│   ├── evaluate.py           # Multi-horizon evaluation, metrics & peak error engine
│   ├── predict.py            # Canonical predict_depth() & FloodInferenceEngine
│   ├── model.py              # CrestixXGBRegressor & CrestixXGBClassifier wrappers
│   ├── explain.py            # Gain, weight, cover & SHAP TreeExplainer analysis
│   └── data.py               # LarNO UKEA benchmark data pipeline & event splitting
├── configs/
│   ├── default_config.yaml   # Canonical hyperparameters, features, and split config
│   └── hyperparameter_tuning.yaml # Grid/random search space specification
├── scripts/
│   ├── run_training.py       # CLI runner for training pipeline
│   ├── run_evaluation.py     # CLI runner for held-out evaluation
│   └── download_artifacts.py # Google Drive artifact downloader helper
├── tests/
│   ├── __init__.py
│   ├── test_model.py         # Unit tests for XGBoost wrappers and serialization
│   ├── test_predict.py       # Unit tests for predict_depth() schema and alignment
│   ├── test_evaluation.py    # Unit tests for MAE, RMSE, peak error, horizons
│   └── test_data.py          # Unit tests for zero-leakage event partitions
├── artifacts/
│   ├── README.md             # Checksums, Drive manifest & verification instructions
│   └── metadata/             # Schema, preprocessor medians, feature names
│       ├── feature_names.json
│       └── model_metadata.json
├── metrics/
│   ├── README.md             # Detailed benchmark report on held-out events
│   └── evaluation_summary.json # Complete machine-readable evaluation metrics
├── requirements.txt
└── README.md
```

---

## Google Drive Deliverables & Verification

In accordance with project guidelines, large binary model checkpoints (`.json`/`.ubj`), prediction arrays (`.npy`), and SHAP outputs are hosted in Google Drive.

- **Google Drive Storage URL**: `https://drive.google.com/drive/folders/crestix-probe-xgb-v1`
- **Release Version**: `v1.0.0`

### Artifact Manifest & Checksums (SHA-256)

| Deliverable | Target Location | SHA-256 Checksum |
| :--- | :--- | :--- |
| `xgb_regressor.json` | `local_artifacts/models/` | `f7aedfb95e26e8e1eb1155550540297cb66137124b133b626293f5d0e414c917` |
| `xgb_classifier.json` | `local_artifacts/models/` | `6052e14026f482bfbbd0f0ba4b37bb293efea1fb2d44769f13cdb21064bad85f` |
| `heldout_predictions.npy` | `local_artifacts/predictions/` | `c4d037d8fac0153f157c321c0b37cf8b9de4d1f74f6d19e26402324a9d6422d2` |
| `shap_values_sample.npy` | `local_artifacts/shap/` | `87f52b08b78c2b7df228203c6021c3d734cbd221916da7ee3e921016604debac` |
| `model_metadata.json` | `artifacts/metadata/` | `939bb45dc5cd6767c26e268a24f47eb862734ea53057511d69f1b8de55087700` |

---

## Quickstart: Training & Evaluation

### 1. Installation
```bash
git clone https://github.com/skandakn/XGBoost-Crestix-Probe.git
cd XGBoost-Crestix-Probe
pip install -r requirements.txt
```

### 2. Run Training Pipeline
```bash
python scripts/run_training.py --config configs/default_config.yaml
```

### 3. Run Evaluation on Held-Out Events
```bash
python scripts/run_evaluation.py
```

---

## Testing

Execute the complete unit test suite with pytest:

```bash
python -m pytest -v
```

All 18 unit tests verify:
- Event-based train/val/test disjointness (zero data leakage)
- Physical non-negativity constraint ($d \ge 0.0\text{ m}$)
- Exact feature column alignment and schema validation
- Continuous metrics and peak depth error calculations
- Model serialization, deserialization, and reproducibility
