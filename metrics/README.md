# LarNO UKEA Benchmark: XGBoost Flood-Depth Forecasting Evaluation Report

## Executive Summary
- **Held-Out Events**: event_09_test_heldout_20yr_benchmark, event_10_test_heldout_100yr_design_flood, event_11_test_heldout_200yr_extreme_inundation
- **Total Held-Out Test Samples**: 44,064
- **Overall MAE**: `0.0326 m` (3.26 cm)
- **Overall RMSE**: `0.0736 m` (7.36 cm)
- **Overall R² Score**: `0.8621`
- **Overall Bias**: `-0.0266 m`
- **Peak Point-wise Absolute Error**: `0.7766 m`

---

## 1. Overall Continuous Performance Metrics

| Metric | Value | Units | Hydrological Interpretation |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | `0.0326` | meters | Average water depth forecasting error across all timesteps |
| **Root Mean Squared Error (RMSE)** | `0.0736` | meters | Sensitivity to high-magnitude runoff deviations |
| **Coefficient of Determination (R²)** | `0.8621` | unitless | Variance explained relative to mean catchment stage |
| **Forecast Bias (Mean Error)** | `-0.0266` | meters | Systematic calibration drift (+ indicates over-prediction) |
| **Maximum Absolute Error** | `0.7766` | meters | Worst-case single timestep divergence |

---

## 2. Event-by-Event Evaluation (Held-Out Storm Events)

| Event Identifier | Sample Count | MAE (m) | RMSE (m) | R² Score | Bias (m) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `event_09_test_heldout_20yr_benchmark` | - | `0.0087` | `0.0139` | `0.9739` | `-0.0014` |
| `event_10_test_heldout_100yr_design_flood` | - | `0.0288` | `0.0595` | `0.9046` | `-0.0229` |
| `event_11_test_heldout_200yr_extreme_inundation` | - | `0.0504` | `0.1009` | `0.8154` | `-0.0452` |

---

## 3. Peak-Depth Prediction Error Analysis

In urban flood emergency management, capturing the maximum flood depth is critical for road closure and evacuation decisions.

| Event Identifier | Actual Peak Depth (m) | Predicted Peak Depth (m) | Peak Error (m) | Peak Relative Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| `event_09_test_heldout_20yr_benchmark` | `0.3743` | `0.4340` | `0.0597` | `+15.94%` |
| `event_10_test_heldout_100yr_design_flood` | `0.8659` | `0.6621` | `0.2037` | `-23.53%` |
| `event_11_test_heldout_200yr_extreme_inundation` | `1.0987` | `0.6620` | `0.4367` | `-39.75%` |

---

## 4. Performance Across Forecast Horizons

Urban flood warning requires multi-horizon lead time foresight from 15 minutes up to 2 hours.

| Horizon (Lead Time) | MAE (m) | RMSE (m) | R² Score | Bias (m) |
| :--- | :--- | :--- | :--- | :--- |
| **15min** | `0.0216` | `0.0486` | `0.9394` | `-0.0148` |
| **30min** | `0.0223` | `0.0499` | `0.9362` | `-0.0178` |
| **45min** | `0.0275` | `0.0582` | `0.9134` | `-0.0228` |
| **60min** | `0.0337` | `0.0687` | `0.8795` | `-0.0283` |
| **90min** | `0.0430` | `0.0900` | `0.7952` | `-0.0371` |
| **120min** | `0.0474` | `0.1070` | `0.7129` | `-0.0386` |

---

## 5. Defensible UKEA Flood-Risk Classification

Thresholds: None (<0.05m), Low (0.05-0.15m), Moderate (0.15-0.30m), High (>=0.30m).

- **Classification Accuracy**: `87.77%`
- **Macro F1-Score**: `0.8619`
- **Weighted F1-Score**: `0.8767`

### Confusion Matrix (Rows: Actual, Columns: Predicted)

| Actual \ Pred | None (<0.05m) | Low (0.05-0.15m) | Moderate (0.15-0.30m) | High (>=0.30m) |
| :--- | :--- | :--- | :--- | :--- |
| **None (<0.05m)** | 17,604 | 500 | 0 | 0 |
| **Low (0.05-0.15m)** | 1,884 | 7,411 | 310 | 0 |
| **Moderate (0.15-0.30m)** | 139 | 1,005 | 6,256 | 142 |
| **High (>=0.30m)** | 40 | 183 | 1,188 | 7,402 |

---

## 6. Verification and Data Leakage Audit
- **Event-based isolation**: 100% verified. No timesteps or sub-catchments from held-out events were used during training or hyperparameter tuning.
- **Inference Feature Order**: Strict column alignment verified against `configs/default_config.yaml`.
- **Non-negativity constraint**: Physical clipping at zero prevents negative flood depths.