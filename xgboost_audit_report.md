# XGBoost Audit Report — Crestix Probe (Person 6)

**Repository:** `skandakn/XGBoost-Crestix-Probe`  
**Audit Date:** 2026-11-14  
**Auditor:** Person 6 (Crestix Probe ML Team)  

---

## 1. Executive Summary

An end-to-end audit of `skandakn/XGBoost-Crestix-Probe` revealed **two separate, incompatible pipelines** coexisting in the repository:
1. **Original XGBoost Pipeline (`src/crestix_xgb/`, PR #1):** Trained on a synthetic 24-sensor, 11-event tabular generator (`LarNODataPipeline.generate_larno_benchmark_dataset(seed=42)`) with **18 features**, target `water_depth_m` stacked across 6 lead horizons (`15`–`120` min), hardcoded `6 / 2 / 3` event splits, and 4-class risk thresholds (`[0.05, 0.15, 0.30]` m).
2. **Person 1 Common Data Pipeline (`data_pipeline/` + `configs/baseline.yaml`, PR #2):** Uses the `32×32` spatial-temporal event arrays in `data/sample/` (`event_001`–`event_005`), **43 engineered features**, primary continuous target `target_depth_lead_30m` (+30 min), binary flood label `target_is_flooded_lead_30m` (`>= 0.03` m), and `EventAwareSplitter(train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_seed=42)`.

**Conclusion:** The original XGBoost results (`MAE = 0.0326 m`, `RMSE = 0.0736 m`, `R² = 0.8621`) are **not comparable** to Linear Regression and Random Forest. To establish a strict apples-to-apples benchmark, XGBoost was retrained and evaluated on Person 1's common dataset, 43-feature schema, target, and event-aware split via [`scripts/evaluate_common_pipeline.py`](scripts/evaluate_common_pipeline.py).

---

## 2. Audit Findings (Tasks 1–10)

| Audit Area | Original Pipeline (`src/crestix_xgb/`) | Common Pipeline (`data_pipeline/` + `configs/baseline.yaml`) |
|---|---|---|
| **Target Variable** | `water_depth_m` (continuous depth in `m` across 6 horizons) + `risk_category` (4-class) | `target_depth_lead_30m` (continuous depth at `+30 min` in `m`) + `target_is_flooded_lead_30m` (binary `>= 0.03 m`) |
| **Input Features** | 18 features (`elevation_m`, `slope_pct`, ..., `horizon_min`) | 43 features (18 topographic, 10 rainfall, 10 hydraulic lag/gradient, 5 spatio-temporal) |
| **Dataset & Preprocessing** | Synthetic 24-sensor generator (`seed=42`, 11 events); train-only median imputation | `data/sample/` (`32×32` grids, 5 events); `TabularPreprocessor` (`scale_features: false`) |
| **Train / Val / Test Split** | Hardcoded 6 / 2 / 3 events (`event_01..06` / `07..08` / `09..11`; 44,064 test rows) | `EventAwareSplitter(0.60, 0.20, 0.20, seed=42)`: Train `['event_002', 'event_003', 'event_005']` (92,160 rows), Val `['event_001']` (30,720 rows), Test `['event_004']` (30,720 rows) |
| **Saved Model & Config** | `local_artifacts/models/xgb_regressor.json` (gitignored, hosted on Google Drive); config in `configs/default_config.yaml` & `artifacts/metadata/model_metadata.json` | Config in `configs/common_xgb_config.yaml`; metadata in `artifacts/metadata/common_model_metadata.json` |
| **Data Leakage Check** | Event-disjoint split; lagged features (`t-15m`, `t-30m`) strictly precede target (`t+horizon`); train-only stats | `check_data_leakage(...)` **PASSED** (disjoint events, no target columns in `X`, no NaNs/Infs) |
| **Output Type** | Continuous water depth (`>= 0.0 m` clipped), plus optional 4-class `XGBClassifier` | Continuous water depth (`>= 0.0 m` clipped); binary flood classification derived from threshold `>= 0.03 m` against `target_is_flooded_lead_30m` |

### Reproduction of Original Results (Task 8)
- `metrics/evaluation_summary.json` (original): `MAE = 0.0326 m`, `RMSE = 0.0736 m`, `R² = 0.8621`, `Bias = -0.0266 m` on 44,064 held-out samples.
- Re-running `LarNODataPipeline(seed=42)` + `CrestixXGBRegressor` with `configs/default_config.yaml` (`xgboost==3.2.0`) yielded: `MAE = 0.0329 m`, `RMSE = 0.0739 m`, `R² = 0.8611`, `Bias = -0.0268 m` on 44,064 held-out samples.

---

## 3. Comparable Evaluation on Common Pipeline

Executed via:
```bash
python scripts/evaluate_common_pipeline.py
```

- **Hyperparameters (`configs/common_xgb_config.yaml`):** `n_estimators=450`, `max_depth=6`, `learning_rate=0.04`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_weight=4`, `gamma=0.05`, `reg_alpha=0.15`, `reg_lambda=1.2`, `early_stopping_rounds=35` (stopped at iteration `179` on validation RMSE).
- **Held-Out Test Split:** `['event_004']` (`30,720` rows, `93.82%` flooded at `>= 0.03 m`).

### Regression Metrics (`target_depth_lead_30m`)
| Metric | Value | Units |
|---|---:|---|
| **MAE** | `0.017830` | `m` |
| **RMSE** | `0.023477` | `m` |
| **R²** | `0.931771` | unitless |
| **Bias** | `+0.002054` | `m` |

### Threshold-Based Classification Metrics (`target_is_flooded_lead_30m`, threshold `>= 0.03 m`)
| Metric | Value |
|---|---:|
| **AUROC** | `0.982230` |
| **PR-AUC** | `0.998823` |
| **F1-score** | `0.973091` |
| **Precision** | `0.990653` |
| **Recall** | `0.956142` |

*(Note: On the validation event `event_001`, which is a light storm with max depth `0.0699 m` and only `2.33%` flooded cells, regression `MAE = 0.013111 m`, `RMSE = 0.014263 m`, `R² = -2.218237`, `AUROC = 0.706442`, `PR-AUC = 0.064523`, `F1 = 0.090679`.)*

---

## 4. Documented Differences from Original Experiment

1. **Dataset:** Replaced synthetic 24-sensor tabular generator (`src/crestix_xgb/data.py`) with Person 1's `32×32` spatial-temporal grid events (`data/sample/`).
2. **Feature Space:** Expanded from 18 features to the common 43-feature specification (`data_pipeline/feature_engineering.py`).
3. **Target & Horizon:** Switched from multi-horizon stacked `water_depth_m` (`15`–`120` min) to fixed `+30 min` lead `target_depth_lead_30m`.
4. **Event Split:** Switched from hardcoded `6/2/3` named events to `EventAwareSplitter(0.60, 0.20, 0.20, seed=42)`.
5. **Classification Evaluation:** Replaced 4-class hazard accuracy/macro-F1 (`[0.05, 0.15, 0.30]` m) with binary flood/no-flood metrics (`AUROC`, `PR-AUC`, `F1`, `Precision`, `Recall`) computed against the common threshold-based label `target_is_flooded_lead_30m` (`>= 0.03 m`).

---

## 5. Deliverables

- Audit Report: `xgboost_audit_report.md`
- Reproducible Evaluation Script: `scripts/evaluate_common_pipeline.py`
- Model Configuration: `configs/common_xgb_config.yaml` and `artifacts/metadata/common_model_metadata.json`
- Updated Results: `metrics/common_evaluation_summary.json`
- Prediction Output: `metrics/common_test_predictions.csv`
- Unit Tests: `tests/test_common_evaluation.py`
