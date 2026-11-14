# Crestix Probe — Common Model Interface Contract (Person 9)

**Scope:** Standardizes prediction inputs, units, feature ordering, spatial metadata, and outputs across:
- `https://github.com/skandakn/XGBoost-Crestix-Probe`
- `https://github.com/skandakn/Linear-Regression---Crestix-Probe`
- `https://github.com/skandakn/Random-Forest---Crestix-Probe`

---

## 1. Repository Inspection Status

| Model Repository | Inspection Findings | Prediction Interface | Current Integration Status |
| :--- | :--- | :--- | :--- |
| **`XGBoost-Crestix-Probe`** | Active codebase with two pipelines: (1) 18-feature `src/crestix_xgb/` (`v1.0.0`) & (2) Person 1's 43-feature `data_pipeline/` + `scripts/evaluate_common_pipeline.py` (`v2.0.0-common`) | `crestix_xgb.predict.FloodInferenceEngine` & `predict_depth(features)` | **Inspected & Wrapped** via `integration/common_contract.py` (`XGBoostContractAdapter`, `CANONICAL_PERSON1_COMMON_43_FEATURES`) |
| **`Linear-Regression---Crestix-Probe`** | Contains Person 7's evaluation framework (`evaluation_metrics.py`, `evaluate_model.py`, `plotting_utils.py`, 9 tests) + Person 8's `comparison.py` report | *Not yet present in repository* | **Pending Linear Regression training/inference module & weights** |
| **`Random-Forest---Crestix-Probe`** | Contains Person 5's tuning & feature engineering (`rf_feature_builder.py` with 11 features, `random_forest_tuning.py`, `best_rf_config.json`, `final_validation_report.md`) | `sklearn.ensemble.RandomForestRegressor` (no saved inference module/weights committed yet) | **Inspected & Wrapped** via `RandomForestContractAdapter` (`CANONICAL_RF_11_FEATURES`); awaiting saved `.joblib` artifact |

---

## 2. Common Prediction Contract

### Input Schema (`StandardizedCellInput`)

| Field | Type | Units | Required | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `event_id` | `str` | — | Yes | Unique flood storm event identifier |
| `grid_row` | `int` | grid index | Yes | Spatial row coordinate (`>= 0`) |
| `grid_col` | `int` | grid index | Yes | Spatial column coordinate (`>= 0`) |
| `timestamp_utc` | `str` | ISO-8601 | Auto/Opt | Observation timestamp (defaults to UTC now if omitted) |
| `forecast_horizon_min` | `int` | minutes | Yes | Forecast lead time (`15, 30, 45, 60, 90, 120`) |
| `elevation_m` | `float` | meters | Yes | Bare-earth DEM elevation |
| `slope_pct` | `float` | percent | Yes | Terrain slope percentage |
| `rain_intensity_mmh` | `float` | mm/hr | Yes | Instantaneous rainfall intensity |
| `previous_water_depth_m` | `float` | meters | Yes | Prior depth (also accepts `previous_water_depth_cm`, auto-converted `/ 100.0`) |
| `aspect_sin`, `aspect_cos` | `float` | dimensionless | Optional | Steepest descent direction components |
| `twi` | `float` | dimensionless | Optional | Topographic Wetness Index |
| `dist_to_drain_m` | `float` | meters | Optional | Distance to nearest urban drainage culvert |
| `impervious_fraction` | `float` | `[0, 1]` | Optional | Built-surface imperviousness fraction |
| `manning_n` | `float` | `s/m^(1/3)` | Optional | Surface roughness coefficient |
| `rain_acc_15m` .. `rain_acc_6h` | `float` | mm | Optional | Rolling rainfall accumulations |
| `depth_lag_15m`, `depth_lag_30m` | `float` | meters | Optional | Antecedent water depth lags (defaults to `previous_water_depth_m`) |
| `depth_rate_of_rise` | `float` | m/hr | Optional | Hydrograph rate of rise |

### Canonical XGBoost 18-Feature Order

Verified against `XGBoost-Crestix-Probe/artifacts/metadata/feature_names.json`:

1. `elevation_m`
2. `slope_pct`
3. `aspect_sin`
4. `aspect_cos`
5. `twi`
6. `dist_to_drain_m`
7. `impervious_fraction`
8. `manning_n`
9. `rain_intensity_mmh`
10. `rain_acc_15m`
11. `rain_acc_30m`
12. `rain_acc_1h`
13. `rain_acc_3h`
14. `rain_acc_6h`
15. `depth_lag_15m`
16. `depth_lag_30m`
17. `depth_rate_of_rise`
18. `horizon_min`

### Canonical Random Forest 11-Feature Order

Verified against `Random-Forest---Crestix-Probe/rf_feature_builder.py` (`FEATURE_NAMES`):

1. `dem`
2. `slope_x`
3. `slope_y`
4. `slope_mag`
5. `elev_above_local_min`
6. `t_norm`
7. `h_init`
8. `rain_t`
9. `rain_prev1`
10. `rain_prev2`
11. `rain_cum`

### Person 1 Common 43-Feature Order (`v2.0.0-common`)

Verified against `XGBoost-Crestix-Probe/artifacts/metadata/common_model_metadata.json` (`CANONICAL_PERSON1_COMMON_43_FEATURES` in `integration/common_contract.py`), predicting `target_depth_lead_30m` (+30 min, meters) with flood threshold `0.03 m`.

### Standardized Output Schema (`StandardizedPredictionOutput`)

Because `FloodInferenceEngine.predict()` natively returns a raw `np.ndarray` in **meters (`m`)** without `event_id` or `(grid_row, grid_col)` metadata, `XGBoostContractAdapter.predict_batch()` wraps the raw prediction and guarantees:

- `event_id` (`str`) — preserved per cell
- `grid_row`, `grid_col` (`int`) — preserved per cell
- `forecast_horizon_min` (`int`) — preserved per prediction
- `predicted_water_depth_m` (`float`, clamped `>= 0.0`)
- `predicted_water_depth_cm` (`float`, `predicted_water_depth_m * 100.0`, aligning with Crestix Probe backend/evaluation cm standard)
- `model_name` (`str`, e.g. `"xgboost"`)
- `model_version` (`str`, e.g. `"1.0.0"`)
- `units` (`"m (and cm)"`)
