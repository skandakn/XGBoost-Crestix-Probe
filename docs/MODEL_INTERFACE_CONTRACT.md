# Crestix Probe — Common Model Interface Contract (Person 9)

**Scope:** Standardizes prediction inputs, units, feature ordering, spatial metadata, and outputs across:
- `https://github.com/skandakn/XGBoost-Crestix-Probe`
- `https://github.com/skandakn/Linear-Regression---Crestix-Probe`
- `https://github.com/skandakn/Random-Forest---Crestix-Probe`

---

## 1. Repository Inspection Status

| Model Repository | Inspection Findings | Prediction Interface | Current Integration Status |
| :--- | :--- | :--- | :--- |
| **`XGBoost-Crestix-Probe`** | Active codebase with `crestix_xgb`, `data_pipeline`, `artifacts/metadata/model_metadata.json`, `artifacts/metadata/feature_names.json` | `crestix_xgb.predict.FloodInferenceEngine` & `predict_depth(features)` | **Inspected & Wrapped** via `integration/common_contract.py` (`XGBoostContractAdapter`) |
| **`Linear-Regression---Crestix-Probe`** | Currently contains Person 7's evaluation framework (`evaluation_metrics.py`, `evaluate_model.py`, `plotting_utils.py`, `tests/`) | *Not yet present in repository* | **Pending LR model & Person 1 schema push** |
| **`Random-Forest---Crestix-Probe`** | Empty GitHub repository (`size: 0`, 0 commits, 0 branches) | *Not yet present in repository* | **Pending Random Forest repository push** |

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
