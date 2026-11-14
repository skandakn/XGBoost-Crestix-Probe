# Crestix Probe — ML Integration Readiness Testing Report (Person 9)

**Date:** 2026-11-14  
**Scope:** Integration readiness audit across `XGBoost-Crestix-Probe`, `Linear-Regression---Crestix-Probe`, and `Random-Forest---Crestix-Probe`.

---

## 1. Observed Environment & Package Audit

Executed `python -m pip list` on the integration host (`Python 3.12.10`):
- **Present:** `numpy 2.5.3`, `pandas 3.0.5`, `pydantic 2.14.3`, `fastapi 0.162.7`, `pytest 9.2.0`
- **Not Installed on Host:** `xgboost`, `scipy`, `scikit-learn`, `shap`, `matplotlib`

---

## 2. Repository-by-Repository Test & Readiness Results

### A. `skandakn/XGBoost-Crestix-Probe`
- **Existing Suite (`python -m pytest -v`):**
  - **Result:** Collection stopped with 8 errors (`ModuleNotFoundError: No module named 'xgboost'` and `No module named 'scipy'`).
  - **Artifacts:** `local_artifacts/models/xgb_regressor.json` is stored externally on Google Drive and not present in Git.
- **Person 9 Integration Suite (`python -m pytest integration_test.py -v`):** **7 passed, 2 skipped** (`0.09s`).
  - Verifies `StandardizedCellInput` validation and `cm -> m` normalization (`PASSED`).
  - Verifies exact 18-feature ordering match against `artifacts/metadata/feature_names.json` (`PASSED`).
  - Verifies exact Person 1 43-feature ordering match against `artifacts/metadata/common_model_metadata.json` (`PASSED`).
  - Verifies spatial `(event_id, grid_row, grid_col)` preservation on mock XGBoost & Random Forest engines (`PASSED`).
  - Verifies `DependencyNotInstalledError` / `ArtifactNotInstalledError` when `xgboost` or weights are absent (`PASSED`).
  - Verifies `samples/sample_input_output.json` fixture alignment (`PASSED`).
  - Skipped (`2`): `test_rf_11_feature_order_if_present` (belongs to RF repo) and `test_live_xgboost_engine_load_and_predict` (`xgboost` / `xgb_regressor.json` absent on host).

### B. `skandakn/Linear-Regression---Crestix-Probe`
- **Combined Suite (`python -m pytest tests/test_evaluation_framework.py integration_test.py -v`):** **14 passed, 4 skipped** (`0.60s`).
  - Person 7 Evaluation Suite (`tests/test_evaluation_framework.py`): **9 passed**.
  - Person 8 Comparison Report (`comparison.py`, `comparison/`): verified no Linear Regression model code or LR weights exist yet.
  - Person 9 Integration Suite (`integration_test.py`): **5 passed, 4 skipped** (repo-specific feature/artifact checks skipped cleanly).
- **Integration Blocker Found:** No Linear Regression training/prediction module or serialized weights have been pushed yet.

### C. `skandakn/Random-Forest---Crestix-Probe`
- **Existing Suites (`python -m pytest -v`):** Collection fails on this host because `matplotlib` (needed by `random_forest_tuning.py` / `tests/test_random_forest_tuning.py`), `scipy`, `scikit-learn`, `joblib`, and `pyyaml` (needed by `src/rf_flood/dataset.py` & `tests/test_random_forest.py`) are not installed.
- **Person 9 Integration Suite (`python -m pytest integration_test.py -v`):** **6 passed, 3 skipped** (`0.33s`).
  - Verifies exact 11-feature match (`CANONICAL_RF_11_FEATURES`) against `rf_feature_builder.FEATURE_NAMES` (`PASSED`), preserves `(event_id, grid_row, grid_col)` on mock estimator predictions (`PASSED`), and raises `DependencyNotInstalledError` / `ArtifactNotInstalledError` cleanly when `scikit-learn`/`joblib` or `artifacts/random_forest_model.joblib` is missing (`PASSED`).
  - Skipped (`3`): XGBoost-specific metadata (`feature_names.json`, `common_model_metadata.json`) and live XGBoost engine check.
- **Integration Findings:** Trained artifact `artifacts/random_forest_model.joblib` and `predict.py` (`load_rf_artifact`, `predict_from_dataframe`) are present, preserving `(event_id, grid_y, grid_x)`. However, Person 4's `src/rf_flood/dataset.py` uses a different feature set (`depth_roc`, `rain_intensity_mm_5min`, `depth_lag_*`, `elevation_m`, etc.) than both Person 5's `rf_feature_builder.py` (11 features) and Person 1's 43-feature schema.

---

## 3. Summary of Action Items for Model Owners

1. **XGBoost Team:** Both the 18-feature (`v1.0.0`, Google Drive) and Person 1 43-feature (`v2.0.0-common`, `scripts/evaluate_common_pipeline.py`) pipelines are documented; ensure deployment installs `requirements.txt` (`xgboost`, `scipy`, `scikit-learn`) and downloads `xgb_regressor.json` (or serializes the common 43-feature checkpoint).
2. **Linear Regression Team:** Push Linear Regression training/prediction module and serialized weights so an adapter can be wired and tested.
3. **Random Forest Team:** Add `matplotlib` to `requirements.txt` if retaining `tests/test_random_forest_tuning.py` in CI, and align `artifacts/random_forest_model.joblib`'s `feature_names` with Person 1's 43-feature schema for apples-to-apples multi-model inference.
