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
- **Person 9 Integration Suite (`python -m pytest integration_test.py -v`):** **5 passed, 1 skipped** (`0.09s`).
  - Verifies `StandardizedCellInput` validation and `cm -> m` normalization (`PASSED`).
  - Verifies exact 18-feature ordering match against `artifacts/metadata/feature_names.json` (`PASSED`).
  - Verifies spatial `(event_id, grid_row, grid_col)` preservation using an explicit mock engine (`PASSED`).
  - Verifies `DependencyNotInstalledError` / `ArtifactNotInstalledError` when `xgboost` or `xgb_regressor.json` is absent (`PASSED`).
  - Verifies `samples/sample_input_output.json` fixture alignment (`PASSED`).
  - Live `FloodInferenceEngine` test (`test_live_xgboost_engine_load_and_predict`) is marked `@pytest.mark.skipif(not (HAS_XGBOOST_PKG and XGB_WEIGHTS_PATH.is_file()))` and was **SKIPPED** on this host due to missing `xgboost` package and un-downloaded `local_artifacts/models/xgb_regressor.json`.

### B. `skandakn/Linear-Regression---Crestix-Probe`
- **Combined Suite (`python -m pytest tests/test_evaluation_framework.py integration_test.py -v`):** **13 passed, 2 skipped** (`0.44s`).
  - Person 7 Evaluation Suite (`tests/test_evaluation_framework.py`): **9 passed**.
  - Person 9 Integration Suite (`integration_test.py`): **4 passed, 2 skipped** (`test_xgboost_feature_names_match_canonical_order_if_present` skipped because `artifacts/metadata/feature_names.json` belongs to the XGBoost repo, and `test_live_xgboost_engine_load_and_predict` skipped due to missing `xgboost`/`xgb_regressor.json`).
- **Integration Blocker Found:** Repository currently contains Person 7's evaluation framework and Person 9's integration files only; no Linear Regression model code, trained weights, or Person 1 common schema has been pushed yet.

### C. `skandakn/Random-Forest---Crestix-Probe`
- **Integration Blocker Found:** Repository is currently empty (`0` commits, `0` branches on GitHub). No Random Forest code, dependencies, or prediction interface exists to test yet.

---

## 3. Summary of Action Items for Model Owners

1. **XGBoost Team:** No code blockers; ensure deployment environment installs `requirements.txt` (`xgboost`, `scipy`, `scikit-learn`) and downloads `xgb_regressor.json` (SHA-256 verified).
2. **Linear Regression & Person 1:** Push Linear Regression training/prediction module + Person 1's finalized schema so `LinearRegressionContractAdapter` can be wired and tested.
3. **Random Forest Team:** Push initial Random Forest codebase and artifact manifest to `skandakn/Random-Forest---Crestix-Probe`.
