# Crestix Probe — Final FastAPI Integration Guide (Person 9)

## 1. Environment & Dependency Setup

Install the consolidated integration dependencies (see `requirements-integration.txt`):

```bash
pip install -r requirements-integration.txt
```

Key verified package baselines:
- `numpy>=1.26.0` (verified locally on `2.5.3`)
- `pandas>=2.0.0` (verified locally on `3.0.5`)
- `pydantic>=2.7.0` (verified locally on `2.14.3`)
- `fastapi>=0.115.0` (verified locally on `0.162.7`)
- `pytest>=8.0.0` (verified locally on `9.2.0`)
- Model-specific dependencies required for live inference: `xgboost>=2.0.0`, `scipy>=1.11.0`, `scikit-learn>=1.3.0`, `joblib>=1.3.0`.

---

## 2. Submodule & Artifact Installation in `Crestix-Probe`

In the main `Crestix-Probe` repository:

```bash
# 1. Attach / update model submodules
git submodule add https://github.com/skandakn/XGBoost-Crestix-Probe.git ml/xgboost
git submodule update --init --recursive

# 2. Download trained XGBoost weights (do NOT commit binaries)
cd ml/xgboost
python scripts/download_artifacts.py --dest local_artifacts
```

Point `Crestix-Probe/backend/.env` (without secrets) to the downloaded weights:

```env
ML_XGBOOST_SUBMODULE_DIR="../ml/xgboost"
ML_XGBOOST_WEIGHTS_PATH="../ml/xgboost/local_artifacts/models/xgb_regressor.json"
```

---

## 3. FastAPI Adapter Wiring (`backend/app/services/ml_adapters.py`)

1. Convert incoming FastAPI request payloads into `StandardizedCellInput` instances from `integration/common_contract.py` (preserving `event_id`, `grid_row`, `grid_col`, and `forecast_horizon_min`).
2. Call `XGBoostContractAdapter(repo_root=..., model_path=...).predict_batch(inputs)`.
3. Cache the loaded `XGBoostContractAdapter` inside `MLModelRegistry` so weights are loaded **once** at startup or first request, never reloaded per request.
4. Catch `ArtifactNotInstalledError` or `DependencyNotInstalledError` and map directly to HTTP `503 MODEL_NOT_INSTALLED` — never substitute fake/random predictions.
5. For model comparison with Person 7's `evaluation_metrics.py`, pass `predicted_water_depth_cm` (already converted `depth_m * 100.0`) so all models are compared in **centimeters (cm)** against the shared `flood_threshold_cm`.

---

## 4. Reproducibility & Error Handling Checklist

- **Feature Order Lock:** `XGBoostContractAdapter` aligns every request to the exact 18-feature order in `artifacts/metadata/feature_names.json`.
- **Non-Negative Depth Invariant:** All predicted depths are clamped `>= 0.0`.
- **Missing Weights / Dependencies:** Raises `ArtifactNotInstalledError` or `DependencyNotInstalledError` cleanly instead of silent fallbacks.
- **No Uncoordinated Internal Edits:** `src/crestix_xgb/*` and `data_pipeline/*` remain untouched.
