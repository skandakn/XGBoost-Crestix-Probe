# Crestix Probe — ML Model Artifact Manifest (Person 9)

Large model binaries and dataset tensors are **never committed to Git**. They are stored externally and verified via SHA-256 checksums before loading in FastAPI.

---

## 1. XGBoost (`skandakn/XGBoost-Crestix-Probe`)

- **Release Tag:** `v1.0.0`
- **Dataset Provenance:** `LarNO_UKEA_Catchment_Benchmark` (`LarNO-UKEA-2026-R1`)
- **External Storage URL:** `https://drive.google.com/drive/folders/crestix-probe-xgb-v1`
- **Download Helper:** `python scripts/download_artifacts.py --drive-url "https://drive.google.com/drive/folders/crestix-probe-xgb-v1"`

| Artifact File | Expected Local Path | Status in Git | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `xgb_regressor.json` | `local_artifacts/models/xgb_regressor.json` | External (not in Git) | `f7aedfb95e26e8e1eb1155550540297cb66137124b133b626293f5d0e414c917` |
| `xgb_classifier.json` | `local_artifacts/models/xgb_classifier.json` | External (not in Git) | `6052e14026f482bfbbd0f0ba4b37bb293efea1fb2d44769f13cdb21064bad85f` |
| `heldout_predictions.npy` | `local_artifacts/predictions/heldout_predictions.npy` | External (not in Git) | `c4d037d8fac0153f157c321c0b37cf8b9de4d1f74f6d19e26402324a9d6422d2` |
| `shap_values_sample.npy` | `local_artifacts/shap/shap_values_sample.npy` | External (not in Git) | `87f52b08b78c2b7df228203c6021c3d734cbd221916da7ee3e921016604debac` |
| `model_metadata.json` | `artifacts/metadata/model_metadata.json` | Committed in repo | `939bb45dc5cd6767c26e268a24f47eb862734ea53057511d69f1b8de55087700` |
| `feature_names.json` | `artifacts/metadata/feature_names.json` | Committed in repo | Committed in repo |

---

## 2. Linear Regression (`skandakn/Linear-Regression---Crestix-Probe`)

- **Status:** `PENDING` — Repository currently contains Person 7's evaluation framework (`evaluation_metrics.py`, `evaluate_model.py`, `plotting_utils.py`), Person 8's model comparison report (`comparison.py`, `comparison/`), and Person 9 integration files. No trained Linear Regression model artifact or training script has been pushed yet.

---

## 3. Random Forest (`skandakn/Random-Forest---Crestix-Probe`)

- **Status:** `AVAILABLE` — Trained `RandomForestRegressor` bundle committed at `artifacts/random_forest_model.joblib` (`4,622,217` bytes) alongside `predict.py`, `train_random_forest.py`, `configs/rf_config.yaml`, `requirements.txt`, and `outputs/evaluation_metrics.json`.
- **Reproducibility Pins (`outputs/evaluation_metrics.json`):** `python 3.11.9`, `scikit-learn 1.9.1`, `numpy 2.4.6`, `pandas 3.1.1`, `joblib 1.6.0`, `random_seed 42`.
- **Dataset Scope Note:** Trained & evaluated on synthetic UKEA `32x32` data (`synthetic_ukea_32x32`; MAE `0.00509 m`, RMSE `0.00886 m`, R² `0.8145`) with `predict.py` expecting `df[artifact["feature_names"]]` (+`event_id`, `grid_y`, `grid_x`). Also contains Person 5's 11-feature `rf_feature_builder.py` & `best_rf_config.json`.
