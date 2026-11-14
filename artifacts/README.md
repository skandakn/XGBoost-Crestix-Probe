# Crestix Probe XGBoost Model Artifacts and Deliverables

This repository stores production model configurations, feature schemas, checksums, and evaluation metrics. In accordance with project policy, large model binaries, raw training tensors, prediction arrays, and SHAP outputs are stored in Google Drive.

## Google Drive Deliverable Storage

- **Google Drive Archive URL**: `https://drive.google.com/drive/folders/crestix-probe-xgb-v1`
- **Release Tag**: `v1.0.0`
- **LarNO Dataset Release**: `LarNO-UKEA-2026-R1`

### Manifest of Deliverables and Checksums

| File / Deliverable | Target Directory | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| `xgb_regressor.json` | `local_artifacts/models/` | `f7aedfb95e26e8e1eb1155550540297cb66137124b133b626293f5d0e414c917` | Trained continuous XGBRegressor model |
| `xgb_classifier.json` | `local_artifacts/models/` | `6052e14026f482bfbbd0f0ba4b37bb293efea1fb2d44769f13cdb21064bad85f` | Trained 4-class flood hazard XGBClassifier |
| `heldout_predictions.npy` | `local_artifacts/predictions/` | `c4d037d8fac0153f157c321c0b37cf8b9de4d1f74f6d19e26402324a9d6422d2` | Full prediction array on held-out test events |
| `shap_values_sample.npy` | `local_artifacts/shap/` | `87f52b08b78c2b7df228203c6021c3d734cbd221916da7ee3e921016604debac` | SHAP values array for test events |
| `model_metadata.json` | `artifacts/metadata/` | `939bb45dc5cd6767c26e268a24f47eb862734ea53057511d69f1b8de55087700` | Feature order, preprocessor stats & params |

---

## Download Instructions

### Using Python CLI Helper:
```bash
python scripts/download_artifacts.py --drive-url "https://drive.google.com/drive/folders/crestix-probe-xgb-v1"
```

### Manual Download & Verification:
1. Download `xgb_regressor.json` and `xgb_classifier.json` into `local_artifacts/models/`.
2. Verify checksum:
```bash
# Windows PowerShell
Get-FileHash -Algorithm SHA256 local_artifacts/models/xgb_regressor.json

# Linux / macOS
sha256sum local_artifacts/models/xgb_regressor.json
```
Ensure the hash matches `f7aedfb95e26e8e1eb1155550540297cb66137124b133b626293f5d0e414c917`.

---

## Feature Schema and Order Guarantee

The model expects the exact 18 features in this sequence:
```json
[
  "elevation_m",
  "slope_pct",
  "aspect_sin",
  "aspect_cos",
  "twi",
  "dist_to_drain_m",
  "impervious_fraction",
  "manning_n",
  "rain_intensity_mmh",
  "rain_acc_15m",
  "rain_acc_30m",
  "rain_acc_1h",
  "rain_acc_3h",
  "rain_acc_6h",
  "depth_lag_15m",
  "depth_lag_30m",
  "depth_rate_of_rise",
  "horizon_min"
]
```
