"""CLI entry point to execute XGBoost flood-depth model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to Python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from crestix_xgb.train import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train Crestix Probe XGBoost Model for LarNO UKEA region.")
    parser.add_argument("--config", type=str, default=str(repo_root / "configs" / "default_config.yaml"), help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default=str(repo_root), help="Root directory for outputs")
    parser.add_argument("--tune", action="store_true", help="Enable validation hyperparameter search")
    args = parser.parse_args()

    print(f"Starting training pipeline with config: {args.config}")
    results = run_training_pipeline(config_path=args.config, output_root=args.output_dir, tune_hyperparameters=args.tune)
    metrics = results["metadata"]["evaluation_metrics"]
    print("\n--- Training & Evaluation Complete ---")
    print(f"Model Version: {results['metadata']['model_version']}")
    print(f"MAE:  {metrics['mae']:.4f} m ({metrics['mae']*100:.2f} cm)")
    print(f"RMSE: {metrics['rmse']:.4f} m ({metrics['rmse']*100:.2f} cm)")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"Bias: {metrics['bias']:.4f} m")
    print(f"Evaluation report written to: {results['metrics_report_path']}")
    print(f"Artifacts manifest written to: {results['artifacts_readme_path']}")


if __name__ == "__main__":
    main()

