"""CLI entry point to evaluate trained XGBoost model on held-out flood events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to Python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from crestix_xgb.data import LarNODataPipeline, TARGET_COLUMN
from crestix_xgb.model import CrestixXGBRegressor, CrestixXGBClassifier
from crestix_xgb.evaluate import evaluate_heldout_events, generate_evaluation_markdown_report
from crestix_xgb.train import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Crestix Probe XGBoost Model on Held-out Events.")
    parser.add_argument("--config", type=str, default=str(repo_root / "configs" / "default_config.yaml"), help="Config file path")
    parser.add_argument("--model-path", type=str, default=str(repo_root / "local_artifacts" / "models" / "xgb_regressor.json"), help="Regressor model file")
    parser.add_argument("--classifier-path", type=str, default=str(repo_root / "local_artifacts" / "models" / "xgb_classifier.json"), help="Classifier model file")
    parser.add_argument("--metadata-path", type=str, default=str(repo_root / "artifacts" / "metadata" / "model_metadata.json"), help="Metadata JSON path")
    parser.add_argument("--output-dir", type=str, default=str(repo_root / "metrics"), help="Metrics directory")
    args = parser.parse_args()

    config = load_config(args.config)
    data_pipe = LarNODataPipeline(config)
    df_raw = data_pipe.generate_larno_benchmark_dataset(seed=42)
    _, _, df_test = data_pipe.split_by_events(df_raw)

    meta_file = Path(args.metadata_path)
    feature_names = data_pipe.feature_names
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta = json.load(f)
            feature_names = meta.get("feature_order", feature_names)

    regressor = CrestixXGBRegressor()
    regressor.load(args.model_path, feature_names=feature_names)

    classifier = None
    if Path(args.classifier_path).exists():
        classifier = CrestixXGBClassifier()
        classifier.load(args.classifier_path, feature_names=feature_names)

    print("Running held-out event evaluation...")
    summary = evaluate_heldout_events(
        model=regressor,
        df_heldout=df_test,
        feature_names=feature_names,
        target_col=TARGET_COLUMN,
        classifier_model=classifier,
        horizons=data_pipe.horizons,
    )

    report_md = generate_evaluation_markdown_report(summary)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "README.md"
    with open(report_path, "w") as f:
        f.write(report_md)

    json_path = out_dir / "evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    overall = summary["overall_continuous_metrics"]
    print("\n--- Evaluation Summary ---")
    print(f"Held-out Events: {summary['heldout_events']}")
    print(f"MAE:  {overall['mae']:.4f} m")
    print(f"RMSE: {overall['rmse']:.4f} m")
    print(f"R2:   {overall['r2']:.4f}")
    print(f"Bias: {overall['bias']:.4f} m")
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()

