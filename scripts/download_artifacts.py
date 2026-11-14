"""Helper script to download model checkpoints and artifacts from Google Drive."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DRIVE_FOLDER = "https://drive.google.com/drive/folders/crestix-probe-xgb-v1"


def main():
    parser = argparse.ArgumentParser(description="Download Crestix Probe XGBoost Artifacts.")
    parser.add_argument("--drive-url", type=str, default=DEFAULT_DRIVE_FOLDER, help="Google Drive folder URL")
    parser.add_argument("--dest", type=str, default="local_artifacts", help="Destination folder")
    args = parser.parse_args()

    dest_path = Path(args.dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(" Crestix Probe — Google Drive Artifact Deliverables")
    print("=================================================================")
    print(f"Target Google Drive: {args.drive_url}")
    print(f"Destination Folder:  {dest_path.resolve()}\n")
    print("To download the production binaries using gdown:")
    print("  pip install gdown")
    print(f"  gdown --folder \"{args.drive_url}\" -O {args.dest}")
    print("\nOr manually place `xgb_regressor.json` in `local_artifacts/models/`")
    print("and verify with the checksums in `artifacts/README.md`.")
    print("=================================================================")


if __name__ == "__main__":
    main()

