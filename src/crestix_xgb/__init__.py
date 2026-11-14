"""
crestix_xgb
===========
Core package for Crestix Probe XGBoost urban flood forecasting.
Provides dataset generation, feature extraction, train/val/test partitioning,
and inference integration for the LarNO UKEA 32x32 benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from data_pipeline.feature_engineering import (
    build_event_tabular_dataframe,
    get_feature_column_names,
)
from data_pipeline.load_data import (
    FloodEventData,
    create_sample_dataset,
    generate_synthetic_larno_event,
    load_all_events,
    load_event_from_directory,
    save_events_to_disk,
)
from data_pipeline.target_generation import (
    TARGET_DEFINITIONS,
    TargetMetadata,
    attach_targets_to_dataframe,
    generate_event_targets,
    get_target_metadata,
)
from data_pipeline.validation import (
    DataLeakageError,
    EventAwareSplitter,
    TabularPreprocessor,
    check_data_leakage,
    validate_tabular_dataset,
)

__version__ = "0.1.0"


def build_training_ready_splits(
    data_dir: str = "data/sample",
    config_path: str = "configs/baseline.yaml",
    primary_target_col: str = "target_depth_lead_30m",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    High-level end-to-end pipeline function for Prompt 3 training lead.

    Loads all flood events from data_dir, extracts features, attaches targets,
    performs strict event-aware splitting, executes data leakage checks, and
    returns (train_df, val_df, test_df, feature_names).

    Parameters
    ----------
    data_dir : str
        Directory containing raw flood event subfolders or .npz archives.
    config_path : str
        Path to baseline.yaml configuration file.
    primary_target_col : str
        Target column for filtering valid lead-time observations.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]
        (train_df, val_df, test_df, feature_names)
    """
    cfg: Dict[str, Any] = {}
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    split_cfg = cfg.get("splitting", {})
    train_ratio = split_cfg.get("train_ratio", 0.60)
    val_ratio = split_cfg.get("val_ratio", 0.20)
    test_ratio = split_cfg.get("test_ratio", 0.20)
    seed = split_cfg.get("random_seed", 42)

    # 1. Load all events
    data_path = Path(data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        # Auto-populate sample dataset if directory is empty
        create_sample_dataset(output_dir=data_path, n_events=5)

    events = load_all_events(data_path)

    # 2. Extract features and attach targets per event
    dfs: List[pd.DataFrame] = []
    for ev in events:
        raw_df = build_event_tabular_dataframe(ev, config=cfg)
        event_df = attach_targets_to_dataframe(
            raw_df, ev, drop_nan_targets=True, primary_target_col=primary_target_col
        )
        dfs.append(event_df)

    consolidated_df = pd.concat(dfs, axis=0, ignore_index=True)
    feature_cols = get_feature_column_names(consolidated_df)

    # 3. Perform Event-Aware Splitting
    splitter = EventAwareSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )
    train_df, val_df, test_df = splitter.split_dataframe(consolidated_df)

    # 4. Check for Data Leakage
    check_data_leakage(
        train_df, val_df, test_df, feature_cols=feature_cols
    )

    return train_df, val_df, test_df, feature_cols


__all__ = [
    "FloodEventData",
    "load_event_from_directory",
    "load_all_events",
    "generate_synthetic_larno_event",
    "save_events_to_disk",
    "create_sample_dataset",
    "build_event_tabular_dataframe",
    "get_feature_column_names",
    "TARGET_DEFINITIONS",
    "TargetMetadata",
    "generate_event_targets",
    "attach_targets_to_dataframe",
    "get_target_metadata",
    "DataLeakageError",
    "EventAwareSplitter",
    "check_data_leakage",
    "TabularPreprocessor",
    "validate_tabular_dataset",
    "build_training_ready_splits",
]

