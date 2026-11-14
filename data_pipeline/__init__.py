"""
data_pipeline
=============
Crestix Probe XGBoost data engineering and feature extraction pipeline.
Target Domain: LarNO UKEA 32x32 benchmark.
"""

from data_pipeline.load_data import (
    FloodEventData,
    create_sample_dataset,
    generate_synthetic_larno_event,
    load_all_events,
    load_event_from_directory,
    save_events_to_disk,
)
from data_pipeline.feature_engineering import (
    build_event_tabular_dataframe,
    compute_hydraulic_lag_features,
    compute_rainfall_features,
    compute_spatial_temporal_features,
    compute_topographic_features,
    get_feature_column_names,
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

__all__ = [
    "FloodEventData",
    "load_event_from_directory",
    "load_all_events",
    "generate_synthetic_larno_event",
    "save_events_to_disk",
    "create_sample_dataset",
    "compute_topographic_features",
    "compute_rainfall_features",
    "compute_hydraulic_lag_features",
    "compute_spatial_temporal_features",
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
]

