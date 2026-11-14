"""
test_validation_and_leakage.py
==============================
Unit tests for event-aware splitting, data leakage detection, and preprocessors.
"""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.feature_engineering import (
    build_event_tabular_dataframe,
    get_feature_column_names,
)
from data_pipeline.load_data import generate_synthetic_larno_event
from data_pipeline.target_generation import attach_targets_to_dataframe
from data_pipeline.validation import (
    DataLeakageError,
    EventAwareSplitter,
    TabularPreprocessor,
    check_data_leakage,
    validate_tabular_dataset,
)


def test_event_aware_splitter_disjoint():
    splitter = EventAwareSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42)
    event_ids = [f"event_{i:03d}" for i in range(1, 11)]

    train_ids, val_ids, test_ids = splitter.split_event_ids(event_ids)

    # Disjointness
    assert set(train_ids).isdisjoint(set(val_ids))
    assert set(train_ids).isdisjoint(set(test_ids))
    assert set(val_ids).isdisjoint(set(test_ids))

    # All events accounted for
    assert sorted(train_ids + val_ids + test_ids) == sorted(event_ids)


def test_check_data_leakage_passes_on_disjoint_data():
    train_df = pd.DataFrame({"event_id": ["ev_1", "ev_2"], "feat_1": [1.0, 2.0]})
    val_df = pd.DataFrame({"event_id": ["ev_3"], "feat_1": [3.0]})
    test_df = pd.DataFrame({"event_id": ["ev_4"], "feat_1": [4.0]})

    report = check_data_leakage(train_df, val_df, test_df, feature_cols=["feat_1"])
    assert report["status"] == "PASSED"


def test_check_data_leakage_detects_event_overlap():
    train_df = pd.DataFrame({"event_id": ["ev_1", "ev_2"], "feat_1": [1.0, 2.0]})
    val_df = pd.DataFrame({"event_id": ["ev_2"], "feat_1": [3.0]})  # Contaminated ev_2!
    test_df = pd.DataFrame({"event_id": ["ev_4"], "feat_1": [4.0]})

    with pytest.raises(DataLeakageError, match="Event overlap detected"):
        check_data_leakage(train_df, val_df, test_df, feature_cols=["feat_1"])


def test_check_data_leakage_detects_target_in_features():
    train_df = pd.DataFrame({"event_id": ["ev_1"], "feat_1": [1.0], "target_depth": [0.5]})
    val_df = pd.DataFrame({"event_id": ["ev_2"], "feat_1": [2.0], "target_depth": [0.6]})
    test_df = pd.DataFrame({"event_id": ["ev_3"], "feat_1": [3.0], "target_depth": [0.7]})

    with pytest.raises(DataLeakageError, match="Target variables leaked"):
        check_data_leakage(
            train_df, val_df, test_df, feature_cols=["feat_1", "target_depth"]
        )


def test_tabular_preprocessor_train_only_fit():
    # Train set has median 5.0
    train_df = pd.DataFrame({
        "elevation": [4.0, 5.0, 6.0],
        "rainfall_intensity_current": [-1.0, 2.0, 3.0],  # Negative intensity should be clipped
    })
    # Val set has a NaN that must be imputed with train median (5.0)
    val_df = pd.DataFrame({
        "elevation": [np.nan, 8.0, 9.0],
        "rainfall_intensity_current": [1.0, 2.0, 3.0],
    })

    prep = TabularPreprocessor(scale_features=False, clip_negative_features=True)
    prep.fit(train_df)

    val_trans = prep.transform(val_df)

    # Imputed with train median: 5.0
    assert np.isclose(val_trans["elevation"].iloc[0], 5.0)
    # Check clipping of negative physical features on train
    train_trans = prep.transform(train_df)
    assert train_trans["rainfall_intensity_current"].iloc[0] == 0.0

