"""Unit tests for LarNO UKEA benchmark data pipeline and zero-leakage event splits."""

import pandas as pd
import pytest

from crestix_xgb.data import (
    LarNODataPipeline,
    FEATURE_NAMES,
    TARGET_COLUMN,
    discretize_flood_risk,
)


def test_larno_dataset_generation():
    df = LarNODataPipeline.generate_larno_benchmark_dataset(seed=42)
    assert not df.empty
    assert TARGET_COLUMN in df.columns
    assert "event_id" in df.columns

    for f in FEATURE_NAMES:
        assert f in df.columns, f"Missing feature in generated data: {f}"

    # Target depths must be physically non-negative
    assert (df[TARGET_COLUMN] >= 0.0).all()


def test_strict_event_based_isolation():
    """Verifies complete disjointness of training, validation, and held-out test events."""
    data_pipe = LarNODataPipeline()
    df = data_pipe.generate_larno_benchmark_dataset(seed=42)
    df_train, df_val, df_test = data_pipe.split_by_events(df)

    train_events = set(df_train["event_id"].unique())
    val_events = set(df_val["event_id"].unique())
    test_events = set(df_test["event_id"].unique())

    # Zero overlap between all sets
    assert train_events.isdisjoint(val_events), "Train and Val events must be disjoint"
    assert train_events.isdisjoint(test_events), "Train and Test events must be disjoint"
    assert val_events.isdisjoint(test_events), "Val and Test events must be disjoint"

    assert len(train_events) == 6
    assert len(val_events) == 2
    assert len(test_events) == 3


def test_preprocessor_fitting_and_transformation():
    data_pipe = LarNODataPipeline()
    df = data_pipe.generate_larno_benchmark_dataset(seed=42)
    df_train, _, _ = data_pipe.split_by_events(df)

    data_pipe.fit_preprocessor(df_train)
    assert len(data_pipe.preprocessor_stats) == len(FEATURE_NAMES)

    # Test transforming dataframe with some missing columns / NaNs
    df_dirty = df_train.head(5).copy()
    df_dirty.loc[0, "elevation_m"] = None
    transformed = data_pipe.transform_features(df_dirty)

    assert not transformed.isna().any().any()
    assert list(transformed.columns) == FEATURE_NAMES

