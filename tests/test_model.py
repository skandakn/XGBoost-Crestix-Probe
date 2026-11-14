"""Unit tests for CrestixXGBRegressor and CrestixXGBClassifier."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from crestix_xgb.model import CrestixXGBRegressor, CrestixXGBClassifier
from crestix_xgb.data import FEATURE_NAMES


def test_regressor_initialization():
    model = CrestixXGBRegressor(params={"max_depth": 4, "learning_rate": 0.1})
    assert model.params["max_depth"] == 4
    assert model.params["learning_rate"] == 0.1
    assert model.model is None


def test_regressor_fit_predict_save_load():
    rng = np.random.default_rng(42)
    n_samples = 80
    X = rng.normal(0, 1, size=(n_samples, len(FEATURE_NAMES)))
    # Generate non-negative true depths
    y = np.clip(np.abs(X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.normal(0, 0.1, n_samples)), 0.0, 3.0)

    df_X = pd.DataFrame(X, columns=FEATURE_NAMES)
    df_y = pd.Series(y)

    model = CrestixXGBRegressor(params={"n_estimators": 10, "max_depth": 3})
    model.fit(df_X, df_y)

    preds = model.predict(df_X)
    assert len(preds) == n_samples
    assert np.all(preds >= 0.0), "Predicted water depth must be non-negative"

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_regressor.json"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = CrestixXGBRegressor()
        loaded_model.load(save_path, feature_names=FEATURE_NAMES)
        loaded_preds = loaded_model.predict(df_X)

        np.testing.assert_allclose(preds, loaded_preds, rtol=1e-5, atol=1e-5)


def test_classifier_fit_predict():
    rng = np.random.default_rng(42)
    n_samples = 60
    X = rng.normal(0, 1, size=(n_samples, len(FEATURE_NAMES)))
    y = rng.integers(0, 4, size=n_samples)

    clf = CrestixXGBClassifier(params={"n_estimators": 10, "max_depth": 3, "num_class": 4})
    clf.fit(X, y, feature_names=FEATURE_NAMES)

    preds = clf.predict(X)
    assert len(preds) == n_samples
    assert set(np.unique(preds)).issubset({0, 1, 2, 3})

    probs = clf.predict_proba(X)
    assert probs.shape == (n_samples, 4)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(n_samples), rtol=1e-5)

