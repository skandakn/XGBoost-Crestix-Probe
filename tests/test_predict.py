"""Unit tests for the clean inference interface: predict_depth(features)."""

import numpy as np
import pandas as pd
import pytest

from crestix_xgb.data import FEATURE_NAMES
from crestix_xgb.model import CrestixXGBRegressor
from crestix_xgb.predict import FloodInferenceEngine, predict_depth, set_default_engine


@pytest.fixture(scope="module")
def trained_engine():
    """Builds a calibrated fixture engine for testing."""
    rng = np.random.default_rng(42)
    n_samples = 100
    X = rng.uniform(0.5, 10.0, size=(n_samples, len(FEATURE_NAMES)))
    y = np.clip(0.05 * X[:, 0] + 0.02 * X[:, 8] + 0.1 * X[:, 14], 0.0, 3.0)

    df_X = pd.DataFrame(X, columns=FEATURE_NAMES)
    df_y = pd.Series(y)

    reg = CrestixXGBRegressor(params={"n_estimators": 15, "max_depth": 3})
    reg.fit(df_X, df_y)

    engine = FloodInferenceEngine()
    engine.regressor = reg
    engine.feature_names = list(FEATURE_NAMES)
    set_default_engine(engine)
    return engine


def test_predict_depth_with_dataframe(trained_engine):
    df_input = pd.DataFrame([{col: 1.0 for col in FEATURE_NAMES}])
    preds = predict_depth(df_input, engine=trained_engine)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 1
    assert preds[0] >= 0.0


def test_predict_depth_with_single_dict(trained_engine):
    sample_dict = {col: 2.5 for col in FEATURE_NAMES}
    preds = predict_depth(sample_dict, engine=trained_engine)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 1
    assert preds[0] >= 0.0


def test_predict_depth_with_list_of_dicts(trained_engine):
    samples = [{col: 1.5 for col in FEATURE_NAMES}, {col: 3.0 for col in FEATURE_NAMES}]
    preds = predict_depth(samples, engine=trained_engine)
    assert len(preds) == 2
    assert np.all(preds >= 0.0)


def test_predict_depth_with_shuffled_column_order(trained_engine):
    """Guarantees inference enforces identical feature ordering regardless of user dataframe order."""
    reversed_features = list(reversed(FEATURE_NAMES))
    values = {col: float(idx + 1) for idx, col in enumerate(FEATURE_NAMES)}

    df_correct = pd.DataFrame([values])[FEATURE_NAMES]
    df_shuffled = pd.DataFrame([values])[reversed_features]

    pred_correct = predict_depth(df_correct, engine=trained_engine)
    pred_shuffled = predict_depth(df_shuffled, engine=trained_engine)

    # Predictions must be strictly identical regardless of column presentation order
    np.testing.assert_allclose(pred_correct, pred_shuffled, rtol=1e-7)


def test_predict_depth_with_numpy_array(trained_engine):
    vec = np.ones(len(FEATURE_NAMES))
    preds = predict_depth(vec, engine=trained_engine)
    assert len(preds) == 1

    mat = np.ones((5, len(FEATURE_NAMES)))
    preds_mat = predict_depth(mat, engine=trained_engine)
    assert len(preds_mat) == 5


def test_predict_depth_with_invalid_dimension(trained_engine):
    with pytest.raises(ValueError, match="Input 1D numpy array has shape"):
        predict_depth(np.ones(5), engine=trained_engine)


def test_predict_risk_interface(trained_engine):
    sample = {col: 1.0 for col in FEATURE_NAMES}
    res = trained_engine.predict_risk(sample)
    assert "predicted_depth_m" in res
    assert "risk_class" in res
    assert "risk_label" in res
    assert len(res["predicted_depth_m"]) == 1
    assert res["risk_class"][0] in [0, 1, 2, 3]

