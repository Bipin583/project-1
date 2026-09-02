"""
Unit tests for LightGBM Model, Serialization, and Evaluation Metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.models.trainer import prepare_feature_arrays, evaluate_predictions, ModelTrainer


@pytest.fixture
def synthetic_training_data():
    """Generate synthetic tabular feature matrix for model testing."""
    rng = np.random.RandomState(42)
    n_samples = 100
    X = rng.randn(n_samples, len(FEATURE_NAMES)).astype(np.float32)
    # Binary labels with 15% positive failure rate
    y = (rng.rand(n_samples) < 0.15).astype(int)
    # Ensure at least 2 positive samples
    y[0] = 1
    y[1] = 1

    df_dict = {name: X[:, i] for i, name in enumerate(FEATURE_NAMES)}
    df_dict["label_failed"] = y
    df = pd.DataFrame(df_dict)
    return df, X, y


def test_lightgbm_training_and_prediction(synthetic_training_data):
    """Verify LightGBM model training, probability prediction bounds, and feature importances."""
    _, X, y = synthetic_training_data
    predictor = LightGBMTestPredictor(random_seed=42, n_estimators=20)
    diag = predictor.train(X_train=X, y_train=y)

    assert diag["n_features"] == 32
    assert diag["n_train_samples"] == 100

    probs = predictor.predict_proba(X)
    assert len(probs) == 100
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    importances = predictor.get_feature_importances()
    assert len(importances) == 32
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-2)


def test_model_serialization_roundtrip(synthetic_training_data, tmp_path: Path):
    """Verify model can be saved and loaded with identical prediction outputs."""
    _, X, y = synthetic_training_data
    predictor = LightGBMTestPredictor(random_seed=42, n_estimators=20)
    predictor.train(X_train=X, y_train=y)
    original_probs = predictor.predict_proba(X[:10])

    save_path = tmp_path / "test_model.joblib"
    predictor.save(str(save_path))
    assert save_path.exists()

    loaded = LightGBMTestPredictor.load(str(save_path))
    loaded_probs = loaded.predict_proba(X[:10])

    np.testing.assert_allclose(original_probs, loaded_probs, rtol=1e-5)


def test_evaluation_metrics_computation():
    """Verify scientific metric calculations for PR-AUC, ROC-AUC, F1, and Brier score."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1, 0.2, 0.1])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["pr_auc"] > 0.8
    assert metrics["roc_auc"] == 1.0
    assert 0.0 <= metrics["brier_score"] <= 0.1


def test_model_trainer_pipeline(synthetic_training_data, tmp_path: Path):
    """Verify ModelTrainer executes full training, validation, and exports reports."""
    df, _, _ = synthetic_training_data
    train_df = df.iloc[:70]
    val_df = df.iloc[70:85]
    test_df = df.iloc[85:]

    trainer = ModelTrainer(output_dir=str(tmp_path), model_version="test_v1")
    report = trainer.train_and_evaluate(train_df=train_df, val_df=val_df, test_df=test_df)

    assert "model_file" in report
    assert Path(report["model_file"]).exists()
    assert "test_metrics" in report
    assert "top_10_features_by_gain" in report
    assert len(report["top_10_features_by_gain"]) <= 10
