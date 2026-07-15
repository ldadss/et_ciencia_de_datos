"""Tests unitarios para los artefactos de Machine Learning."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CLF_PATH = BASE_DIR / "models" / "model_goal_involvement.pkl"
MODEL_REG_PATH = BASE_DIR / "models" / "model_player_rating.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "players_processed.parquet"

NUMERIC_FEATURES = [
    "age", "height_cm", "weight_kg", "minutes_played",
    "shots", "shots_on_target", "key_passes", "successful_passes",
    "total_passes", "pass_accuracy", "dribbles_attempted",
    "successful_dribbles", "crosses", "successful_crosses",
    "distance_covered_km", "sprint_distance_km", "top_speed_kmh",
    "accelerations", "decelerations", "stamina_score",
    "market_value_eur", "expected_goals_xg", "expected_assists_xa",
]
CATEGORICAL_FEATURES = ["position", "preferred_foot", "continent", "tournament_stage"]


@pytest.fixture(scope="module")
def processed_df():
    if not DATA_PATH.exists():
        pytest.skip("Ejecuta primero el pipeline ETL (python -m etl.run_pipeline)")
    return pd.read_parquet(DATA_PATH)


@pytest.fixture(scope="module")
def classifier():
    if not MODEL_CLF_PATH.exists():
        pytest.skip("Ejecuta primero el entrenamiento (python -m models.train_model)")
    return joblib.load(MODEL_CLF_PATH)


@pytest.fixture(scope="module")
def regressor():
    if not MODEL_REG_PATH.exists():
        pytest.skip("Ejecuta primero el entrenamiento (python -m models.train_model)")
    return joblib.load(MODEL_REG_PATH)


def test_classifier_predicts_valid_probabilities(classifier, processed_df):
    sample = processed_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(5)
    proba = classifier.predict_proba(sample)
    assert proba.shape == (5, 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_regressor_predicts_reasonable_ratings(regressor, processed_df):
    sample = processed_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(5)
    predictions = regressor.predict(sample)
    assert len(predictions) == 5
    assert (predictions >= 0).all() and (predictions <= 10).all()


def test_model_handles_unseen_categorical_gracefully(classifier, processed_df):
    sample = processed_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(1).copy()
    sample["continent"] = "Categoria_Nueva_No_Vista"
    # OneHotEncoder(handle_unknown="ignore") no debe lanzar excepción
    proba = classifier.predict_proba(sample)
    assert proba.shape == (1, 2)
