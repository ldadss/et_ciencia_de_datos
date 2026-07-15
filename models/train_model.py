"""
Entrenamiento de modelos de Machine Learning para predicción de rendimiento.

Se entrenan dos modelos complementarios sobre datos de rendimiento a nivel
partido (evitando fuga de datos: solo se usan variables de proceso/físicas/
contextuales, nunca variables derivadas directamente del resultado final):

1. CLASIFICACIÓN: probabilidad de que un jugador tenga una "contribución de
   gol" (marque o asista) en un partido dado su perfil y comportamiento.
2. REGRESIÓN: predicción de la calificación de rendimiento (player_rating)
   del jugador en el partido.

Uso:
    python models/train_model.py
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "players_processed.parquet"
MODELS_DIR = BASE_DIR / "models"

# Features de proceso/contexto disponibles ANTES o INDEPENDIENTES del resultado
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

TARGET_CLASSIFICATION = "goal_involvement"  # 1 si goles+asistencias > 0
TARGET_REGRESSION = "player_rating"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Ejecuta primero el pipeline ETL: "
            "python -m etl.run_pipeline"
        )
    df = pd.read_parquet(DATA_PATH)
    df[TARGET_CLASSIFICATION] = (df["goal_contribution"] > 0).astype(int)
    return df


def train_classification(df: pd.DataFrame) -> dict:
    logger.info("Entrenando modelo de CLASIFICACIÓN (contribución de gol)...")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_CLASSIFICATION]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    logger.info("Métricas clasificación: %s", metrics)

    joblib.dump(pipeline, MODELS_DIR / "model_goal_involvement.pkl")
    return metrics


def train_regression(df: pd.DataFrame) -> dict:
    logger.info("Entrenando modelo de REGRESIÓN (player_rating)...")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_REGRESSION]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=14,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "r2_score": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_test, y_pred), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    logger.info("Métricas regresión: %s", metrics)

    joblib.dump(pipeline, MODELS_DIR / "model_player_rating.pkl")
    return metrics


def main():
    df = load_dataset()
    clf_metrics = train_classification(df)
    reg_metrics = train_regression(df)

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classification_metrics": clf_metrics,
        "regression_metrics": reg_metrics,
    }
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Modelos y métricas guardados en %s", MODELS_DIR)


if __name__ == "__main__":
    main()
