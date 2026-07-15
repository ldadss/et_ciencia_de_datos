"""
Módulo de TRANSFORMACIÓN (Transform) del pipeline ETL.

Responsabilidades:
- Limpieza de datos (tipos, duplicados, valores fuera de rango)
- Normalización de nombres (país/equipo) para el cruce con la fuente externa
- Enriquecimiento con la dimensión "continente"
- Ingeniería de features (métricas por 90 minutos, eficiencia, etc.)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columnas numéricas que jamás deberían ser negativas
NON_NEGATIVE_COLS = [
    "age", "height_cm", "weight_kg", "minutes_played", "goals", "assists",
    "shots", "shots_on_target", "expected_goals_xg", "expected_assists_xa",
    "key_passes", "successful_passes", "total_passes", "dribbles_attempted",
    "successful_dribbles", "tackles", "interceptions", "clearances",
]


def clean_players_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica del dataset principal."""
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    logger.info("Duplicados eliminados: %s", before - len(df))

    # Tipos de fecha
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Elimina filas sin identificadores clave
    df = df.dropna(subset=["player_id", "match_id", "team"])

    # Corrige valores negativos imposibles (clip a 0) en columnas de conteo
    existing_cols = [c for c in NON_NEGATIVE_COLS if c in df.columns]
    for col in existing_cols:
        n_bad = (df[col] < 0).sum()
        if n_bad > 0:
            logger.warning("Corrigiendo %s valores negativos en %s", n_bad, col)
            df[col] = df[col].clip(lower=0)

    # pass_accuracy y save_percentage deben estar entre 0 y 1
    for col in ["pass_accuracy", "save_percentage"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)

    # Texto: quita espacios extra
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    return df


def enrich_with_continent(df: pd.DataFrame, continents: pd.DataFrame) -> pd.DataFrame:
    """Une el dataset principal con la fuente externa país -> continente."""
    merged = df.merge(continents, how="left", on="team")
    missing = merged["continent"].isna().sum()
    if missing:
        logger.warning("%s filas sin continente asignado tras el merge", missing)
        merged["continent"] = merged["continent"].fillna("Desconocido")
    return merged


def add_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de features: métricas normalizadas por 90 minutos y ratios."""
    df = df.copy()
    minutes_safe = df["minutes_played"].replace(0, np.nan)

    df["goals_per90"] = (df["goals"] / minutes_safe * 90).fillna(0)
    df["assists_per90"] = (df["assists"] / minutes_safe * 90).fillna(0)
    df["shots_per90"] = (df["shots"] / minutes_safe * 90).fillna(0)
    df["goal_contribution"] = df["goals"] + df["assists"]
    df["goal_contribution_per90"] = (df["goal_contribution"] / minutes_safe * 90).fillna(0)

    df["shot_accuracy"] = np.where(
        df["shots"] > 0, df["shots_on_target"] / df["shots"], 0
    )
    df["dribble_success_rate"] = np.where(
        df["dribbles_attempted"] > 0,
        df["successful_dribbles"] / df["dribbles_attempted"],
        0,
    )
    df["cross_success_rate"] = np.where(
        df["crosses"] > 0, df["successful_crosses"] / df["crosses"], 0
    )
    df["xg_overperformance"] = df["goals"] - df["expected_goals_xg"]
    df["xa_overperformance"] = df["assists"] - df["expected_assists_xa"]

    # Categoría etaria, útil para segmentación en el dashboard
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 20, 24, 29, 34, 100],
        labels=["Sub-20", "21-24", "25-29", "30-34", "35+"],
    )

    return df


def transform(players: pd.DataFrame, continents: pd.DataFrame) -> pd.DataFrame:
    """Orquesta toda la etapa de transformación."""
    logger.info("Iniciando transformación...")
    df = clean_players_data(players)
    df = enrich_with_continent(df, continents)
    df = add_performance_features(df)
    logger.info("Transformación completa: %s filas, %s columnas", *df.shape)
    return df
