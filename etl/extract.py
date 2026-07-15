"""
Módulo de EXTRACCIÓN (Extract) del pipeline ETL.

Fuentes de datos:
1. Dataset principal: estadísticas de jugadores por partido - Kaggle
   (FIFA World Cup 2026 Player Performance Dataset)
2. Dataset externo: mapeo país -> continente (Our World in Data / GitHub)
   usado para enriquecer el análisis con una dimensión geográfica.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PLAYERS_PATH = BASE_DIR / "data" / "raw" / "fifa_world_cup_2026_player_performance.csv"
EXTERNAL_CONTINENT_PATH = BASE_DIR / "data" / "external" / "country_continent.csv"


def extract_players_data(path: Path = RAW_PLAYERS_PATH) -> pd.DataFrame:
    """Lee el dataset principal de rendimiento de jugadores."""
    logger.info("Extrayendo dataset principal desde %s", path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo fuente en {path}. "
            "Verifica que el CSV esté en data/raw/."
        )
    df = pd.read_csv(path)
    logger.info("Dataset principal extraído: %s filas, %s columnas", *df.shape)
    return df


def extract_continent_data(path: Path = EXTERNAL_CONTINENT_PATH) -> pd.DataFrame:
    """Lee la fuente de datos externa (país -> continente)."""
    logger.info("Extrayendo dataset externo desde %s", path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo externo en {path}. "
            "Ejecuta el script de descarga o revisa data/external/."
        )
    df = pd.read_csv(path)
    logger.info("Dataset externo extraído: %s filas", len(df))
    return df


def extract_all() -> dict:
    """Extrae todas las fuentes de datos y las retorna en un diccionario."""
    return {
        "players": extract_players_data(),
        "continents": extract_continent_data(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = extract_all()
    for name, df in data.items():
        print(f"{name}: {df.shape}")
