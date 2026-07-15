"""
Módulo de CARGA (Load) del pipeline ETL.

Persiste los datos transformados en formato parquet (eficiente) y csv
(fácil de inspeccionar), listos para ser consumidos por el modelo de
Machine Learning y el dashboard.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load(df: pd.DataFrame, filename: str = "players_processed") -> dict:
    """Guarda el DataFrame procesado en parquet y csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = PROCESSED_DIR / f"{filename}.parquet"
    csv_path = PROCESSED_DIR / f"{filename}.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    logger.info("Datos cargados en %s y %s", parquet_path, csv_path)
    return {"parquet": parquet_path, "csv": csv_path}
