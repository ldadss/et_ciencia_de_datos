"""
Orquestador del pipeline ETL completo.

Uso:
    python etl/run_pipeline.py
"""

import logging

from etl.extract import extract_all
from etl.load import load
from etl.transform import transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== INICIO PIPELINE ETL ===")

    raw = extract_all()
    df_transformed = transform(raw["players"], raw["continents"])
    paths = load(df_transformed)

    logger.info("=== PIPELINE ETL FINALIZADO ===")
    logger.info("Archivos generados: %s", paths)
    return df_transformed


if __name__ == "__main__":
    main()
