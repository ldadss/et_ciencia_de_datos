# ⚽ FIFA World Cup 2026 — Player Performance Analytics

Proyecto integrador de la Evaluación Final Transversal — **Programación para la Ciencia de Datos**.

Solución completa de ciencia de datos que integra gestión de datos, un pipeline
ETL, modelos de Machine Learning y un dashboard interactivo, simulando un
entorno profesional.

---

## 1. Caso abordado

Se analiza el rendimiento de los jugadores del **FIFA World Cup 2026** a nivel
partido, con el objetivo de:

- Consolidar y limpiar estadísticas de rendimiento (75 variables originales).
- Enriquecer los datos con una fuente externa (continente por país).
- Predecir el **rendimiento de un jugador** en un partido:
  - Probabilidad de **contribución de gol** (marcar o asistir).
  - **Calificación de rendimiento** (`player_rating`) estimada.
- Explorar los resultados mediante un dashboard interactivo.

## 2. Fuentes de datos

| Fuente | Descripción | Registros |
|---|---|---|
| `data/raw/fifa_world_cup_2026_player_performance.csv` | Estadísticas de jugadores por partido (Kaggle) | 54.600 filas × 75 columnas |
| `data/external/country_continent.csv` | Mapeo país → continente (Our World in Data, vía GitHub) | 287 países |

> ⚠️ **Nota sobre la naturaleza del dataset:** el dataset principal es
> **sintético/simulado**, no estadísticas reales recopiladas de partidos.
> Esto se evidencia en su calidad "perfecta" (0 nulos, 0 duplicados) y en
> que modela un torneo que aún no se disputa. Se eligió igualmente porque
> permite desarrollar un caso realista de ciencia de datos (ETL, ML,
> dashboard) con una estructura de datos verosímil. Este punto se declara
> explícitamente en la presentación del equipo.

## 3. Arquitectura del proyecto

```
proyecto_fifa2026/
├── data/
│   ├── raw/              # Dataset original de Kaggle
│   ├── external/         # Fuente externa país-continente
│   └── processed/        # Salida del pipeline ETL (parquet + csv)
├── etl/
│   ├── extract.py        # Extracción de ambas fuentes
│   ├── transform.py      # Limpieza + enriquecimiento + feature engineering
│   ├── load.py            # Persistencia en parquet/csv
│   └── run_pipeline.py   # Orquestador
├── models/
│   ├── train_model.py    # Entrenamiento de los 2 modelos
│   ├── model_goal_involvement.pkl
│   ├── model_player_rating.pkl
│   └── metrics.json      # Métricas de evaluación
├── dashboard/
│   └── app.py             # Dashboard Dash/Plotly
├── notebooks/
│   └── eda.ipynb          # Análisis exploratorio
├── tests/                 # Tests unitarios (pytest)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml   # Pipeline CI/CD
```

## 4. Pipeline ETL

1. **Extract**: lee el CSV de Kaggle y el CSV externo de continentes.
2. **Transform**:
   - Limpieza: duplicados, tipos de datos, valores negativos imposibles,
     porcentajes fuera de rango (`pass_accuracy`, `save_percentage`).
   - Enriquecimiento: `merge` con la dimensión `continent`.
   - Feature engineering: métricas por 90 minutos, tasas de efectividad
     (tiro, regate, centro), sobre-rendimiento respecto al xG/xA, grupos etarios.
3. **Load**: guarda el resultado en `data/processed/` (parquet + csv).

Ejecutar:
```bash
python -m etl.run_pipeline
```

## 5. Machine Learning

Se entrenan **dos modelos** (Random Forest, dentro de un `Pipeline` de
scikit-learn con preprocesamiento incluido) usando únicamente variables de
proceso/contexto (sin fuga de datos del resultado):

| Modelo | Tipo | Target | Métrica principal |
|---|---|---|---|
| `model_goal_involvement.pkl` | Clasificación | ¿El jugador participó en un gol (marcó o asistió)? | ROC-AUC ≈ 0.87 |
| `model_player_rating.pkl` | Regresión | Calificación de rendimiento (`player_rating`) | R² ≈ 0.98 |

Ver métricas completas en `models/metrics.json`.

Entrenar (requiere haber corrido el ETL primero):
```bash
python -m models.train_model
```

## 6. Dashboard interactivo (Dash + Plotly)

Tres secciones:

1. **Resumen general**: KPIs, goles por selección, top goleadores/asistentes,
   xG vs. goles reales, distribución por posición — con filtros por etapa,
   posición y continente.
2. **Comparador de jugadores**: radar de perfil de rendimiento y evolución
   de calificación por partido entre dos jugadores.
3. **Predicción de rendimiento**: formulario interactivo que consume los
   modelos entrenados y estima la probabilidad de contribución de gol y la
   calificación de rendimiento.

Ejecutar localmente:
```bash
python dashboard/app.py
# abrir http://localhost:8050
```

## 7. Cómo ejecutar todo el proyecto

### Opción A — entorno local
```bash
python -m venv .venv && source .venv/bin/activate      # o .venv\Scripts\activate en Windows
pip install -r requirements.txt

python -m etl.run_pipeline        # 1. ETL
python -m models.train_model      # 2. Entrenamiento
pytest tests/ -v                  # 3. Tests
python dashboard/app.py           # 4. Dashboard
```

### Opción B — Docker
```bash
docker compose up --build
# abrir http://localhost:8050
```
> Nota: `docker-compose.yml` monta `data/` y `models/`, por lo que conviene
> haber corrido el ETL y el entrenamiento al menos una vez antes de levantar
> el contenedor (o extender el `Dockerfile` para ejecutarlos en el build).

## 8. CI/CD

`.github/workflows/ci.yml` ejecuta en cada push/PR:
1. Lint (`flake8`)
2. Pipeline ETL
3. Entrenamiento de modelos
4. Tests unitarios (`pytest`)
5. Build de la imagen Docker

## 9. Tests

11 tests unitarios cubren:
- Limpieza y transformación de datos (`tests/test_etl.py`)
- Predicciones válidas de ambos modelos, incluyendo categorías no vistas
  (`tests/test_model.py`)

```bash
pytest tests/ -v
```

## 10. Stack tecnológico

Python · Pandas · Scikit-learn · Dash/Plotly · Dash Bootstrap Components ·
Docker · GitHub Actions · Pytest · Parquet (PyArrow)

## 11. Autores

_Completar con los nombres del equipo y el aporte individual de cada
integrante, según lo solicitado en la pauta de evaluación (la nota es
individual)._

| Integrante | Aporte principal |
|---|---|
| — | — |
| — | — |
| — | — |
