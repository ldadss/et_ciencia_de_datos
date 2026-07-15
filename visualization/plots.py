"""
Visualizaciones del análisis exploratorio — FIFA World Cup 2026.

Cada función recibe el DataFrame procesado (salida del pipeline ETL) y
devuelve una figura de Plotly, sin efectos secundarios (no llama a
``.show()``). Esto permite reutilizarlas desde notebooks, scripts o el
dashboard, y facilita testearlas.

Uso típico en un notebook:

    import pandas as pd
    from visualization import plots

    df = pd.read_parquet("data/processed/players_processed.parquet")
    plots.top_goal_contributors(df).show()
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def top_goal_contributors(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Barras horizontales apiladas: top N por goles + asistencias."""
    top = df.groupby("player_name", as_index=False).agg(
        team=("team", "first"),
        goles=("goals", "sum"),
        asistencias=("assists", "sum"),
    )
    top["contribucion"] = top["goles"] + top["asistencias"]
    top = top.sort_values("contribucion", ascending=False).head(n)
    top["jugador"] = top["player_name"] + " (" + top["team"] + ")"
    # Orden ascendente para que la barra más alta quede arriba.
    top = top.sort_values("contribucion")

    fig = go.Figure()
    fig.add_bar(
        y=top["jugador"], x=top["goles"],
        name="Goles", orientation="h", marker_color="#2a78d6",
    )
    fig.add_bar(
        y=top["jugador"], x=top["asistencias"],
        name="Asistencias", orientation="h", marker_color="#1baf7a",
    )
    fig.update_layout(
        barmode="stack",
        title=f"Top {n} jugadores por Goles + Asistencias — Mundial 2026",
        xaxis_title="Cantidad",
        height=500,
    )
    return fig


def top_rated_players(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Barras horizontales: top N por calificación de rendimiento promedio."""
    top = (
        df.groupby("player_name", as_index=False)
        .agg(team=("team", "first"), rating=("player_rating", "mean"))
        .sort_values("rating", ascending=False)
        .head(n)
    )
    top["jugador"] = top["player_name"] + " (" + top["team"] + ")"
    top = top.sort_values("rating")

    fig = px.bar(
        top, x="rating", y="jugador", orientation="h",
        color="rating", color_continuous_scale="Purples",
        title=f"Top {n} jugadores por rendimiento promedio — Mundial 2026",
    )
    fig.update_layout(
        xaxis_range=[top["rating"].min() - 0.2, top["rating"].max() + 0.2],
        xaxis_title="Calificación promedio (player_rating)",
        yaxis_title="",
        height=500,
        coloraxis_showscale=False,
    )
    return fig


def goals_over_time(df: pd.DataFrame) -> go.Figure:
    """Líneas: evolución de goles totales por fecha de partido."""
    goles_por_fecha = (
        df.groupby("match_date", as_index=False)["goals"]
        .sum()
        .sort_values("match_date")
    )
    fig = px.line(
        goles_por_fecha,
        x="match_date",
        y="goals",
        markers=True,
        title="Evolución de goles totales por fecha de partido — Mundial 2026",
        labels={"match_date": "Fecha del partido", "goals": "Goles totales"},
    )
    return fig


def goals_vs_xg(df: pd.DataFrame) -> go.Figure:
    """Dispersión: goles reales vs goles esperados (xG), coloreado por posición."""
    fig = px.scatter(
        df,
        x="expected_goals_xg",
        y="goals",
        color="position",
        hover_name="player_name",
        opacity=0.6,
        title="Goles reales vs goles esperados (xG) — Mundial 2026",
        labels={
            "expected_goals_xg": "Goles esperados (xG)",
            "goals": "Goles reales",
            "position": "Posición",
        },
    )
    return fig


def confusion_matrix_heatmap(cm) -> go.Figure:
    """Heatmap de la matriz de confusión del modelo de contribución de gol.

    ``cm`` es la matriz 2x2 devuelta por
    ``sklearn.metrics.confusion_matrix``.
    """
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicción", y="Valor real", color="Cantidad"),
        x=["No contribuye", "Contribuye a gol"],
        y=["No contribuye", "Contribuye a gol"],
        color_continuous_scale="Blues",
        title="Matriz de confusión — Contribución de gol",
    )
    fig.update_layout(height=500, width=550)
    return fig


def third_place_matches(df: pd.DataFrame) -> go.Figure:
    """Barras: goles por partido en la etapa 'Third Place Match'.

    Evidencia que el dataset es sintético: un Mundial real tiene un único
    partido por el tercer lugar, pero aquí la etiqueta agrupa varios
    ``match_id`` distintos.
    """
    tercer_lugar = df[df["tournament_stage"] == "Third Place Match"]
    goles_por_partido = tercer_lugar.groupby("match_id", as_index=False)["goals"].sum()
    n_partidos = goles_por_partido["match_id"].nunique()

    fig = px.bar(
        goles_por_partido,
        x="match_id",
        y="goals",
        title=(
            "Goles por partido en la etapa 'Partido por el tercer puesto' "
            f"({n_partidos} partidos distintos)"
        ),
        labels={"match_id": "ID de partido", "goals": "Goles"},
        color="goals",
        color_continuous_scale="Reds",
    )
    return fig
