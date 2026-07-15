"""
Dashboard interactivo - FIFA World Cup 2026 Player Performance Analytics

Ejecutar:
    python dashboard/app.py

Luego abrir http://localhost:8050 en el navegador.
"""

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dcc, html

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "players_processed.parquet"
MODEL_CLF_PATH = BASE_DIR / "models" / "model_goal_involvement.pkl"
MODEL_REG_PATH = BASE_DIR / "models" / "model_player_rating.pkl"

# ----------------------------------------------------------------------
# Carga de datos y modelos
# ----------------------------------------------------------------------
df = pd.read_parquet(DATA_PATH)
clf_model = joblib.load(MODEL_CLF_PATH)
reg_model = joblib.load(MODEL_REG_PATH)

TEAMS = sorted(df["team"].unique())
POSITIONS = sorted(df["position"].unique())
STAGES = sorted(df["tournament_stage"].unique())
FEET = sorted(df["preferred_foot"].unique())
CONTINENTS = sorted(df["continent"].unique())

NUMERIC_FEATURES = [
    "age", "height_cm", "weight_kg", "minutes_played",
    "shots", "shots_on_target", "key_passes", "successful_passes",
    "total_passes", "pass_accuracy", "dribbles_attempted",
    "successful_dribbles", "crosses", "successful_crosses",
    "distance_covered_km", "sprint_distance_km", "top_speed_kmh",
    "accelerations", "decelerations", "stamina_score",
    "market_value_eur", "expected_goals_xg", "expected_assists_xa",
]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="FIFA World Cup 2026 - Analytics",
)
server = app.server  # necesario para despliegue (gunicorn / docker)

# ----------------------------------------------------------------------
# Componentes reutilizables
# ----------------------------------------------------------------------


def kpi_card(title, value, color="primary"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="text-muted"),
                html.H3(value, className=f"text-{color}"),
            ]
        ),
        className="shadow-sm text-center",
    )


filters = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Etapa del torneo"),
                            dcc.Dropdown(
                                id="filter-stage",
                                options=[{"label": s, "value": s} for s in STAGES],
                                value=None,
                                multi=True,
                                placeholder="Todas las etapas",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Posición"),
                            dcc.Dropdown(
                                id="filter-position",
                                options=[{"label": p, "value": p} for p in POSITIONS],
                                value=None,
                                multi=True,
                                placeholder="Todas las posiciones",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Continente"),
                            dcc.Dropdown(
                                id="filter-continent",
                                options=[{"label": c, "value": c} for c in CONTINENTS],
                                value=None,
                                multi=True,
                                placeholder="Todos los continentes",
                            ),
                        ],
                        md=4,
                    ),
                ]
            )
        ]
    ),
    className="mb-4 shadow-sm",
)

# ----------------------------------------------------------------------
# Tab 1: Resumen general
# ----------------------------------------------------------------------
tab_overview = html.Div(
    [
        html.Br(),
        dbc.Row(id="kpi-row", className="g-3 mb-4"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="graph-goals-by-team"), md=6),
                dbc.Col(dcc.Graph(id="graph-top-scorers"), md=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="graph-xg-vs-goals"), md=6),
                dbc.Col(dcc.Graph(id="graph-position-distribution"), md=6),
            ]
        ),
    ]
)

# ----------------------------------------------------------------------
# Tab 2: Comparador de jugadores
# ----------------------------------------------------------------------
tab_compare = html.Div(
    [
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Jugador A"),
                        dcc.Dropdown(id="player-a", options=sorted(df["player_name"].unique())),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.Label("Jugador B"),
                        dcc.Dropdown(id="player-b", options=sorted(df["player_name"].unique())),
                    ],
                    md=6,
                ),
            ],
            className="mb-4",
        ),
        dcc.Graph(id="graph-radar-comparison"),
        dcc.Graph(id="graph-timeline-comparison"),
    ]
)

# ----------------------------------------------------------------------
# Tab 3: Predicción de rendimiento
# ----------------------------------------------------------------------
tab_predict = html.Div(
    [
        html.Br(),
        dbc.Alert(
            "Ingresa las estadísticas hipotéticas de un jugador en un partido "
            "y el modelo estimará su calificación de rendimiento y la "
            "probabilidad de que participe en un gol (marcando o asistiendo).",
            color="info",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col([html.Label("Posición"), dcc.Dropdown(id="pred-position", options=POSITIONS, value="Forward")], md=3),
                            dbc.Col([html.Label("Pie preferido"), dcc.Dropdown(id="pred-foot", options=FEET, value=FEET[0])], md=3),
                            dbc.Col([html.Label("Continente"), dcc.Dropdown(id="pred-continent", options=CONTINENTS, value=CONTINENTS[0])], md=3),
                            dbc.Col([html.Label("Etapa"), dcc.Dropdown(id="pred-stage", options=STAGES, value=STAGES[0])], md=3),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col([html.Label("Edad"), dbc.Input(id="pred-age", type="number", value=25)], md=2),
                            dbc.Col([html.Label("Minutos jugados"), dbc.Input(id="pred-minutes", type="number", value=90)], md=2),
                            dbc.Col([html.Label("Tiros"), dbc.Input(id="pred-shots", type="number", value=3)], md=2),
                            dbc.Col([html.Label("Tiros al arco"), dbc.Input(id="pred-shots-target", type="number", value=1)], md=2),
                            dbc.Col([html.Label("xG"), dbc.Input(id="pred-xg", type="number", value=0.3, step=0.01)], md=2),
                            dbc.Col([html.Label("xA"), dbc.Input(id="pred-xa", type="number", value=0.2, step=0.01)], md=2),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col([html.Label("Pases clave"), dbc.Input(id="pred-keypasses", type="number", value=2)], md=2),
                            dbc.Col([html.Label("% precisión pases"), dbc.Input(id="pred-passacc", type="number", value=0.85, step=0.01)], md=2),
                            dbc.Col([html.Label("Regates exitosos"), dbc.Input(id="pred-dribbles", type="number", value=2)], md=2),
                            dbc.Col([html.Label("Distancia (km)"), dbc.Input(id="pred-distance", type="number", value=10.5, step=0.1)], md=2),
                            dbc.Col([html.Label("Velocidad máx (km/h)"), dbc.Input(id="pred-topspeed", type="number", value=31.0, step=0.1)], md=2),
                            dbc.Col([html.Label("Valor de mercado (€)"), dbc.Input(id="pred-marketvalue", type="number", value=50_000_000)], md=2),
                        ],
                        className="mb-3",
                    ),
                    dbc.Button("Predecir rendimiento", id="btn-predict", color="primary", className="mt-2"),
                ]
            ),
            className="shadow-sm mb-4",
        ),
        html.Div(id="prediction-output"),
    ]
)

app.layout = dbc.Container(
    [
        html.Br(),
        html.H2("⚽ FIFA World Cup 2026 — Player Performance Analytics", className="mb-1"),
        html.P(
            "Proyecto integrador de Ciencia de Datos: ETL + Machine Learning + Dashboard interactivo.",
            className="text-muted",
        ),
        html.Hr(),
        filters,
        dbc.Tabs(
            [
                dbc.Tab(tab_overview, label="📊 Resumen General"),
                dbc.Tab(tab_compare, label="🆚 Comparador de Jugadores"),
                dbc.Tab(tab_predict, label="🤖 Predicción de Rendimiento"),
            ]
        ),
        html.Br(),
        html.Footer(
            "Proyecto Programación para la Ciencia de Datos — Evaluación Final Transversal",
            className="text-center text-muted mb-3",
        ),
    ],
    fluid=True,
)


# ----------------------------------------------------------------------
# Utilidad: aplica filtros globales
# ----------------------------------------------------------------------
def apply_filters(stage, position, continent):
    dff = df.copy()
    if stage:
        dff = dff[dff["tournament_stage"].isin(stage)]
    if position:
        dff = dff[dff["position"].isin(position)]
    if continent:
        dff = dff[dff["continent"].isin(continent)]
    return dff


# ----------------------------------------------------------------------
# Callbacks: Tab 1 - Resumen general
# ----------------------------------------------------------------------
@app.callback(
    Output("kpi-row", "children"),
    Output("graph-goals-by-team", "figure"),
    Output("graph-top-scorers", "figure"),
    Output("graph-xg-vs-goals", "figure"),
    Output("graph-position-distribution", "figure"),
    Input("filter-stage", "value"),
    Input("filter-position", "value"),
    Input("filter-continent", "value"),
)
def update_overview(stage, position, continent):
    dff = apply_filters(stage, position, continent)

    kpis = [
        dbc.Col(kpi_card("Goles totales", int(dff["goals"].sum())), md=3),
        dbc.Col(kpi_card("Asistencias totales", int(dff["assists"].sum())), md=3),
        dbc.Col(kpi_card("Jugadores únicos", dff["player_id"].nunique()), md=3),
        dbc.Col(kpi_card("Partidos registrados", dff["match_id"].nunique()), md=3),
    ]

    goals_by_team = (
        dff.groupby("team", as_index=False)["goals"]
        .sum()
        .sort_values("goals", ascending=False)
        .head(15)
    )
    fig_team = px.bar(
        goals_by_team, x="team", y="goals", title="Top 15 selecciones por goles",
        color="goals", color_continuous_scale="Blues",
    )
    fig_team.update_layout(xaxis_title="", yaxis_title="Goles")

    top_scorers = (
        dff.groupby("player_name", as_index=False)
        .agg(goles=("goals", "sum"), asistencias=("assists", "sum"))
    )
    top_scorers["contribucion"] = top_scorers["goles"] + top_scorers["asistencias"]
    top_scorers = top_scorers.sort_values("contribucion", ascending=False).head(10)
    fig_scorers = px.bar(
        top_scorers, x="contribucion", y="player_name", orientation="h",
        title="Top 10 jugadores por contribución de gol (G+A)",
        color="contribucion", color_continuous_scale="Oranges",
    )
    fig_scorers.update_layout(yaxis=dict(categoryorder="total ascending"), xaxis_title="Goles + Asistencias", yaxis_title="")

    fig_xg = px.scatter(
        dff[dff["minutes_played"] > 0], x="expected_goals_xg", y="goals",
        color="position", opacity=0.5,
        title="Goles reales vs. xG esperado (por posición)",
        hover_data=["player_name", "team"],
    )

    pos_dist = dff["position"].value_counts().reset_index()
    pos_dist.columns = ["position", "count"]
    fig_pos = px.pie(pos_dist, names="position", values="count", title="Distribución de registros por posición", hole=0.4)

    return kpis, fig_team, fig_scorers, fig_xg, fig_pos


# ----------------------------------------------------------------------
# Callbacks: Tab 2 - Comparador de jugadores
# ----------------------------------------------------------------------
RADAR_METRICS = [
    "offensive_contribution", "defensive_contribution", "possession_impact",
    "pressure_resistance", "creativity_score", "consistency_score",
]


@app.callback(
    Output("graph-radar-comparison", "figure"),
    Output("graph-timeline-comparison", "figure"),
    Input("player-a", "value"),
    Input("player-b", "value"),
)
def update_comparison(player_a, player_b):
    fig_radar = px.line_polar(title="Perfil de rendimiento (promedio del torneo)")
    fig_timeline = px.line(title="Evolución de calificación (player_rating) por partido")

    players = [p for p in [player_a, player_b] if p]
    if not players:
        return fig_radar, fig_timeline

    radar_rows = []
    for p in players:
        sub = df[df["player_name"] == p]
        row = {"player": p}
        for m in RADAR_METRICS:
            row[m] = sub[m].mean()
        radar_rows.append(row)
    radar_df = pd.DataFrame(radar_rows).melt(id_vars="player", var_name="metric", value_name="value")
    fig_radar = px.line_polar(
        radar_df, r="value", theta="metric", color="player", line_close=True,
        title="Perfil de rendimiento (promedio del torneo)",
    )
    fig_radar.update_traces(fill="toself", opacity=0.5)

    timeline_df = df[df["player_name"].isin(players)].sort_values("match_date")
    fig_timeline = px.line(
        timeline_df, x="match_date", y="player_rating", color="player_name",
        markers=True, title="Evolución de calificación (player_rating) por partido",
    )

    return fig_radar, fig_timeline


# ----------------------------------------------------------------------
# Callbacks: Tab 3 - Predicción
# ----------------------------------------------------------------------
@app.callback(
    Output("prediction-output", "children"),
    Input("btn-predict", "n_clicks"),
    State("pred-position", "value"),
    State("pred-foot", "value"),
    State("pred-continent", "value"),
    State("pred-stage", "value"),
    State("pred-age", "value"),
    State("pred-minutes", "value"),
    State("pred-shots", "value"),
    State("pred-shots-target", "value"),
    State("pred-xg", "value"),
    State("pred-xa", "value"),
    State("pred-keypasses", "value"),
    State("pred-passacc", "value"),
    State("pred-dribbles", "value"),
    State("pred-distance", "value"),
    State("pred-topspeed", "value"),
    State("pred-marketvalue", "value"),
    prevent_initial_call=True,
)
def predict_performance(
    n_clicks, position, foot, continent, stage, age, minutes, shots,
    shots_target, xg, xa, keypasses, passacc, dribbles, distance, topspeed,
    marketvalue,
):
    # Se completan con valores medios del dataset las features no capturadas
    # explícitamente en el formulario, para mantener la UI simple.
    base = df[NUMERIC_FEATURES].mean().to_dict()
    base.update(
        {
            "age": age, "minutes_played": minutes, "shots": shots,
            "shots_on_target": shots_target, "expected_goals_xg": xg,
            "expected_assists_xa": xa, "key_passes": keypasses,
            "pass_accuracy": passacc, "successful_dribbles": dribbles,
            "distance_covered_km": distance, "top_speed_kmh": topspeed,
            "market_value_eur": marketvalue,
        }
    )
    row = {**base, "position": position, "preferred_foot": foot, "continent": continent, "tournament_stage": stage}
    X = pd.DataFrame([row])

    proba_goal = clf_model.predict_proba(X)[0, 1]
    pred_rating = reg_model.predict(X)[0]

    return dbc.Row(
        [
            dbc.Col(kpi_card("Probabilidad de contribución de gol", f"{proba_goal * 100:.1f}%", "success"), md=6),
            dbc.Col(kpi_card("Calificación de rendimiento estimada", f"{pred_rating:.2f} / 10", "primary"), md=6),
        ],
        className="mt-3",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
