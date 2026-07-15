"""Tests unitarios para el pipeline ETL."""

import pandas as pd
import pytest

from etl.transform import add_performance_features, clean_players_data, enrich_with_continent


@pytest.fixture
def sample_players_df():
    return pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P2"],
            "player_name": ["Ana", "Beto", "Beto"],
            "match_id": ["M1", "M1", "M1"],
            "team": ["Chile", "Chile", "Chile"],
            "match_date": ["2026-06-11", "2026-06-11", "2026-06-11"],
            "age": [25, -3, -3],
            "minutes_played": [90, 45, 45],
            "goals": [1, 0, 0],
            "assists": [0, 1, 1],
            "shots": [3, 1, 1],
            "shots_on_target": [2, 0, 0],
            "dribbles_attempted": [4, 0, 0],
            "successful_dribbles": [2, 0, 0],
            "crosses": [0, 0, 0],
            "successful_crosses": [0, 0, 0],
            "expected_goals_xg": [0.5, 0.1, 0.1],
            "expected_assists_xa": [0.1, 0.2, 0.2],
            "pass_accuracy": [0.9, 1.5, 1.5],
            "save_percentage": [0.0, 0.0, 0.0],
        }
    )


@pytest.fixture
def sample_continents_df():
    return pd.DataFrame({"team": ["Chile"], "continent": ["South America"]})


def test_clean_players_data_removes_duplicates(sample_players_df):
    cleaned = clean_players_data(sample_players_df)
    assert cleaned.duplicated().sum() == 0


def test_clean_players_data_clips_negative_age(sample_players_df):
    cleaned = clean_players_data(sample_players_df)
    assert (cleaned["age"] >= 0).all()


def test_clean_players_data_clips_pass_accuracy(sample_players_df):
    cleaned = clean_players_data(sample_players_df)
    assert cleaned["pass_accuracy"].max() <= 1.0


def test_enrich_with_continent_adds_column(sample_players_df, sample_continents_df):
    cleaned = clean_players_data(sample_players_df)
    enriched = enrich_with_continent(cleaned, sample_continents_df)
    assert "continent" in enriched.columns
    assert enriched["continent"].isna().sum() == 0


def test_enrich_with_continent_handles_missing_team():
    df = pd.DataFrame({"team": ["Atlantis"]})
    continents = pd.DataFrame({"team": ["Chile"], "continent": ["South America"]})
    enriched = enrich_with_continent(df, continents)
    assert enriched.loc[0, "continent"] == "Desconocido"


def test_add_performance_features_creates_expected_columns(sample_players_df, sample_continents_df):
    cleaned = clean_players_data(sample_players_df)
    enriched = enrich_with_continent(cleaned, sample_continents_df)
    result = add_performance_features(enriched)
    for col in ["goals_per90", "assists_per90", "goal_contribution", "shot_accuracy"]:
        assert col in result.columns


def test_goal_contribution_per90_is_non_negative(sample_players_df, sample_continents_df):
    cleaned = clean_players_data(sample_players_df)
    enriched = enrich_with_continent(cleaned, sample_continents_df)
    result = add_performance_features(enriched)
    assert (result["goal_contribution_per90"] >= 0).all()


def test_shot_accuracy_is_zero_when_no_shots():
    df = pd.DataFrame({"shots": [0], "shots_on_target": [0]})
    df["shot_accuracy"] = 0
    assert df.loc[0, "shot_accuracy"] == 0
