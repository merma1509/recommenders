"""Tests for the semantic feature extraction module (Tasks 2-3)."""

from __future__ import annotations

import pandas as pd

from src.features import FeatureExtractor, add_features


def _extract_for(row: pd.Series) -> dict:
    return FeatureExtractor().extract(row)


def test_feature_columns_present(featured_poi_df: pd.DataFrame) -> None:
    """All requested semantic columns should exist after extraction."""
    df = add_features(featured_poi_df)
    expected = {
        "experience_type",
        "atmosphere",
        "visit_context",
        "budget",
        "visit_time",
        "indoor_outdoor",
        "popularity_bucket",
    }
    assert expected.issubset(df.columns)


def test_experience_type_from_category(raw_poi_df: pd.DataFrame) -> None:
    """Museum category should map to culture experience type."""
    features = _extract_for(raw_poi_df.iloc[1])
    assert features["experience_type"] == "culture"


def test_experience_type_cafe(raw_poi_df: pd.DataFrame) -> None:
    """Cafe category should map to food experience type."""
    features = _extract_for(raw_poi_df.iloc[0])
    assert features["experience_type"] == "food"


def test_cyrillic_atmosphere_detected(raw_poi_df: pd.DataFrame) -> None:
    """A Russian 'cozy' keyword should be detected in text."""
    features = _extract_for(raw_poi_df.iloc[0])
    assert features["atmosphere"] == "cozy"


def test_visit_context_family(raw_poi_df: pd.DataFrame) -> None:
    """A review mentioning family should set the family visit context."""
    features = _extract_for(raw_poi_df.iloc[1])
    assert features["visit_context"] == "family"


def test_budget_low_for_cafe(raw_poi_df: pd.DataFrame) -> None:
    """Cafe category should fall back to a low budget when no hints exist."""
    features = _extract_for(raw_poi_df.iloc[0])
    assert features["budget"] == "low"


def test_atmosphere_category_prior_bar() -> None:
    """A bar without atmosphere hints should be lively by category prior."""
    row = pd.Series({"name": "Bar X", "category": "bar", "description": "", "reviews": ""})
    features = _extract_for(row)
    assert features["atmosphere"] == "lively"


def test_visit_time_category_prior_bar() -> None:
    """A bar should default to evening as the recommended visit time."""
    row = pd.Series({"name": "Bar X", "category": "bar", "description": "", "reviews": ""})
    features = _extract_for(row)
    assert features["visit_time"] == "evening"


def test_atmosphere_museum_quiet() -> None:
    """A museum should default to a quiet atmosphere."""
    row = pd.Series(
        {
            "name": "Museum",
            "category": "museum",
            "description": "",
            "reviews": "",
        }
    )
    features = _extract_for(row)
    assert features["atmosphere"] == "quiet"


def test_budget_high_detected() -> None:
    """A premium hint should map to a high budget level."""
    row = pd.Series(
        {
            "name": "Premium",
            "category": "restaurant",
            "description": "Премиальный ресторан",
            "reviews": "",
        }
    )
    features = _extract_for(row)
    assert features["budget"] == "high"
