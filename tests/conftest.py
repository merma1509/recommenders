"""Pytest fixtures shared across tests: small in-memory POI DataFrames."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def raw_poi_df() -> pd.DataFrame:
    """A small raw POI DataFrame mimicking data/poi_dataset_raw.csv columns."""
    return pd.DataFrame(
        [
            {
                "id": 1001,
                "name": "Surf Coffee",
                "category": "cafe",
                "description": "Уютная кофейня с wifi, завтраки весь день.",
                "city": "Moscow",
                "lat": 55.7558,
                "lon": 37.6173,
                "reviews": "Отличный кофе для работы.",
                "url": "https://example.com/surf",
                "source": "openstreetmap",
            },
            {
                "id": 1005,
                "name": "Третьяковская галерея",
                "category": "museum",
                "description": "Художественный музей с коллекцией живописи.",
                "city": "Moscow",
                "lat": 55.7415,
                "lon": 37.6208,
                "reviews": "Культурный день для всей семьи.",
                "url": "https://example.com/tretyakov",
                "source": "openstreetmap",
            },
        ]
    )


@pytest.fixture
def featured_poi_df(raw_poi_df: pd.DataFrame) -> pd.DataFrame:
    """A DataFrame with semantic features already applied."""
    df = raw_poi_df.copy()
    df["experience_type"] = ["food", "culture"]
    df["atmosphere"] = ["work", "family"]
    df["visit_context"] = ["work", "family"]
    df["budget"] = ["medium", "medium"]
    df["visit_time"] = ["morning", "day"]
    df["indoor_outdoor"] = ["indoor", "indoor"]
    df["popularity_bucket"] = ["medium", "medium"]
    return df
