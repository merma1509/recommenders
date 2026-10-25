"""Tests for the dataset collection module (Task 1)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import Settings
from src.dataset_collection import (
    OSM_CATEGORY_TAGS,
    _approx_city,
    _category_placeholder,
    _infer_category_from_tags,
    _nominatim_to_record,
    build_overpass_query,
)


def test_category_mapping_has_expected_keys() -> None:
    """All required POI categories should be present in the tag mapping."""
    expected = {
        "cafe",
        "restaurant",
        "museum",
        "park",
        "art_gallery",
        "cinema",
        "bar",
        "hotel",
    }
    assert set(OSM_CATEGORY_TAGS).issuperset(expected)


def test_build_overpass_query_contains_categories() -> None:
    """The generated query should mention every configured category."""
    settings = Settings(osm_bounding_box="55.55,37.35,55.95,37.95")
    query = build_overpass_query(settings)
    assert "amenity" in query
    assert "tourism" in query
    for category in OSM_CATEGORY_TAGS:
        assert f".cat_{category}" in query


def test_infer_category_from_tags() -> None:
    """OSM tags should map to the correct category."""
    assert _infer_category_from_tags({"amenity": "cafe"}) == "cafe"
    assert _infer_category_from_tags({"tourism": "museum"}) == "museum"
    assert _infer_category_from_tags({"leisure": "park"}) == "park"
    assert _infer_category_from_tags({"amenity": "pharmacy"}) is None


def test_nominatim_record_normalization() -> None:
    """A Nominatim result with a short name should normalize correctly."""
    item = {
        "osm_id": 123,
        "name": "Cofix",
        "display_name": "Cofix, улица, Москва",
        "lat": "55.7",
        "lon": "37.6",
    }
    record = _nominatim_to_record(item, "cafe", (55.5, 37.3, 55.9, 37.9))
    assert record["id"] == 123
    assert record["category"] == "cafe"
    assert record["name"] == "Cofix"
    assert record["lat"] == "55.7"
    assert record["lon"] == "37.6"


def test_nominatim_record_rejects_address_only() -> None:
    """A result without a short name (address only) should be rejected."""
    item = {"osm_id": 1, "display_name": "улица, Москва", "lat": "55", "lon": "37"}
    with pytest.raises(ValueError):
        _nominatim_to_record(item, "cafe", (55.5, 37.3, 55.9, 37.9))


def test_approx_city_moscow() -> None:
    """The Moscow bbox should produce the 'Moscow' fallback label."""
    assert _approx_city((55.7, 37.5, 55.8, 37.7)) == "Moscow"
    assert _approx_city((10.0, 20.0, 11.0, 21.0)) == "unknown"


def test_category_placeholder_not_empty() -> None:
    """Every category should have a non-empty description placeholder."""
    for category in OSM_CATEGORY_TAGS:
        assert _category_placeholder(category)


def test_collect_offline_roundtrip(tmp_path, monkeypatch) -> None:
    """Collecting in offline mode should load the pre-generated CSV."""
    import src.dataset_collection as dc

    # Write a tiny raw CSV that the offline loader should return.
    csv_path = tmp_path / "poi_dataset_raw.csv"
    pd.DataFrame([{"id": 1, "name": "X"}]).to_csv(csv_path, index=False)

    settings = Settings(offline_mode=True, offline_csv=str(csv_path))
    df = dc.load_offline_dataset(settings)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "X"
