"""Tests for the dataset collection module (Task 1)."""

from __future__ import annotations

import pandas as pd
import pytest

import src.dataset_collection as dc
from src.config import Settings
from src.dataset_collection import (
    OSM_CATEGORY_TAGS,
    _approx_city,
    _cap_records_per_category,
    _category_placeholder,
    _city_from_display_name,
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

    # Write a tiny raw CSV that the offline loader should return.
    csv_path = tmp_path / "poi_dataset_raw.csv"
    pd.DataFrame([{"id": 1, "name": "X"}]).to_csv(csv_path, index=False)

    settings = Settings(offline_mode=True, offline_csv=str(csv_path))
    df = dc.load_offline_dataset(settings)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "X"


def test_city_extraction_skips_country() -> None:
    """City extraction should return the city, not the country/region."""
    display_name = (
        "Третьяковская галерея, Лаврушинский переулок, район Якиманка, "
        "Москва, Центральный федеральный округ, Россия"
    )
    assert _city_from_display_name(display_name) == "Москва"


def test_city_extraction_simple() -> None:
    """A short address should still yield the city."""
    assert _city_from_display_name("Cofix, улица, Москва") == "Москва"


def test_cap_records_per_category_enforces_limit() -> None:
    """The cap must trim each category to the limit while keeping order."""
    records = [
        {"category": "cafe", "id": 1},
        {"category": "cafe", "id": 2},
        {"category": "cafe", "id": 3},
        {"category": "museum", "id": 4},
        {"category": "museum", "id": 5},
    ]
    capped = _cap_records_per_category(records, limit=2)
    # cafe: first 2 kept (1, 2) - 3 dropped; museum: both kept (4, 5)
    assert [r["id"] for r in capped] == [1, 2, 4, 5]


def test_fetch_reviews_uses_wikivoyage(monkeypatch) -> None:
    """Reviews should be populated from a mocked Wikivoyage response."""

    class _FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "query": {
                    "pages": {
                        "1": {
                            "title": "Третьяковская галерея",
                            "extract": "Отличное место для знакомства с русской живописью.",
                        }
                    }
                }
            }

    def _fake_get(url, params=None, timeout=120, headers=None):
        assert "wikivoyage" in url
        return _FakeResponse()

    monkeypatch.setattr(dc.requests, "get", _fake_get)
    result = dc._fetch_reviews(["Третьяковская галерея", "NoSuchPlace"], category="museum")
    assert "Третьяковская галерея" in result
    assert "живописью" in result["Третьяковская галерея"]


def test_is_rejectable_place_text() -> None:
    """Extracts about bands/songs/films must be rejected as place descriptions."""
    assert dc._is_rejectable_place_text(
        "Breakfast Club — американская группа. Их самым большим хитом..."
    )
    assert dc._is_rejectable_place_text(
        "«Берёзка» — русская народная песня."
    )
    assert dc._is_rejectable_place_text(
        "Seasons — дебютный студийный альбом немецкой группы EverEve."
    )
    assert not dc._is_rejectable_place_text(
        "Третьяковская галерея — художественный музей в Москве."
    )
    assert not dc._is_rejectable_place_text(
        "Кусково — усадьба графов Шереметевых на востоке Москвы."
    )


def test_fetch_wikipedia_descriptions_skips_noise(monkeypatch) -> None:
    """Name colliding with a band/article must NOT fill the description."""
    class _FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "query": {
                    "pages": {
                        "1": {
                            "title": "Breakfast Club",
                            "extract": (
                                "Breakfast Club — американская группа. Их самым "
                                "большим хитом стала песня «Right on Track»..."
                            ),
                        }
                    }
                }
            }

    def _fake_get(url, params=None, timeout=120, headers=None):
        return _FakeResponse()

    monkeypatch.setattr(dc.requests, "get", _fake_get)
    result = dc._fetch_wikipedia_descriptions(["Breakfast Club"])
    assert "Breakfast Club" not in result


def test_is_rejectable_wikivoyage_title() -> None:
    """Non-place Wikivoyage pages (phrasebooks, heritage lists) must be skipped.

    Nested district pages ("Москва/...") are NOT rejected here: whether they
    describe the queried POI is decided contextually in `_fetch_reviews` by
    requiring the POI name to appear in the article title.
    """

    assert dc._is_rejectable_wikivoyage_title("Русский разговорник")
    assert dc._is_rejectable_wikivoyage_title("Культурное наследие России/Москва")
    assert not dc._is_rejectable_wikivoyage_title("Москва/Восток")  # contextual decision
    assert not dc._is_rejectable_wikivoyage_title("Третьяковская галерея")


def test_fetch_reviews_district_page_requires_name_in_title(monkeypatch) -> None:
    """A nested Moscow district page is used ONLY when the POI name is in it.

    "Москва/Кусково" describes the Kuskovo estate -> accepted for a POI named
    "Кусково". "Москва/Замоскворечье" merely mentions the gallery -> rejected
    for a POI named "Третьяковская галерея".
    """

    class _FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "query": {
                    "pages": {
                        "1": {
                            "title": "Москва/Кусково",
                            "extract": "Усадьба Кусково находится на востоке Москвы...",
                        },
                        "2": {
                            "title": "Москва/Замоскворечье",
                            "extract": "Добрынинская, Октябрьская...",
                        },
                    }
                }
            }

    def _fake_get(url, params=None, timeout=120, headers=None):
        return _FakeResponse()

    monkeypatch.setattr(dc.requests, "get", _fake_get)
    result = dc._fetch_reviews(["Кусково", "Третьяковская галерея"], category="museum")
    assert "Кусково" in result
    assert "усадьба" in result["Кусково"].lower()
    # The district page that merely mentions the gallery must NOT be used.
    assert "Третьяковская галерея" not in result


def test_fetch_reviews_skips_noisy_articles(monkeypatch) -> None:
    """Search hits about songs/bands/films must NOT pollute the reviews column.

    Regression for the "Breakfast Club" and "Берёзка" false positives that were
    captured before the strict matching + title filtering were added.
    """

    class _FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "query": {
                    "pages": {
                        "1": {
                            "title": "Breakfast Club — американская поп-группа",
                            "extract": "Группа, чей главный хит...",
                        },
                        "2": {
                            "title": "Берёзка (значения)",
                            "extract": "Список значений слова...",
                        },
                    }
                }
            }

    def _fake_get(url, params=None, timeout=120, headers=None):
        return _FakeResponse()

    monkeypatch.setattr(dc.requests, "get", _fake_get)
    result = dc._fetch_reviews(["Breakfast Club", "Берёзка"])
    # The noisy pages must not be treated as reviews for our POIs
    assert "Breakfast Club" not in result
    assert "Берёзка" not in result


def test_category_review_placeholder_not_empty() -> None:
    """Synthetic impressions must be present for ALL dataset categories."""

    for category in (
        "restaurant",
        "bar",
        "cafe",
        "cinema",
        "museum",
        "art_gallery",
        "park",
        "hotel",
    ):
        text = dc._category_review_placeholder(category, "Тестовое место")
        assert text.strip()  # non-empty, category-appropriate impression


def test_fetch_reviews_commercial_ignores_unrelated_district(monkeypatch) -> None:
    """A cinema named 'Свобода' must NOT get a village article from Kursk region.

    Regression: a direct-match article about a toponym in another region
    (не место в Москве) used to leak into the reviews column for commercial
    categories. Such places fall back to synthetic reviews instead.
    """

    class _FakeResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "query": {
                    "pages": {
                        "1": {
                            "title": "Свобода",
                            "extract": (
                                "Свобода — местечко в Золотухинском районе "
                                "Курской области. Здесь расположена Коренная "
                                "пустынь..."
                            ),
                        }
                    }
                }
            }

    def _fake_get(url, params=None, timeout=120, headers=None):
        return _FakeResponse()

    monkeypatch.setattr(dc.requests, "get", _fake_get)
    result = dc._fetch_reviews(["Свобода"], category="cinema")
    assert "Свобода" not in result


def test_save_records_synthetic_review_source(tmp_path, monkeypatch) -> None:
    """Places without a Wikivoyage article get a marked synthetic review."""

    # No network: make both enrichment calls no-ops.
    monkeypatch.setattr(dc, "_fetch_wikipedia_descriptions", lambda names: {})
    monkeypatch.setattr(dc, "_fetch_reviews", lambda names, category="": {})

    records = [
        {
            "id": 1,
            "name": "Кофейня №1",
            "category": "cafe",
            "description": "",
            "city": "Москва",
            "lat": 55.7,
            "lon": 37.6,
            "reviews": "",
            "review_source": "",
            "url": "",
            "source": "nominatim",
        }
    ]
    settings = Settings(data_dir=tmp_path, outputs_dir=tmp_path / "outputs")
    df = dc.save_records(records, settings)

    assert df["review_source"].iloc[0] == dc.REVIEW_SOURCE_CATEGORY
    assert str(df["reviews"].iloc[0]).strip()  # synthetic impression present
    # The raw CSV must contain the new provenance column.
    reloaded = pd.read_csv(settings.data_dir / "poi_dataset_raw.csv")
    assert "review_source" in reloaded.columns
