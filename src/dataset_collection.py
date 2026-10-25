"""Task 1: Dataset collection for points of interest (POI).

Uses the OpenStreetMap Overpass API to fetch real places (cafes, restaurants,
museums, parks, etc.) for a given bounding box. Works fully offline via a
bundled sample dataset when available (see `Settings.offline_mode`).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import Settings, ensure_dirs

# Nominatim geocoder: supports category search within a viewbox and is widely
# reachable. It is a thin layer over the OSM database.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Shortcut to the OSM editors' API used as a fallback when Overpass is blocked.
# Public OSM API allows small bbox downloads via /api/0.6/map?bbox=...
OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map"

# Default Overpass endpoints; the first is preferred, the rest are fallbacks.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# User-Agent: Overpass/OSM/Nominatim require a meaningful identifier to avoid
# rate-limiting.
USER_AGENT = "poi-recommendations-pipeline/0.1 (educational dataset collection)"
# Human-friendly delay between requests to stay polite to the API.
REQUEST_DELAY_SECONDS = 2.0
# Number of attempts per endpoint before trying the next one.
MAX_RETRIES = 2
# HTTP timeout in seconds for a single request.
HTTP_TIMEOUT_SECONDS = 120
# Grid step (degrees) used when scanning via the OSM API fallback.
OSM_API_GRID_STEP = 0.002
# Nominatim recommends a small delay between requests to respect its usage policy.
NOMINATIM_DELAY_SECONDS = 1.1

# Wikipedia REST API used to enrich short descriptions for found places.
WIKIPEDIA_API_URL = "https://ru.wikipedia.org/w/api.php"

# Category -> (OSM element type, OSM key, OSM value).
# Element type may be "node", "way" or "nwr" (nodes + ways + relations).
# Used both to build the Overpass query and to infer the category of each
# returned element from its tags.
OSM_CATEGORY_TAGS: dict[str, tuple[str, str, str]] = {
    "cafe": ("node", "amenity", "cafe"),
    "restaurant": ("node", "amenity", "restaurant"),
    "museum": ("nwr", "tourism", "museum"),
    "park": ("nwr", "leisure", "park"),
    "art_gallery": ("nwr", "tourism", "artwork"),
    "cinema": ("node", "amenity", "cinema"),
    "bar": ("node", "amenity", "bar"),
    "hotel": ("nwr", "tourism", "hotel"),
}

# Fields we retain from the OSM answer and the exact column names in the dataset.
RAW_COLUMNS: list[str] = [
    "id",
    "name",
    "category",
    "description",
    "city",
    "lat",
    "lon",
    "reviews",
    "url",
    "source",
]


def build_overpass_query(settings: Settings) -> str:
    """Build an Overpass QL query covering all POI categories in the bbox.

    Args:
        settings: pipeline settings containing bbox and per-category limits.

    Returns:
        A valid Overpass QL query string.
    """
    min_lat, min_lon, max_lat, max_lon = (float(v) for v in settings.osm_bounding_box.split(","))
    limit = settings.osm_pois_per_category
    bbox_clause = f"({min_lat:.5f},{min_lon:.5f},{max_lat:.5f},{max_lon:.5f})"

    # Each category contributes an out-block for its tags with a count limit.
    blocks: list[str] = []
    for category, (element_type, key, value) in OSM_CATEGORY_TAGS.items():
        selector = f'{element_type}["{key}"="{value}"]'
        blocks.append(
            f"(\n  {selector}{bbox_clause};\n) -> .cat_{category};\n"
            f".cat_{category} out tags center {limit};"
        )
    # Combine: fetch all category results, then output elements as JSON.
    return "[\n  out:json;\n  timeout:60;\n];\n" + "\n\n".join(blocks)


def _parse_element(
    element: dict[str, Any], category: str, bbox: tuple[float, float, float, float]
) -> dict[str, Any] | None:
    """Convert a single OSM element into a normalized POI record.

    Handles both nodes (lat/lon directly) and ways/relations (center geometry).

    Args:
        element: raw JSON element from the Overpass API.
        category: POI category this element belongs to.
        bbox: bounding box tuple, used to derive a city label.

    Returns:
        A dict with the RAW_COLUMNS keys, or None if the element has no name.
    """
    tags: dict[str, Any] = element.get("tags", {}) or {}
    name = tags.get("name") or tags.get("name:ru") or tags.get("name:en")
    if not name:
        # A place without a name is not useful for recommendations.
        return None

    center = element.get("center") or {}
    lat = element.get("lat") or center.get("lat")
    lon = element.get("lon") or center.get("lon")

    return {
        "id": element.get("id"),
        "name": name,
        "category": category,
        "description": tags.get("description") or tags.get("opening_hours") or "",
        "city": tags.get("addr:city") or _approx_city(bbox),
        "lat": lat,
        "lon": lon,
        "reviews": "",  # OSM does not expose reviews; filled later if available.
        "url": tags.get("website") or tags.get("url") or "",
        "source": "openstreetmap",
    }


def _approx_city(bbox: tuple[float, float, float, float]) -> str:
    """Return an approximate city name derived from a bounding box center."""
    center_lat = (bbox[0] + bbox[2]) / 2.0
    center_lon = (bbox[1] + bbox[3]) / 2.0
    # Rough geographic heuristic only used as a fallback label.
    if 55.5 <= center_lat <= 56.0 and 37.0 <= center_lon <= 38.0:
        return "Moscow"
    return "unknown"


def _post_with_retry(query: str) -> dict[str, Any]:
    """POST the Overpass query trying each endpoint with retries.

    Args:
        query: Overpass QL query string.

    Returns:
        Parsed JSON payload from a successful response.

    Raises:
        RuntimeError: if all endpoints/attempts fail.
    """
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    url,
                    data={"data": query},
                    timeout=HTTP_TIMEOUT_SECONDS,
                    headers={"User-Agent": USER_AGENT},
                )
                if response.status_code == 200:
                    return response.json()
                last_error = RuntimeError(
                    f"Overpass API error at {url}: "
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
            except requests.RequestException as exc:  # network/timeout errors
                last_error = exc
            print(f"  Retry {attempt}/{MAX_RETRIES} at {url} failed...")
            time.sleep(REQUEST_DELAY_SECONDS * attempt)
    raise RuntimeError(f"All Overpass endpoints failed: {last_error}")


def fetch_from_overpass(settings: Settings) -> list[dict[str, Any]]:
    """Query the Overpass API and return a list of normalized POI records.

    Falls back to the OSM API grid scan if all Overpass endpoints fail.

    Args:
        settings: pipeline settings.

    Returns:
        A list of POI dicts (RAW_COLUMNS keys) retrieved from OSM.
    """
    query = build_overpass_query(settings)
    try:
        payload = _post_with_retry(query)
    except RuntimeError:
        print("  Overpass unavailable; falling back to the OSM API grid scan...")
        return fetch_from_osm_api(settings)

    elements = payload.get("elements", [])
    records: list[dict[str, Any]] = []
    for element in elements:
        # Determine which category produced this element.
        category = _infer_category(element)
        if category is None:
            continue
        record = _parse_element(element, category, _bbox_tuple(settings))
        if record is not None:
            records.append(record)
    return records


def _bbox_tuple(settings: Settings) -> tuple[float, float, float, float]:
    """Return the bounding box as a tuple of floats."""
    return tuple(float(v) for v in settings.osm_bounding_box.split(","))  # type: ignore[return-value]


# Nominatim geocoder source (primary online source, widely reachable)
def _nominatim_to_record(
    item: dict[str, Any],
    category: str,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Convert a Nominatim search result into a normalized POI record.

    Args:
        item: a single Nominatim JSON result.
        category: the POI category this search was performed for.
        bbox: bounding box used to derive a city label.

    Returns:
        A dict with the RAW_COLUMNS keys.

    Raises:
        ValueError: if the result has no usable short name.
    """
    name = item.get("name")
    if not name:
        # Address-only results (no amenity name) are not useful.
        raise ValueError("result without a short name")
    display_name = item.get("display_name") or ""
    return {
        "id": item.get("osm_id"),
        "name": name,
        "category": category,
        "description": "",  # Nominatim does not provide descriptions.
        "city": _city_from_display_name(display_name) or _approx_city(bbox),
        "lat": item.get("lat"),
        "lon": item.get("lon"),
        "reviews": "",
        "url": "",
        "source": "nominatim",
    }


def _city_from_display_name(display_name: str) -> str:
    """Heuristically extract a city name from a Nominatim display_name."""
    for part in reversed(display_name.split(",")):
        part = part.strip()
        if part and not part[0].isdigit():
            return part
    return ""


def fetch_from_nominatim(settings: Settings) -> list[dict[str, Any]]:
    """Fetch POIs per category from the Nominatim geocoder within the viewbox.

    Nominatim is used as the primary online source because it is widely
    reachable and returns structured results with coordinates and category
    types directly.

    Args:
        settings: pipeline settings.

    Returns:
        A list of normalized POI records.
    """
    min_lat, min_lon, max_lat, max_lon = _bbox_tuple(settings)
    viewbox = f"{min_lon:.5f},{max_lat:.5f},{max_lon:.5f},{min_lat:.5f}"

    # Some categories need alternative queries in Nominatim to reach the target
    # count (e.g. museums are often tagged with tourism=museum).
    category_queries: dict[str, list[str]] = {
        "museum": ["музей", "museum"],
        "art_gallery": ["галерея", "art gallery"],
    }

    records: list[dict[str, Any]] = []
    for category, (_element_type, key, value) in OSM_CATEGORY_TAGS.items():
        queries = category_queries.get(category, [value])
        target_count = settings.osm_pois_per_category
        for query in queries:
            if sum(1 for r in records if r["category"] == category) >= target_count:
                break
            params = {
                "q": query,
                "format": "jsonv2",
                "limit": target_count,
                "viewbox": viewbox,
                "bounded": 1,
                "addressdetails": 0,
                "accept-language": "ru",
            }
            try:
                response = requests.get(
                    NOMINATIM_URL,
                    params=params,
                    timeout=HTTP_TIMEOUT_SECONDS,
                    headers={"User-Agent": USER_AGENT},
                )
                if response.status_code != 200:
                    print(f"  Nominatim HTTP {response.status_code} for {query}")
                    continue
                for item in response.json():
                    try:
                        record = _nominatim_to_record(item, category, _bbox_tuple(settings))
                    except ValueError:
                        continue
                    # De-duplicate by OSM id within the current collection.
                    if any(str(r["id"]) == str(record["id"]) for r in records):
                        continue
                    records.append(record)
            except requests.RequestException as exc:
                print(f"  Nominatim request failed for {query}: {exc}")
            # Respect Nominatim's fair-use policy.
            time.sleep(NOMINATIM_DELAY_SECONDS)
    return records


# OSM API fallback (works when Overpass endpoints are unreachable)
def _element_tags(element: ET.Element) -> dict[str, str]:
    """Collect all OSM tags of an XML element into a dict.

    Args:
        element: an OSM XML node/way element.

    Returns:
        A dict of tag key -> tag value.
    """
    tags: dict[str, str] = {}
    for tag in element.findall("tag"):
        key = tag.get("k")
        value = tag.get("v")
        if key and value:
            tags[key] = value
    return tags


def _element_geometry(
    element: ET.Element, nodes: dict[str, tuple[float, float]]
) -> tuple[float, float] | None:
    """Resolve geometry for an OSM XML element.

    Nodes carry lat/lon directly; ways use the centroid of their member nodes.

    Args:
        element: an OSM XML node or way element.
        nodes: lat/lon lookup for node ids found in the same download.

    Returns:
        (lat, lon) tuple or None if geometry cannot be resolved.
    """
    lat = element.get("lat")
    lon = element.get("lon")
    if lat is not None and lon is not None:
        return float(lat), float(lon)

    # Way: compute centroid from referenced node ids.
    lats: list[float] = []
    lons: list[float] = []
    for nd in element.findall("nd"):
        ref = nd.get("ref")
        if ref and ref in nodes:
            n_lat, n_lon = nodes[ref]
            lats.append(n_lat)
            lons.append(n_lon)
    if lats:
        return sum(lats) / len(lats), sum(lons) / len(lons)
    return None


def _osm_xml_to_records(
    root: ET.Element,
    bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    """Convert an OSM XML response into normalized POI records for our categories.

    Args:
        root: parsed <osm> XML root element.
        bbox: bounding box tuple used to derive a city label.

    Returns:
        A list of POI dicts (RAW_COLUMNS keys).
    """
    nodes: dict[str, tuple[float, float]] = {}
    elements: list[ET.Element] = []

    for child in root:
        if child.tag == "node":
            lat = child.get("lat")
            lon = child.get("lon")
            if lat and lon:
                nodes[child.get("id")] = (float(lat), float(lon))
            elements.append(child)
        elif child.tag == "way":
            elements.append(child)

    records: list[dict[str, Any]] = []
    for element in elements:
        tags = _element_tags(element)
        category = _infer_category_from_tags(tags)
        if category is None:
            continue
        geometry = _element_geometry(element, nodes)
        name = tags.get("name") or tags.get("name:ru") or tags.get("name:en")
        if not name:
            continue
        records.append(
            {
                "id": element.get("id"),
                "name": name,
                "category": category,
                "description": tags.get("description") or tags.get("opening_hours") or "",
                "city": tags.get("addr:city") or _approx_city(bbox),
                "lat": geometry[0] if geometry else None,
                "lon": geometry[1] if geometry else None,
                "reviews": "",
                "url": tags.get("website") or tags.get("url") or "",
                "source": "openstreetmap",
            }
        )
    return records


def _infer_category_from_tags(tags: dict[str, str]) -> str | None:
    """Map OSM tags to a POI category defined in OSM_CATEGORY_TAGS.

    Args:
        tags: OSM tag dict.

    Returns:
        The matching category key or None.
    """
    for category, (_element_type, key, expected_value) in OSM_CATEGORY_TAGS.items():
        if tags.get(key) == expected_value:
            return category
    return None


def fetch_from_osm_api(settings: Settings) -> list[dict[str, Any]]:
    """Scan a grid over the bounding box using the OSM API map endpoint.

    This is a fallback path that works when the Overpass endpoints are not
    reachable: it downloads small bbox chunks and filters them locally for the
    categories defined in OSM_CATEGORY_TAGS.

    Args:
        settings: pipeline settings.

    Returns:
        A list of normalized POI records.
    """
    min_lat, min_lon, max_lat, max_lon = _bbox_tuple(settings)
    records: list[dict[str, Any]] = []

    lat = min_lat
    while lat < max_lat and len(records) < settings.osm_pois_per_category * len(OSM_CATEGORY_TAGS):
        lon = min_lon
        next_lat = lat + OSM_API_GRID_STEP
        while lon < max_lon:
            next_lon = lon + OSM_API_GRID_STEP
            # Clamp to the target limit to avoid extremely wide requests.
            cell_lat2 = min(next_lat, max_lat)
            cell_lon2 = min(next_lon, max_lon)
            bbox_str = f"{lon:.6f},{lat:.6f},{cell_lon2:.6f},{cell_lat2:.6f}"
            response = requests.get(
                OSM_API_URL,
                params={"bbox": bbox_str},
                timeout=HTTP_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                cell_records = _osm_xml_to_records(root, _bbox_tuple(settings))
                records.extend(cell_records)
            # Respect the source server; it has no explicit rate limit, but be kind.
            time.sleep(0.5)
            lon = next_lon
        lat = next_lat

    # De-duplicate by id, keeping the first occurrence.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        rid = str(record["id"])
        if rid in seen:
            continue
        seen.add(rid)
        unique.append(record)
    return unique


def _infer_category(element: dict[str, Any]) -> str | None:
    """Infer the category of an OSM element from its tags.

    Args:
        element: raw OSM JSON element.

    Returns:
        The matching category key from OSM_CATEGORY_TAGS or None.
    """
    tags: dict[str, Any] = element.get("tags", {}) or {}
    for category, (_element_type, key, expected_value) in OSM_CATEGORY_TAGS.items():
        if tags.get(key) == expected_value:
            return category
    return None


def load_offline_dataset(settings: Settings) -> pd.DataFrame:
    """Load a previously collected dataset as a DataFrame.

    Used to make the pipeline reproducible without network access.

    Args:
        settings: pipeline settings with `offline_csv` path.

    Returns:
        DataFrame of collected POI records.
    """
    if not Path(settings.offline_csv).exists():
        raise FileNotFoundError(
            f"Offline dataset not found: {settings.offline_csv}. "
            "Run `make collect` once to generate it."
        )
    return pd.read_csv(settings.offline_csv)


def _fetch_wikipedia_descriptions(names: list[str], max_concurrent: int = 20) -> dict[str, str]:
    """Fetch short intro descriptions from Wikipedia for the given names.

    Only places with an exact (or close) article title get an extract;
    others are left with an empty description.

    Args:
        names: place names to look up.
        max_concurrent: number of articles per API call (Wikipedia batch limit).

    Returns:
        A dict mapping the original name -> short description (may be empty).
    """
    descriptions: dict[str, str] = {}
    for start in range(0, len(names), max_concurrent):
        batch = names[start : start + max_concurrent]
        try:
            response = requests.get(
                WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exintro": 1,
                    "redirects": 1,
                    "titles": "|".join(batch),
                },
                timeout=HTTP_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code != 200:
                continue
            pages = response.json().get("query", {}).get("pages", {})
            for page in pages.values():
                title = page.get("title")
                extract = (page.get("extract") or "").strip()
                if title and extract:
                    # Keep the first matching original name for this article.
                    for name in batch:
                        if name.lower() in title.lower() or title.lower() in name.lower():
                            descriptions.setdefault(name, extract[:300])
                            break
        except requests.RequestException:
            continue
        time.sleep(0.2)
    return descriptions


def save_records(records: list[dict[str, Any]], settings: Settings) -> pd.DataFrame:
    """Persist collected records to CSV and return them as a DataFrame.

    Descriptions are enriched from Wikipedia when available (short intro),
    falling back to a category-based placeholder.

    Args:
        records: normalized list of POI records.
        settings: pipeline settings.

    Returns:
        DataFrame of all collected POI records.
    """
    ensure_dirs(settings)
    names = sorted({r["name"] for r in records})
    wiki_descriptions = _fetch_wikipedia_descriptions(names)

    enriched: list[dict[str, Any]] = []
    for record in records:
        record = dict(record)
        if not record.get("description"):
            record["description"] = wiki_descriptions.get(
                record["name"], _category_placeholder(record["category"])
            )
        enriched.append(record)

    df = pd.DataFrame(enriched, columns=RAW_COLUMNS)
    out_path = settings.data_dir / "poi_dataset_raw.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def _category_placeholder(category: str) -> str:
    """Return a neutral description placeholder for a POI category."""
    placeholders = {
        "cafe": "Уютное место, где можно выпить кофе и перекусить в течение дня.",
        "restaurant": "Ресторан с разнообразным меню, подходящий для обеда или ужина.",
        "museum": "Экспозиция посвящена истории и культуре, интересна взрослым и детям.",
        "park": "Зелёная зона для прогулок и отдыха на свежем воздухе.",
        "art_gallery": "Пространство с выставками современного и классического искусства.",
        "cinema": "Кинотеатр с актуальным репертуаром и современными залами.",
        "bar": "Бар с широким выбором напитков, уместен для встречи с друзьями.",
        "hotel": "Гостиница с комфортными номерами для туристов и командировок.",
    }
    return placeholders.get(category, "")


def collect_dataset(settings: Settings | None = None) -> pd.DataFrame:
    """Collect POI dataset from OSM or load a cached/offline copy.

    Source priority: offline mode (if enabled) -> Nominatim -> Overpass ->
    OSM API grid scan. This keeps the pipeline working even when some public
    endpoints are unreachable.

    Args:
        settings: optional settings override; defaults to get_settings().

    Returns:
        DataFrame with the collected POI records.

    Raises:
        RuntimeError: if no records could be collected from any source.
    """
    settings = settings or Settings()
    if settings.offline_mode:
        return load_offline_dataset(settings)

    records: list[dict[str, Any]] = []
    errors: list[str] = []

    # 1. Nominatim geocoder (most reliable public source).
    try:
        records = fetch_from_nominatim(settings)
    except Exception as exc:  # noqa: BLE001 - any failure should fall through
        errors.append(f"Nominatim: {exc}")
    if not records:
        # 2. Overpass API (with OSM API grid as its own fallback).
        try:
            records = fetch_from_overpass(settings)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Overpass: {exc}")
    if not records:
        raise RuntimeError(
            "No POI records were returned by any online source. Errors: "
            + "; ".join(errors)
            + ". Check the bounding box or enable OFFLINE_MODE."
        )

    df = save_records(records, settings)
    _log_collection_summary(df, settings)
    return df


def _log_collection_summary(df: pd.DataFrame, settings: Settings) -> None:
    """Print a short human-readable summary of the collection step."""
    print(f"[Task 1] Collected {len(df)} POIs")
    if "category" in df:
        print(df["category"].value_counts().to_string())
    print(f"Saved to {settings.data_dir / 'poi_dataset_raw.csv'}")
