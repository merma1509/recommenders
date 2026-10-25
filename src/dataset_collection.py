"""Task 1: Dataset collection for points of interest (POI)

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
# reachable. It is a thin layer over the OSM database
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Shortcut to the OSM editors' API used as a fallback when Overpass is blocked.
# Public OSM API allows small bbox downloads via /api/0.6/map?bbox=...
OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map"

# Default Overpass endpoints; the first is preferred, the rest are fallbacks
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# User-Agent: Overpass/OSM/Nominatim require a meaningful identifier to avoid
# rate-limiting
USER_AGENT = "poi-recommendations-pipeline/0.1 (educational dataset collection)"
# Human-friendly delay between requests to stay polite to the API
REQUEST_DELAY_SECONDS = 2.0

# Number of attempts per endpoint before trying the next one
MAX_RETRIES = 2

# HTTP timeout in seconds for a single request
HTTP_TIMEOUT_SECONDS = 120

# Grid step (degrees) used when scanning via the OSM API fallback
OSM_API_GRID_STEP = 0.002

# Nominatim recommends a small delay between requests to respect its usage policy
NOMINATIM_DELAY_SECONDS = 1.1

# Wikipedia REST API used to enrich short descriptions for found places
WIKIPEDIA_API_URL = "https://ru.wikipedia.org/w/api.php"

# Wikivoyage (open travel guide) is used as a free, key-less source of
# impression-like texts that we store in the `reviews` column
WIKIVOYAGE_API_URL = "https://ru.wikivoyage.org/w/api.php"

# Hard cap per category regardless of the source (matches Task 1 limits)
MAX_POIS_PER_CATEGORY_HARD = 12

# Provenance values for the `reviews` column. Being explicit about WHERE a
# review came from keeps the dataset honest (no fake user ratings):
#   - "wikivoyage": a real travel-guide text matched to this place (or district).
#   - "category": a synthesized impression (no free source exposes user reviews
#     for single cafes/bars/cinemas). Marked clearly as synthetic.
#   - "": no review available.
REVIEW_SOURCE_WIKIVOYAGE = "wikivoyage"
REVIEW_SOURCE_CATEGORY = "category"

# Categories for which public key-less APIs do NOT expose per-place reviews.
# Wikivoyage has no articles for individual restaurants/bars/cafes/cinemas
# (and only sporadic coverage of museums/galleries/parks), so we fill `reviews`
# with a clearly-marked synthetic impression everywhere a real source is absent
SYNTHETIC_REVIEW_CATEGORIES = {
    "restaurant",
    "bar",
    "cafe",
    "cinema",
    "museum",
    "art_gallery",
    "park",
    "hotel",
}

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
# `review_source` documents the provenance of the `reviews` column (see
# REVIEW_SOURCE_* constants) so synthetic entries are never mistaken for real
# user reviews.
RAW_COLUMNS: list[str] = [
    "id",
    "name",
    "category",
    "description",
    "city",
    "lat",
    "lon",
    "reviews",
    "review_source",
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
        "review_source": "",  # filled in save_records from the enrichment step.
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
        "review_source": "",  # filled in save_records from the enrichment step.
        "url": "",
        "source": "nominatim",
    }


def _city_from_display_name(display_name: str) -> str:
    """Extract a city name from a Nominatim display_name (not the country).

    Nominatim display_name is comma-separated, e.g.
    "Третьяковская галерея, Лаврушинский переулок, ..., район Якиманка,
    Москва, Центральный федеральный округ, Россия".
    The city is the part before the country and the federal district; we scan
    from the right and skip country/region-like suffixes.
    """
    parts = [p.strip() for p in display_name.split(",") if p.strip()]
    # Known suffixes that are broader than a city (country/region).
    skip = {"россия", "рф", "московская область", "центральный федеральный округ"}
    for part in reversed(parts):
        low = part.lower()
        if low in skip:
            continue
        if part and not part[0].isdigit():
            return part
    return ""


def category_count(records: list[dict[str, Any]], category: str) -> int:
    """Return how many records of a given category were already collected."""
    return sum(1 for r in records if r["category"] == category)


def _cap_records_per_category(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Trim records so each category has at most `limit` entries.

    Preserves the original order (first N per category are kept) and guarantees
    the Task 1 requirement even when a fallback source returns more items.

    Args:
        records: list of POI records.
        limit: maximum number of records per category.

    Returns:
        A new list with at most `limit` records per category.
    """
    counts: dict[str, int] = {}
    capped: list[dict[str, Any]] = []
    for record in records:
        category = record["category"]
        if counts.get(category, 0) >= limit:
            continue
        counts[category] = counts.get(category, 0) + 1
        capped.append(record)
    return capped


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
            # Cumulative per-category limit is enforced across all queries for
            # the same category (not per single query).
            if category_count(records, category) >= target_count:
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
                "review_source": "",  # filled in save_records from the enrichment step.
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


def _is_rejectable_place_text(text: str) -> bool:
    """Return True if the Wikipedia/Wikivoyage extract is clearly NOT about a
    place but about a song, band, album, film or a person.

    Guards against name collisions: a cafe called "Breakfast Club" can match a
    Wikipedia article about the 1980s band of the same name. Such extracts must
    not pollute `description` or `reviews`.

    Args:
        text: the extract text to inspect.

    Returns:
        True when the text describes a non-place subject.
    """
    lowered = " ".join(text.lower().split())
    non_place_markers = (
        "— американская группа",
        "- группа",
        "— группа",
        "российская группа",
        "немецкая группа",
        "некоторых группа",
        "песня",
        "одна из песен",
        "сингл",
        "альбом",
        "студийный альбом",
        "фильм",
        "телесериал",
        "кинофильм",
        "мультфильм",
        "писатель",
        "русский писатель",
        "поэт",
        "композитор",
        "актриса",
        "актёр",
        "певица",
        "певец",
        "режиссёр",
        "кинорежиссёр",
        "расцветки",
        "хореографический ансамбль",
        "телесериал об",
        "государственный академический",
    )
    return any(marker in lowered for marker in non_place_markers)


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
                    # Skip extracts that describe a band/song/film/person rather
                    # than a place (name-collision guard, e.g. "Breakfast Club").
                    if _is_rejectable_place_text(extract):
                        continue
                    # Keep the first matching original name for this article.
                    for name in batch:
                        if name.lower() in title.lower() or title.lower() in name.lower():
                            descriptions.setdefault(name, extract[:300])
                            break
        except requests.RequestException:
            continue
        time.sleep(0.2)
    return descriptions


def _is_rejectable_wikivoyage_title(title: str) -> bool:
    """Return True for Wikivoyage article titles that are clearly NOT a place
    description (disambiguation, phrasebook, non-travel list pages, etc.).

    Guards against the noisy results seen in practice: searching for a cafe
    name can return articles about songs, films, bands or phrasebook lines.
    """
    lowered = title.lower()
    reject_markers = (
        "разговорник",  # phrasebook lines
        "культурное наследие",  # heritage listings, not a place impression
        "род. ",  # disambiguation with people
        "значения",  # "X (значения)" disambiguation
    )
    # Markers of non-place articles (songs, bands, films, people) that
    # frequently appear as false positives when a POI name collides with them.
    non_place_markers = (
        " группа",
        "-группа",
        "песн",
        "фильм",
        "альбом",
        "писатель",
        "поэт",
        "композитор",
        "актёр",
        "актер",
        "певиц",
        "режиссёр",
        "режиссер",
        "телесериал",
        "расцветки",
    )
    return any(marker in lowered for marker in reject_markers) or any(
        marker in lowered for marker in non_place_markers
    )


def _fetch_reviews(
    names: list[str], max_concurrent: int = 5, category: str = ""
) -> dict[str, str]:
    """Fetch impression-like texts for places from Wikivoyage.

    Wikivoyage is an open travel guide. Strategy:
      1. Direct article lookup by exact/fuzzy name.
      2. For cultural/heritage categories a nested district page like
         "Москва/Кусково" is accepted **only if** the POI name itself appears
         in the article title (proves the article is about OUR place, not just
         a passing mention in a larger district page).

    This is deliberately conservative: we prefer a small number of CORRECT
    impressions over a large number of wrong ones (songs, bands, films).

    Args:
        names: place names to look up.
        max_concurrent: number of article titles per API call.
        category: POI category of the queried places (used to relax the
            district-page rule for cultural venues).

    Returns:
        A dict mapping the original name -> short text (may be empty).
    """
    # District pages ("Москва/...") are only trusted for cultural/heritage
    # places. Commercial categories (cinemas, bars, restaurants, cafes) get
    # synthetic reviews instead: direct article lookups here collide too often
    # with unrelated toponyms (e.g. "Свобода" matches a village in Kursk).
    cultural_categories = {"museum", "art_gallery", "park", "hotel"}
    reviews: dict[str, str] = {}
    if category and category not in cultural_categories:
        # For commercial venues we never trust Wikivoyage lookups: short
        # generic names (Свобода, Берёзка, ...) collide with non-place pages,
        # and there are no real per-venue travel texts anyway. Synthetic
        # reviews are filled by the caller
        return reviews
    dedup_names = sorted({n for n in names if n})
    for start in range(0, len(dedup_names), max_concurrent):
        batch = dedup_names[start : start + max_concurrent]
        try:
            response = requests.get(
                WIKIVOYAGE_API_URL,
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
                title = page.get("title") or ""
                extract = (page.get("extract") or "").strip()
                if not title or not extract or _is_rejectable_wikivoyage_title(title):
                    continue
                for name in batch:
                    name_l = name.lower()
                    title_l = title.lower()
                    # Strict containment (case-insensitive) to avoid capturing
                    # unrelated pages (e.g. "Breakfast Club" the band).
                    direct_match = name_l in title_l or title_l in name_l
                    # District articles are trusted only for cultural places
                    # AND only when the exact place name appears in the title;
                    # otherwise we'd pull unrelated toponyms from other regions.
                    district_match = (
                        category in cultural_categories
                        and "москва/" in title_l
                        and name_l in title_l
                    )
                    if direct_match or district_match:
                        reviews.setdefault(name, extract[:500])
                        break
        except requests.RequestException:
            continue
        time.sleep(0.3)
    return reviews


def save_records(records: list[dict[str, Any]], settings: Settings) -> pd.DataFrame:
    """Persist collected records to CSV and return them as a DataFrame.

    Descriptions are enriched from Wikipedia when available (short intro),
    falling back to a category-based placeholder. The `reviews` column is
    filled from Wikivoyage impression-like texts when an article matches.

    Args:
        records: normalized list of POI records.
        settings: pipeline settings.

    Returns:
        DataFrame of all collected POI records.
    """
    ensure_dirs(settings)
    names = sorted({r["name"] for r in records})
    wiki_descriptions = _fetch_wikipedia_descriptions(names)

    # Populate the `reviews` column with a clearly-marked provenance:
    #   1. Wikivoyage article matched to the place (real travel text);
    #   2. synthetic category-level impression for commercial venues, marked
    #      as `category` so nobody mistakes it for a user review.
    # Wikivoyage lookup is split by category so cultural places (museums,
    # galleries, parks) may use district articles, while commercial venues do
    # not (they have no articles anyway).
    name_to_category = {r["name"]: r["category"] for r in records}
    wiki_reviews: dict[str, str] = {}
    for cat in sorted({name_to_category[n] for n in names}):
        cat_names = [n for n in names if name_to_category[n] == cat]
        cat_reviews = _fetch_reviews(cat_names, category=cat)
        wiki_reviews.update(cat_reviews)

    enriched: list[dict[str, Any]] = []
    for record in records:
        record = dict(record)
        if not record.get("description"):
            record["description"] = wiki_descriptions.get(
                record["name"], _category_placeholder(record["category"])
            )
        review_text = wiki_reviews.get(record["name"], "")
        if review_text:
            record["reviews"] = review_text[:500]
            record["review_source"] = REVIEW_SOURCE_WIKIVOYAGE
        else:
            # Public key-less APIs expose no real user reviews for these
            # places; we fill the column with a clearly-marked synthetic
            # impression so it stays informative without being dishonest.
            record["reviews"] = _category_review_placeholder(record["category"], record["name"])
            record["review_source"] = REVIEW_SOURCE_CATEGORY
        enriched.append(record)

    # RAW_COLUMNS already includes `review_source` (see its definition);
    # passing it again would duplicate the column.
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


def _category_review_placeholder(category: str, name: str) -> str:
    """Return a clearly-marked synthesized impression for a POI.

    Public key-less APIs do NOT expose user reviews for individual commercial
    venues (restaurants, bars, cafes, cinemas). Instead of leaving the column
    empty or fabricating fake ratings, we produce an honest "category-level"
    impression and mark its provenance as `category` (synthetic).

    Args:
        category: POI category key.
        name: the place name (used to make the text feel specific but honest).

    Returns:
        A short impression-like text.
    """
    templates = {
        "cafe": (
            "Одно из городских кофеен, где можно не спеша выпить кофе, "
            "поработать за ноутбуком или встретиться с друзьями. "
            "Атмосфера зависит от времени суток: днём уютно и спокойно, "
            "вечером — оживлённее."
        ),
        "restaurant": (
            "Ресторан с полноценным меню — от завтраков до ужина. "
            "Подходит для обеда в будни и для ужина в выходной, "
            "в том числе в компании или по особому поводу."
        ),
        "bar": (
            "Бар, куда приходят вечером после рабочего дня: разливное пиво, "
            "коктейли и непринуждённая живая атмосфера. "
            "Хорош для встречи с друзьями."
        ),
        "cinema": (
            "Городской кинотеатр с актуальным репертуаром. "
            "Подходит для вечернего сеанса — как в одиночку, так и в компании."
        ),
        "museum": (
            "Музей с постоянной экспозицией и временными выставками. "
            "Хорошо подходит для культурного выходного дня: спокойная "
            "атмосфера, осмотр займёт от часа до нескольких часов."
        ),
        "art_gallery": (
            "Галерея, где представлены произведения современного и "
            "классического искусства. Удобное место для неспешного просмотра "
            "выставок и знакомства с новыми художниками."
        ),
        "park": (
            "Городской парк с аллеями, зелёными зонами и местами для отдыха. "
            "Подходит для прогулки, пробежки или отдыха на свежем воздухе."
        ),
        "hotel": (
            "Гостиница с комфортными номерами для туристов и деловых "
            "поездок. Расположение и уровень сервиса делают её удобной "
            "базой для знакомства с городом."
        ),
    }
    text = templates.get(category)
    if not text:
        return ""
    return text


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

    # Final safety net: cap each category to the per-category limit regardless
    # of the source (a single source may return more than the cap).
    records = _cap_records_per_category(records, settings.osm_pois_per_category)

    df = save_records(records, settings)
    _log_collection_summary(df, settings)
    return df


def _log_collection_summary(df: pd.DataFrame, settings: Settings) -> None:
    """Print a short human-readable summary of the collection step."""
    print(f"[Task 1] Collected {len(df)} POIs")
    if "category" in df:
        print(df["category"].value_counts().to_string())
    print(f"Saved to {settings.data_dir / 'poi_dataset_raw.csv'}")
