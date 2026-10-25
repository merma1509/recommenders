"""Tasks 2-3: Semantic feature definition and extraction.

Defines the semantic feature set useful for POI recommendations and implements
a deterministic rule-based extraction pipeline. An optional LLM-based extractor
is available (uses OpenAI) when an API key is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import Settings

# Feature catalogue (Task 2)
#: Semantic features we compute for every place, with a short rationale.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "experience_type": "Type of experience: food, culture, leisure, walk, nightlife.",
    "atmosphere": "Atmosphere: cozy, quiet, lively, romantic, family, work-friendly.",
    "visit_context": "Visit context: family, friends, date, work.",
    "budget": "Budget level: low, medium, high.",
    "visit_time": "Time of day recommended: morning, day, evening.",
    "indoor_outdoor": "Whether the place is indoors or outdoors.",
    "popularity_bucket": "Popularity bucket derived from OSM metadata.",
}

# Rule-based keyword dictionaries.
# Keys are match tokens (Russian intent words or English feature values) -> feature value.
_EXPERIENCE_TOKENS: dict[str, str] = {
    "coffee": "food",
    "food": "food",
    "restaurant": "food",
    "café": "food",
    "bar": "nightlife",
    "club": "nightlife",
    "music": "nightlife",
    "museum": "culture",
    "exhibition": "culture",
    "art": "culture",
    "theater": "culture",
    "park": "walk",
    "garden": "walk",
    "square": "walk",
    "cinema": "entertainment",
    "movie": "entertainment",
    # Russian intent tokens:
    "кофейн": "food",
    "кафе": "food",
    "ресторан": "food",
    "столов": "food",
    "музей": "culture",
    "галере": "culture",
    "искусств": "culture",
    "театр": "culture",
    "парк": "walk",
    "выставк": "culture",
    "кинотеатр": "entertainment",
    "бар": "nightlife",
    "вечеринк": "nightlife",
}

_ATMOSPHERE_TOKENS: dict[str, str] = {
    "cozy": "cozy",
    "quiet": "quiet",
    "lively": "lively",
    "loud": "lively",
    "romantic": "romantic",
    "family": "family",
    "hustle": "lively",
    # Russian:
    "уютн": "cozy",
    "спокойн": "quiet",
    "тих": "quiet",
    "приглуш": "quiet",
    "атмосфер": "cozy",
    "шумн": "lively",
    "громк": "lively",
    "оживл": "lively",
    "танцплощадк": "lively",
    "романтич": "romantic",
    "семейн": "family",
}

_CONTEXT_TOKENS: dict[str, str] = {
    "family": "family",
    "kids": "family",
    "children": "family",
    "friends": "friends",
    "date": "date",
    "work": "work",
    "meeting": "work",
    # Russian:
    "семь": "family",
    "семейн": "family",
    "дет": "family",
    "друз": "friends",
    "свидан": "date",
    "работ": "work",
    "коллег": "work",
    "командир": "work",
    "делов": "work",
}

# Budget matching: works on OSM "price:" tags and Russian/English hints.
_BUDGET_TOKENS: dict[str, str] = {
    "price:cheap": "low",
    "price:€": "low",
    "price:€€": "medium",
    "price:€€€": "high",
    "недорог": "low",
    "бюджетн": "low",
    "дорог": "high",
    "премиал": "high",
}


def _match_first_tokens(text: str, mapping: dict[str, str]) -> str | None:
    """Return the first mapping value whose key is found in the lowercased text."""
    lowered = text.lower()
    for token, value in mapping.items():
        if token in lowered:
            return value
    return None


# Extraction (Task 3)
@dataclass
class FeatureExtractor:
    """Deterministic rule-based feature extractor.

    Extracts the semantic features defined in FEATURE_DESCRIPTIONS from a
    combined text of name, category, description and reviews.
    """

    def extract(self, row: pd.Series) -> dict[str, Any]:
        """Compute all semantic features for a single POI row.

        Args:
            row: a pandas Series with the POI columns (name, category, ...).

        Returns:
            A dict mapping feature name -> extracted value (str or float).
        """
        text = self._combine(row)
        category = str(row.get("category", "")).lower()

        experience = self._extract_experience(row, text)
        atmosphere = self._extract_atmosphere(text, category)
        visit_context = _match_first_tokens(text, _CONTEXT_TOKENS) or "general"
        budget = self._extract_budget(text, category)
        visit_time = self._extract_visit_time(text, category)
        indoor_outdoor = self._extract_indoor_outdoor(text)
        popularity_bucket = self._extract_popularity(row)

        return {
            "experience_type": experience,
            "atmosphere": atmosphere,
            "visit_context": visit_context,
            "budget": budget,
            "visit_time": visit_time,
            "indoor_outdoor": indoor_outdoor,
            "popularity_bucket": popularity_bucket,
        }

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _combine(row: pd.Series) -> str:
        """Join textual columns for downstream keyword matching."""
        parts = [
            str(row.get("name", "")),
            str(row.get("category", "")),
            str(row.get("description", "")),
            str(row.get("reviews", "")),
        ]
        return " ".join(p for p in parts if p and p != "nan")

    @staticmethod
    def _extract_experience(row: pd.Series, text: str) -> str:
        """Derive the primary experience type.

        The OSM category is a strong prior: cafes/restaurants -> food,
        etc. Keyword fallback covers textual hints.
        """
        category = str(row.get("category", "")).lower()
        if any(k in category for k in ("cafe", "restaurant", "bar")):
            return "nightlife" if category == "bar" else "food"
        if any(k in category for k in ("museum", "art")):
            return "culture"
        if category == "park":
            return "walk"
        if category == "cinema":
            return "entertainment"
        return _match_first_tokens(text, _EXPERIENCE_TOKENS) or "other"

    @staticmethod
    def _extract_atmosphere(text: str, category: str = "") -> str:
        """Extract an atmosphere label from the combined text.

        Category priors help when the free text has no explicit atmosphere
        marker (e.g. cafes tend to be cozy, bars lively, museums quiet).

        Args:
            text: combined place text.
            category: normalized OSM category.

        Returns:
            One of the atmosphere values defined in FEATURE_DESCRIPTIONS.
        """
        matched = _match_first_tokens(text, _ATMOSPHERE_TOKENS)
        if matched:
            return matched
        # Category-level priors apply when text gives no signal.
        if any(k in category for k in ("cafe", "coffee")):
            return "cozy"
        if any(k in category for k in ("bar", "club")):
            return "lively"
        if any(k in category for k in ("museum", "art", "gallery")):
            return "quiet"
        if category == "cinema":
            return "quiet"
        return "neutral"

    @staticmethod
    def _extract_budget(text: str, category: str = "") -> str:
        """Estimate a budget level.

        Uses explicit hints first (OSM 'price' tags, Russian/English keywords),
        then falls back to category priors: low-cost daily places vs premium.

        Args:
            text: combined place text.
            category: normalized OSM category.

        Returns:
            One of "low", "medium", "high".
        """
        lowered = text.lower()
        for token, value in _BUDGET_TOKENS.items():
            if token in lowered:
                return value
        # Category-based priors when no explicit price signal exists.
        if any(k in category for k in ("cafe", "bar", "park", "cinema")):
            return "low"
        if any(k in category for k in ("hotel", "restaurant")):
            return "medium"
        return "medium"

    @staticmethod
    def _extract_visit_time(text: str, category: str = "") -> str:
        """Return a recommended time of day (morning/day/evening) from hints.

        Explicit hints from the free text win over category priors.

        Args:
            text: combined place text.
            category: normalized OSM category.

        Returns:
            One of "morning", "day", "evening".
        """
        lowered = text.lower()
        if any(k in lowered for k in ("breakfast", "morning", "завтрак", "кофе")):
            return "morning"
        if any(k in lowered for k in ("lunch", "afternoon", "обед", "день")):
            return "day"
        if any(k in lowered for k in ("dinner", "evening", "night", "ужин", "вечер", "ноч")):
            return "evening"
        # Category priors: bars and clubs work best in the evening.
        if any(k in category for k in ("bar", "club", "cinema")):
            return "evening"
        if any(k in category for k in ("cafe", "breakfast")):
            return "morning"
        return "day"

    @staticmethod
    def _extract_indoor_outdoor(text: str) -> str:
        """Classify the place as indoor or outdoor."""
        lowered = text.lower()
        outdoor_hints = ("outdoor", "garden", "park", "свежем воздух", "открыт", "сад")
        if any(k in lowered for k in outdoor_hints):
            return "outdoor"
        return "indoor"

    @staticmethod
    def _extract_popularity(row: pd.Series) -> str:
        """Bucket popularity using any available numeric metadata.

        Falls back to a heuristic based on review length in the raw data.
        """
        reviews = row.get("reviews")
        if isinstance(reviews, str) and reviews and reviews != "nan":
            # Favour a longer review text as a weak proxy for popularity.
            return "high" if len(reviews) > 200 else "low"
        return "medium"


def add_features(df: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    """Add semantic feature columns to the POI DataFrame.

    Idempotent: if the feature columns already exist they are not recomputed,
    which prevents duplicate suffixed columns on repeated pipeline runs.

    Args:
        df: raw POI records (RAW_COLUMNS from the collection step).
        settings: optional settings (used to opt into LLM extraction).

    Returns:
        A copy of the input DataFrame with the feature columns appended.
    """
    settings = settings or Settings()
    feature_columns = set(FEATURE_DESCRIPTIONS)
    if feature_columns.issubset(df.columns):
        return df.copy()

    extractor = FeatureExtractor()
    feature_rows = df.apply(extractor.extract, axis=1)
    result = df.copy()
    result = pd.concat([result, feature_rows.apply(pd.Series)], axis=1)
    _persist(result, settings)
    return result


def extract_with_llm(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Optional LLM-based feature extraction (requires OPENAI_API_KEY).

    This is a thin wrapper: if no API key is configured it falls back to the
    rule-based pipeline with a warning, keeping the project runnable offline.

    The LLM is asked to classify each place; missing or malformed answers are
    filled with the rule-based defaults for the same row.

    Args:
        df: raw POI records.
        settings: settings containing the OpenAI configuration.

    Returns:
        Feature-enriched DataFrame.
    """
    if not settings.openai_api_key:
        print("[Task 3] No OPENAI_API_KEY set; falling back to rule-based extractor.")
        return add_features(df, settings)

    # Lazy import to avoid a hard dependency on the openai package.
    from openai import OpenAI  # type: ignore[import-not-found]

    client = OpenAI(api_key=settings.openai_api_key)
    extractor = FeatureExtractor()

    per_index: dict[int, dict[str, Any]] = {}
    for index, row in df.iterrows():
        prompt = _llm_prompt(row)
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_LLM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content or "{}"
            predicted = _parse_llm_json(raw)
        except Exception:  # noqa: BLE001 - network/parse errors fall back
            predicted = {}

        # Merge: keep every requested key, defaulting to the rule-based value.
        defaults = extractor.extract(row)
        merged = {**defaults, **{k: v for k, v in predicted.items() if v}}
        per_index[index] = merged

    out = df.copy()
    extracted = pd.DataFrame.from_dict(per_index, orient="index")
    return pd.concat([out, extracted], axis=1, ignore_index=False)


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object that the LLM may wrap in markdown fences.

    Args:
        raw: the raw text returned by the model.

    Returns:
        A dict of parsed features (possibly empty).
    """
    import json
    import re

    text = raw.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Fall back to the first {...} block when the model adds prose.
    if not text.startswith("{"):
        block = re.search(r"\{.*\}", text, re.DOTALL)
        if block:
            text = block.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _llm_prompt(row: pd.Series) -> str:
    """Build a single-row prompt for the LLM extractor."""
    text = FeatureExtractor._combine(row)
    category = str(row.get("category", "")).lower()
    return (
        "Extract semantic features for the following place.\n"
        f"Category: {category}\n"
        f"Text: {text}\n"
        'Return a JSON object with exactly these keys: "experience_type", '
        '"atmosphere", "visit_context", "budget", "visit_time", '
        '"indoor_outdoor", "popularity_bucket".\n'
        "Allowed values:\n"
        "  experience_type: food | culture | walk | nightlife | entertainment | other\n"
        "  atmosphere: cozy | quiet | lively | romantic | family | work | neutral\n"
        "  visit_context: family | friends | date | work | general\n"
        "  budget: low | medium | high\n"
        "  visit_time: morning | day | evening\n"
        "  indoor_outdoor: indoor | outdoor\n"
        "  popularity_bucket: low | medium | high"
    )


SYSTEM_LLM_PROMPT = (
    "You are a data annotator. Extract semantic features of places for a "
    "recommendation system from the given text. Use the place category as a "
    "strong prior and the text as evidence. Always answer with valid JSON "
    "containing exactly these keys and values from the allowed list provided "
    "by the user. Return only JSON, without extra prose."
)


def _persist(df: pd.DataFrame, settings: Settings) -> None:
    """Write the enriched DataFrame to data/poi_dataset_featured.csv."""
    out_path = settings.data_dir / "poi_dataset_featured.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Tasks 2-3] Feature extraction done: {len(df)} rows -> {out_path}")
