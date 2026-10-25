"""Configuration for the POI recommendations pipeline.

All settings are read from environment variables (loaded from `.env` when present)
with sensible defaults for offline/dev usage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load `.env` variables into the environment if present (project root).
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class Settings:
    """Central configuration object with typed accessors.

    All fields have defaults so the pipeline runs without any external config.
    """

    # --- Data collection (Task 1) ---
    # Bounding box for the OSM query in the format min_lat,min_lon,max_lat,max_lon.
    # Default: central Moscow area.
    osm_bounding_box: str = os.getenv("OSM_BOUNDING_BOX", "55.55,37.35,55.95,37.95")
    # Maximum number of places fetched per category to balance coverage.
    osm_pois_per_category: int = int(os.getenv("OSM_POIS_PER_CATEGORY", "12"))

    # --- Offline mode (deterministic testing without network) ---
    offline_mode: bool = os.getenv("OFFLINE_MODE", "false").lower() == "true"
    offline_csv: str = os.getenv("OFFLINE_CSV", str(DATA_DIR / "poi_dataset_raw.csv"))

    # --- Optional LLM feature extraction (Task 3) ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # --- Paths where artifacts are written ---
    data_dir: Path = DATA_DIR
    outputs_dir: Path = OUTPUTS_DIR


def get_settings() -> Settings:
    """Return the application settings (cached at module import for simplicity)."""
    return Settings()


def ensure_dirs(settings: Settings) -> None:
    """Create output directories if they do not exist yet."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
