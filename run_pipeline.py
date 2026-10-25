"""CLI entry point for the POI recommendations pipeline.

Usage:
    uv run python run_pipeline.py --stage all
    uv run python run_pipeline.py --stage collect
    uv run python run_pipeline.py --stage features
    uv run python run_pipeline.py --stage analyze
    uv run python run_pipeline.py --stage search

Stages map to the test assignment tasks:
    collect  -> Task 1    (dataset collection)
    features -> Tasks 2-3 (semantic feature extraction)
    analyze  -> Task 4    (EDA and visualizations)
    search   -> Task 5    (semantic search experiments)
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import Settings, ensure_dirs
from src.dataset_collection import collect_dataset
from src.eda import run_eda
from src.features import add_features, extract_with_llm
from src.search import run_search

STAGES = ("collect", "features", "analyze", "search", "all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="POI recommendations pipeline")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Which pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM-based feature extraction (requires OPENAI_API_KEY).",
    )
    return parser.parse_args(argv)


def _load_featured_df(settings: Settings) -> pd.DataFrame:
    """Load the feature-enriched dataset or run the previous stages if missing."""
    featured_path = settings.data_dir / "poi_dataset_featured.csv"
    if featured_path.exists():
        return pd.read_csv(featured_path)
    # Fall back: compute features from the raw collected dataset.
    raw_path = settings.data_dir / "poi_dataset_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            "No dataset found. Run `make collect` first (or enable OFFLINE_MODE)."
        )
    return pd.read_csv(raw_path)


def main(argv: list[str] | None = None) -> int:
    """Run the requested pipeline stage(s). Returns process exit code."""
    args = parse_args(argv)
    settings = Settings()
    ensure_dirs(settings)

    stage = args.stage
    if stage == "all":
        stages = ("collect", "features", "analyze", "search")
    else:
        stages = (stage,)

    for current in stages:
        _run_stage(current, settings, args.llm)

    return 0


def _run_stage(stage: str, settings: Settings, use_llm: bool) -> None:
    """Dispatch a single pipeline stage."""
    if stage == "collect":
        collect_dataset(settings)
        return

    # For features, analyze and search we need the (possibly raw) dataset.
    df = _load_featured_df(settings)
    if stage == "features":
        if use_llm:
            extract_with_llm(df, settings)
        else:
            add_features(df, settings)
    elif stage == "analyze":
        featured = _ensure_featured(df, settings, use_llm)
        run_eda(featured, settings)
    elif stage == "search":
        featured = _ensure_featured(df, settings, use_llm)
        run_search(featured, settings)


def _ensure_featured(df: pd.DataFrame, settings: Settings, use_llm: bool) -> pd.DataFrame:
    """Return a feature-enriched DataFrame, computing features if necessary."""
    featured_path = settings.data_dir / "poi_dataset_featured.csv"
    if featured_path.exists():
        return pd.read_csv(featured_path)
    if use_llm:
        return extract_with_llm(df, settings)
    return add_features(df, settings)


if __name__ == "__main__":
    sys.exit(main())
