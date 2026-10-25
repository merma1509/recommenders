"""Shared utility helpers used across the pipeline."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def load_csv_or_none(path: str | Path) -> pd.DataFrame | None:
    """Read a CSV file into a DataFrame, returning None if missing.

    Args:
        path: path to the CSV file.

    Returns:
        DataFrame or None when the file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


def slugify(text: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug.

    Args:
        text: input string.

    Returns:
        Lowercased string with non-alphanumeric characters replaced by '_'.
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "untitled"
