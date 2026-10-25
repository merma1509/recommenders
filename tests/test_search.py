"""Tests for the semantic search module (Task 5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Settings
from src.search import _expand_query_tokens, _keyword_boost_search, run_search


def test_expand_query_tokens_detects_cafe() -> None:
    tokens = _expand_query_tokens("уютное кафе для работы")
    assert "cafe" in tokens
    assert "work" in tokens


def test_keyword_boost_ranks_cafes_first(featured_poi_df: pd.DataFrame) -> None:
    documents = [
        "Уютная кофейня работа wifi cafe",
        "Художественный музей культура museum",
    ]
    indices = _keyword_boost_search("уютное кафе для работы", documents, top_k=2)
    # The cafe document (index 0) must be ranked first.
    assert indices[0] == 0


def test_run_search_outputs_comparison(tmp_path: Path, featured_poi_df: pd.DataFrame) -> None:
    settings = Settings(outputs_dir=tmp_path, data_dir=tmp_path)
    result = run_search(featured_poi_df, settings)
    assert {"query", "method", "rank", "name"}.issubset(result.columns)
    assert len(result) >= 4  # at least two queries * two methods
