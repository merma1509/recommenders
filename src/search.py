"""Task 5: Semantic search experiments over the POI dataset.

Implements two retrieval strategies over the enriched dataset:
  1. TF-IDF vectorizer + cosine similarity over combined text (pure sklearn,
     no external embeddings required).
  2. A small rule-based keyword scorer that boosts places matching the query.

A comparison table (query, top results, notes) is written to outputs/.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Settings

#: Example user queries used for the experiments in the task description.
EXAMPLE_QUERIES: list[str] = [
    "уютное кафе для работы",
    "место для семейного обеда",
    "романтический ресторан на вечер",
    "музей для культурного дня",
    "парк для прогулки на выходных",
]

#: Simple keyword boost rules: Russian query token -> English feature value.
#: Keys may be Russian (intent words) or English (raw feature values).
#: Use stems (e.g. "rabot") to match inflected Russian word forms.
_QUERY_BOOST_TOKENS: dict[str, str] = {
    "кафе": "cafe",
    "ресторан": "restaurant",
    "музей": "museum",
    "парк": "park",
    "кино": "cinema",
    "работ": "work",
    "семь": "family",
    "семейн": "family",
    "романтич": "romantic",
    "уютн": "cozy",
    "вечер": "evening",
    "культур": "culture",
    "прогулк": "walk",
    "природа": "outdoor",
    "друз": "friends",
    "свидан": "date",
    "недорого": "low",
    "бюджет": "low",
    "дорого": "high",
}


def _expand_query_tokens(query: str) -> set[str]:
    """Expand a conversational query with English feature values.

    Closes the gap between Russian user intents and the English feature values
    stored in the dataset, making both TF-IDF and keyword scoring meaningful.
    """
    lowered = query.lower()
    targets: set[str] = set()
    for token, value in _QUERY_BOOST_TOKENS.items():
        if token in lowered:
            targets.add(value)
            targets.add(token)
    # Also add the raw words of the query for term-overlap scoring.
    targets.update(re.findall(r"[a-zа-яё]+", lowered))
    return targets


def _combine_searchable_text(df: pd.DataFrame) -> list[str]:
    """Build one searchable text document per POI row.

    Uses raw text plus extracted semantic features so that keyword and TF-IDF
    search can match across both descriptive text and structured attributes.
    """
    documents: list[str] = []
    for _, row in df.iterrows():
        parts = [
            str(row.get("name", "")),
            str(row.get("category", "")),
            str(row.get("description", "")),
            str(row.get("reviews", "")),
            str(row.get("experience_type", "")),
            str(row.get("atmosphere", "")),
            str(row.get("visit_context", "")),
            str(row.get("budget", "")),
            str(row.get("visit_time", "")),
        ]
        documents.append(" ".join(p for p in parts if p and p != "nan"))
    return documents


def _tfidf_search(
    query: str,
    documents: list[str],
    tfidf: TfidfVectorizer,
    matrix: Any,
    top_k: int,
) -> list[int]:
    """Rank documents by cosine similarity to the query using TF-IDF.

    The query is expanded with its translated feature values, so Russian
    conversational intents can align with English feature tokens in the docs.
    """
    expanded_query = " ".join(_expand_query_tokens(query))
    query_vec = tfidf.transform([expanded_query])
    similarities = cosine_similarity(query_vec, matrix).flatten()
    return similarities.argsort()[::-1][:top_k].tolist()


def _keyword_boost_search(query: str, documents: list[str], top_k: int) -> list[int]:
    """Rank documents by a simple token-boost score.

    This strategy is interpretable and works well for short conversational
    queries, complementing the dense TF-IDF approach.
    """
    query_tokens = _expand_query_tokens(query)
    scores: list[tuple[float, int]] = []
    for idx, doc in enumerate(documents):
        doc_lower = doc.lower()
        hits = sum(1 for token in query_tokens if token in doc_lower)
        # Primary sort by boost hits, secondary by raw term overlap.
        term_overlap = len(set(re.findall(r"\w+", doc_lower)) & query_tokens)
        scores.append((10 * hits + 0.1 * term_overlap, idx))
    scores.sort(key=lambda t: t[0], reverse=True)
    return [idx for _, idx in scores[:top_k]]


def _format_results(df: pd.DataFrame, indices: list[int], query: str) -> list[dict[str, Any]]:
    """Format ranked indices into readable result rows."""
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(indices, start=1):
        row = df.iloc[idx]
        rows.append(
            {
                "rank": rank,
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "atmosphere": row.get("atmosphere", ""),
                "experience_type": row.get("experience_type", ""),
                "visit_context": row.get("visit_context", ""),
                "budget": row.get("budget", ""),
                "visit_time": row.get("visit_time", ""),
            }
        )
    return rows


def run_search(df: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    """Run the search experiments and persist a comparison table.

    Args:
        df: feature-enriched POI DataFrame.
        settings: optional settings override.

    Returns:
        A DataFrame with columns: query, method, rank, name, category, checks.
    """
    settings = settings or Settings()
    documents = _combine_searchable_text(df)
    tfidf = TfidfVectorizer(analyzer="word", token_pattern=r"\w+", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(documents)

    tfidf_topk = 5
    keyword_topk = 5

    log_rows: list[dict[str, Any]] = []
    for query in EXAMPLE_QUERIES:
        tfidf_indices = _tfidf_search(query, documents, tfidf, matrix, tfidf_topk)
        keyword_indices = _keyword_boost_search(query, documents, keyword_topk)

        tfidf_results = _format_results(df, tfidf_indices, query)
        keyword_results = _format_results(df, keyword_indices, query)

        print(f"\nQuery: {query!r}")
        print("  [TF-IDF]    ", _names_only(tfidf_results))
        print("  [Keyword]   ", _names_only(keyword_results))

        for result in tfidf_results:
            row = dict(result)
            row.update({"query": query, "method": "tfidf"})
            log_rows.append(row)
        for result in keyword_results:
            row = dict(result)
            row.update({"query": query, "method": "keyword"})
            log_rows.append(row)

    comparison = pd.DataFrame(log_rows)
    out_path = settings.outputs_dir / "search_comparison.csv"
    comparison.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[Task 5] Comparison table saved to {out_path}")
    return comparison


def _names_only(results: list[dict[str, Any]]) -> list[str]:
    """Return just the names for compact console output."""
    return [r["name"] for r in results]
