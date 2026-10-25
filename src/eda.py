"""Task 4: Exploratory data analysis (EDA) and visualization.

Computes distributions of POI categories, experience types, atmospheres, visit
contexts and budget levels, and stores charts in the outputs directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless generation.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import Settings

#: Columns that should be categorical for plotting purposes.
CATEGORICAL_COLUMNS = [
    "category",
    "experience_type",
    "atmosphere",
    "visit_context",
    "budget",
    "visit_time",
]


def _plot_bar(series: pd.Series, title: str, output_path: Path) -> None:
    """Build and save a horizontal bar chart of a categorical value counts."""
    plt.figure(figsize=(9, 5))
    counts = series.value_counts().sort_values(ascending=True)
    sns.barplot(
        x=counts.values,
        y=counts.index,
        hue=counts.index,
        legend=False,
        palette="viridis",
    )
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_eda(df: pd.DataFrame, settings: Settings | None = None) -> dict[str, Any]:
    """Generate all EDA visualizations and print a numeric summary.

    Args:
        df: feature-enriched POI DataFrame.
        settings: optional settings override.

    Returns:
        A dict with the generated figure paths, for reporting.
    """
    settings = settings or Settings()
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, str] = {}
    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue
        title = f"Distribution of {column.replace('_', ' ').title()}"
        out = settings.outputs_dir / f"eda_distribution_{column}.png"
        _plot_bar(df[column], title, out)
        figures[column] = str(out)

    # Print compact numeric summary for the console/report.
    print("=" * 60)
    print("Task 4: EDA summary")
    print("=" * 60)
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            print(f"\n[{column}]")
            print(df[column].value_counts().to_string())
    print("\n[missing values]")
    print(df.isna().mean().to_string())

    return figures
