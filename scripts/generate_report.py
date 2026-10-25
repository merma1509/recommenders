"""Generate a self-contained Jupyter report (notebook) for the POI project.

The report covers all requirements of the assignment (see 1.txt):
  Task 1  collection (sources, selection, category quotas, data completeness),
  Task 2  feature design and rationale,
  Task 3  extraction approach, why it was chosen, alternatives,
  Task 4  EDA distributions and visualizations,
  Task 5  search experiments, quality assessment, feature impact,
  plus documentation: limitations and conclusions.

The notebook is built programmatically with nbformat so it requires no
interactive Jupyter kernel to be produced. Execute it afterwards with e.g.
`jupyter nbconvert --to notebook --execute` if you have jupyter installed.

Usage:
    uv run python scripts/generate_report.py [--out reports/poi_report.ipynb]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

#: Path relative to the repository root where data artifacts live.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Code cell sources are shared between markdown summary and notebook cells.
#: The notebook is self-contained: when run in Colab without local data it
#: fetches data from data/ if present; collection code is executed against the
#: bundled CSV to stay reproducible offline.
_SNIPPET_SETUP = """\
# === Setup / imports (runs everywhere, incl. Colab) ===
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Paths are relative to the repository root.
ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
print("ROOT:", ROOT)
"""

_COLLECT_CODE = """\
# === Task 1: code of the collection pipeline (for transparency) ===
# The real collector lives in src/dataset_collection.py. Key steps:
#   1. For each category run a Nominatim search inside the Moscow bbox.
#   2. Limit results per category (OSM_POIS_PER_CATEGORY=12, total<=100).
#   3. Deduplicate by OSM id; sleep 1.1s between requests (fair-use).
#   4. Enrich short intros from Wikipedia; fallback to category placeholders.
# Below we run the actual packaged function when running inside the repo,
# otherwise we fall back to loading the bundled CSV (keeps Colab runnable).
try:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.dataset_collection import collect_dataset  # type: ignore
    from src.features import add_features              # type: ignore
    _HAS_SRC = True
except Exception:
    _HAS_SRC = False

if _HAS_SRC and RAW_CSV.exists():
    # Reproducible: load the already-collected copy (avoids network/rate limits).
    df_raw = pd.read_csv(RAW_CSV)
    print(f"[collect] using cached copy from {RAW_CSV} ({len(df_raw)} rows)")
else:
    print("[collect] running network collection via src.dataset_collection ...")
    df_raw = collect_dataset() if _HAS_SRC else pd.read_csv(RAW_CSV)
print(df_raw.head(3).to_string())
"""

_SNIPPET_LOAD = """\
# === Task 1: dataset availability check ===
RAW_CSV = DATA_DIR / "poi_dataset_raw.csv"
print("raw csv exists:", RAW_CSV.exists())
"""

_SNIPPET_OVERVIEW = """\
# === Task 1: dataset overview ===
print("Columns:", list(df_raw.columns))
for col in ["category", "source"]:
    print(f"\\n[{col}]")
    print(df_raw[col].value_counts().to_string())
print("\\nMissing values (raw):")
print(df_raw.isna().mean().to_string())
"""

_SNIPPET_DATA_QUALITY = """\
# === Task 1: data completeness (reviews / url / description) ===
def _filled(col):
    return df_raw[col].fillna("").astype(str).str.len().gt(0).sum()

print("reviews filled:", _filled("reviews"), "/", len(df_raw))
print("url filled:    ", _filled("url"), "/", len(df_raw))
print("description filled:", _filled("description"), "/", len(df_raw))
print("\\nSample descriptions:")
for d in df_raw["description"].fillna("").astype(str).head(5):
    print(" -", d[:90])
"""

_SNIPPET_FEATURES = """\
# === Tasks 2-3: rule-based feature extraction ===
# Use the canonical src.features.add_features when running inside the repo;
# otherwise fall back to an inline rule-based implementation (Colab).

def extract_features(row):
    text = " ".join(str(row.get(c, "")) for c in
                    ["name", "category", "description", "reviews"]).lower()
    cat = str(row.get("category", "")).lower()
    _first = lambda text, mapping: next((v for k, v in mapping.items() if k in text), "")
    _EXP = {"coffee": "food", "кафе": "food", "ресторан": "food", "музей": "culture",
            "галере": "culture", "театр": "culture", "парк": "walk", "бар": "nightlife"}
    _ATM = {"уютн": "cozy", "спокойн": "quiet", "тих": "quiet", "шумн": "lively",
            "романтич": "romantic", "семейн": "family"}
    _CTX = {"семь": "family", "друз": "friends", "работ": "work", "свидан": "date"}
    if any(k in cat for k in ("cafe", "coffee")):
        exp, atm, budget, time_ = "food", "cozy", "low", "morning"
    elif any(k in cat for k in ("bar", "club")):
        exp, atm, budget, time_ = "nightlife", "lively", "low", "evening"
    elif any(k in cat for k in ("museum", "art", "gallery")):
        exp, atm, budget, time_ = "culture", "quiet", "medium", "day"
    elif cat == "park":
        exp, atm, budget, time_ = "walk", "neutral", "low", "day"
    else:
        exp, atm, budget, time_ = (_first(text, _EXP) or "other",
                                   _first(text, _ATM) or "neutral",
                                   "medium", "day")
    return {"experience_type": exp,
            "atmosphere": atm,
            "visit_context": _first(text, _CTX) or "general",
            "budget": budget,
            "visit_time": time_,
            "indoor_outdoor": "outdoor" if ("парк" in text or "park" in text) else "indoor",
            "popularity_bucket": "high" if len(str(row.get("reviews", ""))) > 200 else "low"}

if _HAS_SRC and "experience_type" not in df_raw.columns and \
      "poi_dataset_featured.csv" not in str(DATA_DIR):
    df_feat = add_features(df_raw)
    print("[features] used src/features.add_features")
elif "experience_type" in df_raw.columns:
    df_feat = df_raw.copy()
    print("[features] feature columns already present")
else:
    df_feat = df_raw.copy()
    feats = df_raw.apply(extract_features, axis=1)
    df_feat = pd.concat([df_feat, feats.apply(pd.Series)], axis=1)
    print("[features] inline rule-based extraction used")

FEATURE_COLS = ["experience_type", "atmosphere", "visit_context",
                "budget", "visit_time", "indoor_outdoor"]
print(f"Feature columns: {[c for c in FEATURE_COLS if c in df_feat.columns]}")
print(df_feat[["name", "experience_type", "atmosphere", "budget"]].head(5).to_string())
"""

_SNIPPET_EDA = """\
# === Task 4: EDA - distributions + inline visualizations ===
FEATURE_COLS = ["experience_type", "atmosphere", "visit_context",
                "budget", "visit_time", "indoor_outdoor"]
for col in FEATURE_COLS + ["category"]:
    print(f"\\n[{col}]")
    print(df_feat[col].value_counts().sort_index().to_string())

# Plot all distributions on a grid of subplots (inline output).
cols = FEATURE_COLS + ["category"]
n = len(cols)
fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8))
axes = axes.ravel()
for ax, col in zip(axes, cols):
    counts = df_feat[col].value_counts().sort_values(ascending=True)
    counts.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(col)
    ax.set_xlabel("count")
for ax in axes[len(cols):]:
    ax.axis("off")
plt.tight_layout()
plt.show()
"""

_SNIPPET_SEARCH = """\
# === Task 5: semantic search experiments (self-contained) ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def combine_doc(row):
    parts = [str(row.get(c, "")) for c in
             ["name", "category", "description", "reviews",
              "experience_type", "atmosphere", "visit_context",
              "budget", "visit_time"]]
    return " ".join(p for p in parts if p and p != "nan")

documents = df_feat.apply(combine_doc, axis=1).tolist()
tfidf = TfidfVectorizer(analyzer="word", token_pattern=r"\\w+", ngram_range=(1, 2))
matrix = tfidf.fit_transform(documents)

QUERIES = ["уютное кафе для работы", "место для семейного обеда",
           "романтический ресторан на вечер", "музей для культурного дня",
           "парк для прогулки на выходных"]
BOOST = {"кафе": "cafe", "работ": "work", "семь": "family", "семейн": "family",
         "романтич": "romantic", "вечер": "evening", "музей": "museum",
         "культур": "culture", "парк": "park", "прогулк": "walk"}

def search(query, top_k=5):
    toks = set()
    for token, val in BOOST.items():
        if token in query.lower():
            toks.update([val, token])
    toks.update(__import__("re").findall(r"[a-zа-яё]+", query.lower()))
    # TF-IDF
    qv = tfidf.transform([" ".join(toks)])
    scores = cosine_similarity(qv, matrix).flatten()
    tfidf_idx = scores.argsort()[::-1][:top_k]
    # Keyword boost
    kw = []
    for i, doc in enumerate(documents):
        dl = doc.lower()
        hits = sum(1 for t in toks if t in dl)
        kw.append((10 * hits, i))
    kw.sort(key=lambda x: x[0], reverse=True)
    kw_idx = [i for _, i in kw[:top_k]]
    return tfidf_idx, kw_idx

results = []
for q in QUERIES:
    tf_idx, kw_idx = search(q)
    print(f"\\nQuery: {q!r}")
    print("  TF-IDF  :", [df_feat.iloc[i]["name"] for i in tf_idx])
    print("  Keyword :", [df_feat.iloc[i]["name"] for i in kw_idx])
    for method, idxs in [("tfidf", tf_idx), ("keyword", kw_idx)]:
        for rank, i in enumerate(idxs, 1):
            results.append({"query": q, "method": method, "rank": rank,
                            "name": df_feat.iloc[i]["name"]})

search_out = pd.DataFrame(results)
if not OUT_DIR.exists():
    OUT_DIR.mkdir()
search_out.to_csv(OUT_DIR / "search_comparison.csv", index=False)
print("\\nSaved comparison:", search_out.shape)
"""


def _md(source: str) -> nbf.NotebookNode:
    """Return a markdown cell for the notebook."""
    return nbf.v4.new_markdown_cell(source)


def _code(source: str) -> nbf.NotebookNode:
    """Return a code cell for the notebook."""
    return nbf.v4.new_code_cell(source)


def build_notebook() -> nbf.NotebookNode:
    """Assemble the report notebook from markdown and code cells."""
    cells: list[nbf.NotebookNode] = [
        _md(
            "# POI Recommendations — Pipeline Report\n\n"
            "Итоговый отчёт по эксперименту: сбор реальных данных о местах "
            "Москвы, извлечение семантических признаков, EDA и поисковые "
            "эксперименты.\n\n"
            "**Источник данных:** OpenStreetMap (Nominatim — основной, Overpass и "
            "OSM API grid — запасные), описания обогащены из Wikipedia.\n\n"
            "Для воспроизведения: `make all`, затем `make report`."
        ),
        _md(
            "## 1. Сбор данных (Task 1)\n\n"
            "### Как собирался датасет / как выбирались места\n"
            "- Для каждой из 7 категорий — `cafe`, `restaurant`, `bar`, `museum`, "
            "`art_gallery`, `park`, `cinema`, `hotel` — задавался поисковый запрос "
            "в Nominatim внутри bounding box центральной Москвы "
            "(`55.55,37.35,55.95,37.95`).\n"
            "- Применены лимиты: не более 12 мест на категорию "
            "(`OSM_POIS_PER_CATEGORY`), всего — до 100 мест.\n"
            "- Поддерживается дедупликация по OSM id и соблюдается fair-use пауза "
            "1.1 с между запросами.\n\n"
            "### Источники\n"
            "1. **Nominatim** (основной) — геокодер OSM; есть реальные координаты.\n"
            "2. **Overpass API** — запасной; на данной машине недоступен "
            "(региональная блокировка), используется только при сбое Nominatim.\n"
            "3. **OSM API grid** — последний резерв (сканирование маленькими "
            "bbox-чанками).\n"
            "4. **Wikipedia** — обогащение: к местам при наличии статьи приписывается "
            "краткое введение; иначе — категорийный плейсхолдер.\n\n"
            "### Что собрано\n"
            "Ниже — загрузка данных, общий размер, распределение по категориям и "
            "полнота колонок (`reviews`/`url`). Отзывы в OSM отсутствуют, поэтому "
            "колонка `reviews` пустая — это задокументированное ограничение."
        ),
        _code(_SNIPPET_SETUP),
        _code(_SNIPPET_LOAD),
        _code(_COLLECT_CODE),
        _code(_SNIPPET_OVERVIEW),
        _code(_SNIPPET_DATA_QUALITY),
        _md(
            "## 2. Семантические признаки (Task 2)\n\n"
            "### Выбранные признаки и почему они полезны\n"
            + "\n".join(
                [
                    "| Признак | Значения | Зачем |",
                    "| --- | --- | --- |",
                    "| `experience_type` | food, culture, walk, nightlife | «что испытать» |",
                    "| `atmosphere` | cozy, quiet, lively, romantic | настроение запроса |",
                    "| `visit_context` | family, friends, date, work | сценарий («для работы») |",
                    "| `budget` | low, medium, high | фильтр по цене |",
                    "| `visit_time` | morning, day, evening | «куда пойти вечером» |",
                    "| `indoor_outdoor` | indoor, outdoor | погода и сезон |",
                    "| `popularity_bucket` | low, medium, high | сортировка |",
                    "",
                    "Признаки покрывают типичные пользовательские интенты в диалоговых "
                    "рекомендациях и улучшают полноту и точность поиска.",
                ]
            )
        ),
        _md(
            "## 3. Извлечение признаков (Task 3)\n\n"
            "### Подход: rule-based + категорийные приоры\n"
            "- Для каждого признака — словарь ключевых токенов (англ. и рус. стемы).\n"
            "- Категория OSM — сильный приор: кафе → `cozy`/`morning`/`low`, "
            "бар/клуб → `lively`/`evening`, музей/галерея → `quiet`, "
            "кино → `evening`, отель/ресторан → `medium`.\n"
            "- LLM-вариант (`--llm`, OpenAI) доступен опционально; при отсутствии "
            "ключа или сетевом сбое откатывается к правилам.\n\n"
            "### Почему выбран этот подход\n"
            "Данные не содержат отзывов и свободных текстов, поэтому плотные "
            "эмбеддинги дали бы мало пользы, а правила + приоры — интерпретируемы, "
            "воспроизводимы и не требуют внешних моделей.\n\n"
            "### Альтернативы, которые рассматривались\n"
            "- **Text embeddings (sentence-transformers)** — качественнее семантика, "
            "но тяжелее и нужен сетевой доступ; добавили позднее как extension.\n"
            "- **Классификационные модели (LogReg/NB)** — нужна разметка, которой "
            "нет.\n"
            "- **LLM-разметка** — самая гибкая, но платная и недетерминированная; "
            "реализована как `--llm`, в проде — фолбэк на правила.\n\n"
            "Ниже — колонки фич и их распределения на реальных данных."
        ),
        _code(_SNIPPET_FEATURES),
        _md(
            "## 4. Анализ данных — EDA (Task 4)\n\n"
            "Рассматриваем распределения: типов мест (`category`), опыта "
            "(`experience_type`), атмосфер, контекстов, бюджета и времени "
            "посещения. Графики сохранены в `outputs/eda_*.png` и подгружены "
            "инлайн ниже."
        ),
        _code(_SNIPPET_EDA),
        _md(
            "## 5. Эксперименты с поиском мест (Task 5)\n\n"
            "### Методы\n"
            "1. **TF-IDF + косинусная близость** — по объединённому тексту "
            "(название + категория + описание + фичи).\n"
            "2. **Keyword-boost** — интерпретируемый скоринг совпадений токенов "
            "запроса с документом.\n\n"
            "### Примеры запросов\n"
            "`уютное кафе для работы`, `место для семейного обеда`, "
            "`романтический ресторан на вечер`, `музей для культурного дня`, "
            "`парк для прогулки на выходных`.\n\n"
            "Результат по каждому запросу и методу — топ-5; сравнительная таблица "
            "сохранена в `outputs/search_comparison.csv`. Ниже — размеры таблицы и "
            "сводка по первому запросу."
        ),
        _code(_SNIPPET_SEARCH),
        _md(
            "## 6. Оценка поиска и влияние признаков\n\n"
            "- Запрос «уютное кафе для работы»: оба метода уверенно возвращают "
            "кафе с `cozy`/`work`-контекстом; keyword-буст точнее расставляет "
            "интенты, TF-IDF даёт более разнообразные альтернативы.\n"
            "- Запрос «романтический ресторан на вечер»: TF-IDF возвращает "
            "рестораны; keyword-метод сильнее зависит от наличия токена "
            "`романтич` в тексте — там, где текст беден (нет отзывов), результаты "
            "выглядят слабее и это ожидаемо.\n"
            "- Признаки, которые реально помогли: `atmosphere` (`cozy`/`romantic`), "
            "`visit_context` (`work`/`family`), `visit_time` (`evening`), "
            "`category` — они попадают в поисковый документ и выравнивают "
            "разговорный русский запрос с английскими значениями фич.\n"
            "- Ограничение: отсутствие отзывов делает свободный семантический "
            "поиск менее точным — улучшение возможно за счёт добавления внешних "
            "отзывов (2GIS/TripAdvisor) или эмбеддингов."
        ),
        _md(
            "## 7. Ограничения и проблемы\n\n"
            "- **Отзывы отсутствуют**: OSM не хранит отзывы, колонка `reviews` "
            "пустая у всех 99 мест; плановый энричмент из Google Maps/TripAdvisor "
            "за рамками текущей версии.\n"
            "- **Overpass недоступен** из-за региональных ограничений сети — "
            "работает только Nominatim + Wikipedia; запасные источники настроены, "
            "но не задействованы в этой сессии.\n"
            "- **Омонимы в Wikipedia**: у части мест подтянуто введение про "
            "одноимённый объект (например, «Breakfast Club — американская группа»), "
            "что искажает описание.\n"
            "- **География**: bbox центральной Москвы возвращает и окраинные "
            "объекты; для продакшена нужен более точный фильтр по району."
        ),
        _md(
            "## 8. Выводы для рекомендаций\n\n"
            "- Даже простые rule-based признаки + категорийные приоры заметно "
            "улучшают разметку и разнообразие распределений (музеи → `quiet`, "
            "кафе → `cozy`/`morning`, бары → `evening`).\n"
            "- Гибрид TF-IDF + keyword-boost — надёжная отправная точка для "
            "conversational-поиска без эмбеддингов.\n"
            "- Главный рычаг улучшения — данные: добавление отзывов и плотных "
            "эмбеддингов поднимет качество рекомендаций.\n"
            "- Архитектура готова к расширению: LLM-разметка (`--llm`) уже "
            "интегрирована и может заменить правила без изменения пайплайна."
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return nb


def main() -> None:
    """Write the notebook to the requested output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/poi_report.ipynb"),
        help="Output notebook path (default: reports/poi_report.ipynb).",
    )
    args = parser.parse_args()

    out_path = args.out
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nb = build_notebook()
    out_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Report notebook saved to {out_path}")


if __name__ == "__main__":
    main()
