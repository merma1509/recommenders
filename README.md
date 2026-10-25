# POI Recommendations

Pipeline for collecting and structuring information about points of interest (cafes, restaurants, museums, parks, etc.) enriched with semantic features useful for searching and conversational recommendations.

## Tasks

1. **Dataset collection** — collect 50–100 places from open sources (OpenStreetMap via Nominatim/Overpass; short descriptions are enriched from Wikipedia).
2. **Semantic feature formation** — define features useful for recommendations (experience type, atmosphere, visit context, budget, visit time, popularity).
3. **Feature extraction** — extract features from text descriptions using rule-based methods, with an optional LLM-based modules.
4. **EDA** — analyze the dataset and build visualizations.
5. **Search experiments** — implement semantic search over places and test user queries.

## Project structure

```bash
.
├── Makefile               # task automation
├── pyproject.toml         # uv package configuration
├── env.example            # environment template
├── run_pipeline.py        # CLI entry point
├── data/                  # generated datasets
├── outputs/               # visualizations and results
└── src/
    ├── config.py          # configuration (env vars, defaults)
    ├── dataset_collection.py  # Task 1: OSM data collection
    ├── features.py        # Tasks 2-3: semantic features
    ├── eda.py             # Task 4: EDA and visualizations
    ├── search.py          # Task 5: semantic search
    └── utils.py           # helpers
```

## Quick start

```bash
cp env.example .env        # create environment config
make install               # install dependencies with uv
make all                   # run the full pipeline (collect -> features -> analyze -> search)
```

Individual stages:

````bash
make collect    # Task 1: collect POI dataset from OpenStreetMap
make features   # Tasks 2-3: extract semantic features
Individual stages:

```bash
make collect    # Task 1: collect POI dataset from OpenStreetMap
make features   # Tasks 2-3: extract semantic features
make analyze    # Task 4: EDA and visualizations
make search     # Task 5: semantic search experiments
make report     # generate the Jupyter report notebook (reports/poi_report.ipynb)
````

## Data sources

The collector uses several layered sources to stay resilient:

1. **Nominatim** (OpenStreetMap geocoder) — primary source; searches for each
   POI category within the bounding box and returns real places with coordinates.
2. **Overpass API** — used as a fallback if Nominatim is unreachable.
3. **OSM API grid scan** — last-resort fallback (downloads small bbox chunks).
4. **Wikipedia** — short intro descriptions are attached to collected places
   when an article exists; category-based placeholders fill the rest.

Set `OFFLINE_MODE=true` in `.env` to skip the network entirely and use the
prepared `data/poi_dataset_raw.csv`.

## Configuration

All settings are read from environment variables (see `env.example` and `src/config.py`):

| Variable                | Description                                                    | Default                    |
| ----------------------- | -------------------------------------------------------------- | -------------------------- |
| `OSM_BOUNDING_BOX`      | Bounding box for OSM query (`min_lat,min_lon,max_lat,max_lon`) | central Moscow             |
| `OSM_POIS_PER_CATEGORY` | Max places per category                                        | 12                         |
| `OFFLINE_MODE`          | Use offline sample data instead of the network                 | `false`                    |
| `OFFLINE_CSV`           | Path to prepared dataset for offline mode                      | `data/poi_dataset_raw.csv` |
| `OPENAI_API_KEY`        | API key for optional LLM feature extraction (Task 3)           | unset                      |

## License

For internal/educational use.
