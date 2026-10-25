# Makefile for automating the POI recommendations pipeline
# Dependency management via uv

PYTHON      := uv run python
RUFF        := uv run ruff
DATA_DIR    := data
SRC_DIR     := src

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show the list of available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- installation
.PHONY: install
install: ## Install project dependencies with uv
	uv sync

.PHONY: install-dev
install-dev: ## Install dev dependencies (ruff, pytest)
	uv sync --group dev

.PHONY: install-llm
install-llm: ## Install deps for LLM feature extraction
	uv sync --extra llm

# ---------------------------------------------------------------- checks
.PHONY: lint
lint: ## Run the ruff linter
	$(RUFF) check $(SRC_DIR)

.PHONY: format
format: ## Autoformat code with ruff format
	$(RUFF) format $(SRC_DIR)

.PHONY: test
test: ## Run pytest tests
	$(PYTHON) -m pytest tests/ -v

# ---------------------------------------------------------------- pipeline
.PHONY: collect
collect: ## Task 1: Collect POI dataset from OSM
	$(PYTHON) run_pipeline.py --stage collect

.PHONY: features
features: ## Tasks 2-3: Extract semantic features
	$(PYTHON) run_pipeline.py --stage features

.PHONY: analyze
analyze: ## Task 4: EDA and visualizations
	$(PYTHON) run_pipeline.py --stage analyze

.PHONY: search
search: ## Task 5: Semantic search experiments
	$(PYTHON) run_pipeline.py --stage search

.PHONY: report
report: ## Generate the Jupyter report notebook
	$(PYTHON) scripts/generate_report.py

.PHONY: all
all: collect features analyze search ## Run the full pipeline

# ---------------------------------------------------------------- utilities
.PHONY: clean
clean: ## Remove generated data and artifacts
	rm -rf $(DATA_DIR) outputs/ .pytest_cache

.PHONY: env
env: ## Create .env from template (if missing)
	@test -f .env || cp env.example .env; echo ".env ready"


