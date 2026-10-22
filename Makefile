# ── The Catcher in the RAG — Makefile ────────────────────────────────────────
# Requires: uv, ruff, gcloud (ADC configured), terraform
# All Python commands run inside the uv-managed virtual environment.

PYTHON := uv run python
PYTEST := uv run pytest
RUFF   := uv run ruff

.PHONY: help setup install lint format test eval run-api run-ui clean tf-init tf-plan tf-apply

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────
setup: ## Create venv with Python 3.13 and install all dependencies
	uv venv --python 3.13
	uv sync --all-extras

install: ## Sync dependencies without recreating venv
	uv sync --all-extras

# ── Quality ───────────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	$(RUFF) check src/ tests/ datagen/ evals/ ui/

format: ## Run ruff formatter
	$(RUFF) format src/ tests/ datagen/ evals/ ui/
	$(RUFF) check --fix src/ tests/ datagen/ evals/ ui/
	@command -v terraform >/dev/null 2>&1 && terraform fmt -recursive infra/terraform/ || true

# ── Tests ─────────────────────────────────────────────────────────────────────
test: ## Run unit tests (stub provider, no LLM calls)
	$(PYTEST) tests/ -m "not real_llm"

test-all: ## Run all tests including real-LLM (requires ADC and billing)
	$(PYTEST) tests/

# ── Evaluation ────────────────────────────────────────────────────────────────
eval: ## Run full evaluation harness and produce versioned JSON report
	$(PYTHON) -m evals.run --output evals/reports/

eval-ablations: ## Run ablation suite (no-RAG, no-reranker, stub vs real)
	$(PYTHON) -m evals.run --ablations --output evals/reports/

# ── Run (local dev) ───────────────────────────────────────────────────────────
run-api: ## Start FastAPI dev server (port 8000)
	uv run uvicorn catcher_in_the_rag.api.main:app --reload --port 8000

run-ui: ## Start Streamlit UI (port 8501)
	uv run streamlit run ui/app.py --server.port 8501

# ── Terraform ─────────────────────────────────────────────────────────────────
tf-init: ## Initialize Terraform (requires GCS state bucket to exist)
	terraform -chdir=infra/terraform init

tf-plan: ## Show Terraform plan
	terraform -chdir=infra/terraform plan

tf-apply: ## Apply Terraform changes
	terraform -chdir=infra/terraform apply

tf-destroy: ## Destroy all Terraform-managed resources (CAREFUL)
	terraform -chdir=infra/terraform destroy

# ── Data ──────────────────────────────────────────────────────────────────────
download-data: ## Download CUAD and other datasets (see DATA_SOURCES.md)
	$(PYTHON) -m datagen.download

generate-synthetic: ## Generate synthetic procurement packages
	$(PYTHON) -m datagen.generate --seed 42 --count 300

# ── Clean ─────────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean done."
