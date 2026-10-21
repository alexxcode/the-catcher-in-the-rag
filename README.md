# The Catcher in the RAG

> *catches the invoice before it falls off the cliff.*

**Audit agent for procurement documents** — checks invoices against purchase orders *and* against the contract clauses that govern them, cites the exact clause, and is measured with a reproducible evaluation harness.

[![CI](https://github.com/alexxcode/the-catcher-in-the-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/alexxcode/the-catcher-in-the-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![GCP](https://img.shields.io/badge/cloud-GCP-4285F4)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What makes this different

Three-way matching (PO ↔ goods receipt ↔ invoice) is a commodity. What is rarely shown is an agent that:

- (a) retrieves the **contractual terms** that apply to a given invoice
- (b) justifies each finding with a **verifiable citation** (document id, span offsets, verbatim quote)
- (c) **abstains** when the contract is silent (`bartleby` component, no LLM call)
- (d) is **evaluated** against injected, labeled errors with a reproducible harness

The evaluation harness is the core deliverable, not the chat UI.

---

## Architecture

```
PDF/JSON package
      │
 [1 ingest] → [2 classify] → [3 extract (LLM, schema-constrained)]
                                      │
                         [4 link PO / receipt by id]
                                      │
                         [5 match (deterministic, tolerances)]
                                      │
              [6 retrieve contract terms (hybrid + rerank, scoped to vendor)]
                                      │
              [6b bartleby: abstention gate → NO_EVIDENCE or continue]
                                      │
              [7 apply terms (LLM proposes finding + verbatim quote)]
                                      │
              [8 validate (quote exists, numbers recomputed) → decide (policy)]
                                      │
              approve | human review (with evidence) | reject
```

Orchestration: **LangGraph** state machine · **Gemini** via Vertex AI (ADC, no API keys) · **pgvector** on Cloud SQL · **Langfuse** for traces · **Cloud Run** for hosting.

---

## Quick start

### Prerequisites

- Python 3.13+, [uv](https://github.com/astral-sh/uv), Docker
- GCP project with billing, `gcloud` authenticated via ADC
- Cloud SQL instance with pgvector extension (see `infra/terraform/`)
- Langfuse Hobby account (see `SPEC.md §9`)

### Setup

```bash
# Clone
git clone https://github.com/alexxcode/the-catcher-in-the-rag.git
cd the-catcher-in-the-rag

# Install deps (creates .venv with Python 3.13)
make setup

# Copy and edit environment config
cp .env.example .env
# Edit .env with your GCP project, Cloud SQL connection, Langfuse keys

# Download datasets (see DATA_SOURCES.md)
make download-data

# Generate synthetic evaluation packages
make generate-synthetic

# Run tests (offline, stub provider — no LLM calls)
make test

# Run the full evaluation harness
make eval

# Start the review UI
make run-ui
```

---

## Evaluation

```bash
make eval          # full run → evals/reports/<timestamp>.json
make eval-ablations  # no-RAG vs RAG, with/without reranker, stub vs real LLM
```

The report is the **source of truth for metrics**. Langfuse is for debugging (limited retention).

---

## Repository layout

```
catcher-in-the-rag/
  SPEC.md  README.md  DATA_SOURCES.md  ATTRIBUTION.md  Makefile
  src/catcher_in_the_rag/
    schemas/      # Pydantic models: Contract, PO, Receipt, Invoice, Finding, Decision
    ingest/ extract/ match/ retrieve/ bartleby/ apply_terms/ validate/ policy/
    graph.py      # LangGraph wiring
    providers/    # LLM + embedding interfaces: real (Gemini/Vertex) and stub
    api/          # FastAPI app
  datagen/        # Seeded synthetic generator + error injector
  config/         # models.yaml, thresholds.yaml, policy.yaml
  infra/terraform/  # GCP resources (Cloud SQL, Cloud Run, GCS, Secret Manager…)
  evals/          # Runners, metrics, report writer
  ui/             # Streamlit: review workspace + evaluation dashboard
  tests/
  data/           # gitignored raw; small fixtures committed
```

---

## Dataset sources

See [`DATA_SOURCES.md`](DATA_SOURCES.md) — download scripts are in `datagen/download.py`.  
Attribution: [`ATTRIBUTION.md`](ATTRIBUTION.md).

---

## Stack

Python 3.13 · LangGraph · Pydantic · FastAPI · Streamlit · BM25 + dense hybrid retrieval · Ragas · Langfuse · GCP (Vertex AI, Cloud Run, Cloud SQL + pgvector, Cloud Storage, Secret Manager) · Terraform · GitHub Actions · ruff · pytest

---

## License

MIT — see [LICENSE](LICENSE).  
**Note:** *The Catcher in the Rye* title is used only as a parody project name. No text, quotes, or imagery from that work appears in code, docs, or UI.  
Short quotes from *Bartleby, the Scrivener* (Melville, 1853, public domain) are used for the `bartleby` component.
