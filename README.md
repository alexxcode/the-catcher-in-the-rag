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
 [1 ingest - planned] → [2 classify - planned] → [3 extract (schema-constrained / LLM)]
                                      │
                         [4 link PO / receipt by id]
                                      │
                         [5 match (deterministic, tolerances)]
                                      │
              [6 retrieve contract terms (BM25 + dense hybrid, scoped to vendor)]
                                      │
              [6b bartleby: abstention gate → NO_EVIDENCE or continue]
                                      │
              [7 apply terms (evaluates structured contract terms & rates)]
                                      │
              [8 validate (quote exists, numbers recomputed) → decide (policy)]
                                      │
              approve | human review (with evidence) | reject
```

Orchestration: **LangGraph** state machine · **Gemini** via Vertex AI (ADC credentials, `providers/gemini.py`) with offline deterministic stub · **pgvector** on Cloud SQL · **Langfuse** for traces · **Cloud Run** for hosting.

---

## Benchmark Results (300 Synthetic Packages)

Evaluated against a reproducible corpus of 300 procurement packages covering 10 anomaly categories (A01–A10).

| Metric | Baseline RAG | Ablation (No-RAG) | Delta / Impact |
|---|---|---|---|
| **Overall Decision Accuracy** | **100.0%** (300/300) | **60.0%** (180/300) | **+40.0% Valor de Términos Contractuales Estructurados** |
| **Citation Validity Rate** | **100.0%** (120/120) | 100.0% (30/30) | Strict offset & verbatim verification |
| **Bartleby Correct Abstention** | **100.0%** (30/30 on A09) | 0.0% (0/30 on A09) | Zero ungrounded interpretations |
| **Clean Package (A10) FP Rate** | **0.0%** (0/30) | 0.0% (0/30) | Clean packages pass cleanly |
| **Retrieval Recall@1 / Recall@5** | **100.0%** / **100.0%** | N/A | Hybrid BM25 + dense retrieval |
| **Mean Latency per Package** | **9.9 ms** | 1.1 ms | Fully deterministic offline execution |

### Anomaly Taxonomy Coverage (A01–A10)

| ID | Anomaly Description | Layer | Baseline Precision | Baseline Recall | Baseline F1 / FPR |
|---|---|---|---|---|---|
| **A01** | Unit price above contract rate ceiling | Match + Clause | 100.0% | 100.0% | 1.000 |
| **A02** | Quantity billed > quantity received | 3-Way Match | 100.0% | 100.0% | 1.000 |
| **A03** | Duplicate invoice number detected | 3-Way Match | 100.0% | 100.0% | 1.000 |
| **A04** | Payment terms mismatch against contract | Contract Clause | 100.0% | 100.0% | 1.000 |
| **A05** | Late-delivery penalty applicable | Contract Clause | 100.0% | 100.0% | 1.000 |
| **A06** | Expired contract / auto-renewal notice | Contract Clause | 100.0% | 100.0% | 1.000 |
| **A07** | Unknown vendor or unlinked PO | 3-Way Match | 100.0% | 100.0% | 1.000 |
| **A08** | Total amount exceeds $50,000 threshold | Policy Engine | 100.0% | 100.0% | 1.000 |
| **A09** | Governing clause absent (`bartleby` guard) | Abstention Gate | 100.0% | 100.0% | 1.000 |
| **A10** | Clean package (false-positive control) | Full Pipeline | N/A (Control) | N/A (Control) | FPR: 0.0% |

---

## Interactive Demo & UI

The web application provides two distinct operational screens via Streamlit (`make run-ui`):

1. **Review Workspace (`ui/app.py`)**:
   - **Interactive 4-Way Document Inspection**: Side-by-side tabs for Invoice, Purchase Order, Goods Receipt, and Contract.
   - **Verbatim Citations**: Highlights extracted evidence alongside exact character offsets and document IDs.
   - **Bartleby Melville Guard Banner**: Highlights when and why the agent declined to guess when contractual terms were silent.
   - **Human-in-the-Loop Sign-Off**: Auditor notes and decision confirmation/override.

2. **Evaluation Dashboard (`ui/app.py`)**:
   - **Live Metric Tracking**: Overall decision accuracy, citation validity rate, clean package FP rate, and latency.
   - **Ablation Comparison**: Side-by-side visualization of Baseline RAG vs. No-RAG ablation (+44% accuracy lift).
   - **Failure Case Inspector**: Full transparency into diagnostic failure packages and policy engine rationale.

---

## Quick start

### Prerequisites

- Python 3.13+, [uv](https://github.com/astral-sh/uv), Docker & Docker Compose
- GCP project with billing, `gcloud` authenticated via ADC (for real LLM mode)

### Local Dev Setup

```bash
# Clone
git clone https://github.com/alexxcode/the-catcher-in-the-rag.git
cd the-catcher-in-the-rag

# Install deps (creates .venv with Python 3.13)
make setup

# Generate synthetic evaluation packages (300 packages across A01-A10)
make generate-synthetic

# Run tests (offline, stub provider — 40+ tests, 80%+ coverage)
make test

# Run the full evaluation harness and ablation suite
make eval-ablations

# Run field-by-field unstructured contract extraction benchmark (Phase 2)
make eval-extraction

# Run model extraction bake-off (Flash vs Pro vs Baseline) (Phase 3)
make eval-bakeoff

# Start the interactive Streamlit UI
make run-ui

# Or start the FastAPI server
make run-api
```

### Docker Compose (One Command)

```bash
docker compose up --build
# API runs at http://localhost:8000
# Streamlit UI runs at http://localhost:8501
```

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
