# The Catcher in the RAG — Technical Spec

*Tagline: catches the invoice before it falls off the cliff.*
Repo: `catcher-in-the-rag` · Python package: `catcher_in_the_rag` · Abstention guard: `bartleby`

> Audit agent for procurement documents: checks invoices against purchase orders **and against the contract clauses that govern them**, cites the exact clause, and is measured with a reproducible evaluation harness.
> Audience: the maintainer and any coding agent working in this repo. Read sections 3, 5 and 11 before writing code.

Status: draft v0.4 
---

## 1. Why this exists

Three-way matching (PO ↔ goods receipt ↔ invoice) is already a commodity in open-source demos. What is rarely shown is an agent that (a) retrieves the *contractual terms* that apply to a given invoice (rates, payment terms, penalties, renewals), (b) justifies each finding with a verifiable citation, (c) abstains when the contract is silent, and (d) is **evaluated** against injected, labeled errors.

The Catcher in the RAG is a portfolio-grade, end-to-end pipeline: ingestion → extraction → matching → contract RAG → decision → human review → evaluation dashboard. The evaluation harness is the core deliverable, not the chat UI.

## 2. Magnitude (targets, to be calibrated after the first baseline)

| Dimension | Target |
|---|---|
| Real contracts indexed (CUAD) | ~510 |
| Synthetic procurement packages (contract + PO + receipt + invoice) | 300–500 |
| Injected anomaly classes | 10 (incl. clean and "clause absent") |
| Graph nodes in the agent | ~8 |
| Evaluation metrics tracked | 8 families (section 8) |
| Demo screens | 2 (review workspace, evaluation dashboard) |
| Estimated effort | 10–14 working days |

## 3. Non-negotiable principles

1. **LLM extracts and explains; deterministic code decides.** Approve/review/reject comes from a policy engine, never from free-form LLM judgment.
2. **Every finding carries a citation** (document id, span offsets, verbatim quote). A validator drops any finding whose quote is not found verbatim in the source.
3. **Abstain over guess.** If no relevant clause is retrieved, the output is `NO_EVIDENCE`, not an invented rule. This is enforced by a dedicated component, `bartleby` (see section 5), not left to prompt wording.
4. **No custom model training.** Only API/off-the-shelf models (LLM, embeddings).
5. **Reproducible.** Seeded generators, pinned dependency versions, pinned dataset versions, a one-command eval run.
6. **Offline mode must work.** A deterministic stub provider runs the whole pipeline without LLM calls, so CI and demos never depend on network or cost.
7. **No secrets in the repo.** On GCP, authenticate with service identities (Application Default Credentials); anything else lives in Secret Manager or environment variables.
8. **Provider-agnostic core.** Gemini is the default LLM, but code depends only on the interfaces in `providers/`. Model IDs live in config, never in code (model versions are retired on a fixed schedule).

## 4. Scope

**In:** structured extraction with LLM, document classification, deterministic 3-way matching, contract RAG with citations, policy engine, human-in-the-loop review queue, synthetic data generator with error injection, evaluation harness, tracing, 2-screen demo UI.

**Out (do not build):** OCR or any trained/fine-tuned model, real ERP integration, payments, multi-tenant auth, production hardening, non-English contracts (possible later extension, flagged synthetic).

## 5. Architecture

```
 PDF/JSON package
      │
 [1 ingest] → [2 classify] → [3 extract (LLM, schema-constrained)]
                                      │
                         [4 link PO / receipt by id]
                                      │
                         [5 match (deterministic, tolerances)]
                                      │
              [6 retrieve contract terms (hybrid, scoped to vendor contract)]
                                      │
              [6b bartleby: abstention gate → NO_EVIDENCE or continue]
                                      │
              [7 apply terms (LLM proposes finding + verbatim quote)]
                                      │
              [8 validate (quote exists, numbers recomputed) → decide (policy engine)]
                                      │
              approve | human review (with evidence) | reject
```

**`bartleby` (abstention guard).** Named after Melville's scrivener who answers "I would prefer not to": when the contract is silent, the agent declines to invent a rule. It runs between retrieval and clause application, and again after validation. It returns `NO_EVIDENCE` (with a machine-readable reason) when any of these hold: no retrieved span passes the relevance threshold; the proposed finding has no verbatim supporting quote; or the cited span does not actually address the invoice field in question. Thresholds are deterministic config, calibrated on A09/A10 packages, and every abstention is logged and counted in the evaluation report. `bartleby` must work without any LLM call (it consumes retrieval scores and validator results).

Orchestration: LangGraph state machine with typed state and an interrupt for human review. Every node emits a trace span. Nodes are pure functions over typed state wherever possible, so they are unit-testable without an LLM.

## 6. Data

| Source | Role | License (verified at source) |
|---|---|---|
| CUAD v1 (510 contracts, 41 clause types) | Contract corpus + retrieval ground truth | CC BY 4.0 |
| ContractNLI | Optional: contradiction/entailment test cases | CC BY 4.0 (secondary source; re-verify) |
| LegalBench-RAG (CUAD/ContractNLI subsets only) | Retrieval benchmark comparison | Repo MIT; underlying corpora keep own licenses |
| FATURA (10k invoice images, 50 layouts) | Extraction robustness set | CC BY 4.0 |
| OCDS public procurement data (1–2 chosen publishers) | Realistic vendors, categories, amounts for synthetic POs | Per publisher: record exact license in `DATA_SOURCES.md` |
| Synthetic packages (own generator) | Labeled anomalies, main eval set | Own; publish CC BY 4.0 or MIT, mark as synthetic. Generate only with GA (non-preview) models: preview terms restrict commercial use and disclosing generated output |

**Do not use:** datasets under non-commercial terms (e.g. CC BY-NC), or aggregates with mixed/unclear licenses. **Do not re-host** full third-party corpora: ship download scripts and record URL, version, date and license in `DATA_SOURCES.md`; credit everything in `ATTRIBUTION.md`.

## 7. Anomaly taxonomy (ground-truth labels)

| ID | Anomaly | Detected by |
|---|---|---|
| A01 | Unit price above contract rate | Match + clause |
| A02 | Quantity billed > received | Match |
| A03 | Duplicate invoice | Match |
| A04 | Payment terms differ from contract | Clause |
| A05 | Late-delivery penalty applicable | Clause |
| A06 | Contract expired / auto-renewal triggered | Clause |
| A07 | Unknown vendor or no PO | Match |
| A08 | Amount above approval threshold | Policy |
| A09 | Relevant clause absent → must abstain | Clause (negative) |
| A10 | Clean package (false-positive control) | All |

Each synthetic package stores its labels in a sidecar `labels.json` (anomaly ids, expected clause span, expected decision).

## 8. Evaluation (the main deliverable)

Metrics: extraction field accuracy · retrieval recall@k and MRR against labeled spans · citation validity rate · faithfulness · per-class precision/recall for A01–A10 · correct-abstention rate of `bartleby` (A09), and its over-abstention rate on A01–A08 and A10 (abstaining when evidence existed) · false-positive rate on A10 · cost and latency per package.

Required ablations: no-RAG vs RAG; stub vs real LLM.

Rules: eval is one command (`make eval`), writes a versioned JSON report, and the dashboard reads only that report. Failure cases are shown, not hidden. Initial acceptance targets are set **after** the first baseline run, not before.

## 9. Stack

Python 3.11+ · LangGraph (agent) · Pydantic (typed schemas, structured output) · FastAPI (API) · BM25 + dense hybrid retrieval · Ragas (RAG metrics, supplementary) · pytest · ruff · Docker Compose (local) · GitHub Actions.

**Cloud: everything on GCP (decided).**

| Concern | Choice |
|---|---|
| LLM | Gemini through Google Cloud's generative AI platform (formerly Vertex AI), authenticated with ADC, no API keys |
| Embeddings | A Gemini embedding model, pinned in config (re-embed the corpus if it changes) |
| Hosting (API + UI) | Cloud Run, scale to zero |
| Documents and reports | Cloud Storage |
| Vectors and app state | Default: PostgreSQL with pgvector on Cloud SQL (one DB for vectors, app state and the LangGraph checkpointer). Alternative: Vertex AI Vector Search |
| Secrets, images, CI auth | Secret Manager, Artifact Registry, GitHub Actions with Workload Identity Federation |
| Observability | Langfuse for LLM traces (the one deliberate exception to "everything on GCP"); Cloud Logging for infrastructure logs |
| Cost control | Budget alerts on the project; cost per run recorded in every eval report; cache LLM calls during development |

**Tracing rules.** Default is Langfuse Cloud on the free Hobby tier (at the time of writing: 50k units/month, 30-day data access, 2 users; a unit is a trace, observation or score; re-verify before relying on it). Call it only through a tracing interface in `providers/`, with a no-op implementation for tests and offline mode. Bulk evaluation runs trace a sample, not every package, to stay under the quota. Langfuse is for debugging and demos; the **source of truth for metrics is the versioned JSON report**, never Langfuse (retention is limited). Only synthetic or public data may reach traces.

**Model rules.** Pin exact model IDs in `config/models.yaml` (extraction, clause application, embeddings). Check the platform's model lifecycle page before pinning: models have fixed retirement dates and several Gemini 2.5 models were scheduled to retire in October 2026. Use a cheaper Flash-tier model for extraction and a stronger one only if the ablation justifies it. Use GA models only for dataset generation and published results.

## 10. Repository layout

```
catcher-in-the-rag/
  SPEC.md  README.md  DATA_SOURCES.md  ATTRIBUTION.md  Makefile
  src/catcher_in_the_rag/
    schemas/      # Pydantic models: Contract, PO, Receipt, Invoice, Finding, Decision
    ingest/ extract/ match/ retrieve/ bartleby/ apply_terms/ validate/ policy/
    graph.py      # LangGraph wiring
    providers/    # llm + embedding interfaces, real and stub
    api/
  datagen/        # seeded synthetic generator + error injector
  config/         # models.yaml (pinned model IDs), thresholds, policy tolerances
  infra/          # Dockerfiles, Cloud Run and CI definitions (IaC optional)
  evals/          # runners, metrics, report writer
  ui/
  tests/
  data/           # gitignored raw; small fixtures committed
```

## 11. Rules for coding agents

- Implement against `schemas/` first; change a schema only with a note in the PR description.
- Every node needs a unit test that runs with the stub provider.
- Never call a real LLM in tests. Real-model runs live under `evals/` and are opt-in.
- Do not add dependencies without pinning the version and noting the license.
- Never hardcode a model ID, project ID, region or bucket name: read them from config or environment.
- Never use service-account key files; use ADC locally (`gcloud auth application-default login`) and the attached service identity on Cloud Run.
- Never fabricate citations, metrics, or dataset facts in code, docs or the README. Unknown means `TODO`.
- Never commit third-party corpora, keys, or generated reports with secrets.
- Naming: the only literary name in code is the `bartleby` module. Every other module, class and field stays descriptive (`retrieve`, `Finding`, `Decision`).
- IP: do not reproduce text, quotes, or imagery from *The Catcher in the Rye* in code, docs, UI or tests; the title is used only as a parody project name. Short quotes from *Bartleby, the Scrivener* (Melville, 1853, public domain) are fine.
- Definition of done for any task: tests pass, `ruff` clean, `make eval` still runs, docs updated.

## 12. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M1 | Schemas + synthetic generator + labels | 300+ packages generated deterministically from a seed |
| M2 | Extraction + matching + policy (stub provider) | A01–A03, A07, A08 detected end to end offline |
| M3 | Contract RAG + citation validator + `bartleby` | Recall@k reported; A04–A06 working; `bartleby` returns `NO_EVIDENCE` on A09 with no LLM call |
| M4 | Eval harness + report | `make eval` produces the full metric report and ablations |
| M5 | UI (review workspace + dashboard) + README | Demo runs from `docker compose up`; failure cases visible |

## 13. Open decisions (resolve before M2)

1. ~~LLM provider~~ Decided: Gemini on GCP. Still open: exact model IDs (extraction, clause application) and embedding model, chosen from the lifecycle page after a small bake-off.
2. Vector store: pgvector on Cloud SQL (default) vs Vertex AI Vector Search. 
3. UI: Streamlit (faster) vs React (more polished). Both deploy on Cloud Run.
4. Which OCDS publisher(s) to use, and their exact license.
5. Whether to add a Spanish-language variant as a flagged-synthetic extension after M5.
6. ~~Tracing~~ Decided: Langfuse. Still open: stay on the free Hobby tier or move to Core if the quota is exceeded; apply to Langfuse's discount program for students and open-source projects.
7. Tone: literary humor only in the name and tagline, or also in the UI and agent messages (default until decided: name and tagline only).
