# The Catcher in the RAG

**Catches the invoice before it falls off the cliff.**

[![CI](https://github.com/alexxcode/the-catcher-in-the-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/alexxcode/the-catcher-in-the-rag/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-live%20(staging%2C%20stub)-0071E3)](https://catcher-ui-staging-hrok623gcq-uc.a.run.app)

![Animated walkthrough of the Review Workspace: a package is selected, Run Audit Agent is pressed, the pipeline steps appear with their measured times, and the verdict card and verbatim citations render below](docs/img/demo.gif)

## The idea

In Salinger's novel, Holden pictures himself standing at the edge of a cliff beside a rye field, catching the kids before they run over. This project takes the same post in accounts payable: it stands between the invoice and the payment and catches the ones about to go over the edge — a unit price above the contracted rate, a duplicate, a quantity that was billed but never received.

It follows three rules of conduct:

1. **Check everything against the contract.** A three-way match (purchase order, goods receipt, invoice) plus contract clauses retrieved with RAG.
2. **Cite or stay quiet.** Every clause-backed finding carries a verbatim excerpt that is validated as an exact substring of the contract, with character offsets. If the quote isn't in the source, the finding is dropped.
3. **When there is no evidence, prefer not to.** Bartleby, the abstention guard, is named after Melville's scrivener and his "I would prefer not to" (public domain). No relevant clause, no verdict: the package goes to a human.

| Caught: a price over the contract ceiling | Declined: no clause supports the charge |
|:---:|:---:|
| ![Review Workspace showing a REJECT card for a unit price above the contract ceiling, with the verbatim clause citation, document id and character offsets underneath](docs/img/reject.png) | ![Review Workspace showing an ABSTAIN card: no contract clause for an expedited rush surcharge met the relevance threshold, and the Bartleby abstention banner explains the reason](docs/img/abstain.png) |

## Four possible verdicts

| APPROVE | REVIEW | REJECT | ABSTAIN |
|:---:|:---:|:---:|:---:|
| ![Green APPROVE card: the package passed every check and is within the auto-approval threshold](docs/img/approve.png) | ![Amber REVIEW card: a payment-terms finding that could not be verified against a contract clause, sent to a human](docs/img/review.png) | ![Red REJECT card: a hard policy violation with its verbatim contract citation](docs/img/reject.png) | ![Violet ABSTAIN card: Bartleby found no clause relevant enough to judge the charge](docs/img/abstain.png) |
| Every check passed and the total is within the auto-approval limit. Pay it. | Something needs a person: a soft finding, or one the contract could not confirm. | A hard violation backed by a verbatim clause. Don't pay it. | No clause is relevant enough to judge. Bartleby declines to guess and hands it to a human. |

## How it works

The pipeline is a LangGraph state machine (`src/catcher_in_the_rag/graph.py`). The nodes run in this order:

```mermaid
flowchart LR
    A([package]) --> E[extract]
    E --> M[match<br/>three-way match]
    M --> T
    subgraph T[apply_terms]
        R[retrieve clauses<br/>BM25 + dense] --> B{Bartleby}
    end
    T --> V[validate<br/>verbatim citations]
    V --> D{decide<br/>policy engine}
    D --> AP([APPROVE])
    D --> RV([REVIEW])
    D --> RJ([REJECT])
    D --> AB([ABSTAIN])
```

- **extract** turns the package into typed documents and pulls the governing terms out of the contract text.
- **match** runs the deterministic three-way match.
- **apply_terms** retrieves the relevant clauses and passes them through Bartleby before any clause-backed finding is produced.
- **validate** drops any finding whose citation is not an exact substring of the contract.
- **decide** applies the policy rules and returns one of the four verdicts.

## The RAG, in plain words

The contract is split into clauses. For each question the audit needs to answer ("what is the price ceiling for this SKU?"), clauses are ranked by a hybrid retriever: BM25 for exact words and dense embeddings (`text-embedding-005` on Vertex AI) for meaning. The top passages go through Bartleby before anything is generated: if none is relevant enough, the answer is an abstention, not a guess. When a finding does cite a clause, its offsets point to the exact characters in the source text, so anyone can check the quote.

By default everything runs offline with deterministic stubs (`MAX_LIVE_CALLS=0`): no network calls, the same answer every time. Vertex AI and Gemini are opt-in, and every live call counts against a hard cap.

![CUAD explorer in the UI: a real SEC contract with its annotated clause categories, showing a category marked absent that Bartleby must decline to assert](docs/img/cuad.png)

## Observability (Langfuse)

![Langfuse graph view of one audit: the audit_pipeline span with its child spans extract, match, apply_terms, validate and decide, next to the trace's input and output (decision, justification, findings count)](docs/img/langfuse-graph.png)

Each audit produces one trace: an `audit_pipeline` span with one child span per graph node, timed as the node actually runs. The span metadata carries the decision, the findings count and τ. In the Graph view, Langfuse draws `extract` and `match` side by side because they are sibling spans; in `graph.py` they run one after the other.

Tracing is optional and switches on when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set; without them the tracer is a no-op. The screenshots come from local runs in stub mode. The deployed demo runs without traces.

![Langfuse tree view of one audit_pipeline trace with its five child spans and their durations](docs/img/langfuse-trace-detail.png)

![Langfuse trace table filtered to audit_pipeline: one row per audit with its package, decision and latency metadata](docs/img/langfuse-traces.png)

## Results

Every number below lives in a `GEN` block regenerated from `evals/reports/*.json` (`make sync-readme-tables`). Nothing is typed by hand.

### 1. Out-of-sample retrieval
`MAX_LIVE_CALLS=20 uv run python evals/eval_out_of_sample.py --retrieval-ablation --embedding-provider vertex_ai --start 15 --end 50` → `evals/reports/retrieval_ablation_oos_report.json`

Dense retrieval does the heavy lifting: BM25 alone trails, and the hybrid is indistinguishable from dense on this sample.

<!-- GEN:retrieval_table -->
156 presence cases, CUAD-0016 to CUAD-0050 (35 contracts), embeddings `text-embedding-005`, commit `8a68fe2`. Ties broken by order of appearance in the contract.

| Method | Hit@1 | Hit@5 |
|---|:---:|:---:|
| **BM25** | 30.1% (47/156) [CI95 23.5–37.7] | 68.6% (107/156) [CI95 60.9–75.3] |
| **Dense** | 55.8% (87/156) [CI95 47.9–63.3] | 85.9% (134/156) [CI95 79.6–90.5] |
| **Hybrid (50/50)** | 55.1% (86/156) [CI95 47.3–62.7] | 84.6% (132/156) [CI95 78.1–89.4] |

- Hit@1: the Wilson interval of BM25 does not overlap the others; the three methods are not indistinguishable on this sample. Dense and Hybrid (50/50): indistinguishable on this sample.
- Hit@5: the Wilson interval of BM25 does not overlap the others; the three methods are not indistinguishable on this sample. Dense and Hybrid (50/50): indistinguishable on this sample.
<!-- /GEN -->

### 2. Bartleby's abstention, out of sample
`MAX_LIVE_CALLS=3 uv run python evals/eval_out_of_sample.py --start 15 --end 50 --embedding-provider vertex_ai --llm-provider stub --tau <τ>`

A higher τ buys correct abstentions on absent clauses and pays for them with false abstentions on present ones. The LLM row is not directly comparable with the others (see the note).

<!-- GEN:oos_bartleby_table -->
| Configuration | $\tau$ | Embeddings | Correct abstention (absences) | False abstention (presences) | Macro F1 | Report |
|---|:---:|---|:---:|:---:|:---:|---|
| No LLM, score only | 0.50 | `vertex_ai` | 63/264 = 23.9% [CI95 19.1–29.4] | 8/156 = 5.1% [CI95 2.6–9.8] | 0.4811 | `out_of_sample_cuad_stub_tau0.50_report.json` |
| No LLM, score only | 0.80 | `vertex_ai` | 221/264 = 83.7% [CI95 78.8–87.7] | 72/156 = 46.2% [CI95 38.5–54.0] | 0.6936 | `out_of_sample_cuad_stub_tau0.80_report.json` |
| With LLM verification (3b) | 0.50 (fixed in the eval) | `stub (command default)` | 248/264 = 93.9% [CI95 90.4–96.2] | 74/156 = 47.4% [CI95 39.8–55.2] | 0.7460 | `out_of_sample_cuad_report.json` (git_dirty) |
| Always abstain (trivial) | — | — | 264/264 = 100.0% | 156/156 = 100.0% | 0.3860 | computed from the counts |
<!-- /GEN -->
<!-- GEN:operating_point -->
Without the LLM, $\tau$ = 0.80 (current value in `config/thresholds.yaml`) abstains on 83.7% of absences (221/264) and on 46.2% of presences (72/156). $\tau$ = 0.50 almost never abstains: 71 of 420 queries (16.9%), and on only 23.9% of absences.

> **Note:** The LLM row is not comparable with the rows without the LLM: it used `stub` embeddings and it was generated with `git_dirty: true`.
<!-- /GEN -->
<!-- GEN:mode3_diagnosis -->
> **Bartleby: three measured claims**
> (a) **The retrieval score alone is a weak signal.** Validation AUC-ROC 0.5909 [CI95 0.3885–0.7933] with `text-embedding-005` (CUAD, contracts 1–15, `calibration_cuad_report.json`); the interval includes 0.50.
>
> (b) **LLM verification supplies the abstention.** With the same embeddings and $\tau$ = 0.50 (`bartleby_llm_verification_report.json`), correct abstention on absences goes from 22/79 (27.9%) without the LLM to 50/79 (63.3%) with it.
>
> (c) **False abstentions remain high.** With the LLM, out of sample: 74/156 = 47.4% [CI95 39.8–55.2] of present clauses (`out_of_sample_cuad_report.json`). Each false abstention is an unnecessary human review: 74 in 420 queries (17.6 per 100). Time or cost per review: NOT MEASURED.
<!-- /GEN -->

### 3. LLM verification (mode 3)
`make eval-bartleby-llm` → `evals/reports/bartleby_llm_verification_report.json`

Measured with τ fixed in the eval and whatever embeddings that run's environment used (the report does not record them). LLM verification adds most of the correct abstentions, and false abstentions, latency and cost rise with it.

<!-- GEN:llm_verification_table -->
| Guard mode | $\tau$ | LLM verification | Correct abstention (recall) | False abstention (error) | Macro F1 | Mean latency | Estimated cost / 1k* |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pure Vector Hybrid (tau=0.50)** | **0.50** | `Off` | **27.9% (22/79)** | **0.0% (0/26)** | **0.4564** | 17.1 ms | $0.0000 |
| **Pure Vector Hybrid (tau=0.70)** | **0.70** | `Off` | **40.5% (32/79)** | **26.9% (7/26)** | **0.4777** | 16.9 ms | $0.0000 |
| **Bartleby Condition 3 (tau=0.50 + LLM Verification)** | **0.50** | `Gemini 3.5 Flash` | **63.3% (50/79)** | **42.3% (11/26)** | **0.5714** | 3303.5 ms | $2.3367 |
<!-- /GEN -->
<!-- GEN:llm_verification_footnote -->
*\*Cost estimated from measured tokens (98,140 prompt / 10,905 candidate, with thinking tokens counted as output) and official pay-as-you-go Vertex AI rates ($1.50 / $9.00 per million tokens for Gemini 3.5 Flash, per [`config/pricing.yaml`](config/pricing.yaml)). With Gemini 3.8 Flash the estimated cost is ~$0.88/1k at the current introductory rate ($0.75 / $3.75 until December 31, 2026) and ~$1.77/1k at the standard rate ($1.50 / $7.50 from January 1, 2027).*
<!-- /GEN -->

### 4. Calibrating τ
`MAX_LIVE_CALLS=80 uv run python -m evals.calibrate_cuad --embedding-provider vertex_ai` → `evals/reports/calibration_cuad_report.json`

On real contracts the retrieval score barely separates present clauses from absent ones: the validation interval includes chance.

<!-- GEN:domain_calibration_table -->
| Domain | Source dataset | Calibrated threshold ($\tau$) | Validation Macro F1 | AUC-ROC (separability) | Empirical transfer performance |
|---|---|:---:|:---:|:---:|---|
| **Synthetic templates (+ Rule 3a heuristic*)** | Datagen M1 (300 packages; not evidence) | **0.50** | **1.0000** | **0.1204** | F1 with the 3a denial regex: 1.0000. Pure score without 3a: F1 0.4737 (abstention 0.0%). |
| **Real SEC contracts (transfer)** | CUAD v1, contracts 1–15, train/val 60/40 | **0.50** (fixed) | **0.4496** | **0.5909** | Correct abstention on absences 25.8% (74.2% pass through). |
| **Real SEC contracts (recalibrated)** | CUAD v1, contracts 1–15, train/val 60/40 | **0.80** (recalibrated, not applied) | **0.5457** | **0.5909** | Correct abstention 54.8%, false abstentions 36.4% (Always abstain: 0.4247). |
<!-- /GEN -->
<!-- GEN:domain_calibration_diagnosis -->
> **CUAD calibration diagnosis** (`evals/reports/calibration_cuad_report.json`)
> Synthetic (reference, not evidence): $\tau = 0.50$, validation Macro F1 1.0000, AUC-ROC 0.1204.
> CUAD: 15 contracts (train 9, validation 6, seed 42), embeddings `text-embedding-005`.
>
> 1. **AUC-ROC (Wilcoxon-Mann-Whitney):** train 0.7708 [CI95 0.6202–0.9214]; validation 0.5909 [CI95 0.3885–0.7933]; the validation interval includes 0.50: indistinguishable from chance on this sample.
> 2. **Threshold:** at $\tau = 0.50$, correct abstention on validation absences is 25.8%. At $\tau = 0.80$ (train optimum) it is 54.8%, with 36.4% false abstentions and Macro F1 0.5457 (Always abstain: 0.4247; difference +0.1210).
<!-- /GEN -->

### 5. Synthetic control — not evidence
`LLM_PROVIDER=stub EMBEDDING_PROVIDER=stub MAX_LIVE_CALLS=0 make eval-ablations` → `evals/reports/canonical_baseline_report.json`

Templates written by `datagen/`, measured with the τ in `config/thresholds.yaml`. It checks that the wiring works, nothing more.

<!-- GEN:synthetic_ablation_table -->
| Metric | With contract terms (baseline) | Without terms (three-way match only) | Impact / delta |
|---|:---:|:---:|:---:|
| **Overall decision accuracy** | **91.3%** (274/300) | **60.0%** (180/300) | **ablation: no clause evaluator (+31.3% degradation)** |
| **Verbatim citation validity** | **100.0%** (64/64) [synthetic templates] | N/A | Zero invalid synthetic citations |
| **Bartleby correct abstention** | **100.0%** (30/30 on A09*) | 0.0% (0/30 on A09) | Silence detection (via the 3a template heuristic*) |
| **False positive rate (A10)** | **0.0%** (0/30) | 0.0% (0/30) | Clean packages approved without friction |
| **Retrieval Recall@5** | **100.0%** [synthetic templates] | N/A | Retrieval on synthetic templates |
| **Mean latency per package** | **16.9 ms** | 0.0 ms | Local deterministic execution |
<!-- /GEN -->

## Honest limitations

- **Synthetic is not evidence.** The synthetic packages come from templates the generator wrote; section 5 only shows the wiring works.
- **Rule 3a is a template heuristic.** It was written against the synthetic templates, and the synthetic Bartleby scores lean on it.
- **Stub embeddings in part of the measurements.** The LLM row in section 2 used stub embeddings and was generated from a dirty tree; section 5 runs on stub embeddings; the section 3 report does not record which embeddings it used.
- **τ comes from CUAD.** `config/thresholds.yaml` uses the τ calibrated on CUAD (section 4). With that τ and stub embeddings, synthetic A06 packages move from REJECT to REVIEW (a NOT_VERIFIABLE finding); the synthetic tests pin the τ from their own calibration. Section 5 reflects that change.
- **False abstentions remain high.** Each one is an unnecessary human review (section 2).
- **The demo corpus is not the measured corpus.** The demo ships 30 synthetic packages (3 per type, seed 42) and 6 CUAD contracts. The metrics are measured on 300 packages from the same generator and on CUAD contracts 16–50 (τ calibrated on contracts 1–15). Package numbering does not match between the two. The screenshots come from local runs on the full corpora.
- **Reranker:** not measured, disabled.
- **NOT MEASURED:** LLM abstention with `text-embedding-005` out of sample, the cost of a human review, latency over a distributed network.

## Try it

**Live demo.** [Staging on Cloud Run](https://catcher-ui-staging-hrok623gcq-uc.a.run.app), in stub mode with no calls to Gemini. It scales to zero and may be switched off.

**Run it locally.**

```bash
uv sync
make run-api      # FastAPI on port 8000
make run-ui       # Streamlit on port 8501
make verify       # lint, tests, claim audit, report checks, terraform validate
```

**Deploy it (Cloud Run staging).**

1. Authenticate: `gcloud auth login && gcloud auth application-default login`.
2. Build, push and plan: `PROJECT_ID=<project> REGION=<region> scripts/deploy.sh`. It prints the service URL and the smoke command; it applies nothing.
3. Apply by hand after reading the plan: `terraform -chdir=infra/terraform apply tfplan`, then `scripts/smoke.sh <API_URL>`.

On the first deploy the Artifact Registry repository doesn't exist yet: run `terraform -chdir=infra/terraform apply -target=google_artifact_registry_repository.images -var container_image=none` first (the variable is required; that apply only creates the repository). With `MAX_LIVE_CALLS=0`, the Cloud Run default, the service runs on stubs only.

> **Don't run a bare `terraform apply`.** Use `scripts/deploy.sh`, then `apply tfplan`.

## Stack, datasets and licenses

- **Stack:** Python, LangGraph, Pydantic, FastAPI, Streamlit, rank-bm25, Vertex AI (Gemini, `text-embedding-005`), Langfuse, Terraform on Cloud Run, uv.
- **Data:** [CUAD v1](https://github.com/TheAtticusProject/cuad) by The Atticus Project, CC BY 4.0. `data/demo/cuad_demo.json` is an unmodified subset redistributed under the same license; the full corpus comes from `make download-data`. Synthetic packages are generated by `datagen/`.
- **License:** MIT (`pyproject.toml`). Third-party credits are in [ATTRIBUTION.md](ATTRIBUTION.md).

## About the names

*The Catcher in the RAG* is a parody of J.D. Salinger's *The Catcher in the Rye*: no text from the novel appears here, only the picture of someone standing at the edge so nobody falls. *Bartleby* is the clerk in Herman Melville's *Bartleby, the Scrivener* (1853), who answers every request with "I would prefer not to." The guard borrows his manners: when the contract says nothing, it says nothing either.
