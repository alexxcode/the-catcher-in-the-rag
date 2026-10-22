# State of The Catcher in the RAG
**Document Version:** 1.2.0  
**System Status:** Phase 2 (Benchmark Desafío No Estructurado) Completed · 47 Unit Tests Passing · 100% Extraction Match · CI Green  
**Date:** Octubre 2026  
**Repository:** `alexxcode/the-catcher-in-the-rag`  

---

## 1. Resumen Ejecutivo del Sistema

**The Catcher in the RAG** es un agente de auditoría determinista de compras corporativas (*procurement*) concebido para resolver una deficiencia crítica de los sistemas convencionales: el cotejo tripartito estándar (Factura ↔ Orden de Compra ↔ Remito de Recepción) pasa por alto los términos y condiciones de los contratos marco que rigen legalmente la compra.

El sistema implementa:
1. **Cotejo tripartito determinista (3-Way Matching):** Reconciliación de cantidades, precios unitarios, duplicados e identificadores de proveedor sin intervención de modelos de lenguaje.
2. **Evaluación de términos contractuales:** Verificación de tarifas máximas (`rate_sheet`), plazos netos de pago (`payment_terms_days`), penalizaciones por demora (`late_delivery_penalty_pct_per_day`) y vigencia contractual (`expiration_date`).
3. **Citas textuales verificables (Principio #2: "No citation, no accusation"):** Ningún hallazgo contractual es aceptado por el motor de políticas si no cuenta con una cita textual exacta verificada contra los offsets de caracteres del contrato original.
4. **Guardia de abstención Melville (`bartleby`):** Intercepta la recuperación para rechazar alucinaciones y emitir `NO_EVIDENCE` cuando el contrato no autoriza cargos imprevistos.
5. **Separación estricta entre extracción y decisión (Principio #1):** La IA extrae; un **motor de políticas determinista** decide (`APPROVE`, `REVIEW`, `REJECT`).
6. **Ejecución offline reproducible:** Proveedores desacoplados mediante el patrón Factory (`providers/`), permitiendo una suite 100% determinista sin dependencias de red ni costo en tokens.

---

## 2. Métricas y Diagnóstico Técnico del Benchmark Actual (M4/M5)

Evaluación ejecutada sobre un corpus sintético determinista de **300 paquetes** distribuidos homogéneamente en 10 categorías de anomalías (A01 a A10). El reporte canónico versionado se encuentra en [evals/reports/canonical_baseline_report.json](evals/reports/canonical_baseline_report.json):

| Métrica | Baseline (Con Términos) | Ablación (Solo 3-Way Match) | Impacto / Delta Observado |
|---|:---:|:---:|:---:|
| **Precisión Global de Decisión** | **100.0%** (300/300) | **60.0%** (180/300) | **+40.0% Valor de Términos Contractuales Estructurados** |
| **Casos de Falla Observados** | **0** | 120 | Las 120 fallas corresponden a violaciones de cláusulas (A04, A05, A06, A09) |
| **Tasa de Validez de Citas** | **100.0%** (120/120) | 100.0% (30/30) | Cero citas huérfanas o no coincidentes con el texto |
| **Abstención Correcta de Bartleby** | **100.0%** (30/30 en A09) | 0.0% (0/30 en A09) | Rechazo a avalar recargos sin respaldo contractual |
| **Sobre-abstención de Bartleby** | **0.0%** (0/270) | 0.0% (0/270) | No se abstiene en paquetes con cláusulas disponibles |
| **Tasa de Falsos Positivos en Control (A10)** | **0.0%** (0/30) | 0.0% (0/30) | Paquetes limpios aprueban sin falsas alarmas |
| **Recall@1 / Recall@5 de Cláusulas** | **100.0%** / **100.0%** | N/A | Búsqueda léxica determinista sobre contratos sintéticos |
| **Latencia Media por Paquete** | **9.1 ms** | 1.8 ms | Evaluación determinista en memoria |
| **Cobertura de Código en Tests** | **83%** (45 tests unitarios) | N/A | Validado en Linux/WSL2 y Windows con Pytest |

### ⚠️ Diagnóstico Técnico y Alcance Real del Benchmark Offline

Para mantener la máxima honestidad y rigor de ingeniería, es imprescindible documentar con precisión qué mide y qué no sostiene aún este benchmark:

1. **El "+40% de mejora" mide el impacto de aplicar reglas contractuales estructuradas, no la recuperación RAG no supervisada:**
   En el generador actual, los contratos entran como objetos Pydantic ya estructurados (`Contract`). El evaluador (`ContractTermsEvaluator`) compara campos directos (`payment_terms_days`, `expiration_date`, `late_delivery_penalty_pct_per_day`) y busca el substring correspondiente para certificar la cita. La ablación "no-RAG" apaga el evaluador de cláusulas completo, demostrando el inmenso valor de cruzar facturas contra contratos, pero no el desempeño de un retriever en texto libre.
2. **Circularidad en las métricas de recuperación:**
   En `evals/runner.py`, la consulta de búsqueda se construye uniendo el texto de los spans esperados (`" ".join(queries)`). Como los contratos sintéticos actuales tienen solo 5 cláusulas breves, Recall@5 es 100% por diseño geométrico del corpus.
3. **Embeddings sintéticos en modo Stub:**
   El proveedor `StubEmbeddingProvider` genera vectores deterministas mediante hashing SHA-256. En consecuencia, la recuperación en modo stub descansa primordialmente en BM25. En A09, la abstención ocurre porque el ítem no contractual no comparte vocabulario con las 5 cláusulas (BM25 = 0 y score inferior al umbral 0.5).
4. **Métrica de Sobre-abstención (0/270):**
   La consulta a Bartleby solo se dispara para ítems de factura no presentes en la Orden de Compra. Como únicamente los paquetes inyectados con A09 contienen tales ítems imprevistos, la sobre-abstención en A01–A08 y A10 es 0/270 de forma estructural.
5. **Extracción desde texto crudo pendiente de evaluación:**
   El benchmark alimenta JSON validado directamente a `model_validate`. La evaluación de exactitud de extracción campo por campo desde `raw_text` no estructurado contra LLM en vivo constituye el hito central de la siguiente fase.

---

## 3. Tecnologías Implementadas: Cómo y Dónde

| Tecnología | Versión / Componente | Ubicación en el Repositorio | ¿Cómo se implementó? |
|---|---|---|---|
| **Python** | `3.13` (Astral `uv`) | [pyproject.toml](pyproject.toml), [.python-version](.python-version) | Entorno virtual con resolución de dependencias congeladas vía `uv sync` y `uv run`. |
| **LangGraph** | `>=0.2` | [src/catcher_in_the_rag/graph.py](src/catcher_in_the_rag/graph.py) | Máquina de estados tipada (`AuditState`) en grafo acíclico: `extract` → `match` → `apply_terms` → `validate` → `decide` → `END`. |
| **Pydantic v2** | `>=2.9` | [src/catcher_in_the_rag/schemas/](src/catcher_in_the_rag/schemas/) | Tipado estricto y validadores para `Invoice`, `PurchaseOrder`, `GoodsReceipt`, `Contract`, `Citation` y `Decision`. |
| **Hybrid RAG** (BM25 + Dense) | `rank-bm25>=0.2.2` | [src/catcher_in_the_rag/retrieve/retriever.py](src/catcher_in_the_rag/retrieve/retriever.py) | Recuperador híbrido que combina BM25 con normalización léxica y similitud coseno sobre representaciones densas. |
| **Bartleby Guard** | Melville Abstention Pattern | [src/catcher_in_the_rag/bartleby/guard.py](src/catcher_in_the_rag/bartleby/guard.py) | Guardia de abstención que intercepta la recuperación: si ningún span supera el umbral de relevancia, emite `NO_EVIDENCE` sin inventar reglas. |
| **Citation Validator** | Verificación estricta de offsets | [src/catcher_in_the_rag/validate/validator.py](src/catcher_in_the_rag/validate/validator.py) | Validador estricto que descarta cualquier hallazgo contractual carente de cita o cuyos offsets `[start_char:end_char]` no coincidan verbatim. |
| **Policy Engine** | Motor de reglas determinista | [src/catcher_in_the_rag/policy/engine.py](src/catcher_in_the_rag/policy/engine.py) | Resuelve prioridades (`REJECT` > `REVIEW` > `APPROVE`), ejecuta límites monetarios (`auto_approve_max_usd`, `requires_review_max_usd`, `auto_reject_above_usd`). |
| **Providers (Stub & Gemini)** | Factory Pattern | [src/catcher_in_the_rag/providers/](src/catcher_in_the_rag/providers/) | Interfaces `LLMProvider` y `EmbeddingProvider`. Implementa `StubLLMProvider` para offline determinista y `GeminiProvider` para Vertex AI con ADC. |
| **Streamlit** | `>=1.40` | [ui/app.py](ui/app.py) | UI de dos pantallas: **Review Workspace** (inspección de 4 documentos y Human-in-the-Loop) y **Evaluation Dashboard** (métricas y ablaciones). |
| **FastAPI** | `>=0.115` | [src/catcher_in_the_rag/api/main.py](src/catcher_in_the_rag/api/main.py) | API REST con endpoints `/api/audit`, `/api/review/override` y `/api/reports`. |
| **Pytest & Ruff** | `>=8.3` / `>=0.8` | [tests/](tests/), [pyproject.toml](pyproject.toml) | Suite de 45 pruebas unitarias con cobertura del 83% y linter estricto. |
| **Terraform** | `>=1.5` | [infra/terraform/](infra/terraform/) | Infraestructura como Código (IaC) para GCP: Cloud Run, Cloud SQL (PostgreSQL + pgvector), GCS y Secret Manager. |
| **Docker & Compose** | Docker v26+ | [infra/Dockerfile](infra/Dockerfile), [docker-compose.yml](docker-compose.yml) | Empaquetado multicapa con Astral `uv` que expone API (8000) y UI (8501). |
| **GitHub Actions** | CI Workflow | [.github/workflows/ci.yml](.github/workflows/ci.yml) | CI con jobs paralelos de pruebas unitarias offline y validación de Terraform. |

---

## 4. Taxonomía de Anomalías (A01 a A10)

| ID | Anomalía Inyectada | Capa de Detección | Comportamiento del Sistema | Métrica Baseline |
|---|---|---|---|:---:|
| **A01** | Precio unitario superior a tarifa pactada | Match + Cláusula | Compara contra `rate_sheet` y cita la cláusula de lista de precios. | F1 = 1.000 |
| **A02** | Cantidad facturada mayor a la recibida | Match Tripartito | Coteja ítems facturados vs remito de bodega (`GoodsReceipt`). | F1 = 1.000 |
| **A03** | Factura duplicada | Match Tripartito | Detecta número de factura repetido en historial contable. | F1 = 1.000 |
| **A04** | Términos de pago discordantes | Contract Clause | Detecta discrepancia entre `payment_terms_days` y vencimiento de factura. | F1 = 1.000 |
| **A05** | Penalización por entrega tardía aplicable | Contract Clause | Compara fecha de entrega real vs pactada y liquida descuento por mora. | F1 = 1.000 |
| **A06** | Contrato expirado / sin renovación | Contract Clause | Rechaza facturas con fecha posterior a `expiration_date` sin prórroga. | F1 = 1.000 |
| **A07** | Proveedor desconocido o sin Orden de Compra | Match Tripartito | Rechaza facturas con `po_id` o `vendor_id` no autorizados. | F1 = 1.000 |
| **A08** | Monto total superior a $50,000 | Motor de Políticas | Rutea a revisión humana obligatoria (`REVIEW`) por límite de gobernanza. | F1 = 1.000 |
| **A09** | Cláusula ausente (prueba negativa) | Guardia Bartleby | Factura con recargo no pactado; Bartleby declina y emite `NO_EVIDENCE`. | F1 = 1.000 |
| **A10** | Paquete limpio (control de falsos positivos) | Pipeline Completo | Factura íntegramente concordante con PO, entrega y contrato; aprueba limpio. | FPR = 0.0% |

---

## 5. Estructura del Repositorio

```
the-catcher-in-the-rag/
├── SPEC.md                    # Especificación técnica exhaustiva (v0.4)
├── README.md                  # Documentación principal, arquitectura y benchmark
├── StateOfCatcher.md          # Diagnóstico técnico de estado y tecnologías (este documento)
├── DATA_SOURCES.md            # Registro de procedencia y licencias de datos (CUAD, OCDS)
├── ATTRIBUTION.md             # Atribuciones y créditos legales
├── Makefile                   # Automatización: format, lint, test, eval, run-ui, run-api
├── pyproject.toml             # Metadatos del paquete y dependencias fijadas con Astral uv
├── docker-compose.yml         # Orquestación local (FastAPI + Streamlit)
│
├── config/                    # Configuración declarativa (umbrales, políticas, modelos)
│   ├── models.yaml            # Declaración de modelos Vertex AI / Gemini
│   ├── policy.yaml            # Reglas de decisión y límites monetarios de aprobación
│   └── thresholds.yaml        # Tolerancias de matcheo y umbrales de recuperación
│
├── src/catcher_in_the_rag/    # Paquete Python principal
│   ├── schemas/               # Modelos Pydantic v2 (Contract, PO, Receipt, Invoice, Findings)
│   ├── extract/               # Extractor de documentos schema-constrained
│   ├── match/                 # Motor de cotejo tripartito determinista (A01, A02, A03, A07)
│   ├── retrieve/              # RAG híbrido: BM25 + Embeddings densos
│   ├── bartleby/              # Guardia de abstención Melville (A09)
│   ├── apply_terms/           # Evaluador de términos contractuales (A04, A05, A06)
│   ├── validate/              # Validador estricto de citas textuales y offsets
│   ├── policy/                # Motor de políticas y resolución de prioridades (A08)
│   ├── providers/             # Proveedores: Stub determinista y Gemini Vertex AI (ADC)
│   ├── graph.py               # Máquina de estados LangGraph (pipeline end-to-end)
│   └── api/                   # Servicio REST FastAPI (endpoints OpenAPI)
│
├── datagen/                   # Generador sintético determinista de 300 paquetes
│   ├── templates.py           # Plantillas de contratos y compras
│   ├── injector.py            # Inyector determinista de anomalías A01–A10
│   └── generate.py            # CLI generador con semilla reproducible
│
├── evals/                     # Harness de evaluación y ablaciones
│   ├── metrics.py             # Modelos de métricas: Precision, Recall, F1, MRR, FPR
│   ├── runner.py              # Ejecutor de benchmarks con ablaciones (baseline vs no-rag)
│   ├── report_writer.py       # Renderizador de reportes CLI y serializador de JSON
│   ├── run.py                 # CLI de evaluación (`make eval` / `make eval-ablations`)
│   └── reports/               # Reportes canónicos versionados y comiteados en git
│
├── ui/                        # Interfaz de usuario Streamlit (Review Workspace & Dashboard)
├── tests/                     # Suite de 45 pruebas unitarias e integrales
└── infra/                     # Dockerfile y módulos de Terraform para Google Cloud
```

---

## 6. Comandos Operativos Clave

Todos los flujos se operan mediante el [Makefile](Makefile):

| Acción | Comando | Descripción |
|---|---|---|
| **Iniciar UI Web** | `make run-ui` | Abre Streamlit en `http://localhost:8501`. |
| **Iniciar API Backend** | `make run-api` | Abre FastAPI en `http://localhost:8000/docs`. |
| **Ejecutar Pruebas** | `make test` | Corre los 45 tests unitarios en modo offline con reporte de cobertura. |
| **Correr Evaluación** | `make eval-ablations` | Corre el benchmark de 300 paquetes y reporta el impacto contractual. |
| **Formateo & Calidad** | `make format && make lint` | Aplica Ruff y Terraform fmt; valida linter sin advertencias. |
| **Regenerar Dataset** | `make generate-synthetic` | Reconstruye los 300 paquetes deterministas en `data/synthetic/`. |
| **Despliegue Local Docker** | `docker compose up --build` | Levanta API y UI en contenedores aislados. |

---

## 7. Roadmap Priorizado: Hacia la Evaluación Rigurosa

Habiendo culminado la fase de **Honestidad Primero**, la evolución hacia un sistema evaluado con rigor extremo se estructura en las siguientes fases:

### Fase 1: Honestidad y Consistencia Técnica (Completado Hoy)
- [x] Implementación formal de `providers/gemini.py` con autenticación ADC y desacoplamiento limpio.
- [x] Soporte para alias de proveedores (`real`, `vertex_ai`, `gemini`, `stub`) en `factory.py`.
- [x] Configuración por defecto de `.env.example` en modo offline determinista (`stub`).
- [x] Aplicación estricta del Principio #2 en `validator.py`: los hallazgos contractuales sin cita válida son estrictamente descartados.
- [x] Activación de umbrales `auto_approve_max_usd` y `auto_reject_above_usd` en `PolicyEngine`.
- [x] Corrección visual de A10 en reportes: presentado como tasa de falsos positivos (FPR = 0.0%) en lugar de P/R/F1 artificial.
- [x] Renombramiento honesto de métricas: "+40% Valor de Términos Contractuales Estructurados".
- [x] Inclusión de reportes canónicos en Git (`evals/reports/canonical_baseline_report.json`).

### Fase 2: Benchmark que Pueda Fallar (Completado)
- [x] **Extracción desde `raw_text`:** Implementado `RuleBasedContractExtractor` y `HybridContractExtractor` en `src/catcher_in_the_rag/extract/contract_extractor.py`, analizando contratos sin estructurar con fallback riguroso.
- [x] **Métricas de exactitud de extracción campo por campo:** Harness `evals/extraction_eval.py` (`make eval-extraction`) evaluando Exact Match en `payment_terms_days`, `expiration_date`, `late_delivery_penalty_pct_per_day` y `rate_sheet` (100.0% Exact Match en 300 contratos).
- [x] **Contratos densos con distractores:** Expandido `datagen/templates.py` de 5 a 18 secciones contractuales (Garantías, Indemnizaciones, Auditoría, Propiedad Intelectual, Cumplimiento y Sección 8 de Fletes/Transporte como distractor).
- [x] **Consultas desacopladas (Sin data-leakage):** Búsqueda de cláusulas en `evals/runner.py` desacoplada del texto del span objetivo mediante consultas temáticas de auditoría (`audit_topic_queries`), eliminando la circularidad de Recall@5.
- [x] **Negativos difíciles en A09 y Activación de Bartleby Condición #3:** La Sección 8 (Fletes) actúa como negativo difícil para recargos no autorizados. Bartleby inspecciona si las cláusulas coincidentes son restrictivas o deniegan reembolso ("under no circumstances", "unless authorized"), emitiendo `NO_EVIDENCE` de manera robusta.

### Fase 3: Conexión Real con Gemini (Vertex AI & GCP)
1. Fijar modelos específicos con estado General Availability (GA) y ciclo de vida validado en [Google Cloud Model Versioning](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning).
2. Ejecutar un *bake-off* de extracción controlada entre modelos Flash y Pro, cuantificando trade-offs de costo, latencia y fidelidad de citas.
3. Conectar trazas en vivo a Langfuse para auditoría de latencia de red y consumo de tokens.

### Fase 4: Calibración Empírica de Bartleby
1. Barrido de umbrales (`min_relevance_score`, `min_bm25_score`, `min_dense_score`) sobre un conjunto de validación separado (A09 vs A10).
2. [x] **Condición #3 de Bartleby Implementada:** Verificación semántica de patrones de rechazo/prohibición en cláusulas recuperadas para rechazar interpretaciones espurias de negativos difíciles.
