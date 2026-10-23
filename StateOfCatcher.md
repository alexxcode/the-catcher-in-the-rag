# State of The Catcher in the RAG
**Document Version:** 1.5.0  
**System Status:** Phase 4 Remediada y Calibrada con Rigor Empírico · 54 Unit Tests Passing · Pinned Active GA Models · End-to-End Raw Text Pipeline  
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
7. **Pipeline de extremo a extremo desde texto (`raw_text`):** El extractor analiza el texto crudo del contrato y suministra los campos estructurados a los nodos de matching, términos y validación.

---

## 2. Métricas y Diagnóstico Técnico del Benchmark Actual

Evaluación ejecutada sobre un corpus sintético determinista de **300 paquetes** distribuidos homogéneamente en 10 categorías de anomalías (A01 a A10). El reporte canónico versionado se encuentra en [evals/reports/canonical_baseline_report.json](evals/reports/canonical_baseline_report.json):

| Métrica | Baseline (Con Términos) | Ablación (Solo 3-Way Match) | Impacto / Delta Observado |
|---|:---:|:---:|:---:|
| **Precisión Global de Decisión** | **100.0%** (300/300) | **60.0%** (180/300) | **+40.0% Valor de Términos Contractuales Estructurados** |
| **Casos de Falla Observados** | **0** | 120 | Las 120 fallas corresponden a violaciones de cláusulas (A04, A05, A06, A09) |
| **Tasa de Validez de Citas** | **100.0%** (120/120) | 100.0% (30/30) | Cero citas huérfanas o no coincidentes con el texto |
| **Abstención Correcta de Bartleby** | **100.0%** (30/30 en A09) | 0.0% (0/30 en A09) | Rechazo a avalar recargos sin respaldo contractual |
| **Sobre-abstención de Bartleby (Validada)** | **0.0%** (0/180) | N/A | Denominador real evaluando cláusulas legítimas de flete, pago y tarifas |
| **Tasa de Falsos Positivos en Control (A10)** | **0.0%** (0/30) | 0.0% (0/30) | Paquetes limpios aprueban sin falsas alarmas |
| **Recall@1 / Recall@5 de Cláusulas** | **100.0%** / **100.0%** | N/A | Búsqueda léxica desacoplada sin data-leakage |
| **Latencia Media por Paquete** | **9.1 ms** | 1.8 ms | Evaluación determinista en memoria |
| **Cobertura de Código en Tests** | **81%** (54 tests unitarios) | N/A | Validado en Linux/WSL2 y Windows con Pytest |

---

## 3. Remediación Rigurosa de Fases 2, 3 y 4

En respuesta a la auditoría técnica y empírica, se aplicaron las siguientes correcciones de ingeniería:

1. **Pinning de Modelos GA Activos (2026):**
   - Se descartaron los modelos retirados (`gemini-2.0-flash-001` y `gemini-1.5-pro-002`).
   - Se fijaron y verificaron en vivo sobre Vertex AI en `config/models.yaml`:
     - **Extracción estructurada:** `gemini-2.5-flash` (GA, verificado en vivo con Vertex AI).
     - **Razonamiento de cláusulas:** `gemini-2.5-pro` (GA, verificado en vivo con Vertex AI).
     - **Embeddings:** `text-embedding-005` (768 dims, activo sin retiro programado).
     - **Reranker:** `flashrank` (`ms-marco-TinyBERT-L-2-v2`, ligero CPU-only).
   - Migración al SDK moderno `google.genai` (`genai.Client(vertexai=True)`), suprimiendo advertencias obsoletas de `vertexai.generative_models`.

2. **Eliminación Total de Fallbacks Silenciosos en el Bake-Off:**
   - Se eliminó el bloque `try/except` que degradaba silenciosamente a stub o a reglas sin avisar.
   - Si se ejecuta en modo simulación/offline, la fila queda explícitamente etiquetada como `[Offline Stub Simulation]` con proveedor `stub`.
   - Se implementó `LLMContractExtractor` en modo estricto: las fallas de extracción del modelo no se enmascaran con regex; se computan como error explícito en `error_count` y `error_rate`.
   - Los costos se calculan sobre tokens reales medidos ($0.075 / 1M input, $0.30 / 1M output para Flash; $1.25 / $5.00 para Pro), o se marcan transparentemente como `N/A (Stub)`.

3. **Integración Extremo a Extremo desde Texto (`raw_text`):**
   - El nodo de extracción (`create_extract_node` en `graph.py`) y `DocumentExtractor` ahora procesan directamente el texto no estructurado del contrato (`pkg.contract.raw_text`), extrayendo `rate_sheet`, plazos de pago y penalizaciones antes de alimentar los nodos de matching y evaluación.

4. **Contratos con Redacción Diversa (Donde las Reglas Fallan):**
   - Se introdujo `style="diverse"` en `datagen/templates.py`, empleando formatos tabulares no estándar ("Agreed Cap of $..."), números en prosa y cláusulas con redacción natural.
   - Se verificó que `RuleBasedContractExtractor` falla en estos contratos (0% de exactitud en listas de precios), demostrando empíricamente por qué un LLM es indispensable y justificando su costo operativo.

5. **Calibración Empírica de Bartleby sin 0/0 y con Selección en la Meseta:**
   - **Denominador Real:** En los paquetes no-A09, se consultan cláusulas legítimas (flete estándar de Sección 8, plazos netos de pago, tarifas pactadas). Bartleby no debe abstenerse; si lo hace, se registra sobre-abstención real.
   - **Split Train / Validation:** Partición estratificada (60% calibración, 40% validación retenida).
   - **Centro de la Meseta:** Entre todos los umbrales que empatan en F1 máximo con mínima sobre-abstención, el calibrador selecciona el **centro/mediana de la meseta** (`min_relevance_score: 0.55`, `min_evidence_spans: 1`) en lugar de fijar arbitrariamente el extremo (0.80).

---

## 4. Estructura del Repositorio

```
the-catcher-in-the-rag/
├── SPEC.md                    # Especificación técnica exhaustiva (v0.4)
├── README.md                  # Documentación principal, arquitectura y benchmark
├── StateOfCatcher.md          # Diagnóstico técnico de estado y tecnologías (este documento)
├── Makefile                   # Automatización de tareas de desarrollo y pruebas
├── config/
│   ├── models.yaml            # Configuración de modelos GA activos (gemini-3.5-flash, gemini-3.1-pro)
│   ├── thresholds.yaml        # Umbrales calibrados empíricamente (Bartleby: 0.55 / 1 span)
│   └── policy_rules.yaml      # Reglas de negocio y gobernanza financiera
├── src/catcher_in_the_rag/
│   ├── graph.py               # Grafo LangGraph con extracción end-to-end desde raw_text
│   ├── extract/
│   │   ├── extractor.py       # DocumentExtractor integrado con contract_extractor
│   │   └── contract_extractor.py # RuleBased, LLMContractExtractor y HybridContractExtractor
│   ├── match/                 # Cotejo tripartito determinista (3-Way Matching)
│   ├── apply_terms/           # RAG de términos y evaluación de cláusulas
│   ├── bartleby/              # Guardia de abstención Melville (F1 calibrado en la meseta)
│   ├── validate/              # Validador de citas verbatim con offsets
│   ├── policy/                # Motor de políticas y gobierno de pagos
│   └── providers/             # Conectores Gemini (google.genai SDK) y Stub determinista
├── evals/
│   ├── bakeoff.py             # Bake-off transparente Flash vs Pro vs Rules sin fallbacks
│   ├── calibrate_bartleby.py  # Calibrador con partición Train/Val y denominador real
│   ├── runner.py              # Runner de evaluación de 300 paquetes y ablaciones
│   └── reports/               # Reportes canónicos JSON versionados
└── tests/                     # 54 tests unitarios pasando (81% de cobertura)
```
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

### Fase 3: Conexión Real con Gemini (Vertex AI & GCP) (Completado y Verificado en Vivo)
- [x] **Habilitación de API y Autenticación ADC:** Activada la API de Vertex AI (`aiplatform.googleapis.com`) en el proyecto `the-catcher-in-the-rag` autenticado sin API keys mediante ADC.
- [x] **Fijación de Modelos GA Activos en `config/models.yaml`:**
  - Extracción estructurada: `gemini-2.5-flash` (GA activo en Vertex AI, latencia media 5.6s, $0.20/1k).
  - Razonamiento de cláusulas: `gemini-2.5-pro` (GA activo en Vertex AI, latencia media 8.8s, $3.34/1k).
  - Embeddings de recuperación: `text-embedding-005` (768 dimensiones).
  - Reranker: `flashrank` (`ms-marco-TinyBERT-L-2-v2`, ligero CPU-only sin costos externos).
- [x] **Ejecución Empírica del Bake-off (`make eval-bakeoff`):** Sin stubs ni simulaciones; conexión real a Google Cloud Vertex AI reportada en `evals/reports/bakeoff_latest.json`:
  - **Rule-Based Baseline:** 0.6 ms | $0.00 / 1k | 0.0% errores | 100.0% exactitud global
  - **Gemini 2.5 Flash:** 5610.8 ms | $0.20 / 1k | 0.0% errores | 100.0% exactitud global
  - **Gemini 2.5 Pro:** 8830.6 ms | $3.34 / 1k | 0.0% errores | 100.0% exactitud global
- [x] **Trazabilidad y Observabilidad con Langfuse:** Implementado `providers/tracing.py` con interfaz abstracta `BaseTracer`, `NoOpTracer` para ejecución offline y `LangfuseTracer` para auditoría de latencia y consumo de tokens en GCP.

### Fase 4: Calibración Empírica de Bartleby (Completado)
- [x] **Barrido de Umbrales en `evals/calibrate_bartleby.py` (`make calibrate-bartleby`):** Grid search sistemático sobre los 300 paquetes cruzando A09 (ground-truth abstención) contra no-A09. Demostró 100.0% de abstención en A09 y 0.0% de sobre-abstención en paquetes legítimos (F1 = 1.0000).
- [x] **Fijación de Umbrales Óptimos en `config/thresholds.yaml`:**
  - `min_relevance_score: 0.50` (centro de meseta)
  - `min_evidence_spans: 1`
- [x] **Condición #3 de Bartleby Validada:** Verificación semántica de patrones de rechazo/prohibición en cláusulas recuperadas para rechazar interpretaciones espurias de negativos difíciles.

### Fase 5: Ingesta y Evaluación con Datos Reales (CUAD / OCDS) (Completado)
- [x] **Descargador Automatizado de Corpora Externos:** `datagen/download.py` (`make download-data`) descarga y extrae de forma reproducible CUAD v1 (510 contratos, 41 tipos de cláusulas, CC BY 4.0) y OCDS 1.1 (Open Contracting Partnership, Apache 2.0).
- [x] **Loaders de Ingesta:** `src/catcher_in_the_rag/ingest/cuad_loader.py` y `ocds_loader.py` mapeando anotaciones expertas de cláusulas, offsets verbatim, contratos SEC reales y compradores/proveedores públicos.
- [x] **Benchmark de Recuperación en Contratos Reales (CUAD):**
  - **Recall@5:** **75.8%** en contratos corporativos densos de hasta 175.000 caracteres (MRR: 0.5455, latencia media: 83.7 ms).
- [x] **Benchmark de Abstención de Bartleby en Negativos Reales (CUAD):**
  - **Tasa de Abstención Correcta:** **97.3%** en categorías contractuales genuinamente ausentes (`answers: []`).
  - **Tasa de Falsas Afirmaciones:** **2.7%** (solo 1 caso espurio en contratos reales no estructurados).
- [x] **Suite de Pruebas Unitarias & CI:** 58 tests unitarios pasando al 100% con 83% de cobertura en ingest y 82% global, 0 errores de Ruff, fixtures autocontenidas para ejecución offline en GitHub Actions CI.
- [x] **Commit y CI Verificado:** Commit `5bee38f` y `9fadc5c` en `main` verificados con GitHub Actions CI verde.

### Fase 6: Interfaz de Usuario y Explorador CUAD en Streamlit (Completado)
- [x] **Pantalla Interactiva `📜 CUAD Real SEC Contracts`:** Integración nativa de 510 contratos comerciales de la SEC con buscador reactivo y métricas KPI de volumen y categorías.
- [x] **Inspección de 41 Categorías de Cláusulas:** Filtro por estado (*Present* con offsets verbatim vs *Absent* como negativos ground-truth para Bartleby).
- [x] **Playground en Vivo de Hybrid RAG & Bartleby:** Recuperación en tiempo real (BM25 léxico + embeddings densos) sobre contratos de más de 50.000 caracteres (~40 ms) con evaluación del umbral de Bartleby (0.50).
- [x] **Simulación de Auditoría de Compras con Contrato SEC:** Auditoría 4-way en 3 escenarios (*Compliant*, *Unauthorized Surcharge A09* con activación de Melville `NO_EVIDENCE`, y *Price Variance A01*).
- [x] **Dashboard de Evaluación Multi-Pista:** Selector horizontal que unifica en una sola vista los benchmarks sintéticos (300 paquetes), el benchmark de CUAD en contratos reales, y el Bake-Off empírico de Vertex AI (Flash vs Pro vs Reglas).

### Fase 7: Observabilidad en Tiempo Real con Langfuse (Completado)
- [x] **Compatibilidad Universal con Langfuse (v5.x OpenTelemetry y v2.x/v3.x Legacy):** `LangfuseTracer` en `src/catcher_in_the_rag/providers/tracing.py` con detección dinámica de `start_observation` y fallback transparente a `client.trace`.
- [x] **Configuración Flexible de Servidor:** Soporte para `LANGFUSE_BASE_URL` (estándar v5) y `LANGFUSE_HOST` en variables de entorno.
- [x] **Instrumentación del Grafo de Auditoría (`graph.py`):** Cada ejecución de `audit_pipeline()` emite una traza de ejecución enriquecida con latencia en milisegundos, paquete, proveedor, monto total facturado, decisión final de política, hallazgos validados y estado de abstención de Bartleby.
- [x] **Instrumentación de Modelos Generativos (`providers/gemini.py`):** Registro de llamadas LLM en `extract_structured` y `generate_text` extrayendo el conteo exacto de tokens de entrada (`prompt_token_count`), candidatos de salida (`candidates_token_count`) y totales (`total_token_count`) desde `usage_metadata`.
- [x] **Indicador de Estado en la Interfaz Web (`ui/app.py`):** Monitorización visual en la barra lateral que reporta si el sistema opera con `Langfuse Active` o con el `NoOpTracer` offline.
- [x] **Suite de Pruebas Unitarias Expandida:** 61 pruebas unitarias pasando al 100% con 83% de cobertura de código global (95% en `tracing.py` y 90% en `graph.py`).
- [x] **Commit y CI Verificado:** Commits `5bee38f`, `9fadc5c` y `9b97d5d` en `main` verificados con GitHub Actions CI verde.
