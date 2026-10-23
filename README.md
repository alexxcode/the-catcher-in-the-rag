# 🛡️ The Catcher in the RAG

> *"catches the invoice before it falls off the cliff."*

[![CI](https://github.com/alexxcode/the-catcher-in-the-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/alexxcode/the-catcher-in-the-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13%20%7C%203.12-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-61%20passing%20%2883%25%20coverage%29-brightgreen)
![GCP Vertex AI](https://img.shields.io/badge/GCP-Vertex%20AI%20%28ADC%29-4285F4?logo=googlecloud&logoColor=white)
![Models](https://img.shields.io/badge/Models-Gemini%202.5%20Flash%20%26%20Pro-8E75C2?logo=google&logoColor=white)
![Observability](https://img.shields.io/badge/Observability-Langfuse%20Cloud%20%28OTel%20v5%29-FF6B6B)
![Real Corpus](https://img.shields.io/badge/Corpus-CUAD%20v1%20%28510%20SEC%20Contracts%29-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**The Catcher in the RAG** es un agente de auditoría y conciliación financiera para flujos de adquisiciones (*Procurement*). A diferencia de los cotejos tradicionales, no solo valida la coherencia aritmética entre facturas, órdenes de compra y recepciones de mercancías (**Three-Way Matching**), sino que recupera, interpreta y audita **las cláusulas contractuales vinculantes** que rigen la transacción (techos de tarifas, plazos de pago, penalizaciones por demora y renovación automática), respaldando cada hallazgo con **citas textuales infalsificables** y **absteniéndose de alucinar** cuando el contrato guarda silencio.

El núcleo del proyecto es un **harness de evaluación científica reproducible** medido tanto en un corpus sintético adversarial de 300 paquetes como en el dataset real **CUAD v1** (510 contratos comerciales de la SEC anotados por juristas).

---

## 📑 Tabla de Contenidos

1. [Diferenciadores Clave y Filosofía](#-diferenciadores-clave-y-filosofía)
2. [Arquitectura del Pipeline LangGraph](#-arquitectura-del-pipeline-langgraph)
3. [Resultados Empíricos de Benchmarks](#-resultados-empíricos-de-benchmarks)
   - [Benchmark Sintético (300 Paquetes & Ablaciones)](#1-benchmark-sintético-300-paquetes-y-ablaciones)
   - [Procedimiento y Benchmark en Contratos Reales de la SEC (CUAD v1 & OCDS)](#2-procedimiento-y-benchmark-en-contratos-reales-de-la-sec-cuad-v1--ocds)
   - [Bake-off de Extracción en Vertex AI (Flash vs. Pro vs. Reglas)](#3-bake-off-de-extracción-en-google-cloud-vertex-ai)
4. [Metodología de Calibración de Bartleby](#-metodología-de-calibración-de-bartleby)
5. [Observabilidad en Tiempo Real con Langfuse](#-observabilidad-en-tiempo-real-con-langfuse)
6. [Interfaz de Usuario Streamlit](#-interfaz-de-usuario-streamlit)
7. [Taxonomía de Anomalías (A01–A10)](#-taxonomía-de-anomalías-a01a10)
8. [Inicio Rápido y Comandos Operativos](#-inicio-rápido-y-comandos-operativos)
9. [Estructura del Repositorio](#-estructura-del-repositorio)
10. [Procedencia de Datos y Licencias](#-procedencia-de-datos-y-licencias)

---

## 💡 Diferenciadores Clave y Filosofía

| Principio | Implementación en The Catcher in the RAG |
|---|---|
| **1. El LLM Extrae, el Código Decide** | Las decisiones financieras (`APPROVE`, `REVIEW`, `REJECT`, `ABSTAIN`) son emitidas por un motor de políticas determinista ([`policy/engine.py`](src/catcher_in_the_rag/policy/engine.py)). Jamás se delega la aprobación monetaria al criterio estocástico de un prompt. |
| **2. Citas Textuales Infalsificables** | Todo hallazgo contractual exige document ID, rango de caracteres `[start_char:end_char]` y cita literal. El validador ([`validate/validator.py`](src/catcher_in_the_rag/validate/validator.py)) descarta implacablemente cualquier hallazgo cuya cita no coincida byte a byte con el contrato. |
| **3. Preferiría No Hacerlo (`bartleby`)** | Cuando un contrato no regula un cargo (ej. recargos de flete o seguros no pactados), el guardia de abstención [Bartleby](src/catcher_in_the_rag/bartleby/guard.py) emite `NO_EVIDENCE` y `ABSTAIN`. La arquitectura prefiere no dictaminar antes que inventar reglas contractuales. |
| **4. Desacoplamiento Offline / Determinista** | La suite completa de 61 tests unitarios y las evaluaciones basales corren en CI de forma 100% offline y determinista con el proveedor `stub`, eliminando costos de red y flakiness en integración continua. |
| **5. Autenticación ADC sin Secretos en Git** | En entornos cloud reales, la conexión a Gemini en Google Cloud Vertex AI opera mediante Application Default Credentials (ADC), sin almacenar API keys en el repositorio ni en variables de entorno fijas. |

---

## 🏛️ Arquitectura del Pipeline LangGraph

El flujo de auditoría se modela como una máquina de estados dirigida mediante **LangGraph**:

```
                       PDF / Texto Bruto de Contrato y Documentos
                                           │
                              ┌────────────┴────────────┐
                              ▼                         ▼
                    [1. Ingesta OCDS/CUAD]   [2. Extracción de Términos]
                              │              (Schema-Constrained LLM)
                              └────────────┬────────────┘
                                           ▼
                              [3. Enlace Documental por ID]
                                 (PO ↔ Receipt ↔ Invoice)
                                           │
                                           ▼
                             [4. Cotejo Tripartito 3-Way]
                            (Tolerancias de Precio y Cantidad)
                                           │
                                           ▼
                      [5. Recuperación Híbrida de Términos RAG]
                        (BM25 Léxico + Embeddings Densos 768d)
                                           │
                                           ▼
                   ┌───────────────────────────────────────────────┐
                   │  [6. Guardia de Abstención Melville: BARTLEBY] │
                   │    ¿Existen cláusulas relevantes (s ≥ 0.50)?   │
                   └───────┬───────────────────────────────┬───────┘
                     SÍ    │                               │  NO (o Restricción)
                           ▼                               ▼
                 [7. Evaluación de Cláusulas]     [Veredicto: ABSTAIN]
                 (Tarifas, Pagos, Penalidades)    (NO_EVIDENCE Emitido)
                           │                               │
                           ▼                               ▼
                 [8. Validador Verbatim] ─────────► [9. Motor de Políticas]
                 (Verificación de Citas/Offsets)   (Resolución Determinista)
                                                           │
                                                           ▼
                                            APPROVE │ REVIEW │ REJECT │ ABSTAIN
```

---

## 📊 Resultados Empíricos de Benchmarks

### 1. Benchmark Sintético (300 Paquetes y Ablaciones)
Evaluado de forma automatizada sobre 300 paquetes generados proceduralmente con inyección controlada de errores ([`evals/reports/canonical_baseline_report.json`](evals/reports/canonical_baseline_report.json)):

| Métrica de Evaluación | Full RAG (Baseline) | Ablación Sin-RAG (Solo 3-Way) | Impacto / Delta |
|---|:---:|:---:|:---:|
| **Exactitud Global de Decisión** | **100.0%** (300/300) | **60.0%** (180/300) | **+40.0% Valor del RAG Contractual** |
| **Validez de Citas Verbatim** | **100.0%** (120/120) | N/A | Cero alucinaciones de citas |
| **Abstención Correcta de Bartleby** | **100.0%** (30/30 en A09) | 0.0% (0/30 en A09) | Detección de silencios contractuales |
| **Tasa de Falsos Positivos (A10)** | **0.0%** (0/30) | 0.0% (0/30) | Paquetes limpios aprobados sin fricción |
| **Recall@5 de Recuperación** | **100.0%** | N/A | Recuperación multi-sección robusta |
| **Latencia Media por Paquete** | **9.9 ms** | 1.1 ms | Ejecución determinista local |

> **Conclusión Científica:** Prescindir de la recuperación de cláusulas contractuales degrada la exactitud del sistema en un **40%**, dejando escapar discrepancias críticas en plazos de pago (A04), penalizaciones por entrega tardía (A05) y vigencia de acuerdos (A06).

---

### 2. Procedimiento y Benchmark en Contratos Reales de la SEC (CUAD v1 & OCDS)

Para contrastar el comportamiento del sistema fuera del laboratorio sintético, se integró el estándar de oro de comprensión de contratos corporativos: **CUAD v1** (*Contract Understanding Atticus Dataset*, 510 contratos presentados ante la **SEC EDGAR**, con hasta 175.000 caracteres por documento) junto con releases estándar de compras públicas **OCDS 1.1** (*Open Contracting Data Standard*).

#### A. Flujo de Adquisición y Descarga (`datagen/download.py`)
El pipeline incluye un script desacoplado con control de integridad, hashes y licenciamiento:
```bash
make download-data
# Descarga automatizada:
# 1. CUAD_v1.json (~50 MB, The Atticus Project, bajo licencia CC BY 4.0)
# 2. ocds_sample.json (Open Contracting Partnership, bajo licencia ODC-BY)
# Almacenamiento local en data/real/ (aislado e ignorado por Git)
```

#### B. Arquitectura de Ingesta y Mapeo Ontológico (`ingest/`)
Los contratos corporativos de la SEC no están estructurados como bases de datos. El proceso de ingesta traduce el formato nativo SQuAD 2.0 a entidades auditables:
* **Mapeo Ontológico de 41 Categorías Jurídicas ([`cuad_loader.py`](src/catcher_in_the_rag/ingest/cuad_loader.py)):** Mapea las 41 clases legales anotadas por juristas a tópicos contractuales auditables por la política:
  * `Payment Terms` ➔ Días de crédito, plazos de facturación y descuentos por pronto pago.
  * `Price Restrictions / Most Favored Nation` ➔ Topes de precios y prohibiciones de ajustes unilaterales.
  * `Liquidated Damages / Penalties` ➔ Tasas de penalización por mora en entrega.
  * `Governing Law / Termination` ➔ Jurisdicción y cláusulas de rescisión anticipada.
* **Preservación de Spans Verbatim con Coordenadas Físicas:** Cada cláusula retiene su offset exacto de caracteres `[start_char, end_char]` y número de párrafo dentro del texto completo de 50 a 80 páginas, posibilitando la trazabilidad y la citación infalsificable.
* **Normalización OCDS ([`ocds_loader.py`](src/catcher_in_the_rag/ingest/ocds_loader.py)):** Estructura transacciones públicas enlazando el ID de licitación (`tender`), el contrato adjudicado (`award`) y los hitos de entrega correspondientes.

#### C. Metodología del Benchmark Real ([`evals/cuad_benchmark.py`](evals/cuad_benchmark.py))
El harness de evaluación somete al RAG híbrido (BM25 + Dense Vectors) a dos tipos de pruebas:
1. **Recuperación Positiva (Cláusula Presente):** Evalúa si, ante una consulta sobre una política de auditoría (ej. *"What are the penalty terms for late delivery?"*), el motor RAG es capaz de posicionar la cláusula real anotada por los abogados dentro de su ranking.
2. **Evaluación de Ausencia Negativa (Anti-Alucinación de Bartleby):** Se evalúan consultas sobre cláusulas que el contrato expresamente **no contiene** (`answers: []`). Si el sistema inventa una cláusula o cita un fragmento irrelevante para sancionar al proveedor, comete una *afirmación espuria*; si emite `ABSTAIN` y `NO_EVIDENCE`, se considera una **abstención correcta**.

#### D. Resultados Empíricos Oficiales
Resultados certificados sobre contratos auténticos de la SEC mediante [`evals/reports/cuad_benchmark_latest.json`](evals/reports/cuad_benchmark_latest.json):

| Métrica | Valor Obtenido | Interpretación y Significado Práctico |
|---|:---:|---|
| **Recall@1** | `39.4%` | La cláusula exacta es la primera recomendación del ranking en 4 de cada 10 consultas. |
| **Recall@3** | `69.7%` | Casi el 70% de las cláusulas críticas están en el top 3 de relevancia. |
| **Recall@5** | **75.8% – 81.6%** | Alta cobertura en contratos de 50.000 a 175.000 caracteres sin pre-procesamiento manual. |
| **Mean Reciprocal Rank (MRR)** | **0.5455** | Posición recíproca media sobresaliente para contratos densos y extensos. |
| **Abstención Correcta (Bartleby)** | **97.3%** | En ausencias ground-truth reales, el guardia se niega a alucinar cargos o penalizaciones. |
| **Tasa de Falsa Afirmación Espuria** | **2.7%** | Rango mínimo de falsos positivos en documentos reales no estructurados. |
| **Latencia Media de Recuperación** | **83.7 ms** | Indexación léxica BM25 y scoring semántico en sub-décima de segundo por contrato. |

#### E. Reproducción del Benchmark en Local
Cualquier usuario puede reproducir este benchmark ejecutando:
```bash
make download-data   # Descarga los contratos de la SEC y releases OCDS
make eval-cuad       # Ejecuta la suite de evaluación CUAD
```
Los resultados se serializan automáticamente en `evals/reports/cuad_benchmark_latest.json` y se grafican de forma interactiva en las pantallas **"CUAD Real SEC Contracts"** y **"Evaluation Dashboard"** de la interfaz Streamlit.

---

### 3. Bake-Off de Extracción en Google Cloud Vertex AI
Comparativa transparente en vivo sobre contratos reales sin fallbacks silenciosos ([`evals/reports/bakeoff_latest.json`](evals/reports/bakeoff_latest.json)):

| Modelo / Candidato | Proveedor | Latencia Media | Costo Estimado / 1k docs | Tasa de Error | Exactitud de Extracción |
|---|---|:---:|:---:|:---:|:---:|
| **Rule-Based Baseline** | Regex / Heurística | **0.6 ms** | **$0.00** | 0.0% | 100.0% (sintético estándar) / *0% (estilos diversos)* |
| **Gemini 2.5 Flash (GA)** | Vertex AI (ADC) | **5,610.8 ms** | **$0.20** | **0.0%** | **100.0% (todos los estilos)** |
| **Gemini 2.5 Pro (GA)** | Vertex AI (ADC) | **8,830.6 ms** | **$3.34** | **0.0%** | **100.0% (todos los estilos)** |

> **Hallazgo:** Mientras que las reglas manuales fallan en contratos con formatos tabulares no estándar o redacción narrativa variada, **Gemini 2.5 Flash** ofrece exactitud perfecta a un costo 16 veces menor que el tier Pro, validándolo como el modelo de producción óptimo para la etapa de extracción.

---

## 🎯 Metodología de Calibración de Bartleby

El umbral de decisión del guardia no fue fijado arbitrariamente. Se desarrolló un calibrador riguroso ([`evals/calibrate_bartleby.py`](evals/calibrate_bartleby.py)):

1. **Split Train / Validation:** Partición estratificada 60/40 para evitar sobreajuste.
2. **Denominador Real contra Sobre-Abstención:** Para evitar divisiones indeterminadas ($0/0$), el calibrador contrasta paquetes con anomalía contractual real (donde Bartleby debe abstenerse) contra paquetes válidos con cláusulas de transporte estándar (donde abstenerse constituye un error de falso positivo).
3. **Selección en el Centro de la Meseta:** En lugar de adoptar umbrales extremos (ej. $0.80$), el sistema selecciona la **mediana de la meseta de máxima puntuación F1**:
   $$\text{min\_relevance\_score} = 0.50 \qquad \text{min\_evidence\_spans} = 1$$
4. **Condición #3 (Inspección Semántica de Restricciones):** Bartleby no solo evalúa el puntaje de similitud, sino que detecta prohibiciones expresas (*"under no circumstances"*, *"unless authorized by written amendment"*), asegurando que cláusulas distractoras no se interpreten espuriamente como autorizaciones de pago.

---

## 📡 Observabilidad en Tiempo Real con Langfuse

El sistema cuenta con instrumentación integral bajo el estándar **OpenTelemetry v5** mediante [`providers/tracing.py`](src/catcher_in_the_rag/providers/tracing.py):

* **Trazabilidad de Grafo:** Cada invocación de `audit_pipeline()` registra latencia, ID de paquete, proveedor, monto total, decisión final y si Bartleby ejerció abstención.
* **Trazabilidad de Tokens y Costos:** Cada llamada a Gemini captura `prompt_token_count`, `candidates_token_count` y tokens totales directamente desde `usage_metadata`, permitiendo un control presupuestario granular en tiempo real.
* **Modo Cero Overhead:** Si no se configuran variables de entorno, opera el `NoOpTracer` sin dependencias de red.

---

## 🖥️ Interfaz de Usuario Streamlit

La interfaz de usuario interactiva ([`ui/app.py`](ui/app.py)) incluye tres entornos operativos principales:

1. **📋 Review Workspace:**
   * Explorador cuatripartito interactivo (Factura, PO, Recepción, Contrato).
   * Visualizador de citas textuales con resaltado de offsets y verificación verbatim.
   * Banner de activación de Bartleby con justificación en caso de abstención.
   * Panel de intervención humana (*Human-in-the-Loop*) para confirmación o anulación de decisiones.

2. **📜 CUAD Real SEC Contracts:**
   * Navegador entre 510 contratos auténticos de la SEC con búsqueda y filtrado en tiempo real.
   * Inspector de las 41 categorías legales con diferenciación entre cláusulas presentes y negativos ausentes.
   * Playground interactivo de RAG Híbrido: ejecución de consultas de auditoría con desglose visual de puntajes BM25, densos y combinados.
   * Simulador de compras reales: prueba auditorías completas sobre contratos de la SEC en escenarios conformes o con recargos no autorizados.

3. **📊 Evaluation Dashboard:**
   * Selector multi-pista unificado para explorar el benchmark sintético (300 paquetes), el benchmark de CUAD y el Bake-Off de modelos Vertex AI.

---

## 🏷️ Taxonomía de Anomalías (A01–A10)

| Código | Categoría | Descripción Técnica | Mecanismo de Detección |
|:---:|---|---|---|
| **A01** | *Price Ceiling Violation* | Precio unitario facturado excede el techo pactado | Cotejo contra rate sheet extraída |
| **A02** | *Quantity Discrepancy* | Cantidad facturada superior a la recibida en muelle | Three-way match determinista |
| **A03** | *Duplicate Billing* | Número de factura duplicado para el mismo proveedor | Registro hash de facturas previas |
| **A04** | *Payment Terms Mismatch* | Plazo facturado (ej. Net 15) discrepa del contrato (Net 30) | Recuperación RAG de cláusula de pago |
| **A05** | *Late Delivery Penalty* | Entrega posterior a la fecha pactada sujeta a multa | Evaluación de penalización por día |
| **A06** | *Contract Expiration* | Facturación con fecha posterior al vencimiento | Comparación temporal contra vigencia |
| **A07** | *Unlinked Purchase Order* | Factura huérfana o proveedor no registrado | Validación de grafo relacional |
| **A08** | *Monetary Approval Threshold* | Monto de compra superior a $50,000 USD | Regla de gobernanza financiera |
| **A09** | *Governing Clause Absent* | Recargo no contemplado en el acuerdo (ej. flete extra) | Guardia de abstención **Bartleby** |
| **A10** | *Clean Package* | Paquete totalmente conforme (control negativo) | Aprobación automática sin fricción |

---

## 🚀 Inicio Rápido y Comandos Operativos

### Prerrequisitos
* Python 3.12 o 3.13 con soporte para [Astral `uv`](https://github.com/astral-sh/uv).
* Google Cloud CLI (`gcloud`) autenticado con ADC (para pruebas en vivo con Gemini).

### Instalación y Pruebas Locales

```bash
# 1. Clonar el repositorio
git clone https://github.com/alexxcode/the-catcher-in-the-rag.git
cd the-catcher-in-the-rag

# 2. Configurar entorno y dependencias
make setup

# 3. Formateo y validación de calidad de código (Ruff)
make format
make lint

# 4. Correr la suite completa de 61 tests unitarios (83% de cobertura)
make test
```

### Ejecución de Benchmarks

```bash
# Generar el dataset sintético reproducible (300 paquetes)
make generate-synthetic

# Ejecutar el benchmark basal con ablación Sin-RAG
make eval-ablations

# Calibrar empíricamente los umbrales de Bartleby
make calibrate-bartleby

# Descargar corpora reales (CUAD v1 y OCDS 1.1)
make download-data

# Correr el benchmark sobre los contratos reales de la SEC
make eval-cuad

# Ejecutar el bake-off en Vertex AI (requiere gcloud auth)
make eval-bakeoff
```

### Iniciar la Interfaz Web y API

```bash
# Iniciar la interfaz web Streamlit (puerto 8501)
make run-ui

# Iniciar la API REST FastAPI (puerto 8000, docs en /docs)
make run-api
```

---

## 📁 Estructura del Repositorio

```
the-catcher-in-the-rag/
├── SPEC.md                    # Especificación técnica exhaustiva del sistema
├── README.md                  # Documentación principal, arquitectura y métricas
├── StateOfCatcher.md          # Bitácora histórica de fases y diagnósticos técnicos
├── DATA_SOURCES.md            # Registro de procedencia y licencias de datasets
├── ATTRIBUTION.md             # Créditos legales y atribuciones académicas
├── Makefile                   # Automatización: test, lint, format, eval, run-ui
├── pyproject.toml             # Metadatos del paquete y dependencias fijadas
├── config/
│   ├── models.yaml            # Configuración de modelos GA activos (Gemini 2.5)
│   ├── thresholds.yaml        # Umbrales empíricos (Bartleby: 0.50 / 1 span)
│   └── policy_rules.yaml      # Reglas deterministas de aprobación financiera
├── src/catcher_in_the_rag/
│   ├── graph.py               # Máquina de estados LangGraph con instrumentación OTel
│   ├── schemas/               # Modelos Pydantic v2 (Contract, PO, Invoice, Finding)
│   ├── extract/               # Extracción estructurada y extractores híbridos
│   ├── match/                 # Motor de cotejo tripartito (Three-Way Match)
│   ├── retrieve/              # RAG Híbrido: BM25 Okapi + Embeddings densos
│   ├── bartleby/              # Guardia de abstención Melville (F1 calibrado)
│   ├── apply_terms/           # RAG de términos y evaluador de penalidades
│   ├── validate/              # Validador estricto de citas verbatim y offsets
│   ├── policy/                # Motor de políticas y gobernanza de pagos
│   ├── ingest/                # Loaders para datasets reales (CUAD y OCDS)
│   ├── providers/             # Proveedores: Gemini (ADC), Stub offline y Langfuse
│   └── api/                   # Servicio REST FastAPI
├── datagen/                   # Generador procedimental determinista y downloader
├── evals/                     # Harness de evaluación, runners y calibradores
│   └── reports/               # Reportes canónicos JSON versionados en Git
├── ui/                        # Interfaz interactiva de usuario en Streamlit
└── tests/                     # Suite de 61 tests unitarios e integrales (83% cobertura)
```

---

## 📜 Procedencia de Datos y Licencias

* **CUAD v1:** *Contract Understanding Atticus Dataset*, publicado por The Atticus Project bajo licencia [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
* **OCDS 1.1:** *Open Contracting Data Standard*, mantenido por la Open Contracting Partnership bajo licencia [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
* **Atribución Literaria:** El título *The Catcher in the RAG* es empleado únicamente como parodia técnica. No se reproduce texto, citas ni imágenes de la obra original de J.D. Salinger. Las citas alusivas a *Bartleby, the Scrivener* (Herman Melville, 1853) pertenecen al dominio público.

---

## ⚖️ Licencia

Distribuido bajo la licencia **MIT**. Consulte el archivo [LICENSE](LICENSE) para más información.
