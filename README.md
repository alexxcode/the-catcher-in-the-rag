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
| **4. Desacoplamiento Offline / Determinista** | La suite completa de tests unitarios y de integración (67 pasados, 2 omitidos sin datos de CUAD locales; 69 pasando con `make download-data`) y las evaluaciones basales corren en CI de forma 100% offline y determinista con el proveedor `stub`, eliminando costos de red y flakiness en integración continua. |
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

| Métrica de Evaluación | Con Términos Contractuales (Baseline) | Sin Términos (Solo Cotejo 3-Way) | Impacto / Delta |
|---|:---:|:---:|:---:|
| **Exactitud Global de Decisión** | **100.0%** (300/300) | **60.0%** (180/300) | **+40.0% Valor de Términos Contractuales** |
| **Validez de Citas Verbatim** | **100.0%** (120/120) | N/A | Cero alucinaciones de citas |
| **Abstención Correcta de Bartleby** | **100.0%** (30/30 en A09*) | 0.0% (0/30 en A09) | Detección de silencios (vía heurística 3a*) |
| **Tasa de Falsos Positivos (A10)** | **0.0%** (0/30) | 0.0% (0/30) | Paquetes limpios aprobados sin fricción |
| **Recall@5 de Recuperación** | **100.0%** | N/A | Recuperación en plantillas sintéticas |
| **Latencia Media por Paquete** | **9.9 ms** | 1.1 ms | Ejecución determinista local |

> **Nota Metodológica:** Esta ablación evalúa el impacto de aplicar las reglas contractuales (penalizaciones por retraso, plazos de pago netos y topes de precios) frente a un cotejo tripartito puramente aritmético. Prescindir de los términos contractuales estructurados degrada la decisión financiera en un **40%**, dejando escapar cargos indebidos (A04, A05, A06). (*La abstención del 100% en A09 en datos sintéticos depende de la heurística de plantilla 3a descrita abajo; el score de recuperación pura en contratos abiertos se evalúa en CUAD).*

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

#### D. Resultados Empíricos y Reproducibilidad
Los resultados se serializan automáticamente con procedencia criptográfica (SHA de commit de Git, timestamp ISO con microsegundos y comando exacto) en [`evals/reports/cuad_benchmark_latest.json`](evals/reports/cuad_benchmark_latest.json):

* **Comando de Reproducción Offline (Stub / CI):** `make eval-cuad` o `uv run python -m evals.run --eval-cuad`
* **Comando de Reproducción Offline (Stub / CI):** `make eval-cuad` o `uv run python -m evals.run --eval-cuad`
* **Comando de Recalibración:** `make calibrate-cuad` o `uv run python -m evals.calibrate_cuad`
* **Commit de Referencia:** Verificado en Git SHA registrado en metadata de reporte.

| Métrica | Modo Offline (`stub-hash-768d`) | Vertex AI (`text-embedding-005` Live) | Intervalo de Confianza (95% Wilson Stub) | Interpretación y Significado Práctico |
|---|:---:|:---:|:---:|---|
| **Recall@1** | 35.5% | *Pendiente de cuota Vertex AI* | — | Precisión de primer ranking en contratos no estructurados. |
| **Recall@3** | 72.4% | *Pendiente de cuota Vertex AI* | — | Cláusula relevante localizada dentro del top 3 de candidatos. |
| **Recall@5** | 81.6% | *Pendiente de cuota Vertex AI* | — | Cobertura agregada en contratos extensos (50k a 175k caracteres). |
| **MRR** | 0.5336 | *Pendiente de cuota Vertex AI* | — | Posición recíproca media del fragmento probatorio real. |
| **Abstención en Negativos** | **51.5%** | *Pendiente de cuota Vertex AI* | `[41.8%, 61.1%]` | Ante cláusulas ausentes reales (`answers: []`), Bartleby rehúsa alucinar. |
| **Abstención Errónea (Positivos)** | **14.5%** | *Pendiente de cuota Vertex AI* | `[8.3%, 24.1%]` | Tasa de sobre-abstención en presencia de cláusulas legítimas válidas. |
| **Falsa Afirmación Espuria** | **48.5%** | *Pendiente de cuota Vertex AI* | — | Riesgo residual de afirmar falsamente una penalización inexistente. |

#### E. Reproducción del Benchmark en Local
Cualquier usuario puede verificar y regenerar estos reportes ejecutando:
```bash
make download-data   # Descarga los contratos de la SEC y releases OCDS
make eval-cuad       # Ejecuta la suite de evaluación CUAD con procedencia (Git SHA + microsegundos)
make calibrate-cuad  # Ejecuta el barrido de calibración empírica sobre CUAD (Train/Val 60/40)
```
Los resultados se serializan en `evals/reports/cuad_benchmark_latest.json` y se grafican de forma interactiva en las pantallas **"CUAD Real SEC Contracts"** y **"Evaluation Dashboard"** de la interfaz Streamlit.

---

### 3. Bake-Off de Extracción en Google Cloud Vertex AI
Comparativa reproducible evaluada sobre contratos sintéticos con redacción y estilos no estándar (`style="diverse"`) tras el retiro formal de la serie Gemini 2.5 en Vertex AI el 20 de octubre de 2026.

* **Comando de Reproducción:** `uv run python evals/bakeoff.py --sample-size 5 --output evals/reports/` (ejecutado con credenciales ADC en Vertex AI sobre muestra $n=5$ diversa; objetivo de validación pre-staging: $n \ge 30$)
* **Reporte Oficial:** [`evals/reports/bakeoff_latest.json`](evals/reports/bakeoff_latest.json)

| Modelo / Candidato | Proveedor | Latencia Media | Costo / 1k docs | Error% | Payment Terms | Rate Sheet | Exactitud Global Conjunta |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rule-Based Baseline (Regex/Parsing)** | `rule_based` | **0.8 ms** | **$0.00** | 0.0% | **100.0%** | **0.0%** | **0.0%** |
| **Gemini 3.5 Flash (GA Active)** | `vertex_ai` | **7836.5 ms** | **$0.20** | 0.0% | **100.0%** | **100.0%** | **100.0%** |
| **Gemini 3.8 Flash (GA Active)** | `vertex_ai` | **22889.6 ms** | **$0.26** | 0.0% | **100.0%** | **100.0%** | **100.0%** |

> **Diagnóstico Forense y Fidelidad de la Medición:**
> En los contratos con redacción variada (`style="diverse"`), el extractor por reglas deterministas captura con éxito los plazos de pago estándar (100%), pero **colapsa al 0% en la tabla de tarifas** debido a la dispersión de estructuras narrativas, llevando la exactitud conjunta a 0.0%. En los contratos evaluados en vivo con Vertex AI mediante esquema estructurado estricto, **Gemini 3.5 Flash** y **Gemini 3.8 Flash** recuperaron fielmente las tarifas tabulares y los plazos netos.
>
> **Salvaguarda de Ciclo de Vida y Fail-Fast:**
> - `gemini-3.5-flash` es GA en Vertex AI con fecha de retiro fijada para el **19 de mayo de 2027 o posterior** (región endpoint: `global`).
> - `gemini-3.8-flash` es GA con retiro para el **2 de septiembre de 2027 o posterior** (región endpoint: `global`).
> - Se implementó un canario semanal en GitHub Actions (`.github/workflows/model_canary.yml`) y sonda de verificación local (`make check-models`).

---

### 4. Seguridad de la API y Control Estricto de Gasto
Para prevenir la degradación de presupuesto y proteger los modelos en Vertex AI:

1. **API Privada en Cloud Run (IAM de Mínimo Privilegio):**
   El servicio Cloud Run de la API elimina `allUsers` y exige autenticación IAM. Únicamente la cuenta de servicio de la UI (`catcher-run-sa`) con el rol `roles/run.invoker` tiene autorización de invocación directa servicio a servicio.
2. **Protección de `/api/audit` con PIN y Rate Limiting:**
   El endpoint `/api/audit` valida la cabecera `X-API-Key` o `Authorization: Bearer <PIN>` sincronizada desde Secret Manager (`demo-access-pin`) con comparación en tiempo constante (`secrets.compare_digest`). Incluye un limitador de tasa deslizante que extrae la IP real del cliente detrás del proxy de Cloud Run (`X-Forwarded-For`), fijado a 30 peticiones por minuto.
3. **Fallback Seguro en Modo Público (Zero-Cost Stub):**
   Las peticiones no autenticadas que lleguen por acceso público ejecutan automáticamente la auditoría utilizando el proveedor `stub` offline ($0.00 de costo, sin llamadas a Vertex AI), garantizando que ningún actor externo pueda drenar la cuota de Google Cloud. En entornos de producción, la UI implementa cierre estricto (*fail-closed*) si el PIN no se encuentra configurado.
4. **Alertas Presupuestarias y Circuit Breaker:**
   Terraform provisiona una alerta de facturación en Google Cloud al 50%, 80% y 100% del presupuesto mensual configurado (`spend_basis = "CURRENT_SPEND"`).

---

## 🎯 Metodología de Calibración de Bartleby: Brecha de Transferencia de Dominio

El guardia de abstención fue sometido a calibración empírica tanto en el dominio sintético original como en contratos auténticos de la SEC ([`evals/reports/calibration_cuad_report.json`](evals/reports/calibration_cuad_report.json)):

### Comparativa Multi-Dominio (Sintético vs CUAD Real)

| Dominio Evaluado | Dataset de Origen | Umbral Calibrado ($\tau$) | Macro F1 Validación | AUC-ROC Separabilidad | Desempeño Empírico en Transferencia |
|---|---|:---:|:---:|:---:|---|
| **Plantillas Sintéticas (+ Heurística 3a*)** | Datagen M1 (300 paquetes) | **0.50** | **1.0000** | **0.1204** | F1 1.0000 debido a regex de denegación 3a ("under no circumstances"). Score puro sin 3a da F1 0.4737 (abstención 0.0%). |
| **Contratos SEC Reales (Transfer)** | CUAD v1 Train/Val (60/40) | **0.50** (fijado) | **0.4418** | **0.5552** | **Sub-abstención:** ante ausencias solo abstiene 26.9%, dejando pasar el 73.1% como alucinaciones. |
| **Contratos SEC Reales (Recalibrado)** | CUAD v1 Train/Val (60/40) | **0.80** (recalibrado) | **0.4982** | **0.5552** | Eleva abstención en ausencias a 80.8%, pero incrementa falsas abstenciones al 80.6% (vs clasificador ciego: 0.4262). |

> **Diagnóstico Empírico Fundamental: El Límite de la Similitud Vectorial y Separabilidad AUC-ROC**
> En datos sintéticos controlados, un umbral de relevancia $\tau = 0.50$ logra un Macro F1 de **1.0000** en validación retenida (gracias al filtro de negación explícita de la Condición #3a de Bartleby), aun cuando el puntaje de recuperación léxica por sí solo invierte la señal (AUC-ROC **0.1204** debido al alto solapamiento de palabras clave en cláusulas que prohíben recargos). Sin embargo, al transferir directamente ese umbral a contratos reales de la SEC (CUAD, evaluado sobre 50 contratos con BM25 + embeddings stub; Train: 30, Val: 20, semilla 42):
>
> 1. **Colapso de Separabilidad:** El AUC-ROC no paramétrico (Wilcoxon-Mann-Whitney) es de **0.5552** en validación (0.4940 en entrenamiento). Esto demuestra matemáticamente que la distribución de puntajes híbridos (BM25 + embeddings hash) entre cláusulas presentes y cláusulas ausentes es **casi indistinguible del azar** (~0.50).
> 2. **Falso Óptimo y Clasificador Trivial:** A $\tau = 0.50$, Bartleby sufre de **sub-abstención** (deja pasar el 73.1% de ausencias). Al barrer la rejilla de 0.20 a 0.90, elevar el umbral a $\tau = 0.80$ mejora el Macro F1 a 0.4982, pero a costa de rechazar erróneamente el 80.6% de las cláusulas presentes. Además, este desempeño apenas supera en un marginal 0.07 al clasificador trivial ciego ("Siempre Abstenerse", Macro F1 = 0.4262).
>
> **Conclusión Arquitectónica:** Ningún umbral estático sobre recuperación basada en BM25 / embeddings de hash es suficiente para gobernar contratos comerciales reales. Este resultado justifica y motiva la **Condición #3 de Bartleby: Verificación Semántica por LLM (`verify_relevance_with_llm`)**, evaluada en [`evals/reports/bartleby_llm_verification_report.json`](evals/reports/bartleby_llm_verification_report.json), donde el modelo evalúa materialmente la pertinencia semántica del pasaje antes de validar la evidencia contractual.

### Evaluación de la Condición #3 de Bartleby: Verificación Semántica por LLM

Medición empírica en vivo sobre 15 contratos auténticos de la SEC (CUAD, 105 evaluaciones de consulta) comparando el filtro de umbral puro frente a la verificación semántica activa con Vertex AI (`gemini-3.5-flash`), inspeccionando los candidatos del top-5 (`max_verification_candidates=5`):

| Modo de Guardia | $\tau$ | Verificación LLM | Abstención Correcta (Recall) | Falsa Abstención (Error) | Macro F1 | Latencia Media | Costo Estimado / 1k* |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pure Vector Hybrid (tau=0.50)** | **0.50** | `Desactivada` | **27.9% (22/79)** | **0.0% (0/26)** | **0.4564** | 35.9 ms | $0.0000 |
| **Pure Vector Hybrid (tau=0.70)** | **0.70** | `Desactivada` | **40.5% (32/79)** | **26.9% (7/26)** | **0.4777** | 37.3 ms | $0.0000 |
| **Bartleby Condition 3 (tau=0.50 + LLM Verification)** | **0.50** | `Gemini 3.5 Flash` | **63.3% (50/79)** | **46.2% (12/26)** | **0.5575** | 4511.1 ms | $2.4094 |

*\*Costo estimado a partir de tokens medidos (100,766 prompt / 11,315 candidate, incluyendo tokens de pensamiento contabilizados como salida) y tarifas pay-as-you-go oficiales en Vertex AI ($1.50 / $9.00 por millón de tokens para Gemini 3.5 Flash según [`config/pricing.yaml`](config/pricing.yaml)). Con Gemini 3.8 Flash, el costo estimado resulta en ~$0.88/1k con la tarifa introductoria vigente ($0.75 / $3.75 hasta el 31 de diciembre de 2026) y en ~$1.77/1k con la tarifa estándar ($1.50 / $7.50 a partir del 1 de enero de 2027).*

> **Diagnóstico Empírico: Rendimiento Condicionado del Verificador y Cuello de Botella de Recuperación**
> 1. **Inspección de Casos Condicionados a Recuperación:** En los 26 casos con cláusula presente ([`evals/reports/bartleby_llm_verification_report.json`](evals/reports/bartleby_llm_verification_report.json)), cuando la recuperación ubicó la cláusula correcta dentro del top-5 (16 casos), el verificador LLM confirmó **12 de 16 casos (75.0%)**. Cuando la recuperación fracasó por completo trayendo únicamente fragmentos irrelevantes (10 casos), el verificador **rechazó correctamente 8 de los 10 pasajes (80.0%)**, actuando como filtro de distractores espurios.
> 2. **El Cuello de Botella es la Recuperación:** En los 26 casos presentes, la búsqueda léxica/hash ubica la cláusula correcta en el rango #1 solo en el **26.9%** de los casos (7/26), y en el top-5 solo en el **61.5%** (16/26). La tasa de falsa abstención (46.2%, 12/26) está gobernada principalmente por las ausencias en el top-5 recuperado.
> 3. **Ganancia Neta en Discriminación:** El filtro semántico eleva la abstención correcta ante ausencias reales del 27.9% al **63.3%** (50/79), reduciendo la falsa abstención al **46.2%** (12/26 frente a 17/26 en la corrida inicial sin enriquecimiento de categorías) y alcanzando el Macro F1 más alto de todas las arquitecturas evaluadas (**0.5575**).
> 4. **Trade-off Operativo:** Añade ~4.5 segundos de latencia media por consulta y un costo proyectado de ~$2.41 por cada 1,000 consultas auditadas en Gemini 3.5 Flash (~$0.88 en Gemini 3.8 Flash). Fijar `thinking_budget: 0` reduce la latencia de 5,153 ms a 4,511 ms (una mejora del 12.5%), suprimiendo tokens de pensamiento redundantes en clasificación binaria.

1. **Split Train / Validation:** Partición reproducible 60/40 (semilla 42) sobre 50 contratos de la SEC para evaluar transferencia sin sobreajuste.
2. **Métricas sin Umbral (AUC-ROC):** Mide la capacidad intrínseca del ranking de recuperación de separar presencia vs ausencia con independencia del umbral.
3. **Contraste contra Baselines Triviales:** Evaluación estricta frente a clasificadores ciegos (*Always Abstain* y *Never Abstain*).
4. **Condición #3 (Verificación Semántica por LLM):** El guardia combina el filtro de similitud con una verificación semántica que inspecciona si el fragmento efectivamente otorga o niega la cláusula consultada.

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

# 4. Correr la suite completa de 69 tests unitarios e integrales (84% de cobertura)
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
│   ├── models.yaml            # Configuración de modelos GA activos (Gemini 3.5 Flash)
│   ├── thresholds.yaml        # Umbrales empíricos (Bartleby: 0.50 / 1 span)
│   └── policy_rules.yaml      # Reglas deterministas de aprobación financiera
├── infra/                     # Infraestructura como Código (IaC) y contenedor
│   ├── Dockerfile             # Imagen hardened (usuario appuser sin root, curl healthcheck)
│   ├── entrypoint.sh          # Entrypoint dual (API/UI) con inicialización reproducible
│   └── terraform/             # Despliegue en Cloud Run v2, IAM de mínimo privilegio y alerta presupuestaria
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
└── tests/                     # Suite de 69 tests unitarios e integrales (84% cobertura)
```

---

## 📜 Procedencia de Datos y Licencias

* **CUAD v1:** *Contract Understanding Atticus Dataset*, publicado por The Atticus Project bajo licencia [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
* **OCDS 1.1:** *Open Contracting Data Standard*, mantenido por la Open Contracting Partnership bajo licencia [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
* **Atribución Literaria:** El título *The Catcher in the RAG* es empleado únicamente como parodia técnica. No se reproduce texto, citas ni imágenes de la obra original de J.D. Salinger. Las citas alusivas a *Bartleby, the Scrivener* (Herman Melville, 1853) pertenecen al dominio público.

---

## ⚖️ Licencia

Distribuido bajo la licencia **MIT**. Consulte el archivo [LICENSE](LICENSE) para más información.
