# The Catcher in the RAG

Auditor determinista de compras corporativas (procurement) con verificación contractual y guardia de abstención Melville (Bartleby).

## Arquitectura y Ejecución
- **Pipeline:** Extracción (`extract/`), cotejo tripartito (`match/`), recuperación híbrida BM25/vectorial (`retrieve/`), guardia de abstención (`bartleby/`), evaluación de términos (`apply_terms/`), validación verbatim (`validate/`) y motor de políticas (`policy/`).
- **Proveedores:** `GeminiProvider` en Google Cloud Vertex AI (autenticación ADC) y `StubProvider` determinista offline.
- **Ejecución:** Pruebas offline con `make test`, puerta completa con `make verify`, UI en `make run-ui` (puerto 8501) y API en `make run-api` (puerto 8000).

## Limitaciones
- Embeddings stub (`stub-hash-768d`) en las mediciones de CUAD salvo la calibración (sección 2, contratos 1–15) y la sección 4 (`text-embedding-005`); la Regla 3a es una heurística de plantilla sintética; los datos sintéticos no constituyen evidencia empírica de producción.
- **NOT MEASURED:** abstención fuera de muestra (contratos 16–50) con el τ recalibrado, y latencia en red distribuida. El τ recalibrado no está aplicado en `config/thresholds.yaml`.
- **Reranker:** no medido, desactivado.

## Resultados Medidos

### 1. Benchmark CUAD (`make eval-cuad` -> `evals/reports/cuad_benchmark_latest.json`)
<!-- GEN:cuad_table -->
| Métrica | Modo Offline (`stub-hash-768d`) | Vertex AI (`text-embedding-005` Live) | Intervalo de Confianza (95% Wilson Stub) | Interpretación y Significado Práctico |
|---|:---:|:---:|:---:|---|
| **Recall@1** | 35.5% | NOT MEASURED | — | Precisión de primer ranking en contratos no estructurados. |
| **Recall@3** | 72.4% | NOT MEASURED | — | Cláusula relevante localizada dentro del top 3 de candidatos. |
| **Recall@5** | 81.6% | NOT MEASURED | — | Cobertura agregada en contratos extensos (50k a 175k caracteres). |
| **MRR** | 0.5336 | NOT MEASURED | — | Posición recíproca media del fragmento probatorio real. |
| **Abstención en Negativos** | **51.5%** | NOT MEASURED | `[41.8%, 61.1%]` | Ante cláusulas ausentes reales (`answers: []`), Bartleby rehúsa alucinar. |
| **Abstención Errónea (Positivos)** | **14.5%** | NOT MEASURED | `[8.3%, 24.1%]` | Tasa de sobre-abstención en presencia de cláusulas legítimas válidas. |
| **Falsa Afirmación Espuria** | **48.5%** | NOT MEASURED | — | Riesgo residual de afirmar falsamente una penalización inexistente. |
<!-- /GEN -->

### 2. Calibración de Dominio (`MAX_LIVE_CALLS=80 uv run python -m evals.calibrate_cuad --embedding-provider vertex_ai` -> `evals/reports/calibration_cuad_report.json`)
<!-- GEN:domain_calibration_table -->
| Dominio Evaluado | Dataset de Origen | Umbral Calibrado ($\tau$) | Macro F1 Validación | AUC-ROC Separabilidad | Desempeño Empírico en Transferencia |
|---|---|:---:|:---:|:---:|---|
| **Plantillas Sintéticas (+ Heurística 3a*)** | Datagen M1 (300 paquetes; no es evidencia) | **0.50** | **1.0000** | **0.1204** | F1 con regex de denegación 3a: 1.0000. Score puro sin 3a: F1 0.4737 (abstención 0.0%). |
| **Contratos SEC Reales (Transfer)** | CUAD v1, contratos 1–15, Train/Val 60/40 | **0.50** (fijado) | **0.4496** | **0.5909** | Abstención correcta en ausencias 25.8% (74.2% pasan). |
| **Contratos SEC Reales (Recalibrado)** | CUAD v1, contratos 1–15, Train/Val 60/40 | **0.80** (recalibrado, no aplicado) | **0.5457** | **0.5909** | Abstención correcta 54.8%, falsas abstenciones 36.4% (Siempre Abstenerse: 0.4247). |
<!-- /GEN -->
<!-- GEN:domain_calibration_diagnosis -->
> **Diagnóstico de calibración en CUAD** (`evals/reports/calibration_cuad_report.json`)
> Sintético (referencia, no evidencia): $\tau = 0.50$, Macro F1 validación 1.0000, AUC-ROC 0.1204.
> CUAD: 15 contratos (entrenamiento 9, validación 6, semilla 42), embeddings `text-embedding-005`.
>
> 1. **AUC-ROC (Wilcoxon-Mann-Whitney):** entrenamiento 0.7708 [IC95 0.6202–0.9214]; validación 0.5909 [IC95 0.3885–0.7933]; el intervalo de validación incluye 0.50: no distinguible del azar en esta muestra.
> 2. **Umbral:** a $\tau = 0.50$ la abstención correcta en ausencias de validación es 25.8%. A $\tau = 0.80$ (óptimo de entrenamiento) es 54.8%, con 36.4% de falsas abstenciones y Macro F1 0.5457 (Siempre Abstenerse: 0.4247; diferencia +0.1210).
<!-- /GEN -->

### 3. Verificación Semántica LLM (`make eval-bartleby-llm` -> `evals/reports/bartleby_llm_verification_report.json`)
<!-- GEN:llm_verification_table -->
| Modo de Guardia | $\tau$ | Verificación LLM | Abstención Correcta (Recall) | Falsa Abstención (Error) | Macro F1 | Latencia Media | Costo Estimado / 1k* |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pure Vector Hybrid (tau=0.50)** | **0.50** | `Desactivada` | **27.9% (22/79)** | **0.0% (0/26)** | **0.4564** | 17.1 ms | $0.0000 |
| **Pure Vector Hybrid (tau=0.70)** | **0.70** | `Desactivada` | **40.5% (32/79)** | **26.9% (7/26)** | **0.4777** | 16.9 ms | $0.0000 |
| **Bartleby Condition 3 (tau=0.50 + LLM Verification)** | **0.50** | `Gemini 3.5 Flash` | **63.3% (50/79)** | **42.3% (11/26)** | **0.5714** | 3303.5 ms | $2.3367 |
<!-- /GEN -->
<!-- GEN:llm_verification_footnote -->
*\*Costo estimado a partir de tokens medidos (98,140 prompt / 10,905 candidate, incluyendo tokens de pensamiento contabilizados como salida) y tarifas pay-as-you-go oficiales en Vertex AI ($1.50 / $9.00 por millón de tokens para Gemini 3.5 Flash según [`config/pricing.yaml`](config/pricing.yaml)). Con Gemini 3.8 Flash, el costo estimado resulta en ~$0.88/1k con la tarifa introductoria vigente ($0.75 / $3.75 hasta el 31 de diciembre de 2026) y en ~$1.77/1k con la tarifa estándar ($1.50 / $7.50 a partir del 1 de enero de 2027).*
<!-- /GEN -->
<!-- GEN:mode3_diagnosis -->
> **Bartleby: tres afirmaciones medidas**
> (a) **El score de recuperación solo es una señal débil.** AUC-ROC de validación 0.5909 [IC95 0.3885–0.7933] con `text-embedding-005` (CUAD, contratos 1–15, `calibration_cuad_report.json`); el intervalo incluye 0.50.
>
> (b) **La verificación con LLM aporta la abstención.** Con los mismos embeddings y $\tau$ = 0.50 (`bartleby_llm_verification_report.json`), la abstención correcta en ausencias pasa de 22/79 (27.9%) sin LLM a 50/79 (63.3%) con LLM.
>
> (c) **Las falsas abstenciones siguen altas.** Con LLM, fuera de muestra: 74/156 = 47.4% [IC95 39.8–55.2] de las cláusulas presentes (`out_of_sample_cuad_report.json`). Cada falsa abstención es una revisión humana innecesaria: 74 en 420 consultas (17.6 por cada 100). Tiempo o costo por revisión: NOT MEASURED.
<!-- /GEN -->

### 3b. Evaluación Fuera de Muestra (`uv run python evals/eval_out_of_sample.py --start 15 --end 50 --llm-provider vertex_ai` -> `evals/reports/out_of_sample_cuad_report.json`)
Reportado con `git_dirty: true`. Según el operador, solo estaba sin commitear el reporte hermano de Bartleby; git no registra qué archivos estaban modificados. El registro crudo (`evals/reports/raw/out_of_sample_cuad.jsonl`) no trae `query` ni `contract_id` por llamada y su número de líneas no coincide con el número de consultas del reporte, así que los conteos no se pueden cruzar consulta por consulta. No se repitió por costo.
<!-- GEN:oos_bartleby_table -->
| Configuración | $\tau$ | Embeddings | Abstención correcta (ausencias) | Falsa abstención (presencias) | Macro F1 | Reporte |
|---|:---:|---|:---:|:---:|:---:|---|
| Sin LLM, solo score | 0.50 | `vertex_ai` | 63/264 = 23.9% [IC95 19.1–29.4] | 8/156 = 5.1% [IC95 2.6–9.8] | 0.4811 | `out_of_sample_cuad_stub_tau0.50_report.json` |
| Sin LLM, solo score | 0.80 | `vertex_ai` | 221/264 = 83.7% [IC95 78.8–87.7] | 72/156 = 46.2% [IC95 38.5–54.0] | 0.6936 | `out_of_sample_cuad_stub_tau0.80_report.json` |
| Con verificación LLM (3b) | 0.50 (fijo en el eval) | `stub (valor por defecto del comando)` | 248/264 = 93.9% [IC95 90.4–96.2] | 74/156 = 47.4% [IC95 39.8–55.2] | 0.7460 | `out_of_sample_cuad_report.json` (git_dirty) |
| Siempre abstenerse (trivial) | — | — | 264/264 = 100.0% | 156/156 = 100.0% | 0.3860 | calculado de los conteos |
<!-- /GEN -->

### 4. Recuperación fuera de muestra (`MAX_LIVE_CALLS=20 uv run python evals/eval_out_of_sample.py --retrieval-ablation --embedding-provider vertex_ai --start 15 --end 50` -> `evals/reports/retrieval_ablation_oos_report.json`)
<!-- GEN:retrieval_table -->
156 casos de presencia, CUAD-0016 to CUAD-0050 (35 contracts), embeddings `text-embedding-005`, commit `3cdff0d`. Empates resueltos por orden de aparición en el contrato.

| Método | Hit@1 | Hit@5 |
|---|:---:|:---:|
| **BM25** | 30.1% (47/156) [IC95 23.5–37.7] | 68.6% (107/156) [IC95 60.9–75.3] |
| **Denso** | 55.8% (87/156) [IC95 47.9–63.3] | 85.9% (134/156) [IC95 79.6–90.5] |
| **Híbrido (50/50)** | 55.1% (86/156) [IC95 47.3–62.7] | 84.6% (132/156) [IC95 78.1–89.4] |

- Hit@1: el intervalo de Wilson de BM25 no se solapa con los demás; los tres métodos no son indistinguibles en esta muestra. Denso y Híbrido (50/50): indistinguibles en esta muestra.
- Hit@5: el intervalo de Wilson de BM25 no se solapa con los demás; los tres métodos no son indistinguibles en esta muestra. Denso y Híbrido (50/50): indistinguibles en esta muestra.
<!-- /GEN -->

### 5. Bake-Off de Modelos (`evals/bakeoff.py` -> `evals/reports/bakeoff_latest.json`)
<!-- GEN:bakeoff_table -->
| Modelo / Candidato | Proveedor | Latencia Media | Costo / 1k docs | Error% | Payment Terms | Rate Sheet | Exactitud Global Conjunta |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rule-Based Baseline (Regex/Parsing)** | `rule_based` | **2.0 ms** | **$0.00** | 0.0% | **100.0%** | **0.0%** | **0.0%** |
| **Gemini 3.5 Flash (GA Active)** | `vertex_ai` | **2299.6 ms** | **$4.67** | 0.0% | **100.0%** | **100.0%** | **100.0%** |
| **Gemini 3.8 Flash (GA Active)** | `vertex_ai` | **48095.9 ms** | **$2.15** | 0.0% | **100.0%** | **100.0%** | **100.0%** |
<!-- /GEN -->
<!-- GEN:bakeoff_diagnosis -->
> **Diagnóstico Forense y Fidelidad de la Medición:**
> En los contratos con redacción variada (`style="diverse"`), el extractor por reglas deterministas captura con éxito los plazos de pago estándar (100%), pero **colapsa al 0% en la tabla de tarifas** debido a la dispersión de estructuras narrativas, llevando la exactitud conjunta a 0.0%. En los contratos evaluados en vivo con Vertex AI mediante esquema estructurado estricto, **Gemini 3.5 Flash** y **Gemini 3.8 Flash** recuperaron fielmente las tarifas tabulares y los plazos netos.
>
> **Dinámica de Latencia y Razonamiento:** Con `thinking_budget: 0`, Gemini 3.5 Flash alcanza una latencia media de **2299.6 ms** por extracción de documento complejo, mientras que Gemini 3.8 Flash promedió **48.1 segundos** (hipótesis preliminar a investigar formalmente en Fase 4: sobrecosto de latencia por deliberación interna o reintentos de endpoint, aún no medido de forma aislada). Los costos se calcularon rigurosamente a partir de tarifas GA en Vertex AI ($1.50 / $9.00 por 1M tokens para 3.5 Flash; $0.75 / $3.75 introductorio para 3.8 Flash).
>
> **Salvaguarda de Ciclo de Vida y Fail-Fast:**
> - `gemini-3.5-flash` es GA en Vertex AI con fecha de retiro fijada para el **19 de mayo de 2027 o posterior** (región endpoint: `global`).
> - `gemini-3.8-flash` es GA con retiro para el **2 de septiembre de 2027 o posterior** (región endpoint: `global`).
> - Se implementó un canario semanal en GitHub Actions (`.github/workflows/model_canary.yml`) y sonda de verificación local (`make check-models`).
<!-- /GEN -->

### 6. Control Sintético de Integridad (`make eval-ablations` -> `evals/reports/canonical_baseline_report.json`)
<!-- GEN:synthetic_ablation_table -->
| Métrica de Evaluación | Con Términos Contractuales (Baseline) | Sin Términos (Solo Cotejo 3-Way) | Impacto / Delta |
|---|:---:|:---:|:---:|
| **Exactitud Global de Decisión** | **100.0%** (300/300) | **60.0%** (180/300) | **ablación: sin evaluador de cláusulas (+40.0% degradación)** |
| **Validez de Citas Verbatim** | **100.0%** (120/120) [plantillas sintéticas] | N/A | Cero citas sintéticas inválidas |
| **Abstención Correcta de Bartleby** | **100.0%** (30/30 en A09*) | 0.0% (0/30 en A09) | Detección de silencios (vía heurística de plantilla 3a*) |
| **Tasa de Falsos Positivos (A10)** | **0.0%** (0/30) | 0.0% (0/30) | Paquetes limpios aprobados sin fricción |
| **Recall@5 de Recuperación** | **100.0%** [plantillas sintéticas] | N/A | Recuperación en plantillas sintéticas |
| **Latencia Media por Paquete** | **16.7 ms** | 0.0 ms | Ejecución determinista local |
<!-- /GEN -->
