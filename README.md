# The Catcher in the RAG

Auditor determinista de compras corporativas (procurement) con verificación contractual y guardia de abstención Melville (Bartleby).

## Arquitectura y Ejecución
- **Pipeline:** Extracción (`extract/`), cotejo tripartito (`match/`), recuperación híbrida BM25/vectorial (`retrieve/`), guardia de abstención (`bartleby/`), evaluación de términos (`apply_terms/`), validación verbatim (`validate/`) y motor de políticas (`policy/`).
- **Proveedores:** `GeminiProvider` en Google Cloud Vertex AI (autenticación ADC) y `StubProvider` determinista offline.
- **Ejecución:** Pruebas offline con `make test`, puerta completa con `make verify`, UI en `make run-ui` (puerto 8501) y API en `make run-api` (puerto 8000).

## Despliegue en Cloud Run (staging)
1. Autenticarse: `gcloud auth login && gcloud auth application-default login`.
2. Construir, subir la imagen y planificar: `PROJECT_ID=<proyecto> REGION=<región> scripts/deploy.sh`. Imprime la URL del servicio y el comando de smoke; no aplica nada.
3. Aplicar a mano tras revisar el plan: `terraform -chdir=infra/terraform apply tfplan`. Luego `scripts/smoke.sh <API_URL>`.

En el primer despliegue el repositorio de Artifact Registry aún no existe: aplica antes `terraform -chdir=infra/terraform apply -target=google_artifact_registry_repository.images -var container_image=none` (la variable es obligatoria; ese apply solo crea el repositorio). Con `MAX_LIVE_CALLS=0` (valor por defecto en Cloud Run) el servicio corre solo con stub.
No ejecutes terraform apply a secas: usa deploy.sh y luego apply tfplan.

## Demo en vivo
Corpus de demostración: 30 paquetes sintéticos (3 por tipo, semilla 42). Las métricas del README se miden sobre 300 paquetes generados con la misma fórmula; la numeración no coincide.

## Limitaciones
- La Regla 3a es una heurística de plantilla sintética; los datos sintéticos no son evidencia empírica.
- `config/thresholds.yaml` usa el τ calibrado en CUAD (sección 4). Con ese τ y embeddings stub, los paquetes sintéticos A06 pasan de REJECT a REVIEW (hallazgo NOT_VERIFIABLE); los tests sintéticos fijan el τ de su propia calibración. La sección 5 refleja ese cambio.
- **NOT MEASURED:** abstención con LLM y `text-embedding-005` fuera de muestra, costo de la revisión humana, latencia en red distribuida.
- **Reranker:** no medido, desactivado.

## Resultados medidos
Cada cifra sale de un bloque `GEN` regenerado desde `evals/reports/*.json` (`make sync-readme-tables`).

### 1. Recuperación fuera de muestra (`MAX_LIVE_CALLS=20 uv run python evals/eval_out_of_sample.py --retrieval-ablation --embedding-provider vertex_ai --start 15 --end 50` -> `evals/reports/retrieval_ablation_oos_report.json`)
<!-- GEN:retrieval_table -->
156 casos de presencia, CUAD-0016 to CUAD-0050 (35 contracts), embeddings `text-embedding-005`, commit `8a68fe2`. Empates resueltos por orden de aparición en el contrato.

| Método | Hit@1 | Hit@5 |
|---|:---:|:---:|
| **BM25** | 30.1% (47/156) [IC95 23.5–37.7] | 68.6% (107/156) [IC95 60.9–75.3] |
| **Denso** | 55.8% (87/156) [IC95 47.9–63.3] | 85.9% (134/156) [IC95 79.6–90.5] |
| **Híbrido (50/50)** | 55.1% (86/156) [IC95 47.3–62.7] | 84.6% (132/156) [IC95 78.1–89.4] |

- Hit@1: el intervalo de Wilson de BM25 no se solapa con los demás; los tres métodos no son indistinguibles en esta muestra. Denso y Híbrido (50/50): indistinguibles en esta muestra.
- Hit@5: el intervalo de Wilson de BM25 no se solapa con los demás; los tres métodos no son indistinguibles en esta muestra. Denso y Híbrido (50/50): indistinguibles en esta muestra.
<!-- /GEN -->

### 2. Abstención de Bartleby fuera de muestra (`MAX_LIVE_CALLS=3 uv run python evals/eval_out_of_sample.py --start 15 --end 50 --embedding-provider vertex_ai --llm-provider stub --tau <τ>`)
<!-- GEN:oos_bartleby_table -->
| Configuración | $\tau$ | Embeddings | Abstención correcta (ausencias) | Falsa abstención (presencias) | Macro F1 | Reporte |
|---|:---:|---|:---:|:---:|:---:|---|
| Sin LLM, solo score | 0.50 | `vertex_ai` | 63/264 = 23.9% [IC95 19.1–29.4] | 8/156 = 5.1% [IC95 2.6–9.8] | 0.4811 | `out_of_sample_cuad_stub_tau0.50_report.json` |
| Sin LLM, solo score | 0.80 | `vertex_ai` | 221/264 = 83.7% [IC95 78.8–87.7] | 72/156 = 46.2% [IC95 38.5–54.0] | 0.6936 | `out_of_sample_cuad_stub_tau0.80_report.json` |
| Con verificación LLM (3b) | 0.50 (fijo en el eval) | `stub (valor por defecto del comando)` | 248/264 = 93.9% [IC95 90.4–96.2] | 74/156 = 47.4% [IC95 39.8–55.2] | 0.7460 | `out_of_sample_cuad_report.json` (git_dirty) |
| Siempre abstenerse (trivial) | — | — | 264/264 = 100.0% | 156/156 = 100.0% | 0.3860 | calculado de los conteos |
<!-- /GEN -->
<!-- GEN:operating_point -->
Sin LLM, $\tau$ = 0.80 (valor actual de `config/thresholds.yaml`) se abstiene en 83.7% de las ausencias (221/264) y en 46.2% de las presencias (72/156). $\tau$ = 0.50 casi nunca se abstiene: 71 de 420 consultas (16.9%), y solo en 23.9% de las ausencias.

> **Nota:** La fila con LLM no es comparable con las filas sin LLM: usó embeddings `stub` y se generó con `git_dirty: true`.
<!-- /GEN -->
<!-- GEN:mode3_diagnosis -->
> **Bartleby: tres afirmaciones medidas**
> (a) **El score de recuperación solo es una señal débil.** AUC-ROC de validación 0.5909 [IC95 0.3885–0.7933] con `text-embedding-005` (CUAD, contratos 1–15, `calibration_cuad_report.json`); el intervalo incluye 0.50.
>
> (b) **La verificación con LLM aporta la abstención.** Con los mismos embeddings y $\tau$ = 0.50 (`bartleby_llm_verification_report.json`), la abstención correcta en ausencias pasa de 22/79 (27.9%) sin LLM a 50/79 (63.3%) con LLM.
>
> (c) **Las falsas abstenciones siguen altas.** Con LLM, fuera de muestra: 74/156 = 47.4% [IC95 39.8–55.2] de las cláusulas presentes (`out_of_sample_cuad_report.json`). Cada falsa abstención es una revisión humana innecesaria: 74 en 420 consultas (17.6 por cada 100). Tiempo o costo por revisión: NOT MEASURED.
<!-- /GEN -->

### 3. Verificación con LLM, modo 3 (`make eval-bartleby-llm` -> `evals/reports/bartleby_llm_verification_report.json`)
Medido con τ fijo en el eval y los embeddings del entorno de esa corrida (el reporte no los registra).
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

### 4. Calibración de τ (`MAX_LIVE_CALLS=80 uv run python -m evals.calibrate_cuad --embedding-provider vertex_ai` -> `evals/reports/calibration_cuad_report.json`)
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

### 5. Control sintético — no es evidencia (`make eval-ablations` -> `evals/reports/canonical_baseline_report.json`)
Plantillas generadas por `datagen/`; medido con el τ de `config/thresholds.yaml` (`LLM_PROVIDER=stub EMBEDDING_PROVIDER=stub MAX_LIVE_CALLS=0 make eval-ablations`).
<!-- GEN:synthetic_ablation_table -->
| Métrica de Evaluación | Con Términos Contractuales (Baseline) | Sin Términos (Solo Cotejo 3-Way) | Impacto / Delta |
|---|:---:|:---:|:---:|
| **Exactitud Global de Decisión** | **91.3%** (274/300) | **60.0%** (180/300) | **ablación: sin evaluador de cláusulas (+31.3% degradación)** |
| **Validez de Citas Verbatim** | **100.0%** (64/64) [plantillas sintéticas] | N/A | Cero citas sintéticas inválidas |
| **Abstención Correcta de Bartleby** | **100.0%** (30/30 en A09*) | 0.0% (0/30 en A09) | Detección de silencios (vía heurística de plantilla 3a*) |
| **Tasa de Falsos Positivos (A10)** | **0.0%** (0/30) | 0.0% (0/30) | Paquetes limpios aprobados sin fricción |
| **Recall@5 de Recuperación** | **100.0%** [plantillas sintéticas] | N/A | Recuperación en plantillas sintéticas |
| **Latencia Media por Paquete** | **16.9 ms** | 0.0 ms | Ejecución determinista local |
<!-- /GEN -->
