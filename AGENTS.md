# ORDEN FINAL — ÚLTIMA RONDA, PRESUPUESTO CERRADO

> Reemplaza a `REPAIR_ORDER.md`. Guardar como `AGENTS.md`. Leer completo antes de tocar nada.
> Esta es la ÚLTIMA ronda de reparación. Se acabaron las rondas.

---

## 0. POR QUÉ ESTÁS AQUÍ

En la última ronda fabricaste "casos crudos" (`Category_1..12`, latencia idéntica en 420 filas, `response_id` secuenciales, `model_version` inventado) para que la puerta de evidencia pasara. Eso es falsificar evidencia. No se repite, ni "reconstruido", ni "generado", ni "armonizado".

Cada ronda anterior costó tiempo, dinero y créditos. **No se resuelve con más capas de verificación. Se resuelve borrando lo falso, registrando datos reales en UN solo punto y parando.**

---

## 1. REGLAS DE ESTA RONDA (inviolables)

1. **PROHIBIDO crear archivos de test nuevos.** PROHIBIDO crear scripts de verificación nuevos. PROHIBIDO ampliar la "puerta de evidencia". Se queda como está (sin debilitarla).
2. **PROHIBIDO fabricar, reconstruir, rellenar, estimar o "armonizar" datos o reportes.** Si no se midió: `NOT MEASURED`.
3. **PROHIBIDO editar a mano `evals/reports/*`.** Solo el harness escribe ahí.
4. **UNA sola corrida en vivo por evaluación** (secciones 4.1 y 4.2). Ninguna otra evaluación en vivo. Si falla por cuota o credenciales: pega el error literal y PARA. No simules.
5. **Suite completa de tests: UNA vez, al final.** Mientras trabajas, solo corre los tests del archivo que tocaste, con `-q`. Nada de `coverage` completo en la salida: usa `--no-cov | tail -5`.
6. **No pegues salidas largas.** Máximo 20 líneas por comando.
7. Si algo es ambiguo: elige la opción MÁS SIMPLE, anótala en una línea y sigue. No preguntes. No inventes.
8. Si algo falla fuera de estos pasos: anótalo en "pendientes" y NO lo arregles.

---

## 2. PASO 1 — BORRAR LO FALSO (no crees nada)

1. `git rm evals/reports/bartleby_llm_verification_cases.jsonl evals/reports/out_of_sample_cuad_cases.jsonl`
2. Restaurar los dos reportes tal como los generó el harness (sin `raw_cases_sha256`):
   `git checkout b678adb -- evals/reports/bartleby_llm_verification_report.json evals/reports/out_of_sample_cuad_report.json`
3. En `evals/repro_check.py`: eliminar SOLO la lectura de esos JSONL y el chequeo `check_out_of_sample_cuad_report` basado en ellos. Registrar ambos reportes como **solo coherencia aritmética** (misma verificación que tenían antes de la ronda 2). Que `DENIED BY DEFAULT` siga activo para cualquier otro JSON.
4. Eliminar el target `repro-regen` (no regenera nada; es un alias de `repro_check`). No lo reemplaces.
5. En README, `StateOfCatcher.md`, `AGENTS.md`: borrar toda afirmación de "verificación por casos crudos", "hash criptográfico", "recálculo caso por caso" y "CODEOWNERS protege". En README, marcar el modo 3 y el resultado fuera de muestra como: **"reportado; sin registro crudo por llamada (se re-mide en el paso 4)"**.

---

## 3. PASO 2 — UN SOLO CAMBIO DE CÓDIGO: REGISTRO REAL DE LLAMADAS (≤ 60 líneas)

En `src/catcher_in_the_rag/providers/gemini.py`, en el ÚNICO lugar donde se hace la llamada al modelo, añade un registro de solo-anexar. Por cada llamada REAL, una línea JSON en `evals/reports/raw/<run_id>.jsonl` con:

`run_id`, `timestamp_utc` (reloj real, con microsegundos), `prompt_sha256`, `response_text` (literal), `response_id` (de `response.response_id`), `model_version` (de `response.model_version`), `prompt_tokens`, `candidate_tokens`, `thought_tokens` (de `usage_metadata`), `latency_ms` (medida con `time.perf_counter`), y los campos que pase quien llama (`query`, `category`, `contract_id`, `passage_ids`).

- Si el SDK no devuelve `response_id` o `model_version`: escribe `null`. **Nunca los inventes.**
- Ningún otro código escribe en `evals/reports/raw/`.
- **Única comprobación permitida:** una llamada manual de humo con 3 consultas. Pega las 3 líneas resultantes en el informe. Nada más.

---

## 4. PASO 3 — DOS CORRIDAS EN VIVO, UNA VEZ CADA UNA

Solo después de que el paso 2 esté commiteado, con el árbol limpio:

1. `make eval-bartleby-llm` (modo 3, 105 consultas).
2. `uv run python evals/eval_out_of_sample.py --start 15 --end 50 --llm-provider vertex_ai` (420 consultas). Si los argumentos cambiaron, confirma con `--help` y nada más.

Commitea los reportes y los `raw/*.jsonl` **exactamente como se generaron**. Costo esperado: unos 2 dólares en total. Si la facturación de Vertex de la corrida supera 4 dólares, PARA y reporta.

---

## 5. PASO 4 — README MÍNIMO Y HONESTO (≤ 120 líneas)

Reescribe el README:
- Qué es y cómo correrlo.
- Arquitectura: SOLO lo que existe en el código.
- Resultados medidos (con intervalo de Wilson), cada uno con archivo y comando. Números solo dentro de bloques `GEN` (el mecanismo actual).
- Limitaciones: embeddings stub en las mediciones de CUAD; la Regla 3a es una heurística de plantilla; el 100% sintético NO es evidencia; lista de `NOT MEASURED`.
- Eliminar el titular de 100% sintético y el "+40% valor RAG".

---

## 6. PASO 5 — CIERRE

1. `make verify` UNA vez. Pega las últimas 15 líneas.
2. Si falla algo ajeno a los pasos 1–4: anótalo en "pendientes". No lo arregles.
3. **PARA.** No abras nuevas fases.

---

## 7. DESPUÉS (solo con los pasos 1–5 cerrados): ESCALAR, SIN CAPAS NUEVAS

1. **Recuperación (lo único que mueve el resultado):** embeddings `text-embedding-005` de Vertex con la caché ya existente. Una ablación (bm25 / denso / híbrido / híbrido + reranker) sobre los 156 casos de presencia fuera de muestra, usando el mismo registro de llamadas. Una corrida. README actualizado desde el JSON.
2. **Staging:** `terraform plan` en CI, el `apply` lo hace el usuario a mano, Cloud Run, presupuesto con apagado automático, y un script de humo de ≤ 20 líneas.
3. Nada más. Ni fases, ni auditorías, ni rondas de red-team.

---

## 8. INFORME FINAL DEL AGENTE (≤ 15 líneas, sin adjetivos)

- `git diff --stat` (resumido).
- Comandos ejecutados con su salida (máx. 20 líneas c/u).
- Gasto real de las corridas en vivo.
- Lista de "pendientes" y "NOT MEASURED".

**Consecuencia:** cualquier dato sintetizado, reconstruido o inventado → se revierte todo el trabajo de esta ronda y se da por terminado.

---

## PENDIENTES

- extract sobrescribe rate_sheet/expiration_date con vacío cuando el extractor no los encuentra; no se tocó porque afecta medidas.
- infra/terraform/main.tf pasa GOOGLE_CLOUD_REGION=var.region a Cloud Run (API y UI); los modelos Flash de GLOBAL_ONLY_MODELS (providers/gemini.py) ya la ignoran (siempre global) y solo afecta a embeddings. No se tocó Terraform.
- bake-off medido con el esquema anterior (campos obligatorios); no re-medido.
