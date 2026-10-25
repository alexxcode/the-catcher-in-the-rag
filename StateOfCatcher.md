# State of The Catcher in the RAG
**Document Version:** 0.4.0  
**System Status:** Hardened Evidence Gate Active · Zero-Cost Public Demo Shield  
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

## 2. Métricas Empíricas de Recuperación en CUAD

Evaluación de recuperación y abstención sobre contratos comerciales de la SEC ([`evals/reports/cuad_benchmark_latest.json`](evals/reports/cuad_benchmark_latest.json)):

<!-- GEN:cuad_table -->
| Métrica | Modo Offline (`stub-hash-768d`) | Vertex AI (`text-embedding-005` Live) | Intervalo de Confianza (95% Wilson Stub) | Interpretación y Significado Práctico |
|---|:---:|:---:|:---:|---|
| **Recall@1** | 26.9% | NOT MEASURED | — | Precisión de primer ranking en contratos no estructurados. |
| **Recall@3** | 46.2% | NOT MEASURED | — | Cláusula relevante localizada dentro del top 3 de candidatos. |
| **Recall@5** | 61.5% | NOT MEASURED | — | Cobertura agregada en contratos extensos (50k a 175k caracteres). |
| **MRR** | 0.3995 | NOT MEASURED | — | Posición recíproca media del fragmento probatorio real. |
| **Abstención en Negativos** | **27.9%** | NOT MEASURED | `[18.9%, 39.0%]` | Ante cláusulas ausentes reales (`answers: []`), Bartleby rehúsa alucinar. |
| **Abstención Errónea (Positivos)** | **65.4%** | NOT MEASURED | `[46.2%, 80.6%]` | Tasa de sobre-abstención en presencia de cláusulas legítimas válidas. |
| **Falsa Afirmación Espuria** | **0.0%** | NOT MEASURED | — | Riesgo residual de afirmar falsamente una penalización inexistente. |
<!-- /GEN -->

---

## 3. Bake-Off de Extracción en Vertex AI

Evaluación sobre contratos con redacción diversa (`style="diverse"`) ([`evals/reports/bakeoff_latest.json`](evals/reports/bakeoff_latest.json)):

<!-- GEN:bakeoff_table -->
| Modelo / Candidato | Proveedor | Latencia Media | Costo / 1k docs | Error% | Payment Terms | Rate Sheet | Exactitud Global Conjunta |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rule-Based Baseline (Regex/Parsing)** | `rule_based` | **2.0 ms** | **$0.00** | 0.0% | **100.0%** | **0.0%** | **0.0%** |
| **Gemini 3.5 Flash (GA Active)** | `vertex_ai` | **2299.6 ms** | **$4.67** | 0.0% | **100.0%** | **100.0%** | **100.0%** |
| **Gemini 3.8 Flash (GA Active)** | `vertex_ai` | **48095.9 ms** | **$2.15** | 0.0% | **100.0%** | **100.0%** | **100.0%** |
<!-- /GEN -->

---

## 4. Estructura del Repositorio

```
the-catcher-in-the-rag/
├── SPEC.md                    # Especificación técnica exhaustiva
├── README.md                  # Documentación principal, arquitectura y benchmark
├── StateOfCatcher.md          # Diagnóstico técnico de estado y tecnologías
├── Makefile                   # Automatización de tareas de desarrollo y pruebas
├── config/
│   ├── models.yaml            # Configuración de modelos GA activos
│   ├── thresholds.yaml        # Umbrales calibrados empíricamente
│   └── policy_rules.yaml      # Reglas de negocio y gobernanza financiera
├── src/catcher_in_the_rag/
│   ├── graph.py               # Grafo LangGraph con extracción end-to-end
│   ├── extract/               # Extracción estructurada de contratos
│   ├── match/                 # Cotejo tripartito determinista
│   ├── apply_terms/           # RAG de términos y evaluación de cláusulas
│   ├── bartleby/              # Guardia de abstención Melville
│   ├── validate/              # Validador de citas verbatim con offsets
│   ├── policy/                # Motor de políticas determinista
│   └── providers/             # Conectores Vertex AI y Stub determinista
├── evals/
│   ├── repro_check.py         # Verificación reproducible estricta
│   ├── COMMANDS.md            # Registro canónico de comandos de evaluación
│   └── reports/               # Reportes canónicos JSON versionados
└── tests/                     # Suite de pruebas unitarias e integración
```

---

## 5. Comandos Operativos Clave

Todos los flujos se operan mediante el `Makefile`:

- `make verify`: Ejecuta la puerta completa de evidencia (Linter, Tests, Auditoría de Afirmaciones, Repro-Check, Procedencia, Terraform).
- `make run-ui`: Inicia la interfaz web en el puerto 8501.
- `make run-api`: Inicia el servidor de API en el puerto 8000.
- `make test`: Ejecuta las pruebas unitarias offline deterministas.
