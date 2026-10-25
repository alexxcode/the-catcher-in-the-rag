# Registro Canónico de Comandos de Evaluación

Este documento registra los comandos oficiales y reproducibles para generar cada reporte en `evals/reports/`.
Conforme a la Ley L1 y Ley L6, cada archivo JSON commiteado debe provenir de uno de estos comandos exactos, ejecutado con el working tree de Git limpio.

| Archivo de Reporte | Comando Canónico | Script de Origen | Descripción |
|---|---|---|---|
| `canonical_baseline_report.json` | `python -m evals.run --baseline` | `evals/run.py` | Benchmark sintético canónico (300 paquetes, términos contractuales completos). |
| `canonical_norag_report.json` | `python -m evals.run --norag` | `evals/run.py` | Ablación sintética: cotejo tripartito sin evaluación de términos contractuales. |
| `calibration_bartleby_report.json` | `python -m evals.calibrate_bartleby` | `evals/calibrate_bartleby.py` | Calibración de umbrales sobre datos sintéticos (180 dev / 120 val). |
| `calibration_cuad_report.json` | `python -m evals.calibrate_cuad` | `evals/calibrate_cuad.py` | Calibración de dominio en contratos SEC CUAD v1 (30 dev / 20 val). |
| `cuad_benchmark_latest.json` | `python -m evals.run --eval-cuad` | `evals/cuad_benchmark.py` | Benchmark de recuperación léxica y abstención negativa en CUAD v1. |
| `bartleby_llm_verification_report.json` | `python -m evals.verify_bartleby_llm --sample-size 15 --provider vertex_ai` | `evals/verify_bartleby_llm.py` | Evaluación de Modo 1 (tau=0.50), Modo 2 (tau=0.70) y Modo 3 (+LLM verification). |
| `bakeoff_latest.json` | `python -m evals.run --bakeoff --sample-size 30 --output evals/reports/` | `evals/bakeoff.py` | Bake-off de extracción estructurada en Vertex AI (n=30, Gemini 3.5 vs 3.8 vs reglas). |
| `audit_rejected_cases_report.json` | `python -m evals.audit_rejected_cases` | `evals/audit_rejected_cases.py` | Auditoría caso por caso de los 26 casos con cláusula presente en CUAD dev. |
| `mode3_variance_report.json` | `python evals/measure_mode3_runs.py` | `evals/measure_mode3_runs.py` | Medición de varianza y determinismo en 3 corridas consecutivas de Modo 3. |
| `out_of_sample_cuad_report.json` | `python evals/eval_out_of_sample.py --start 15 --end 50 --llm-provider vertex_ai` | `evals/eval_out_of_sample.py` | Evaluación fuera de muestra en 35 contratos held-out (CUAD 16–50) con definiciones oficiales. |
