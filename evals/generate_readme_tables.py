"""Generate honest, reproducible Markdown tables for README.md directly from JSON reports.

Ensures that documentation can never diverge from empirical benchmark artifacts:
- Reads evals/reports/cuad_benchmark_latest.json
- Reads evals/reports/calibration_cuad_report.json
- Reads evals/reports/bakeoff_latest.json
- STRICT: Never uses fallback default constants for missing data; fails loudly on missing keys.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_cuad_table() -> str:
    cuad_path = Path("evals/reports/cuad_benchmark_latest.json")
    if not cuad_path.exists():
        raise FileNotFoundError(f"Required report {cuad_path} not found. Run benchmark first.")

    with cuad_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rep = data["report"]
    r1 = rep["recall_at_1"] * 100
    r3 = rep["recall_at_3"] * 100
    r5 = rep["recall_at_5"] * 100
    mrr = rep["mean_reciprocal_rank"]
    corr_abs = rep["bartleby_correct_abstention_rate"] * 100
    corr_ci = rep["bartleby_correct_abstention_ci95"]
    false_abs = rep["bartleby_false_abstention_rate"] * 100
    false_ci = rep["bartleby_false_abstention_ci95"]
    false_assert = rep["bartleby_false_assertion_rate"] * 100

    table = f"""| Métrica | Modo Offline (`stub-hash-768d`) | Vertex AI (`text-embedding-005` Live) | Intervalo de Confianza (95% Wilson Stub) | Interpretación y Significado Práctico |
|---|:---:|:---:|:---:|---|
| **Recall@1** | {r1:.1f}% | *Pendiente de cuota Vertex AI* | — | Precisión de primer ranking en contratos no estructurados. |
| **Recall@3** | {r3:.1f}% | *Pendiente de cuota Vertex AI* | — | Cláusula relevante localizada dentro del top 3 de candidatos. |
| **Recall@5** | {r5:.1f}% | *Pendiente de cuota Vertex AI* | — | Cobertura agregada en contratos extensos (50k a 175k caracteres). |
| **MRR** | {mrr:.4f} | *Pendiente de cuota Vertex AI* | — | Posición recíproca media del fragmento probatorio real. |
| **Abstención en Negativos** | **{corr_abs:.1f}%** | *Pendiente de cuota Vertex AI* | `[{corr_ci[0] * 100:.1f}%, {corr_ci[1] * 100:.1f}%]` | Ante cláusulas ausentes reales (`answers: []`), Bartleby rehúsa alucinar. |
| **Abstención Errónea (Positivos)** | **{false_abs:.1f}%** | *Pendiente de cuota Vertex AI* | `[{false_ci[0] * 100:.1f}%, {false_ci[1] * 100:.1f}%]` | Tasa de sobre-abstención en presencia de cláusulas legítimas válidas. |
| **Falsa Afirmación Espuria** | **{false_assert:.1f}%** | *Pendiente de cuota Vertex AI* | — | Riesgo residual de afirmar falsamente una penalización inexistente. |"""
    return table


def generate_bakeoff_table() -> str:
    bakeoff_path = Path("evals/reports/bakeoff_latest.json")
    if not bakeoff_path.exists():
        raise FileNotFoundError(f"Required report {bakeoff_path} not found. Run bakeoff first.")

    with bakeoff_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["candidates"]
    rows = []
    for c in candidates:
        name = c["candidate_name"]
        prov = c["provider_type"]
        lat = f"{c['mean_latency_ms']:.1f} ms"
        cost = (
            f"${c['estimated_cost_per_1k_usd']:.2f}"
            if not c.get("is_simulated", False)
            else "N/A (Stub)"
        )
        err = f"{c['error_rate'] * 100:.1f}%"
        pmt = f"{c['field_accuracies']['payment_terms_days'] * 100:.1f}%"
        rates = f"{c['field_accuracies']['rate_sheet'] * 100:.1f}%"
        overall = f"{c['overall_perfect_accuracy'] * 100:.1f}%"
        rows.append(
            f"| **{name}** | `{prov}` | **{lat}** | **{cost}** | {err} | **{pmt}** | **{rates}** | **{overall}** |"
        )

    header = """| Modelo / Candidato | Proveedor | Latencia Media | Costo / 1k docs | Error% | Payment Terms | Rate Sheet | Exactitud Global Conjunta |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|"""
    return header + "\n" + "\n".join(rows)


def generate_domain_calibration_table() -> str:
    cal_cuad_path = Path("evals/reports/calibration_cuad_report.json")
    if not cal_cuad_path.exists():
        raise FileNotFoundError(
            f"Required report {cal_cuad_path} not found. Run calibration first."
        )

    cal_bartleby_path = Path("evals/reports/calibration_bartleby_report.json")
    if not cal_bartleby_path.exists():
        raise FileNotFoundError(
            f"Required report {cal_bartleby_path} not found. Run calibrate-bartleby first."
        )

    with cal_cuad_path.open("r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    with cal_bartleby_path.open("r", encoding="utf-8") as f:
        synth_data = json.load(f)

    # STRICT: No default values. Fail if keys are missing from report JSON.
    synth_t = synth_data["optimal_threshold"]
    synth_f1 = synth_data["validation_macro_f1"]
    synth_auc = synth_data["auc_roc_val"]

    cuad_t = cuad_data["cuad_optimal_threshold"]
    val_synth = cuad_data["validation_f1_at_synthetic_threshold"]
    val_cuad = cuad_data["validation_f1_at_optimal"]
    val_c_synth = cuad_data["val_correct_abstention_at_synthetic"] * 100
    val_c_opt = cuad_data["val_correct_abstention_at_optimal"] * 100
    val_f_opt = cuad_data["val_false_abstention_at_optimal"] * 100
    auc_val = cuad_data["auc_roc_val"]
    always_f1 = cuad_data["baseline_always_abstain"]["macro_f1"]

    table = f"""| Dominio Evaluado | Dataset de Origen | Umbral Calibrado ($\\tau$) | Macro F1 Validación | AUC-ROC Separabilidad | Desempeño Empírico en Transferencia |
|---|---|:---:|:---:|:---:|---|
| **Plantillas Sintéticas** | Datagen M1 (300 paquetes) | **{synth_t:.2f}** | **{synth_f1:.4f}** | **{synth_auc:.4f}** | Alta separabilidad sintética entre cláusulas estándar y recargos anómalos. |
| **Contratos SEC Reales (Transfer)** | CUAD v1 Train/Val (60/40) | **{synth_t:.2f}** (fijado) | **{val_synth:.4f}** | **{auc_val:.4f}** | **Sub-abstención:** ante ausencias solo abstiene {val_c_synth:.1f}%, dejando pasar el {100 - val_c_synth:.1f}% como alucinaciones. |
| **Contratos SEC Reales (Recalibrado)** | CUAD v1 Train/Val (60/40) | **{cuad_t:.2f}** (recalibrado) | **{val_cuad:.4f}** | **{auc_val:.4f}** | Eleva abstención en ausencias a {val_c_opt:.1f}%, pero incrementa falsas abstenciones al {val_f_opt:.1f}% (vs clasificador ciego: {always_f1:.4f}). |"""
    return table


if __name__ == "__main__":
    print("=== CUAD BENCHMARK TABLE ===")
    print(generate_cuad_table())
    print("\n=== BAKEOFF TABLE ===")
    print(generate_bakeoff_table())
    print("\n=== DOMAIN CALIBRATION TABLE ===")
    print(generate_domain_calibration_table())
